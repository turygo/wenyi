"""Deterministic hidden-EPUB interruption/resume integration harness."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

import trans_novel.benchmark.integration.artifacts as integration_artifacts
from trans_novel.benchmark.artifacts import sha256_bytes
from trans_novel.benchmark.integration.artifacts import (
    IntegrationError,
    IntegrationIntegrityError,
    integration_relative_path,
    integration_sha256,
    read_integration_canonical,
    validate_all_candidate_paths,
    validate_result_contract,
    write_integration_json,
)
from trans_novel.benchmark.integration.preflight import preflight
from trans_novel.benchmark.integration.resume import (
    candidate_store,
    run_candidate,
    validate_restart_prefixes,
)
from trans_novel.benchmark.run import JsonlCallTelemetrySink, model_client
from trans_novel.benchmark.schema import Candidate, CandidateSpec, StrictModel
from trans_novel.config import ModelRoles
from trans_novel.llm.generation import GenerationOptions

_HEX64 = r"^[0-9a-f]{64}$"
_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"


class IntegrationSpec(StrictModel):
    """Immutable, strict request for one hidden-book integration run."""

    schema_version: Literal[1]
    benchmark_id: str = Field(min_length=1, pattern=_SAFE_ID)
    corpus_sha256: str = Field(pattern=_HEX64)
    candidate_spec_sha256: str = Field(pattern=_HEX64)
    book_id: str = Field(min_length=1, pattern=_SAFE_ID)
    candidate_ids: list[str] = Field(min_length=2, max_length=3)
    interrupt_after_committed_batches: int = Field(ge=1)
    output_mono: Literal[True]
    output_bilingual: Literal[True]
    bilingual_order: Literal["target_first", "source_first"]
    source_language: Literal["en"]
    target_language: Literal["zh"]

    @staticmethod
    def _validate_ids(ids: list[str]) -> list[str]:
        folded = [value.casefold() for value in ids]
        if (
            len(set(ids)) != len(ids)
            or len(set(folded)) != len(folded)
            or any(not re.fullmatch(_SAFE_ID, value) for value in ids)
        ):
            raise ValueError("candidate_ids must contain unique safe IDs")
        return ids

    def model_post_init(self, __context: Any) -> None:
        self._validate_ids(self.candidate_ids)


class BenchmarkInterruption(ValueError):
    """Expected safe-boundary interruption after a committed translation batch."""


class _CommitHook:
    def __init__(self, limit: int) -> None:
        self.limit, self.armed = limit, True
        self.reached = self.raised = False
        self.committed: list[dict[str, int]] = []

    def after_batch_committed(self, chapter_index: int, start: int, count: int) -> None:
        if not self.armed:
            return
        self.committed.append({"chapter": chapter_index, "start": start, "count": count})
        if len(self.committed) >= self.limit:
            self.reached = self.raised = True
            self.armed = False
            raise BenchmarkInterruption(
                f"benchmark interruption after {len(self.committed)} committed batches"
            )


def _prepare_integration(
    runner: Any,
    corpus_dir: str | os.PathLike[str],
    book_spec_path: str | os.PathLike[str],
    candidate_spec_path: str | os.PathLike[str],
    integration_spec_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    corpus, book_spec = Path(corpus_dir).expanduser(), Path(book_spec_path).expanduser()
    candidates, integration_input = (
        Path(candidate_spec_path).expanduser(),
        Path(integration_spec_path).expanduser(),
    )
    out = Path(out_dir).expanduser().resolve()
    runner._created_client_ids.clear()
    spec, candidate_spec, source, source_hash, lineage = preflight(
        corpus, book_spec, candidates, integration_input, spec_type=IntegrationSpec
    )
    out.mkdir(parents=True, exist_ok=True)
    chosen = list(lineage["selected"])
    validate_all_candidate_paths(out, source, list(spec.candidate_ids), spec.book_id)
    request = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "corpus_sha256": spec.corpus_sha256,
        "book_spec_sha256": lineage["book_spec_sha256"],
        "candidate_spec_sha256": lineage["candidate_spec_sha256"],
        "integration_spec_sha256": lineage["integration_spec_sha256"],
        "book_id": spec.book_id,
        "source_sha256": source_hash,
        "source_language": spec.source_language,
        "target_language": spec.target_language,
        "candidate_ids": list(spec.candidate_ids),
        "candidates": {
            c.candidate_id: {
                "pipeline_variant": c.pipeline_variant,
                "translator_model": c.translator_model,
                "analyst_model": c.analyst_model,
                "editor_model": c.editor_model,
                "fast_model": c.fast_model,
                "temperature": candidate_spec.temperature,
                "seed": candidate_spec.seed,
            }
            for c in chosen
        },
        "interrupt_after_committed_batches": spec.interrupt_after_committed_batches,
        "output_mono": spec.output_mono,
        "output_bilingual": spec.output_bilingual,
        "bilingual_order": spec.bilingual_order,
    }
    request_path, state_path = out / "integration_request.json", out / "integration_state.json"
    if not request_path.exists() and any(
        p.exists()
        for p in (
            state_path,
            out / "integration.json",
            out / "integration_complete.json",
            out / "candidates",
        )
    ):
        raise IntegrationError("integration request missing for existing artifacts")
    request_created = not request_path.exists()
    if request_path.exists() and read_integration_canonical(request_path) != request:
        raise IntegrationError("integration immutable request mismatch")
    if request_created:
        write_integration_json(request_path, request)
    if not state_path.exists() and not request_created:
        raise IntegrationError("integration state missing for existing request")
    state = (
        read_integration_canonical(state_path)
        if state_path.exists()
        else {
            "schema_version": 1,
            "benchmark_id": spec.benchmark_id,
            "candidate_ids": list(spec.candidate_ids),
            "candidates": {cid: {"status": "pending"} for cid in spec.candidate_ids},
        }
    )
    if (
        state.get("schema_version") != 1
        or state.get("benchmark_id") != spec.benchmark_id
        or state.get("candidate_ids") != list(spec.candidate_ids)
        or set(state.get("candidates", {})) != set(spec.candidate_ids)
    ):
        raise IntegrationError("integration state mismatch")
    allowed = {"pending", "interrupted", "resuming", "completed", "failed"}
    if any(
        not isinstance(v, dict) or v.get("status") not in allowed
        for v in state["candidates"].values()
    ):
        raise IntegrationError("integration state contains invalid candidate status")
    return {
        "out": out,
        "spec": spec,
        "candidate_spec": candidate_spec,
        "source": source,
        "source_hash": source_hash,
        "lineage": lineage,
        "chosen": chosen,
        "request": request,
        "request_path": request_path,
        "state_path": state_path,
        "state": state,
        "request_created": request_created,
    }


def _resume_terminal(data: dict[str, Any]) -> dict[str, Any] | None:
    out, spec, lineage, state = data["out"], data["spec"], data["lineage"], data["state"]
    integration_path, complete_path = out / "integration.json", out / "integration_complete.json"
    if not (complete_path.exists() or integration_path.exists()):
        return None
    validated = integration_artifacts.validate_terminal_artifacts(out)
    integration, complete = validated["integration"], validated["complete"]
    expected_lineage = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "corpus_sha256": spec.corpus_sha256,
        "book_spec_sha256": lineage["book_spec_sha256"],
        "candidate_spec_sha256": lineage["candidate_spec_sha256"],
        "integration_spec_sha256": lineage["integration_spec_sha256"],
        "book_id": spec.book_id,
        "source_sha256": data["source_hash"],
    }
    if any(integration.get(k) != v for k, v in expected_lineage.items()):
        raise IntegrationError("integration manifest lineage mismatch")
    failed = []
    for cid in spec.candidate_ids:
        entry, completion = integration["candidates"][cid], complete["candidates"][cid]
        current = state["candidates"][cid]
        if current.get("status") != completion["status"]:
            raise IntegrationError("integration state terminal status mismatch")
        if current.get("result_sha256") != entry["result_sha256"]:
            raise IntegrationError("integration result lineage mismatch")
        if "boundary_event_count" in current:
            validate_restart_prefixes(
                current,
                candidate_store=candidate_store(out / "candidates" / cid / "state"),
                telemetry_path=out / "candidates" / cid / "telemetry.jsonl",
            )
        if completion["status"] == "failed":
            failed.append(cid)
    return {
        **integration,
        "out_dir": str(out),
        "no_op": True,
        "failed_candidates": failed,
        "integration_sha256": validated["integration_sha256"],
        "resumed": True,
    }


def _execute_integration(data: dict[str, Any], runner: Any) -> dict[str, Any]:
    out, spec, candidate_spec = data["out"], data["spec"], data["candidate_spec"]
    state, state_path = data["state"], data["state_path"]
    request_path, chosen = data["request_path"], data["chosen"]
    results: dict[str, dict[str, Any]] = {}
    write_integration_json(state_path, state)
    for candidate in chosen:
        cid, candidate_state = candidate.candidate_id, state["candidates"][candidate.candidate_id]
        if candidate_state.get("status") in {"completed", "failed"}:
            path = out / "candidates" / cid / "result.json"
            if not path.exists():
                raise IntegrationError("terminal candidate result is missing")
            value = read_integration_canonical(path)
            validate_result_contract(
                value,
                candidate_id=cid,
                expected_lineage={**data["request"], "schema_version": 1},
                request_sha256=integration_sha256(request_path),
                status=candidate_state["status"],
                out=out,
            )
            if candidate_state.get("result_sha256") != integration_sha256(path):
                raise IntegrationError("terminal candidate result hash mismatch")
            results[cid] = value
        else:
            results[cid] = runner._run_candidate(
                spec,
                candidate_spec,
                candidate,
                data["source"],
                data["source_hash"],
                out,
                state,
                state_path,
            )
    manifests, completions = {}, {}
    for cid in sorted(results):
        path = out / "candidates" / cid / "result.json"
        digest, relative = integration_sha256(path), integration_relative_path(path, out)
        status = "completed" if results[cid].get("passed") else "failed"
        manifests[cid] = {"result_path": relative, "result_sha256": digest}
        completions[cid] = {"status": status, "result_path": relative, "result_sha256": digest}
    integration = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "corpus_sha256": spec.corpus_sha256,
        "book_spec_sha256": data["lineage"]["book_spec_sha256"],
        "candidate_spec_sha256": data["lineage"]["candidate_spec_sha256"],
        "integration_spec_sha256": data["lineage"]["integration_spec_sha256"],
        "book_id": spec.book_id,
        "source_sha256": data["source_hash"],
        "candidates": manifests,
    }
    integration_path, complete_path = out / "integration.json", out / "integration_complete.json"
    write_integration_json(integration_path, integration)
    complete = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "integration_sha256": integration_sha256(integration_path),
        "terminal": True,
        "candidates": completions,
    }
    write_integration_json(complete_path, complete)
    return {
        **integration,
        "out_dir": str(out),
        "no_op": False,
        "failed_candidates": [cid for cid, value in results.items() if not value.get("passed")],
        "integration_sha256": sha256_bytes(integration_path.read_bytes()),
        "resumed": not data["request_created"],
    }


class IntegrationRunner:
    """Run or resume isolated candidate integrations under one output directory."""

    def __init__(
        self, *, client_factory: Callable[..., Any] | None = None, client: Any | None = None
    ) -> None:
        self.client_factory, self.client = client_factory, client
        self._created_client_ids: set[int] = set()

    def _client(
        self,
        spec: CandidateSpec,
        candidate: Candidate,
        options: GenerationOptions,
        sink: JsonlCallTelemetrySink,
    ) -> Any:
        roles = ModelRoles(
            translator=[candidate.translator_model],
            analyst=[candidate.analyst_model],
            editor=[candidate.editor_model],
            fast=[candidate.fast_model],
        )
        if self.client is not None:
            raise IntegrationError(
                "IntegrationRunner requires a client_factory for distinct clients"
            )
        client = model_client(
            spec,
            candidate.translator_model,
            "translator",
            options,
            self.client_factory,
            sink,
            roles=roles,
        )
        identity = id(client)
        if identity in self._created_client_ids:
            raise IntegrationIntegrityError("client_factory returned a singleton client")
        self._created_client_ids.add(identity)
        return client

    def run(
        self,
        corpus_dir: str | os.PathLike[str],
        book_spec_path: str | os.PathLike[str],
        candidate_spec_path: str | os.PathLike[str],
        integration_spec_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        data = _prepare_integration(
            self, corpus_dir, book_spec_path, candidate_spec_path, integration_spec_path, out_dir
        )
        terminal = _resume_terminal(data)
        return terminal if terminal is not None else _execute_integration(data, self)

    def _run_candidate(
        self,
        spec: IntegrationSpec,
        candidate_spec: CandidateSpec,
        candidate: Candidate,
        source: Path,
        source_hash: str,
        out: Path,
        state: dict[str, Any],
        state_path: Path,
    ) -> dict[str, Any]:
        return run_candidate(
            self._client,
            _CommitHook,
            BenchmarkInterruption,
            spec,
            candidate_spec,
            candidate,
            source,
            source_hash,
            out,
            state,
            state_path,
        )


__all__ = [
    "BenchmarkInterruption",
    "IntegrationError",
    "IntegrationIntegrityError",
    "IntegrationRunner",
    "IntegrationSpec",
]
