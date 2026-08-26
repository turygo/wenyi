"""Deterministic hidden-EPUB interruption/resume integration harness.

The harness is intentionally benchmark-only: it validates all immutable inputs before
opening a model client, then delegates translation, routing, telemetry, readiness, and
assembly to the production pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from trans_novel.benchmark.corpus import (
    canonical_json,
    load_book_spec,
    sha256_bytes,
    validate_corpus,
)
from trans_novel.benchmark.epub_check import validate_epub_triplet
from trans_novel.benchmark.runner import (
    _GENERATION_FIELDS,
    BenchmarkError,
    FullRunner,
    _JsonlTelemetrySink,
    _model_client,
    load_candidate_spec,
    validate_candidate_capabilities,
)
from trans_novel.benchmark.schema import Candidate, CandidateSpec, StrictModel
from trans_novel.config import Config, ModelRoles
from trans_novel.llm.generation import GenerationOptions
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.model_profiles import parse_model_selection
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.readiness import assemble_readiness_problems
from trans_novel.pipeline.runstore import RunStore

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
        self.limit = limit
        self.armed = True
        self.reached = False
        self.raised = False
        self.committed: list[dict[str, int]] = []

    def after_batch_committed(self, chapter_index: int, start: int, count: int) -> None:
        if not self.armed:
            return
        identity = {"chapter": chapter_index, "start": start, "count": count}
        self.committed.append(identity)
        if len(self.committed) >= self.limit:
            self.reached = True
            self.raised = True
            self.armed = False
            raise BenchmarkInterruption(
                f"benchmark interruption after {len(self.committed)} committed batches"
            )


class IntegrationError(BenchmarkError):
    """Invalid immutable input, state, or integration evidence."""


class IntegrationIntegrityError(IntegrationError):
    """Corrupt persisted lineage that must abort the whole integration."""


def _canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise IntegrationError(f"invalid JSON artifact {path}: {error}") from error


def _sha(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise IntegrationError(f"cannot hash {path}: {error}") from error


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise IntegrationError(f"path escapes declared root: {path}") from error


def _read_canonical(path: Path) -> Any:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != _canonical_bytes(value):
        raise IntegrationError(f"non-canonical JSON artifact: {path}")
    return value


def _validate_candidate_paths(
    root: Path,
    source: Path,
    *paths: Path,
) -> None:
    """Reject escape, symlink escape, case-fold collisions, and aliases."""
    resolved_root = root.resolve()
    resolved_source = source.resolve()
    resolved: list[Path] = []
    for path in paths:
        current = path.resolve(strict=False)
        try:
            current.relative_to(resolved_root)
        except ValueError as error:
            raise IntegrationError(f"candidate path escapes integration root: {path}") from error
        resolved.append(current)
    values = [str(path).casefold() for path in (resolved_source, *resolved)]
    if len(values) != len(set(values)):
        raise IntegrationError("candidate artifact paths alias")


def _validate_all_candidate_paths(
    root: Path,
    source: Path,
    candidate_ids: list[str],
    book_id: str,
) -> None:
    planned: list[Path] = []
    for cid in candidate_ids:
        candidate_root = root / "candidates" / cid
        state_dir = candidate_root / "state"
        output_dir = candidate_root / "outputs"
        planned.extend(
            (
                candidate_root,
                state_dir,
                output_dir,
                output_dir / f"{book_id}.epub",
                output_dir / f"{book_id}-bi.epub",
                candidate_root / "telemetry.jsonl",
                candidate_root / "telemetry.first.jsonl",
                candidate_root / "telemetry.resume.jsonl",
                candidate_root / "canary.json",
                candidate_root / "result.json",
            )
        )
    _validate_candidate_paths(root, source, *planned)


def _events(store: RunStore) -> list[dict[str, Any]]:
    path = Path(store.event_log_path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    except Exception as error:
        raise IntegrationError(f"invalid RunStore event log: {error}") from error
    return rows


def _events_from_bytes(raw: bytes) -> list[dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        for line in raw.decode("utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event is not an object")
                rows.append(value)
        return rows
    except Exception as error:
        raise IntegrationError(f"invalid RunStore event prefix: {error}") from error


def _authenticated_event_prefix(
    store: RunStore,
    *,
    count: int,
    size: int,
    digest: str,
) -> list[dict[str, Any]]:
    path = Path(store.event_log_path)
    raw = path.read_bytes()
    if count < 0 or size < 0 or len(raw) < size:
        raise IntegrationError("event prefix boundary is beyond persisted events")
    prefix = raw[:size]
    if hashlib.sha256(prefix).hexdigest() != digest:
        raise IntegrationIntegrityError("interrupted event prefix mismatch")
    rows = _events_from_bytes(prefix)
    if len(rows) != count:
        raise IntegrationIntegrityError("interrupted event prefix count mismatch")
    return rows


def _telemetry_records(path: Path) -> list[CallAttemptTelemetry]:
    try:
        return [
            CallAttemptTelemetry.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as error:
        raise IntegrationError(f"invalid telemetry artifact {path}: {error}") from error


def _authenticated_telemetry_prefix(
    path: Path,
    *,
    count: int,
    size: int,
    digest: str,
) -> list[CallAttemptTelemetry]:
    raw = path.read_bytes()
    if count < 0 or size < 0 or len(raw) < size:
        raise IntegrationError("telemetry prefix boundary is beyond persisted attempts")
    prefix = raw[:size]
    if hashlib.sha256(prefix).hexdigest() != digest:
        raise IntegrationIntegrityError("interrupted telemetry prefix mismatch")
    try:
        rows = [
            CallAttemptTelemetry.model_validate(json.loads(line))
            for line in prefix.decode("utf-8").splitlines()
            if line.strip()
        ]
    except Exception as error:
        raise IntegrationIntegrityError("interrupted telemetry prefix is invalid") from error
    if len(rows) != count:
        raise IntegrationIntegrityError("interrupted telemetry prefix count mismatch")
    return rows


def _canary_routes(records: list[CallAttemptTelemetry]) -> list[dict[str, str]]:
    return [{"agent": record.agent, "operation": record.operation} for record in records]


def _validate_restart_prefixes(
    state_entry: dict[str, Any],
    *,
    candidate_store: RunStore,
    telemetry_path: Path,
) -> tuple[list[dict[str, Any]], list[CallAttemptTelemetry]]:
    """Authenticate and semantically validate the persisted restart boundaries."""
    required = (
        "boundary_event_count",
        "event_prefix_size",
        "event_prefix_sha256",
        "first_telemetry_count",
        "telemetry_prefix_size",
        "telemetry_prefix_sha256",
        "canary_telemetry_count",
    )
    if any(
        not isinstance(state_entry.get(key), int)
        for key in required
        if key.endswith(("count", "size"))
    ):
        raise IntegrationError("interrupted candidate evidence is incomplete")
    interruption = state_entry.get("interruption")
    if not isinstance(interruption, dict) or not isinstance(interruption.get("batches"), list):
        raise IntegrationError("interruption identity evidence is incomplete")
    first_rows = _authenticated_event_prefix(
        candidate_store,
        count=int(
            state_entry.get("first_boundary_event_count", state_entry["boundary_event_count"])
        ),
        size=int(state_entry.get("first_event_prefix_size", state_entry["event_prefix_size"])),
        digest=str(
            state_entry.get("first_event_prefix_sha256", state_entry["event_prefix_sha256"])
        ),
    )
    first_batches = _batches(first_rows)
    identities = [
        {"chapter": row["chapter"], "start": row["start"], "count": row["count"]}
        for row in first_batches
    ]
    if identities != interruption["batches"] or interruption.get("count") != len(identities):
        raise IntegrationIntegrityError("interruption identities do not match event prefix")
    event_rows = _authenticated_event_prefix(
        candidate_store,
        count=state_entry["boundary_event_count"],
        size=state_entry["event_prefix_size"],
        digest=state_entry["event_prefix_sha256"],
    )
    before = state_entry.get("before_target_hashes")
    if not isinstance(before, list) or _batches(event_rows) != before:
        raise IntegrationIntegrityError("interruption target hashes do not match event prefix")
    first_telemetry_count = state_entry["first_telemetry_count"]
    first_telemetry_size = state_entry.get(
        "first_telemetry_prefix_size", state_entry["telemetry_prefix_size"]
    )
    first_telemetry_digest = state_entry.get(
        "first_telemetry_prefix_sha256", state_entry["telemetry_prefix_sha256"]
    )
    telemetry_rows = _authenticated_telemetry_prefix(
        telemetry_path,
        count=first_telemetry_count,
        size=first_telemetry_size,
        digest=first_telemetry_digest,
    )
    canary_count = state_entry["canary_telemetry_count"]
    if canary_count != 3 or len(telemetry_rows) < canary_count:
        raise IntegrationError("canary telemetry prefix is incomplete")
    routes = _canary_routes(telemetry_rows[:canary_count])
    if state_entry.get("canary_telemetry_routes") != routes:
        raise IntegrationIntegrityError("canary telemetry routes do not match prefix")
    attempt_count = state_entry.get("attempt_telemetry_count", first_telemetry_count)
    attempt_size = state_entry.get(
        "attempt_telemetry_prefix_size", state_entry["telemetry_prefix_size"]
    )
    attempt_digest = state_entry.get(
        "attempt_telemetry_prefix_sha256", state_entry["telemetry_prefix_sha256"]
    )
    if (
        not isinstance(attempt_count, int)
        or not isinstance(attempt_size, int)
        or not isinstance(attempt_digest, str)
    ):
        raise IntegrationError("resume telemetry boundary is incomplete")
    _authenticated_telemetry_prefix(
        telemetry_path,
        count=attempt_count,
        size=attempt_size,
        digest=attempt_digest,
    )
    return event_rows, telemetry_rows


def _batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chapter": int(row["chapter"]),
            "start": int(row["start_index"]),
            "count": int(row["count"]),
            "target_sha256": str(row["target_sha256"]),
        }
        for row in rows
        if row.get("event") == "batch_translated" and not row.get("back_matter")
    ]


def _candidate_store(state_dir: Path) -> RunStore:
    candidates = (
        [
            path
            for path in state_dir.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        ]
        if state_dir.exists()
        else []
    )
    if len(candidates) != 1:
        raise IntegrationError("candidate Application state root is missing or ambiguous")
    return RunStore(str(candidates[0]))


def _skips(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chapter": int(row["chapter"]),
            "start": int(row["start_index"]),
            "count": int(row["count"]),
            "target_sha256": str(row.get("target_sha256", "")),
        }
        for row in rows
        if row.get("event") == "batch_skipped" and row.get("reason") == "already_translated"
    ]


def _normalized_model(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return parse_model_selection(value).model


def _telemetry_evidence(
    path: Path,
    *,
    candidate: Candidate,
    candidate_spec: CandidateSpec,
    start_index: int = 0,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
            "unknown_required_usage_count": 1,
            "logical_call_count": 0,
            "attempt_count": 0,
            "operation_count": 0,
            "agent_count": 0,
            "retry_count": 0,
            "translate_call_count": 0,
            "valid": False,
        }
    try:
        raw_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records = [CallAttemptTelemetry.model_validate(item) for item in raw_records]
        records = records[start_index:]
    except Exception:
        return {
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
            "unknown_required_usage_count": 1,
            "logical_call_count": 0,
            "attempt_count": 0,
            "operation_count": 0,
            "agent_count": 0,
            "retry_count": 0,
            "translate_call_count": 0,
            "valid": False,
        }
    expected = {
        "translator": _normalized_model(candidate.primary_model),
        "analyst": _normalized_model(candidate.primary_model),
        "editor": _normalized_model(candidate.editor_model),
        "reviewer": _normalized_model(candidate_spec.fast_model),
        "preparer": _normalized_model(candidate_spec.fast_model),
        "light-translator": _normalized_model(candidate_spec.fast_model),
    }
    expected_operations = {
        "translator": {
            "translate.batch",
            "translate.back_matter",
            "translate.lint_fix",
            "translate.review_fix",
            "title.translate",
            "integration.canary.translate",
        },
        "analyst": {"analyzer.analyze", "prescan.name_terms", "glossary.audit"},
        "preparer": {
            "prescan.digest",
            "prescan.book_synopsis",
            "prescan.term_mine",
            "glossary.extract",
        },
        "editor": {"polish.batch", "naturalize.rewrite", "integration.canary.polish"},
        "reviewer": {
            "language.detect",
            "review.chapter",
            "backtranslate.check",
            "consistency.check",
            "naturalize.screen",
            "naturalize.pair",
            "naturalize.fidelity",
            "integration.canary.review",
        },
        "light-translator": {"backtranslate.translate", "translate.back_matter"},
    }
    mismatches = 0
    unknown = 0
    reasoning = 0
    for record in records:
        expected_model = expected.get(record.agent)
        if (
            expected_model is None
            or record.operation not in expected_operations.get(record.agent, set())
            or record.provider != candidate_spec.provider
            or _normalized_model(record.requested_model) != expected_model
            or _normalized_model(record.resolved_model) != expected_model
            or record.reasoning_enabled
            or record.status != "success"
            or record.temperature != 0.1
            or record.seed is not None
        ):
            mismatches += 1
        if record.billed_usage_unknown:
            unknown += 1
        reasoning += record.reasoning_tokens
    return {
        "reasoning_tokens": reasoning,
        "model_mismatch_count": mismatches,
        "unknown_required_usage_count": unknown,
        "logical_call_count": len({record.logical_call_id for record in records}),
        "attempt_count": len(records),
        "operation_count": len({record.operation for record in records}),
        "agent_count": len({record.agent for record in records}),
        "retry_count": sum(
            1 for record in records if record.attempt_index > 1 or record.retry_class is not None
        ),
        "translate_call_count": sum(
            1 for record in records if record.operation == "translate.batch"
        ),
        "valid": True,
    }


def _usage_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"unknown_required_usage_count": 1, "valid": False}
    try:
        value = _read_json(path)
        if value.get("schema_version") != 2:
            raise ValueError("usage schema mismatch")
        totals = value["totals"]
        required = (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        )
        if any(
            isinstance(totals.get(key), bool)
            or not isinstance(totals.get(key), int)
            or totals[key] < 0
            for key in required
        ):
            raise ValueError("invalid usage totals")
    except Exception:
        return {"unknown_required_usage_count": 1, "valid": False}
    attempts = sum(
        int(slot.get("attempts", 0))
        for slot in (value.get("by_agent") or {}).values()
        if isinstance(slot, dict)
    )
    logical_calls = sum(
        int(slot.get("logical_calls", 0))
        for slot in (value.get("by_agent") or {}).values()
        if isinstance(slot, dict)
    )
    return {
        "unknown_required_usage_count": 0,
        "valid": True,
        "calls": int(totals["calls"]),
        "attempts": attempts,
        "logical_calls": logical_calls,
        "prompt_tokens": int(totals["prompt_tokens"]),
        "completion_tokens": int(totals["completion_tokens"]),
        "total_tokens": int(totals["total_tokens"]),
    }


def _failure_code(error: BaseException) -> str:
    message = str(error).casefold()
    known = (
        ("resume event boundary", "resume_event_boundary"),
        ("resume telemetry boundary", "resume_telemetry_boundary"),
        ("resume reuse proof", "resume_reuse_proof"),
        ("resume translation call attribution", "resume_telemetry_attribution"),
        ("usage and application telemetry", "usage_telemetry_mismatch"),
        ("crashed resume evidence", "crashed_resume_evidence"),
        ("event slice mismatch", "resume_event_slice"),
        ("telemetry prefix", "telemetry_prefix"),
        ("event prefix", "event_prefix"),
        ("canary", "canary_evidence"),
        ("phase timing", "phase_timing"),
        ("bilingual output", "bilingual_output"),
        ("readiness", "readiness"),
        ("first telemetry", "first_telemetry_evidence"),
    )
    for fragment, code in known:
        if fragment in message:
            return code
    return "integration_error"


def _timestamp_ms(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(parsed.timestamp() * 1000)


def _recovered_active_duration_ms(
    started_at: Any,
    *,
    events: list[dict[str, Any]],
    telemetry: list[CallAttemptTelemetry],
) -> int:
    start_ms = _timestamp_ms(started_at)
    if start_ms is None:
        return 0
    ends: list[int] = []
    for row in events:
        for key in ("ts", "at", "timestamp", "created_at"):
            value = _timestamp_ms(row.get(key))
            if value is not None:
                ends.append(value)
                break
    for record in telemetry:
        begin = _timestamp_ms(record.started_at)
        if begin is not None:
            ends.append(begin + max(0, int(record.elapsed_ms)))
    if not ends:
        return 0
    return max(0, max(ends) - start_ms)


def validate_terminal_artifacts(root: Path | str) -> dict[str, Any]:
    """Authenticate one terminal Phase 9 producer directory.

    This is the sole terminal-artifact validator.  Phase 8 report consumers must
    use it instead of interpreting a reduced integration manifest.
    """
    out = Path(root).expanduser().resolve()
    request_path = out / "integration_request.json"
    integration_path = out / "integration.json"
    complete_path = out / "integration_complete.json"
    try:
        request = _read_canonical(request_path)
        integration = _read_canonical(integration_path)
        complete = _read_canonical(complete_path)
    except Exception as error:
        if isinstance(error, IntegrationError):
            raise
        raise IntegrationError(f"invalid terminal integration artifacts: {error}") from error
    if not all(isinstance(item, dict) for item in (request, integration, complete)):
        raise IntegrationError("terminal integration artifacts must be objects")
    request_required = (
        "schema_version",
        "benchmark_id",
        "corpus_sha256",
        "book_spec_sha256",
        "candidate_spec_sha256",
        "integration_spec_sha256",
        "book_id",
        "source_sha256",
        "source_language",
        "target_language",
        "candidate_ids",
        "candidates",
        "interrupt_after_committed_batches",
        "output_mono",
        "output_bilingual",
        "bilingual_order",
    )
    if any(key not in request for key in request_required):
        raise IntegrationError("integration request contract is incomplete")
    if (
        request["schema_version"] != 1
        or not isinstance(request["benchmark_id"], str)
        or not isinstance(request["book_id"], str)
        or not isinstance(request["candidate_ids"], list)
        or len(request["candidate_ids"]) not in {2, 3}
        or any(
            not isinstance(cid, str) or not re.fullmatch(_SAFE_ID, cid)
            for cid in request["candidate_ids"]
        )
        or len({cid.casefold() for cid in request["candidate_ids"]})
        != len(request["candidate_ids"])
        or not isinstance(request["candidates"], dict)
        or set(request["candidates"]) != set(request["candidate_ids"])
        or request["source_language"] != "en"
        or request["target_language"] != "zh"
        or request["output_mono"] is not True
        or request["output_bilingual"] is not True
        or request["bilingual_order"] not in {"target_first", "source_first"}
        or type(request["interrupt_after_committed_batches"]) is not int
        or request["interrupt_after_committed_batches"] < 1
        or any(
            not isinstance(request.get(key), str) or not re.fullmatch(_HEX64, request[key])
            for key in (
                "corpus_sha256",
                "book_spec_sha256",
                "candidate_spec_sha256",
                "integration_spec_sha256",
                "source_sha256",
            )
        )
    ):
        raise IntegrationError("integration request contract is invalid")
    request_hash = _sha(request_path)
    lineage_keys = (
        "schema_version",
        "benchmark_id",
        "corpus_sha256",
        "book_spec_sha256",
        "candidate_spec_sha256",
        "integration_spec_sha256",
        "book_id",
        "source_sha256",
    )
    if any(integration.get(key) != request[key] for key in lineage_keys):
        raise IntegrationError("integration manifest request lineage mismatch")
    if (
        not isinstance(integration.get("candidates"), dict)
        or not isinstance(complete.get("candidates"), dict)
        or complete.get("schema_version") != 1
        or complete.get("benchmark_id") != request["benchmark_id"]
        or complete.get("integration_sha256") != _sha(integration_path)
        or complete.get("terminal") is not True
        or set(integration["candidates"]) != set(request["candidate_ids"])
        or set(complete["candidates"]) != set(request["candidate_ids"])
    ):
        raise IntegrationError("integration completion manifest is invalid")
    candidates: dict[str, dict[str, Any]] = {}
    candidate_fields = (
        "provider",
        "primary_model",
        "editor_model",
        "fast_model",
        "temperature",
        "seed",
    )
    for candidate_id, candidate in request["candidates"].items():
        if (
            not isinstance(candidate, dict)
            or any(key not in candidate for key in candidate_fields)
            or not isinstance(candidate.get("provider"), str)
            or not isinstance(candidate.get("primary_model"), str)
            or not isinstance(candidate.get("fast_model"), str)
            or not isinstance(candidate.get("editor_model"), str)
            or isinstance(candidate.get("temperature"), bool)
            or not isinstance(candidate.get("temperature"), int | float)
            or (
                candidate.get("seed") is not None
                and (
                    isinstance(candidate.get("seed"), bool)
                    or not isinstance(candidate.get("seed"), int)
                )
            )
        ):
            raise IntegrationError(f"integration request candidate {candidate_id} is invalid")
    providers = {candidate["provider"] for candidate in request["candidates"].values()}
    if len(providers) != 1:
        raise IntegrationError("integration candidates must share one provider")
    fast_models = {candidate["fast_model"] for candidate in request["candidates"].values()}
    if len(fast_models) != 1:
        raise IntegrationError("integration candidates must share one fast model")
    provider = next(iter(providers))
    if any(
        candidate["temperature"] != 0.1 or candidate["seed"] is not None
        for candidate in request["candidates"].values()
    ):
        raise IntegrationError("integration candidates require fixed generation controls")
    try:
        candidate_spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": request["benchmark_id"],
                "provider": provider,
                "fast_model": next(iter(fast_models)),
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "candidates": [
                    {
                        "candidate_id": cid,
                        "primary_model": request["candidates"][cid]["primary_model"],
                        "editor_model": request["candidates"][cid]["editor_model"],
                    }
                    for cid in request["candidate_ids"]
                ],
            }
        )
        validate_candidate_capabilities(candidate_spec)
    except Exception as error:
        raise IntegrationError(
            f"integration candidate capabilities are invalid: {error}"
        ) from error
    for cid in sorted(request["candidate_ids"]):
        entry = integration["candidates"].get(cid)
        centry = complete["candidates"].get(cid)
        if (
            not isinstance(entry, dict)
            or not isinstance(centry, dict)
            or not isinstance(entry.get("result_path"), str)
            or Path(entry["result_path"]).is_absolute()
            or ".." in Path(entry["result_path"]).parts
            or centry.get("result_path") != entry["result_path"]
            or centry.get("result_sha256") != entry.get("result_sha256")
            or centry.get("status") not in {"completed", "failed"}
            or not isinstance(entry.get("result_sha256"), str)
            or not re.fullmatch(_HEX64, entry["result_sha256"])
        ):
            raise IntegrationError(f"integration candidate {cid} manifest is invalid")
        result_path = (out / entry["result_path"]).resolve()
        if _safe_relative(result_path, out) != entry["result_path"] or not result_path.is_file():
            raise IntegrationError(f"integration candidate {cid} result path is invalid")
        if _sha(result_path) != entry["result_sha256"]:
            raise IntegrationIntegrityError(f"integration candidate {cid} result hash mismatch")
        result = _read_canonical(result_path)
        if not isinstance(result, dict):
            raise IntegrationError(f"integration candidate {cid} result is invalid")
        expected_lineage = {key: request[key] for key in lineage_keys}
        _validate_result_contract(
            result,
            candidate_id=cid,
            expected_lineage=expected_lineage,
            request_sha256=request_hash,
            status=centry["status"],
            out=out,
        )
        if (centry["status"] == "completed") != result["passed"]:
            raise IntegrationError(f"integration candidate {cid} status mismatch")
        candidates[cid] = {
            "result": result,
            "result_path": entry["result_path"],
            "result_sha256": entry["result_sha256"],
            "status": centry["status"],
        }
    return {
        "request": request,
        "request_sha256": request_hash,
        "integration": integration,
        "integration_sha256": _sha(integration_path),
        "complete": complete,
        "integration_complete_sha256": _sha(complete_path),
        "candidates": candidates,
        "root": out,
    }


def _validate_result_contract(
    value: dict[str, Any],
    *,
    candidate_id: str,
    expected_lineage: dict[str, Any],
    request_sha256: str,
    status: str,
    out: Path,
) -> None:
    if value.get("schema_version") != 1 or value.get("candidate_id") != candidate_id:
        raise IntegrationError("integration result schema mismatch")
    lineage_keys = (
        "corpus_sha256",
        "book_spec_sha256",
        "candidate_spec_sha256",
        "integration_spec_sha256",
        "source_sha256",
        "benchmark_id",
        "book_id",
    )
    if any(value.get(key) != expected_lineage.get(key) for key in lineage_keys):
        raise IntegrationError("integration result lineage mismatch")
    if value.get("request_sha256") != request_sha256 or type(value.get("passed")) is not bool:
        raise IntegrationError("integration result request lineage mismatch")
    base_counters = (
        "resume_duplicate_operations",
        "reasoning_tokens",
        "model_mismatch_count",
        "unknown_required_usage_count",
    )
    if any(type(value.get(key)) is not int or value[key] < 0 for key in base_counters):
        raise IntegrationError("integration result counters are invalid")
    if any(
        type(value.get(key)) is not bool
        for key in ("canary_passed", "expected_interruption_observed", "readiness_passed")
    ):
        raise IntegrationError("integration result predicates are invalid")
    if any(
        not isinstance(value.get(key), dict) or type(value[key].get("structural_pass")) is not bool
        for key in ("structural", "mono", "bilingual")
    ):
        raise IntegrationError("integration structural predicates are invalid")
    if status == "failed":
        if value["passed"]:
            raise IntegrationError("failed integration result pass predicate is invalid")
        return
    if status != "completed":
        raise IntegrationError("integration result status is invalid")
    counters = (
        *base_counters,
        "committed_batches",
        "skipped_batches",
        "remaining_batches",
        "repeated_batches",
    )
    if any(type(value.get(key)) is not int or value[key] < 0 for key in counters):
        raise IntegrationError("completed integration result counters are invalid")
    required_booleans = ("canary_passed", "expected_interruption_observed", "readiness_passed")
    if any(type(value.get(key)) is not bool for key in required_booleans):
        raise IntegrationError("integration result predicates are invalid")
    if any(
        not isinstance(value.get(key), dict) or type(value[key].get("structural_pass")) is not bool
        for key in ("structural", "mono", "bilingual")
    ):
        raise IntegrationError("integration structural predicates are invalid")
    evidence_pass = bool(
        value["canary_passed"]
        and value["expected_interruption_observed"]
        and value["readiness_passed"]
        and value["resume_duplicate_operations"] == 0
        and value["structural"]["structural_pass"]
        and value["mono"]["structural_pass"]
        and value["bilingual"]["structural_pass"]
        and value["reasoning_tokens"] == 0
        and value["model_mismatch_count"] == 0
        and value["unknown_required_usage_count"] == 0
    )
    if value["passed"] != evidence_pass:
        raise IntegrationError("integration result pass evidence mismatch")
    if not evidence_pass:
        return
    timings = value.get("phase_timings_ms")
    if (
        not isinstance(timings, dict)
        or any(
            type(timings.get(key)) is not int or timings[key] < 0
            for key in ("prepare", "translate", "quality")
        )
        or any(
            type(timings.get(key)) is not int or timings[key] <= 0
            for key in ("first_attempt", "resume", "total")
        )
        or timings["total"] < timings["first_attempt"] + timings["resume"]
    ):
        raise IntegrationError("integration phase timing evidence is missing")
    output_paths = value.get("output_paths")
    output_hashes = value.get("output_sha256")
    if (
        not isinstance(output_paths, dict)
        or not isinstance(output_hashes, dict)
        or set(output_paths) != {"mono", "bilingual"}
        or set(output_hashes) != {"mono", "bilingual"}
    ):
        raise IntegrationError("integration output evidence is missing")
    for output_key in ("mono", "bilingual"):
        relative = output_paths.get(output_key)
        if not isinstance(relative, str):
            raise IntegrationError("integration output evidence is missing")
        path = (out / relative).resolve()
        if (
            _safe_relative(path, out) != relative
            or not path.is_file()
            or _sha(path) != output_hashes.get(output_key)
        ):
            raise IntegrationIntegrityError("integration output evidence is missing or tampered")
    evidence_paths = (
        ("telemetry_path", "telemetry_sha256"),
        ("first_telemetry_path", "first_telemetry_sha256"),
        ("resume_telemetry_path", "resume_telemetry_sha256"),
        ("usage_path", "usage_sha256"),
    )
    physical: dict[str, Path] = {}
    for evidence_key, hash_key in evidence_paths:
        relative = value.get(evidence_key)
        if not isinstance(relative, str):
            raise IntegrationError("integration physical evidence is missing")
        path = (out / relative).resolve()
        if (
            _safe_relative(path, out) != relative
            or not path.is_file()
            or _sha(path) != value.get(hash_key)
        ):
            raise IntegrationIntegrityError("integration physical evidence is missing or tampered")
        physical[evidence_key] = path
    try:
        telemetry = _telemetry_records(physical["telemetry_path"])
        first = _telemetry_records(physical["first_telemetry_path"])
        resume = _telemetry_records(physical["resume_telemetry_path"])
    except IntegrationError:
        raise
    for key, rows in (
        ("first_telemetry_count", first),
        ("resume_telemetry_count", resume),
    ):
        if type(value.get(key)) is not int or value[key] < 0 or value[key] != len(rows):
            raise IntegrationIntegrityError("integration telemetry record count mismatch")
    resume_attempt_count = value.get("resume_attempt_telemetry_count")
    if (
        type(resume_attempt_count) is not int
        or resume_attempt_count < 0
        or resume_attempt_count > len(resume)
    ):
        raise IntegrationIntegrityError("integration telemetry record count mismatch")
    telemetry_counts = value.get("telemetry_counts")
    expected_counts = {
        "logical_call_count": len({record.logical_call_id for record in telemetry}),
        "attempt_count": len(telemetry),
        "operation_count": len({record.operation for record in telemetry}),
        "agent_count": len({record.agent for record in telemetry}),
        "retry_count": sum(
            1 for record in telemetry if record.attempt_index > 1 or record.retry_class is not None
        ),
        "translate_call_count": sum(
            1 for record in telemetry if record.operation == "translate.batch"
        ),
    }
    if not isinstance(telemetry_counts, dict) or any(
        type(telemetry_counts.get(key)) is not int
        or telemetry_counts[key] < 0
        or telemetry_counts[key] != expected
        for key, expected in expected_counts.items()
    ):
        raise IntegrationIntegrityError("integration telemetry count mismatch")
    usage = _usage_evidence(physical["usage_path"])
    if not usage.get("valid") or usage.get("unknown_required_usage_count") != 0:
        raise IntegrationIntegrityError("integration usage evidence is invalid")


def _node_phase_timings(store: RunStore) -> dict[str, int]:
    state = store.load_state()
    totals = {"prepare": 0, "translate": 0, "quality": 0}
    seen = {"prepare": False, "translate": False, "quality": False}
    phases = {
        "prepare": {"prepare", "analyze", "digest", "mine_terms", "name_terms", "book_synopsis"},
        "translate": {"translate", "titles"},
        "quality": {
            "polish",
            "naturalize",
            "review",
            "backtranslate",
            "consistency_qa",
            "report",
            "assemble",
        },
    }
    phase_by_node = {node: phase for phase, nodes in phases.items() for node in nodes}
    for node_id, node in state.nodes.items():
        if not node.started_at or not node.finished_at:
            continue
        base = node_id.split(":", 1)[0].casefold()
        phase = phase_by_node.get(base)
        if phase is None:
            raise IntegrationError(f"unknown production node timing: {node_id}")
        try:
            started = datetime.fromisoformat(node.started_at.replace("Z", "+00:00"))
            finished = datetime.fromisoformat(node.finished_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise IntegrationError(f"invalid production node timing: {node_id}") from error
        elapsed = (finished - started).total_seconds() * 1000
        if elapsed < 0:
            raise IntegrationError(f"production node timing is negative: {node_id}")
        totals[phase] += round(elapsed)
        seen[phase] = True
    if not all(seen.values()):
        raise IntegrationError("required production phase timing evidence is missing")
    return totals


def _quality_config(spec: CandidateSpec, candidate: Candidate, state_dir: Path) -> Config:
    """Reuse FullRunner's production quality configuration, not a second profile."""
    config = FullRunner._config(
        spec,
        candidate.primary_model,
        candidate.editor_model,
        quality=True,
        state_dir=str(state_dir),
    )
    config.source_lang = "en"
    config.target_lang = "zh"
    config.output.mono = True
    config.output.bilingual = True
    return config


class IntegrationRunner:
    """Run or resume isolated candidate integrations under one output directory."""

    def __init__(
        self,
        *,
        client_factory: Callable[..., Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.client_factory = client_factory
        self.client = client
        self._created_client_ids: set[int] = set()

    def _client(
        self,
        spec: CandidateSpec,
        candidate: Candidate,
        options: GenerationOptions,
        sink: _JsonlTelemetrySink,
    ) -> Any:
        roles = ModelRoles(
            primary=candidate.primary_model,
            editor=candidate.editor_model,
            fast=spec.fast_model,
        )
        if self.client is not None:
            raise IntegrationError(
                "IntegrationRunner requires a client_factory for distinct clients"
            )
        client = _model_client(
            spec,
            candidate.primary_model,
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

    @staticmethod
    def _canary(
        client: Any,
        candidate: Candidate,
        spec: CandidateSpec,
        sink: _JsonlTelemetrySink,
    ) -> dict[str, Any]:
        calls = (
            (
                "translator",
                "integration.canary.translate",
                "文学翻译",
                "synthetic canary source",
                candidate.primary_model,
            ),
            (
                "editor",
                "integration.canary.polish",
                "中文润色编辑",
                "synthetic canary target",
                candidate.editor_model,
            ),
            (
                "reviewer",
                "integration.canary.review",
                "译文审校",
                "synthetic canary target",
                spec.fast_model,
            ),
        )
        try:
            for agent, operation, marker, text, _model in calls:
                client.complete(
                    [
                        {"role": "system", "content": marker},
                        {"role": "user", "content": f"[0] {text}"},
                    ],
                    json_mode=True,
                    agent=agent,
                    operation=operation,
                )
        except Exception as error:
            return {
                "schema_version": 1,
                "passed": False,
                "reason": type(error).__name__,
                "roles": [row[0] for row in calls],
                "provider": spec.provider,
                "primary_model": candidate.primary_model,
                "editor_model": candidate.editor_model,
                "fast_model": spec.fast_model,
                "temperature": spec.temperature,
                "seed": spec.seed,
                "reasoning_tokens": 0,
                "model_mismatch_count": 0,
                "unknown_required_usage_count": 1,
            }
        records = sink.records[-3:]
        expected_models = [
            _normalized_model(candidate.primary_model),
            _normalized_model(candidate.editor_model),
            _normalized_model(spec.fast_model),
        ]
        if len(records) != 3:
            return {
                "schema_version": 1,
                "passed": False,
                "reason": "missing_canary_telemetry",
                "unknown_required_usage_count": 1,
                "reasoning_tokens": 0,
                "model_mismatch_count": 0,
            }
        reasoning = 0
        mismatch = 0
        unknown = 0
        for index, (record, expected) in enumerate(zip(records, expected_models, strict=True)):
            try:
                value = CallAttemptTelemetry.model_validate(record)
            except Exception:
                unknown += 1
                continue
            expected_agent, expected_operation = calls[index][0], calls[index][1]
            reasoning += value.reasoning_tokens
            unknown += int(value.billed_usage_unknown)
            mismatch += int(
                value.agent != expected_agent
                or value.operation != expected_operation
                or value.provider != spec.provider
                or _normalized_model(value.requested_model) != expected
                or _normalized_model(value.resolved_model) != expected
                or value.reasoning_enabled
                or value.status != "success"
                or value.temperature != spec.temperature
                or value.seed is not None
            )
        passed = reasoning == 0 and mismatch == 0 and unknown == 0
        return {
            "schema_version": 1,
            "passed": passed,
            "roles": [row[0] for row in calls],
            "provider": spec.provider,
            "primary_model": candidate.primary_model,
            "editor_model": candidate.editor_model,
            "fast_model": spec.fast_model,
            "temperature": spec.temperature,
            "seed": spec.seed,
            "reasoning_tokens": reasoning,
            "model_mismatch_count": mismatch,
            "unknown_required_usage_count": unknown,
        }

    def _preflight(
        self,
        corpus_dir: Path,
        book_spec_path: Path,
        candidate_spec_path: Path,
        integration_spec_path: Path,
    ) -> tuple[IntegrationSpec, CandidateSpec, Path, str, dict[str, Any]]:
        corpus_dir = corpus_dir.resolve()
        book_spec_path = book_spec_path.resolve()
        candidate_spec_path = candidate_spec_path.resolve()
        integration_spec_path = integration_spec_path.resolve()
        corpus_value = validate_corpus(corpus_dir)
        corpus_hash = corpus_value.get("corpus_sha256")
        raw_candidate_hash = _sha(candidate_spec_path)
        try:
            import yaml

            raw_integration = yaml.safe_load(integration_spec_path.read_text(encoding="utf-8"))
            spec = IntegrationSpec.model_validate(raw_integration)
        except Exception as error:
            raise IntegrationError(f"invalid IntegrationSpec: {error}") from error
        if spec.corpus_sha256 != corpus_hash:
            raise IntegrationError("integration corpus hash mismatch")
        if spec.candidate_spec_sha256 != raw_candidate_hash:
            raise IntegrationError("integration candidate spec hash mismatch")
        candidate_spec = load_candidate_spec(candidate_spec_path)
        if candidate_spec.benchmark_id != spec.benchmark_id:
            raise IntegrationError("benchmark_id mismatch")
        if candidate_spec.temperature != 0.1 or candidate_spec.seed is not None:
            raise IntegrationError("candidate generation must be temperature 0.1 and seed None")
        options = validate_candidate_capabilities(candidate_spec)
        selected = {candidate.candidate_id: candidate for candidate in candidate_spec.candidates}
        if any(cid not in selected for cid in spec.candidate_ids):
            raise IntegrationError("integration selects an unknown candidate")
        chosen = [selected[cid] for cid in spec.candidate_ids]
        book_spec = load_book_spec(book_spec_path)
        hidden = [
            book
            for book in book_spec.books
            if book.book_id == spec.book_id and book.split == "hidden"
        ]
        if len(hidden) != 1:
            raise IntegrationError("book_id must select exactly one hidden book")
        source = Path(hidden[0].path).expanduser()
        if not source.is_absolute():
            source = book_spec_path.parent / source
        source = source.resolve()
        try:
            source.relative_to(book_spec_path.parent)
        except ValueError as error:
            raise IntegrationError("hidden source escapes BOOK_SPEC root") from error
        if (
            source.suffix.lower() != ".epub"
            or not source.is_file()
            or not os.access(source, os.R_OK)
        ):
            raise IntegrationError("selected hidden source must be a readable EPUB")
        manifest = _read_json(corpus_dir / "source_manifest.json")
        rows = {row.get("book_id"): row for row in manifest.get("books", [])}
        source_hash = _sha(source)
        manifest_row = rows.get(spec.book_id)
        if (
            not manifest_row
            or manifest_row.get("split") != "hidden"
            or manifest_row.get("source_sha256") != source_hash
        ):
            raise IntegrationError("hidden source does not match frozen source manifest")
        runner_rows = []
        runner_path = corpus_dir / "runner_segments.jsonl"
        if runner_path.exists():
            runner_rows = [
                json.loads(line)
                for line in runner_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if any(
            row.get("subset") != "hidden"
            and rows.get(row.get("book_id"), {}).get("source_sha256") == source_hash
            for row in runner_rows
        ):
            raise IntegrationError("hidden source aliases a screen/formal corpus book")
        if options.temperature != 0.1 or options.seed is not None:
            raise IntegrationError("generation options must be exact")
        return (
            spec,
            candidate_spec,
            source,
            source_hash,
            {
                "book_spec_sha256": _sha(book_spec_path),
                "candidate_spec_sha256": raw_candidate_hash,
                "integration_spec_sha256": _sha(integration_spec_path),
                "selected": chosen,
                "source_sha256": source_hash,
            },
        )

    def run(
        self,
        corpus_dir: str | os.PathLike[str],
        book_spec_path: str | os.PathLike[str],
        candidate_spec_path: str | os.PathLike[str],
        integration_spec_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        corpus = Path(corpus_dir).expanduser()
        book_spec = Path(book_spec_path).expanduser()
        candidates = Path(candidate_spec_path).expanduser()
        integration_input = Path(integration_spec_path).expanduser()
        out = Path(out_dir).expanduser().resolve()
        self._created_client_ids.clear()
        spec, candidate_spec, source, source_hash, lineage = self._preflight(
            corpus, book_spec, candidates, integration_input
        )
        out.mkdir(parents=True, exist_ok=True)
        chosen: list[Candidate] = lineage["selected"]
        _validate_all_candidate_paths(out, source, list(spec.candidate_ids), spec.book_id)
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
                candidate.candidate_id: {
                    "provider": candidate_spec.provider,
                    "primary_model": candidate.primary_model,
                    "editor_model": candidate.editor_model,
                    "fast_model": candidate_spec.fast_model,
                    "temperature": candidate_spec.temperature,
                    "seed": candidate_spec.seed,
                }
                for candidate in chosen
            },
            "interrupt_after_committed_batches": spec.interrupt_after_committed_batches,
            "output_mono": spec.output_mono,
            "output_bilingual": spec.output_bilingual,
            "bilingual_order": spec.bilingual_order,
        }
        request_path = out / "integration_request.json"
        state_path = out / "integration_state.json"
        if not request_path.exists() and any(
            path.exists()
            for path in (
                state_path,
                out / "integration.json",
                out / "integration_complete.json",
                out / "candidates",
            )
        ):
            raise IntegrationError("integration request missing for existing artifacts")
        request_created = not request_path.exists()
        if request_path.exists() and _read_canonical(request_path) != request:
            raise IntegrationError("integration immutable request mismatch")
        if request_created:
            _atomic_json(request_path, request)
        state_path = out / "integration_state.json"
        if not state_path.exists() and not request_created:
            raise IntegrationError("integration state missing for existing request")
        state = (
            _read_canonical(state_path)
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
            not isinstance(value, dict) or value.get("status") not in allowed
            for value in state["candidates"].values()
        ):
            raise IntegrationError("integration state contains invalid candidate status")
        integration_path = out / "integration.json"
        complete_path = out / "integration_complete.json"
        if complete_path.exists() or integration_path.exists():
            validated = validate_terminal_artifacts(out)
            integration = validated["integration"]
            complete = validated["complete"]
            expected_hash = validated["integration_sha256"]
            expected_lineage = {
                "schema_version": 1,
                "benchmark_id": spec.benchmark_id,
                "corpus_sha256": spec.corpus_sha256,
                "book_spec_sha256": lineage["book_spec_sha256"],
                "candidate_spec_sha256": lineage["candidate_spec_sha256"],
                "integration_spec_sha256": lineage["integration_spec_sha256"],
                "book_id": spec.book_id,
                "source_sha256": source_hash,
            }
            if any(integration.get(key) != expected_lineage[key] for key in expected_lineage):
                raise IntegrationError("integration manifest lineage mismatch")
            failed: list[str] = []
            for cid in spec.candidate_ids:
                entry = integration["candidates"][cid]
                completion_entry = complete["candidates"][cid]
                if state["candidates"][cid].get("status") != completion_entry["status"]:
                    raise IntegrationError("integration state terminal status mismatch")
                if state["candidates"][cid].get("result_sha256") != entry["result_sha256"]:
                    raise IntegrationError("integration result lineage mismatch")
                state_entry = state["candidates"][cid]
                if "boundary_event_count" in state_entry:
                    _validate_restart_prefixes(
                        state_entry,
                        candidate_store=_candidate_store(out / "candidates" / cid / "state"),
                        telemetry_path=out / "candidates" / cid / "telemetry.jsonl",
                    )
                if completion_entry["status"] == "failed":
                    failed.append(cid)
            return {
                **integration,
                "out_dir": str(out),
                "no_op": True,
                "failed_candidates": failed,
                "integration_sha256": expected_hash,
                "resumed": True,
            }
        _atomic_json(state_path, state)
        results: dict[str, dict[str, Any]] = {}
        for candidate in chosen:
            cid = candidate.candidate_id
            candidate_state = state["candidates"][cid]
            if candidate_state.get("status") in {"completed", "failed"}:
                result_path = out / "candidates" / cid / "result.json"
                if not result_path.exists():
                    raise IntegrationError("terminal candidate result is missing")
                reused = _read_canonical(result_path)
                _validate_result_contract(
                    reused,
                    candidate_id=cid,
                    expected_lineage={**request, "schema_version": 1},
                    request_sha256=_sha(request_path),
                    status=candidate_state["status"],
                    out=out,
                )
                if candidate_state.get("result_sha256") != _sha(result_path):
                    raise IntegrationError("terminal candidate result hash mismatch")
                results[cid] = reused
                continue
            results[cid] = self._run_candidate(
                spec,
                candidate_spec,
                candidate,
                source,
                source_hash,
                out,
                state,
                state_path,
            )
        manifest_candidates: dict[str, Any] = {}
        completion_candidates: dict[str, Any] = {}
        for cid in sorted(results):
            path = out / "candidates" / cid / "result.json"
            digest = _sha(path)
            relative = _safe_relative(path, out)
            status = "completed" if results[cid].get("passed") else "failed"
            manifest_candidates[cid] = {"result_path": relative, "result_sha256": digest}
            completion_candidates[cid] = {
                "status": status,
                "result_path": relative,
                "result_sha256": digest,
            }
        integration = {
            "schema_version": 1,
            "benchmark_id": spec.benchmark_id,
            "corpus_sha256": spec.corpus_sha256,
            "book_spec_sha256": lineage["book_spec_sha256"],
            "candidate_spec_sha256": lineage["candidate_spec_sha256"],
            "integration_spec_sha256": lineage["integration_spec_sha256"],
            "book_id": spec.book_id,
            "source_sha256": source_hash,
            "candidates": manifest_candidates,
        }
        _atomic_json(integration_path, integration)
        complete = {
            "schema_version": 1,
            "benchmark_id": spec.benchmark_id,
            "integration_sha256": sha256_bytes(_canonical_bytes(integration)),
            "terminal": True,
            "candidates": completion_candidates,
        }
        _atomic_json(complete_path, complete)
        integration_hash = sha256_bytes(integration_path.read_bytes())
        failed_candidates = [cid for cid, item in results.items() if not item.get("passed")]
        return {
            **integration,
            "out_dir": str(out),
            "no_op": False,
            "failed_candidates": failed_candidates,
            "integration_sha256": integration_hash,
            "resumed": not request_created,
        }

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
        cid = candidate.candidate_id
        root = out / "candidates" / cid
        state_dir = root / "state"
        output_dir = root / "outputs"
        telemetry_path = root / "telemetry.jsonl"
        canary_path = root / "canary.json"
        result_path = root / "result.json"
        first_telemetry_path = root / "telemetry.first.jsonl"
        resume_telemetry_path = root / "telemetry.resume.jsonl"
        _validate_candidate_paths(
            out,
            source,
            root,
            state_dir,
            output_dir,
            output_dir / f"{spec.book_id}.epub",
            output_dir / f"{spec.book_id}-bi.epub",
            telemetry_path,
            first_telemetry_path,
            resume_telemetry_path,
            canary_path,
            result_path,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        options = GenerationOptions(**_GENERATION_FIELDS)
        result: dict[str, Any] = {
            "schema_version": 1,
            "candidate_id": cid,
            "passed": False,
            "canary_passed": False,
            "expected_interruption_observed": False,
            "resume_duplicate_operations": 0,
            "readiness_passed": False,
            "readiness_problem_count": 0,
            "readiness_codes": [],
            "structural": {"structural_pass": False},
            "mono": {"structural_pass": False},
            "bilingual": {"structural_pass": False},
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
            "unknown_required_usage_count": 0,
            "source_sha256": source_hash,
            "benchmark_id": spec.benchmark_id,
            "book_id": spec.book_id,
            "interruption": {"count": 0, "batches": []},
        }
        request_value = _read_canonical(out / "integration_request.json")
        result["request_sha256"] = _sha(out / "integration_request.json")
        for key in (
            "corpus_sha256",
            "book_spec_sha256",
            "candidate_spec_sha256",
            "integration_spec_sha256",
        ):
            result[key] = request_value[key]
        started = time.monotonic()
        candidate_status = state["candidates"][cid]["status"]
        if candidate_status not in {"pending", "interrupted", "resuming"}:
            raise IntegrationError("candidate is not resumable")
        state_entry = dict(state["candidates"][cid])
        try:
            sink = _JsonlTelemetrySink(telemetry_path)
            config = _quality_config(candidate_spec, candidate, state_dir)
            config.output.bilingual_order = spec.bilingual_order
            mono_path = output_dir / f"{spec.book_id}.epub"
            if mono_path.resolve() == source.resolve():
                raise IntegrationError("mono output aliases source EPUB")
            if candidate_status == "pending":
                canary_client = self._client(candidate_spec, candidate, options, sink)
                canary = self._canary(canary_client, candidate, candidate_spec, sink)
                canary_telemetry_count = len(sink.records)
                _atomic_json(canary_path, canary)
                result["reasoning_tokens"] = int(canary.get("reasoning_tokens", 0))
                result["model_mismatch_count"] = int(canary.get("model_mismatch_count", 0))
                result["unknown_required_usage_count"] = int(
                    canary.get("unknown_required_usage_count", 0)
                )
                result["canary_passed"] = bool(canary.get("passed"))
                if not result["canary_passed"]:
                    raise IntegrationError("synthetic three-role canary failed")
                first_client = self._client(candidate_spec, candidate, options, sink)
                hook = _CommitHook(spec.interrupt_after_committed_batches)
                state["candidates"][cid] = {
                    "status": "pending",
                    "state_path": _safe_relative(state_dir, out),
                }
                _atomic_json(state_path, state)
                try:
                    Application(config, client=first_client, batch_commit_hook=hook).run_all(
                        str(source), out_format="epub", out_path=str(mono_path)
                    )
                except BenchmarkInterruption as exc:
                    candidate_store = _candidate_store(state_dir)
                    interrupted_events = _events(candidate_store)
                    persisted = _batches(interrupted_events)
                    if (
                        not hook.reached
                        or not hook.raised
                        or not persisted
                        or hook.committed[-1]
                        != {
                            "chapter": persisted[-1]["chapter"],
                            "start": persisted[-1]["start"],
                            "count": persisted[-1]["count"],
                        }
                    ):
                        raise IntegrationError("unverified benchmark interruption") from exc
                    boundary_event_count = len(interrupted_events)
                    before_batches = persisted
                    event_path = Path(candidate_store.event_log_path)
                    event_bytes = event_path.read_bytes()
                    telemetry_bytes = telemetry_path.read_bytes()
                    first_telemetry_path.write_bytes(telemetry_bytes)
                    first_wall_ms = int((time.monotonic() - started) * 1000)
                    state_entry = {
                        "status": "interrupted",
                        "state_path": _safe_relative(state_dir, out),
                        "canary_sha256": _sha(canary_path),
                        "interruption": {
                            "count": len(hook.committed),
                            "batches": hook.committed,
                        },
                        "canary_telemetry_routes": _canary_routes(
                            [
                                CallAttemptTelemetry.model_validate(value)
                                for value in sink.records[:canary_telemetry_count]
                            ]
                        ),
                        "boundary_event_count": boundary_event_count,
                        "first_boundary_event_count": boundary_event_count,
                        "attempt_event_boundary_count": boundary_event_count,
                        "event_prefix_size": len(event_bytes),
                        "event_prefix_sha256": hashlib.sha256(event_bytes).hexdigest(),
                        "first_event_prefix_size": len(event_bytes),
                        "first_event_prefix_sha256": hashlib.sha256(event_bytes).hexdigest(),
                        "before_target_hashes": before_batches,
                        "first_telemetry_path": _safe_relative(first_telemetry_path, out),
                        "first_telemetry_sha256": _sha(first_telemetry_path),
                        "canary_telemetry_count": canary_telemetry_count,
                        "first_telemetry_count": len(sink.records),
                        "telemetry_prefix_size": len(telemetry_bytes),
                        "telemetry_prefix_sha256": hashlib.sha256(telemetry_bytes).hexdigest(),
                        "first_telemetry_prefix_size": len(telemetry_bytes),
                        "first_telemetry_prefix_sha256": hashlib.sha256(
                            telemetry_bytes
                        ).hexdigest(),
                        "attempt_telemetry_count": len(sink.records),
                        "attempt_telemetry_prefix_size": len(telemetry_bytes),
                        "attempt_telemetry_prefix_sha256": hashlib.sha256(
                            telemetry_bytes
                        ).hexdigest(),
                        "resume_wall_ms": 0,
                        "resume_durations_ms": [],
                        "first_wall_ms": first_wall_ms,
                    }
                    result["expected_interruption_observed"] = True
                    result["interruption"] = state_entry["interruption"]
                    state["candidates"][cid] = state_entry
                    _atomic_json(state_path, state)
                else:
                    raise IntegrationError("configured interruption was not observed")
            else:
                state_entry = state["candidates"][cid]
                if not canary_path.exists():
                    raise IntegrationError("resuming candidate is missing canary evidence")
                canary = _read_canonical(canary_path)
                if state_entry.get("canary_sha256") != _sha(canary_path):
                    raise IntegrationIntegrityError("resuming candidate canary hash mismatch")
                if candidate_status in {"interrupted", "resuming"}:
                    candidate_store = _candidate_store(state_dir)
                    _validate_restart_prefixes(
                        state_entry,
                        candidate_store=candidate_store,
                        telemetry_path=telemetry_path,
                    )
                    canary_telemetry_count = state_entry["canary_telemetry_count"]
                    result["expected_interruption_observed"] = True
                    result["interruption"] = state_entry["interruption"]
                result["canary_passed"] = bool(canary.get("passed"))
                result["reasoning_tokens"] = int(canary.get("reasoning_tokens", 0))
                result["model_mismatch_count"] = int(canary.get("model_mismatch_count", 0))
                result["unknown_required_usage_count"] = int(
                    canary.get("unknown_required_usage_count", 0)
                )
            candidate_store = _candidate_store(state_dir)
            first_events = _events(candidate_store)
            boundary_event_count = int(
                state_entry.get(
                    "attempt_event_boundary_count", state_entry.get("boundary_event_count", 0)
                )
            )
            attempt_telemetry_count = int(
                state_entry.get(
                    "attempt_telemetry_count", state_entry.get("first_telemetry_count", 0)
                )
            )
            if candidate_status != "pending":
                before_batches = list(state_entry["before_target_hashes"])
                if len(first_events) < boundary_event_count:
                    raise IntegrationError("resume event boundary is beyond persisted events")
            if candidate_status == "resuming":
                prior_ids = {
                    (row["chapter"], row["start"], row["count"]): row["target_sha256"]
                    for row in before_batches
                }
                newly_committed = _batches(first_events[boundary_event_count:])
                for row in newly_committed:
                    identity = (row["chapter"], row["start"], row["count"])
                    if identity in prior_ids:
                        if prior_ids[identity] != row["target_sha256"]:
                            raise IntegrationIntegrityError("resume committed target hash changed")
                        raise IntegrationIntegrityError("resume retranslated committed batch")
                old_event_boundary = boundary_event_count
                old_telemetry_boundary = attempt_telemetry_count
                old_event_size = int(state_entry.get("event_prefix_size", 0))
                old_telemetry_size = int(
                    state_entry.get(
                        "attempt_telemetry_prefix_size", state_entry.get("telemetry_prefix_size", 0)
                    )
                )
                if newly_committed:
                    before_batches.extend(newly_committed)
                telemetry_records = _telemetry_records(telemetry_path)
                if len(telemetry_records) < attempt_telemetry_count:
                    raise IntegrationError("resume telemetry boundary is beyond persisted attempts")
                boundary_event_count = len(first_events)
                attempt_telemetry_count = len(telemetry_records)
                event_path = Path(candidate_store.event_log_path)
                event_raw = event_path.read_bytes()
                telemetry_raw = telemetry_path.read_bytes()
                if len(event_raw) < old_event_size or len(telemetry_raw) < old_telemetry_size:
                    raise IntegrationError("crashed resume evidence is truncated")
                recovered_event_raw = event_raw[old_event_size:]
                recovered_telemetry_raw = telemetry_raw[old_telemetry_size:]
                recovered_events = (
                    _events_from_bytes(recovered_event_raw) if recovered_event_raw else []
                )
                recovered_telemetry = [
                    CallAttemptTelemetry.model_validate(json.loads(line))
                    for line in recovered_telemetry_raw.decode("utf-8").splitlines()
                    if line.strip()
                ]
                if _batches(recovered_events) != newly_committed:
                    raise IntegrationIntegrityError("crashed resume event slice mismatch")
                recovered_duration_ms = _recovered_active_duration_ms(
                    state_entry.get("resume_started_at"),
                    events=recovered_events,
                    telemetry=recovered_telemetry,
                )
                if recovered_duration_ms:
                    durations = list(state_entry.get("resume_durations_ms", []))
                    durations.append(recovered_duration_ms)
                    state_entry["resume_durations_ms"] = durations
                    state_entry["resume_wall_ms"] = (
                        int(state_entry.get("resume_wall_ms", 0)) + recovered_duration_ms
                    )
                state_entry["recovered_resume"] = {
                    "event_boundary_count": old_event_boundary,
                    "event_count": len(recovered_events),
                    "event_sha256": hashlib.sha256(recovered_event_raw).hexdigest(),
                    "telemetry_boundary_count": old_telemetry_boundary,
                    "telemetry_count": len(recovered_telemetry),
                    "telemetry_sha256": hashlib.sha256(recovered_telemetry_raw).hexdigest(),
                    "active_duration_ms": recovered_duration_ms,
                }
                state_entry.update(
                    {
                        "before_target_hashes": before_batches,
                        "boundary_event_count": boundary_event_count,
                        "attempt_event_boundary_count": boundary_event_count,
                        "event_prefix_size": len(event_raw),
                        "event_prefix_sha256": hashlib.sha256(event_raw).hexdigest(),
                        "attempt_telemetry_count": attempt_telemetry_count,
                        "attempt_telemetry_prefix_size": len(telemetry_raw),
                        "attempt_telemetry_prefix_sha256": hashlib.sha256(
                            telemetry_raw
                        ).hexdigest(),
                        "telemetry_prefix_size": len(telemetry_raw),
                        "telemetry_prefix_sha256": hashlib.sha256(telemetry_raw).hexdigest(),
                    }
                )
                _atomic_json(
                    state_path, {**state, "candidates": {**state["candidates"], cid: state_entry}}
                )
            before_ids = {
                (row["chapter"], row["start"], row["count"]): row["target_sha256"]
                for row in before_batches
            }
            result["before_target_hashes"] = before_batches
            sink_resume = _JsonlTelemetrySink(telemetry_path)
            first_telemetry_count = int(state_entry["first_telemetry_count"])
            resume_attempt_telemetry_count = int(
                state_entry.get("attempt_telemetry_count", first_telemetry_count)
            )
            resume_attempt_started = time.monotonic()
            resume_client = self._client(candidate_spec, candidate, options, sink_resume)
            state["candidates"][cid] = {
                **state_entry,
                "status": "resuming",
                "state_path": _safe_relative(state_dir, out),
                "resume_attempt": int(state_entry.get("resume_attempt", 0)) + 1,
                "resume_started_wall_ms": int((time.monotonic() - started) * 1000),
                "resume_started_at": datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z"),
            }
            _atomic_json(state_path, state)
            result_value = Application(config, client=resume_client).run_all(
                str(source), out_format="epub", out_path=str(mono_path)
            )
            telemetry_raw = telemetry_path.read_bytes()
            first_telemetry_prefix_size = int(
                state_entry.get("first_telemetry_prefix_size", state_entry["telemetry_prefix_size"])
            )
            current_attempt_prefix_size = int(
                state_entry.get(
                    "attempt_telemetry_prefix_size", state_entry["telemetry_prefix_size"]
                )
            )
            resume_raw = telemetry_raw[first_telemetry_prefix_size:]
            current_resume_raw = telemetry_raw[current_attempt_prefix_size:]
            resume_telemetry_path.write_bytes(resume_raw)
            current_resume_telemetry = _telemetry_evidence(
                telemetry_path,
                candidate=candidate,
                candidate_spec=candidate_spec,
                start_index=resume_attempt_telemetry_count,
            )
            current_resume_elapsed = max(
                int((time.monotonic() - resume_attempt_started) * 1000),
                0,
            )
            events = _events(candidate_store)
            all_batches = _batches(events)
            post_events = events[boundary_event_count:]
            remaining_batches = len(_batches(post_events))
            post_skips = _skips(post_events)
            skip_ids = [
                (row["chapter"], row["start"], row["count"], row["target_sha256"])
                for row in post_skips
            ]
            required_ids = [
                (row["chapter"], row["start"], row["count"], row["target_sha256"])
                for row in before_batches
            ]
            skip_counts = Counter(skip_ids)
            required_counts = Counter(required_ids)
            reuse_complete = (
                not post_skips
                and remaining_batches == 0
                and all(
                    any(
                        item["chapter"] == row["chapter"]
                        and item["start"] == row["start"]
                        and item["count"] == row["count"]
                        and item["target_sha256"] == row["target_sha256"]
                        for item in all_batches
                    )
                    for row in before_batches
                )
            )
            if not reuse_complete and (
                any(identity not in required_counts for identity in skip_counts)
                or any(skip_counts[identity] < count for identity, count in required_counts.items())
            ):
                raise IntegrationError("resume reuse proof is incomplete or mismatched")
            repeated = [
                row
                for row in _batches(post_events)
                if (row["chapter"], row["start"], row["count"]) in before_ids
            ]
            final_ids = {
                (row["chapter"], row["start"], row["count"]): row["target_sha256"]
                for row in all_batches
            }
            changed = [
                identity
                for identity, target_hash in before_ids.items()
                if final_ids.get(identity) != target_hash
            ]
            result["resume_duplicate_operations"] = len(repeated) + len(changed)
            result["committed_batches"] = len(before_batches)
            result["skipped_batches"] = len(post_skips)
            result["remaining_batches"] = remaining_batches
            result["repeated_batches"] = len(repeated)
            result["final_target_hashes"] = all_batches
            if state_entry.get("recovered_resume") is not None:
                result["recovered_resume"] = state_entry["recovered_resume"]
            readiness_problems = assemble_readiness_problems(candidate_store)
            result["readiness_problem_count"] = len(readiness_problems)
            result["readiness_codes"] = sorted(
                {
                    hashlib.sha256(problem.encode("utf-8")).hexdigest()[:16]
                    for problem in readiness_problems
                }
            )
            result["readiness_passed"] = not readiness_problems
            telemetry = _telemetry_evidence(
                telemetry_path, candidate=candidate, candidate_spec=candidate_spec
            )
            usage = _usage_evidence(Path(candidate_store.usage_path))
            application_attempt_count = max(0, telemetry["attempt_count"] - canary_telemetry_count)
            if usage["valid"] and usage["attempts"] != application_attempt_count:
                raise IntegrationError("RunStore usage and application telemetry mismatch")
            result["telemetry_counts"] = {
                key: telemetry[key]
                for key in (
                    "logical_call_count",
                    "attempt_count",
                    "operation_count",
                    "agent_count",
                    "retry_count",
                    "translate_call_count",
                )
            }
            result["unknown_required_usage_count"] = int(
                telemetry["unknown_required_usage_count"]
            ) + int(usage["unknown_required_usage_count"])
            result["reasoning_tokens"] = telemetry["reasoning_tokens"]
            result["model_mismatch_count"] = telemetry["model_mismatch_count"]
            if (
                current_resume_telemetry["valid"]
                and current_resume_telemetry["translate_call_count"] != remaining_batches
            ):
                raise IntegrationError("resume translation call attribution mismatch")
            outputs = result_value.get("outputs", [])
            bilingual_path = output_dir / f"{spec.book_id}-bi.epub"
            if str(bilingual_path) not in outputs and not bilingual_path.exists():
                raise IntegrationError("bilingual output missing")
            result["output_paths"] = {
                "mono": _safe_relative(mono_path, out),
                "bilingual": _safe_relative(bilingual_path, out),
            }
            result["output_sha256"] = {
                "mono": _sha(mono_path),
                "bilingual": _sha(bilingual_path),
            }
            structural = validate_epub_triplet(source, mono_path, bilingual_path)
            if not telemetry_path.exists():
                raise IntegrationError("telemetry artifact missing")
            result["telemetry_path"] = _safe_relative(telemetry_path, out)
            result["telemetry_sha256"] = _sha(telemetry_path)
            first_relative = state_entry.get("first_telemetry_path")
            first_hash = state_entry.get("first_telemetry_sha256")
            first_count = state_entry.get("first_telemetry_count")
            first_path = (
                (out / first_relative).resolve() if isinstance(first_relative, str) else out
            )
            if (
                not isinstance(first_relative, str)
                or not isinstance(first_hash, str)
                or type(first_count) is not int
                or _safe_relative(first_path, out) != first_relative
                or not first_path.is_file()
                or _sha(first_path) != first_hash
            ):
                raise IntegrationError("first telemetry evidence is missing or tampered")
            if len(_telemetry_records(first_path)) != first_count:
                raise IntegrationError("first telemetry evidence count mismatch")
            usage_path = Path(candidate_store.usage_path)
            if not usage_path.exists():
                raise IntegrationError("RunStore usage artifact missing")
            result["usage_path"] = _safe_relative(usage_path, out)
            result["first_telemetry_path"] = state_entry.get("first_telemetry_path")
            result["first_telemetry_sha256"] = state_entry.get("first_telemetry_sha256")
            result["first_telemetry_count"] = state_entry.get("first_telemetry_count")
            result["resume_telemetry_path"] = _safe_relative(resume_telemetry_path, out)
            result["resume_telemetry_sha256"] = _sha(resume_telemetry_path)
            result["resume_telemetry_count"] = len(resume_raw.splitlines())
            result["resume_attempt_telemetry_count"] = len(current_resume_raw.splitlines())
            first_wall_ms = int(state_entry.get("first_wall_ms", 0))
            prior_resume_ms = int(state_entry.get("resume_wall_ms", 0))
            prior_resume_durations = list(state_entry.get("resume_durations_ms", []))
            cumulative_resume_ms = prior_resume_ms + current_resume_elapsed
            resume_durations = [*prior_resume_durations, current_resume_elapsed]
            total_wall_ms = first_wall_ms + cumulative_resume_ms
            state_entry.update(
                {
                    "resume_attempt": int(state_entry.get("resume_attempt", 0)) + 1,
                    "resume_wall_ms": cumulative_resume_ms,
                    "resume_durations_ms": resume_durations,
                    "resume_telemetry_count": len(resume_raw.splitlines()),
                }
            )
            phase_timings = _node_phase_timings(candidate_store)
            result["phase_timings_ms"] = {
                **phase_timings,
                "first_attempt": first_wall_ms,
                "resume": cumulative_resume_ms,
                "total": total_wall_ms,
            }
            result["usage_sha256"] = _sha(usage_path)
            result["structural"] = structural
            result["mono"] = structural.get("mono", {"structural_pass": False})
            result["bilingual"] = structural.get("bilingual", {"structural_pass": False})
            result["passed"] = bool(
                result["canary_passed"]
                and result["expected_interruption_observed"]
                and result["readiness_passed"]
                and result["resume_duplicate_operations"] == 0
                and result["mono"].get("structural_pass") is True
                and result["bilingual"].get("structural_pass") is True
                and structural.get("structural_pass") is True
                and result["unknown_required_usage_count"] == 0
                and result["reasoning_tokens"] == 0
                and result["model_mismatch_count"] == 0
            )
            result["wall_time_seconds"] = total_wall_ms / 1000
            result["total_wall_ms"] = total_wall_ms
            request_value = _read_canonical(out / "integration_request.json")
            result["request_sha256"] = _sha(out / "integration_request.json")
            for key in (
                "corpus_sha256",
                "book_spec_sha256",
                "candidate_spec_sha256",
                "integration_spec_sha256",
            ):
                result[key] = request_value[key]
            final_status = "completed" if result["passed"] else "failed"
            state["candidates"][cid] = {
                **state_entry,
                "status": final_status,
                "state_path": _safe_relative(state_dir, out),
                "result_path": _safe_relative(result_path, out),
            }
        except IntegrationIntegrityError:
            raise
        except Exception as error:
            reasons = result.setdefault("failure_reasons", [])
            class_name = type(error).__name__
            if class_name not in reasons:
                reasons.append(class_name)
            code = _failure_code(error)
            if code not in reasons:
                reasons.append(code)
            result["failure_code"] = code
            result["unknown_required_usage_count"] = max(
                1, int(result.get("unknown_required_usage_count", 0))
            )
            result["wall_time_seconds"] = time.monotonic() - started
            state["candidates"][cid] = {
                **state_entry,
                "status": "failed",
                "state_path": _safe_relative(state_dir, out),
                "reason": code,
            }
        _atomic_json(result_path, result)
        state["candidates"][cid]["result_sha256"] = _sha(result_path)
        _atomic_json(state_path, state)
        return result


__all__ = [
    "BenchmarkInterruption",
    "IntegrationError",
    "IntegrationIntegrityError",
    "IntegrationRunner",
    "IntegrationSpec",
    "validate_terminal_artifacts",
]
