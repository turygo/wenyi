"""Candidate/book execution stages for the benchmark runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import atomic_json, read_jsonl, sha256_bytes, write_jsonl
from trans_novel.benchmark.run.candidate import candidate_models
from trans_novel.benchmark.run.evidence import candidate_store, segment_rows, target_hash
from trans_novel.benchmark.run.telemetry import JsonlCallTelemetrySink
from trans_novel.benchmark.schema import Candidate, CandidateSpec
from trans_novel.llm.usage import usage_delta
from trans_novel.pipeline.state import RunStore, clone_closed_runstore


def validate_completed(
    runner: Any,
    out: Path,
    immutable: dict[str, Any],
    state: dict[str, Any],
    by_key: dict[str, Any],
    expected: int,
    candidate_by_id: dict[str, Candidate],
    book_by_id: dict[str, tuple[Any, str]],
    minimal_by_pair: dict[tuple[str, str, str, str], Candidate],
    spec: CandidateSpec,
    error_type: type[Exception],
) -> dict[str, Any] | None:
    if state.get("status") != "completed":
        return None
    if len(by_key) != expected:
        raise error_type("completed benchmark candidate count mismatch")
    for row in by_key.values():
        candidate = candidate_by_id.get(row.get("candidate_id"))
        book_id = row.get("book_id")
        if candidate is None or book_id not in book_by_id:
            raise error_type("completed benchmark candidate identity mismatch")
        minimal_row = None
        if candidate.pipeline_variant == "polish":
            minimal = minimal_by_pair.get(candidate_models(candidate))
            minimal_row = next(
                (
                    value
                    for value in by_key.values()
                    if value.get("candidate_id") == (minimal.candidate_id if minimal else None)
                    and value.get("book_id") == book_id
                    and value.get("replicate") == row.get("replicate")
                ),
                None,
            )
            if minimal_row is None:
                raise error_type("polish candidate is missing its minimal arm")
        runner._validate_artifact(
            out,
            row,
            spec=spec,
            candidate=candidate,
            book_id=book_id,
            source_sha256=book_by_id[book_id][1],
            replicate=row.get("replicate"),
            minimal_row=minimal_row,
        )
    return immutable | {"status": "completed", "branch_count": len(by_key)}


def prepare_polish(
    out: Path,
    state_root: Path,
    minimal: Candidate,
    book_id: str,
    replicate: int,
    error_type: type[Exception],
) -> None:
    minimal_root = (
        out
        / "candidates"
        / minimal.candidate_id
        / _safe_book_id(book_id)
        / f"r{replicate}"
        / "state"
    )
    minimal_store = candidate_store(minimal_root, error_type=error_type)
    destination_store = state_root / Path(minimal_store.run_dir).name
    if destination_store.is_dir() and (destination_store / "manifest.json").is_file():
        return
    state_root.mkdir(parents=True, exist_ok=True)
    clone_closed_runstore(minimal_store.run_dir, str(destination_store))
    cloned = RunStore(str(destination_store))
    cloned_state = cloned.load_state()
    from trans_novel.pipeline.state import NodeState

    for node_key in list(cloned_state.nodes):
        if node_key.split(":", 1)[0] in {
            "polish",
            "titles",
            "deterministic_qa",
            "report",
            "assemble",
        }:
            cloned_state.nodes[node_key] = NodeState(node_id=node_key)
    cloned.save_state(cloned_state)


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _safe_book_id(value: str) -> str:
    safe = "".join(char if (char.isalnum() or char in "-_") else "_" for char in value).strip("_")
    return f"{safe or 'book'}-{_safe_id(value)[:12]}"


def _safe_application(
    config: Any, client: Any, source_path: Path, output_path: Path
) -> dict[str, Any]:
    from trans_novel.pipeline import Application

    return Application(config, client=client).run_all(
        str(source_path), out_format="epub", out_path=str(output_path)
    )


def _execute_application(
    runner: Any,
    out: Path,
    spec: CandidateSpec,
    options: Any,
    candidate: Candidate,
    book: Any,
    source_path: Path,
    source_sha: str,
    replicate: int,
    root: Path,
    state_root: Path,
    minimal_row: dict[str, Any] | None,
    initial_rows: list[dict[str, Any]] | None,
    usage_before: dict[str, Any],
    error_type: type[Exception],
) -> dict[str, Any]:
    telemetry_path = root / "telemetry.jsonl"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.touch(exist_ok=True)
    client = runner._client(spec, candidate, options, JsonlCallTelemetrySink(telemetry_path))
    output_path = root / "outputs" / f"{book.book_id}.epub"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _safe_application(
        runner._config(
            candidate, pipeline_variant=candidate.pipeline_variant, state_dir=str(state_root)
        ),
        client,
        source_path,
        output_path,
    )
    store = candidate_store(state_root, error_type=error_type)
    final_rows = segment_rows(store, source_sha, error_type=error_type)
    initial_rows = final_rows if initial_rows is None else initial_rows
    segments_path = root / "segments.jsonl"
    write_jsonl(segments_path, final_rows)
    outputs = [
        Path(value).expanduser().resolve()
        for value in result.get("outputs", [])
        if isinstance(value, str)
    ]
    if not outputs or any(not path.is_file() for path in outputs):
        raise error_type("production benchmark did not create output EPUBs")
    relative_outputs = [str(path.relative_to(out)) for path in outputs]
    usage = store.load_usage() or {}
    return {
        "artifact_key": runner._artifact_key(spec, candidate, book.book_id, source_sha, replicate),
        "candidate_id": candidate.candidate_id,
        "pipeline_variant": candidate.pipeline_variant,
        "book_id": book.book_id,
        "replicate": replicate,
        "source_sha256": source_sha,
        "state_path": str(state_root.relative_to(out)),
        "segments_path": str(segments_path.relative_to(out)),
        "segments_sha256": sha256_bytes(segments_path.read_bytes()),
        "telemetry_path": str(telemetry_path.relative_to(out)),
        "telemetry_sha256": sha256_bytes(telemetry_path.read_bytes()),
        "outputs": relative_outputs,
        "output_hashes": {
            relative: sha256_bytes((out / relative).read_bytes()) for relative in relative_outputs
        },
        "initial_targets_sha256": target_hash(initial_rows),
        "final_targets_sha256": target_hash(final_rows),
        "usage": usage,
        "polish_incremental_usage": usage_delta(usage, usage_before)
        if candidate.pipeline_variant == "polish"
        else usage_delta({}, {}),
        "translator_model": candidate.translator_model,
        "analyst_model": candidate.analyst_model,
        "editor_model": candidate.editor_model,
        "fast_model": candidate.fast_model,
    }


def execute_candidate(
    runner: Any,
    out: Path,
    spec: CandidateSpec,
    options: Any,
    candidate: Candidate,
    book: Any,
    source_path: Path,
    source_sha: str,
    replicate: int,
    by_key: dict[str, Any],
    minimal_by_pair: dict[tuple[str, str, str, str], Candidate],
    error_type: type[Exception],
) -> tuple[str, dict[str, Any] | None]:
    key = runner._artifact_key(spec, candidate, book.book_id, source_sha, replicate)
    if key in by_key:
        minimal_row = None
        if candidate.pipeline_variant == "polish":
            minimal = minimal_by_pair.get(candidate_models(candidate))
            minimal_row = next(
                (
                    value
                    for value in by_key.values()
                    if minimal is not None
                    and value.get("candidate_id") == minimal.candidate_id
                    and value.get("book_id") == book.book_id
                    and value.get("replicate") == replicate
                ),
                None,
            )
            if minimal_row is None:
                raise error_type("polish candidate is missing its minimal arm")
        runner._validate_artifact(
            out,
            by_key[key],
            spec=spec,
            candidate=candidate,
            book_id=book.book_id,
            source_sha256=source_sha,
            replicate=replicate,
            minimal_row=minimal_row,
        )
        return key, None
    root = (
        out / "candidates" / candidate.candidate_id / _safe_book_id(book.book_id) / f"r{replicate}"
    )
    state_root = root / "state"
    minimal_row = None
    initial_rows = None
    usage_before: dict[str, Any] = {}
    if candidate.pipeline_variant == "polish":
        minimal = minimal_by_pair.get(candidate_models(candidate))
        if minimal is None:
            raise error_type("polish candidate requires a matching minimal candidate")
        minimal_key = runner._artifact_key(spec, minimal, book.book_id, source_sha, replicate)
        minimal_row = by_key.get(minimal_key)
        if minimal_row is None:
            raise error_type("minimal arm must complete before polish arm is created")
        runner._validate_artifact(
            out,
            minimal_row,
            spec=spec,
            candidate=minimal,
            book_id=book.book_id,
            source_sha256=source_sha,
            replicate=replicate,
        )
        prepare_polish(out, state_root, minimal, book.book_id, replicate, error_type)
        initial_rows = read_jsonl(out / minimal_row["segments_path"], error_type=error_type)
        usage_before = minimal_row["usage"]
    row = _execute_application(
        runner,
        out,
        spec,
        options,
        candidate,
        book,
        source_path,
        source_sha,
        replicate,
        root,
        state_root,
        minimal_row,
        initial_rows,
        usage_before,
        error_type,
    )
    runner._validate_artifact(
        out,
        row,
        spec=spec,
        candidate=candidate,
        book_id=book.book_id,
        source_sha256=source_sha,
        replicate=replicate,
        minimal_row=minimal_row,
    )
    return key, row


def execute(
    runner: Any,
    out: Path,
    state_path: Path,
    rows_path: Path,
    state: dict[str, Any],
    by_key: dict[str, Any],
    spec: CandidateSpec,
    options: Any,
    selected_books: list[tuple[Any, Path, bytes]],
    ordered_candidates: list[Candidate],
    minimal_by_pair: dict[tuple[str, str, str, str], Candidate],
    expected: int,
    replicate_count: int,
    error_type: type[Exception],
) -> dict[str, Any]:
    atomic_json(
        state_path,
        {"schema_version": 1, "status": "running", "artifacts": state.get("artifacts", {})},
    )
    try:
        for book, source_path, source_bytes in selected_books:
            source_sha = sha256_bytes(source_bytes)
            for replicate in range(1, replicate_count + 1):
                for candidate in ordered_candidates:
                    key, row = execute_candidate(
                        runner,
                        out,
                        spec,
                        options,
                        candidate,
                        book,
                        source_path,
                        source_sha,
                        replicate,
                        by_key,
                        minimal_by_pair,
                        error_type,
                    )
                    if row is None:
                        continue
                    by_key[key] = row
                    atomic_json(
                        rows_path, sorted(by_key.values(), key=lambda item: item["artifact_key"])
                    )
                    artifacts = dict(state.get("artifacts", {}))
                    artifacts[key] = {
                        "candidate_id": candidate.candidate_id,
                        "pipeline_variant": candidate.pipeline_variant,
                        "book_id": book.book_id,
                        "replicate": replicate,
                    }
                    state = {"schema_version": 1, "status": "running", "artifacts": artifacts}
                    atomic_json(state_path, state)
        if len(by_key) != expected:
            raise error_type("benchmark candidate count mismatch")
        atomic_json(
            state_path,
            {"schema_version": 1, "status": "completed", "artifacts": state.get("artifacts", {})},
        )
        return {"status": "completed", "branch_count": len(by_key)}
    except Exception as error:
        atomic_json(
            state_path,
            {
                "schema_version": 1,
                "status": "failed",
                "artifacts": state.get("artifacts", {}),
                "error": str(error),
            },
        )
        if isinstance(error, error_type):
            raise
        raise error_type(str(error)) from error


__all__ = ["execute", "execute_candidate", "prepare_polish", "validate_completed"]
