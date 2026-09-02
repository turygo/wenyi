"""Run manifest and candidate artifact identity/validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import (
    ArtifactError,
    canonical_json,
    read_jsonl,
    sha256_bytes,
)


def candidate_artifact_key(
    spec: Any,
    candidate: Any,
    book_id: str,
    source_sha256: str,
    replicate: int,
    *,
    generation: dict[str, Any],
    schema_version: int,
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "schema_version": schema_version,
                "benchmark_id": spec.benchmark_id,
                "candidate": candidate.model_dump(mode="python"),
                "generation": generation,
                "pipeline_variant": candidate.pipeline_variant,
                "quality": candidate.pipeline_variant,
                "source_sha256": source_sha256,
                "replicate": replicate,
            }
        ).encode("utf-8")
    )


def validate_candidate_artifact(
    out: Path,
    row: dict[str, Any],
    *,
    spec: Any,
    candidate: Any,
    book_id: str,
    source_sha256: str,
    replicate: int,
    minimal_row: dict[str, Any] | None = None,
    generation: dict[str, Any],
    schema_version: int,
    target_hash: Any,
    candidate_store: Any,
    error_type: type[Exception] = ArtifactError,
) -> None:
    required = {
        "artifact_key",
        "candidate_id",
        "pipeline_variant",
        "book_id",
        "replicate",
        "source_sha256",
        "state_path",
        "segments_path",
        "segments_sha256",
        "telemetry_path",
        "telemetry_sha256",
        "outputs",
        "output_hashes",
        "initial_targets_sha256",
        "final_targets_sha256",
        "usage",
        "polish_incremental_usage",
        "translator_model",
        "analyst_model",
        "editor_model",
        "fast_model",
    }
    if set(row) != required:
        raise error_type("candidate artifact fields invalid")
    expected = candidate_artifact_key(
        spec,
        candidate,
        book_id,
        source_sha256,
        replicate,
        generation=generation,
        schema_version=schema_version,
    )
    values = {
        "artifact_key": expected,
        "candidate_id": candidate.candidate_id,
        "pipeline_variant": candidate.pipeline_variant,
        "book_id": book_id,
        "replicate": replicate,
        "source_sha256": source_sha256,
        "translator_model": candidate.translator_model,
        "analyst_model": candidate.analyst_model,
        "editor_model": candidate.editor_model,
        "fast_model": candidate.fast_model,
    }
    if any(row.get(key) != value for key, value in values.items()):
        raise error_type("candidate artifact identity mismatch")
    for field, digest_field in (
        ("segments_path", "segments_sha256"),
        ("telemetry_path", "telemetry_sha256"),
    ):
        path = out / row[field]
        if not path.is_file() or sha256_bytes(path.read_bytes()) != row[digest_field]:
            raise error_type(f"candidate artifact {field} is missing or changed")
    rows = read_jsonl(out / row["segments_path"], error_type=error_type)
    if target_hash(rows) != row["final_targets_sha256"]:
        raise error_type("candidate final target hash mismatch")
    if minimal_row is None and row["initial_targets_sha256"] != row["final_targets_sha256"]:
        raise error_type("minimal candidate initial target hash mismatch")
    if (
        minimal_row is not None
        and row["initial_targets_sha256"] != minimal_row["final_targets_sha256"]
    ):
        raise error_type("polish candidate initial target hash mismatch")
    usage = candidate_store(out / row["state_path"], error_type=error_type).load_usage() or {}
    if usage != row["usage"]:
        raise error_type("candidate usage metadata mismatch")
    from trans_novel.llm.usage import usage_delta

    if row["polish_incremental_usage"] != (
        usage_delta(usage, minimal_row["usage"]) if minimal_row is not None else usage_delta({}, {})
    ):
        raise error_type("candidate polish usage metadata mismatch")
    for relative, digest in row["output_hashes"].items():
        if not (out / relative).is_file() or sha256_bytes((out / relative).read_bytes()) != digest:
            raise error_type("candidate output is missing or changed")


__all__ = [
    "ArtifactError",
    "candidate_artifact_key",
    "validate_candidate_artifact",
]
