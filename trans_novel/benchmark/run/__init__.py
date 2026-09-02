from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trans_novel.benchmark import contracts
from trans_novel.benchmark.artifacts import atomic_json, read_json
from trans_novel.benchmark.corpus import (
    canonical_json,
    load_book_spec,
    resolve_books,
    sha256_bytes,
)
from trans_novel.benchmark.run.artifacts import candidate_artifact_key, validate_candidate_artifact
from trans_novel.benchmark.run.candidate import (
    attach_telemetry_sink,
    candidate_models,
    load_candidate_spec,
    model_client,
    validate_candidate_capabilities,
)
from trans_novel.benchmark.run.evidence import candidate_store, target_hash
from trans_novel.benchmark.run.execution import execute, validate_completed
from trans_novel.benchmark.run.telemetry import (
    CollectingCallTelemetrySink,
    JsonlCallTelemetrySink,
)
from trans_novel.benchmark.schema import Candidate, CandidateSpec
from trans_novel.config import Config, LLMConfig, ModelRoles, PipelineConfig
from trans_novel.ingest.epub.reader import read_epub
from trans_novel.llm import GenerationOptions

PROMPT_VERSION = "production-translator-v1"


class BenchmarkError(ValueError):
    """Invalid benchmark input or an integrity failure in a run directory."""


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _safe_book_id(value: str) -> str:
    safe = "".join(char if (char.isalnum() or char in "-_") else "_" for char in value).strip("_")
    return f"{safe or 'book'}-{_safe_id(value)[:12]}"


def _run_full(
    runner: FullRunner,
    book_spec_path: str | os.PathLike[str],
    candidates_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    book_spec = load_book_spec(book_spec_path)
    resolved = resolve_books(book_spec, Path(book_spec_path).expanduser().resolve())
    formal = [(book, path, data) for book, path, data in resolved if book.split == "formal"]
    if len(formal) != 6:
        raise BenchmarkError("full benchmark requires exactly six formal chapter EPUBs")
    for book, path, _data in formal:
        document = read_epub(
            str(path), source_lang=book_spec.source_language, target_lang=book_spec.target_language
        )
        if len(document.chapters) != 1:
            raise BenchmarkError(
                f"{book.book_id} must contain exactly one chapter; found {len(document.chapters)}"
            )
    return runner._run(
        book_spec_path,
        candidates_path,
        out_dir,
        mode="full",
        selected_books=formal,
        replicate_count=load_candidate_spec(candidates_path).replicates,
    )


def _run_canary(
    runner: FullRunner,
    book_spec_path: str | os.PathLike[str],
    candidates_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    book_id: str | None = None,
) -> dict[str, Any]:
    book_spec = load_book_spec(book_spec_path)
    resolved = resolve_books(book_spec, Path(book_spec_path).expanduser().resolve())
    screen = [(book, path, data) for book, path, data in resolved if book.split == "screen"]
    if book_id is not None:
        screen = [row for row in screen if row[0].book_id == book_id]
    if not screen:
        raise BenchmarkError("canary requires a matching screen chapter EPUB")
    return runner._run(
        book_spec_path,
        candidates_path,
        out_dir,
        mode="canary",
        selected_books=[sorted(screen, key=lambda row: row[0].book_id)[0]],
        replicate_count=1,
    )


class FullRunner:
    """Run original formal chapter EPUBs through the production quality pipeline."""

    def __init__(
        self, *, client_factory: Callable[..., Any] | None = None, client: Any | None = None
    ) -> None:
        self.client_factory = client_factory
        self.client = client

    @staticmethod
    def _config(
        candidate: Candidate,
        *,
        pipeline_variant: str = "minimal",
        state_dir: str,
    ) -> Config:
        preset = "quality" if pipeline_variant == "polish" else "balanced"
        return Config(
            llm=LLMConfig(
                models=ModelRoles(
                    translator=[candidate.translator_model],
                    analyst=[candidate.analyst_model],
                    editor=[candidate.editor_model],
                    fast=[candidate.fast_model],
                )
            ),
            quality=preset,
            source_lang="en",
            target_lang="zh",
            pipeline=PipelineConfig.for_quality(preset),
            state_dir=state_dir,
        )

    def _client(
        self,
        spec: CandidateSpec,
        candidate: Candidate,
        options: GenerationOptions,
        sink: JsonlCallTelemetrySink,
    ) -> Any:
        if self.client is not None:
            attach_telemetry_sink(self.client, sink, required=True)
            return self.client
        roles = ModelRoles(
            translator=[candidate.translator_model],
            analyst=[candidate.analyst_model],
            editor=[candidate.editor_model],
            fast=[candidate.fast_model],
        )
        return model_client(
            spec,
            candidate.translator_model,
            "translator",
            options,
            self.client_factory,
            sink,
            roles=roles,
        )

    def _artifact_key(
        self,
        spec: CandidateSpec,
        candidate: Candidate,
        book_id: str,
        source_sha256: str,
        replicate: int,
    ) -> str:
        return candidate_artifact_key(
            spec,
            candidate,
            book_id,
            source_sha256,
            replicate,
            generation=contracts.GENERATION_FIELDS,
            schema_version=contracts.RUN_SCHEMA_VERSION,
        )

    def _validate_artifact(
        self,
        out: Path,
        row: dict[str, Any],
        *,
        spec: CandidateSpec,
        candidate: Candidate,
        book_id: str,
        source_sha256: str,
        replicate: int,
        minimal_row: dict[str, Any] | None = None,
    ) -> None:
        validate_candidate_artifact(
            out,
            row,
            spec=spec,
            candidate=candidate,
            book_id=book_id,
            source_sha256=source_sha256,
            replicate=replicate,
            minimal_row=minimal_row,
            generation=contracts.GENERATION_FIELDS,
            schema_version=contracts.RUN_SCHEMA_VERSION,
            target_hash=target_hash,
            candidate_store=candidate_store,
            error_type=BenchmarkError,
        )

    def _run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        mode: str,
        selected_books: list[tuple[Any, Path, bytes]],
        replicate_count: int,
    ) -> dict[str, Any]:
        source_spec_path = Path(book_spec_path).expanduser().resolve()
        spec = load_candidate_spec(candidates_path)
        options = validate_candidate_capabilities(spec)
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        immutable = {
            "schema_version": contracts.RUN_SCHEMA_VERSION,
            "run_mode": mode,
            "benchmark_id": spec.benchmark_id,
            "book_spec_sha256": sha256_bytes(source_spec_path.read_bytes()),
            "candidate_spec_sha256": sha256_bytes(
                canonical_json(spec.model_dump(mode="python")).encode("utf-8")
            ),
            "generation": dict(contracts.GENERATION_FIELDS),
            "pipeline_variants": {
                candidate.candidate_id: candidate.pipeline_variant for candidate in spec.candidates
            },
            "book_ids": [book.book_id for book, _, _ in selected_books],
            "replicates": replicate_count,
        }
        run_path = out / "run.json"
        if run_path.exists() and read_json(run_path, error_type=BenchmarkError) != immutable:
            raise BenchmarkError("immutable benchmark run identity mismatch")
        if not run_path.exists():
            atomic_json(run_path, immutable)
        state_path = out / "run_state.json"
        state = (
            read_json(state_path, error_type=BenchmarkError)
            if state_path.exists()
            else {"schema_version": 1, "status": "pending", "artifacts": {}}
        )
        rows_path = out / "candidates.json"
        existing = read_json(rows_path, error_type=BenchmarkError) if rows_path.exists() else []
        if not isinstance(existing, list):
            raise BenchmarkError("candidate artifact index must be a list")
        by_key = {row["artifact_key"]: row for row in existing}
        expected = len(selected_books) * len(spec.candidates) * replicate_count
        candidate_by_id = {candidate.candidate_id: candidate for candidate in spec.candidates}
        book_by_id = {
            book.book_id: (book, sha256_bytes(data)) for book, _path, data in selected_books
        }
        ordered = sorted(
            spec.candidates,
            key=lambda candidate: (candidate.pipeline_variant != "minimal", candidate.candidate_id),
        )
        minimal_by_pair = {
            candidate_models(candidate): candidate
            for candidate in ordered
            if candidate.pipeline_variant == "minimal"
        }
        completed = validate_completed(
            self,
            out,
            immutable,
            state,
            by_key,
            expected,
            candidate_by_id,
            book_by_id,
            minimal_by_pair,
            spec,
            BenchmarkError,
        )
        if completed is not None:
            return completed
        return immutable | execute(
            self,
            out,
            state_path,
            rows_path,
            state,
            by_key,
            spec,
            options,
            selected_books,
            ordered,
            minimal_by_pair,
            expected,
            replicate_count,
            BenchmarkError,
        )

    def run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        return _run_full(self, book_spec_path, candidates_path, out_dir)


class CanaryRunner(FullRunner):
    """Run one real screen chapter EPUB through the same production quality pipeline."""

    def run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        book_id: str | None = None,
    ) -> dict[str, Any]:
        return _run_canary(self, book_spec_path, candidates_path, out_dir, book_id)


__all__ = [
    "PROMPT_VERSION",
    "BenchmarkError",
    "CanaryRunner",
    "CollectingCallTelemetrySink",
    "FullRunner",
    "JsonlCallTelemetrySink",
    "attach_telemetry_sink",
    "candidate_artifact_key",
    "candidate_models",
    "load_candidate_spec",
    "model_client",
    "validate_candidate_artifact",
    "validate_candidate_capabilities",
]
