"""Hidden-book integration preflight and production configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import read_json
from trans_novel.benchmark.corpus import load_book_spec, validate_corpus
from trans_novel.benchmark.integration.artifacts import IntegrationError, integration_sha256
from trans_novel.benchmark.run import (
    load_candidate_spec,
    validate_candidate_capabilities,
)
from trans_novel.benchmark.schema import CandidateSpec
from trans_novel.config import Config, LLMConfig, ModelRoles, PipelineConfig


def preflight(
    corpus_dir: Path,
    book_spec_path: Path,
    candidate_spec_path: Path,
    integration_spec_path: Path,
    spec_type: Any,
) -> tuple[Any, CandidateSpec, Path, str, dict[str, Any]]:
    corpus_dir = corpus_dir.resolve()
    book_spec_path = book_spec_path.resolve()
    candidate_spec_path = candidate_spec_path.resolve()
    integration_spec_path = integration_spec_path.resolve()
    corpus_value = validate_corpus(corpus_dir)
    corpus_hash = corpus_value.get("corpus_sha256")
    raw_candidate_hash = integration_sha256(candidate_spec_path)
    try:
        import yaml

        raw_integration = yaml.safe_load(integration_spec_path.read_text(encoding="utf-8"))
        spec = spec_type.model_validate(raw_integration)
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
        book for book in book_spec.books if book.book_id == spec.book_id and book.split == "hidden"
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
    if source.suffix.lower() != ".epub" or not source.is_file() or not os.access(source, os.R_OK):
        raise IntegrationError("selected hidden source must be a readable EPUB")
    manifest = read_json(corpus_dir / "source_manifest.json", error_type=IntegrationError)
    rows = {row.get("book_id"): row for row in manifest.get("books", [])}
    source_hash = integration_sha256(source)
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
            "book_spec_sha256": integration_sha256(book_spec_path),
            "candidate_spec_sha256": raw_candidate_hash,
            "integration_spec_sha256": integration_sha256(integration_spec_path),
            "selected": chosen,
            "source_sha256": source_hash,
        },
    )


def quality_config(spec: CandidateSpec, candidate: Any, state_dir: Path) -> Config:
    preset = "quality" if candidate.pipeline_variant == "polish" else "balanced"
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
        state_dir=str(state_dir),
    )


__all__ = ["preflight", "quality_config"]
