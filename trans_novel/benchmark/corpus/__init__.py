"""Deterministic corpus scan, build, and validation orchestration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from trans_novel.benchmark.corpus.identity import (
    WORD_COUNTER,
    canonical_json,
    count_words,
    passage_id,
    segment_id,
    sha256_bytes,
    source_digest,
)
from trans_novel.benchmark.corpus.selection import (
    QUOTA_TARGETS,
    STRATA,
    runner_record,
    validate_selection,
)
from trans_novel.benchmark.corpus.sources import (
    load_book_spec,
    load_selection,
    resolve_books,
    scan_books,
)
from trans_novel.benchmark.corpus.validation import (
    validate_corpus_artifacts,
    validate_runner_leakage,
)
from trans_novel.ingest import load_document
from trans_novel.pipeline.state import RUN_INPUT_SCHEMA_VERSION

SCHEMA_VERSION = 1


class CorpusError(ValueError):
    """A corpus input or emitted artifact failed the benchmark contract."""


def scan(spec_path: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> Path:
    """Scan every source once and write the human-selection inventory."""
    from trans_novel.benchmark.artifacts import write_jsonl

    spec_file = Path(spec_path).expanduser().resolve()
    spec = _load_spec(spec_file)
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise CorpusError(f"output directory already exists: {out}")
    books, segments, _ = _scan(spec, spec_file)
    out.mkdir(parents=True)
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
        "source_language": "en",
        "target_language": "zh",
        "books": books,
    }
    (out / "inventory.json").write_text(canonical_json(inventory) + "\n", encoding="utf-8")
    write_jsonl(out / "segments.jsonl", segments)
    return out


def build(
    spec_path: str | os.PathLike[str],
    selection_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> Path:
    """Reparse sources and freeze passage-level runner/evaluator artifacts."""
    from trans_novel.benchmark.artifacts import write_jsonl

    spec_file = Path(spec_path).expanduser().resolve()
    selection_file = Path(selection_path).expanduser().resolve()
    spec = _load_spec(spec_file)
    selection = _load_selection(selection_file)
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise CorpusError(f"output directory already exists: {out}")
    books, _, by_id = _scan(spec, spec_file)
    rows = validate_selection(spec, selection, by_id, error_type=CorpusError)
    runner = [runner_record(row) for row in rows]
    validate_runner_leakage(runner, error_type=CorpusError)
    challenge_keys = [
        {
            "passage_id": row["passage_id"],
            "challenge_type": row["selection"].context.challenge_type,
            "answer_key": row["selection"].context.answer_key,
            "rationale": row["selection"].context.rationale,
        }
        for row in rows
        if row["selection"].context is not None
    ]
    quotas = {
        name: sum(row["word_count"] for row in rows if row["selection"].subset == name)
        for name in (*QUOTA_TARGETS, "hidden", "formal")
    }
    quotas["formal"] = quotas["continuous"] + quotas["stratified"] + quotas["context"]
    corpus = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_name": selection.benchmark_name,
        "word_counter": WORD_COUNTER,
        "parser_schema": RUN_INPUT_SCHEMA_VERSION,
        "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
        "books": books,
        "passages": [
            {
                "passage_id": row["passage_id"],
                "subset": row["selection"].subset,
                "book_id": row["selection"].book_id,
                "chapter_index": row["selection"].chapter_index,
                "start": row["selection"].start_segment_index,
                "end": row["selection"].end_segment_index,
                "word_count": row["word_count"],
                "strata": row["selection"].strata,
            }
            for row in rows
        ],
        "quotas": {
            "targets": QUOTA_TARGETS,
            "actual": quotas,
            "tolerance": selection.quota_tolerance,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
        "books": [
            {
                "book_id": book.book_id,
                "source_sha256": by_id[book.book_id]["sha"],
                "basename": by_id[book.book_id]["path"].name,
                "split": book.split,
                "format": by_id[book.book_id]["doc"].fmt,
                "title": by_id[book.book_id]["doc"].title,
                "chapter_count": len(by_id[book.book_id]["doc"].chapters),
                "parser_schema": RUN_INPUT_SCHEMA_VERSION,
            }
            for book in spec.books
        ],
    }
    semantics = {"corpus": corpus, "runner_segments": runner, "challenge_keys": challenge_keys}
    corpus["corpus_sha256"] = sha256_bytes(canonical_json(semantics).encode("utf-8"))
    out.mkdir(parents=True)
    (out / "corpus.json").write_text(canonical_json(corpus) + "\n", encoding="utf-8")
    (out / "source_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    write_jsonl(out / "runner_segments.jsonl", runner)
    write_jsonl(out / "challenge_keys.jsonl", challenge_keys)
    return out


def validate_corpus(corpus_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate frozen artifacts without opening any original source book."""

    try:
        return validate_corpus_artifacts(
            corpus_dir,
            schema_version=SCHEMA_VERSION,
            word_counter=WORD_COUNTER,
            quota_targets=QUOTA_TARGETS,
            strata=STRATA,
            error_type=CorpusError,
        )
    except CorpusError:
        raise
    except Exception as error:
        raise CorpusError(str(error)) from error


def _load_spec(path: Path) -> Any:
    try:
        return load_book_spec(path)
    except Exception as error:
        raise CorpusError(str(error)) from error


def _load_selection(path: Path) -> Any:
    try:
        return load_selection(path)
    except Exception as error:
        raise CorpusError(str(error)) from error


def _scan(
    spec: Any, spec_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    try:
        return scan_books(spec, spec_path, parser=load_document, error_type=CorpusError)
    except CorpusError:
        raise
    except Exception as error:
        raise CorpusError(str(error)) from error


__all__ = [
    "SCHEMA_VERSION",
    "WORD_COUNTER",
    "CorpusError",
    "build",
    "canonical_json",
    "count_words",
    "load_book_spec",
    "passage_id",
    "resolve_books",
    "scan",
    "segment_id",
    "sha256_bytes",
    "source_digest",
    "validate_corpus",
    "validate_corpus_artifacts",
    "validate_runner_leakage",
]
