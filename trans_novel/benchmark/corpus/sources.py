"""Book specification loading and deterministic source scanning."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from trans_novel.benchmark.corpus.identity import count_words, segment_id, sha256_bytes
from trans_novel.benchmark.corpus.selection import suggestion_tags
from trans_novel.benchmark.schema import BookSpec, Selection
from trans_novel.ingest import load_document
from trans_novel.pipeline.state import RUN_INPUT_SCHEMA_VERSION


class CorpusSourceError(ValueError):
    """A source specification or book cannot be loaded."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CorpusSourceError(f"cannot read YAML {path}: {error}") from error
    if not isinstance(raw, dict):
        raise CorpusSourceError(f"YAML root must be a mapping: {path}")
    return raw


def load_book_spec(path: str | os.PathLike[str]) -> BookSpec:
    source = Path(path).expanduser().resolve()
    try:
        return BookSpec.model_validate(_load_yaml(source))
    except Exception as error:
        if isinstance(error, CorpusSourceError):
            raise
        raise CorpusSourceError(f"invalid BookSpec {source}: {error}") from error


def load_selection(path: str | os.PathLike[str]) -> Selection:
    source = Path(path).expanduser().resolve()
    try:
        return Selection.model_validate(_load_yaml(source))
    except Exception as error:
        if isinstance(error, CorpusSourceError):
            raise
        raise CorpusSourceError(f"invalid Selection {source}: {error}") from error


def resolve_books(
    spec: BookSpec,
    spec_path: Path,
    *,
    error_type: type[Exception] = CorpusSourceError,
) -> list[tuple[Any, Path, bytes]]:
    resolved: list[tuple[Any, Path, bytes]] = []
    seen_paths: set[Path] = set()
    seen_hashes: dict[str, str] = {}
    for book in spec.books:
        path = Path(book.path).expanduser()
        if not path.is_absolute():
            path = spec_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise error_type(f"book source is not a regular file: {path}")
        if path in seen_paths:
            raise error_type(f"physical source appears more than once: {path}")
        seen_paths.add(path)
        data = path.read_bytes()
        digest = sha256_bytes(data)
        prior = seen_hashes.get(digest)
        if prior is not None:
            raise error_type(f"physical source bytes reused by {prior} and {book.book_id}")
        seen_hashes[digest] = book.book_id
        resolved.append((book, path, data))
    return resolved


def _jsonable(value: Any) -> Any:
    try:
        import json

        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def scan_books(
    spec: BookSpec,
    spec_path: Path,
    *,
    parser: Callable[..., Any] = load_document,
    error_type: type[Exception] = CorpusSourceError,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    books: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    by_id: dict[str, Any] = {}
    for book, path, data in resolve_books(spec, spec_path, error_type=error_type):
        try:
            doc = parser(str(path), "en", "zh")
        except Exception as error:
            raise error_type(f"cannot parse {book.book_id}: {error}") from error
        digest = sha256_bytes(data)
        by_id[book.book_id] = {"book": book, "path": path, "data": data, "doc": doc, "sha": digest}
        nonempty = [
            (chapter, segment)
            for chapter in doc.chapters
            for segment in chapter.segments
            if segment.source.strip()
        ]
        books.append(
            {
                "book_id": book.book_id,
                "split": book.split,
                "source_sha256": digest,
                "basename": path.name,
                "format": doc.fmt,
                "title": doc.title,
                "chapter_count": len(doc.chapters),
                "segment_count": len(nonempty),
                "word_count": sum(count_words(segment.source) for _, segment in nonempty),
                "parser_schema": RUN_INPUT_SCHEMA_VERSION,
                "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
            }
        )
        for chapter in doc.chapters:
            for segment in chapter.segments:
                if segment.source.strip():
                    segments.append(
                        {
                            "book_id": book.book_id,
                            "split": book.split,
                            "source_sha256": digest,
                            "segment_id": segment_id(
                                digest, chapter.index, segment.index, segment.source
                            ),
                            "chapter_index": chapter.index,
                            "index": segment.index,
                            "source": segment.source,
                            "kind": segment.kind,
                            "suggestion_tags": suggestion_tags(segment),
                            "cont": segment.cont,
                            "anchor": segment.anchor,
                            "resource_href": segment.resource_href,
                            "meta": _jsonable(segment.meta),
                            "word_count": count_words(segment.source),
                            "char_count": len(segment.source),
                        }
                    )
    return books, segments, by_id


__all__ = ["CorpusSourceError", "load_book_spec", "load_selection", "resolve_books", "scan_books"]
