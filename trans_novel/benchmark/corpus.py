"""Deterministic, local-only benchmark corpus scan, build, and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from trans_novel.benchmark.schema import (
    BookSpec,
    ContextChallenge,
    EmittedChallengeKey,
    EmittedManifest,
    EmittedRunnerRecord,
    PassageSelection,
    SegmentCoordinate,
    Selection,
)
from trans_novel.ingest.models import Document, Segment
from trans_novel.ingest.segmenter import load_document
from trans_novel.pipeline.state import RUN_INPUT_SCHEMA_VERSION

SCHEMA_VERSION = 1
WORD_COUNTER = "en-v1"
WORD_RE = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*|\d+(?:[.,]\d+)*")
DIALOGUE_CHARS = "“”\"‘’'«»「」『』"
STRATA = (
    "narrative",
    "dialogue",
    "literary",
    "long_sentence",
    "idiom_metaphor_wordplay",
    "terminology",
    "numbers_entities",
    "special_format",
)
QUOTA_TARGETS = {"screen": 10_000, "continuous": 30_000, "stratified": 15_000, "context": 5_000}
RUNNER_FORBIDDEN_KEYS = frozenset(
    {
        "answer_key",
        "rationale",
        "model",
        "model_id",
        "primary_model",
        "editor_model",
        "fast_model",
        "candidate_id",
    }
)


def _validate_runner_leakage(records: Iterable[dict[str, Any]]) -> None:
    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in RUNNER_FORBIDDEN_KEYS:
                    raise CorpusError(f"runner record contains forbidden key {key!r} at {path}")
                walk(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{path}[{index}]")

    for index, record in enumerate(records):
        walk(record, f"runner[{index}]")


class CorpusError(ValueError):
    """A corpus input or emitted artifact failed the benchmark contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def source_digest(source: str) -> str:
    return sha256_bytes(source.encode("utf-8"))


def segment_id(book_sha256: str, chapter_index: int, segment_index: int, source: str) -> str:
    return f"{book_sha256}:c{chapter_index:04d}:s{segment_index:04d}:{source_digest(source)[:8]}"


def passage_id(
    book_id: str,
    chapter_index: int,
    start_segment_index: int,
    end_segment_index: int,
    sources: Iterable[str],
) -> str:
    joined = "\n".join(sources)
    return (
        f"{book_id}:c{chapter_index:04d}:s{start_segment_index:04d}-"
        f"{end_segment_index:04d}:{source_digest(joined)[:12]}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CorpusError(f"cannot read YAML {path}: {error}") from error
    if not isinstance(raw, dict):
        raise CorpusError(f"YAML root must be a mapping: {path}")
    return raw


def load_book_spec(path: str | os.PathLike[str]) -> BookSpec:
    source = Path(path).expanduser().resolve()
    try:
        return BookSpec.model_validate(_load_yaml(source))
    except Exception as error:
        if isinstance(error, CorpusError):
            raise
        raise CorpusError(f"invalid BookSpec {source}: {error}") from error


def load_selection(path: str | os.PathLike[str]) -> Selection:
    source = Path(path).expanduser().resolve()
    try:
        return Selection.model_validate(_load_yaml(source))
    except Exception as error:
        if isinstance(error, CorpusError):
            raise
        raise CorpusError(f"invalid Selection {source}: {error}") from error


def _resolve_books(spec: BookSpec, spec_path: Path) -> list[tuple[Any, Path, bytes]]:
    resolved: list[tuple[Any, Path, bytes]] = []
    seen_paths: set[Path] = set()
    seen_hashes: dict[str, str] = {}
    for book in spec.books:
        path = Path(book.path).expanduser()
        if not path.is_absolute():
            path = spec_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise CorpusError(f"book source is not a regular file: {path}")
        if path in seen_paths:
            raise CorpusError(f"physical source appears more than once: {path}")
        seen_paths.add(path)
        data = path.read_bytes()
        digest = sha256_bytes(data)
        prior = seen_hashes.get(digest)
        if prior is not None:
            raise CorpusError(f"physical source bytes reused by {prior} and {book.book_id}")
        seen_hashes[digest] = book.book_id
        resolved.append((book, path, data))
    return resolved


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _suggestion_tags(segment: Segment) -> list[str]:
    source = segment.source
    stripped = source.strip()
    tags: list[str] = []
    if any(char in source for char in DIALOGUE_CHARS) or stripped.startswith(("—", "–", "- ")):
        tags.append("dialogue")
    if re.search(r"\d+(?:[.,]\d+)*", source) or re.search(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", source
    ):
        tags.append("numbers_entities")
    if any(count_words(sentence) >= 40 for sentence in re.split(r"[.!?]+", source)):
        tags.append("long_sentence")
    meta = segment.meta if isinstance(segment.meta, dict) else {}
    if segment.kind == "heading" or segment.anchor or segment.resource_href or meta:
        tags.append("special_format")
    return tags


def _segment_record(
    book: Any, book_sha: str, chapter_index: int, segment: Segment
) -> dict[str, Any]:
    return {
        "book_id": book.book_id,
        "split": book.split,
        "source_sha256": book_sha,
        "segment_id": segment_id(book_sha, chapter_index, segment.index, segment.source),
        "chapter_index": chapter_index,
        "index": segment.index,
        "source": segment.source,
        "kind": segment.kind,
        "cont": segment.cont,
        "anchor": segment.anchor,
        "resource_href": segment.resource_href,
        "meta": _jsonable(segment.meta),
        "word_count": count_words(segment.source),
        "char_count": len(segment.source),
        "suggestion_tags": _suggestion_tags(segment),
    }


def _scan_books(
    spec: BookSpec, spec_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    books: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    by_id: dict[str, Any] = {}
    for book, path, data in _resolve_books(spec, spec_path):
        try:
            doc = load_document(str(path), "en", "zh")
        except Exception as error:
            raise CorpusError(f"cannot parse {book.book_id}: {error}") from error
        digest = sha256_bytes(data)
        by_id[book.book_id] = {"book": book, "path": path, "data": data, "doc": doc, "sha": digest}
        nonempty = [
            (chapter, segment)
            for chapter in doc.chapters
            for segment in chapter.segments
            if segment.source.strip()
        ]
        book_row = {
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
        books.append(book_row)
        for chapter in doc.chapters:
            for segment in chapter.segments:
                if segment.source.strip():
                    segments.append(_segment_record(book, digest, chapter.index, segment))
    return books, segments, by_id


def scan(spec_path: str | os.PathLike[str], out_dir: str | os.PathLike[str]) -> Path:
    """Scan every source once and write the human-selection inventory."""
    spec_file = Path(spec_path).expanduser().resolve()
    spec = load_book_spec(spec_file)
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise CorpusError(f"output directory already exists: {out}")
    books, segments, _ = _scan_books(spec, spec_file)
    out.mkdir(parents=True)
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
        "source_language": "en",
        "target_language": "zh",
        "books": books,
    }
    (out / "inventory.json").write_text(canonical_json(inventory) + "\n", encoding="utf-8")
    with (out / "segments.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in segments:
            handle.write(canonical_json(row) + "\n")
    return out


def _coordinates(doc: Document) -> dict[tuple[int, int], Segment]:
    result: dict[tuple[int, int], Segment] = {}
    for chapter in doc.chapters:
        for segment in chapter.segments:
            if segment.source.strip():
                result[(chapter.index, segment.index)] = segment
    return result


def _coord_key(coordinate: SegmentCoordinate) -> tuple[int, int]:
    return coordinate.chapter_index, coordinate.segment_index


def _range_segments(doc: Document, selection: PassageSelection) -> list[Segment]:
    coords = _coordinates(doc)
    result: list[Segment] = []
    for index in range(selection.start_segment_index, selection.end_segment_index + 1):
        segment = coords.get((selection.chapter_index, index))
        if segment is None:
            raise CorpusError(
                f"missing segment coordinate {selection.book_id}:"
                f"c{selection.chapter_index}:s{index}"
            )
        result.append(segment)
    return result


def _validate_quota(name: str, actual: int, target: int, tolerance: float) -> None:
    if abs(actual - target) / target > tolerance:
        raise CorpusError(f"quota {name}: actual={actual} target={target} tolerance={tolerance}")


def _validate_selection(
    spec: BookSpec,
    selection: Selection,
    by_id: dict[str, Any],
    *,
    enforce_quotas: bool = True,
) -> list[dict[str, Any]]:
    if {book.book_id for book in spec.books} != set(by_id):
        raise CorpusError("source scan does not match BookSpec")
    target_coords: set[tuple[str, int, int]] = set()
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    context_refs: list[tuple[str, int, int]] = []
    for item in selection.passages:
        source = by_id.get(item.book_id)
        if source is None:
            raise CorpusError(f"unknown book_id: {item.book_id}")
        book = source["book"]
        allowed = {
            "screen": {"screen"},
            "continuous": {"formal"},
            "stratified": {"formal"},
            "context": {"formal"},
            "hidden": {"hidden"},
        }[item.subset]
        if book.split not in allowed:
            raise CorpusError(f"subset={item.subset} cannot use split={book.split}: {item.book_id}")
        segments = _range_segments(source["doc"], item)
        coords = _coordinates(source["doc"])
        selected_coords = [
            (item.book_id, item.chapter_index, segment.index) for segment in segments
        ]
        if any(coord in target_coords for coord in selected_coords):
            raise CorpusError(f"selected target segment overlap: {item.book_id}")
        target_coords.update(selected_coords)
        joined = "\n".join(segment.source for segment in segments)
        pid = passage_id(
            item.book_id,
            item.chapter_index,
            item.start_segment_index,
            item.end_segment_index,
            [segment.source for segment in segments],
        )
        if pid in used_ids:
            raise CorpusError(f"duplicate passage_id: {pid}")
        used_ids.add(pid)
        context = item.context
        if context is not None:
            before = [_coord_key(coord) for coord in context.source_before]
            after = [_coord_key(coord) for coord in context.source_after]
            if len(set(before)) != len(before) or len(set(after)) != len(after):
                raise CorpusError(f"duplicate context coordinate: {pid}")
            if any(coord not in coords for coord in before + after):
                raise CorpusError(f"context reference does not exist: {pid}")
            target_start = (item.chapter_index, item.start_segment_index)
            target_end = (item.chapter_index, item.end_segment_index)
            if any(coord >= target_start for coord in before):
                raise CorpusError(f"source_before must precede target: {pid}")
            if any(coord <= target_end for coord in after):
                raise CorpusError(f"source_after must follow target: {pid}")
            if before != sorted(before) or after != sorted(after):
                raise CorpusError(f"context references must be source-ordered: {pid}")
            frozen = [_coord_key(coord) for coord in context.frozen_target_before]
            if frozen != before:
                raise CorpusError(f"frozen_target_before coordinates mismatch: {pid}")
            context_refs.extend((item.book_id, *coord) for coord in before + after)
        records.append(
            {
                "selection": item,
                "book": book,
                "source": source,
                "segments": segments,
                "passage_id": pid,
                "word_count": count_words(joined),
            }
        )
    if any(coord in target_coords for coord in context_refs):
        raise CorpusError("context reference is also a selected target segment")
    # Selection order is canonicalized to BookSpec order, then chapter/start.
    order = {book.book_id: index for index, book in enumerate(spec.books)}
    records.sort(
        key=lambda row: (
            order[row["book"].book_id],
            row["selection"].chapter_index,
            row["selection"].start_segment_index,
        )
    )
    for index, row in enumerate(records):
        if row["passage_id"] != records[index]["passage_id"]:
            raise CorpusError("unreachable passage ordering error")
    buckets: dict[str, int] = defaultdict(int)
    formal_books: dict[str, set[str]] = defaultdict(set)
    continuous_ranges: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    stratified_books: dict[str, set[str]] = defaultdict(set)
    for row in records:
        item = row["selection"]
        buckets[item.subset] += row["word_count"]
        if item.subset in {"continuous", "stratified", "context"}:
            formal_books[item.subset].add(item.book_id)
        if item.subset == "stratified":
            for stratum in item.strata:
                stratified_books[stratum].add(item.book_id)
        if item.subset in {"stratified", "context"} and not 150 <= row["word_count"] <= 350:
            raise CorpusError(
                f"{item.subset} passage word count must be 150..350: {row['passage_id']}={row['word_count']}"
            )
        if item.subset == "continuous":
            continuous_ranges[item.book_id].append(
                (item.chapter_index, item.start_segment_index, item.end_segment_index)
            )
    for book_id, ranges in continuous_ranges.items():
        ranges.sort()
        for prior, current in pairwise(ranges):
            if current[0] == prior[0] and current[1] <= prior[2]:
                raise CorpusError(f"continuous ranges overlap: {book_id}")
            if current[0] == prior[0] and current[1] != prior[2] + 1:
                raise CorpusError(f"continuous ranges have a gap within chapter: {book_id}")
    if enforce_quotas:
        for name, target in QUOTA_TARGETS.items():
            _validate_quota(name, buckets[name], target, selection.quota_tolerance)
        formal_total = buckets["continuous"] + buckets["stratified"] + buckets["context"]
        _validate_quota("formal", formal_total, 50_000, selection.quota_tolerance)
        if (
            len(
                set().union(
                    *(formal_books[name] for name in ("continuous", "stratified", "context"))
                )
            )
            < 6
        ):
            raise CorpusError("at least six formal books must contribute target passages")
        if len(formal_books["continuous"]) < 3:
            raise CorpusError("at least three continuous books must contribute passages")
        for stratum in STRATA:
            if len(stratified_books[stratum]) < 3:
                raise CorpusError(f"stratum {stratum} must occur in at least three formal books")
    return records


def _runner_record(row: dict[str, Any]) -> dict[str, Any]:
    item: PassageSelection = row["selection"]
    source = row["source"]
    book_sha = source["sha"]
    segments = [
        {
            "segment_id": segment_id(book_sha, item.chapter_index, segment.index, segment.source),
            "index": segment.index,
            "source": segment.source,
            "kind": segment.kind,
            "cont": segment.cont,
            "anchor": segment.anchor,
            "resource_href": segment.resource_href,
            "meta": _jsonable(segment.meta),
        }
        for segment in row["segments"]
    ]
    context: dict[str, Any] | None = None
    if item.context is not None:
        challenge: ContextChallenge = item.context
        coords = _coordinates(source["doc"])
        context = {
            "challenge_type": challenge.challenge_type,
            "source_before": [
                {
                    "segment_id": segment_id(book_sha, *coord, coords[coord].source),
                    "source": coords[coord].source,
                }
                for coord in (_coord_key(value) for value in challenge.source_before)
            ],
            "source_after": [
                {
                    "segment_id": segment_id(book_sha, *coord, coords[coord].source),
                    "source": coords[coord].source,
                }
                for coord in (_coord_key(value) for value in challenge.source_after)
            ],
            "frozen_target_before": [
                {
                    "segment_id": segment_id(
                        book_sha,
                        value.chapter_index,
                        value.segment_index,
                        coords[(value.chapter_index, value.segment_index)].source,
                    ),
                    "target": value.target,
                }
                for value in challenge.frozen_target_before
            ],
        }
    return {
        "passage_id": row["passage_id"],
        "subset": item.subset,
        "book_id": item.book_id,
        "chapter_index": item.chapter_index,
        "start": item.start_segment_index,
        "end": item.end_segment_index,
        "word_count": row["word_count"],
        "strata": item.strata,
        "segments": segments,
        "context": context,
    }


def build(
    spec_path: str | os.PathLike[str],
    selection_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
) -> Path:
    """Reparse sources and freeze passage-level runner/evaluator artifacts."""
    spec_file = Path(spec_path).expanduser().resolve()
    selection_file = Path(selection_path).expanduser().resolve()
    spec = load_book_spec(spec_file)
    selection = load_selection(selection_file)
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise CorpusError(f"output directory already exists: {out}")
    books, _, by_id = _scan_books(spec, spec_file)
    rows = _validate_selection(spec, selection, by_id)
    runner = [_runner_record(row) for row in rows]
    _validate_runner_leakage(runner)
    challenge_keys = []
    for row, _ in zip(rows, runner, strict=True):
        if row["selection"].context is not None:
            challenge = row["selection"].context
            challenge_keys.append(
                {
                    "passage_id": row["passage_id"],
                    "challenge_type": challenge.challenge_type,
                    "answer_key": challenge.answer_key,
                    "rationale": challenge.rationale,
                }
            )
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
    corpus_without_hash = dict(corpus)
    semantics = {
        "corpus": corpus_without_hash,
        "runner_segments": runner,
        "challenge_keys": challenge_keys,
    }
    corpus["corpus_sha256"] = sha256_bytes(canonical_json(semantics).encode("utf-8"))
    out.mkdir(parents=True)
    (out / "corpus.json").write_text(canonical_json(corpus) + "\n", encoding="utf-8")
    (out / "source_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with (out / "runner_segments.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in runner:
            handle.write(canonical_json(row) + "\n")
    with (out / "challenge_keys.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in challenge_keys:
            handle.write(canonical_json(row) + "\n")
    return out


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"invalid JSON {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CorpusError(f"cannot read JSONL {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise CorpusError(f"blank JSONL line {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise CorpusError(f"JSONL row is not an object {path}:{line_number}")
        rows.append(row)
    return rows


def _parse_artifact(model: Any, value: Any, label: str) -> Any:
    try:
        return model.model_validate(value)
    except Exception as error:
        raise CorpusError(f"invalid {label}: {error}") from error


def _manifest_books(value: Any) -> dict[str, dict[str, Any]]:
    manifest = _parse_artifact(EmittedManifest, value, "source manifest")
    result: dict[str, dict[str, Any]] = {}
    for row_model in manifest.books:
        row = row_model.model_dump(mode="python")
        basename = row["basename"]
        if row["parser_schema"] != RUN_INPUT_SCHEMA_VERSION:
            raise CorpusError(f"manifest parser schema mismatch: {row['book_id']}")
        if (
            Path(basename).is_absolute()
            or Path(basename).name != basename
            or "/" in basename
            or "\\" in basename
            or basename in {".", ".."}
        ):
            raise CorpusError(f"manifest basename must be a basename only: {basename!r}")
        if row["book_id"] in result:
            raise CorpusError(f"duplicate manifest book: {row['book_id']}")
        result[row["book_id"]] = row
    return result


def validate_corpus(corpus_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate frozen artifacts without opening any original source book."""
    root = Path(corpus_dir).expanduser().resolve()
    corpus = _read_json(root / "corpus.json")
    manifest_raw = _read_json(root / "source_manifest.json")
    runner_input = _read_jsonl(root / "runner_segments.jsonl")
    challenge_input = _read_jsonl(root / "challenge_keys.jsonl")
    _validate_runner_leakage(runner_input)
    manifest = _parse_artifact(EmittedManifest, manifest_raw, "source manifest")
    manifest_value = manifest.model_dump(mode="python")
    manifest_books = _manifest_books(manifest_value)
    runner_models = [
        _parse_artifact(EmittedRunnerRecord, row, f"runner record {index}")
        for index, row in enumerate(runner_input)
    ]
    runner = [model.model_dump(mode="python") for model in runner_models]
    challenge_models = [
        _parse_artifact(EmittedChallengeKey, row, f"challenge key {index}")
        for index, row in enumerate(challenge_input)
    ]
    challenge_keys = [model.model_dump(mode="python") for model in challenge_models]
    if corpus.get("schema_version") != SCHEMA_VERSION or corpus.get("word_counter") != WORD_COUNTER:
        raise CorpusError("unsupported corpus schema or word counter")
    if corpus.get("run_input_schema_version") != RUN_INPUT_SCHEMA_VERSION:
        raise CorpusError("run input schema version mismatch")
    if manifest_value["run_input_schema_version"] != RUN_INPUT_SCHEMA_VERSION:
        raise CorpusError("source manifest run input schema version mismatch")
    corpus_passages = corpus.get("passages")
    if not isinstance(corpus_passages, list) or len(corpus_passages) != len(runner):
        raise CorpusError("corpus passage summaries do not match runner")
    challenge_by_id = {row["passage_id"]: row for row in challenge_keys}
    if len(challenge_by_id) != len(challenge_keys):
        raise CorpusError("duplicate challenge key passage_id")
    target_coords: set[tuple[str, int, int]] = set()
    references: set[tuple[str, int, int]] = set()
    recomputed_passages: list[dict[str, Any]] = []
    for row, summary in zip(runner, corpus_passages, strict=True):
        book_id = row.get("book_id")
        book = manifest_books.get(book_id)
        if book is None:
            raise CorpusError(f"runner references unknown book: {book_id}")
        book_sha = book.get("source_sha256")
        segments = row.get("segments")
        if not isinstance(segments, list) or not segments:
            raise CorpusError(f"runner passage has no segments: {row.get('passage_id')}")
        indexes = [segment["index"] for segment in segments]
        if indexes != list(range(row["start"], row["end"] + 1)):
            raise CorpusError(f"runner segment range is not contiguous: {row.get('passage_id')}")
        for segment in segments:
            expected = segment_id(
                book_sha, row["chapter_index"], segment["index"], segment["source"]
            )
            if segment["segment_id"] != expected:
                raise CorpusError(f"segment ID mismatch: {segment.get('segment_id')}")
            coord = (book_id, row["chapter_index"], segment["index"])
            if coord in target_coords:
                raise CorpusError(f"selected segment overlap: {coord}")
            target_coords.add(coord)
        joined = "\n".join(segment["source"] for segment in segments)
        expected_passage = passage_id(
            book_id, row["chapter_index"], row["start"], row["end"], [s["source"] for s in segments]
        )
        if row.get("passage_id") != expected_passage:
            raise CorpusError(f"passage ID mismatch: {row.get('passage_id')}")
        if row.get("word_count") != count_words(joined):
            raise CorpusError(f"passage word count mismatch: {row.get('passage_id')}")
        expected_summary = {
            "passage_id": row.get("passage_id"),
            "subset": row.get("subset"),
            "book_id": row.get("book_id"),
            "chapter_index": row.get("chapter_index"),
            "start": row.get("start"),
            "end": row.get("end"),
            "word_count": row.get("word_count"),
            "strata": row.get("strata"),
        }
        if summary != expected_summary:
            raise CorpusError(f"runner/corpus summary mismatch: {row.get('passage_id')}")
        if row["start"] != segments[0]["index"] or row["end"] != segments[-1]["index"]:
            raise CorpusError(f"runner segment range mismatch: {row.get('passage_id')}")
        if row["chapter_index"] < 0 or row["start"] < 0 or row["end"] < row["start"]:
            raise CorpusError(f"invalid runner range: {row.get('passage_id')}")
        context = row.get("context")
        if row.get("subset") == "context":
            if not isinstance(context, dict) or set(context) != {
                "challenge_type",
                "source_before",
                "source_after",
                "frozen_target_before",
            }:
                raise CorpusError(f"malformed context runner record: {row.get('passage_id')}")
            key = challenge_by_id.get(row["passage_id"])
            if key is None or set(key) != {
                "passage_id",
                "challenge_type",
                "answer_key",
                "rationale",
            }:
                raise CorpusError(f"missing or malformed challenge key: {row.get('passage_id')}")
            if key["challenge_type"] != context.get("challenge_type"):
                raise CorpusError(f"context challenge type mismatch: {row.get('passage_id')}")
            before_ids: list[str] = []
            for field in ("source_before", "source_after"):
                refs = context.get(field)
                if not isinstance(refs, list):
                    raise CorpusError(f"malformed context references: {row.get('passage_id')}")
                prior: tuple[int, int] | None = None
                for ref in refs:
                    if set(ref) != {"segment_id", "source"}:
                        raise CorpusError(
                            f"malformed context source reference: {row.get('passage_id')}"
                        )
                    coord = _segment_coord_from_id(ref["segment_id"], book_sha)
                    if prior is not None and coord <= prior:
                        raise CorpusError(
                            f"context references are not source-ordered: {row.get('passage_id')}"
                        )
                    prior = coord
                    expected_ref_id = segment_id(book_sha, coord[0], coord[1], ref["source"])
                    if ref["segment_id"] != expected_ref_id:
                        raise CorpusError(f"context segment ID mismatch: {row.get('passage_id')}")
                    target_start = (row["chapter_index"], row["start"])
                    target_end = (row["chapter_index"], row["end"])
                    if field == "source_before" and coord >= target_start:
                        raise CorpusError(
                            f"context source_before does not precede target: {row.get('passage_id')}"
                        )
                    if field == "source_after" and coord <= target_end:
                        raise CorpusError(
                            f"context source_after does not follow target: {row.get('passage_id')}"
                        )
                    references.add((book_id, coord[0], coord[1]))
                    if field == "source_before":
                        before_ids.append(ref["segment_id"])
            frozen = context["frozen_target_before"]
            if not isinstance(frozen, list) or len(frozen) != len(before_ids):
                raise CorpusError(
                    f"frozen targets do not match source_before: {row.get('passage_id')}"
                )
            for frozen_row, before_id in zip(frozen, before_ids, strict=True):
                if (
                    set(frozen_row) != {"segment_id", "target"}
                    or frozen_row["segment_id"] != before_id
                ):
                    raise CorpusError(f"frozen target reference mismatch: {row.get('passage_id')}")
        elif context is not None:
            raise CorpusError(f"non-context runner record has context: {row.get('passage_id')}")
        else:
            if row["passage_id"] in challenge_by_id:
                raise CorpusError(f"challenge key has non-context passage: {row.get('passage_id')}")
        recomputed_passages.append(row)
    if target_coords & references:
        raise CorpusError("context reference is a selected target")
    if set(challenge_by_id) != {
        row["passage_id"] for row in runner if row.get("subset") == "context"
    }:
        raise CorpusError("challenge key set does not match context runner set")
    # The persisted corpus is authoritative for quotas, but recompute it from runner text.
    actual = defaultdict(int, dict.fromkeys((*QUOTA_TARGETS, "hidden", "formal"), 0))
    formal_books: set[str] = set()
    continuous_books: set[str] = set()
    stratified_books: dict[str, set[str]] = defaultdict(set)
    continuous_ranges: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in runner:
        subset = row.get("subset")
        if subset not in {"screen", "continuous", "stratified", "context", "hidden"}:
            raise CorpusError(f"unknown runner subset: {subset}")
        split = manifest_books[row["book_id"]].get("split")
        expected_split = {"screen": "screen", "hidden": "hidden"}.get(subset, "formal")
        if split != expected_split:
            raise CorpusError(f"runner subset/split mismatch: {row['passage_id']}")
        actual[subset] += row["word_count"]
        if subset in {"continuous", "stratified", "context"}:
            formal_books.add(row["book_id"])
        if subset == "continuous":
            continuous_books.add(row["book_id"])
            continuous_ranges[row["book_id"]].append(
                (row["chapter_index"], row["start"], row["end"])
            )
        if subset in {"stratified"}:
            for stratum in row.get("strata", []):
                stratified_books[stratum].add(row["book_id"])
        if subset in {"stratified", "context"} and not 150 <= row["word_count"] <= 350:
            raise CorpusError(f"{subset} passage word count must be 150..350: {row['passage_id']}")
    for book_id, ranges in continuous_ranges.items():
        ranges.sort()
        for prior, current in pairwise(ranges):
            if current[0] == prior[0] and current[1] <= prior[2]:
                raise CorpusError(f"continuous ranges overlap: {book_id}")
            if current[0] == prior[0] and current[1] != prior[2] + 1:
                raise CorpusError(f"continuous ranges have a gap within chapter: {book_id}")
    actual["formal"] = actual["continuous"] + actual["stratified"] + actual["context"]
    quotas = corpus.get("quotas", {})
    if quotas.get("actual") != dict(actual):
        raise CorpusError("quota totals do not match runner")
    tolerance = quotas.get("tolerance")
    if not isinstance(tolerance, int | float) or not 0 <= tolerance <= 0.20:
        raise CorpusError("invalid quota tolerance")
    for name, target in QUOTA_TARGETS.items():
        _validate_quota(name, actual[name], target, tolerance)
    _validate_quota("formal", actual["formal"], 50_000, tolerance)
    if len(formal_books) < 6:
        raise CorpusError(f"formal book coverage is {len(formal_books)}, requires 6")
    if len(continuous_books) < 3:
        raise CorpusError(f"continuous book coverage is {len(continuous_books)}, requires 3")
    for stratum in STRATA:
        if len(stratified_books[stratum]) < 3:
            raise CorpusError(
                f"stratum {stratum} coverage is {len(stratified_books[stratum])}, requires 3"
            )
    semantics = {
        "corpus": {key: value for key, value in corpus.items() if key != "corpus_sha256"},
        "runner_segments": runner,
        "challenge_keys": challenge_keys,
    }
    digest = sha256_bytes(canonical_json(semantics).encode("utf-8"))
    if corpus.get("corpus_sha256") != digest:
        raise CorpusError("corpus_sha256 mismatch")
    split_counts = {
        split: sum(1 for row in runner if manifest_books[row["book_id"]]["split"] == split)
        for split in ("screen", "formal", "hidden")
    }
    bucket_counts = {
        subset: sum(1 for row in runner if row["subset"] == subset)
        for subset in ("screen", "continuous", "stratified", "context", "hidden")
    }
    book_counts = {
        book_id: sum(1 for row in runner if row["book_id"] == book_id)
        for book_id in sorted(manifest_books)
    }
    return {
        "corpus_sha256": digest,
        "runner_count": len(runner),
        "challenge_count": len(challenge_keys),
        "book_count": len(manifest_books),
        "word_counts": dict(actual),
        "split_counts": split_counts,
        "bucket_counts": bucket_counts,
        "book_counts": book_counts,
    }


def _segment_coord_from_id(value: str, book_sha: str) -> tuple[int, int]:
    suffix = (
        value[len(book_sha) + 1 :]
        if isinstance(value, str) and value.startswith(book_sha + ":")
        else ""
    )
    match = re.fullmatch(r"c(\d{4}):s(\d{4}):([0-9a-f]{8})", suffix)
    if match is None:
        raise CorpusError(f"invalid segment reference: {value}")
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "QUOTA_TARGETS",
    "SCHEMA_VERSION",
    "STRATA",
    "WORD_COUNTER",
    "CorpusError",
    "build",
    "canonical_json",
    "count_words",
    "load_book_spec",
    "load_selection",
    "passage_id",
    "scan",
    "segment_id",
    "source_digest",
    "validate_corpus",
]
