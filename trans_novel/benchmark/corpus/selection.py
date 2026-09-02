"""Pure segment selection, quota, and runner-record policy."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import pairwise
from typing import Any

from trans_novel.benchmark.corpus.identity import count_words, passage_id, segment_id
from trans_novel.benchmark.schema import (
    ContextChallenge,
    PassageSelection,
    SegmentCoordinate,
    Selection,
)
from trans_novel.ingest import Document, Segment

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


def jsonable(value: Any) -> Any:
    try:
        import json

        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def suggestion_tags(segment: Segment) -> list[str]:
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


def coordinates(doc: Document) -> dict[tuple[int, int], Segment]:
    return {
        (chapter.index, segment.index): segment
        for chapter in doc.chapters
        for segment in chapter.segments
        if segment.source.strip()
    }


def range_segments(
    doc: Document, selection: PassageSelection, *, error_type: type[Exception]
) -> list[Segment]:
    coords = coordinates(doc)
    result: list[Segment] = []
    for index in range(selection.start_segment_index, selection.end_segment_index + 1):
        segment = coords.get((selection.chapter_index, index))
        if segment is None:
            raise error_type(
                f"missing segment coordinate {selection.book_id}:c{selection.chapter_index}:s{index}"
            )
        result.append(segment)
    return result


def coord_key(coordinate: SegmentCoordinate) -> tuple[int, int]:
    return coordinate.chapter_index, coordinate.segment_index


def validate_quota(
    name: str, actual: int, target: int, tolerance: float, *, error_type: type[Exception]
) -> None:
    if abs(actual - target) / target > tolerance:
        raise error_type(f"quota {name}: actual={actual} target={target} tolerance={tolerance}")


def _validate_selection_quotas(
    records: list[dict[str, Any]],
    selection: Selection,
    enforce_quotas: bool,
    error_type: type[Exception],
) -> None:
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
            for name in item.strata:
                stratified_books[name].add(item.book_id)
        if item.subset in {"stratified", "context"} and not 150 <= row["word_count"] <= 350:
            raise error_type(
                f"{item.subset} passage word count must be 150..350: {row['passage_id']}"
            )
        if item.subset == "continuous":
            continuous_ranges[item.book_id].append(
                (item.chapter_index, item.start_segment_index, item.end_segment_index)
            )
    for book_id, values in continuous_ranges.items():
        for prior, current in pairwise(sorted(values)):
            if current[0] == prior[0] and current[1] <= prior[2]:
                raise error_type(f"continuous ranges overlap: {book_id}")
            if current[0] == prior[0] and current[1] != prior[2] + 1:
                raise error_type(f"continuous ranges have a gap within chapter: {book_id}")
    if not enforce_quotas:
        return
    for name, target in QUOTA_TARGETS.items():
        validate_quota(
            name, buckets[name], target, selection.quota_tolerance, error_type=error_type
        )
    validate_quota(
        "formal",
        buckets["continuous"] + buckets["stratified"] + buckets["context"],
        50_000,
        selection.quota_tolerance,
        error_type=error_type,
    )
    if (
        len(set().union(*(formal_books[name] for name in ("continuous", "stratified", "context"))))
        < 6
    ):
        raise error_type("at least six formal books must contribute target passages")
    if len(formal_books["continuous"]) < 3:
        raise error_type("at least three continuous books must contribute passages")
    for name in STRATA:
        if len(stratified_books[name]) < 3:
            raise error_type(f"stratum {name} must occur in at least three formal books")


def validate_selection(
    spec: Any,
    selection: Selection,
    by_id: dict[str, Any],
    *,
    enforce_quotas: bool = True,
    error_type: type[Exception],
) -> list[dict[str, Any]]:
    if {book.book_id for book in spec.books} != set(by_id):
        raise error_type("source scan does not match BookSpec")
    target_coords: set[tuple[str, int, int]] = set()
    records: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    context_refs: list[tuple[str, int, int]] = []
    for item in selection.passages:
        source = by_id.get(item.book_id)
        if source is None:
            raise error_type(f"unknown book_id: {item.book_id}")
        allowed = {
            "screen": {"screen"},
            "continuous": {"formal"},
            "stratified": {"formal"},
            "context": {"formal"},
            "hidden": {"hidden"},
        }[item.subset]
        if source["book"].split not in allowed:
            raise error_type(
                f"subset={item.subset} cannot use split={source['book'].split}: {item.book_id}"
            )
        segments = range_segments(source["doc"], item, error_type=error_type)
        coords = coordinates(source["doc"])
        selected_coords = [
            (item.book_id, item.chapter_index, segment.index) for segment in segments
        ]
        if any(coord in target_coords for coord in selected_coords):
            raise error_type(f"selected target segment overlap: {item.book_id}")
        target_coords.update(selected_coords)
        pid = passage_id(
            item.book_id,
            item.chapter_index,
            item.start_segment_index,
            item.end_segment_index,
            [segment.source for segment in segments],
        )
        if pid in used_ids:
            raise error_type(f"duplicate passage_id: {pid}")
        used_ids.add(pid)
        context = item.context
        if context is not None:
            before = [coord_key(coord) for coord in context.source_before]
            after = [coord_key(coord) for coord in context.source_after]
            if len(set(before)) != len(before) or len(set(after)) != len(after):
                raise error_type(f"duplicate context coordinate: {pid}")
            if any(coord not in coords for coord in before + after):
                raise error_type(f"context reference does not exist: {pid}")
            target_start = (item.chapter_index, item.start_segment_index)
            target_end = (item.chapter_index, item.end_segment_index)
            if any(coord >= target_start for coord in before) or any(
                coord <= target_end for coord in after
            ):
                raise error_type(f"context reference is on the wrong side of target: {pid}")
            if (
                before != sorted(before)
                or after != sorted(after)
                or [coord_key(value) for value in context.frozen_target_before] != before
            ):
                raise error_type(
                    f"context references are not source-ordered or frozen targets mismatch: {pid}"
                )
            context_refs.extend((item.book_id, *coord) for coord in before + after)
        records.append(
            {
                "selection": item,
                "book": source["book"],
                "source": source,
                "segments": segments,
                "passage_id": pid,
                "word_count": count_words("\n".join(segment.source for segment in segments)),
            }
        )
    if any(coord in target_coords for coord in context_refs):
        raise error_type("context reference is also a selected target segment")
    order = {book.book_id: index for index, book in enumerate(spec.books)}
    records.sort(
        key=lambda row: (
            order[row["book"].book_id],
            row["selection"].chapter_index,
            row["selection"].start_segment_index,
        )
    )
    _validate_selection_quotas(records, selection, enforce_quotas, error_type)
    return records


def runner_record(row: dict[str, Any]) -> dict[str, Any]:
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
            "meta": jsonable(segment.meta),
        }
        for segment in row["segments"]
    ]
    context: dict[str, Any] | None = None
    if item.context is not None:
        challenge: ContextChallenge = item.context
        coords = coordinates(source["doc"])
        context = {
            "challenge_type": challenge.challenge_type,
            "source_before": [
                {
                    "segment_id": segment_id(book_sha, *coord, coords[coord].source),
                    "source": coords[coord].source,
                }
                for coord in (coord_key(value) for value in challenge.source_before)
            ],
            "source_after": [
                {
                    "segment_id": segment_id(book_sha, *coord, coords[coord].source),
                    "source": coords[coord].source,
                }
                for coord in (coord_key(value) for value in challenge.source_after)
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


__all__ = [
    "DIALOGUE_CHARS",
    "QUOTA_TARGETS",
    "STRATA",
    "coord_key",
    "coordinates",
    "jsonable",
    "range_segments",
    "runner_record",
    "suggestion_tags",
    "validate_quota",
    "validate_selection",
]
