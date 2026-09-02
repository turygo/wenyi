"""Validation of frozen benchmark corpus artifacts."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import (
    ArtifactError,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_bytes,
)
from trans_novel.benchmark.corpus.identity import count_words, passage_id, segment_id
from trans_novel.benchmark.schema import EmittedChallengeKey, EmittedManifest, EmittedRunnerRecord
from trans_novel.pipeline.state import RUN_INPUT_SCHEMA_VERSION


def validate_runner_leakage(
    records: Iterable[dict[str, Any]], *, error_type: type[Exception] = ArtifactError
) -> None:
    forbidden = {
        "answer_key",
        "rationale",
        "model",
        "model_id",
        "primary_model",
        "translator_model",
        "analyst_model",
        "editor_model",
        "fast_model",
        "candidate_id",
    }

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in forbidden:
                    raise error_type(f"runner record contains forbidden key {key!r} at {path}")
                walk(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                walk(nested, f"{path}[{index}]")

    for index, record in enumerate(records):
        walk(record, f"runner[{index}]")


def _model(model: Any, value: Any, label: str, error_type: type[Exception]) -> Any:
    try:
        return model.model_validate(value)
    except Exception as error:
        raise error_type(f"invalid {label}: {error}") from error


def _manifest_books(value: Any, error_type: type[Exception]) -> dict[str, dict[str, Any]]:
    manifest = _model(EmittedManifest, value, "source manifest", error_type)
    result: dict[str, dict[str, Any]] = {}
    for item in manifest.books:
        row = item.model_dump(mode="python")
        basename = row["basename"]
        if row["parser_schema"] != RUN_INPUT_SCHEMA_VERSION:
            raise error_type(f"manifest parser schema mismatch: {row['book_id']}")
        if (
            Path(basename).is_absolute()
            or Path(basename).name != basename
            or "/" in basename
            or "\\" in basename
            or basename in {".", ".."}
        ):
            raise error_type(f"manifest basename must be a basename only: {basename!r}")
        if row["book_id"] in result:
            raise error_type(f"duplicate manifest book: {row['book_id']}")
        result[row["book_id"]] = row
    return result


def _segment_coord(value: str, book_sha: str, error_type: type[Exception]) -> tuple[int, int]:
    suffix = (
        value[len(book_sha) + 1 :]
        if isinstance(value, str) and value.startswith(book_sha + ":")
        else ""
    )
    match = re.fullmatch(r"c(\d{4}):s(\d{4}):([0-9a-f]{8})", suffix)
    if match is None:
        raise error_type(f"invalid segment reference: {value}")
    return int(match.group(1)), int(match.group(2))


def _validate_corpus_rows(
    runner: list[dict[str, Any]],
    summaries: list[Any],
    challenge_by_id: dict[str, Any],
    manifest_books: dict[str, Any],
    error_type: type[Exception],
) -> tuple[set[tuple[str, int, int]], set[tuple[str, int, int]]]:
    targets: set[tuple[str, int, int]] = set()
    refs: set[tuple[str, int, int]] = set()
    for row, summary in zip(runner, summaries, strict=True):
        book_id = row["book_id"]
        book = manifest_books.get(book_id)
        if book is None:
            raise error_type(f"runner references unknown book: {book_id}")
        book_sha = book["source_sha256"]
        segments = row["segments"]
        if [s["index"] for s in segments] != list(range(row["start"], row["end"] + 1)):
            raise error_type(f"runner segment range is not contiguous: {row.get('passage_id')}")
        for segment in segments:
            if segment["segment_id"] != segment_id(
                book_sha, row["chapter_index"], segment["index"], segment["source"]
            ):
                raise error_type(f"segment ID mismatch: {segment.get('segment_id')}")
            coord = (book_id, row["chapter_index"], segment["index"])
            if coord in targets:
                raise error_type(f"selected segment overlap: {coord}")
            targets.add(coord)
        expected = passage_id(
            book_id, row["chapter_index"], row["start"], row["end"], [s["source"] for s in segments]
        )
        if row["passage_id"] != expected:
            raise error_type(f"passage ID mismatch: {row.get('passage_id')}")
        if row["word_count"] != count_words("\n".join(s["source"] for s in segments)):
            raise error_type(f"passage word count mismatch: {row.get('passage_id')}")
        if summary != {
            key: row.get(key)
            for key in (
                "passage_id",
                "subset",
                "book_id",
                "chapter_index",
                "start",
                "end",
                "word_count",
                "strata",
            )
        }:
            raise error_type(f"runner/corpus summary mismatch: {row.get('passage_id')}")
        context = row.get("context")
        if row["subset"] == "context":
            if not isinstance(context, dict) or set(context) != {
                "challenge_type",
                "source_before",
                "source_after",
                "frozen_target_before",
            }:
                raise error_type(f"malformed context runner record: {row.get('passage_id')}")
            key = challenge_by_id.get(row["passage_id"])
            if (
                key is None
                or set(key) != {"passage_id", "challenge_type", "answer_key", "rationale"}
                or key["challenge_type"] != context["challenge_type"]
            ):
                raise error_type(f"missing or malformed challenge key: {row.get('passage_id')}")
            before: list[str] = []
            for field in ("source_before", "source_after"):
                prior: tuple[int, int] | None = None
                for ref in context[field]:
                    coord = _segment_coord(ref["segment_id"], book_sha, error_type)
                    if prior is not None and coord <= prior:
                        raise error_type(
                            f"context references are not source-ordered: {row.get('passage_id')}"
                        )
                    prior = coord
                    if ref["segment_id"] != segment_id(book_sha, *coord, ref["source"]):
                        raise error_type(f"context segment ID mismatch: {row.get('passage_id')}")
                    if field == "source_before" and coord >= (row["chapter_index"], row["start"]):
                        raise error_type(
                            f"context source_before does not precede target: {row.get('passage_id')}"
                        )
                    if field == "source_after" and coord <= (row["chapter_index"], row["end"]):
                        raise error_type(
                            f"context source_after does not follow target: {row.get('passage_id')}"
                        )
                    refs.add((book_id, *coord))
                    if field == "source_before":
                        before.append(ref["segment_id"])
            frozen = context["frozen_target_before"]
            if len(frozen) != len(before) or any(
                item.get("segment_id") != ident for item, ident in zip(frozen, before, strict=True)
            ):
                raise error_type(f"frozen target reference mismatch: {row.get('passage_id')}")
        elif context is not None or row["passage_id"] in challenge_by_id:
            raise error_type(
                f"non-context runner record has context or challenge key: {row.get('passage_id')}"
            )
    return targets, refs


def validate_corpus_artifacts(
    root: str | os.PathLike[str],
    *,
    schema_version: int,
    word_counter: str,
    quota_targets: dict[str, int],
    strata: tuple[str, ...],
    error_type: type[Exception] = ArtifactError,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    corpus = read_json(root_path / "corpus.json", error_type=error_type)
    manifest_raw = read_json(root_path / "source_manifest.json", error_type=error_type)
    runner_input = read_jsonl(root_path / "runner_segments.jsonl", error_type=error_type)
    challenge_input = read_jsonl(root_path / "challenge_keys.jsonl", error_type=error_type)
    validate_runner_leakage(runner_input, error_type=error_type)
    manifest_books = _manifest_books(manifest_raw, error_type)
    runner = [
        _model(EmittedRunnerRecord, row, f"runner record {i}", error_type).model_dump(mode="python")
        for i, row in enumerate(runner_input)
    ]
    challenge_keys = [
        _model(EmittedChallengeKey, row, f"challenge key {i}", error_type).model_dump(mode="python")
        for i, row in enumerate(challenge_input)
    ]
    if corpus.get("schema_version") != schema_version or corpus.get("word_counter") != word_counter:
        raise error_type("unsupported corpus schema or word counter")
    if (
        corpus.get("run_input_schema_version") != RUN_INPUT_SCHEMA_VERSION
        or manifest_raw.get("run_input_schema_version") != RUN_INPUT_SCHEMA_VERSION
    ):
        raise error_type("run input schema version mismatch")
    summaries = corpus.get("passages")
    if not isinstance(summaries, list) or len(summaries) != len(runner):
        raise error_type("corpus passage summaries do not match runner")
    challenge_by_id = {row["passage_id"]: row for row in challenge_keys}
    if len(challenge_by_id) != len(challenge_keys):
        raise error_type("duplicate challenge key passage_id")
    targets, refs = _validate_corpus_rows(
        runner, summaries, challenge_by_id, manifest_books, error_type
    )
    actual = defaultdict(int, dict.fromkeys((*quota_targets, "hidden", "formal"), 0))
    formal: set[str] = set()
    if targets & refs:
        raise error_type("context reference is a selected target")
    if set(challenge_by_id) != {row["passage_id"] for row in runner if row["subset"] == "context"}:
        raise error_type("challenge key set does not match context runner set")
    continuous: set[str] = set()
    stratified: dict[str, set[str]] = defaultdict(set)
    ranges: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in runner:
        subset = row["subset"]
        expected_split = {"screen": "screen", "hidden": "hidden"}.get(subset, "formal")
        if manifest_books[row["book_id"]]["split"] != expected_split:
            raise error_type(f"runner subset/split mismatch: {row['passage_id']}")
        actual[subset] += row["word_count"]
        if subset in {"continuous", "stratified", "context"}:
            formal.add(row["book_id"])
        if subset == "continuous":
            continuous.add(row["book_id"])
            ranges[row["book_id"]].append((row["chapter_index"], row["start"], row["end"]))
        if subset == "stratified":
            for name in row["strata"]:
                stratified[name].add(row["book_id"])
        if subset in {"stratified", "context"} and not 150 <= row["word_count"] <= 350:
            raise error_type(f"{subset} passage word count must be 150..350: {row['passage_id']}")
    for book_id, values in ranges.items():
        for prior, current in pairwise(sorted(values)):
            if current[0] == prior[0] and current[1] <= prior[2]:
                raise error_type(f"continuous ranges overlap: {book_id}")
            if current[0] == prior[0] and current[1] != prior[2] + 1:
                raise error_type(f"continuous ranges have a gap within chapter: {book_id}")
    actual["formal"] = actual["continuous"] + actual["stratified"] + actual["context"]
    if corpus.get("quotas", {}).get("actual") != dict(actual):
        raise error_type("quota totals do not match runner")
    tolerance = corpus.get("quotas", {}).get("tolerance")
    if not isinstance(tolerance, int | float) or not 0 <= tolerance <= 0.20:
        raise error_type("invalid quota tolerance")
    for name, target in (*quota_targets.items(), ("formal", 50_000)):
        if abs(actual[name] - target) / target > tolerance:
            raise error_type(
                f"quota {name}: actual={actual[name]} target={target} tolerance={tolerance}"
            )
    if len(formal) < 6:
        raise error_type(f"formal book coverage is {len(formal)}, requires 6")
    if len(continuous) < 3:
        raise error_type(f"continuous book coverage is {len(continuous)}, requires 3")
    for name in strata:
        if len(stratified[name]) < 3:
            raise error_type(f"stratum {name} coverage is {len(stratified[name])}, requires 3")
    semantics = {
        "corpus": {key: value for key, value in corpus.items() if key != "corpus_sha256"},
        "runner_segments": runner,
        "challenge_keys": challenge_keys,
    }
    digest = sha256_bytes(canonical_json(semantics).encode("utf-8"))
    if corpus.get("corpus_sha256") != digest:
        raise error_type("corpus_sha256 mismatch")
    return {
        "corpus_sha256": digest,
        "runner_count": len(runner),
        "challenge_count": len(challenge_keys),
        "book_count": len(manifest_books),
        "word_counts": dict(actual),
        "split_counts": {
            split: sum(1 for row in runner if manifest_books[row["book_id"]]["split"] == split)
            for split in ("screen", "formal", "hidden")
        },
        "bucket_counts": {
            subset: sum(1 for row in runner if row["subset"] == subset)
            for subset in ("screen", "continuous", "stratified", "context", "hidden")
        },
        "book_counts": {
            book_id: sum(1 for row in runner if row["book_id"] == book_id)
            for book_id in sorted(manifest_books)
        },
    }


__all__ = ["validate_corpus_artifacts", "validate_runner_leakage"]
