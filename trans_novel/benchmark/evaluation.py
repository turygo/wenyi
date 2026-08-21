"""Deterministic offline model-blind evaluation packs."""

from __future__ import annotations

import hashlib
import html
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from trans_novel.benchmark.corpus import canonical_json, count_words, sha256_bytes, validate_corpus
from trans_novel.benchmark.schema import (
    AbsoluteResponse,
    ContextResponse,
    EvaluationSpec,
    MQMResponse,
    PairwiseResponse,
    PolishResponse,
    PosteditResponse,
)

SCHEMA_VERSION = 1
_KIND_ORDER = ("absolute", "pairwise", "polish", "mqm", "context", "postedit")
_SINGLE_LABEL = "译文"
_RESPONSE_MODELS = {
    "absolute": AbsoluteResponse,
    "pairwise": PairwiseResponse,
    "polish": PolishResponse,
    "mqm": MQMResponse,
    "context": ContextResponse,
    "postedit": PosteditResponse,
}
_FORBIDDEN_KEYS = {
    "candidate_id",
    "model",
    "model_id",
    "primary_model",
    "editor_model",
    "fast_model",
    "provider",
    "artifact_id",
    "artifact_key",
    "stage",
    "operation",
    "agent",
    "strategy",
    "context_strategy",
}


class EvaluationError(ValueError):
    """Evaluation input or artifact violates the contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"invalid JSON: {path}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise EvaluationError(f"cannot read JSONL: {path}") from error
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(f"invalid JSONL: {path}") from error
        if not isinstance(value, dict):
            raise EvaluationError(f"JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def _hash(value: Any, *, without: str | None = None) -> str:
    if isinstance(value, dict) and without is not None:
        value = {key: item for key, item in value.items() if key != without}
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _atomic_json(
    path: Path, value: Any, mode: int | None = None, *, escape_lt: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            serialized = canonical_json(value)
            if escape_lt:
                serialized = serialized.replace("<", "\\u003c")
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            with suppress(OSError):
                os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def _spec(value: str | os.PathLike[str] | dict[str, Any]) -> EvaluationSpec:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = yaml.safe_load(Path(value).expanduser().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise EvaluationError(f"cannot read evaluation spec: {value}") from error
    if not isinstance(raw, dict):
        raise EvaluationError("evaluation spec must be a mapping")
    try:
        return EvaluationSpec.model_validate(raw)
    except Exception as error:
        raise EvaluationError(f"invalid evaluation spec: {error}") from error


def _seed(seed: int, purpose: str, scope: Iterable[str] = ()) -> int:
    material = {"seed": seed, "purpose": purpose, "scope": list(scope)}
    return int.from_bytes(hashlib.sha256(canonical_json(material).encode()).digest()[:16], "big")


def _aid(seed: int, kind: str, scope: Iterable[str]) -> str:
    """Return a deterministic random-looking assignment identifier."""
    return _hash(
        {
            "seed": seed,
            "purpose": "assignment_id",
            "kind": kind,
            "scope": list(scope),
        }
    )[:32]


def _sanitize_assignment_ids(
    assignments: list[tuple[str, dict[str, Any]]],
    mapping: dict[str, Any],
    seed: int,
    forbidden: set[str],
) -> None:
    remap: dict[str, str] = {}
    for _, item in assignments:
        original = item["assignment_id"]
        candidate = original
        counter = 0
        while any(token and token in candidate for token in forbidden):
            candidate = _aid(seed, "safe_assignment_id", (original, str(counter)))
            counter += 1
        if candidate != original:
            remap[original] = candidate
            item["assignment_id"] = candidate
    if not remap:
        return
    updated: dict[str, Any] = {}
    for original, value in mapping.items():
        new_key = remap.get(original, original)
        copied = deepcopy(value)
        duplicate = copied.get("duplicate_of")
        if duplicate in remap:
            copied["duplicate_of"] = remap[duplicate]
        updated[new_key] = copied
    mapping.clear()
    mapping.update(updated)


def _unit_id(corpus_hash: str, passage: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    return _hash(
        {
            "corpus_sha256": corpus_hash,
            "passage_id": passage["passage_id"],
            "segment_ids": [s["segment_id"] for s in segments],
            "source": "\n".join(str(s.get("source", "")) for s in segments),
        }
    )


def build_units(rows: list[dict[str, Any]], corpus_hash: str) -> list[dict[str, Any]]:
    """Group whole ordered segments into 150--350 word evaluation units."""
    result: list[dict[str, Any]] = []
    for passage in rows:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        words = 0
        for segment in passage.get("segments", []):
            count = int(segment.get("word_count", count_words(str(segment.get("source", "")))))
            # A group may be short because the next indivisible segment would
            # cross the upper bound.  Never create a multi-segment unit over
            # 350 words; the boundary exception applies only to one segment.
            if current and words + count > 350:
                groups.append(current)
                current, words = [], 0
            current.append(segment)
            words += count
        if current:
            groups.append(current)
        for group in groups:
            source = "\n".join(str(s.get("source", "")) for s in group)
            actual = count_words(source)
            result.append(
                {
                    "unit_id": _unit_id(corpus_hash, passage, group),
                    "passage_id": passage["passage_id"],
                    "book_id": passage.get("book_id"),
                    "subset": passage.get("subset"),
                    "chapter_index": passage.get("chapter_index", 0),
                    "strata": list(passage.get("strata", [])),
                    "segment_ids": [s["segment_id"] for s in group],
                    "segments": group,
                    "source": source,
                    "word_count": actual,
                    "boundary": not 150 <= actual <= 350,
                }
            )
    return result


def _select(
    units: list[dict[str, Any]], target: int, seed: int, purpose: str
) -> list[dict[str, Any]]:
    if not units:
        return []
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_book[str(unit.get("book_id"))].append(unit)
    books = sorted(by_book)
    random.Random(_seed(seed, "unit_selection", [purpose])).shuffle(books)
    for book in books:
        by_book[book].sort(key=lambda row: row["unit_id"])
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    words_by_book: dict[str, int] = defaultdict(int)
    cap = target * 0.25
    while True:
        progressed = False
        for book in books:
            available = [row for row in by_book[book] if row["unit_id"] not in used]
            if not available:
                continue
            under_cap = [row for row in available if words_by_book[book] + row["word_count"] <= cap]
            row = under_cap[0] if under_cap else None
            if row is None:
                # Relax only when this book is currently least represented.
                row = min(
                    available,
                    key=lambda value: (
                        words_by_book[book] / max(1, sum(item["word_count"] for item in chosen)),
                        value["unit_id"],
                    ),
                )
            chosen.append(row)
            used.add(row["unit_id"])
            words_by_book[book] += row["word_count"]
            progressed = True
            if sum(item["word_count"] for item in chosen) >= target:
                return chosen
        if not progressed:
            return chosen


def _load_inputs(
    corpus_dir: Path, run_dir: Path, spec: EvaluationSpec
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    try:
        checked = validate_corpus(corpus_dir)
    except Exception as error:
        raise EvaluationError(f"invalid corpus: {error}") from error
    if checked.get("corpus_sha256") != spec.corpus_sha256:
        raise EvaluationError("corpus hash mismatch")
    rows = _read_jsonl(corpus_dir / "runner_segments.jsonl")
    challenge = {row["passage_id"]: row for row in _read_jsonl(corpus_dir / "challenge_keys.jsonl")}
    candidates = _read_json(run_dir / "candidates.json")
    if not isinstance(candidates, list):
        raise EvaluationError("candidates.json must be a list")
    return rows, challenge, candidates


def _records(
    run: Path,
    candidates: list[dict[str, Any]],
    wanted: set[str],
    corpus_rows: list[dict[str, Any]] | None = None,
) -> dict[tuple[str, int, str, str, str, str], dict[str, Any]]:
    """Load one provenance-exact record for each candidate/replicate output."""
    result: dict[tuple[str, int, str, str, str, str], dict[str, Any]] = {}
    passage_by_segment = {
        str(segment["segment_id"]): str(passage["passage_id"])
        for passage in (corpus_rows or [])
        for segment in passage.get("segments", [])
        if isinstance(segment, dict) and "segment_id" in segment
    }

    def put(
        candidate: str,
        replicate: int,
        surface: str,
        passage_id: str,
        strategy: str,
        row: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        segment_id = str(row.get("segment_id", ""))
        if not segment_id or not passage_id:
            raise EvaluationError("artifact segment provenance is incomplete")
        value = {
            "source": str(row.get("source", "")),
            "target": str(
                row.get("target", row.get("final", row.get("final_after_full_pipeline", "")))
            ),
            "raw": str(
                row.get(
                    "translation_raw", row.get("raw", row.get("raw_after_translation_lint", ""))
                )
            ),
            "proposal": row.get("polish_proposal", row.get("proposal")),
            "polish_accepted": row.get("polish_accepted"),
            "metadata": dict(metadata),
        }
        key = (candidate, replicate, surface, passage_id, segment_id, strategy)
        prior = result.get(key)
        if prior is not None and prior != value:
            raise EvaluationError("conflicting duplicate segment output")
        result[key] = value

    def load_artifact(
        candidate: str,
        replicate: int,
        surface: str,
        strategy: str,
        reference: str,
        candidate_row: dict[str, Any],
        *,
        editor: bool,
    ) -> None:
        passages = run / reference / "passages"
        if not passages.is_dir():
            raise EvaluationError(f"missing artifact passages: {reference}")
        for passage_path in sorted(passages.glob("*.json")):
            passage = _read_json(passage_path)
            if not isinstance(passage, dict):
                raise EvaluationError("artifact passage must be an object")
            passage_id = str(passage.get("passage_id", ""))
            actual_replicate = int(passage.get("replicate", replicate))
            if actual_replicate != replicate:
                raise EvaluationError("artifact replicate mismatch")
            resolved_model = passage.get("editor_model" if editor else "primary_model")
            if resolved_model is None:
                resolved_model = candidate_row.get("editor_model" if editor else "primary_model")
            metadata = {
                "artifact_id": reference,
                "artifact_key": passage.get("artifact_key", reference),
                "stage": "editor_final" if editor else "translation_final",
                "stage_name": "attribution_final"
                if surface == "attribution_final"
                else "context_diagnostic",
                "provider": passage.get("provider", candidate_row.get("provider")),
                "requested_model_id": candidate_row.get(
                    "editor_model" if editor else "primary_model"
                ),
                "resolved_model_id": resolved_model,
                "replicate": replicate,
                "surface": surface,
                "context_strategy": strategy,
                "raw_artifact_id": candidate_row.get("raw_artifact_id"),
                "editor_artifact_id": candidate_row.get("editor_artifact_id"),
            }
            for row in passage.get("segments", []):
                if isinstance(row, dict):
                    put(candidate, replicate, surface, passage_id, strategy, row, metadata)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise EvaluationError("candidate manifest must be an object")
        cid = candidate.get("candidate_id")
        if cid not in wanted:
            continue
        try:
            replicate = int(candidate["replicate"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvaluationError("candidate replicate is invalid") from error
        translations = candidate.get("translation_artifacts", {})
        editors = candidate.get("editor_artifacts", {})
        if not isinstance(translations, dict) or not isinstance(editors, dict):
            raise EvaluationError("candidate artifact references are invalid")
        edited = bool(candidate.get("editor_model"))
        for scope, reference in translations.items():
            if not isinstance(reference, str):
                raise EvaluationError("translation artifact reference is invalid")
            _, _, strategy = str(scope).partition(":")
            strategy = strategy or "c2"
            if strategy == "c2" and not edited:
                load_artifact(cid, replicate, "polish", "c2", reference, candidate, editor=False)
            load_artifact(cid, replicate, "context", strategy, reference, candidate, editor=False)
            if strategy == "c2" and not edited:
                load_artifact(
                    cid, replicate, "attribution_final", "c2", reference, candidate, editor=False
                )
        if edited:
            for scope, reference in editors.items():
                if not isinstance(reference, str):
                    raise EvaluationError("editor artifact reference is invalid")
                if (str(scope).partition(":")[2] or "c2") != "c2":
                    continue
                # Edited C2 records are the layer-one raw/proposal pair.  The
                # translation artifact remains the source of the raw text;
                # the editor artifact supplies proposal/acceptance provenance.
                load_artifact(cid, replicate, "polish", "c2", reference, candidate, editor=True)
                load_artifact(
                    cid, replicate, "attribution_final", "c2", reference, candidate, editor=True
                )
        for row in candidate.get("stage", []):
            if not isinstance(row, dict):
                continue
            segment_id = str(row.get("segment_id", ""))
            passage_id = str(row.get("passage_id", "")) or passage_by_segment.get(segment_id, "")
            metadata = {
                "artifact_id": candidate.get("branch_artifact_id"),
                "artifact_key": candidate.get("branch_artifact_id"),
                "stage": "full_final",
                "stage_name": "full_final",
                "provider": candidate.get("provider"),
                "requested_model_id": candidate.get("primary_model"),
                "resolved_model_id": candidate.get("primary_model"),
                "replicate": replicate,
                "surface": "full_final",
                "context_strategy": "c2",
                "raw_artifact_id": candidate.get("raw_artifact_id"),
                "editor_artifact_id": candidate.get("branch_artifact_id"),
            }
            put(
                cid,
                replicate,
                "full_final",
                passage_id,
                "c2",
                {
                    "segment_id": segment_id,
                    "source": row.get("source", ""),
                    "final_after_full_pipeline": row.get(
                        "final_after_full_pipeline", row.get("final", "")
                    ),
                    "raw_after_translation_lint": row.get(
                        "raw_after_translation_lint", row.get("translation_raw", "")
                    ),
                    "polish_proposal": row.get("polish_proposal"),
                },
                metadata,
            )
    if not result:
        raise EvaluationError("run has no eligible candidate output")
    return result


def _output(
    records: dict[tuple[str, int, str, str, str, str], dict[str, Any]],
    unit: dict[str, Any],
    candidate: str,
    replicate: int,
    surface: str,
    strategy: str = "c2",
) -> dict[str, Any] | None:
    values = [
        records.get((candidate, replicate, surface, unit["passage_id"], segment_id, strategy))
        for segment_id in unit["segment_ids"]
    ]
    if any(value is None for value in values):
        return None
    checked: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict):
            raise EvaluationError("artifact output is invalid")
        target = str(value.get("target", ""))
        source = str(value.get("source", ""))
        metadata = value.get("metadata")
        missing = [
            field
            for field, present in (
                ("target", bool(target.strip())),
                ("source", bool(source.strip())),
                ("metadata", isinstance(metadata, dict)),
            )
            if not present
        ]
        if missing:
            raise EvaluationError(
                "artifact output provenance incomplete "
                f"unit={unit['unit_id']} replicate={replicate} "
                f"surface={surface} strategy={strategy} fields={','.join(missing)}"
            )
        checked.append(value)
    return {
        "source": unit["source"],
        "target": "\n".join(str(value["target"]) for value in checked),
        "raw": "\n".join(str(value.get("raw", "")) for value in checked),
        "proposal": "\n".join(
            str(value.get("proposal", "")) for value in checked if value.get("proposal") is not None
        ),
        "polish_accepted": all(value.get("polish_accepted") is True for value in checked),
        "metadata": [dict(value["metadata"]) for value in checked],
    }


def _make_assignments(
    spec: EvaluationSpec,
    units: list[dict[str, Any]],
    records: dict[tuple[str, int, str, str, str, str], dict[str, Any]],
    challenge: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    replicate_count = max(
        (int(record["metadata"].get("replicate", 1)) for record in records.values()),
        default=1,
    )

    def replica(index: int, purpose: str, scope: Iterable[str]) -> int:
        if replicate_count <= 1:
            return 1
        offset = _seed(spec.seed, purpose, scope) % replicate_count
        return 1 + ((index + offset) % replicate_count)

    context_units = [
        unit
        for unit in units
        if unit.get("subset") == "context" and unit["passage_id"] in challenge
    ]
    selected = {
        kind: _select(
            context_units if kind == "context" else units,
            getattr(spec, kind).target_source_words,
            spec.seed,
            kind,
        )
        for kind in _KIND_ORDER
        if kind != "pairwise"
    }
    pair_count = len(spec.enabled_surfaces)
    pair_target = spec.pairwise.target_source_words // pair_count
    pair_remainder = spec.pairwise.target_source_words % pair_count
    selected_pairwise = {
        surface: _select(
            units,
            pair_target + (1 if index < pair_remainder else 0),
            spec.seed,
            f"pairwise:{surface}",
        )
        for index, surface in enumerate(spec.enabled_surfaces)
    }

    def position(
        candidate: str,
        replicate_index: int,
        surface: str,
        strategy: str,
        output: dict[str, Any],
        label: str | None = None,
    ) -> dict[str, Any]:
        metadata = output.get("metadata", [])
        first = dict(metadata[0]) if metadata else {}
        first.update(
            {
                "candidate_id": candidate,
                "replicate": replicate_index,
                "surface": surface,
                "context_strategy": strategy,
                "polish_accepted": output.get("polish_accepted"),
            }
        )
        first["segment_provenance"] = metadata
        if label is not None:
            first["polish_position"] = label
        return first

    def add(
        kind: str,
        unit: dict[str, Any],
        payload: dict[str, Any],
        positions: list[dict[str, Any]],
        surface: str,
        strategy: str = "c2",
    ) -> None:
        aid = _aid(spec.seed, kind, (unit["unit_id"], str(len(base))))
        item = {
            "assignment_id": aid,
            "kind": kind,
            "unit_id": unit["unit_id"],
            "calibration": False,
            "surface": _SINGLE_LABEL,
            **payload,
        }
        base.append(item)
        metadata = {
            "kind": kind,
            "unit_id": unit["unit_id"],
            "surface": surface,
            "positions": positions,
            "duplicate_of": None,
            "calibration": False,
        }
        if kind != "pairwise":
            metadata["strategy"] = strategy
        mapping[aid] = metadata

    for kind in ("absolute", "mqm", "postedit"):
        for index, unit in enumerate(selected[kind]):
            for surface in spec.enabled_surfaces:
                for candidate_index, candidate in enumerate(spec.candidate_ids):
                    replicate = replica(
                        index,
                        "replicate_split",
                        (kind, candidate, surface, unit["unit_id"], str(candidate_index)),
                    )
                    output = _output(records, unit, candidate, replicate, surface)
                    if output is None:
                        raise EvaluationError(
                            f"missing candidate/surface output for {kind}: {candidate}/{surface}"
                        )
                    if kind == "absolute":
                        payload = {
                            "source": unit["source"],
                            "target": output["target"],
                            "dimensions": [
                                "fidelity",
                                "naturalness",
                                "style_voice",
                                "consistency",
                                "context_handling",
                                "readability",
                                "format_integrity",
                            ],
                        }
                    elif kind == "mqm":
                        payload = {
                            "source": unit["source"],
                            "target": output["target"],
                            "segment_ids": unit["segment_ids"],
                        }
                    else:
                        payload = {"source": unit["source"], "target": output["target"]}
                    add(
                        kind,
                        unit,
                        payload,
                        [position(candidate, replicate, surface, "c2", output)],
                        surface,
                    )

    for index, unit in enumerate(selected["polish"]):
        for candidate_index, candidate in enumerate(spec.candidate_ids):
            replicate = replica(
                index,
                "replicate_split",
                ("polish", candidate, unit["unit_id"], str(candidate_index)),
            )
            output = _output(records, unit, candidate, replicate, "polish")
            if output is None or not output["proposal"].strip():
                if output is None:
                    raise EvaluationError(f"missing polish candidate output: {candidate}")
                continue
            add(
                "polish",
                unit,
                {
                    "source": unit["source"],
                    "target_a": output["raw"],
                    "target_b": output["proposal"],
                    "labels": ["A", "B"],
                },
                [
                    position(candidate, replicate, "polish", "c2", output, "raw"),
                    position(candidate, replicate, "polish", "c2", output, "proposal"),
                ],
                "polish",
            )

    pairs = [
        (left, right)
        for index, left in enumerate(spec.candidate_ids)
        for right in spec.candidate_ids[index + 1 :]
    ]
    for surface in spec.enabled_surfaces:
        pair_units = selected_pairwise[surface]
        if len(pair_units) < len(spec.candidate_ids) - 1:
            raise EvaluationError("pairwise surface cannot form a connected graph")
        for index, unit in enumerate(pair_units):
            left, right = pairs[index % len(pairs)]
            replicate = replica(
                index,
                "replicate_split",
                ("pairwise", surface, unit["unit_id"]),
            )
            output_a = _output(records, unit, left, replicate, surface)
            output_b = _output(records, unit, right, replicate, surface)
            if output_a is None or output_b is None:
                raise EvaluationError(
                    f"missing pairwise candidate/surface output: {left}/{right}/{surface}"
                )
            add(
                "pairwise",
                unit,
                {
                    "source": unit["source"],
                    "target_a": output_a["target"],
                    "target_b": output_b["target"],
                    "labels": ["A", "B"],
                },
                [
                    position(left, replicate, surface, "c2", output_a, "A"),
                    position(right, replicate, surface, "c2", output_b, "B"),
                ],
                surface,
            )

    for index, unit in enumerate(selected["context"]):
        key = challenge[unit["passage_id"]]
        for strategy in ("c0", "c1", "c2"):
            for candidate_index, candidate in enumerate(spec.candidate_ids):
                replicate = replica(
                    index,
                    "replicate_split",
                    ("context", candidate, strategy, unit["unit_id"], str(candidate_index)),
                )
                output = _output(records, unit, candidate, replicate, "context", strategy)
                if output is None:
                    raise EvaluationError(
                        f"missing context strategy output: {candidate}/{strategy}"
                    )
                add(
                    "context",
                    unit,
                    {
                        "source": unit["source"],
                        "current_output": output["target"],
                        "answer_key": key.get("answer_key", ""),
                        "rationale": key.get("rationale", ""),
                    },
                    [position(candidate, replicate, "context", strategy, output)],
                    "context",
                    strategy,
                )
    if any(not any(item["kind"] == kind for item in base) for kind in _KIND_ORDER):
        raise EvaluationError("every evaluation kind needs an eligible assignment")
    return base, mapping


def _provenance_identifiers(
    spec: dict[str, Any],
    secret: dict[str, Any],
) -> set[str]:
    identifiers = {str(value) for value in spec.get("evaluation_spec", {}).get("candidate_ids", [])}
    metadata_keys = {
        "candidate_id",
        "provider",
        "model",
        "model_id",
        "requested_model_id",
        "resolved_model_id",
        "artifact_id",
        "artifact_key",
        "raw_artifact_id",
        "editor_artifact_id",
        "stage",
        "stage_name",
        "operation",
        "agent",
    }

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in metadata_keys and value:
            identifiers.add(value)

    visit(secret)
    return {value for value in identifiers if value}


def _allocate(
    base: list[dict[str, Any]],
    spec: EvaluationSpec,
    mapping: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    by_kind = {kind: [item for item in base if item["kind"] == kind] for kind in _KIND_ORDER}
    ordered = sorted(spec.raters)
    counts = {
        "absolute": 3,
        "polish": 3,
        "context": 3,
        "pairwise": 2,
        "mqm": 2,
        "postedit": 1,
    }

    def swapped(item: dict[str, Any], metadata: dict[str, Any]) -> None:
        if item["kind"] not in {"pairwise", "polish"}:
            return
        item["target_a"], item["target_b"] = item["target_b"], item["target_a"]
        positions = metadata.get("positions", [])
        if len(positions) >= 2:
            positions[0], positions[1] = positions[1], positions[0]

    def presentation(
        source: dict[str, Any],
        rater: str,
        purpose: str,
        ordinal: int,
        *,
        calibration: bool = False,
        reverse: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        aid = _aid(spec.seed, purpose, (rater, source["assignment_id"], str(ordinal)))
        copy = dict(source)
        copy["assignment_id"] = aid
        copy["calibration"] = calibration
        metadata = deepcopy(mapping[source["assignment_id"]])
        metadata["calibration"] = calibration
        metadata["duplicate_of"] = None
        if reverse:
            swapped(copy, metadata)
        mapping[aid] = metadata
        return rater, copy

    for rater in ordered:
        for kind in _KIND_ORDER:
            pool = by_kind[kind]
            if not pool:
                raise EvaluationError(f"every evaluation kind needs an eligible assignment: {kind}")
            offset = _seed(spec.seed, "calibration", [rater, kind]) % len(pool)
            for index in range(5):
                source = pool[(offset + index) % len(pool)]
                result.append(
                    presentation(
                        source,
                        rater,
                        "calibration",
                        index,
                        calibration=True,
                    )
                )

    positions: dict[str, int] = defaultdict(int)
    postedit_positions: dict[tuple[str, str, str], int] = defaultdict(int)
    postedit_keys: dict[str, tuple[str, str, str]] = {}
    for source in base:
        if source["kind"] != "postedit":
            continue
        metadata = mapping[source["assignment_id"]]
        source_position = metadata.get("positions")
        if not isinstance(source_position, list) or len(source_position) != 1:
            raise EvaluationError("postedit assignment provenance is invalid")
        position = source_position[0]
        candidate = position.get("candidate_id")
        surface = metadata.get("surface")
        if not isinstance(candidate, str) or not isinstance(surface, str):
            raise EvaluationError("postedit assignment partition is invalid")
        postedit_keys[source["assignment_id"]] = ("postedit", candidate, surface)
    postedit_starts: dict[tuple[str, str, str], int] = {}
    postedit_totals = [0] * len(ordered)
    for key in sorted(set(postedit_keys.values())):
        count = sum(value == key for value in postedit_keys.values())
        preferred = _seed(spec.seed, "base_rater_allocation", list(key)) % len(ordered)
        choices: list[tuple[tuple[int, int, int], int, list[int]]] = []
        for offset in range(len(ordered)):
            projection = list(postedit_totals)
            for index in range(count):
                projection[(offset + index) % len(ordered)] += 1
            distance = min(
                (offset - preferred) % len(ordered),
                (preferred - offset) % len(ordered),
            )
            choices.append(
                (
                    (max(projection) - min(projection), distance, offset),
                    offset,
                    projection,
                )
            )
        _, selected_offset, postedit_totals = min(choices, key=lambda choice: choice[0])
        postedit_starts[key] = selected_offset
    for source in base:
        kind = source["kind"]
        if kind == "postedit":
            key = postedit_keys[source["assignment_id"]]
            ordinal = postedit_positions[key]
            postedit_positions[key] += 1
            offset = postedit_starts[key]
            assigned = [ordered[(offset + ordinal) % len(ordered)]]
        else:
            index = positions[kind]
            positions[kind] += 1
            offset = _seed(spec.seed, "base_rater_allocation", [kind]) % len(ordered)
            assigned = [ordered[(offset + index + n) % len(ordered)] for n in range(counts[kind])]
        for ordinal, rater in enumerate(assigned):
            result.append(
                presentation(
                    source,
                    rater,
                    "presentation",
                    ordinal,
                    reverse=kind in {"pairwise", "polish"} and ordinal % 2 == 1,
                )
            )

    per_rater_kind: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rater, item in result:
        if not item["calibration"]:
            per_rater_kind[(rater, item["kind"])].append(item)
    for rater in ordered:
        for kind in _KIND_ORDER:
            rows = list(per_rater_kind[(rater, kind)])
            random.Random(
                _seed(
                    spec.seed,
                    "hidden_duplicates",
                    [rater, kind],
                )
            ).shuffle(rows)
            duplicate_count = int(len(rows) * spec.hidden_duplicate_fraction + 0.5)
            for ordinal, source in enumerate(rows[:duplicate_count]):
                rater_id, copy = presentation(
                    source,
                    rater,
                    "duplicate",
                    ordinal,
                    reverse=False,
                )
                duplicate_mapping = mapping[copy["assignment_id"]]
                duplicate_mapping["duplicate_of"] = source["assignment_id"]
                if kind in {"pairwise", "polish"}:
                    swapped(copy, duplicate_mapping)
                result.append((rater_id, copy))
    for source in base:
        mapping.pop(source["assignment_id"], None)
    return sorted(
        result,
        key=lambda row: (row[0], not row[1]["calibration"], row[1]["assignment_id"]),
    )


_CONTENT_KEYS = {
    "source",
    "target",
    "target_a",
    "target_b",
    "current_output",
    "answer_key",
    "rationale",
    "note",
    "study_protocol",
    "eligibility_text",
    "consent_text",
    "compensation_text",
    "retention_text",
    "kind",
    "unit_id",
    "assignment_id",
    "surface",
    "calibration",
    "labels",
    "dimensions",
    "segment_ids",
    "pack_hash",
    "rater_id",
    "schema_version",
}


def _leak(value: Any, identifiers: set[str], key_name: str = "") -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_KEYS:
                return True
            if _leak(child, identifiers, key):
                return True
        return False
    if isinstance(value, list):
        return any(_leak(child, identifiers, key_name) for child in value)
    if isinstance(value, str) and key_name not in _CONTENT_KEYS:
        return any(
            token
            and re.search(
                rf"(?<![A-Za-z0-9._-]){re.escape(token)}(?![A-Za-z0-9._-])",
                value,
            )
            for token in identifiers
        )
    return False


_INDEX = (
    '<!doctype html><meta charset="utf-8"><title>Offline Evaluation</title>'
    '<script src="app.js" defer></script><link rel="stylesheet" href="style.css">'
    '<main><h1>离线评估</h1><p id="eligibility">参与资格：__ELIGIBILITY__</p>'
    '<p id="consent-notice">知情同意：__CONSENT__</p>'
    '<p id="compensation">报酬：__COMPENSATION__</p>'
    '<p id="retention">数据保留：__RETENTION__</p>'
    '<label><input id="consent" type="checkbox">我同意参与</label>'
    '<button id="start" disabled>开始</button>'
    '<button id="download" disabled>下载</button><p id="status"></p>'
    '<section id="assignment"></section></main>'
    '<script>window.PACK_HASH="__PACK_HASH__";window.RATER_ID="__RATER__";</script>'
)

_JS = r"""(()=> {
const PACK=window.PACK_HASH,RATER=window.RATER_ID;
const storageKey=`trans-novel-evaluation:${PACK}:${RATER}`;
const requiredState=["schema_version","consented_at","responses","current_assignment_id",
  "current_started_at","current_started_ms","active_ms","current_draft"];
const section=document.getElementById("assignment"), consent=document.getElementById("consent");
const start=document.getElementById("start"), download=document.getElementById("download");
const status=document.getElementById("status");
let assignments=[], state, visibleStart=0;
const emptyState=()=>({schema_version:1,consented_at:null,responses:[],
  current_assignment_id:null,current_started_at:null,current_started_ms:null,
  active_ms:0,current_draft:null});
const validState=value=>{
  if(!value||typeof value!=="object"||Array.isArray(value)||value.schema_version!==1||
     Object.keys(value).some(k=>!requiredState.includes(k))||
     requiredState.some(k=>!Object.prototype.hasOwnProperty.call(value,k))||
     (value.consented_at!==null&&typeof value.consented_at!=="string")||
     !Array.isArray(value.responses)||!Number.isFinite(value.active_ms)||value.active_ms<0||
     (value.current_assignment_id!==null&&typeof value.current_assignment_id!=="string")||
     (value.current_started_at!==null&&typeof value.current_started_at!=="string")||
     (value.current_started_ms!==null&&(!Number.isFinite(value.current_started_ms)||value.current_started_ms<0))||
     (value.current_draft!==null&&(typeof value.current_draft!=="object"||Array.isArray(value.current_draft)||
       (value.current_draft.errors!==undefined&&(!Array.isArray(value.current_draft.errors)||
         !value.current_draft.errors.every(error=>error&&typeof error==="object"&&!Array.isArray(error)))))) return false;
  const seen=new Set;
  return value.responses.every(row=>row&&typeof row==="object"&&!Array.isArray(row)&&
    typeof row.assignment_id==="string"&&!seen.has(row.assignment_id)&&seen.add(row.assignment_id));
};
const save=()=>localStorage.setItem(storageKey,JSON.stringify(state));
const load=()=>{try{const value=JSON.parse(localStorage.getItem(storageKey)||"null");
  state=validState(value)?value:emptyState()}catch(_){state=emptyState()}};
const assignment=()=>assignments.find(row=>row.assignment_id===state.current_assignment_id);
const submittedIds=()=>new Set(state.responses.map(row=>row.assignment_id));
const flush=endMs=>{if(visibleStart!==0){state.active_ms+=Math.max(0,endMs-visibleStart);visibleStart=0;save()}};
const resume=()=>{if(state.current_assignment_id!==null&&visibleStart===0){visibleStart=Date.now()}};
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const labels={"a_much_better":"A 大幅更好","a_slightly_better":"A 略好",tie:"两者相当",
  "b_slightly_better":"B 略好","b_much_better":"B 大幅更好",
  "clearly_improved":"明显改善","slightly_improved":"略有改善","no_material_change":"没有明显变化",
  "fluent_but_semantic_damage":"表达流畅但语义受损","quality_declined":"质量下降",
  correct:"正确",incorrect:"不正确",uncertain:"无法判断",
  critical:"严重",major:"主要",minor:"次要",
  mistranslation:"误译",omission:"漏译",addition:"增译",hallucination:"幻觉",
  terminology:"术语",named_entity:"专名",pronoun_reference:"指代",style_register:"文体",
  fluency:"流畅度",formatting:"格式"};
const options=(values,selected)=>`<option value=""></option>`+values.map(v=>`<option value="${v}" ${selected===v?"selected":""}>${labels[v]??v}</option>`).join("");
const draftValue=(key,defaultValue="")=>state.current_draft&&state.current_draft[key]!==undefined?state.current_draft[key]:defaultValue;
const render=()=>{
  const done=submittedIds(), next=assignments.find(row=>!done.has(row.assignment_id));
  download.disabled=done.size!==assignments.length;
  if(!state.consented_at){section.textContent="请先同意参与；评估内容将在本地加载。";start.disabled=true;return}
  start.disabled=false;
  if(state.current_assignment_id!==next.assignment_id||state.current_started_ms===null){
    if(state.current_assignment_id!==next.assignment_id){
      state.active_ms=0;state.current_draft={};
      state.current_started_at=new Date().toISOString();state.current_started_ms=Date.now();
    }
    state.current_assignment_id=next.assignment_id;save()
  }
  resume();
  const item=assignment(); if(!item)return;
  const heading=`<h2>${esc(item.kind)}</h2><p>${esc(item.source||"")}</p>`;
  let body="";
  if(item.kind==="absolute") body=`<p>${esc(item.target||"")}</p><div class="fields">${
    ["fidelity","naturalness","style_voice","consistency","context_handling","readability","format_integrity"]
    .map(k=>`<label>${k}<select data-field="${k}">${options(["1","2","3","4","5"],String(draftValue(k,"")))}</select></label>`).join("")
    }</div><label>备注<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  if(item.kind==="pairwise") body=`<p>A: ${esc(item.target_a||"")}</p><p>B: ${esc(item.target_b||"")}</p>
    <label>偏好<select data-field="preference">${options(["a_much_better","a_slightly_better","tie","b_slightly_better","b_much_better"],draftValue("preference"))}</select></label>
    <label>备注<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  if(item.kind==="polish") body=`<p>A: ${esc(item.target_a||"")}</p><p>B: ${esc(item.target_b||"")}</p>
    <label>结果<select data-field="outcome">${options(["clearly_improved","slightly_improved","no_material_change","fluent_but_semantic_damage","quality_declined"],draftValue("outcome"))}</select></label>
    <label>备注<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  if(item.kind==="context") body=`<p>${esc(item.current_output||"")}</p><p>问题：${esc(item.source||"")}</p>
    <label>判断<select data-field="judgment">${options(["correct","incorrect","uncertain"],draftValue("judgment"))}</select></label>
    <label>说明<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  if(item.kind==="postedit") body=`<p>${esc(item.target||"")}</p>
    <label>编辑结果<textarea data-field="edited_target">${esc(draftValue("edited_target"))}</textarea></label>
    <label>备注<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  if(item.kind==="mqm") body=`<p>${esc(item.target||"")}</p><div id="errors"></div>
    <button type="button" id="add-error">添加错误</button><label>备注<textarea data-field="note">${esc(draftValue("note"))}</textarea></label>`;
  section.innerHTML=heading+`<form id="response-form">${body}<button type="submit">提交</button></form>`;
  const form=document.getElementById("response-form");
  const saveDraft=()=>{
    const draft=Object.fromEntries([...form.querySelectorAll("[data-field]")].map(el=>[el.dataset.field,el.value]));
    if(item.kind==="mqm")draft.errors=[...form.querySelectorAll(".mqm-error")].map(row=>Object.fromEntries(
      [...row.querySelectorAll("[data-error]")].map(el=>[el.dataset.error,el.value])));
    state.current_draft=draft;save()
  };
  form.querySelectorAll("[data-field]").forEach(input=>input.addEventListener("input",saveDraft));
  if(item.kind==="mqm"){
    const errors=document.getElementById("errors"), addError=(draft={})=>{
      const row=document.createElement("fieldset");row.className="mqm-error";
      row.innerHTML=`<label>段落<select data-error="segment_id">${options(item.segment_ids||[],"")}</select></label>
        <label>严重程度<select data-error="severity">${options(["critical","major","minor"],"")}</select></label>
        <label>错误类型<select data-error="type">${options(["mistranslation","omission","addition","hallucination","terminology","named_entity","pronoun_reference","style_register","fluency","formatting"],"")}</select></label>
        <input data-error="source_quote" placeholder="原文摘录"><input data-error="target_quote" placeholder="译文摘录">
        <input data-error="note" placeholder="备注" required><button type="button" class="remove-error">删除</button>`;
      errors.appendChild(row);
      row.querySelectorAll("[data-error]").forEach(input=>{
        if(draft[input.dataset.error]!==undefined)input.value=draft[input.dataset.error]||"";
        input.addEventListener("input",saveDraft)
      });
      row.querySelector(".remove-error").onclick=()=>{row.remove();saveDraft()};
    };
    document.getElementById("add-error").onclick=()=>addError();
    draftValue("errors",[]).forEach(error=>addError(error));
  }
  form.onsubmit=event=>{
    event.preventDefault();
    const values=Object.fromEntries([...form.querySelectorAll("[data-field]")].map(el=>[el.dataset.field,el.value]));
    const absoluteFields=["fidelity","naturalness","style_voice","consistency","context_handling","readability","format_integrity"];
    const pairValues=["a_much_better","a_slightly_better","tie","b_slightly_better","b_much_better"];
    const polishValues=["clearly_improved","slightly_improved","no_material_change","fluent_but_semantic_damage","quality_declined"];
    const contextValues=["correct","incorrect","uncertain"];
    let valid=true;
    if(item.kind==="absolute")valid=absoluteFields.every(key=>/^[1-5]$/.test(values[key]||""));
    if(item.kind==="pairwise")valid=pairValues.includes(values.preference);
    if(item.kind==="polish")valid=polishValues.includes(values.outcome);
    if(item.kind==="context")valid=contextValues.includes(values.judgment)&&
      (values.judgment==="correct"||Boolean(values.note&&values.note.trim()));
    if(item.kind==="postedit")valid=Boolean(values.edited_target&&values.edited_target.trim());
    let errors=[];
    if(item.kind==="mqm"){
      errors=[...form.querySelectorAll(".mqm-error")].map(row=>Object.fromEntries(
        [...row.querySelectorAll("[data-error]")].map(el=>[el.dataset.error,el.value.trim()])));
      const seenErrors=new Set(), segmentIds=new Set(item.segment_ids||[]);
      valid=errors.every(error=>segmentIds.has(error.segment_id)&&
        ["critical","major","minor"].includes(error.severity)&&
        ["mistranslation","omission","addition","hallucination","terminology","named_entity","pronoun_reference","style_register","fluency","formatting"].includes(error.type)&&
        Boolean(error.note)&&!seenErrors.has([error.segment_id,error.severity,error.type,error.note].join("\u0000"))&&
        seenErrors.add([error.segment_id,error.severity,error.type,error.note].join("\u0000")));
    }
    if(!valid){status.textContent="请完整填写后再提交。";return}
    const endMs=Date.now();flush(endMs);
    const startedMs=state.current_started_ms===null?endMs:state.current_started_ms;
    const wallMs=Math.max(0,endMs-startedMs);
    if(state.active_ms>wallMs)state.active_ms=wallMs;
    const response={schema_version:1,assignment_id:item.assignment_id,rater_id:RATER,pack_hash:PACK,
      started_at:state.current_started_at,submitted_at:new Date(endMs).toISOString(),active_ms:state.active_ms,kind:item.kind};
    const addNote=()=>{if(values.note&&values.note.trim())response.note=values.note};
    if(item.kind==="mqm")response.errors=errors.map(error=>{
      const output={segment_id:error.segment_id,severity:error.severity,type:error.type,note:error.note};
      if(error.source_quote)output.source_quote=error.source_quote;
      if(error.target_quote)output.target_quote=error.target_quote;
      return output;
    }),addNote();
    else if(item.kind==="absolute"){
      absoluteFields.forEach(key=>{response[key]=Number(values[key])});
      addNote();
    } else {
      Object.assign(response,values);addNote();if(!values.note||!values.note.trim())delete response.note;
    }
    state.responses=state.responses.concat([Object.freeze(response)]);state.current_assignment_id=null;
    state.current_started_at=null;state.current_started_ms=null;state.active_ms=0;
    state.current_draft=null;visibleStart=0;save();render();
  };
};
consent.addEventListener("change",()=>{start.disabled=!consent.checked});
start.addEventListener("click",()=>{if(!state.consented_at){state.consented_at=new Date().toISOString();save()}resume();render()});
download.addEventListener("click",()=>{if(download.disabled)return;flush(Date.now());const blob=new Blob([JSON.stringify({
  schema_version:1,pack_sha256:PACK,rater_id:RATER,consented_at:state.consented_at,responses:state.responses
})],{type:"application/json"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download=`responses-${RATER}.json`;a.click()});
document.addEventListener("visibilitychange",()=>{if(document.hidden)flush(Date.now());else resume()});
load();fetch("./assignments.json").then(response=>response.ok?response.json():Promise.reject()).then(value=>{
  if(!value||value.pack_sha256!==PACK||value.rater_id!==RATER||!Array.isArray(value.assignments))throw new Error();
  assignments=value.assignments;load();
  const ids=new Set(assignments.map(row=>row.assignment_id));
  if(!state.responses.every(row=>ids.has(row.assignment_id)&&row.rater_id===RATER&&row.pack_hash===PACK&&
    assignments.find(item=>item.assignment_id===row.assignment_id)?.kind===row.kind))state=emptyState();
  consent.checked=Boolean(state.consented_at);start.disabled=!state.consented_at;render();
}).catch(()=>{section.textContent="无法加载本地 assignments.json";status.textContent="加载失败"});
})();"""
_CSS = (
    "body{font-family:system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem}"
    "label{display:block;margin:.6rem 0}textarea{display:block;min-height:4rem;width:100%}"
    "select,input,button{font:inherit;margin:.2rem}.fields{display:grid;grid-template-columns:repeat(2,1fr)}"
    "#assignment{white-space:normal}button:disabled{opacity:.5}.mqm-error{margin:.5rem 0}"
)


def _render_index(
    semantic_hash: str,
    rater: str,
    protocol: Any,
) -> str:
    values = (
        ("__PACK_HASH__", semantic_hash),
        ("__RATER__", rater),
        ("__ELIGIBILITY__", html.escape(protocol.eligibility_text)),
        ("__CONSENT__", html.escape(protocol.consent_text)),
        ("__COMPENSATION__", html.escape(protocol.compensation_text)),
        ("__RETENTION__", html.escape(protocol.retention_text)),
    )
    rendered = _INDEX
    for marker, value in values:
        rendered = rendered.replace(marker, value)
    return rendered


def _validate_run_lineage(run: Path, spec: EvaluationSpec) -> None:
    run_path, state_path = run / "run.json", run / "run_state.json"
    run_json = _read_json(run_path)
    state = _read_json(state_path)
    if not isinstance(run_json, dict) or not isinstance(state, dict):
        raise EvaluationError("run lineage manifests must be objects")
    if state.get("status") != "completed":
        raise EvaluationError("run is not completed")
    mode = run_json.get("run_mode")
    if mode in {"attribution", "canary"}:
        expected_keys = {
            "schema_version",
            "run_mode",
            "prompt_version",
            "benchmark_id",
            "spec_sha256",
            "corpus_sha256",
            "preparation_sha256",
            "canary_sample_id",
        }
        if set(run_json) != expected_keys:
            raise EvaluationError("Phase 5 run manifest fields invalid")
        if (
            not isinstance(run_json["schema_version"], int)
            or isinstance(run_json["schema_version"], bool)
            or run_json["schema_version"] != 1
            or not isinstance(run_json["prompt_version"], str)
        ):
            raise EvaluationError("Phase 5 run manifest values invalid")
        if run_json["benchmark_id"] != spec.benchmark_id:
            raise EvaluationError("run benchmark mismatch")
        if run_json["corpus_sha256"] != spec.corpus_sha256:
            raise EvaluationError("run corpus mismatch")
        if (
            not isinstance(run_json["spec_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", run_json["spec_sha256"])
            or not isinstance(run_json["preparation_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", run_json["preparation_sha256"])
        ):
            raise EvaluationError("run manifest hashes invalid")
        if mode == "attribution" and run_json["canary_sample_id"] is not None:
            raise EvaluationError("attribution run cannot declare canary sample")
        if mode == "canary" and (
            not isinstance(run_json["canary_sample_id"], str) or not run_json["canary_sample_id"]
        ):
            raise EvaluationError("canary sample is invalid")
    elif mode == "full":
        expected_keys = {
            "schema_version",
            "run_mode",
            "corpus_sha256",
            "preparation_sha256",
            "benchmark_id",
            "spec_sha256",
            "replicates",
        }
        if set(run_json) != expected_keys:
            raise EvaluationError("Phase 6 run manifest fields invalid")
        if (
            not isinstance(run_json["schema_version"], int)
            or isinstance(run_json["schema_version"], bool)
            or run_json["schema_version"] != 1
            or run_json["run_mode"] != "full"
        ):
            raise EvaluationError("Phase 6 run manifest values invalid")
        if (
            run_json["benchmark_id"] != spec.benchmark_id
            or run_json["corpus_sha256"] != spec.corpus_sha256
            or not isinstance(run_json["preparation_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", run_json["preparation_sha256"])
            or not isinstance(run_json["spec_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", run_json["spec_sha256"])
            or not isinstance(run_json["replicates"], int)
            or isinstance(run_json["replicates"], bool)
            or run_json["replicates"] < 1
        ):
            raise EvaluationError("Phase 6 run manifest values invalid")
    else:
        raise EvaluationError("unknown run mode")
    if _hash(run_json) != spec.run_hash:
        raise EvaluationError("run hash mismatch")


def _write_raters(
    pack: Path,
    spec: EvaluationSpec,
    assignments: list[tuple[str, dict[str, Any]]],
    semantic_hash: str,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rater, item in assignments:
        grouped[rater].append(item)
    hashes: dict[str, str] = {}
    for rater in sorted(spec.raters):
        root = pack / "raters" / rater
        root.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": 1,
            "pack_sha256": semantic_hash,
            "rater_id": rater,
            "study_protocol": spec.study_protocol.model_dump(mode="python"),
            "assignments": grouped[rater],
        }
        _atomic_json(root / "assignments.json", envelope, escape_lt=True)
        _atomic_json(
            root / "response-schema.json", {"schema_version": 1, "kinds": list(_KIND_ORDER)}
        )
        (root / "app.js").write_text(_JS, encoding="utf-8", newline="\n")
        (root / "style.css").write_text(_CSS, encoding="utf-8", newline="\n")
        (root / "index.html").write_text(
            _render_index(semantic_hash, rater, spec.study_protocol),
            encoding="utf-8",
            newline="\n",
        )
        for child in root.iterdir():
            hashes[str(child.relative_to(pack))] = sha256_bytes(child.read_bytes())
    return hashes


def generate_pack(
    corpus_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    evaluation_spec: str | os.PathLike[str] | dict[str, Any],
    out: str | os.PathLike[str],
) -> Path:
    pack = Path(out).expanduser()
    if pack.exists():
        raise EvaluationError("pack output already exists")
    spec = _spec(evaluation_spec)
    corpus = Path(corpus_dir).expanduser().resolve()
    run = Path(run_dir).expanduser().resolve()
    _validate_run_lineage(run, spec)
    rows, challenge, candidates = _load_inputs(corpus, run, spec)
    records = _records(run, candidates, set(spec.candidate_ids), rows)
    units = build_units(rows, spec.corpus_sha256)
    base, mapping = _make_assignments(spec, units, records, challenge)
    assignments = _allocate(base, spec, mapping)
    _sanitize_assignment_ids(assignments, mapping, spec.seed, set(spec.candidate_ids))
    if _leak([item for _, item in assignments], set(spec.candidate_ids)):
        raise EvaluationError("candidate metadata leaked into assignments")
    pack.mkdir(parents=True)
    task_counts = {kind: sum(1 for item in base if item["kind"] == kind) for kind in _KIND_ORDER}
    units_by_id = {unit["unit_id"]: unit for unit in units}
    task_words = {
        kind: sum(
            units_by_id[unit_id]["word_count"]
            for unit_id in {item["unit_id"] for item in base if item["kind"] == kind}
            if unit_id in units_by_id
        )
        for kind in _KIND_ORDER
    }
    book_balance: dict[str, Any] = {}
    for kind in _KIND_ORDER:
        words_by_book: dict[str, int] = defaultdict(int)
        for item in base:
            if item["kind"] != kind:
                continue
            unit = units_by_id.get(item["unit_id"])
            if unit is not None:
                words_by_book[str(unit["book_id"])] += unit["word_count"]
        total = sum(words_by_book.values())
        cap = total * 0.25
        offending = {book: words for book, words in words_by_book.items() if words > cap}
        book_balance[kind] = {
            "words_by_book": dict(sorted(words_by_book.items())),
            "book_cap_relaxed": bool(offending),
            "offending_books": offending,
        }
    allocation_counts: dict[str, dict[str, int]] = {
        rater: dict.fromkeys(_KIND_ORDER, 0) for rater in spec.raters
    }
    calibration_counts: dict[str, int] = dict.fromkeys(spec.raters, 0)
    for rater, item in assignments:
        if item["calibration"]:
            calibration_counts[rater] += 1
        else:
            allocation_counts[rater][item["kind"]] += 1
    allocation = {
        "base_counts": allocation_counts,
        "calibration_counts": calibration_counts,
        "calibration_units_per_rater": 30,
    }
    postedit_scope: dict[tuple[str, str], dict[str, Any]] = {}
    for rater, item in assignments:
        if item["calibration"] or item["kind"] != "postedit":
            continue
        metadata = mapping[item["assignment_id"]]
        if metadata.get("duplicate_of") is not None:
            continue
        candidate = metadata["positions"][0]["candidate_id"]
        scope = (candidate, metadata["surface"])
        row = postedit_scope.setdefault(scope, {"outputs": 0, "raters": set()})
        row["outputs"] += 1
        row["raters"].add(rater)
    postedit_diversity = {
        f"{candidate}:{surface}": {
            "outputs": value["outputs"],
            "raters": sorted(value["raters"]),
            "unavoidable_one_output": value["outputs"] < 2,
        }
        for (candidate, surface), value in sorted(postedit_scope.items())
    }
    allocation["postedit_diversity"] = postedit_diversity
    if any(
        value["outputs"] >= 2 and len(value["raters"]) < 2 for value in postedit_diversity.values()
    ):
        raise EvaluationError("postedit outputs require two distinct raters")
    semantic = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "run_hash": spec.run_hash,
        "corpus_sha256": spec.corpus_sha256,
        "allocation": allocation,
        "evaluation_spec": spec.model_dump(mode="python"),
        "tasks": task_counts,
        "task_source_words": task_words,
        "book_balance": book_balance,
        "assignment_semantic_sha256": _hash([item for _, item in assignments]),
        "mapping_semantic_sha256": _hash(mapping),
        "base_assignment_count": len(base),
        "assignment_count": len(assignments),
        "raters": sorted(spec.raters),
    }
    semantic_hash = _hash(semantic)
    secret = {"schema_version": 1, "pack_semantic_hash": semantic_hash, "assignments": mapping}
    _atomic_json(pack / "secret_mapping.json", secret, 0o600)
    manifest = {
        **semantic,
        "pack_semantic_sha256": semantic_hash,
        "rater_files": {},
        "secret_mapping_sha256": sha256_bytes((pack / "secret_mapping.json").read_bytes()),
    }
    manifest["rater_files"] = _write_raters(pack, spec, assignments, semantic_hash)
    manifest["pack_sha256"] = _hash(manifest)
    _atomic_json(pack / "pack.json", manifest)
    return pack


def validate_pack(pack_dir: str | os.PathLike[str]) -> dict[str, Any]:
    pack = Path(pack_dir).expanduser().resolve()
    manifest = _read_json(pack / "pack.json")
    try:
        manifest_spec = _spec(manifest["evaluation_spec"])
    except (KeyError, EvaluationError) as error:
        raise EvaluationError("evaluation spec manifest invalid") from error
    semantic_keys = (
        "schema_version",
        "benchmark_id",
        "run_hash",
        "corpus_sha256",
        "evaluation_spec",
        "tasks",
        "task_source_words",
        "book_balance",
        "allocation",
        "assignment_semantic_sha256",
        "mapping_semantic_sha256",
        "base_assignment_count",
        "assignment_count",
        "raters",
    )
    semantic = {key: manifest.get(key) for key in semantic_keys}
    semantic_hash = _hash(semantic)
    if manifest.get("pack_semantic_sha256") != semantic_hash:
        raise EvaluationError("pack semantic hash mismatch")
    if manifest.get("pack_sha256") != _hash(manifest, without="pack_sha256"):
        raise EvaluationError("pack hash mismatch")
    secret_path = pack / "secret_mapping.json"
    secret = _read_json(secret_path)
    if secret.get("pack_semantic_hash") != semantic_hash:
        raise EvaluationError("secret semantic hash mismatch")
    owners: dict[str, str] = {}
    if manifest.get("secret_mapping_sha256") != sha256_bytes(secret_path.read_bytes()):
        raise EvaluationError("secret mapping hash mismatch")
    identifiers = _provenance_identifiers(manifest, secret)
    mapping_assignments = secret.get("assignments")
    if not isinstance(mapping_assignments, dict):
        raise EvaluationError("secret mapping assignments invalid")
    seen: set[str] = set()
    seen_items: list[dict[str, Any]] = []
    expected_files = manifest.get("rater_files", {})
    if not isinstance(expected_files, dict):
        raise EvaluationError("rater file manifest invalid")
    for rater in manifest.get("raters", []):
        root = pack / "raters" / rater
        envelope = _read_json(root / "assignments.json")
        if envelope.get("pack_sha256") != semantic_hash:
            raise EvaluationError("rater semantic hash mismatch")
        items = envelope.get("assignments")
        if not isinstance(items, list):
            raise EvaluationError("rater assignments invalid")
        for item in items:
            aid = item.get("assignment_id") if isinstance(item, dict) else None
            if aid in seen or aid not in mapping_assignments:
                raise EvaluationError("assignment mapping mismatch")
            seen.add(aid)
            owners[aid] = rater
            seen_items.append(item)
            if _leak(item, identifiers):
                raise EvaluationError("candidate leakage in rater payload")
        expected_static = {
            "index.html": _render_index(
                semantic_hash,
                rater,
                manifest_spec.study_protocol,
            ).encode("utf-8"),
            "app.js": _JS.encode("utf-8"),
            "style.css": _CSS.encode("utf-8"),
        }
        for name, expected_bytes in expected_static.items():
            path = root / name
            if not path.is_file() or path.read_bytes() != expected_bytes:
                raise EvaluationError("static rater asset mismatch")
        for name in (
            "index.html",
            "app.js",
            "style.css",
            "assignments.json",
            "response-schema.json",
        ):
            path = root / name
            if not path.is_file():
                raise EvaluationError("missing rater asset")
            recorded = expected_files.get(str(path.relative_to(pack)))
            if recorded is None or recorded != sha256_bytes(path.read_bytes()):
                raise EvaluationError("rater asset hash mismatch")
            # Structured assignment assets are checked by metadata key/value;
            # human content fields are intentionally excluded by _leak.
            if path.suffix == ".json" and _leak(_read_json(path), identifiers):
                raise EvaluationError("candidate leakage in rater asset")
    if len(seen_items) != manifest.get("assignment_count"):
        raise EvaluationError("assignment count mismatch")
    if _hash(mapping_assignments) != manifest.get("mapping_semantic_sha256"):
        raise EvaluationError("mapping semantic hash mismatch")
    if set(mapping_assignments) != seen:
        raise EvaluationError("secret mapping assignment set mismatch")
    calibration_counts: dict[str, int] = defaultdict(int)
    for item in seen_items:
        aid = item["assignment_id"]
        metadata = mapping_assignments[aid]
        if (
            metadata.get("kind") != item.get("kind")
            or metadata.get("unit_id") != item.get("unit_id")
            or bool(metadata.get("calibration")) != bool(item.get("calibration"))
            or not isinstance(metadata.get("positions"), list)
            or not metadata["positions"]
        ):
            raise EvaluationError("assignment provenance mismatch")
        if item.get("calibration"):
            calibration_counts[owners[aid]] += 1
        duplicate = metadata.get("duplicate_of")
        if duplicate is not None and (
            duplicate not in mapping_assignments or owners.get(duplicate) != owners[aid]
        ):
            raise EvaluationError("duplicate mapping mismatch")
    if any(calibration_counts[rater] != 30 for rater in manifest.get("raters", [])):
        raise EvaluationError("calibration allocation mismatch")
    if _hash(seen_items) != manifest.get("assignment_semantic_sha256"):
        raise EvaluationError("assignment semantic hash mismatch")
    _pair_adjudication([], mapping_assignments)
    return {
        "pack_sha256": manifest["pack_sha256"],
        "pack_semantic_sha256": semantic_hash,
        "assignment_count": len(seen),
        "rater_count": len(manifest.get("raters", [])),
    }


def _utc(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise EvaluationError("timestamp must be UTC Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise EvaluationError("invalid timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvaluationError("timestamp must be UTC")
    return parsed


def _responses(
    path: Path,
    pack_hash: str,
    rater: str,
    assignments: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    envelope = _read_json(path)
    required = {"schema_version", "pack_sha256", "rater_id", "consented_at", "responses"}
    if set(envelope) != required:
        raise EvaluationError("response envelope fields invalid")
    if (
        envelope.get("schema_version") != 1
        or envelope.get("pack_sha256") != pack_hash
        or envelope.get("rater_id") != rater
    ):
        raise EvaluationError("response identity mismatch")
    _utc(envelope.get("consented_at"))
    if not isinstance(envelope.get("responses"), list):
        raise EvaluationError("responses must be a list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for raw in envelope["responses"]:
        if not isinstance(raw, dict):
            raise EvaluationError("response must be an object")
        aid = raw.get("assignment_id")
        if aid in seen or aid not in assignments:
            raise EvaluationError("response assignment set invalid")
        seen.add(aid)
        expected = assignments[aid]
        if (
            raw.get("rater_id") != rater
            or raw.get("pack_hash") != pack_hash
            or raw.get("kind") != expected["kind"]
        ):
            raise EvaluationError("response identity mismatch")
        try:
            parsed = _RESPONSE_MODELS[expected["kind"]].model_validate(raw)
        except Exception as error:
            raise EvaluationError(f"invalid {expected['kind']} response") from error
        start, end = _utc(parsed.started_at), _utc(parsed.submitted_at)
        if end < start or parsed.active_ms > (end - start).total_seconds() * 1000:
            raise EvaluationError("invalid active time")
        if expected["kind"] == "mqm":
            errors = parsed.errors
            keys = [(error.segment_id, error.severity, error.type, error.note) for error in errors]
            valid_segments = set(expected.get("segment_ids", []))
            if len(keys) != len(set(keys)) or any(
                error.segment_id not in valid_segments for error in errors
            ):
                raise EvaluationError("invalid MQM segment")
        rows.append(parsed.model_dump(mode="python"))
        counts[expected["kind"]] += 1
    if seen != set(assignments):
        raise EvaluationError("incomplete response file")
    return rows, dict(counts)


_PREFERENCE_REVERSE = {
    "a_much_better": "b_much_better",
    "a_slightly_better": "b_slightly_better",
    "tie": "tie",
    "b_slightly_better": "a_slightly_better",
    "b_much_better": "a_much_better",
}


def _pair_adjudication(
    rows: list[dict[str, Any]],
    mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    relations: dict[str, dict[str, Any]] = {}
    for aid, metadata in mapping.items():
        if metadata.get("kind") != "pairwise":
            continue
        positions = metadata.get("positions")
        if not isinstance(positions, list) or len(positions) != 2:
            raise EvaluationError("pair mapping positions invalid")
        if metadata.get("calibration") or metadata.get("duplicate_of") is not None:
            continue
        surface, unit_id = metadata.get("surface"), metadata.get("unit_id")
        if not isinstance(surface, str) or not isinstance(unit_id, str):
            raise EvaluationError("pair mapping relation fields invalid")
        canonical_positions: list[list[Any]] = []
        candidates: set[str] = set()
        for position in positions:
            if not isinstance(position, dict):
                raise EvaluationError("pair mapping position invalid")
            candidate, replicate = position.get("candidate_id"), position.get("replicate")
            if (
                not isinstance(candidate, str)
                or candidate in candidates
                or not isinstance(replicate, int)
                or isinstance(replicate, bool)
                or replicate < 1
                or not isinstance(position.get("segment_provenance"), list)
            ):
                raise EvaluationError("pair mapping position provenance invalid")
            candidates.add(candidate)
            canonical_positions.append([candidate, replicate])
        canonical_positions.sort(key=lambda value: value[0])
        relation_value = {
            "surface": surface,
            "unit_id": unit_id,
            "positions": canonical_positions,
        }
        relation_key = _hash(relation_value)
        relation = relations.setdefault(
            relation_key,
            {"key": relation_value, "assignment_ids": []},
        )
        relation["assignment_ids"].append(aid)
        if len(relation["assignment_ids"]) > 2:
            raise EvaluationError("pair relation has more than two primary presentations")
    for relation in relations.values():
        if len(set(relation["assignment_ids"])) != 2:
            raise EvaluationError("pair relation must have two primary presentations")
    by_assignment = {row["assignment_id"]: row for row in rows if row.get("kind") == "pairwise"}
    output: list[dict[str, Any]] = []
    for relation_key, relation in sorted(relations.items()):
        responses = [
            by_assignment[aid] for aid in relation["assignment_ids"] if aid in by_assignment
        ]
        if len(responses) < 2:
            continue
        if len(responses) != 2:
            raise EvaluationError("pair relation response cardinality invalid")
        normalized: list[str] = []
        for response in responses:
            metadata = mapping[response["assignment_id"]]
            first = metadata["positions"][0]["candidate_id"]
            canonical_first = relation["key"]["positions"][0][0]
            preference = response["preference"]
            if first != canonical_first:
                preference = _PREFERENCE_REVERSE[preference]
            normalized.append(preference)
        if normalized[0] == normalized[1]:
            continue
        output.append(
            {
                "relation_key": relation_key,
                "surface": relation["key"]["surface"],
                "unit_id": relation["key"]["unit_id"],
                "positions": relation["key"]["positions"],
                "responses": [
                    {
                        "assignment_id": response["assignment_id"],
                        "rater_id": response["rater_id"],
                        "preference": preference,
                    }
                    for response, preference in zip(responses, normalized, strict=True)
                ],
            }
        )
    return output


def import_responses(
    pack_dir: str | os.PathLike[str],
    responses_dir: str | os.PathLike[str],
    out: str | os.PathLike[str],
) -> Path:
    pack = Path(pack_dir).expanduser().resolve()
    response_root = Path(responses_dir).expanduser().resolve()
    evaluation = Path(out).expanduser()
    validate_pack(pack)
    manifest = _read_json(pack / "pack.json")
    pack_hash = manifest["pack_semantic_sha256"]
    expected = set(manifest["raters"])
    response_paths = sorted(response_root.glob("*.json"))
    supplied: dict[str, Path] = {}
    for path in response_paths:
        name = path.name
        if not name.startswith("responses-") or not name.endswith(".json"):
            raise EvaluationError(f"unexpected response file: {name}")
        rater = name[len("responses-") : -len(".json")]
        if not rater or rater not in expected or rater in supplied:
            raise EvaluationError(f"unexpected response file: {name}")
        supplied[rater] = path
    if not supplied:
        raise EvaluationError("no response files")
    if evaluation.exists() and not evaluation.is_dir():
        raise EvaluationError("evaluation output is not directory")
    assignment_by_rater: dict[str, dict[str, dict[str, Any]]] = {}
    calibration_ids: set[str] = set()
    for rater in expected:
        envelope = _read_json(pack / "raters" / rater / "assignments.json")
        assignment_by_rater[rater] = {
            item["assignment_id"]: item for item in envelope["assignments"]
        }
        calibration_ids.update(
            item["assignment_id"]
            for item in envelope["assignments"]
            if item.get("calibration") is True
        )
    validated: dict[str, tuple[list[dict[str, Any]], dict[str, int], bytes]] = {}
    for rater, path in supplied.items():
        rows, counts = _responses(path, pack_hash, rater, assignment_by_rater[rater])
        validated[rater] = (rows, counts, path.read_bytes())
    if not evaluation.exists():
        evaluation.mkdir(parents=True)
        _atomic_json(
            evaluation / "evaluation.json",
            {
                "schema_version": 1,
                "pack_sha256": manifest["pack_sha256"],
                "pack_semantic_sha256": pack_hash,
                "spec": manifest["evaluation_spec"],
                "secret_mapping_sha256": manifest["secret_mapping_sha256"],
                "expected_raters": sorted(expected),
            },
        )
        _atomic_json(
            evaluation / "import_state.json",
            {
                "schema_version": 1,
                "pack_sha256": manifest["pack_sha256"],
                "pack_semantic_sha256": pack_hash,
                "status": "incomplete",
                "raters": {},
            },
        )
    state = _read_json(evaluation / "import_state.json")
    immutable = _read_json(evaluation / "evaluation.json")
    if (
        immutable.get("pack_sha256") != manifest["pack_sha256"]
        or immutable.get("pack_semantic_sha256") != pack_hash
        or state.get("pack_sha256") != manifest["pack_sha256"]
        or state.get("pack_semantic_sha256") != pack_hash
    ):
        raise EvaluationError("immutable evaluation hash mismatch")
    prior = state.get("raters", {})
    if not isinstance(prior, dict):
        raise EvaluationError("import state raters invalid")
    for rater, entry in prior.items():
        source_path = evaluation / "source_responses" / f"{rater}.json"
        if not source_path.is_file() or sha256_bytes(source_path.read_bytes()) != entry.get(
            "response_sha256"
        ):
            raise EvaluationError("stored response source tampered")
    for rater, (_, _, raw) in validated.items():
        digest = sha256_bytes(raw)
        if rater in prior and prior[rater]["response_sha256"] != digest:
            raise EvaluationError("conflicting duplicate response")
    complete_path = evaluation / "evaluation_complete.json"
    if complete_path.exists() and state.get("status") == "complete":
        completion = _read_json(complete_path)
        if completion.get("evaluation_sha256") != sha256_bytes(
            (evaluation / "evaluation.json").read_bytes()
        ) or completion.get("import_state_sha256") != sha256_bytes(
            (evaluation / "import_state.json").read_bytes()
        ):
            raise EvaluationError("completion manifest hash mismatch")
        for name, digest in completion.get("derived_files", {}).items():
            path = evaluation / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                raise EvaluationError("completed derived artifact hash mismatch")
        if all(
            rater in prior and prior[rater]["response_sha256"] == sha256_bytes(raw)
            for rater, (_, _, raw) in validated.items()
        ):
            return evaluation
    source = evaluation / "source_responses"
    source.mkdir(parents=True, exist_ok=True)
    for rater, (_, _, raw) in validated.items():
        target = source / f"{rater}.json"
        if not target.exists():
            target.write_bytes(raw)
    all_rows: list[dict[str, Any]] = []
    new_raters = dict(prior)
    for rater in sorted(expected):
        path = source / f"{rater}.json"
        if not path.exists():
            continue
        rows, counts = _responses(path, pack_hash, rater, assignment_by_rater[rater])
        new_raters[rater] = {
            "response_sha256": sha256_bytes(path.read_bytes()),
            "row_counts": counts,
        }
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: (row["assignment_id"], row["rater_id"]))
    (evaluation / "responses.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in all_rows),
        encoding="utf-8",
        newline="\n",
    )
    mqm = [row for row in all_rows if row["kind"] == "mqm"]
    edits = [row for row in all_rows if row["kind"] == "postedit"]
    (evaluation / "mqm_errors.jsonl").write_text(
        "".join(
            canonical_json(
                {**error, "assignment_id": row["assignment_id"], "rater_id": row["rater_id"]}
            )
            + "\n"
            for row in mqm
            for error in row["errors"]
        ),
        encoding="utf-8",
        newline="\n",
    )
    (evaluation / "post_edits.jsonl").write_text(
        "".join(canonical_json(row) + "\n" for row in edits),
        encoding="utf-8",
        newline="\n",
    )
    mapping_for_adjudication: dict[str, dict[str, Any]] = {}
    for assignments_for_rater in assignment_by_rater.values():
        mapping_for_adjudication.update(assignments_for_rater)
    _atomic_json(
        evaluation / "adjudication_needed.json",
        _pair_adjudication(
            all_rows,
            {
                aid: metadata
                for aid, metadata in _read_json(pack / "secret_mapping.json")
                .get("assignments", {})
                .items()
                if aid in mapping_for_adjudication
            },
        ),
    )
    status = "complete" if set(new_raters) == expected else "incomplete"
    _atomic_json(
        evaluation / "import_state.json",
        {
            "schema_version": 1,
            "pack_sha256": manifest["pack_sha256"],
            "pack_semantic_sha256": pack_hash,
            "status": status,
            "raters": new_raters,
        },
    )
    if status == "complete":
        derived_names = (
            "responses.jsonl",
            "mqm_errors.jsonl",
            "post_edits.jsonl",
            "adjudication_needed.json",
        )
        derived_files = {
            name: sha256_bytes((evaluation / name).read_bytes()) for name in derived_names
        }
        copied_files = {
            f"source_responses/{rater}.json": sha256_bytes((source / f"{rater}.json").read_bytes())
            for rater in sorted(expected)
        }
        _atomic_json(
            evaluation / "evaluation_complete.json",
            {
                "schema_version": 1,
                "pack_sha256": manifest["pack_sha256"],
                "pack_semantic_sha256": pack_hash,
                "evaluation_sha256": sha256_bytes((evaluation / "evaluation.json").read_bytes()),
                "import_state_sha256": sha256_bytes(
                    (evaluation / "import_state.json").read_bytes()
                ),
                "derived_files": {**derived_files, **copied_files},
            },
        )
    return evaluation


pack = generate_pack
generate = generate_pack
validate = validate_pack
import_pack = import_responses

__all__ = [
    "EvaluationError",
    "build_units",
    "generate",
    "generate_pack",
    "import_pack",
    "import_responses",
    "pack",
    "validate",
    "validate_pack",
]
