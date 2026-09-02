"""Deterministic review sampling, blinding, and request construction."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from itertools import combinations
from typing import Any

from trans_novel.benchmark.artifacts import canonical_json, sha256_bytes

ERROR_TYPES = (
    "omission",
    "addition",
    "mistranslation",
    "named_entity",
    "terminology",
    "pronoun_reference",
    "context",
    "style_register",
    "fluency",
)
SEVERITIES = ("critical", "major", "minor")
SEVERITY_WEIGHTS = {"critical": 25, "major": 8, "minor": 2}


def semantic_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def rank_key(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).hexdigest()


def category_for_row(row: dict[str, Any]) -> str:
    if row.get("lint_findings") or row.get("deterministic_findings"):
        return "risk"
    source = str(row["source"])
    if any(mark in source for mark in ('"', "“", "”", "‘", "’")):
        return "dialogue"
    if len(source) >= 500:
        return "long_sentence"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source):
        return "terminology"
    return "narrative"


def select_book(rows: list[dict[str, Any]], *, seed: int, limit: int) -> list[dict[str, Any]]:
    quotas = {
        "risk": max(1, limit // 4),
        "dialogue": max(1, limit // 4),
        "terminology": max(1, limit // 6),
        "long_sentence": max(1, limit // 6),
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[category_for_row(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category, quota in quotas.items():
        ordered = sorted(
            by_category.get(category, []),
            key=lambda row: rank_key(seed, f"sample:{category}", row["segment_id"]),
        )
        for row in ordered[:quota]:
            selected.append(row)
            selected_ids.add(row["segment_id"])
    remaining = sorted(
        [row for row in rows if row["segment_id"] not in selected_ids],
        key=lambda row: rank_key(seed, "sample:fill", row["segment_id"]),
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return sorted(selected[:limit], key=lambda row: (row["chapter_index"], row["segment_index"]))


def review_prompt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": "Bilingual literary translation reviewer",
        "instructions": [
            "Compare each blinded Chinese translation against the English source.",
            "Fidelity outranks fluency. Do not reward style when meaning is lost or added.",
            "Report only concrete errors supported by exact source and target quotes.",
            "Use critical only for a passage-level meaning failure, major for material meaning or consistency errors, and minor for local wording or fluency defects.",
            "Return one review for every unit and no prose outside the required JSON object.",
        ],
        "error_types": list(ERROR_TYPES),
        "severities": list(SEVERITIES),
        "output_contract": {
            "schema_version": 1,
            "review_sha256": "<from shard>",
            "shard_id": "<from shard>",
            "reviews": [
                {
                    "unit_id": "<from unit>",
                    "winner": "A|B|tie",
                    "verdict_reason": "specific comparative reason",
                    "findings": [
                        {
                            "finding_id": "stable id unique within the unit",
                            "side": "A|B",
                            "type": "one configured error type",
                            "severity": "critical|major|minor",
                            "source_quote": "exact source substring",
                            "target_quote": "exact substring from the selected side",
                            "reason": "specific explanation",
                        }
                    ],
                }
            ],
        },
    }


def _group_candidate_rows(
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    candidate_ids = sorted({row["candidate_id"] for row in candidate_rows})
    by_scope: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for artifact in candidate_rows:
        candidate = artifact["candidate_id"]
        book_id = artifact["book_id"]
        replicate = artifact["replicate"]
        for row in artifact["rows"]:
            key = (book_id, replicate, str(row.get("segment_id")))
            by_scope[key][candidate] = row
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (book_id, replicate, source_segment_id), values in sorted(by_scope.items()):
        if set(values) != set(candidate_ids):
            raise ValueError("candidate segment sets do not match")
        baseline = values[candidate_ids[0]]
        if any(row.get("source") != baseline.get("source") for row in values.values()):
            raise ValueError("candidate source text mismatch")
        category_row = {
            **baseline,
            "lint_findings": [
                finding for row in values.values() for finding in row.get("lint_findings", [])
            ],
            "deterministic_findings": [
                finding
                for row in values.values()
                for finding in row.get("deterministic_findings", [])
            ],
        }
        by_book[book_id].append(
            {
                "book_id": book_id,
                "replicate": replicate,
                "segment_id": source_segment_id,
                "chapter_index": baseline["chapter_index"],
                "chapter_title": baseline["chapter_title"],
                "segment_index": baseline["segment_index"],
                "category": category_for_row(category_row),
                "source": baseline["source"],
                "source_sha256": baseline.get("source_sha256"),
                "candidate_targets": {
                    candidate: values[candidate]["target"] for candidate in candidate_ids
                },
            }
        )
    return candidate_ids, by_book


def _build_blinded_units(
    spec: Any,
    candidate_ids: list[str],
    sampled: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units: list[dict[str, Any]] = []
    secret: dict[str, Any] = {}
    for row in sampled:
        for first_candidate, second_candidate in combinations(candidate_ids, 2):
            pair = [first_candidate, second_candidate]
            orientation_key = f"{row['segment_id']}:{first_candidate}:{second_candidate}"
            if int(rank_key(spec.seed, "orientation", orientation_key), 16) % 2:
                pair.reverse()
            unit_id = semantic_hash(
                {
                    "book_id": row["book_id"],
                    "replicate": row["replicate"],
                    "segment_id": row["segment_id"],
                    "source_sha256": row["source_sha256"],
                    "candidate_pair": sorted(pair),
                }
            )
            units.append(
                {
                    "unit_id": unit_id,
                    "book_id": row["book_id"],
                    "replicate": row["replicate"],
                    "segment_id": row["segment_id"],
                    "chapter_index": row["chapter_index"],
                    "chapter_title": row["chapter_title"],
                    "segment_index": row["segment_index"],
                    "category": row["category"],
                    "source": row["source"],
                    "targets": {
                        "A": row["candidate_targets"][pair[0]],
                        "B": row["candidate_targets"][pair[1]],
                    },
                }
            )
            secret[unit_id] = {"A": pair[0], "B": pair[1]}
    if not units:
        raise ValueError("review sampling produced no units")
    return units, secret


def build_review_plan(spec: Any, candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_ids, by_book = _group_candidate_rows(candidate_rows)
    sampled = [
        row
        for book_id in sorted(by_book)
        for row in select_book(by_book[book_id], seed=spec.seed, limit=spec.segments_per_book)
    ]
    units, secret = _build_blinded_units(spec, candidate_ids, sampled)
    semantic = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "run_sha256": spec.run_sha256,
        "spec": spec.model_dump(mode="python"),
        "candidate_count": len(candidate_ids),
        "book_ids": sorted(by_book),
        "unit_count": len(units),
        "units_sha256": semantic_hash(units),
        "prompt_sha256": semantic_hash(review_prompt()),
    }
    review_sha = semantic_hash(semantic)
    shard_count = min(spec.shard_count, len(units))
    shard_payloads: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for index in range(shard_count):
        shard_id = f"shard-{index + 1:03d}"
        payload = {
            "schema_version": 1,
            "review_sha256": review_sha,
            "shard_id": shard_id,
            "units": units[index::shard_count],
        }
        shard_payloads.append(payload)
        shards.append({"shard_id": shard_id, "unit_count": len(payload["units"])})
    return {
        "semantic": semantic,
        "review_sha256": review_sha,
        "shards": shards,
        "shard_payloads": shard_payloads,
        "prompt": review_prompt(),
        "secret": secret,
    }


__all__ = [
    "ERROR_TYPES",
    "SEVERITIES",
    "SEVERITY_WEIGHTS",
    "build_review_plan",
    "category_for_row",
    "rank_key",
    "review_prompt",
    "select_book",
    "semantic_hash",
]
