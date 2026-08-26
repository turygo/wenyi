"""Deterministic sampling and evidence-backed subagent review artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trans_novel.benchmark.corpus import canonical_json, count_words, sha256_bytes

_ERROR_TYPES = (
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
_SEVERITIES = ("critical", "major", "minor")
_WEIGHTS = {"critical": 25, "major": 8, "minor": 2}


class ReviewArtifactError(ValueError):
    """A review request, shard response, or frozen artifact is invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ReviewSpec(_StrictModel):
    schema_version: Literal[1]
    benchmark_id: str = Field(min_length=1)
    run_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    segments_per_book: int = Field(default=24, ge=4, le=200)
    shard_count: int = Field(default=8, ge=1, le=32)

    @field_validator("seed")
    @classmethod
    def reject_bool_seed(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("seed must be an integer")
        return value


class ReviewFinding(_StrictModel):
    finding_id: str = Field(pattern=r"^[A-Za-z0-9._-]{1,128}$")
    side: Literal["A", "B"]
    type: Literal[
        "omission",
        "addition",
        "mistranslation",
        "named_entity",
        "terminology",
        "pronoun_reference",
        "context",
        "style_register",
        "fluency",
    ]
    severity: Literal["critical", "major", "minor"]
    source_quote: str = Field(min_length=1)
    target_quote: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class UnitReview(_StrictModel):
    unit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    winner: Literal["A", "B", "tie"]
    verdict_reason: str = Field(min_length=1)
    findings: list[ReviewFinding]

    @model_validator(mode="after")
    def decisive_error_exists(self) -> UnitReview:
        ids = [finding.finding_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding_id values must be unique within a unit")
        if self.winner == "A" and not any(finding.side == "B" for finding in self.findings):
            raise ValueError("winner A requires evidence against B")
        if self.winner == "B" and not any(finding.side == "A" for finding in self.findings):
            raise ValueError("winner B requires evidence against A")
        return self


class ShardReview(_StrictModel):
    schema_version: Literal[1]
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_id: str = Field(pattern=r"^shard-[0-9]{3}$")
    reviews: list[UnitReview]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReviewArtifactError(f"invalid JSON artifact {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as error:
        raise ReviewArtifactError(f"invalid JSONL artifact {path}: {error}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ReviewArtifactError(f"JSONL rows must be objects: {path}")
    return rows


def _load_spec(value: str | os.PathLike[str] | dict[str, Any]) -> ReviewSpec:
    try:
        if isinstance(value, dict):
            raw = value
        else:
            raw = yaml.safe_load(Path(value).read_text(encoding="utf-8"))
        return ReviewSpec.model_validate(raw)
    except Exception as error:
        raise ReviewArtifactError(f"invalid ReviewSpec: {error}") from error


def _hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _rank(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).hexdigest()


def _category(row: dict[str, Any]) -> str:
    if row.get("review_findings") or row.get("backtranslation_findings"):
        return "risk"
    source = str(row["source"])
    if any(mark in source for mark in ('"', "“", "”", "‘", "’")):
        return "dialogue"
    if len(source) >= 500:
        return "long_sentence"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", source):
        return "terminology"
    return "narrative"


def _select_book(rows: list[dict[str, Any]], *, seed: int, limit: int) -> list[dict[str, Any]]:
    quotas = {
        "risk": max(1, limit // 4),
        "dialogue": max(1, limit // 4),
        "terminology": max(1, limit // 6),
        "long_sentence": max(1, limit // 6),
    }
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[_category(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category, quota in quotas.items():
        ordered = sorted(
            by_category.get(category, []),
            key=lambda row: _rank(seed, f"sample:{category}", row["segment_id"]),
        )
        for row in ordered[:quota]:
            selected.append(row)
            selected_ids.add(row["segment_id"])
    remaining = sorted(
        [row for row in rows if row["segment_id"] not in selected_ids],
        key=lambda row: _rank(seed, "sample:fill", row["segment_id"]),
    )
    selected.extend(remaining[: max(0, limit - len(selected))])
    return sorted(selected[:limit], key=lambda row: (row["chapter_index"], row["segment_index"]))


def _review_prompt() -> dict[str, Any]:
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
        "error_types": list(_ERROR_TYPES),
        "severities": list(_SEVERITIES),
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


def prepare_review(
    run_dir: str | os.PathLike[str],
    review_spec: str | os.PathLike[str] | dict[str, Any],
    out_dir: str | os.PathLike[str],
) -> Path:
    run = Path(run_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise ReviewArtifactError("review output already exists")
    spec = _load_spec(review_spec)
    run_manifest_path = run / "run.json"
    run_manifest = _read_json(run_manifest_path)
    if (
        run_manifest.get("schema_version") != 2
        or run_manifest.get("run_mode") != "full"
        or run_manifest.get("benchmark_id") != spec.benchmark_id
        or sha256_bytes(run_manifest_path.read_bytes()) != spec.run_sha256
    ):
        raise ReviewArtifactError("review run identity mismatch")
    state = _read_json(run / "run_state.json")
    if state.get("status") != "completed":
        raise ReviewArtifactError("review requires a completed full run")
    candidates = _read_json(run / "candidates.json")
    raw_candidate_ids = {row.get("candidate_id") for row in candidates}
    if not 2 <= len(raw_candidate_ids) <= 6 or any(
        not isinstance(value, str) for value in raw_candidate_ids
    ):
        raise ReviewArtifactError("comparative review requires two to six candidates")
    candidate_ids = sorted(raw_candidate_ids)
    by_scope: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for artifact in candidates:
        candidate = artifact.get("candidate_id")
        book_id = artifact.get("book_id")
        replicate = artifact.get("replicate")
        path = run / str(artifact.get("segments_path"))
        if (
            not isinstance(candidate, str)
            or not isinstance(book_id, str)
            or not isinstance(replicate, int)
        ):
            raise ReviewArtifactError("candidate review provenance is invalid")
        for row in _read_jsonl(path):
            key = (book_id, replicate, str(row.get("segment_id")))
            by_scope[key][candidate] = row
    by_book: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (book_id, replicate, source_segment_id), values in sorted(by_scope.items()):
        if set(values) != set(candidate_ids):
            raise ReviewArtifactError("candidate segment sets do not match")
        baseline = values[candidate_ids[0]]
        if any(row.get("source") != baseline.get("source") for row in values.values()):
            raise ReviewArtifactError("candidate source text mismatch")
        category_row = {
            **baseline,
            "review_findings": [
                finding for row in values.values() for finding in row.get("review_findings", [])
            ],
            "backtranslation_findings": [
                finding
                for row in values.values()
                for finding in row.get("backtranslation_findings", [])
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
                "category": _category(category_row),
                "source": baseline["source"],
                "source_sha256": baseline.get("source_sha256"),
                "candidate_targets": {
                    candidate: values[candidate]["target"] for candidate in candidate_ids
                },
            }
        )
    sampled = [
        row
        for book_id in sorted(by_book)
        for row in _select_book(by_book[book_id], seed=spec.seed, limit=spec.segments_per_book)
    ]
    units: list[dict[str, Any]] = []
    secret: dict[str, Any] = {}
    for row in sampled:
        for first_candidate, second_candidate in combinations(candidate_ids, 2):
            pair = [first_candidate, second_candidate]
            orientation_key = f"{row['segment_id']}:{first_candidate}:{second_candidate}"
            if int(_rank(spec.seed, "orientation", orientation_key), 16) % 2:
                pair.reverse()
            unit_id = _hash(
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
        raise ReviewArtifactError("review sampling produced no units")
    semantic = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "run_sha256": spec.run_sha256,
        "spec": spec.model_dump(mode="python"),
        "candidate_count": len(candidate_ids),
        "book_ids": sorted(by_book),
        "unit_count": len(units),
        "units_sha256": _hash(units),
        "prompt_sha256": _hash(_review_prompt()),
    }
    review_sha = _hash(semantic)
    shard_count = min(spec.shard_count, len(units))
    shards: list[dict[str, Any]] = []
    for index in range(shard_count):
        shard_id = f"shard-{index + 1:03d}"
        payload = {
            "schema_version": 1,
            "review_sha256": review_sha,
            "shard_id": shard_id,
            "units": units[index::shard_count],
        }
        shards.append({"shard_id": shard_id, "unit_count": len(payload["units"])})
        _atomic_json(out / "shards" / f"{shard_id}.json", payload)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "prompt.json", _review_prompt())
    _atomic_json(out / "secret_mapping.json", {"schema_version": 1, "units": secret})
    manifest = {
        **semantic,
        "review_sha256": review_sha,
        "shards": shards,
        "files": {},
    }
    manifest["files"] = {
        str(path.relative_to(out)): sha256_bytes(path.read_bytes())
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    _atomic_json(out / "review.json", manifest)
    return out


def _unit_index(review: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_json(review / "review.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ReviewArtifactError("review file manifest invalid")
    for relative, digest in files.items():
        path = review / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            raise ReviewArtifactError("review input artifact changed")
    semantic = {
        key: manifest[key]
        for key in (
            "schema_version",
            "benchmark_id",
            "run_sha256",
            "spec",
            "candidate_count",
            "book_ids",
            "unit_count",
            "units_sha256",
            "prompt_sha256",
        )
    }
    if _hash(semantic) != manifest.get("review_sha256"):
        raise ReviewArtifactError("review semantic hash mismatch")
    units: dict[str, dict[str, Any]] = {}
    ordered_shards: list[list[dict[str, Any]]] = []
    shard_ids: set[str] = set()
    for shard in manifest.get("shards", []):
        shard_id = shard.get("shard_id")
        if not isinstance(shard_id, str) or shard_id in shard_ids:
            raise ReviewArtifactError("review shard identity invalid")
        shard_ids.add(shard_id)
        payload = _read_json(review / "shards" / f"{shard_id}.json")
        if (
            payload.get("schema_version") != 1
            or payload.get("review_sha256") != manifest.get("review_sha256")
            or payload.get("shard_id") != shard_id
            or not isinstance(payload.get("units"), list)
            or len(payload["units"]) != shard.get("unit_count")
        ):
            raise ReviewArtifactError("review shard payload invalid")
        ordered_shards.append(payload["units"])
        for unit in payload["units"]:
            unit_id = unit.get("unit_id")
            if not isinstance(unit_id, str) or unit_id in units:
                raise ReviewArtifactError("review unit identity invalid")
            units[unit_id] = unit
    ordered_units = [
        shard_units[offset]
        for offset in range(max((len(rows) for rows in ordered_shards), default=0))
        for shard_units in ordered_shards
        if offset < len(shard_units)
    ]
    if len(units) != manifest.get("unit_count") or _hash(ordered_units) != manifest.get(
        "units_sha256"
    ):
        raise ReviewArtifactError("review unit set changed")
    return manifest, units


def finalize_review(
    review_dir: str | os.PathLike[str],
    results_dir: str | os.PathLike[str],
) -> Path:
    review = Path(review_dir).expanduser().resolve()
    results = Path(results_dir).expanduser().resolve()
    manifest, units = _unit_index(review)
    secret = _read_json(review / "secret_mapping.json").get("units", {})
    reviewed: dict[str, dict[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        path = results / f"{shard_id}.json"
        if not path.is_file():
            raise ReviewArtifactError(f"missing shard review: {shard_id}")
        try:
            parsed = ShardReview.model_validate(_read_json(path))
        except Exception as error:
            raise ReviewArtifactError(f"invalid shard review {shard_id}: {error}") from error
        if parsed.review_sha256 != manifest["review_sha256"] or parsed.shard_id != shard_id:
            raise ReviewArtifactError("shard review identity mismatch")
        expected_payload = _read_json(review / "shards" / f"{shard_id}.json")
        expected_ids = {unit["unit_id"] for unit in expected_payload["units"]}
        actual_ids = {item.unit_id for item in parsed.reviews}
        if expected_ids != actual_ids or len(parsed.reviews) != len(actual_ids):
            raise ReviewArtifactError("shard review unit set mismatch")
        for item in parsed.reviews:
            unit = units[item.unit_id]
            for finding in item.findings:
                target = unit["targets"][finding.side]
                if finding.source_quote not in unit["source"]:
                    raise ReviewArtifactError("finding source quote is not present")
                if finding.target_quote not in target:
                    raise ReviewArtifactError("finding target quote is not present")
            reviewed[item.unit_id] = item.model_dump(mode="python")
        result_hashes[path.name] = sha256_bytes(path.read_bytes())
    if set(reviewed) != set(units):
        raise ReviewArtifactError("review result set is incomplete")
    findings: list[dict[str, Any]] = []
    wins: Counter[str] = Counter()
    by_book: dict[str, Counter[str]] = defaultdict(Counter)
    errors: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_words: Counter[str] = Counter()
    for unit_id in sorted(reviewed):
        unit = units[unit_id]
        mapping = secret[unit_id]
        review_row = reviewed[unit_id]
        for candidate in set(mapping.values()):
            source_words[candidate] += count_words(unit["source"])
        winner = review_row["winner"]
        normalized_winner = "tie" if winner == "tie" else mapping[winner]
        wins[normalized_winner] += 1
        by_book[unit["book_id"]][normalized_winner] += 1
        for finding in review_row["findings"]:
            candidate = mapping[finding["side"]]
            key = f"{finding['severity']}:{finding['type']}"
            errors[candidate][key] += 1
            row = {
                "unit_id": unit_id,
                "book_id": unit["book_id"],
                "segment_id": unit["segment_id"],
                "candidate_id": candidate,
                **finding,
            }
            row.pop("side", None)
            findings.append(row)
            if len(examples[candidate]) < 20:
                examples[candidate].append(row)
    candidates = sorted(
        {candidate for mapping in secret.values() for candidate in mapping.values()}
    )
    candidate_summary: dict[str, Any] = {}
    for candidate in candidates:
        severity_counts = {
            severity: sum(
                count for key, count in errors[candidate].items() if key.startswith(f"{severity}:")
            )
            for severity in _SEVERITIES
        }
        weighted = sum(_WEIGHTS[severity] * severity_counts[severity] for severity in _SEVERITIES)
        words = source_words[candidate]
        candidate_summary[candidate] = {
            "source_words": words,
            "wins": wins[candidate],
            "ties": wins["tie"],
            "severity_counts": severity_counts,
            "type_counts": {
                error_type: sum(
                    count
                    for key, count in errors[candidate].items()
                    if key.endswith(f":{error_type}")
                )
                for error_type in _ERROR_TYPES
            },
            "weighted_errors_per_10k": None if not words else weighted / words * 10000,
            "examples": examples[candidate],
        }
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            candidate_summary[candidate]["severity_counts"]["critical"],
            candidate_summary[candidate]["severity_counts"]["major"],
            candidate_summary[candidate]["weighted_errors_per_10k"],
            -candidate_summary[candidate]["wins"],
            candidate,
        ),
    )
    first = candidate_summary[ranking[0]]
    second = candidate_summary[ranking[1]]
    first_key = (
        first["severity_counts"]["critical"],
        first["severity_counts"]["major"],
        first["weighted_errors_per_10k"],
        -first["wins"],
    )
    second_key = (
        second["severity_counts"]["critical"],
        second["severity_counts"]["major"],
        second["weighted_errors_per_10k"],
        -second["wins"],
    )
    comparison = {
        "schema_version": 1,
        "review_sha256": manifest["review_sha256"],
        "winner": ranking[0] if first_key < second_key else None,
        "ranking": ranking,
        "candidates": candidate_summary,
        "by_book": {book: dict(sorted(counts.items())) for book, counts in sorted(by_book.items())},
    }
    findings.sort(
        key=lambda row: (row["candidate_id"], row["book_id"], row["segment_id"], row["finding_id"])
    )
    findings_path = review / "findings.jsonl"
    findings_path.write_text(
        "".join(canonical_json(row) + "\n" for row in findings), encoding="utf-8"
    )
    _atomic_json(review / "comparison.json", comparison)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "winner": comparison["winner"],
        "reviewed_units": len(reviewed),
        "finding_count": len(findings),
        "ranking": ranking,
    }
    _atomic_json(review / "summary.json", summary)
    complete = {
        "schema_version": 1,
        "review_sha256": manifest["review_sha256"],
        "result_files": result_hashes,
        "derived_files": {
            name: sha256_bytes((review / name).read_bytes())
            for name in ("findings.jsonl", "comparison.json", "summary.json")
        },
    }
    _atomic_json(review / "review_complete.json", complete)
    return review


def validate_review(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    review = Path(review_dir).expanduser().resolve()
    manifest, units = _unit_index(review)
    complete_path = review / "review_complete.json"
    if complete_path.exists():
        complete = _read_json(complete_path)
        if complete.get("review_sha256") != manifest["review_sha256"]:
            raise ReviewArtifactError("review completion identity mismatch")
        for name, digest in complete.get("derived_files", {}).items():
            path = review / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                raise ReviewArtifactError("review derived artifact changed")
    return {
        "review_sha256": manifest["review_sha256"],
        "unit_count": len(units),
        "status": "complete" if complete_path.exists() else "prepared",
    }


__all__ = [
    "ReviewArtifactError",
    "ReviewSpec",
    "finalize_review",
    "prepare_review",
    "validate_review",
]
