"""Strict review models, evidence validation, and canonical publication."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trans_novel.benchmark.artifacts import atomic_json, sha256_bytes, write_jsonl
from trans_novel.benchmark.contracts import RUN_SCHEMA_VERSION
from trans_novel.benchmark.corpus import count_words
from trans_novel.benchmark.review.sampling import (
    ERROR_TYPES,
    SEVERITIES,
    SEVERITY_WEIGHTS,
    build_review_plan,
    semantic_hash,
)


class ReviewArtifactError(ValueError):
    """A review request, shard response, or frozen artifact is invalid."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class ReviewSpec(StrictModel):
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


class ReviewFinding(StrictModel):
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


class UnitReview(StrictModel):
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


class ShardReview(StrictModel):
    schema_version: Literal[1]
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shard_id: str = Field(pattern=r"^shard-[0-9]{3}$")
    reviews: list[UnitReview]


def _read_review_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReviewArtifactError(f"invalid JSON artifact {path}: {error}") from error


def _read_review_jsonl(path: Path) -> list[dict[str, Any]]:
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


def prepare_review_artifacts(
    run_dir: str | os.PathLike[str],
    review_spec: str | os.PathLike[str] | dict[str, Any],
    out_dir: str | os.PathLike[str],
) -> Path:
    run = Path(run_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise ReviewArtifactError("review output already exists")
    try:
        raw = (
            review_spec
            if isinstance(review_spec, dict)
            else yaml.safe_load(Path(review_spec).read_text(encoding="utf-8"))
        )
        spec = ReviewSpec.model_validate(raw)
    except Exception as error:
        raise ReviewArtifactError(f"invalid ReviewSpec: {error}") from error
    run_manifest_path = run / "run.json"
    run_manifest = _read_review_json(run_manifest_path)
    if (
        run_manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or run_manifest.get("run_mode") != "full"
        or run_manifest.get("benchmark_id") != spec.benchmark_id
        or sha256_bytes(run_manifest_path.read_bytes()) != spec.run_sha256
    ):
        raise ReviewArtifactError("review run identity mismatch")
    state = _read_review_json(run / "run_state.json")
    if state.get("status") != "completed":
        raise ReviewArtifactError("review requires a completed full run")
    candidates = _read_review_json(run / "candidates.json")
    raw_candidate_ids = {row.get("candidate_id") for row in candidates}
    if not 2 <= len(raw_candidate_ids) <= 6 or any(not isinstance(value, str) for value in raw_candidate_ids):  # fmt: skip
        raise ReviewArtifactError("comparative review requires two to six candidates")
    candidate_rows: list[dict[str, Any]] = []
    for artifact in candidates:
        candidate = artifact.get("candidate_id")
        book_id = artifact.get("book_id")
        replicate = artifact.get("replicate")
        path = run / str(artifact.get("segments_path"))
        if not isinstance(candidate, str) or not isinstance(book_id, str) or not isinstance(replicate, int):  # fmt: skip
            raise ReviewArtifactError("candidate review provenance is invalid")
        candidate_rows.append(
            {
                "candidate_id": candidate,
                "book_id": book_id,
                "replicate": replicate,
                "rows": _read_review_jsonl(path),
            }
        )
    try:
        plan = build_review_plan(spec, candidate_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ReviewArtifactError(str(error)) from error
    out.mkdir(parents=True, exist_ok=True)
    for payload in plan["shard_payloads"]:
        atomic_json(out / "shards" / f"{payload['shard_id']}.json", payload)
    atomic_json(out / "prompt.json", plan["prompt"])
    atomic_json(out / "secret_mapping.json", {"schema_version": 1, "units": plan["secret"]})
    manifest = {
        **plan["semantic"],
        "review_sha256": plan["review_sha256"],
        "shards": plan["shards"],
        "files": {},
    }
    manifest["files"] = {
        str(path.relative_to(out)): sha256_bytes(path.read_bytes())
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    atomic_json(out / "review.json", manifest)
    return out


def index_review(review: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _read_review_json(review / "review.json")
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
    if semantic_hash(semantic) != manifest.get("review_sha256"):
        raise ReviewArtifactError("review semantic hash mismatch")
    units: dict[str, dict[str, Any]] = {}
    ordered_shards: list[list[dict[str, Any]]] = []
    shard_ids: set[str] = set()
    for shard in manifest.get("shards", []):
        shard_id = shard.get("shard_id")
        if not isinstance(shard_id, str) or shard_id in shard_ids:
            raise ReviewArtifactError("review shard identity invalid")
        shard_ids.add(shard_id)
        payload = _read_review_json(review / "shards" / f"{shard_id}.json")
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
    if len(units) != manifest.get("unit_count") or semantic_hash(ordered_units) != manifest.get(
        "units_sha256"
    ):
        raise ReviewArtifactError("review unit set changed")
    return manifest, units


def _load_review_results(
    review: Path,
    results: Path,
    manifest: dict[str, Any],
    units: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    reviewed: dict[str, dict[str, Any]] = {}
    result_hashes: dict[str, str] = {}
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        path = results / f"{shard_id}.json"
        if not path.is_file():
            raise ReviewArtifactError(f"missing shard review: {shard_id}")
        try:
            parsed = ShardReview.model_validate(_read_review_json(path))
        except Exception as error:
            raise ReviewArtifactError(f"invalid shard review {shard_id}: {error}") from error
        if parsed.review_sha256 != manifest["review_sha256"] or parsed.shard_id != shard_id:
            raise ReviewArtifactError("shard review identity mismatch")
        expected_payload = _read_review_json(review / "shards" / f"{shard_id}.json")
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
    return reviewed, result_hashes


def _collect_evidence(
    reviewed: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
    secret: dict[str, Any],
) -> tuple[Any, ...]:
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
    return findings, wins, by_book, errors, examples, source_words


def _summarize_candidates(
    secret: dict[str, Any],
    wins: Counter[str],
    errors: dict[str, Counter[str]],
    examples: dict[str, list[dict[str, Any]]],
    source_words: Counter[str],
) -> dict[str, Any]:
    candidates = sorted(
        {candidate for mapping in secret.values() for candidate in mapping.values()}
    )
    summary: dict[str, Any] = {}
    for candidate in candidates:
        severity_counts = {
            severity: sum(
                count for key, count in errors[candidate].items() if key.startswith(f"{severity}:")
            )
            for severity in SEVERITIES
        }
        weighted = sum(
            SEVERITY_WEIGHTS[severity] * severity_counts[severity] for severity in SEVERITIES
        )
        words = source_words[candidate]
        summary[candidate] = {
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
                for error_type in ERROR_TYPES
            },
            "weighted_errors_per_10k": None if not words else weighted / words * 10000,
            "examples": examples[candidate],
        }
    return summary


def _build_comparison(
    manifest: dict[str, Any],
    secret: dict[str, Any],
    evidence: tuple[Any, ...],
) -> tuple[dict[str, Any], list[str]]:
    _findings, wins, by_book, errors, examples, source_words = evidence
    candidates = _summarize_candidates(secret, wins, errors, examples, source_words)
    ranking = sorted(
        candidates,
        key=lambda candidate: (
            candidates[candidate]["severity_counts"]["critical"],
            candidates[candidate]["severity_counts"]["major"],
            candidates[candidate]["weighted_errors_per_10k"],
            -candidates[candidate]["wins"],
            candidate,
        ),
    )
    first = candidates[ranking[0]]
    second = candidates[ranking[1]]
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
    return {
        "schema_version": 1,
        "review_sha256": manifest["review_sha256"],
        "winner": ranking[0] if first_key < second_key else None,
        "ranking": ranking,
        "candidates": candidates,
        "by_book": {book: dict(sorted(counts.items())) for book, counts in sorted(by_book.items())},
    }, ranking


def _publish_review_results(
    review: Path,
    manifest: dict[str, Any],
    result_hashes: dict[str, str],
    findings: list[dict[str, Any]],
    comparison: dict[str, Any],
    ranking: list[str],
    reviewed_units: int,
) -> None:
    findings.sort(
        key=lambda row: (row["candidate_id"], row["book_id"], row["segment_id"], row["finding_id"])
    )
    write_jsonl(review / "findings.jsonl", findings)
    atomic_json(review / "comparison.json", comparison)
    atomic_json(
        review / "summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "winner": comparison["winner"],
            "reviewed_units": reviewed_units,
            "finding_count": len(findings),
            "ranking": ranking,
        },
    )
    atomic_json(
        review / "review_complete.json",
        {
            "schema_version": 1,
            "review_sha256": manifest["review_sha256"],
            "result_files": result_hashes,
            "derived_files": {
                name: sha256_bytes((review / name).read_bytes())
                for name in ("findings.jsonl", "comparison.json", "summary.json")
            },
        },
    )


def finalize_review_artifacts(
    review_dir: str | os.PathLike[str],
    results_dir: str | os.PathLike[str],
) -> Path:
    review = Path(review_dir).expanduser().resolve()
    results = Path(results_dir).expanduser().resolve()
    manifest, units = index_review(review)
    secret = _read_review_json(review / "secret_mapping.json").get("units", {})
    reviewed, result_hashes = _load_review_results(review, results, manifest, units)
    evidence = _collect_evidence(reviewed, units, secret)
    findings = evidence[0]
    comparison, ranking = _build_comparison(manifest, secret, evidence)
    _publish_review_results(
        review, manifest, result_hashes, findings, comparison, ranking, len(reviewed)
    )
    return review


def validate_review_artifacts(review_dir: str | os.PathLike[str]) -> dict[str, Any]:
    review = Path(review_dir).expanduser().resolve()
    manifest, units = index_review(review)
    complete_path = review / "review_complete.json"
    if complete_path.exists():
        complete = _read_review_json(complete_path)
        if complete.get("review_sha256") != manifest["review_sha256"]:
            raise ReviewArtifactError("review completion identity mismatch")
        for name, digest in complete.get("derived_files", {}).items():
            path = review / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                raise ReviewArtifactError("review derived artifact changed")
    return {"review_sha256": manifest["review_sha256"], "unit_count": len(units), "status": "complete" if complete_path.exists() else "prepared"}  # fmt: skip


__all__ = [
    "ReviewArtifactError",
    "ReviewFinding",
    "ReviewSpec",
    "ShardReview",
    "index_review",
    "prepare_review_artifacts",
    "validate_review_artifacts",
]
