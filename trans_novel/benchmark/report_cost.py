"""Phase 8 cost, performance, failure and repricing facts.

This module deliberately consumes only validated benchmark evidence.  It never
returns source/target text or request/response digests: those are transient
validation inputs and are not part of the frozen pricing facts.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trans_novel.benchmark.corpus import canonical_json, count_words, sha256_bytes, validate_corpus
from trans_novel.benchmark.pricing import (
    CostQuote,
    PriceSnapshot,
    UnknownCost,
    load_price_snapshot,
    quote_usage,
)
from trans_novel.benchmark.report_schema import ReportSpec
from trans_novel.benchmark.runner import (
    BenchmarkError,
    FullRunner,
    _glossary,
    _locked_terms,
    _safe_id,
    _validate_segment_hashes,
    load_preparation_bundle,
    validate_preparation,
)
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.llm.usage import merge_usage_summaries, usage_delta
from trans_novel.pipeline import lint

__all__ = ["CostAnalysisError", "analyze_cost_system", "reprice_cost_system"]


class CostAnalysisError(ValueError):
    """Malformed or inconsistent cost evidence."""


_FIELDS = (
    "schema_version",
    "logical_call_id",
    "attempt_index",
    "started_at",
    "elapsed_ms",
    "stage",
    "agent",
    "operation",
    "provider",
    "requested_model",
    "resolved_model",
    "reasoning_enabled",
    "reasoning_effort",
    "temperature",
    "seed",
    "json_mode",
    "max_tokens",
    "status",
    "retry_class",
    "http_status",
    "finish_reason",
    "response_id",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
    "billed_usage_unknown",
    "request_sha256",
    "response_sha256",
)
_ENRICHED = {"benchmark_id", "candidate_id", "run_id", "book_id"}
_HEX64 = set("0123456789abcdef")


def _fail(message: str) -> None:
    raise CostAnalysisError(message)


def _json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except Exception:  # pragma: no cover - exact source is diagnostic only
        _fail(f"invalid JSON artifact: {path.name}")


def _canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _price_hash(snapshot: PriceSnapshot) -> str:
    return sha256_bytes(canonical_json(snapshot.model_dump(mode="json")).encode("utf-8"))


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a nonblank string")
    return value.strip()


def _safe_rel(value: Any, name: str) -> Path:
    if not isinstance(value, str):
        _fail(f"{name} must be a relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        _fail(f"{name} is not safe")
    return path


def _dec(value: Any, name: str) -> Decimal:
    if isinstance(value, bool):
        _fail(f"{name} is not a Decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        _fail(f"{name} is not a Decimal")
    if not result.is_finite():
        _fail(f"{name} is not finite")
    return result


def _money(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _rate_key(value: Decimal) -> str:
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return result


def _int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _merge_counts(target: dict[str, int], source: Mapping[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _empty_usage() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "totals": dict.fromkeys(
            (
                "calls",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
            ),
            0,
        ),
    }


def _usage_totals(value: Any) -> dict[str, int] | None:
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 2
        or not isinstance(value.get("totals"), dict)
    ):
        _fail("invalid usage summary")
    totals = value["totals"]
    result: dict[str, int] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    ):
        if key in totals:
            result[key] = _int(totals[key], f"usage totals {key}")
    return result


def _add_usage(known: dict[str, int], attempt: CallAttemptTelemetry) -> None:
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    ):
        known[field] = known.get(field, 0) + int(getattr(attempt, field))


def _parse_telemetry(
    path: Path, *, artifact_id: str, benchmark_id: str, expected_book: str | None, run_ids: set[str]
) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        _fail(f"missing telemetry artifact: {artifact_id}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        _fail(f"cannot read telemetry artifact: {artifact_id}")
    attempts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    by_call: dict[str, list[int]] = defaultdict(list)
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception:
            _fail(f"malformed telemetry JSON: {artifact_id}")
        keys = set(value) if isinstance(value, dict) else set()
        if keys != set(_FIELDS) and keys != set(_FIELDS) | _ENRICHED:
            _fail(f"telemetry envelope schema mismatch: {artifact_id}")
        enriched = keys == set(_FIELDS) | _ENRICHED
        candidate_context = value.get("candidate_id", "producer")
        book_context = value.get("book_id", expected_book)
        if enriched:
            for name in ("benchmark_id", "candidate_id", "run_id"):
                _text(value.get(name), name)
            if value["benchmark_id"] != benchmark_id:
                _fail("telemetry benchmark context mismatch")
            run_ids.add(value["run_id"])
        if book_context is not None:
            _text(book_context, "book_id")
            if expected_book is not None and book_context != expected_book:
                _fail("telemetry book context mismatch")
        try:
            parsed = CallAttemptTelemetry.model_validate({k: value[k] for k in _FIELDS})
        except Exception:
            _fail(f"invalid telemetry record: {artifact_id}")
        key = (artifact_id, parsed.logical_call_id, parsed.attempt_index)
        if key in seen:
            _fail("duplicate physical telemetry attempt")
        seen.add(key)
        by_call[parsed.logical_call_id].append(parsed.attempt_index)
        attempts.append(
            {"telemetry": parsed, "candidate_id": candidate_context, "book_id": book_context}
        )
    for indices in by_call.values():
        if sorted(indices) != list(range(1, len(indices) + 1)):
            _fail("telemetry attempt indices are not contiguous")
    return attempts


def _source_records(artifact_dir: Path, *, kind: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    passages = artifact_dir / "passages"
    if passages.exists():
        for path in sorted(passages.glob("*.json")):
            row = _json(path)
            if (
                not isinstance(row, dict)
                or row.get("status") != "complete"
                or not isinstance(row.get("segments"), list)
            ):
                _fail("incomplete passage artifact")
            if not isinstance(row.get("passage_id"), str) or not row["passage_id"].strip():
                _fail("passage identifier is missing")
            for segment in row["segments"]:
                if not isinstance(segment, dict):
                    _fail("invalid stage segment")
                source = segment.get("source")
                if not isinstance(source, str):
                    _fail("stage source is invalid")
                final = segment.get("final")
                final_hash = segment.get("final_sha256")
                valid_final = (
                    isinstance(final, str)
                    and bool(final.strip())
                    and isinstance(final_hash, str)
                    and sha256_bytes(final.encode()) == final_hash
                )
                record = {
                    **segment,
                    "passage_id": row.get("passage_id"),
                    "book_id": row.get("book_id"),
                    "source_words": count_words(source),
                    "completed": valid_final,
                    "final": final,
                    "final_hash": final_hash,
                    "lint_findings": segment.get("translation_lint_issues", []),
                    "review_findings": segment.get("review_findings", []),
                    "backtranslation_findings": segment.get("backtranslation_findings", []),
                    "polish_accepted": segment.get("polish_accepted"),
                    "polish_rejection_reasons": segment.get("polish_rejection_reasons", []),
                    "consistency_findings": segment.get("consistency_findings", []),
                    "_passage_file": path.name,
                }
                for name in (
                    "alignment_errors",
                    "protocol_errors",
                    "json_errors",
                    "fallback",
                    "required_node_failures",
                ):
                    if name in segment:
                        record[name] = segment[name]
                result.append(record)
    return result


def _full_stage_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    stage = row.get("stage")
    if not isinstance(stage, list):
        _fail("full candidate stage is invalid")
    result = []
    for segment in stage:
        if not isinstance(segment, dict):
            _fail("full candidate stage is invalid")
        source = segment.get("source")
        final = segment.get("final_after_full_pipeline")
        final_hash = segment.get("final_sha256")
        if not isinstance(source, str) or not isinstance(segment.get("segment_id"), str):
            _fail("full stage source is invalid")
        record = {
            "book_id": row.get("book_id"),
            "segment_id": segment["segment_id"],
            "source": source,
            "source_words": count_words(source),
            "completed": isinstance(final, str)
            and bool(final.strip())
            and isinstance(final_hash, str)
            and sha256_bytes(final.encode()) == final_hash,
            "final": final,
            "final_hash": final_hash,
            "lint_findings": segment.get("lint_findings", []),
            "review_findings": segment.get("review_findings", []),
            "backtranslation_findings": segment.get("backtranslation_findings", []),
            "polish_accepted": segment.get("polish_accepted"),
            "polish_rejection_reasons": segment.get("polish_rejection_reasons", []),
            "consistency_findings": segment.get("consistency_findings", []),
        }
        for name in (
            "alignment_errors",
            "protocol_errors",
            "json_errors",
            "fallback",
            "required_node_failures",
        ):
            if name in segment:
                record[name] = segment[name]
        result.append(record)
    return result


def _quote_attempt(
    snapshot: PriceSnapshot, attempt: CallAttemptTelemetry
) -> tuple[Decimal | None, dict[str, Any] | None, str | None]:
    if attempt.status == "error":
        return None, None, "failed_attempt"
    if attempt.billed_usage_unknown:
        return None, None, "billed_usage_unknown"
    model = snapshot.models.get(attempt.requested_model)
    if model is None:
        _fail(f"no pricing for requested model {attempt.requested_model!r}")
    bands = sorted({rule.time_band for rule in model.rules})
    if len(bands) != 1:
        _fail("missing exact time band")
    usage = {
        field: int(getattr(attempt, field))
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "reasoning_tokens",
        )
    }
    try:
        quote = quote_usage(snapshot, attempt.requested_model, usage, time_band=bands[0])
    except Exception as exc:
        _fail(str(exc))
    if isinstance(quote, UnknownCost):
        return None, None, "billed_usage_unknown"
    return (
        quote.total_cost,
        {
            "currency": quote.currency,
            "model": quote.model_id,
            "time_band": quote.time_band,
            "min_prompt_tokens": quote.min_prompt_tokens,
            "max_prompt_tokens": quote.max_prompt_tokens,
            "uncached_input_cost": _money(quote.uncached_input_cost),
            "cached_input_cost": _money(quote.cached_input_cost),
            "output_cost": _money(quote.output_cost),
            "total_cost": _money(quote.total_cost),
        },
        None,
    )


def _empty_aggregate() -> dict[str, Any]:
    return {
        "lower_bound": Decimal(0),
        "complete": True,
        "unknown_count": 0,
        "unknown_reasons": [],
        "by_book": {},
        "by_operation": {},
        "by_model": {},
        "cache_savings": Decimal(0),
        "retry_cost": Decimal(0),
        "unknown_retry_count": 0,
        "book_attribution_complete": True,
    }


def _aggregate(artifacts: Sequence[dict[str, Any]], snapshot: PriceSnapshot) -> dict[str, Any]:
    agg = _empty_aggregate()
    for artifact in artifacts:
        source_book_map = artifact.get("source_words_by_book", {})
        for item in artifact.get("attempts", []):
            raw_book = item.get("book_id")
            book = raw_book or artifact.get("book_id")
            if len(source_book_map) > 1:
                if (
                    not isinstance(raw_book, str)
                    or not raw_book.strip()
                    or raw_book not in source_book_map
                ):
                    agg["book_attribution_complete"] = False
                    book = None
            elif book is None and len(source_book_map) == 1:
                book = next(iter(source_book_map))
            attempt: CallAttemptTelemetry = item["telemetry"]
            cost, quote_meta, unknown = _quote_attempt(snapshot, attempt)
            if unknown:
                agg["complete"] = False
                agg["unknown_count"] += 1
                agg["unknown_reasons"].append(unknown)
                if attempt.attempt_index > 1:
                    agg["unknown_retry_count"] += 1
            else:
                assert cost is not None
                agg["lower_bound"] += cost
                uncached_usage = {
                    "prompt_tokens": attempt.prompt_tokens,
                    "completion_tokens": attempt.completion_tokens,
                    "total_tokens": attempt.total_tokens,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": attempt.prompt_tokens,
                    "reasoning_tokens": attempt.reasoning_tokens,
                }
                try:
                    uncached = quote_usage(
                        snapshot,
                        attempt.requested_model,
                        uncached_usage,
                        time_band=quote_meta["time_band"],
                    )  # type: ignore[index]
                    if isinstance(uncached, CostQuote):
                        agg["cache_savings"] += max(Decimal(0), uncached.total_cost - cost)
                except Exception:
                    _fail("cannot compute cache savings")
                if attempt.attempt_index > 1:
                    agg["retry_cost"] += cost
            op = agg["by_operation"].setdefault(attempt.operation, _empty_aggregate())
            model = agg["by_model"].setdefault(attempt.requested_model, _empty_aggregate())
            children = [op, model]
            if book is not None:
                children.insert(0, agg["by_book"].setdefault(book, _empty_aggregate()))
            for child in children:
                if unknown:
                    child["complete"] = False
                    child["unknown_count"] += 1
                    child["unknown_reasons"].append(unknown)
                else:
                    child["lower_bound"] += cost  # type: ignore[operator]
    agg["unknown_reasons"] = sorted(agg["unknown_reasons"])
    return agg


def _serial_aggregate(agg: dict[str, Any], *, words: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "api_cost_lower_bound": _money(agg["lower_bound"]),
        "api_cost": _money(agg["lower_bound"] if agg["complete"] else None),
        "cost_complete": bool(agg["complete"]),
        "unknown_count": agg["unknown_count"],
        "unknown_reasons": sorted(agg["unknown_reasons"]),
        "cache_savings_lower_bound": _money(agg["cache_savings"]),
        "retry_cost_lower_bound": _money(agg["retry_cost"]),
        "unknown_retry_count": agg["unknown_retry_count"],
        "validated_source_words": words,
        "cost_per_100k": _money(agg["lower_bound"] * Decimal(100000) / Decimal(words))
        if agg["complete"] and words
        else None,
        "validated_words_per_cny": _money(Decimal(words) / agg["lower_bound"])
        if agg["complete"] and words and agg["lower_bound"]
        else None,
        "by_book": {},
        "by_operation": {},
        "by_model": {},
        "book_attribution_complete": bool(agg.get("book_attribution_complete", True)),
    }
    for name in ("by_book", "by_operation", "by_model"):
        for key, child in sorted(agg[name].items()):
            result[name][key] = _serial_aggregate(child, words=words if name != "by_book" else 0)
    return result


def _attempt_view(
    attempt: CallAttemptTelemetry, quote: dict[str, Any] | None, book_id: str | None = None
) -> dict[str, Any]:
    result = {
        name: getattr(attempt, name)
        for name in (
            "logical_call_id",
            "attempt_index",
            "started_at",
            "elapsed_ms",
            "agent",
            "operation",
            "status",
            "retry_class",
            "http_status",
            "requested_model",
            "resolved_model",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "reasoning_tokens",
            "billed_usage_unknown",
        )
    }
    if book_id is not None:
        result["book_id"] = book_id
    if quote is not None:
        result["quote"] = quote
    return result


def _metrics(
    artifacts: Sequence[dict[str, Any]], segments: Mapping[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    attempt_pairs = [
        (artifact["artifact_id"], item["telemetry"])
        for artifact in artifacts
        for item in artifact.get("attempts", [])
    ]
    attempts = [item[1] for item in attempt_pairs]
    elapsed = [int(x.elapsed_ms) for x in attempts]
    successes = [x for x in attempts if x.status in {"success", "empty_response"}]
    requested = sum(int(value["source_words"]) for value in segments.values())
    completed = sum(int(value["source_words"]) for value in segments.values() if value["completed"])
    ops = Counter(x.operation for x in attempts)
    statuses = Counter(x.status for x in attempts)
    retry = Counter(x.retry_class for x in attempts if x.retry_class is not None)
    http = Counter(str(x.http_status) for x in attempts if x.http_status is not None)
    lint = Counter()
    review = Counter()
    back = Counter()
    consistency = Counter()
    rejection_reasons = Counter()
    lint_fixes = lint_accepted = lint_rejected = required_failures = 0
    review_unresolved = back_unresolved = consistency_unresolved = 0
    finding_rows: list[tuple[str, str | None, dict[str, Any]]] = []
    for artifact in artifacts:
        for segment in artifact.get("segments", []):
            if isinstance(segment, dict):
                finding_rows.append((artifact["artifact_id"], artifact.get("kind"), segment))
    if not finding_rows:
        finding_rows = [("__segments__", None, value) for value in segments.values()]
    seen_by_kind: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for owner, artifact_kind, value in finding_rows:
        for kind, target in (
            ("lint", lint),
            ("review", review),
            ("backtranslation", back),
            ("consistency", consistency),
        ):
            if kind == "lint" and artifact_kind == "editor":
                continue
            field = {
                "lint": "lint_findings",
                "review": "review_findings",
                "backtranslation": "backtranslation_findings",
                "consistency": "consistency_findings",
            }[kind]
            for finding in value.get(field, []):
                key = (owner, canonical_json(finding))
                if key in seen_by_kind[kind]:
                    continue
                seen_by_kind[kind].add(key)
                if isinstance(finding, dict) and isinstance(finding.get("type"), str):
                    target[finding["type"]] += 1
                if kind == "review" and isinstance(finding, dict) and finding.get("fixed") is False:
                    review_unresolved += 1
                elif kind == "backtranslation" and isinstance(finding, dict):
                    back_unresolved += 1
                elif kind == "consistency" and isinstance(finding, dict):
                    consistency_unresolved += 1
                elif kind == "lint" and isinstance(finding, dict) and finding.get("fixed") is True:
                    lint_fixes += 1
        accepted = value.get("polish_accepted")
        if accepted is True:
            lint_accepted += 1
            lint_fixes += 1
        elif accepted is False:
            lint_rejected += 1
        reasons = value.get("polish_rejection_reasons")
        if isinstance(reasons, list):
            for reason in reasons:
                if isinstance(reason, str):
                    rejection_reasons[reason] += 1
    logical = {(artifact_id, x.logical_call_id) for artifact_id, x in attempt_pairs}
    unknown = sum(1 for x in attempts if x.billed_usage_unknown or x.status == "error")
    reasoning = sum(x.reasoning_tokens for x in attempts)
    mismatch = sum(
        1
        for x in attempts
        if x.resolved_model is not None and x.resolved_model != x.requested_model
    )
    elapsed_success = sum(x.elapsed_ms for x in successes)
    completion = sum(x.completion_tokens for x in successes)
    by_request: dict[tuple[str, str], set[str]] = defaultdict(set)
    for artifact_id, x in attempt_pairs:
        if (
            x.operation == "translate.batch"
            and x.status in {"success", "empty_response"}
            and x.request_sha256
        ):
            by_request[(artifact_id, x.request_sha256)].add(x.logical_call_id)
    duplicate = sum(max(0, len(ids) - 1) for ids in by_request.values())

    def explicit(name: str) -> int | None:
        present = False
        total = 0
        for value in segments.values():
            if name not in value:
                continue
            present = True
            raw = value[name]
            if isinstance(raw, bool):
                total += int(raw)
            elif isinstance(raw, int):
                total += raw
            elif isinstance(raw, list):
                total += len(raw)
            else:
                return None
        return total if present else None

    alignment = explicit("alignment_errors")
    protocol = explicit("protocol_errors")
    json_errors = explicit("json_errors")
    fallback = explicit("fallback")
    required_failures = explicit("required_node_failures")
    return {
        "requested_words": requested,
        "completed_words": completed,
        "completion_rate": (Decimal(completed) / Decimal(requested) if requested else None),
        "attempts": len(attempts),
        "successful_attempts": statuses["success"],
        "empty_attempts": statuses["empty_response"],
        "error_attempts": statuses["error"],
        "failed_attempts": statuses["error"],
        "logical_calls": len(logical),
        "retry_attempts": sum(1 for x in attempts if x.attempt_index > 1),
        "retry_class_counts": dict(sorted(retry.items())),
        "http_status_counts": dict(sorted(http.items())),
        "operation_counts": dict(sorted(ops.items())),
        "status_counts": dict(sorted(statuses.items())),
        "alignment_errors": alignment,
        "protocol_errors": protocol,
        "json_errors": json_errors,
        "fallbacks": fallback,
        "unknown_count": unknown,
        "billed_unknown": sum(x.billed_usage_unknown for x in attempts),
        "reasoning_tokens": reasoning,
        "resolved_model_mismatch": mismatch,
        "latency_p50_ms": _nearest(elapsed, 0.5),
        "latency_p95_ms": _nearest(elapsed, 0.95),
        "wall_time_ms": sum(elapsed),
        "completion_tokens_per_second": (
            Decimal(completion) / (Decimal(elapsed_success) / Decimal(1000))
            if elapsed_success
            else None
        ),
        "resume_duplicate_operations": duplicate,
        "lint_findings": dict(sorted(lint.items())),
        "review_findings": dict(sorted(review.items())),
        "backtranslation_findings": dict(sorted(back.items())),
        "consistency_findings": dict(sorted(consistency.items())),
        "lint_fixes": lint_fixes,
        "polish_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "lint_accepted": lint_accepted,
        "lint_rejected": lint_rejected,
        "required_node_failures": required_failures,
        "review_unresolved": review_unresolved,
        "backtranslation_unresolved": back_unresolved,
        "consistency_unresolved": consistency_unresolved,
    }


def _nearest(values: Sequence[int], probability: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))]


def _decimalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _money(value)
    if isinstance(value, dict):
        return {key: _decimalize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_decimalize(val) for val in value]
    return value


def _validate_run(
    run_dir: Path, spec: ReportSpec, preparation_sha: str, corpus_sha: str
) -> tuple[str, dict[str, Any]]:
    run = _json(run_dir / "run.json")
    if (
        not isinstance(run, dict)
        or run.get("schema_version") != 1
        or run.get("run_mode") not in {"attribution", "full"}
    ):
        _fail("completed run manifest is invalid")
    expected_keys = (
        {
            "schema_version",
            "run_mode",
            "prompt_version",
            "benchmark_id",
            "spec_sha256",
            "corpus_sha256",
            "preparation_sha256",
            "canary_sample_id",
        }
        if run.get("run_mode") == "attribution"
        else {
            "schema_version",
            "run_mode",
            "corpus_sha256",
            "preparation_sha256",
            "benchmark_id",
            "spec_sha256",
            "replicates",
        }
    )
    if set(run) != expected_keys:
        _fail("completed run manifest schema mismatch")
    if (
        run.get("benchmark_id") != spec.benchmark_id
        or run.get("preparation_sha256") != preparation_sha
        or run.get("corpus_sha256") != corpus_sha
    ):
        _fail("run provenance mismatch")
    state = _json(run_dir / "run_state.json")
    if not isinstance(state, dict) or state.get("status") != "completed":
        _fail("run is not completed")
    digest = _canonical_hash(run)
    if digest != spec.run_hash:
        _fail("run hash mismatch")
    return run["run_mode"], run


def _prepare_artifacts(
    preparation_dir: Path, bundle: Any, benchmark_id: str, run_ids: set[str]
) -> list[dict[str, Any]]:
    result = []
    for book_id, book in sorted(bundle.books.items()):
        if not book.telemetry_path:
            _fail("preparation telemetry path is missing")
        attempts = _parse_telemetry(
            preparation_dir / _safe_rel(book.telemetry_path, "preparation telemetry path"),
            artifact_id=f"preparation:{book_id}",
            benchmark_id=benchmark_id,
            expected_book=book_id,
            run_ids=run_ids,
        )
        result.append(
            {
                "artifact_id": f"preparation:{book_id}",
                "kind": "preparation",
                "book_id": book_id,
                "candidate_ids": [],
                "source_words": 0,
                "source_words_by_book": {book_id: 0},
                "attempts": attempts,
                "relative": book.telemetry_path,
                "usage": book.usage,
            }
        )
    return result


def _artifact_payload(
    run_dir: Path,
    relative: str,
    artifact_id: str,
    kind: str,
    book_id: str | None,
    benchmark_id: str,
    corpus_sha: str,
    preparation_sha: str,
    run_ids: set[str],
) -> dict[str, Any]:
    rel = _safe_rel(relative, "artifact reference")
    path = run_dir / rel
    if not path.is_dir():
        _fail("referenced artifact missing")
    manifest = _json(path / "manifest.json")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != kind
    ):
        _fail("artifact manifest invalid")
    if (
        kind in {"translation", "editor"}
        and manifest.get("artifact_key") != artifact_id.split(":", 1)[-1]
    ):
        _fail("artifact key mismatch")
    if (
        manifest.get("corpus_sha256") != corpus_sha
        or manifest.get("preparation_sha256") != preparation_sha
    ):
        _fail("artifact provenance mismatch")
    if book_id is not None and manifest.get("book_id") not in {None, book_id}:
        _fail("artifact book mismatch")
    passages = path / "passages"
    if kind in {"translation", "editor"}:
        files = sorted(passages.glob("*.json")) if passages.exists() else []
        scope = manifest.get("scope")
        expected_passages: set[str] | None = None
        if isinstance(scope, dict) and isinstance(scope.get("passage_ids"), list):
            if any(
                not isinstance(value, str) or not value.strip() for value in scope["passage_ids"]
            ):
                _fail("passage scope identifiers are invalid")
            expected_passages = set(scope["passage_ids"])
        elif kind == "translation":
            _fail("translation manifest scope is invalid")
        if kind == "translation" and expected_passages is not None:
            expected_files = sorted(_safe_id(value) + ".json" for value in expected_passages)
            if expected_files != [file.name for file in files]:
                _fail("translation passage set mismatch")
        if not files:
            _fail("artifact passages are missing")
        for file in files:
            passage = _json(file)
            if (
                not isinstance(passage, dict)
                or passage.get("status") != "complete"
                or passage.get("artifact_key") != manifest.get("artifact_key")
            ):
                _fail("passage provenance is invalid")
            passage_id = passage.get("passage_id")
            if (
                not isinstance(passage_id, str)
                or not passage_id.strip()
                or _safe_id(passage_id) + ".json" != file.name
            ):
                _fail("passage identifier provenance is invalid")
            if expected_passages is not None and passage_id not in expected_passages:
                _fail("passage identifier is outside manifest scope")
            segments = passage.get("segments")
            if not isinstance(segments, list) or not segments:
                _fail("passage segments are invalid")
            for segment in segments:
                if (
                    not isinstance(segment, dict)
                    or not isinstance(segment.get("segment_id"), str)
                    or not isinstance(segment.get("source"), str)
                ):
                    _fail("passage segment provenance is invalid")
                if "source_hash" in passage:
                    expected_source = sha256_bytes(
                        canonical_json(
                            [
                                {"segment_id": x.get("segment_id"), "source": x.get("source")}
                                for x in segments
                            ]
                        ).encode()
                    )
                    if passage["source_hash"] != expected_source:
                        _fail("passage source hash mismatch")
    store_path = path
    if kind in {"raw", "branch"}:
        stores = [
            child for child in path.iterdir() if child.is_dir() and (child / "chapters_v2").is_dir()
        ]
        if not stores and not (path / "usage.json").exists():
            stores = [
                child
                for child in path.iterdir()
                if child.is_dir() and (child / "usage.json").exists()
            ]
        if len(stores) > 1:
            _fail("full artifact store layout is invalid")
        if stores:
            store_path = stores[0]
        elif not (path / "usage.json").exists():
            _fail("full artifact store layout is invalid")
    usage = _json(store_path / "usage.json")
    totals = _usage_totals(usage)
    attempts = _parse_telemetry(
        path / "telemetry.jsonl",
        artifact_id=artifact_id,
        benchmark_id=benchmark_id,
        expected_book=book_id,
        run_ids=run_ids,
    )
    known = dict.fromkeys(
        (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        ),
        0,
    )
    for item in attempts:
        if not item["telemetry"].billed_usage_unknown:
            _add_usage(known, item["telemetry"])
    if kind != "branch" and not any(
        item["telemetry"].billed_usage_unknown or item["telemetry"].status == "error"
        for item in attempts
    ):
        for key, value in known.items():
            if key in totals and totals[key] != value:
                _fail("usage and telemetry totals mismatch")
    segments = _source_records(path, kind=kind)
    source_unique = {(x.get("book_id"), x.get("segment_id")): x for x in segments}
    source_by_book = {
        book: sum(
            value["source_words"]
            for (book_key, _), value in source_unique.items()
            if book_key == book
        )
        for book in sorted({book for book, _ in source_unique})
        if book is not None
    }
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "book_id": book_id or manifest.get("book_id"),
        "candidate_ids": [],
        "source_words": sum(x["source_words"] for x in source_unique.values()),
        "source_words_by_book": source_by_book,
        "attempts": attempts,
        "segments": segments,
        "relative": relative,
        "usage": usage,
        "_store_path": store_path,
        "manifest": manifest,
    }


def _validate_full_stage_against_stores(
    row: dict[str, Any], raw_artifact: dict[str, Any], branch_artifact: dict[str, Any]
) -> list[dict[str, Any]]:
    raw_path = raw_artifact.get("_store_path")
    branch_path = branch_artifact.get("_store_path")
    if not isinstance(raw_path, Path) or not isinstance(branch_path, Path):
        _fail("full RunStore evidence is missing")
    try:
        _, raw_rows, _ = FullRunner._readonly_store(raw_path)
        branch_state, branch_rows, _ = FullRunner._readonly_store(branch_path)
        report = (
            _json(branch_path / "report.json") if (branch_path / "report.json").exists() else {}
        )
    except Exception:
        _fail("full RunStore evidence is invalid")
    raw_by_id = {value["segment_id"]: value for value in raw_rows}
    branch_by_id = {value["segment_id"]: value for value in branch_rows}
    stage = row.get("stage")
    if (
        not isinstance(stage, list)
        or {value.get("segment_id") for value in stage if isinstance(value, dict)} != set(raw_by_id)
        or set(raw_by_id) != set(branch_by_id)
    ):
        _fail("full stage segment set mismatch")
    consistency_issues: list[dict[str, Any]] = []
    node = branch_state.nodes.get("consistency_qa") if hasattr(branch_state, "nodes") else None
    if node is not None and isinstance(getattr(node, "output", None), dict):
        raw_issues = node.output.get("issues")
        if isinstance(raw_issues, list):
            consistency_issues.extend(issue for issue in raw_issues if isinstance(issue, dict))
    if isinstance(report, dict) and isinstance(report.get("consistency_issues"), list):
        consistency_issues.extend(
            issue for issue in report["consistency_issues"] if isinstance(issue, dict)
        )
    unique_issues: list[dict[str, Any]] = []
    seen_issues: set[str] = set()
    for issue in consistency_issues:
        identity = canonical_json(issue)
        if identity not in seen_issues:
            seen_issues.add(identity)
            unique_issues.append(issue)
    validated: list[dict[str, Any]] = []
    for index, original in enumerate(stage):
        if not isinstance(original, dict):
            _fail("full stage record is invalid")
        value = dict(original)
        segment_id = value.get("segment_id")
        if not isinstance(segment_id, str):
            _fail("full stage record is invalid")
        raw = raw_by_id[segment_id]
        branch = branch_by_id[segment_id]
        final = value.get("final_after_full_pipeline")
        if (
            value.get("source") != raw.get("source")
            or value.get("raw_after_translation_lint") != raw.get("final_after_full_pipeline")
            or final != branch.get("final_after_full_pipeline")
            or value.get("review_findings") != branch.get("review_findings")
            or value.get("lint_findings") != branch.get("lint_findings")
            or value.get("backtranslation_findings") != branch.get("backtranslation_findings")
            or not isinstance(final, str)
            or value.get("final_sha256") != sha256_bytes(final.encode())
        ):
            _fail("full stage provenance mismatch")
        matching = [dict(issue) for issue in unique_issues if issue.get("segment_id") == segment_id]
        book_level = [dict(issue) for issue in unique_issues if "segment_id" not in issue]
        if book_level and index == 0:
            for issue in book_level:
                issue["artifact_id"] = branch_artifact.get("artifact_id")
            matching.extend(book_level)
        value["consistency_findings"] = matching
        validated.append(value)
    return _full_stage_records({"book_id": row.get("book_id"), "stage": validated})


def _finalize_artifact(artifact: dict[str, Any], snapshot: PriceSnapshot) -> dict[str, Any]:
    """Attach deterministic quote and aggregate facts to one loaded artifact."""
    for item in artifact.get("attempts", []):
        attempt = item.get("telemetry")
        if not isinstance(attempt, CallAttemptTelemetry):
            _fail("artifact telemetry is invalid")
        _, quote, _ = _quote_attempt(snapshot, attempt)
        item["quote"] = quote
    aggregate = _aggregate([artifact], snapshot)
    artifact["_aggregate"] = aggregate
    artifact["_normalized_attempts"] = [
        _attempt_view(item["telemetry"], item.get("quote"), item.get("book_id"))
        for item in artifact.get("attempts", [])
    ]
    return artifact


def _candidate_words(
    artifacts: Sequence[dict[str, Any]], mode: str, rows: Sequence[dict[str, Any]] = ()
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    segments: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        for value in artifact.get("segments", []):
            key = (
                str(value.get("book_id") or artifact.get("book_id") or ""),
                str(value.get("segment_id")),
            )
            existing = segments.get(key)
            if existing is None or artifact["kind"] in {"editor", "branch"}:
                segments[key] = value
    return segments, {
        book: sum(v["source_words"] for (book, _), v in segments.items())
        for book in sorted({book for book, _ in segments})
    }


def _validate_attribution_artifacts(artifacts: Mapping[str, dict[str, Any]], bundle: Any) -> None:
    translations = {
        artifact.get("manifest", {}).get("artifact_key"): artifact
        for artifact in artifacts.values()
        if artifact.get("kind") == "translation"
    }
    for artifact in artifacts.values():
        if artifact.get("kind") not in {"translation", "editor"}:
            continue
        editor = artifact["kind"] == "editor"
        try:
            for segment in artifact.get("segments", []):
                _validate_segment_hashes(segment, editor=editor)
        except BenchmarkError as exc:
            _fail(str(exc))
        if not editor:
            continue
        manifest = artifact.get("manifest", {})
        translation_key = manifest.get("translation_artifact_key")
        translation = translations.get(translation_key)
        if not isinstance(translation_key, str) or translation is None:
            _fail("editor translation provenance mismatch")
        editor_files = {segment.get("_passage_file") for segment in artifact.get("segments", [])}
        translation_files = {
            segment.get("_passage_file") for segment in translation.get("segments", [])
        }
        editor_keys = [
            (segment.get("_passage_file"), segment.get("segment_id"))
            for segment in artifact.get("segments", [])
        ]
        translation_keys = [
            (segment.get("_passage_file"), segment.get("segment_id"))
            for segment in translation.get("segments", [])
        ]
        if (
            editor_files != translation_files
            or len(editor_keys) != len(set(editor_keys))
            or len(translation_keys) != len(set(translation_keys))
        ):
            _fail("editor translation passage set mismatch")
        baselines = {
            (segment.get("_passage_file"), segment.get("segment_id")): segment
            for segment in translation.get("segments", [])
        }
        for segment in artifact.get("segments", []):
            baseline = baselines.get((segment.get("_passage_file"), segment.get("segment_id")))
            if baseline is None:
                _fail("editor translation segment provenance mismatch")
            for field in (
                "segment_id",
                "source",
                "translation_raw",
                "translation_after_lint",
                "translation_lint_issues",
                "translation_raw_sha256",
                "translation_after_lint_sha256",
            ):
                if segment.get(field) != baseline.get(field):
                    _fail("editor translation baseline mismatch")
            book_id = segment.get("book_id")
            if not isinstance(book_id, str) or book_id not in bundle.books:
                _fail("editor passage book provenance mismatch")
            prep = bundle.books[book_id]
            try:
                gate = lint.polish_gate(
                    segment["source"],
                    segment["translation_raw"],
                    segment["polish_proposal"],
                    locked_terms=_locked_terms(_glossary(prep.glossary)),
                    src_lang="en",
                )
            except Exception as exc:
                _fail(f"editor polish gate validation failed: {exc}")
            if (
                segment.get("polish_proposal") != gate.proposal
                or segment.get("polish_accepted") != gate.accepted
                or segment.get("polish_rejection_reasons") != list(gate.rejection_reasons)
                or segment.get("final") != gate.selected
            ):
                _fail("editor polish gate evidence mismatch")


def _normalize(
    artifacts: Sequence[dict[str, Any]],
    run: dict[str, Any],
    spec: ReportSpec,
    price_hash: str,
    candidate_artifacts: Mapping[str, Sequence[str]],
    candidate_words: Mapping[str, Mapping[str, int]],
    physical_ids: Sequence[str],
    prep_ids: Mapping[str, str],
    system: Mapping[str, Any],
    initial_insufficient_data: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    values = []
    for artifact in sorted(artifacts, key=lambda x: x["artifact_id"]):
        attempts = []
        for item in sorted(
            artifact["attempts"],
            key=lambda x: (x["telemetry"].logical_call_id, x["telemetry"].attempt_index),
        ):
            attempts.append(
                _attempt_view(item["telemetry"], item.get("quote"), item.get("book_id"))
            )
        values.append(
            {
                "artifact_id": artifact["artifact_id"],
                "kind": artifact["kind"],
                "book_id": artifact.get("book_id"),
                "candidate_ids": sorted(artifact.get("candidate_ids", [])),
                "source_words": int(artifact.get("source_words", 0)),
                "source_words_by_book": {
                    k: int(v) for k, v in sorted(artifact.get("source_words_by_book", {}).items())
                },
                "attempts": attempts,
            }
        )
    return {
        "schema_version": 1,
        "run_mode": run["run_mode"],
        "benchmark_id": spec.benchmark_id,
        "run_hash": spec.run_hash,
        "preparation_sha256": spec.preparation_sha256,
        "original_price_snapshot_sha256": spec.price_snapshot_sha256,
        "price_snapshot_sha256": price_hash,
        "artifacts": values,
        "candidate_artifact_ids": {k: sorted(v) for k, v in sorted(candidate_artifacts.items())},
        "physical_artifact_ids": sorted(physical_ids),
        "preparation_artifact_ids_by_book": dict(sorted(prep_ids.items())),
        "candidate_source_words": {
            k: int(v)
            for k, v in sorted(
                {
                    k: sum(x.values()) if isinstance(x, Mapping) else int(x)
                    for k, x in candidate_words.items()
                }.items()
            )
        },
        "candidate_source_words_by_book": {
            k: {b: int(w) for b, w in sorted(v.items())} for k, v in sorted(candidate_words.items())
        },
        "system_facts": system,
        "initial_insufficient_data": [dict(x) for x in initial_insufficient_data],
        "bootstrap_seed": spec.bootstrap_seed,
        "bootstrap_replicates": spec.bootstrap_replicates,
    }


def _validate_normalized(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        _fail("normalized pricing facts schema mismatch")
    required = {
        "schema_version",
        "run_mode",
        "benchmark_id",
        "run_hash",
        "preparation_sha256",
        "original_price_snapshot_sha256",
        "price_snapshot_sha256",
        "artifacts",
        "candidate_artifact_ids",
        "physical_artifact_ids",
        "preparation_artifact_ids_by_book",
        "candidate_source_words",
        "candidate_source_words_by_book",
        "system_facts",
        "initial_insufficient_data",
        "bootstrap_seed",
        "bootstrap_replicates",
    }
    if set(value) != required:
        _fail("normalized pricing facts keys mismatch")
    initial = value["initial_insufficient_data"]
    if not isinstance(initial, list) or any(
        not isinstance(item, dict)
        or set(item) != {"scope", "reason"}
        or not isinstance(item["scope"], str)
        or not isinstance(item["reason"], str)
        for item in initial
    ):
        _fail("normalized initial insufficiency facts invalid")
    if value["run_mode"] not in {"attribution", "full"}:
        _fail("normalized run mode invalid")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list):
        _fail("normalized artifact facts invalid")
    allowed_kinds = {"preparation", "translation", "editor", "raw", "branch"}
    by_id: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        artifact_keys = {
            "artifact_id",
            "kind",
            "book_id",
            "candidate_ids",
            "source_words",
            "source_words_by_book",
            "attempts",
        }
        if not isinstance(artifact, dict) or set(artifact) != artifact_keys:
            _fail("normalized artifact keys mismatch")
        aid = artifact["artifact_id"]
        book_id = artifact["book_id"]
        candidate_ids = artifact["candidate_ids"]
        words_by_book = artifact["source_words_by_book"]
        if (
            not isinstance(aid, str)
            or not aid
            or aid in by_id
            or artifact["kind"] not in allowed_kinds
            or (book_id is not None and not isinstance(book_id, str))
            or not isinstance(candidate_ids, list)
            or any(not isinstance(x, str) for x in candidate_ids)
            or len(candidate_ids) != len(set(candidate_ids))
            or not isinstance(words_by_book, dict)
            or any(
                not isinstance(k, str) or not isinstance(v, int) or isinstance(v, bool) or v < 0
                for k, v in words_by_book.items()
            )
            or not isinstance(artifact["attempts"], list)
        ):
            _fail("normalized artifact identity invalid")
        _int(artifact["source_words"], "normalized source_words")
        if artifact["source_words"] != sum(words_by_book.values()):
            _fail("normalized artifact source word sum mismatch")
        by_id[aid] = artifact
        for attempt in artifact["attempts"]:
            allowed = {
                "logical_call_id",
                "attempt_index",
                "started_at",
                "elapsed_ms",
                "agent",
                "operation",
                "status",
                "retry_class",
                "http_status",
                "requested_model",
                "resolved_model",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "reasoning_tokens",
                "billed_usage_unknown",
                "quote",
                "book_id",
            }
            if not isinstance(attempt, dict) or not set(attempt) <= allowed:
                _fail("normalized attempt contains forbidden fields")
            if any(
                not isinstance(attempt.get(x), str) or not attempt[x]
                for x in (
                    "logical_call_id",
                    "started_at",
                    "agent",
                    "operation",
                    "status",
                    "requested_model",
                )
            ):
                _fail("normalized attempt identity invalid")
            if "book_id" in attempt and (
                attempt["book_id"] is not None and not isinstance(attempt["book_id"], str)
            ):
                _fail("normalized attempt book invalid")
            _int(attempt.get("attempt_index"), "normalized attempt_index", minimum=1)
            _int(attempt.get("elapsed_ms"), "normalized elapsed_ms")
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cache_hit_tokens",
                "cache_miss_tokens",
                "reasoning_tokens",
            ):
                _int(attempt.get(name), f"normalized {name}")
            if not isinstance(attempt.get("billed_usage_unknown"), bool):
                _fail("normalized billed usage flag invalid")
            if (
                "quote" in attempt
                and attempt["quote"] is not None
                and not isinstance(attempt["quote"], dict)
            ):
                _fail("normalized quote invalid")
    candidate_refs = value["candidate_artifact_ids"]
    if not isinstance(candidate_refs, dict):
        _fail("normalized candidate refs invalid")
    for candidate, ids in candidate_refs.items():
        if (
            not isinstance(candidate, str)
            or not isinstance(ids, list)
            or len(ids) != len(set(ids))
            or any(not isinstance(x, str) for x in ids)
        ):
            _fail("normalized candidate artifact refs invalid")
        for aid in ids:
            if aid not in by_id or candidate not in by_id[aid]["candidate_ids"]:
                _fail("normalized candidate artifact ref dangling")
    physical = value["physical_artifact_ids"]
    if (
        not isinstance(physical, list)
        or len(physical) != len(set(physical))
        or any(not isinstance(x, str) or x not in by_id for x in physical)
    ):
        _fail("normalized physical artifact ref dangling")
    prep_refs = value["preparation_artifact_ids_by_book"]
    if not isinstance(prep_refs, dict):
        _fail("normalized preparation refs invalid")
    for book, aid in prep_refs.items():
        if (
            not isinstance(book, str)
            or not isinstance(aid, str)
            or aid not in by_id
            or by_id[aid]["kind"] != "preparation"
            or by_id[aid]["book_id"] != book
        ):
            _fail("normalized preparation ref dangling")
    words = value["candidate_source_words"]
    words_by_candidate = value["candidate_source_words_by_book"]
    if (
        not isinstance(words, dict)
        or not isinstance(words_by_candidate, dict)
        or set(words) != set(candidate_refs)
        or set(words_by_candidate) != set(candidate_refs)
    ):
        _fail("normalized source word refs invalid")
    for candidate in candidate_refs:
        _int(words[candidate], "normalized candidate source words")
        if not isinstance(words_by_candidate[candidate], dict):
            _fail("normalized candidate book words invalid")
        for book, count in words_by_candidate[candidate].items():
            if not isinstance(book, str):
                _fail("normalized candidate book ref invalid")
            _int(count, "normalized candidate book words")
        if words[candidate] != sum(words_by_candidate[candidate].values()):
            _fail("normalized candidate source word sum mismatch")
        owned: dict[str, int] = defaultdict(int)
        for aid in candidate_refs[candidate]:
            for book, count in by_id[aid]["source_words_by_book"].items():
                owned[book] += count
        for book, count in words_by_candidate[candidate].items():
            if count > owned.get(book, 0):
                _fail("normalized candidate source ownership mismatch")
    return value


def _reprice_from_facts(
    facts: dict[str, Any], snapshot: PriceSnapshot, human: dict[str, Any], rates: Sequence[Decimal]
) -> dict[str, Any]:
    artifacts = []
    for raw in facts["artifacts"]:
        attempts = []
        for value in raw["attempts"]:
            try:
                telemetry = CallAttemptTelemetry.model_validate(
                    {
                        "schema_version": 1,
                        "logical_call_id": value["logical_call_id"],
                        "attempt_index": value["attempt_index"],
                        "started_at": value["started_at"],
                        "elapsed_ms": value["elapsed_ms"],
                        "stage": None,
                        "agent": value["agent"],
                        "operation": value["operation"],
                        "provider": "normalized",
                        "requested_model": value["requested_model"],
                        "resolved_model": value.get("resolved_model"),
                        "reasoning_enabled": False,
                        "reasoning_effort": None,
                        "temperature": None,
                        "seed": None,
                        "json_mode": False,
                        "max_tokens": None,
                        "status": value["status"],
                        "retry_class": value.get("retry_class"),
                        "http_status": value.get("http_status"),
                        "finish_reason": None,
                        "response_id": None,
                        "prompt_tokens": value["prompt_tokens"],
                        "completion_tokens": value["completion_tokens"],
                        "total_tokens": value["total_tokens"],
                        "cache_hit_tokens": value["cache_hit_tokens"],
                        "cache_miss_tokens": value["cache_miss_tokens"],
                        "reasoning_tokens": value["reasoning_tokens"],
                        "billed_usage_unknown": value["billed_usage_unknown"],
                        "request_sha256": "0" * 64,
                        "response_sha256": None,
                    }
                )
            except Exception:
                _fail("normalized attempt is invalid")
            _, quote, _ = _quote_attempt(snapshot, telemetry)
            attempts.append(
                {"telemetry": telemetry, "quote": quote, "book_id": value.get("book_id")}
            )
        artifacts.append({**raw, "attempts": attempts})
    by_id = {x["artifact_id"]: x for x in artifacts}
    physical_ids = set(facts["physical_artifact_ids"])
    physical = [x for x in artifacts if x["artifact_id"] in physical_ids]
    candidates = {}
    for candidate, ids in facts["candidate_artifact_ids"].items():
        selected = [by_id[x] for x in ids]
        words = int(facts["candidate_source_words"].get(candidate, 0))
        candidates[candidate] = _serial_aggregate(_aggregate(selected, snapshot), words=words)
    physical_cost = _serial_aggregate(
        _aggregate(physical, snapshot), words=sum(int(x.get("source_words", 0)) for x in physical)
    )
    output = {
        "schema_version": 1,
        "input_hashes": {
            "run_hash": facts["run_hash"],
            "preparation_sha256": facts["preparation_sha256"],
            "price_snapshot_sha256": _price_hash(snapshot),
            "human": human.get("input_hashes", {}),
        },
        "candidate_costs": candidates,
        "physical_spend": physical_cost,
        "million_word_estimate": _million_estimates(
            facts["candidate_artifact_ids"],
            by_id,
            facts["candidate_source_words_by_book"],
            snapshot,
            seed=facts["bootstrap_seed"],
            replicates=facts["bootstrap_replicates"],
        ),
        "effective_costs": {},
        "system_metrics": facts["system_facts"],
        "normalized_pricing_facts": {**facts, "price_snapshot_sha256": _price_hash(snapshot)},
    }
    _effective(output, candidates, human, rates)
    output["insufficient_data"] = [
        dict(item) for item in facts.get("initial_insufficient_data", [])
    ]
    for cid, cost in candidates.items():
        if cost["unknown_count"]:
            output["insufficient_data"].append(
                {"scope": f"candidate={cid}", "reason": "unknown_cost"}
            )
        if not cost["validated_source_words"]:
            output["insufficient_data"].append(
                {"scope": f"candidate={cid}", "reason": "zero_source_words"}
            )
        if not cost.get("book_attribution_complete", True):
            output["insufficient_data"].append(
                {"scope": f"candidate={cid}", "reason": "missing_book_cost_attribution"}
            )
    output["insufficient_data"] = [
        {"scope": scope, "reason": reason}
        for scope, reason in sorted(
            {(item["scope"], item["reason"]) for item in output["insufficient_data"]}
        )
    ]
    return _decimalize(output)


def _effective(
    output: dict[str, Any],
    candidates: Mapping[str, dict[str, Any]],
    human: Mapping[str, Any],
    rates: Sequence[Decimal],
) -> None:
    post = human.get("postedit", {}) if isinstance(human, Mapping) else {}
    for cid, cost in candidates.items():
        output["effective_costs"].setdefault(cid, {})
        surfaces = post.get(cid, {}) if isinstance(post, Mapping) else {}
        for rate in rates:
            key = _rate_key(rate)
            entries = {}
            for surface, value in sorted(surfaces.items()):
                macro = value.get("macro", {}) if isinstance(value, Mapping) else {}
                minutes = macro.get("minutes_per_10k")
                if minutes is None:
                    entries[surface] = {"value": None, "reason": "missing_postedit"}
                    continue
                mins = _dec(minutes, "postedit minutes")
                if mins < 0:
                    _fail("postedit minutes must be nonnegative")
                words = Decimal(cost["validated_source_words"])
                projected = mins * words / Decimal(10000)
                api = cost["api_cost"]
                lower = (
                    _dec(cost["api_cost_lower_bound"], "api lower bound")
                    + projected / Decimal(60) * rate
                )
                entries[surface] = {
                    "value": _money(_dec(api, "api cost") + projected / Decimal(60) * rate)
                    if api is not None
                    else None,
                    "lower_bound": _money(lower),
                    "projected_minutes": _money(projected),
                    "minutes_per_10k": _money(mins),
                    "reason": None if api is not None else "unknown_cost",
                }
            output["effective_costs"][cid][key] = entries


def _million_estimates(
    candidate_artifact_ids: Mapping[str, Sequence[str]],
    artifacts_by_id: Mapping[str, dict[str, Any]],
    source_by_book: Mapping[str, Mapping[str, int]],
    snapshot: PriceSnapshot,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate, ids in sorted(candidate_artifact_ids.items()):
        by_book: dict[str, dict[str, Any]] = {}
        estimates: list[Decimal] = []
        invalid_multi_book = False
        for aid in ids:
            artifact = artifacts_by_id[aid]
            source_map = artifact.get("source_words_by_book", {})
            if len(source_map) > 1:
                attempts = artifact.get("attempts", [])
                if not attempts or any(
                    not isinstance(item.get("book_id"), str)
                    or item.get("book_id") not in source_map
                    for item in attempts
                ):
                    invalid_multi_book = True
        candidate_book_attribution_complete = not invalid_multi_book
        complete = candidate_book_attribution_complete
        for book, words in sorted(source_by_book.get(candidate, {}).items()):
            prep = [artifacts_by_id[x] for x in ids if x == f"preparation:{book}"]
            variable: list[dict[str, Any]] = []
            for aid in ids:
                if aid == f"preparation:{book}":
                    continue
                artifact = artifacts_by_id[aid]
                source_map = artifact.get("source_words_by_book", {})
                if artifact.get("book_id") != book and book not in source_map:
                    continue
                if len(source_map) > 1:
                    attempts = [
                        item for item in artifact.get("attempts", []) if item.get("book_id") == book
                    ]
                    if candidate_book_attribution_complete and attempts:
                        variable.append({**artifact, "attempts": attempts})
                elif len(source_map) == 1:
                    if book in source_map:
                        variable.append(artifact)
                elif artifact.get("book_id") == book:
                    variable.append(artifact)
            fixed_agg = _aggregate(prep, snapshot) if prep else _empty_aggregate()
            fixed = fixed_agg["lower_bound"]
            if candidate_book_attribution_complete:
                variable_agg = _aggregate(variable, snapshot)
                variable_cost: Decimal | None = variable_agg["lower_bound"]
                attribution_complete = bool(
                    fixed_agg.get("book_attribution_complete", True)
                    and variable_agg.get("book_attribution_complete", True)
                )
                point = (
                    fixed + (variable_cost * Decimal(1_000_000) / Decimal(words))
                    if words and attribution_complete
                    else None
                )
                book_complete = bool(
                    words
                    and fixed_agg["complete"]
                    and variable_agg["complete"]
                    and attribution_complete
                )
            else:
                variable_cost = None
                point = None
                book_complete = False
            by_book[book] = {
                "source_words": words,
                "fixed_preparation_lower_bound": _money(fixed),
                "variable_lower_bound": _money(variable_cost),
                "estimate_lower_bound": _money(point),
                "complete": book_complete,
            }
            if point is not None:
                estimates.append(point)
            complete = complete and book_complete
        lower = (
            sum(
                (
                    Decimal(x["estimate_lower_bound"])
                    for x in by_book.values()
                    if x["estimate_lower_bound"] is not None
                ),
                Decimal(0),
            )
            / Decimal(len(by_book))
            if by_book and candidate_book_attribution_complete
            else None
        )
        if complete and estimates:
            rng = random.Random(seed)
            samples = [
                sum((estimates[rng.randrange(len(estimates))] for _ in estimates), Decimal(0))
                / Decimal(len(estimates))
                for _ in range(replicates)
            ]
            samples.sort()
            low = samples[max(0, math.ceil(len(samples) * 0.025) - 1)]
            high = samples[min(len(samples) - 1, math.ceil(len(samples) * 0.975) - 1)]
            value = sum(estimates, Decimal(0)) / Decimal(len(estimates))
        else:
            low = high = value = None
        result[candidate] = {
            "value": _money(value),
            "lower_bound": _money(lower),
            "lower95": _money(low),
            "upper95": _money(high),
            "by_book": by_book,
            "complete": complete,
            "assumption": "equal-book mean; fixed preparation plus variable artifact cost per validated source word",
        }
    return result


def analyze_cost_system(
    corpus_dir: Path,
    run_dir: Path,
    preparation_dir: Path,
    price_path: Path,
    human_facts: dict[str, Any],
    spec: ReportSpec,
) -> dict[str, Any]:
    try:
        corpus_dir = Path(corpus_dir)
        run_dir = Path(run_dir)
        preparation_dir = Path(preparation_dir)
        price_path = Path(price_path)
        try:
            corpus_facts = validate_corpus(corpus_dir)
        except Exception:
            _fail("corpus validation failed")
        if corpus_facts.get("corpus_sha256") != spec.corpus_sha256:
            _fail("corpus hash mismatch")
        snapshot = load_price_snapshot(price_path)
        if snapshot.currency != "CNY":
            _fail("price currency must be CNY")
        if _price_hash(snapshot) != spec.price_snapshot_sha256:
            _fail("price snapshot hash mismatch")
        try:
            validate_preparation(preparation_dir)
            bundle, preparation_sha = load_preparation_bundle(preparation_dir)
        except (BenchmarkError, Exception) as exc:
            if isinstance(exc, CostAnalysisError):
                raise
            _fail("preparation validation failed")
        if preparation_sha != spec.preparation_sha256:
            _fail("preparation hash mismatch")
        if bundle.corpus_sha256 != spec.corpus_sha256:
            _fail("preparation corpus hash mismatch")
        mode, run = _validate_run(run_dir, spec, preparation_sha, bundle.corpus_sha256)
        run_ids: set[str] = set()
        prep = _prepare_artifacts(preparation_dir, bundle, spec.benchmark_id, run_ids)
        artifacts_by_id: dict[str, dict[str, Any]] = {x["artifact_id"]: x for x in prep}
        candidate_artifact_ids: dict[str, list[str]] = defaultdict(list)
        candidate_segments: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
        rows = _json(run_dir / "candidates.json")
        if not isinstance(rows, list):
            _fail("candidate manifest is invalid")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("candidate_id"), str):
                _fail("candidate row is invalid")
            cid = row["candidate_id"]
            refs: list[tuple[str, str, str | None]] = []
            if mode == "attribution":
                for field in ("translation_artifacts", "editor_artifacts"):
                    mapping = row.get(field, {})
                    if not isinstance(mapping, dict):
                        _fail("candidate artifact mapping invalid")
                    for relative in mapping.values():
                        rel = _safe_rel(relative, "candidate artifact path")
                        aid = rel.name
                        refs.append((aid, str(rel), None))
            else:
                for field, kind in (("raw_artifact_id", "raw"), ("branch_artifact_id", "branch")):
                    aid = row.get(field)
                    if not isinstance(aid, str):
                        _fail("full candidate artifact id missing")
                    refs.append(
                        (
                            aid,
                            f"{('raw' if kind == 'raw' else 'branches')}/{aid}",
                            row.get("book_id"),
                        )
                    )
            for raw_id, relative, book in refs:
                kind = (
                    "editor"
                    if "_shared" in relative
                    else (
                        "translation"
                        if "translation/" in relative
                        else ("raw" if relative.startswith("raw/") else "branch")
                    )
                )
                aid = f"{kind}:{raw_id}"
                artifact = artifacts_by_id.get(aid)
                if artifact is None:
                    artifact = _artifact_payload(
                        run_dir,
                        relative,
                        aid,
                        kind,
                        book,
                        spec.benchmark_id,
                        spec.corpus_sha256,
                        preparation_sha,
                        run_ids,
                    )
                    artifacts_by_id[aid] = artifact
                if mode == "full":
                    expected_telemetry_hash = (
                        row.get("raw_telemetry_sha256")
                        if kind == "raw"
                        else row.get("branch_telemetry_sha256")
                    )
                    telemetry_path = (
                        run_dir / _safe_rel(relative, "artifact reference") / "telemetry.jsonl"
                    )
                    if not isinstance(
                        expected_telemetry_hash, str
                    ) or expected_telemetry_hash != sha256_bytes(telemetry_path.read_bytes()):
                        _fail("full telemetry hash mismatch")
                    allocated = row.get("allocated_usage")
                    if not isinstance(allocated, dict) or set(allocated) != {
                        "preparation",
                        "raw",
                        "branch_increment",
                    }:
                        _fail("full usage allocation is invalid")
                    artifact.setdefault("_store_usage", artifact["usage"])
                    if kind == "raw" and allocated["raw"] != artifact["_store_usage"]:
                        _fail("raw allocated usage mismatch")
                    if kind == "branch":
                        raw_artifact = artifacts_by_id.get(f"raw:{row.get('raw_artifact_id')}")
                        if raw_artifact is None:
                            _fail("branch raw artifact is missing")
                        expected_increment = usage_delta(
                            artifact["_store_usage"],
                            raw_artifact.get("_store_usage", raw_artifact["usage"]),
                        )
                        if allocated["branch_increment"] != expected_increment:
                            _fail("branch allocated usage mismatch")
                        artifact["usage"] = allocated["branch_increment"]
                        branch_totals = _usage_totals(artifact["usage"])
                        known_increment = dict.fromkeys(
                            (
                                "prompt_tokens",
                                "completion_tokens",
                                "total_tokens",
                                "cache_hit_tokens",
                                "cache_miss_tokens",
                            ),
                            0,
                        )
                        for item in artifact["attempts"]:
                            if (
                                not item["telemetry"].billed_usage_unknown
                                and item["telemetry"].status != "error"
                            ):
                                _add_usage(known_increment, item["telemetry"])
                        for key, value in known_increment.items():
                            if key in branch_totals and branch_totals[key] != value:
                                _fail("branch increment usage and telemetry mismatch")
                if mode == "full" and kind == "branch":
                    raw_artifact = artifacts_by_id.get(f"raw:{row.get('raw_artifact_id')}")
                    if raw_artifact is None:
                        _fail("full raw artifact is missing")
                    stage_segments = _validate_full_stage_against_stores(
                        row, raw_artifact, artifact
                    )
                    artifact["segments"] = stage_segments
                    artifact["source_words"] = sum(x["source_words"] for x in stage_segments)
                    artifact["source_words_by_book"] = {
                        row.get("book_id"): sum(x["source_words"] for x in stage_segments)
                    }
                if cid not in artifact["candidate_ids"]:
                    artifact["candidate_ids"].append(cid)
                if aid not in candidate_artifact_ids[cid]:
                    candidate_artifact_ids[cid].append(aid)
                for segment in artifact.get("segments", []):
                    key = (
                        str(segment.get("book_id") or book or ""),
                        str(segment.get("segment_id")),
                    )
                    if (
                        artifact["kind"] in {"editor", "branch"}
                        or key not in candidate_segments[cid]
                    ):
                        candidate_segments[cid][key] = segment
            if mode == "full":
                replicate = row.get("replicate")
                book_id = row.get("book_id")
                if (
                    not isinstance(book_id, str)
                    or book_id not in bundle.books
                    or not isinstance(replicate, int)
                ):
                    _fail("full candidate row identity is invalid")
                expected_allocation_id = FullRunner._preparation_allocation_id(
                    cid, book_id, preparation_sha
                )
                if row.get("preparation_allocation_id") != expected_allocation_id:
                    _fail("full preparation allocation id mismatch")
                expected_preparation = bundle.books[book_id].usage if replicate == 1 else {}
                if row["allocated_usage"]["preparation"] != expected_preparation:
                    _fail("full preparation allocation mismatch")
            for book in bundle.books:
                prep_id = f"preparation:{book}"
                if prep_id not in candidate_artifact_ids[cid]:
                    candidate_artifact_ids[cid].append(prep_id)
                    artifacts_by_id[prep_id]["candidate_ids"].append(cid)
        if mode == "attribution":
            _validate_attribution_artifacts(artifacts_by_id, bundle)
        known_candidates = set(candidate_artifact_ids)
        if len(run_ids) > 1:
            _fail("telemetry run context mismatch")
        for artifact in artifacts_by_id.values():
            for item in artifact.get("attempts", []):
                context_candidate = item.get("candidate_id")
                if context_candidate not in known_candidates and context_candidate not in {
                    "preparation",
                    "producer",
                    "shared",
                    "system",
                }:
                    _fail("telemetry candidate context mismatch")
        for artifact in artifacts_by_id.values():
            _finalize_artifact(artifact, snapshot)
        if mode == "attribution":
            try:
                corpus_rows = [
                    json.loads(line)
                    for line in (corpus_dir / "runner_segments.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line.strip()
                ]
            except Exception:
                _fail("corpus runner evidence is invalid")
            corpus_by_passage: dict[str, tuple[str, list[tuple[str, str]]]] = {}
            for corpus_row in corpus_rows:
                if (
                    not isinstance(corpus_row, dict)
                    or not isinstance(corpus_row.get("passage_id"), str)
                    or not isinstance(corpus_row.get("book_id"), str)
                ):
                    _fail("corpus runner evidence is invalid")
                corpus_segments = corpus_row.get("segments")
                if not isinstance(corpus_segments, list):
                    _fail("corpus runner evidence is invalid")
                expected = [
                    (segment.get("segment_id"), segment.get("source"))
                    for segment in corpus_segments
                    if isinstance(segment, dict)
                ]
                if len(expected) != len(corpus_segments) or any(
                    not isinstance(sid, str) or not isinstance(source, str)
                    for sid, source in expected
                ):
                    _fail("corpus runner evidence is invalid")
                if corpus_row["passage_id"] in corpus_by_passage:
                    _fail("duplicate corpus passage evidence")
                corpus_by_passage[corpus_row["passage_id"]] = (
                    corpus_row["book_id"],
                    sorted(expected),
                )
            artifact_groups: dict[str, dict[str, list[tuple[str, str]]]] = {}
            for artifact in artifacts_by_id.values():
                if artifact["kind"] not in {"translation", "editor"}:
                    continue
                groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for segment in artifact.get("segments", []):
                    groups[str(segment.get("passage_id"))].append(
                        (str(segment.get("segment_id")), str(segment.get("source")))
                    )
                artifact_groups[artifact["artifact_id"]] = groups
                for passage_id, actual in groups.items():
                    expected = corpus_by_passage.get(passage_id)
                    if (
                        expected is None
                        or len(actual) != len(set(actual))
                        or sorted(actual) != expected[1]
                    ):
                        _fail("attribution passage source provenance mismatch")
                    if any(
                        segment.get("book_id") != expected[0]
                        for segment in artifact["segments"]
                        if segment.get("passage_id") == passage_id
                    ):
                        _fail("attribution passage book provenance mismatch")
                if artifact["kind"] == "editor":
                    translation_key = artifact.get("manifest", {}).get("translation_artifact_key")
                    translation = (
                        artifact_groups.get(f"translation:{translation_key}")
                        if isinstance(translation_key, str)
                        else None
                    )
                    if translation is None or groups != translation:
                        _fail("editor translation provenance mismatch")
        expected_actual: dict[str, Any] = {}
        if mode == "full":
            seen_allocations: set[str] = set()
            for row in rows:
                allocation_id = row["preparation_allocation_id"]
                if row.get("replicate") == 1:
                    if allocation_id in seen_allocations:
                        _fail("duplicate preparation allocation in actual usage")
                    seen_allocations.add(allocation_id)
                    expected_actual = merge_usage_summaries(
                        expected_actual, row["allocated_usage"]["preparation"]
                    )
        for artifact in artifacts_by_id.values():
            if not artifact["artifact_id"].startswith("preparation:"):
                expected_actual = merge_usage_summaries(expected_actual, artifact["usage"])
        actual_path = run_dir / "actual_usage.json"
        if not actual_path.exists() or _json(actual_path) != expected_actual:
            _fail("actual benchmark usage integrity failure")
        if mode == "attribution":
            discovered = {
                f"translation:{path.parent.name}"
                for path in (run_dir / "translation").glob("*/manifest.json")
            }
            discovered |= {
                f"editor:{path.parent.name}"
                for path in (run_dir / "candidates" / "_shared").glob("*/manifest.json")
            }
            expected = {
                artifact["artifact_id"]
                for artifact in artifacts_by_id.values()
                if not artifact["artifact_id"].startswith("preparation:")
            }
            if discovered != expected:
                _fail("unreferenced or missing run artifact")
        else:
            discovered = {
                f"raw:{path.parent.name}" for path in (run_dir / "raw").glob("*/manifest.json")
            }
            discovered |= {
                f"branch:{path.parent.name}"
                for path in (run_dir / "branches").glob("*/manifest.json")
            }
            expected = {
                artifact["artifact_id"]
                for artifact in artifacts_by_id.values()
                if not artifact["artifact_id"].startswith("preparation:")
            }
            if discovered != expected:
                _fail("unreferenced or missing full artifact")
        candidates = {}
        system = {}
        source_by_book = {}
        book_attribution_insufficient: set[tuple[str, str]] = set()
        for cid in sorted(candidate_artifact_ids):
            selected = [artifacts_by_id[aid] for aid in candidate_artifact_ids[cid]]
            segments = candidate_segments[cid]
            words_by_book = {
                book: sum(v["source_words"] for (b, _), v in segments.items() if b == book)
                for book in sorted({b for b, _ in segments})
            }
            source_by_book[cid] = words_by_book
            agg = _aggregate(selected, snapshot)
            candidate_cost = _serial_aggregate(agg, words=sum(words_by_book.values()))
            for book, book_cost in candidate_cost["by_book"].items():
                if book in words_by_book:
                    book_cost["validated_source_words"] = words_by_book[book]
                    if book_cost["cost_complete"] and words_by_book[book]:
                        amount = Decimal(book_cost["api_cost_lower_bound"])
                        book_cost["cost_per_100k"] = _money(
                            amount * Decimal(100000) / Decimal(words_by_book[book])
                        )
                        book_cost["validated_words_per_cny"] = (
                            _money(Decimal(words_by_book[book]) / amount) if amount else None
                        )
            candidates[cid] = candidate_cost
            metric = _metrics(selected, segments)
            metric["completion_rate"] = _decimalize(metric["completion_rate"])
            metric["by_book"] = {}
            for book in sorted({v["book_id"] for v in segments.values()}):
                book_artifacts = []
                attribution_complete = True
                for artifact in selected:
                    source_map = artifact.get("source_words_by_book", {})
                    filtered_segments = [
                        segment
                        for segment in artifact.get("segments", [])
                        if isinstance(segment, dict) and segment.get("book_id") == book
                    ]
                    if len(source_map) > 1:
                        attempts = artifact.get("attempts", [])
                        valid = bool(attempts) and all(
                            isinstance(item.get("book_id"), str)
                            and item.get("book_id") in source_map
                            for item in attempts
                        )
                        if not valid:
                            attribution_complete = False
                            continue
                        selected_attempts = [
                            item for item in attempts if item.get("book_id") == book
                        ]
                        if selected_attempts:
                            book_artifacts.append(
                                {
                                    **artifact,
                                    "attempts": selected_attempts,
                                    "segments": filtered_segments,
                                }
                            )
                    elif len(source_map) == 1:
                        if book in source_map:
                            book_artifacts.append({**artifact, "segments": filtered_segments})
                    elif artifact.get("book_id") == book:
                        book_artifacts.append({**artifact, "segments": filtered_segments})
                book_metric = _metrics(
                    book_artifacts, {(b, s): v for (b, s), v in segments.items() if b == book}
                )
                book_metric["book_attribution_complete"] = attribution_complete
                if not attribution_complete:
                    book_attribution_insufficient.add((cid, book))
                metric["by_book"][book] = book_metric
            system[cid] = _decimalize(metric)
        physical = list(artifacts_by_id.values())
        physical_cost = _serial_aggregate(
            _aggregate(physical, snapshot), words=sum(a.get("source_words", 0) for a in physical)
        )
        million = _million_estimates(
            candidate_artifact_ids,
            artifacts_by_id,
            source_by_book,
            snapshot,
            seed=spec.bootstrap_seed,
            replicates=spec.bootstrap_replicates,
        )
        initial_insufficient: list[dict[str, Any]] = []
        for cid, cost in candidates.items():
            scope = f"candidate={cid}"
            if cost["unknown_count"]:
                initial_insufficient.append({"scope": scope, "reason": "unknown_cost"})
            if not cost["validated_source_words"]:
                initial_insufficient.append({"scope": scope, "reason": "zero_source_words"})
            if not cost.get("book_attribution_complete", True):
                initial_insufficient.append(
                    {"scope": scope, "reason": "missing_book_cost_attribution"}
                )
            for name in (
                "alignment_errors",
                "protocol_errors",
                "json_errors",
                "fallbacks",
                "required_node_failures",
            ):
                if system[cid].get(name) is None:
                    initial_insufficient.append(
                        {"scope": scope, "reason": "missing_system_evidence"}
                    )
            if human_facts.get("postedit", {}).get(cid) is None:
                initial_insufficient.append({"scope": scope, "reason": "missing_postedit"})
        for cid, book in sorted(book_attribution_insufficient):
            initial_insufficient.append(
                {"scope": f"candidate={cid}/book={book}", "reason": "missing_book_cost_attribution"}
            )
        initial_insufficient = [
            {"scope": scope, "reason": reason}
            for scope, reason in sorted(
                {(item["scope"], item["reason"]) for item in initial_insufficient}
            )
        ]
        normalized = _normalize(
            list(artifacts_by_id.values()),
            run,
            spec,
            _price_hash(snapshot),
            candidate_artifact_ids,
            source_by_book,
            [a["artifact_id"] for a in physical],
            {b: f"preparation:{b}" for b in bundle.books},
            system,
            initial_insufficient,
        )
        output = {
            "schema_version": 1,
            "input_hashes": {
                "run_hash": spec.run_hash,
                "preparation_sha256": spec.preparation_sha256,
                "price_snapshot_sha256": _price_hash(snapshot),
                "human": human_facts.get("input_hashes", {}),
            },
            "candidate_costs": candidates,
            "physical_spend": physical_cost,
            "million_word_estimate": million,
            "effective_costs": {},
            "system_metrics": system,
            "normalized_pricing_facts": normalized,
            "insufficient_data": [dict(x) for x in initial_insufficient],
        }
        _effective(output, candidates, human_facts, spec.editor_hourly_rates)
        output["insufficient_data"] = sorted(
            output["insufficient_data"], key=lambda x: (x["scope"], x["reason"])
        )
        return _decimalize(output)
    except CostAnalysisError:
        raise
    except Exception as exc:
        raise CostAnalysisError(str(exc)) from exc


def reprice_cost_system(
    normalized_pricing_facts: dict[str, Any],
    price_path: Path,
    human_facts: dict[str, Any],
    editor_hourly_rates: Sequence[Decimal],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    try:
        facts = _validate_normalized(normalized_pricing_facts)
        snapshot = load_price_snapshot(Path(price_path))
        if snapshot.currency != "CNY":
            _fail("price currency must be CNY")
        rates = [_dec(x, "editor hourly rate") for x in editor_hourly_rates]
        if any(x <= 0 for x in rates):
            _fail("editor hourly rates must be positive")
        if (
            bootstrap_seed != facts["bootstrap_seed"]
            or bootstrap_replicates != facts["bootstrap_replicates"]
        ):
            _fail("bootstrap parameters changed")
        return _reprice_from_facts(facts, snapshot, human_facts, rates)
    except CostAnalysisError:
        raise
    except Exception as exc:
        raise CostAnalysisError(str(exc)) from exc
