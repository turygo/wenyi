"""Usage, cost, and system metric aggregation for benchmark reports."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import sha256_bytes
from trans_novel.benchmark.pricing import CostQuote, load_price_snapshot, quote_usage
from trans_novel.llm.usage import merge_usage_summaries


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate_id")
        usage = row.get("usage")
        if not isinstance(candidate, str) or not isinstance(usage, dict):
            raise ValueError("candidate usage provenance is invalid")
        try:
            result[candidate] = merge_usage_summaries(result.get(candidate, {}), usage)
        except ValueError as error:
            raise ValueError(f"candidate usage is invalid: {error}") from error
    return result


def price_band(snapshot: Any, model_id: str, started_at: object) -> str:
    model = snapshot.models.get(model_id)
    if model is None:
        return "all"
    bands = {rule.time_band for rule in model.rules}
    if "all" in bands:
        return "all"
    if bands != {"peak", "off_peak"} or not isinstance(started_at, str):
        raise ValueError(f"unsupported price time bands for {model_id}")
    try:
        instant = datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as error:
        raise ValueError("telemetry started_at is invalid") from error
    peak = instant.weekday() < 5 and (1 <= instant.hour < 4 or 6 <= instant.hour < 10)
    return "peak" if peak else "off_peak"


def aggregate_costs(
    rows: list[dict[str, Any]],
    telemetry_attempts: list[list[dict[str, Any]]],
    price_path: str | Path,
) -> dict[str, Any]:
    snapshot = load_price_snapshot(price_path)
    usage = aggregate_usage(rows)
    candidates: dict[str, Any] = {}
    grouped: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["candidate_id"]), []).append((row, telemetry_attempts[index]))
    for candidate, artifacts in sorted(grouped.items()):
        by_model: dict[str, dict[str, Any]] = {}
        total: Decimal | None = Decimal(0)
        for _artifact, attempts in artifacts:
            for attempt in attempts:
                provider = attempt.get("provider")
                model = attempt.get("resolved_model") or attempt.get("requested_model")
                if not isinstance(provider, str) or not isinstance(model, str):
                    raise ValueError("telemetry model provenance is invalid")
                model_id = f"{provider}:{model}"
                model_total = by_model.setdefault(
                    model_id,
                    {
                        "calls": 0,
                        "uncached_input_cost": Decimal(0),
                        "cached_input_cost": Decimal(0),
                        "output_cost": Decimal(0),
                        "total_cost": Decimal(0),
                        "reason": None,
                        "time_bands": {},
                    },
                )
                model_total["calls"] += 1
                try:
                    time_band = price_band(snapshot, model_id, attempt.get("started_at"))
                    model_total["time_bands"][time_band] = (
                        model_total["time_bands"].get(time_band, 0) + 1
                    )
                    quote = quote_usage(
                        snapshot,
                        model_id,
                        attempt,
                        time_band=time_band,
                        billed_usage_unknown=bool(attempt.get("billed_usage_unknown")),
                    )
                except (TypeError, ValueError) as error:
                    model_total["reason"] = str(error)
                    total = None
                    continue
                if not isinstance(quote, CostQuote):
                    model_total["reason"] = quote.reason
                    total = None
                    continue
                for field in (
                    "uncached_input_cost",
                    "cached_input_cost",
                    "output_cost",
                    "total_cost",
                ):
                    model_total[field] += getattr(quote, field)
                if total is not None:
                    total += quote.total_cost
        quotes = [
            {
                "model": model,
                **{
                    key: str(value) if isinstance(value, Decimal) else value
                    for key, value in values.items()
                },
            }
            for model, values in sorted(by_model.items())
        ]
        candidates[candidate] = {
            "usage": usage[candidate],
            "quotes": quotes,
            "api_cost": None if total is None else str(total),
            "currency": snapshot.currency,
        }
    return {
        "schema_version": 1,
        "price_sha256": sha256_bytes(Path(price_path).read_bytes()),
        "candidates": candidates,
    }


def aggregate_system(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    for candidate, artifacts in sorted(grouped.items()):
        totals = merge_usage_summaries({}, aggregate_usage(artifacts)[candidate])
        by_model = totals.get("by_model", {})
        candidates[candidate] = {
            "artifact_count": len(artifacts),
            "book_count": len({row["book_id"] for row in artifacts}),
            "output_count": sum(len(row.get("outputs", [])) for row in artifacts),
            "failed_attempts": sum(
                int(value.get("failed_attempts", 0)) for value in by_model.values()
            ),
            "reasoning_tokens": sum(
                int(value.get("reasoning_tokens", 0))
                for value in totals.get("by_agent", {}).values()
            ),
            "all_outputs_present": all(row.get("output_hashes") for row in artifacts),
        }
    return {"schema_version": 1, "candidates": candidates}


__all__ = ["aggregate_costs", "aggregate_system", "aggregate_usage", "price_band"]
