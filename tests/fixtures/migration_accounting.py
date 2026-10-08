from __future__ import annotations

from typing import Any

from tests.fixtures.migration_bundle import RecordingError


def _assert_protocol_accounting(
    persisted: dict[str, Any],
    protocol: dict[str, Any],
) -> None:
    counters = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
    )
    physical = (*counters, "attempts", "failed_attempts")
    agent_fields = (*physical, "reasoning_tokens")
    expected_totals = dict.fromkeys(counters, 0)
    expected_dimensions = {
        "by_agent": {},
        "by_operation": {},
        "by_provider": {},
        "by_model": {},
        "by_stage": {},
    }
    for call in protocol["calls"]:
        usage = call.get("token_usage")
        if not isinstance(usage, dict):
            raise RecordingError("model protocol response lacks explicit token accounting")
        values = {
            "calls": 1,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cache_hit_tokens": usage["prompt_cache_hit_tokens"],
            "cache_miss_tokens": usage["prompt_cache_miss_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
        }
        for key, amount in values.items():
            if type(amount) is not int or amount < 0:
                raise RecordingError("model protocol contains invalid token accounting")
            if key in expected_totals:
                expected_totals[key] += amount
        dimensions = {
            "by_agent": call["agent"],
            "by_operation": call["operation"],
            "by_provider": call["provider"],
            "by_model": f"{call['provider']}:{call['model']}",
            "by_stage": call["stage"],
        }
        for dimension, name in dimensions.items():
            if name is None:
                continue
            fields = (
                counters
                if dimension == "by_stage"
                else agent_fields
                if dimension in {"by_agent", "by_operation"}
                else physical
            )
            slot = expected_dimensions[dimension].setdefault(name, dict.fromkeys(fields, 0))
            for key, amount in values.items():
                if key in slot:
                    slot[key] += amount
            if "attempts" in slot:
                slot["attempts"] += 1
                slot["failed_attempts"] += 0
    if not isinstance(persisted.get("totals"), dict):
        raise RecordingError("persisted usage lacks token totals")
    for key, amount in expected_totals.items():
        if persisted["totals"].get(key) != amount:
            raise RecordingError(f"persisted usage total does not match model responses: {key}")
    for dimension, expected in expected_dimensions.items():
        actual = persisted.get(dimension)
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise RecordingError(
                f"persisted usage dimension does not match model responses: {dimension}"
            )
        for name, expected_slot in expected.items():
            if not isinstance(actual[name], dict) or any(
                actual[name].get(key) != value for key, value in expected_slot.items()
            ):
                raise RecordingError(
                    f"persisted usage counters do not match model responses: {dimension}.{name}"
                )
