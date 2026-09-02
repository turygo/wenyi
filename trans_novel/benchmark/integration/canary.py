"""Synthetic canary request and telemetry checks."""

from __future__ import annotations

from typing import Any

from trans_novel.benchmark.run import JsonlCallTelemetrySink
from trans_novel.benchmark.schema import Candidate, CandidateSpec
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.model_profiles import parse_model_selection, parse_provider_model


def normalized_model(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return parse_model_selection(value).model


def model_identity(value: str) -> tuple[str, str, bool]:
    provider, model = parse_provider_model(value)
    selection = parse_model_selection(model)
    return provider, selection.model, selection.thinking != "off"


def canary_routes(records: list[CallAttemptTelemetry]) -> list[dict[str, str]]:
    return [{"agent": record.agent, "operation": record.operation} for record in records]


def run_canary(
    client: Any,
    candidate: Candidate,
    spec: CandidateSpec,
    sink: JsonlCallTelemetrySink,
) -> dict[str, Any]:
    calls = (
        (
            "translator",
            "integration.canary.translate",
            "文学翻译",
            "synthetic canary source",
            candidate.translator_model,
        ),
        (
            "editor",
            "integration.canary.polish",
            "中文润色编辑",
            "synthetic canary target",
            candidate.editor_model,
        ),
    )
    try:
        for agent, operation, marker, text, _model in calls:
            client.complete(
                [
                    {"role": "system", "content": marker},
                    {"role": "user", "content": f"[0] {text}"},
                ],
                json_mode=True,
                agent=agent,
                operation=operation,
            )
    except Exception as error:
        return {
            "schema_version": 1,
            "passed": False,
            "reason": type(error).__name__,
            "roles": [row[0] for row in calls],
            "translator_model": candidate.translator_model,
            "analyst_model": candidate.analyst_model,
            "editor_model": candidate.editor_model,
            "fast_model": candidate.fast_model,
            "temperature": spec.temperature,
            "seed": spec.seed,
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
            "unknown_required_usage_count": 1,
        }
    records = sink.records[-2:]
    expected_models = [
        model_identity(candidate.translator_model),
        model_identity(candidate.editor_model),
    ]
    if len(records) != 2:
        return {
            "schema_version": 1,
            "passed": False,
            "reason": "missing_canary_telemetry",
            "unknown_required_usage_count": 1,
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
        }
    reasoning = 0
    mismatch = 0
    unknown = 0
    for index, (record, expected) in enumerate(zip(records, expected_models, strict=True)):
        try:
            value = CallAttemptTelemetry.model_validate(record)
        except Exception:
            unknown += 1
            continue
        expected_agent, expected_operation = calls[index][0], calls[index][1]
        expected_provider, expected_model, expected_reasoning = expected
        if not expected_reasoning:
            reasoning += value.reasoning_tokens
        unknown += int(value.billed_usage_unknown)
        mismatch += int(
            value.agent != expected_agent
            or value.operation != expected_operation
            or value.provider != expected_provider
            or normalized_model(value.requested_model) != expected_model
            or normalized_model(value.resolved_model) != expected_model
            or value.reasoning_enabled != expected_reasoning
            or value.status != "success"
            or value.temperature != spec.temperature
            or value.seed is not None
        )
    passed = reasoning == 0 and mismatch == 0 and unknown == 0
    return {
        "schema_version": 1,
        "passed": passed,
        "roles": [row[0] for row in calls],
        "translator_model": candidate.translator_model,
        "analyst_model": candidate.analyst_model,
        "editor_model": candidate.editor_model,
        "fast_model": candidate.fast_model,
        "temperature": spec.temperature,
        "seed": spec.seed,
        "reasoning_tokens": reasoning,
        "model_mismatch_count": mismatch,
        "unknown_required_usage_count": unknown,
    }


__all__ = ["canary_routes", "model_identity", "normalized_model", "run_canary"]
