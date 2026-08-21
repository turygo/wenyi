"""Phase 8 report integration, publication gates, stable reports, and repricing."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from trans_novel.benchmark.integration import IntegrationError, validate_terminal_artifacts
from trans_novel.benchmark.report_cost import (
    CostAnalysisError,
    analyze_cost_system,
    reprice_cost_system,
)
from trans_novel.benchmark.report_human import HumanAnalysisError, analyze_human
from trans_novel.benchmark.report_schema import ReportSpec, load_report_spec

__all__ = ["ReportError", "build_report", "reprice_report", "validate_report"]

FILES = (
    "summary.json",
    "reproducibility.json",
    "candidates.csv",
    "quality_by_book.csv",
    "mqm_errors.csv",
    "pairwise.csv",
    "context_ablation.csv",
    "polish_effect.csv",
    "cost_by_operation.csv",
    "failures.csv",
    "pareto.csv",
    "report.html",
    "report_manifest.json",
)
PRICE_INDEPENDENT = (
    "quality_by_book.csv",
    "mqm_errors.csv",
    "pairwise.csv",
    "context_ablation.csv",
    "polish_effect.csv",
)


class ReportError(ValueError):
    """Invalid input, report, or create-only destination."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: (
            format(item, "f")
            if isinstance(item, Decimal)
            else (_ for _ in ()).throw(TypeError(f"not JSON serializable: {type(item).__name__}"))
        ),
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(
        _canonical(value) if not isinstance(value, bytes | bytearray) else bytes(value)
    ).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _read_canonical_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportError(f"invalid canonical JSON: {path.name}: {exc}") from exc
    if _json_bytes(value) != raw:
        raise ReportError(f"noncanonical JSON bytes: {path.name}")
    return value


def _raw_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_decimal(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, dict):
        return {str(k): _dump_decimal(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_dump_decimal(v) for v in value]
    return value


def _spec_json(spec: ReportSpec) -> dict[str, Any]:
    return _dump_decimal(spec.model_dump(mode="json"))


def _err(exc: Exception) -> ReportError:
    if isinstance(exc, ReportError):
        return exc
    return ReportError(str(exc))


def _input_hashes(
    spec: ReportSpec, human: Mapping[str, Any], cost: Mapping[str, Any]
) -> dict[str, str]:
    expected = {
        "corpus_sha256": spec.corpus_sha256,
        "run_hash": spec.run_hash,
        "preparation_sha256": spec.preparation_sha256,
        "pack_sha256": spec.pack_sha256,
        "evaluation_sha256": spec.evaluation_sha256,
        "price_snapshot_sha256": spec.price_snapshot_sha256,
    }
    h = human.get("input_hashes")
    c = cost.get("input_hashes")
    if not isinstance(h, Mapping) or not isinstance(c, Mapping):
        raise ReportError("analyzer input hashes missing")
    checks = {
        "corpus_sha256": h.get("corpus_sha256"),
        "pack_sha256": h.get("pack_sha256"),
        "evaluation_sha256": h.get("evaluation_sha256"),
        "run_hash": c.get("run_hash"),
        "preparation_sha256": c.get("preparation_sha256"),
        "price_snapshot_sha256": c.get("price_snapshot_sha256"),
    }
    if any(checks[k] != v for k, v in expected.items()):
        raise ReportError("analyzer input hash mismatch")
    completion_hash = h.get("evaluation_complete_sha256")
    if (
        not isinstance(completion_hash, str)
        or len(completion_hash) != 64
        or any(char not in "0123456789abcdef" for char in completion_hash)
    ):
        raise ReportError("evaluation completion hash missing")
    expected["evaluation_complete_sha256"] = completion_hash
    return expected


def _validate_integration(
    path: Path | None, spec: ReportSpec, candidates: set[str]
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    if path is None:
        return None, None
    try:
        validated = validate_terminal_artifacts(path.resolve().parent)
    except (IntegrationError, OSError, ValueError) as exc:
        raise ReportError(f"invalid integration terminal artifacts: {exc}") from exc
    request = validated["request"]
    value = validated["integration"]
    selected = set(value["candidates"])
    if (
        request.get("benchmark_id") != spec.benchmark_id
        or request.get("corpus_sha256") != spec.corpus_sha256
        or value.get("benchmark_id") != spec.benchmark_id
        or value.get("corpus_sha256") != spec.corpus_sha256
        or len(selected) not in {2, 3}
        or selected - candidates
    ):
        raise ReportError("integration schema or lineage invalid")
    raw_hash = validated["integration_sha256"]
    complete_hash = validated["integration_complete_sha256"]
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": spec.benchmark_id,
        "corpus_sha256": spec.corpus_sha256,
        "candidates": {},
        "integration_sha256": raw_hash,
        "integration_complete_sha256": complete_hash,
    }
    for cid in sorted(selected):
        item = validated["candidates"][cid]
        result = item["result"]
        status = item["status"]
        checks = {
            "completion_status": status == "completed",
            "producer_passed": result["passed"] is True,
            "canary_passed": result["canary_passed"] is True,
            "expected_interruption_observed": result["expected_interruption_observed"] is True,
            "resume_duplicate_operations_zero": result["resume_duplicate_operations"] == 0,
            "readiness_passed": result["readiness_passed"] is True,
            "mono_structural_pass": result["mono"]["structural_pass"] is True,
            "bilingual_structural_pass": result["bilingual"]["structural_pass"] is True,
            "reasoning_tokens_zero": result["reasoning_tokens"] == 0,
            "model_mismatch_zero": result["model_mismatch_count"] == 0,
            "unknown_required_usage_zero": result["unknown_required_usage_count"] == 0,
        }
        failed_predicates = sorted(key for key, passed_check in checks.items() if not passed_check)
        normalized["candidates"][cid] = {
            "passed": not failed_predicates,
            "structural_pass": bool(
                result["mono"]["structural_pass"] and result["bilingual"]["structural_pass"]
            ),
            "duplicate_completed_segments": result["resume_duplicate_operations"],
            "required_node_failures": 0,
            "reasoning_tokens": result["reasoning_tokens"],
            "resolved_model_mismatch": result["model_mismatch_count"],
            "billed_usage_unknown": result["unknown_required_usage_count"],
            "failures": failed_predicates,
            "status": status,
            "integration_detail": {**checks, "failed_predicates": failed_predicates},
        }
    return normalized, {
        "integration_sha256": raw_hash,
        "integration_complete_sha256": complete_hash,
    }


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _metric(metrics: Mapping[str, Any], *keys: str) -> Any:
    value: Any = metrics
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _entity_ids(human: Mapping[str, Any]) -> list[tuple[str, str]]:
    absolute = human.get("absolute", {})
    result = []
    if isinstance(absolute, Mapping):
        for cid, surfaces in absolute.items():
            if isinstance(surfaces, Mapping):
                result.extend((str(cid), str(surface)) for surface in surfaces)
    return sorted(set(result))


def _known_candidates(human: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for key in ("absolute", "mqm", "polish", "context", "postedit"):
        value = human.get(key)
        if isinstance(value, Mapping):
            result.update(str(item) for item in value)
    pairwise = human.get("pairwise")
    if isinstance(pairwise, Mapping):
        for surface in pairwise.values():
            if isinstance(surface, Mapping) and isinstance(surface.get("candidates"), Mapping):
                result.update(str(item) for item in surface["candidates"])
    return result


def _scope_parts(scope: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for token in scope.replace("/", " ").split():
        if "=" in token:
            key, value = token.split("=", 1)
            parts[key.strip()] = value.strip()
    return parts


def _scope_matches(scope: Any, cid: str, surface: str, book: str | None = None) -> bool:
    if not isinstance(scope, str):
        return False
    parts = _scope_parts(scope)
    if parts.get("candidate") not in (None, cid) or parts.get("surface") not in (None, surface):
        return False
    return parts.get("book") in (None, book) if book is not None else "book" not in parts


def _has_global_insuff(human: Mapping[str, Any]) -> bool:
    for item in (
        human.get("insufficient_data", [])
        if isinstance(human.get("insufficient_data"), list)
        else []
    ):
        scope = (
            item.get("scope")
            if isinstance(item, Mapping)
            else (item[0] if isinstance(item, list | tuple) and len(item) == 2 else None)
        )
        if (
            isinstance(scope, str)
            and "candidate" not in _scope_parts(scope)
            and "surface" not in _scope_parts(scope)
            and "book" not in _scope_parts(scope)
        ):
            return True
    return False


def _has_global_cost_insuff(cost: Mapping[str, Any]) -> bool:
    for item in (
        cost.get("insufficient_data", []) if isinstance(cost.get("insufficient_data"), list) else []
    ):
        if not isinstance(item, Mapping) or item.get("reason") == "unknown_cost":
            continue
        scope = item.get("scope")
        if not isinstance(scope, str) or "candidate" not in _scope_parts(scope):
            return True
    return False


def _integration_row(integration: Mapping[str, Any] | None, cid: str) -> Mapping[str, Any] | None:
    rows = integration.get("candidates") if isinstance(integration, Mapping) else None
    return (
        rows.get(cid) if isinstance(rows, Mapping) and isinstance(rows.get(cid), Mapping) else None
    )


def _gate_entity(
    cid: str,
    surface: str,
    human: Mapping[str, Any],
    cost: Mapping[str, Any],
    spec: ReportSpec,
    integration: Mapping[str, Any] | None,
    books: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    gates = spec.gates
    reasons: set[str] = set()
    by_book: dict[str, Any] = {}
    abs_scope = _metric(human, "absolute", cid, surface) or {}
    mqm_scope = _metric(human, "mqm", cid, surface) or {}
    sys_scope = _metric(cost, "system_metrics", cid) or {}
    polish = _metric(human, "polish", cid) or {}
    irow = _integration_row(integration, cid)
    rel = human.get("reliability", {}) if isinstance(human.get("reliability"), Mapping) else {}
    insufficient_rows = (
        human.get("insufficient_data", [])
        if isinstance(human.get("insufficient_data"), list)
        else []
    )
    for item in insufficient_rows:
        if isinstance(item, Mapping):
            scope, reason = item.get("scope"), item.get("reason")
        elif isinstance(item, list | tuple) and len(item) == 2:
            scope, reason = item
        else:
            continue
        if (
            isinstance(scope, str)
            and isinstance(reason, str)
            and _scope_matches(scope, cid, surface)
        ):
            reasons.add(f"{scope}:{reason}")
    cost_insufficient_rows = (
        cost.get("insufficient_data", []) if isinstance(cost.get("insufficient_data"), list) else []
    )
    for item in cost_insufficient_rows:
        scope, reason = (
            (item.get("scope"), item.get("reason"))
            if isinstance(item, Mapping)
            else (
                (item[0], item[1])
                if isinstance(item, list | tuple) and len(item) == 2
                else (None, None)
            )
        )
        if (
            isinstance(scope, str)
            and isinstance(reason, str)
            and reason != "unknown_cost"
            and _scope_matches(scope, cid, surface)
        ):
            reasons.add(f"{scope}:{reason}")
    macro = abs_scope.get("macro", {}) if isinstance(abs_scope, Mapping) else {}
    dims = macro.get("dimensions", {}) if isinstance(macro, Mapping) else {}
    fidelity = _num(_metric(dims, "fidelity", "raw_mean"))
    natural = _num(_metric(dims, "naturalness", "raw_mean"))
    if fidelity is None or fidelity < float(gates.fidelity_mean_min):
        reasons.add("fidelity")
    if natural is None or natural < float(gates.naturalness_mean_min):
        reasons.add("naturalness")
    major95 = _num(_metric(mqm_scope, "macro", "major_rate_upper95"))
    critical = _metric(mqm_scope, "macro", "agreed", "severity", "critical", "count")
    if critical is None or critical > gates.critical_max:
        reasons.add("critical")
    if major95 is None or major95 > float(gates.major_per_10k_upper95_max):
        reasons.add("major_upper95")
    completion = _num(_metric(sys_scope, "completion_rate"))
    if completion is None or completion < float(gates.completion_min):
        reasons.add("completion")
    for field, gate_name in (
        ("protocol_errors", "protocol_errors"),
        ("json_errors", "json_errors"),
        ("alignment_errors", "alignment_errors"),
        ("required_node_failures", "required_node_failures"),
        ("resume_duplicate_operations", "resume_duplicate_operations"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("resolved_model_mismatch", "resolved_model_mismatch"),
    ):
        value = _metric(sys_scope, field)
        limit = getattr(gates, gate_name + "_max", 0)
        if value is None or value > limit:
            reasons.add(gate_name)
    harm = (
        sum(int(v or 0) for v in (polish.get("mqm_semantic_harm", {}) or {}).values())
        if isinstance(polish, Mapping)
        else None
    )
    harm_rate = _num(_metric(polish, "macro", "harm_wilson_upper95"))
    if harm is None or harm > gates.polish_major_semantic_harm_max:
        reasons.add("polish_semantic_harm")
    if harm_rate is None or harm_rate > float(gates.polish_harm_rate_upper95_max):
        reasons.add("polish_harm_upper95")
    required_alpha = (
        "fidelity",
        "naturalness",
        "style_voice",
        "consistency",
        "context_handling",
        "readability",
        "format_integrity",
        "pairwise_winner",
        "context_correctness",
        "mqm_severity",
        "mqm_type",
    )
    for key in required_alpha:
        value = _num(rel.get(key)) if isinstance(rel, Mapping) else None
        if value is None or value < float(gates.krippendorff_alpha_min):
            reasons.add(f"alpha:{key}")
    for book in books:
        b_abs = _metric(abs_scope, "by_book", book) or {}
        b_mqm = _metric(mqm_scope, "by_book", book) or {}
        b_sys = _metric(sys_scope, "by_book", book) or {}
        bcomp = b_abs.get("composite", {}) if isinstance(b_abs, Mapping) else {}
        bdims = b_abs.get("dimensions", {}) if isinstance(b_abs, Mapping) else {}
        bn = bcomp.get("n_units")
        brate = _num(_metric(b_mqm, "agreed", "severity", "major", "rate_per_10k"))
        bf = _num(_metric(bdims, "fidelity", "raw_mean"))
        bnatur = _num(_metric(bdims, "naturalness", "raw_mean"))
        breasons = set()
        for item in insufficient_rows:
            scope, reason = (
                (item.get("scope"), item.get("reason"))
                if isinstance(item, Mapping)
                else (
                    (item[0], item[1])
                    if isinstance(item, list | tuple) and len(item) == 2
                    else (None, None)
                )
            )
            if (
                isinstance(scope, str)
                and isinstance(reason, str)
                and _scope_matches(scope, cid, surface, book)
            ):
                breasons.add(f"{scope}:{reason}")
        for item in cost_insufficient_rows:
            scope, reason = (
                (item.get("scope"), item.get("reason"))
                if isinstance(item, Mapping)
                else (
                    (item[0], item[1])
                    if isinstance(item, list | tuple) and len(item) == 2
                    else (None, None)
                )
            )
            if (
                isinstance(scope, str)
                and isinstance(reason, str)
                and reason != "unknown_cost"
                and _scope_matches(scope, cid, surface, book)
            ):
                breasons.add(f"{scope}:{reason}")
        if not isinstance(bn, int) or bn < 10:
            breasons.add("insufficient_book_sample")
        else:
            if bf is None or bf < float(gates.fidelity_mean_min):
                breasons.add("fidelity")
            if bnatur is None or bnatur < float(gates.naturalness_mean_min):
                breasons.add("naturalness")
        bcritical = _metric(b_mqm, "agreed", "severity", "critical", "count")
        if bcritical is None or bcritical > gates.critical_max:
            breasons.add("critical")
        if brate is None or brate > float(gates.per_book_major_per_10k_max):
            breasons.add("per_book_major")
        if b_sys:
            completion_book = _num(b_sys.get("completion_rate"))
            if completion_book is None or completion_book < float(gates.completion_min):
                breasons.add("completion")
            for field, gate_name in (
                ("protocol_errors", "protocol_errors"),
                ("json_errors", "json_errors"),
                ("alignment_errors", "alignment_errors"),
                ("required_node_failures", "required_node_failures"),
                ("resume_duplicate_operations", "resume_duplicate_operations"),
                ("reasoning_tokens", "reasoning_tokens"),
                ("resolved_model_mismatch", "resolved_model_mismatch"),
            ):
                value = b_sys.get(field)
                if value is None or value > getattr(gates, gate_name + "_max", 0):
                    breasons.add(gate_name)
        if breasons:
            reasons.update(f"{book}:{x}" for x in sorted(breasons))
        by_book[book] = {
            "gate_pass": not breasons,
            "reasons": sorted(breasons),
            "fidelity_raw": bf,
            "naturalness_raw": bnatur,
            "major_rate_per_10k": brate,
            "n_units": bn,
        }
    # Human-level pending applies to all entities and is never auto-adjudicated.
    if int(human.get("pending_adjudication_count", 0) or 0) > 0:
        reasons.add("pending_adjudication")
    without = not reasons
    integration_reason = None
    if integration is None:
        integration_reason = "integration_missing"
    elif irow is None:
        integration_reason = "integration:not_selected"
    elif not (
        irow.get("passed") is True
        and irow.get("structural_pass") is True
        and all(
            irow.get(k) == 0
            for k in (
                "duplicate_completed_segments",
                "required_node_failures",
                "reasoning_tokens",
                "resolved_model_mismatch",
                "billed_usage_unknown",
            )
        )
        and not irow.get("failures")
    ):
        integration_reason = (
            "integration_failed:"
            + ",".join(irow.get("integration_detail", {}).get("failed_predicates", ["unknown"]))
            if isinstance(irow.get("integration_detail"), Mapping)
            else "integration_failed"
        )
    gate_reasons = set(reasons)
    if integration_reason:
        gate_reasons.add(integration_reason)
    return (
        {
            "gate_pass_without_integration": without,
            "gate_pass": without and integration_reason is None,
            "reasons": sorted(gate_reasons),
        },
        by_book,
        sorted(reasons),
    )


def _candidate_metrics(
    cid: str, surface: str, human: Mapping[str, Any], cost: Mapping[str, Any]
) -> dict[str, Any]:
    abs_scope = _metric(human, "absolute", cid, surface) or {}
    macro = abs_scope.get("macro", {}) if isinstance(abs_scope, Mapping) else {}
    dims = macro.get("dimensions", {}) if isinstance(macro, Mapping) else {}
    comp = macro.get("composite", {}) if isinstance(macro, Mapping) else {}
    mqm = _metric(human, "mqm", cid, surface) or {}
    mm = mqm.get("macro", {}) if isinstance(mqm, Mapping) else {}
    sys = _metric(cost, "system_metrics", cid) or {}
    cc = _metric(cost, "candidate_costs", cid) or {}
    pair = _metric(human, "pairwise", surface, "candidates", cid) or {}
    eff = _metric(cost, "effective_costs", cid) or {}
    effective = (
        {
            rate: (_metric(eff, rate, surface, "value"))
            for rate in sorted(eff)
            if _metric(eff, rate, surface, "value") is not None
        }
        if isinstance(eff, Mapping)
        else {}
    )
    return {
        "entity_id": f"{cid}@{surface}",
        "candidate_id": cid,
        "surface": surface,
        "composite": comp.get("value"),
        "composite_lower95": comp.get("lower95"),
        "fidelity_raw": _metric(dims, "fidelity", "raw_mean"),
        "fidelity_lower95": _metric(dims, "fidelity", "lower95"),
        "naturalness_raw": _metric(dims, "naturalness", "raw_mean"),
        "critical": _metric(mm, "agreed", "severity", "critical", "count"),
        "major_upper95": mm.get("major_rate_upper95"),
        "mqm_upper95": mm.get("weighted_points_upper95"),
        "bt_field_win": pair.get("field_win"),
        "bt_lower95": pair.get("field_win_lower95"),
        "api_cost": cc.get("api_cost"),
        "effective_costs": effective,
        "wall_p95_ms": sys.get("latency_p95_ms"),
        "system": sys,
    }


def _ranking_tuple(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    vals = (
        row.get("critical"),
        row.get("major_upper95"),
        row.get("mqm_upper95"),
        row.get("fidelity_lower95"),
        row.get("bt_lower95"),
        _decimal(row.get("api_cost")),
    )
    if any(v is None for v in vals):
        return None
    return (int(vals[0]), float(vals[1]), float(vals[2]), -float(vals[3]), -float(vals[4]), vals[5])


def _recommendations(rows: list[dict[str, Any]], spec: ReportSpec, status: str) -> dict[str, Any]:
    if status != "final":
        return {}
    by_surface: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("gate_pass")
            and row.get("api_cost") is not None
            and row.get("wall_p95_ms") is not None
        ):
            by_surface.setdefault(row["surface"], []).append(row)
    out: dict[str, Any] = {}
    for surface, values in sorted(by_surface.items()):
        cheapest = min(values, key=lambda x: (_decimal(x["api_cost"]), x["entity_id"]))
        eff: dict[str, Any] = {}
        rates = sorted({r for row in values for r in row.get("effective_costs", {})})
        for rate in rates:
            eligible = [r for r in values if rate in r.get("effective_costs", {})]
            if eligible:
                chosen = min(
                    eligible, key=lambda x: (_decimal(x["effective_costs"][rate]), x["entity_id"])
                )
                eff[rate] = {
                    "entity_id": chosen["entity_id"],
                    "value": chosen["effective_costs"][rate],
                    "statistical_tie": sum(
                        _decimal(x["effective_costs"][rate])
                        == _decimal(chosen["effective_costs"][rate])
                        for x in eligible
                    )
                    > 1,
                }
        ranked = [(r, _ranking_tuple(r)) for r in values]
        ranked = [(r, t) for r, t in ranked if t is not None]
        if ranked:
            ranked.sort(key=lambda x: (x[1], x[0]["entity_id"]))
            chosen, tup = ranked[0]
            same = sum(t == tup for _, t in ranked) > 1
            high = {
                "entity_id": chosen["entity_id"],
                "tuple": [format(x, "f") if isinstance(x, Decimal) else x for x in tup],
                "statistical_tie": same,
            }
        else:
            high = None
        out[surface] = {
            "cheapest": {
                "entity_id": cheapest["entity_id"],
                "value": cheapest["api_cost"],
                "statistical_tie": sum(
                    _decimal(x["api_cost"]) == _decimal(cheapest["api_cost"]) for x in values
                )
                > 1,
            },
            "effective_value": eff,
            "highest_quality": high,
        }
    return out


def _exact_frontier(
    rows: list[dict[str, Any]], dimensions: Mapping[str, str]
) -> tuple[list[str], dict[str, str]]:
    valid: dict[str, dict[str, Decimal]] = {}
    excluded: dict[str, str] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        unknown = None
        values: dict[str, Decimal] = {}
        for field in dimensions:
            value = _decimal(row.get(field))
            if value is None or not value.is_finite():
                unknown = field
                break
            values[field] = value
        if unknown is not None:
            excluded[candidate_id] = f"unknown:{unknown}"
        else:
            valid[candidate_id] = values
    frontier: list[str] = []
    for candidate in sorted(valid):
        dominated = False
        for other in sorted(valid):
            if candidate == other:
                continue
            no_worse = True
            strictly_better = False
            for field, direction in dimensions.items():
                left, right = valid[other][field], valid[candidate][field]
                if direction == "min":
                    if left > right:
                        no_worse = False
                        break
                    strictly_better |= left < right
                else:
                    if left < right:
                        no_worse = False
                        break
                    strictly_better |= left > right
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier, dict(sorted(excluded.items()))


def _pareto(rows: list[dict[str, Any]], status: str, *, diagnostic: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for surface in sorted({r["surface"] for r in rows}):
        values = [
            r
            for r in rows
            if r["surface"] == surface
            and (r.get("gate_pass_without_integration") if diagnostic else r.get("gate_pass"))
        ]
        eligible = [
            r
            for r in values
            if r.get("api_cost") is not None
            and r.get("mqm_upper95") is not None
            and r.get("composite_lower95") is not None
            and r.get("bt_lower95") is not None
            and r.get("wall_p95_ms") is not None
        ]
        if not diagnostic and status != "final":
            eligible = []
        dimensions = {
            "api_cost": "min",
            "mqm_upper95": "min",
            "composite_lower95": "max",
            "bt_lower95": "max",
            "wall_p95_ms": "min",
        }
        frontier_rows = [
            {
                "candidate_id": r["entity_id"],
                "api_cost": r["api_cost"],
                "mqm_upper95": r["mqm_upper95"],
                "composite_lower95": r["composite_lower95"],
                "bt_lower95": r["bt_lower95"],
                "wall_p95_ms": r["wall_p95_ms"],
            }
            for r in eligible
        ]
        frontier, excluded = _exact_frontier(frontier_rows, dimensions) if eligible else ([], {})
        result[surface] = {"api": frontier, "excluded": excluded}
        rates = sorted({rate for r in eligible for rate in r.get("effective_costs", {})})
        result[surface]["effective"] = {}
        for rate in rates:
            vals = [r for r in eligible if rate in r.get("effective_costs", {})]
            effective_rows = [
                {
                    "candidate_id": r["entity_id"],
                    "api_cost": r["effective_costs"][rate],
                    "mqm_upper95": r["mqm_upper95"],
                    "composite_lower95": r["composite_lower95"],
                    "bt_lower95": r["bt_lower95"],
                    "wall_p95_ms": r["wall_p95_ms"],
                }
                for r in vals
            ]
            frontier_rate, _ = _exact_frontier(effective_rows, dimensions) if vals else ([], {})
            result[surface]["effective"][rate] = frontier_rate
    return result


def _status(
    human: Mapping[str, Any], cost: Mapping[str, Any], integration: Mapping[str, Any] | None
) -> str:
    if _has_global_insuff(human) or _has_global_cost_insuff(cost):
        return "insufficient_data"
    reliability = human.get("reliability")
    required_alpha = (
        "fidelity",
        "naturalness",
        "style_voice",
        "consistency",
        "context_handling",
        "readability",
        "format_integrity",
        "pairwise_winner",
        "context_correctness",
        "mqm_severity",
        "mqm_type",
    )
    if isinstance(reliability, Mapping):
        try:
            if any(
                reliability.get(key) is None or float(reliability.get(key)) < 0.67
                for key in required_alpha
            ):
                return "needs_recalibration"
        except (TypeError, ValueError):
            return "needs_recalibration"
    if integration is None:
        return "provisional"
    integration_rows = integration.get("candidates", {}) if isinstance(integration, Mapping) else {}
    if not isinstance(integration_rows, Mapping) or not integration_rows:
        return "provisional"
    if not any(
        isinstance(row, Mapping)
        and row.get("passed") is True
        and row.get("structural_pass") is True
        and all(
            row.get(k) == 0
            for k in (
                "duplicate_completed_segments",
                "required_node_failures",
                "reasoning_tokens",
                "resolved_model_mismatch",
                "billed_usage_unknown",
            )
        )
        and not row.get("failures")
        for row in integration_rows.values()
    ):
        return "provisional"
    if int(human.get("pending_adjudication_count", 0) or 0):
        return "provisional"
    return "final"


def _assert_no_text(value: Any) -> None:
    forbidden = {"source", "target", "edited_target", "quote", "answer_key", "rationale"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                raise ReportError(f"text-bearing fact key: {key}")
            _assert_no_text(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_text(item)


def _csv_bytes(headers: list[str], rows: list[list[Any]]) -> bytes:
    import io

    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(
            ["" if x is None else (str(x).lower() if isinstance(x, bool) else x) for x in row]
        )
    return out.getvalue().encode("utf-8")


def _assemble(
    spec: ReportSpec,
    human: dict[str, Any],
    cost: dict[str, Any],
    integration: dict[str, Any] | None,
    input_hashes: dict[str, str],
    prior_hash: str | None = None,
    price_lineage: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bytes]]:
    _assert_no_text(human)
    _assert_no_text(cost.get("normalized_pricing_facts"))
    _assert_no_text(cost.get("system_metrics"))
    entities = _entity_ids(human)
    books = (
        list(human.get("denominators", {}).get("books", []))
        if isinstance(human.get("denominators"), Mapping)
        else []
    )
    status = _status(human, cost, integration)
    rows: list[dict[str, Any]] = []
    failures: list[list[Any]] = []
    quality: list[list[Any]] = []
    withheld: dict[str, list[str]] = {}
    for cid, surface in entities:
        gate, by_book, _ = _gate_entity(cid, surface, human, cost, spec, integration, books)
        row = _candidate_metrics(cid, surface, human, cost)
        row.update(gate)
        rows.append(row)
        reasons = set(gate["reasons"])
        cc = _metric(cost, "candidate_costs", cid) or {}
        if cc.get("unknown_count"):
            reasons.add("unknown_cost")
            row["gate_pass"] = False
            row["gate_pass_without_integration"] = False
        if row.get("api_cost") is None:
            reasons.add("missing_metric:api_cost")
        if reasons:
            withheld[row["entity_id"]] = sorted(reasons)
        for book in books:
            b = by_book.get(book, {})
            quality.append(
                [
                    row["entity_id"],
                    cid,
                    surface,
                    book,
                    b.get("n_units"),
                    b.get("fidelity_raw"),
                    b.get("naturalness_raw"),
                    _metric(human, "absolute", cid, surface, "by_book", book, "composite", "value"),
                    _metric(
                        human,
                        "mqm",
                        cid,
                        surface,
                        "by_book",
                        book,
                        "agreed",
                        "severity",
                        "critical",
                        "count",
                    ),
                    b.get("major_rate_per_10k"),
                    _metric(human, "mqm", cid, surface, "by_book", book, "weighted_points_per_10k"),
                    b.get("gate_pass"),
                    ";".join(b.get("reasons", [])),
                ]
            )
        for reason in sorted(reasons):
            failures.append([row["entity_id"], "global", reason, "publication"])
    if status == "final" and not any(row.get("gate_pass") is True for row in rows):
        status = "provisional"
    recommendations = _recommendations(rows, spec, status)
    pareto = _pareto(rows, status)
    diagnostic = _pareto(rows, status, diagnostic=True)
    summary = {
        "schema_version": 1,
        "status": status,
        "input_hashes": input_hashes,
        "candidates": rows,
        "gates": _spec_json(spec)["gates"],
        "pareto": pareto,
        "diagnostic_frontier": diagnostic,
        "recommendations": recommendations,
        "withheld_reasons": withheld,
        "pending_adjudication_count": human.get("pending_adjudication_count", 0),
        "integration": integration,
        "million_word_estimate": cost.get("million_word_estimate"),
    }
    repro = {
        "schema_version": 1,
        "report_spec": _spec_json(spec),
        "human_facts": human,
        "normalized_pricing_facts": cost.get("normalized_pricing_facts"),
        "system_metrics": cost.get("system_metrics"),
        "million_word_estimate": cost.get("million_word_estimate"),
        "input_hashes": input_hashes,
        "bootstrap": {
            "seed": spec.bootstrap_seed,
            "replicates": spec.bootstrap_replicates,
            "method": "hierarchical_cluster",
            "discards": _metric(human, "pairwise", "bootstrap_discarded") or {},
        },
        "integration_facts": integration,
        "quality_performance_sha256": _sha({"human": human, "system": cost.get("system_metrics")}),
        "prior_report_hash": prior_hash,
    }
    rate_keys = [
        format(rate, "f").rstrip("0").rstrip(".") if "." in format(rate, "f") else format(rate, "f")
        for rate in spec.editor_hourly_rates
    ]
    candidate_headers = (
        [
            "entity_id",
            "candidate_id",
            "surface",
            "gate_pass",
            "gate_pass_without_integration",
            "composite",
            "composite_lower95",
            "fidelity_raw",
            "fidelity_lower95",
            "naturalness_raw",
            "critical",
            "major_upper95",
            "mqm_upper95",
            "bt_field_win",
            "bt_lower95",
            "api_cost",
        ]
        + [f"effective_cost_rate_{key.replace('.', '_')}" for key in rate_keys]
        + ["wall_p95_ms", "status"]
    )
    headers = {
        "candidates.csv": candidate_headers,
        "quality_by_book.csv": [
            "entity_id",
            "candidate_id",
            "surface",
            "book_id",
            "n_units",
            "fidelity_raw",
            "naturalness_raw",
            "composite",
            "critical",
            "major_rate_per_10k",
            "mqm_weighted_per_10k",
            "gate_pass",
            "reasons",
        ],
        "mqm_errors.csv": [
            "entity_id",
            "book_id",
            "agreement",
            "classification",
            "category",
            "count",
            "rate_per_10k",
            "event_wilson_upper95",
            "weighted_points_lower95",
            "weighted_points_upper95",
            "major_rate_lower95",
            "major_rate_upper95",
            "pending_adjudication_count",
        ],
        "pairwise.csv": [
            "surface",
            "candidate_id",
            "ability",
            "ability_lower95",
            "ability_upper95",
            "field_win",
            "field_win_lower95",
            "field_win_upper95",
            "ratings",
            "units",
        ],
        "cost_by_operation.csv": [
            "record_type",
            "scope",
            "candidate_id",
            "operation",
            "book_id",
            "value",
            "lower_bound",
            "lower95",
            "upper95",
            "complete",
            "source_words",
            "fixed_preparation_lower_bound",
            "variable_lower_bound",
            "estimate_lower_bound",
            "assumption",
            "unknown_count",
            "cache_savings",
            "retry_cost",
        ],
        "context_ablation.csv": [
            "candidate_id",
            "strategy",
            "book_id",
            "correct",
            "incorrect",
            "uncertain",
            "accuracy",
            "uncertain_rate",
            "lift_from_c0",
            "lift_lower95",
            "lift_upper95",
        ],
        "polish_effect.csv": [
            "candidate_id",
            "book_id",
            "improved",
            "neutral",
            "harm",
            "total",
            "improved_rate",
            "harm_rate",
            "net",
            "harm_upper95",
            "semantic_harm",
        ],
        "failures.csv": ["entity_id", "scope", "reason", "gate"],
        "pareto.csv": [
            "surface",
            "frontier_kind",
            "rate",
            "entity_id",
            "is_frontier",
            "exclusion_reason",
        ],
    }
    mqm_rows: list[list[Any]] = []
    for cid, surface in entities:
        scope = _metric(human, "mqm", cid, surface) or {}
        for book, book_scope in sorted(
            (scope.get("by_book", {}) if isinstance(scope, Mapping) else {}).items()
        ):
            for classification in ("raw", "agreed"):
                section = (
                    book_scope.get(classification, {}) if isinstance(book_scope, Mapping) else {}
                )
                for category, value in sorted(
                    (section.get("severity", {}) if isinstance(section, Mapping) else {}).items()
                ):
                    mqm_rows.append(
                        [
                            f"{cid}@{surface}",
                            book,
                            classification,
                            "severity",
                            category,
                            value.get("count"),
                            value.get("rate_per_10k"),
                            _metric(book_scope, "event_wilson_upper95", category)
                            if classification == "agreed"
                            else None,
                            _metric(scope, "macro", "weighted_points_lower95"),
                            _metric(scope, "macro", "weighted_points_upper95"),
                            _metric(scope, "macro", "major_rate_lower95"),
                            _metric(scope, "macro", "major_rate_upper95"),
                            book_scope.get("pending_adjudication_count"),
                        ]
                    )
                for category, value in sorted(
                    (section.get("type", {}) if isinstance(section, Mapping) else {}).items()
                ):
                    mqm_rows.append(
                        [
                            f"{cid}@{surface}",
                            book,
                            classification,
                            "type",
                            category,
                            value.get("count"),
                            value.get("rate_per_10k"),
                            None,
                            _metric(scope, "macro", "weighted_points_lower95"),
                            _metric(scope, "macro", "weighted_points_upper95"),
                            _metric(scope, "macro", "major_rate_lower95"),
                            _metric(scope, "macro", "major_rate_upper95"),
                            book_scope.get("pending_adjudication_count"),
                        ]
                    )
    pair_rows: list[list[Any]] = []
    for surface, section in sorted((human.get("pairwise", {}) or {}).items()):
        if not isinstance(section, Mapping):
            continue
        for cid, value in sorted((section.get("candidates", {}) or {}).items()):
            pair_rows.append(
                [
                    surface,
                    cid,
                    value.get("ability"),
                    value.get("ability_lower95"),
                    value.get("ability_upper95"),
                    value.get("field_win"),
                    value.get("field_win_lower95"),
                    value.get("field_win_upper95"),
                    section.get("ratings"),
                    section.get("units"),
                ]
            )
    context_rows: list[list[Any]] = []
    for cid, context_value in sorted((human.get("context", {}) or {}).items()):
        strategies = (
            context_value.get("by_strategy", context_value)
            if isinstance(context_value, Mapping)
            else {}
        )
        lifts = context_value.get("lift", {}) if isinstance(context_value, Mapping) else {}
        for strategy, section in sorted((strategies or {}).items()):
            lift = lifts.get(strategy, {}) if isinstance(lifts, Mapping) else {}
            per_book_lift = lift.get("by_book", {}) if isinstance(lift, Mapping) else {}
            for book, value in sorted(
                (section.get("by_book", {}) if isinstance(section, Mapping) else {}).items()
            ):
                context_rows.append(
                    [
                        cid,
                        strategy,
                        book,
                        value.get("correct"),
                        value.get("incorrect"),
                        value.get("uncertain"),
                        value.get("accuracy"),
                        value.get("uncertain_rate"),
                        per_book_lift.get(book),
                        None,
                        None,
                    ]
                )
            macro = section.get("macro", {}) if isinstance(section, Mapping) else {}
            context_rows.append(
                [
                    cid,
                    strategy,
                    "",
                    macro.get("correct"),
                    macro.get("incorrect"),
                    macro.get("uncertain"),
                    macro.get("accuracy"),
                    macro.get("uncertain_rate"),
                    lift.get("value"),
                    lift.get("lower95"),
                    lift.get("upper95"),
                ]
            )
    polish_rows: list[list[Any]] = []
    for cid, section in sorted((human.get("polish", {}) or {}).items()):
        for book, value in sorted(
            (section.get("by_book", {}) if isinstance(section, Mapping) else {}).items()
        ):
            polish_rows.append(
                [
                    cid,
                    book,
                    value.get("improved"),
                    value.get("neutral"),
                    value.get("harm"),
                    value.get("total"),
                    value.get("improved_rate"),
                    value.get("harm_rate"),
                    value.get("net"),
                    value.get("harm_upper95"),
                    (section.get("mqm_semantic_harm", {}) or {}).get(book),
                ]
            )
        macro = section.get("macro", {}) if isinstance(section, Mapping) else {}
        polish_rows.append(
            [
                cid,
                "",
                macro.get("improved"),
                macro.get("neutral"),
                macro.get("harm"),
                macro.get("total"),
                macro.get("improved_rate"),
                macro.get("harm_rate"),
                macro.get("net"),
                macro.get("harm_upper95"),
                None,
            ]
        )
    cost_rows: list[list[Any]] = []
    for cid, value in sorted((cost.get("candidate_costs", {}) or {}).items()):
        for operation, op in sorted(
            (value.get("by_operation", {}) if isinstance(value, Mapping) else {}).items()
        ):
            cost_rows.append(
                [
                    "operation",
                    "candidate",
                    cid,
                    operation,
                    "",
                    "",
                    op.get("api_cost_lower_bound"),
                    "",
                    "",
                    op.get("cost_complete"),
                    "",
                    "",
                    "",
                    "",
                    "",
                    op.get("unknown_count"),
                    op.get("cache_savings_lower_bound"),
                    op.get("retry_cost_lower_bound"),
                ]
            )
    physical = (
        cost.get("physical_spend", {}) if isinstance(cost.get("physical_spend"), Mapping) else {}
    )
    cost_rows.append(
        [
            "operation",
            "physical",
            "",
            "",
            "",
            "",
            physical.get("api_cost_lower_bound"),
            "",
            "",
            physical.get("cost_complete"),
            "",
            "",
            "",
            "",
            "",
            physical.get("unknown_count"),
            physical.get("cache_savings_lower_bound"),
            physical.get("retry_cost_lower_bound"),
        ]
    )
    million = (
        cost.get("million_word_estimate", {})
        if isinstance(cost.get("million_word_estimate"), Mapping)
        else {}
    )
    for cid, estimate in sorted(million.items()):
        for book, book_value in sorted(
            (estimate.get("by_book", {}) if isinstance(estimate, Mapping) else {}).items()
        ):
            cost_rows.append(
                [
                    "million_word_estimate",
                    "book",
                    cid,
                    "",
                    book,
                    estimate.get("value"),
                    estimate.get("lower_bound"),
                    estimate.get("lower95"),
                    estimate.get("upper95"),
                    book_value.get("complete"),
                    book_value.get("source_words"),
                    book_value.get("fixed_preparation_lower_bound"),
                    book_value.get("variable_lower_bound"),
                    book_value.get("estimate_lower_bound"),
                    estimate.get("assumption"),
                    "",
                    "",
                    "",
                ]
            )
        cost_rows.append(
            [
                "million_word_estimate",
                "global",
                cid,
                "",
                "",
                estimate.get("value"),
                estimate.get("lower_bound"),
                estimate.get("lower95"),
                estimate.get("upper95"),
                estimate.get("complete"),
                "",
                "",
                "",
                "",
                estimate.get("assumption"),
                "",
                "",
                "",
            ]
        )
    csvs: dict[str, bytes] = {}
    csvs["candidates.csv"] = _csv_bytes(
        headers["candidates.csv"],
        [
            [
                r.get("entity_id"),
                r.get("candidate_id"),
                r.get("surface"),
                r.get("gate_pass"),
                r.get("gate_pass_without_integration"),
                r.get("composite"),
                r.get("composite_lower95"),
                r.get("fidelity_raw"),
                r.get("fidelity_lower95"),
                r.get("naturalness_raw"),
                r.get("critical"),
                r.get("major_upper95"),
                r.get("mqm_upper95"),
                r.get("bt_field_win"),
                r.get("bt_lower95"),
                r.get("api_cost"),
            ]
            + [r.get("effective_costs", {}).get(key) for key in rate_keys]
            + [r.get("wall_p95_ms"), status]
            for r in sorted(rows, key=lambda x: x["entity_id"])
        ],
    )
    csvs["quality_by_book.csv"] = _csv_bytes(
        headers["quality_by_book.csv"], sorted(quality, key=lambda x: (x[0], x[3]))
    )
    csvs["mqm_errors.csv"] = _csv_bytes(
        headers["mqm_errors.csv"], sorted(mqm_rows, key=lambda x: (x[0], x[1], x[2], x[3]))
    )
    csvs["failures.csv"] = _csv_bytes(headers["failures.csv"], sorted(failures))
    csvs["pairwise.csv"] = _csv_bytes(
        headers["pairwise.csv"], sorted(pair_rows, key=lambda x: (x[0], x[1]))
    )
    csvs["context_ablation.csv"] = _csv_bytes(
        headers["context_ablation.csv"], sorted(context_rows, key=lambda x: (x[0], x[1], x[2]))
    )
    csvs["polish_effect.csv"] = _csv_bytes(
        headers["polish_effect.csv"], sorted(polish_rows, key=lambda x: (x[0], x[1]))
    )
    csvs["cost_by_operation.csv"] = _csv_bytes(
        headers["cost_by_operation.csv"],
        sorted(cost_rows, key=lambda x: (x[0] != "candidate", x[1], x[2])),
    )
    prows = []
    for surface, value in sorted(pareto.items()):
        for eid in value.get("api", []):
            prows.append([surface, "api", "", eid, True, ""])
        for rate, ids in sorted(value.get("effective", {}).items()):
            for eid in ids:
                prows.append([surface, "effective", rate, eid, True, ""])
    csvs["pareto.csv"] = _csv_bytes(headers["pareto.csv"], sorted(prows))
    lineage = dict(
        price_lineage
        or {
            "original_price_snapshot_sha256": spec.price_snapshot_sha256,
            "current_price_snapshot_sha256": spec.price_snapshot_sha256,
            "parent_report_semantic_sha256": prior_hash,
        }
    )
    aggregate = {
        "status": status,
        "gates": summary["gates"],
        "denominators": human.get("denominators"),
        "input_hashes": input_hashes,
        "reproducibility_hash": repro["quality_performance_sha256"],
        "price_lineage": lineage,
        "million_word_estimate": cost.get("million_word_estimate"),
        "candidates": rows,
        "pareto": pareto,
        "recommendations": recommendations,
        "withheld_reasons": withheld,
        "failures": failures,
    }
    escaped = json.dumps(
        aggregate, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).replace("<", "\\u003c")
    candidate_rows_html = "".join(
        f"<tr><td>{html.escape(str(row.get('entity_id')))}</td><td>{html.escape(str(row.get('surface')))}</td><td>{html.escape(str(row.get('gate_pass')))}</td><td>{html.escape(str(row.get('fidelity_lower95')))}</td><td>{html.escape(str(row.get('api_cost')))}</td></tr>"
        for row in rows
    )
    html_text = (
        """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"><style>body{font:14px sans-serif;margin:2rem}table{border-collapse:collapse}td,th{padding:.3rem;border:1px solid #ccc}</style></head><body><h1>Benchmark report</h1><h2>Status</h2><p id="status"></p><h2>Candidates and conservative evidence</h2><table><thead><tr><th>Entity</th><th>Surface</th><th>Gate</th><th>Fidelity lower95</th><th>API cost</th></tr></thead><tbody>"""
        + candidate_rows_html
        + """</tbody></table><h2>Publication evidence</h2><pre id="evidence"></pre><script>const report="""
        + escaped
        + """;document.getElementById("status").textContent=report.status;document.getElementById("evidence").textContent=JSON.stringify({gates:report.gates,denominators:report.denominators,input_hashes:report.input_hashes,price_lineage:report.price_lineage,million_word_estimate:report.million_word_estimate,pareto:report.pareto,recommendations:report.recommendations,withheld_reasons:report.withheld_reasons,failures:report.failures},null,2);</script></body></html>"""
    )
    outputs = {
        "summary.json": _json_bytes(summary),
        "reproducibility.json": _json_bytes(repro),
        **csvs,
        "report.html": html_text.encode("utf-8"),
    }
    return summary, repro, outputs


def _manifest(request_hash: str, outputs: Mapping[str, bytes]) -> dict[str, Any]:
    records = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in sorted(outputs.items())
    }
    repro = json.loads(outputs["reproducibility.json"].decode("utf-8"))
    semantic = _sha({"request_sha256": request_hash, "files": records})
    manifest = {
        "schema_version": 1,
        "request_sha256": request_hash,
        "report_semantic_sha256": semantic,
        "files": records,
        "input_hashes": repro.get("input_hashes"),
        "report_spec_sha256": _sha(repro.get("report_spec")),
        "original_price_snapshot_sha256": repro.get(
            "original_price_snapshot_sha256",
            repro.get("report_spec", {}).get("price_snapshot_sha256"),
        ),
        "current_price_snapshot_sha256": repro.get(
            "current_price_snapshot_sha256",
            repro.get("report_spec", {}).get("price_snapshot_sha256"),
        ),
        "integration_sha256": (repro.get("integration_facts") or {}).get("integration_sha256")
        if isinstance(repro.get("integration_facts"), Mapping)
        else None,
        "integration_complete_sha256": (repro.get("integration_facts") or {}).get(
            "integration_complete_sha256"
        )
        if isinstance(repro.get("integration_facts"), Mapping)
        else None,
        "parent_report_semantic_sha256": repro.get("prior_report_hash"),
        "completed": True,
    }
    manifest["report_sha256"] = _sha({k: v for k, v in manifest.items() if k != "report_sha256"})
    return manifest


def _write_create_only(
    out_dir: Path, outputs: dict[str, bytes], request_hash: str
) -> dict[str, Any]:
    manifest = _manifest(request_hash, outputs)
    all_outputs = {**outputs, "report_manifest.json": _json_bytes(manifest)}
    if out_dir.exists():
        try:
            old = validate_report(out_dir)
        except Exception as exc:
            raise ReportError(f"existing report invalid: {exc}") from exc
        if old.get("request_sha256") == request_hash:
            if (out_dir / "report_manifest.json").read_bytes() != _json_bytes(manifest) or any(
                (out_dir / name).read_bytes() != data
                for name, data in all_outputs.items()
                if name != "report_manifest.json"
            ):
                raise ReportError("existing report bytes differ for identical request")
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            return {
                "out_dir": str(out_dir),
                "status": summary.get("status"),
                "report_sha256": old.get("report_sha256"),
                "request_sha256": request_hash,
                "no_op": True,
            }
        raise ReportError("output already exists for a different request")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=str(out_dir.parent)))
    try:
        for name, data in all_outputs.items():
            p = tmp / name
            p.write_bytes(data)
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        dfd = os.open(tmp, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
        os.replace(tmp, out_dir)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    summary = json.loads(outputs["summary.json"].decode("utf-8"))
    return {
        "out_dir": str(out_dir),
        "status": summary.get("status"),
        "report_sha256": manifest["report_sha256"],
        "request_sha256": request_hash,
        "no_op": False,
    }


def build_report(
    corpus_dir: Path,
    run_dir: Path,
    preparation_dir: Path,
    pack_dir: Path,
    evaluation_dir: Path,
    price_path: Path,
    report_spec_path: Path,
    out_dir: Path,
    *,
    integration_path: Path | None = None,
) -> dict[str, Any]:
    try:
        spec = load_report_spec(Path(report_spec_path))
        human = analyze_human(Path(corpus_dir), Path(pack_dir), Path(evaluation_dir), spec)
        cost = analyze_cost_system(
            Path(corpus_dir), Path(run_dir), Path(preparation_dir), Path(price_path), human, spec
        )
        hashes = _input_hashes(spec, human, cost)
        integration, ihash = _validate_integration(
            Path(integration_path) if integration_path else None, spec, _known_candidates(human)
        )
        request = {
            "operation": "build",
            "report_spec": _spec_json(spec),
            "input_hashes": hashes,
            "integration_sha256": None if ihash is None else ihash["integration_sha256"],
        }
        return _write_create_only(
            Path(out_dir), _assemble(spec, human, cost, integration, hashes)[2], _sha(request)
        )
    except (ReportError, HumanAnalysisError, CostAnalysisError, OSError, ValueError) as exc:
        raise _err(exc) from exc


def validate_report(report_dir: Path) -> dict[str, Any]:
    root = Path(report_dir)
    if not root.is_dir() or root.is_symlink():
        raise ReportError("report directory invalid")
    names = sorted(p.name for p in root.iterdir())
    if names != sorted(FILES):
        raise ReportError("report file set mismatch")
    manifest = json.loads((root / "report_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(FILES) - {"report_manifest.json"}:
        raise ReportError("manifest file map mismatch")
    for name, record in files.items():
        p = root / name
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or p.is_symlink()
            or not p.is_file()
            or _raw_hash(p) != record["sha256"]
            or p.stat().st_size != record["size"]
        ):
            raise ReportError(f"report file hash/size mismatch: {name}")
    expected_semantic = _sha({"request_sha256": manifest.get("request_sha256"), "files": files})
    expected_report = _sha({k: v for k, v in manifest.items() if k != "report_sha256"})
    if (
        manifest.get("report_semantic_sha256") != expected_semantic
        or manifest.get("report_sha256") != expected_report
        or manifest.get("completed") is not True
    ):
        raise ReportError("manifest hash mismatch")
    return manifest


def reprice_report(report_dir: Path, new_price_path: Path, out_dir: Path) -> dict[str, Any]:
    try:
        from trans_novel.benchmark.pricing import load_price_snapshot
        from trans_novel.benchmark.report_cost import _price_hash

        old = validate_report(Path(report_dir))
        root = Path(report_dir)
        repro = json.loads((root / "reproducibility.json").read_text(encoding="utf-8"))
        spec = ReportSpec.model_validate(repro["report_spec"])
        snapshot = load_price_snapshot(Path(new_price_path))
        new_hash = _price_hash(snapshot)
        spec = spec.model_copy(update={"price_snapshot_sha256": new_hash})
        human = repro["human_facts"]
        cost = reprice_cost_system(
            repro["normalized_pricing_facts"],
            Path(new_price_path),
            human,
            spec.editor_hourly_rates,
            bootstrap_seed=spec.bootstrap_seed,
            bootstrap_replicates=spec.bootstrap_replicates,
        )
        hashes = dict(repro["input_hashes"])
        hashes["price_snapshot_sha256"] = new_hash
        integration = repro.get("integration_facts")
        request = {
            "prior_report_sha256": old["report_semantic_sha256"],
            "new_price_snapshot_sha256": new_hash,
            "report_spec_with_new_price_hash": _sha(_spec_json(spec)),
            "input_hashes": hashes,
            "integration_sha256": None
            if integration is None
            else integration.get("integration_sha256"),
        }
        _, _, outputs = _assemble(
            spec,
            human,
            cost,
            integration,
            hashes,
            old["report_semantic_sha256"],
            {
                "original_price_snapshot_sha256": repro.get(
                    "original_price_snapshot_sha256", repro["report_spec"]["price_snapshot_sha256"]
                ),
                "current_price_snapshot_sha256": new_hash,
                "parent_report_semantic_sha256": old["report_semantic_sha256"],
            },
        )
        new_repro = json.loads(outputs["reproducibility.json"].decode("utf-8"))
        new_repro["original_price_snapshot_sha256"] = repro.get(
            "original_price_snapshot_sha256", repro["report_spec"]["price_snapshot_sha256"]
        )
        new_repro["current_price_snapshot_sha256"] = new_hash
        outputs["reproducibility.json"] = _json_bytes(new_repro)
        for name in PRICE_INDEPENDENT:
            outputs[name] = (root / name).read_bytes()
        return _write_create_only(Path(out_dir), outputs, _sha(request))
    except (ReportError, CostAnalysisError, OSError, ValueError) as exc:
        raise _err(exc) from exc
