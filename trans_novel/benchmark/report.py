"""Deterministic report over production benchmark artifacts and subagent review facts."""

from __future__ import annotations

import html
import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from trans_novel.benchmark.corpus import canonical_json, sha256_bytes
from trans_novel.benchmark.pricing import CostQuote, load_price_snapshot, quote_usage
from trans_novel.benchmark.review import ReviewArtifactError, validate_review
from trans_novel.llm.usage import merge_usage_summaries

__all__ = ["ReportError", "build_report", "validate_report"]


class ReportError(ValueError):
    """Frozen benchmark facts cannot produce a valid report."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReportError(f"invalid JSON artifact {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as error:
        raise ReportError(f"invalid JSONL artifact {path}: {error}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ReportError(f"JSONL rows must be objects: {path}")
    return rows


def _price_band(snapshot: Any, model_id: str, started_at: object) -> str:
    model = snapshot.models.get(model_id)
    if model is None:
        return "all"
    bands = {rule.time_band for rule in model.rules}
    if "all" in bands:
        return "all"
    if bands != {"peak", "off_peak"} or not isinstance(started_at, str):
        raise ReportError(f"unsupported price time bands for {model_id}")
    try:
        instant = datetime.fromisoformat(started_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as error:
        raise ReportError("telemetry started_at is invalid") from error
    peak = instant.weekday() < 5 and (1 <= instant.hour < 4 or 6 <= instant.hour < 10)
    return "peak" if peak else "off_peak"


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


def _aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate_id")
        usage = row.get("usage")
        if not isinstance(candidate, str) or not isinstance(usage, dict):
            raise ReportError("candidate usage provenance is invalid")
        try:
            result[candidate] = merge_usage_summaries(result.get(candidate, {}), usage)
        except ValueError as error:
            raise ReportError(f"candidate usage is invalid: {error}") from error
    return result


def _costs(run: Path, rows: list[dict[str, Any]], price_path: Path) -> dict[str, Any]:
    snapshot = load_price_snapshot(price_path)
    usage = _aggregate_usage(rows)
    candidates: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    for candidate, artifacts in sorted(grouped.items()):
        by_model: dict[str, dict[str, Any]] = {}
        total: Decimal | None = Decimal(0)
        for artifact in artifacts:
            telemetry_path = run / str(artifact.get("telemetry_path"))
            expected_sha = artifact.get("telemetry_sha256")
            if (
                not telemetry_path.is_file()
                or not isinstance(expected_sha, str)
                or sha256_bytes(telemetry_path.read_bytes()) != expected_sha
            ):
                raise ReportError("candidate telemetry is missing or changed")
            for attempt in _read_jsonl(telemetry_path):
                provider = attempt.get("provider")
                model = attempt.get("resolved_model") or attempt.get("requested_model")
                if not isinstance(provider, str) or not isinstance(model, str):
                    raise ReportError("telemetry model provenance is invalid")
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
                    time_band = _price_band(snapshot, model_id, attempt.get("started_at"))
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
        "price_sha256": sha256_bytes(price_path.read_bytes()),
        "candidates": candidates,
    }


def _system(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), []).append(row)
    for candidate, artifacts in sorted(grouped.items()):
        totals = merge_usage_summaries(
            {},
            _aggregate_usage(artifacts)[candidate],
        )
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


def _render_html(summary: dict[str, Any], comparison: dict[str, Any], costs: dict[str, Any]) -> str:
    winner = html.escape(str(summary.get("winner") or "none"))
    rows = []
    for candidate in comparison.get("ranking", []):
        quality = comparison["candidates"][candidate]
        cost = costs.get("candidates", {}).get(candidate, {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate)}</td>"
            f"<td>{quality['severity_counts']['critical']}</td>"
            f"<td>{quality['severity_counts']['major']}</td>"
            f"<td>{quality['severity_counts']['minor']}</td>"
            f"<td>{quality['weighted_errors_per_10k']:.3f}</td>"
            f"<td>{quality['wins']}</td>"
            f"<td>{html.escape(str(cost.get('api_cost')))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><meta charset='utf-8'><title>Wenyi Benchmark Report</title>"
        "<style>body{font-family:system-ui;max-width:90rem;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.45rem}</style>"
        f"<h1>Wenyi Benchmark</h1><p>Winner: <strong>{winner}</strong></p>"
        "<table><thead><tr><th>Candidate</th><th>Critical</th><th>Major</th>"
        "<th>Minor</th><th>Weighted errors / 10k</th><th>Wins</th><th>API cost</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        "<p>See comparison.json and findings.jsonl for evidence-backed reasons.</p>"
    )


def build_report(
    run_dir: str | os.PathLike[str],
    review_dir: str | os.PathLike[str],
    price_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    integration_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    review = Path(review_dir).expanduser().resolve()
    price = Path(price_path).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    if out.exists():
        raise ReportError("report output already exists")
    try:
        review_status = validate_review(review)
    except ReviewArtifactError as error:
        raise ReportError(str(error)) from error
    if review_status["status"] != "complete":
        raise ReportError("report requires a completed review")
    run_manifest_path = run / "run.json"
    run_manifest = _read_json(run_manifest_path)
    run_state = _read_json(run / "run_state.json")
    if run_manifest.get("schema_version") != 2 or run_state.get("status") != "completed":
        raise ReportError("report requires a completed production benchmark")
    run_sha = sha256_bytes(run_manifest_path.read_bytes())
    review_manifest = _read_json(review / "review.json")
    if review_manifest.get("run_sha256") != run_sha or review_manifest.get(
        "benchmark_id"
    ) != run_manifest.get("benchmark_id"):
        raise ReportError("review does not belong to this benchmark run")
    rows = _read_json(run / "candidates.json")
    if not isinstance(rows, list) or not rows:
        raise ReportError("candidate artifact index is empty")
    comparison = _read_json(review / "comparison.json")
    review_summary = _read_json(review / "summary.json")
    if comparison.get("review_sha256") != review_status["review_sha256"]:
        raise ReportError("review comparison identity mismatch")
    run_candidates = {row.get("candidate_id") for row in rows}
    if run_candidates != set(comparison.get("candidates", {})):
        raise ReportError("review candidate set does not match the benchmark run")
    costs = _costs(run, rows, price)
    system = _system(rows)
    integration = (
        _read_json(Path(integration_path).expanduser().resolve()) if integration_path else None
    )
    status = "final"
    if any(
        value.get("failed_attempts") or not value.get("all_outputs_present")
        for value in system["candidates"].values()
    ) or any(value.get("api_cost") is None for value in costs["candidates"].values()):
        status = "provisional"
    summary = {
        "schema_version": 1,
        "status": status,
        "winner": comparison.get("winner") if status == "final" else None,
        "reviewed_units": review_summary.get("reviewed_units"),
        "finding_count": review_summary.get("finding_count"),
        "ranking": comparison.get("ranking", []),
        "run_sha256": run_sha,
        "review_sha256": review_status["review_sha256"],
        "price_sha256": sha256_bytes(price.read_bytes()),
        "integration_sha256": (
            None
            if integration_path is None
            else sha256_bytes(Path(integration_path).expanduser().resolve().read_bytes())
        ),
    }
    out.mkdir(parents=True)
    _atomic_json(out / "summary.json", summary)
    _atomic_json(out / "comparison.json", comparison)
    _atomic_json(out / "costs.json", costs)
    _atomic_json(out / "system.json", system)
    if integration is not None:
        _atomic_json(out / "integration.json", integration)
    (out / "findings.jsonl").write_bytes((review / "findings.jsonl").read_bytes())
    (out / "report.html").write_text(_render_html(summary, comparison, costs), encoding="utf-8")
    files = {
        str(path.relative_to(out)): sha256_bytes(path.read_bytes())
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "status": status,
        "files": files,
        "report_sha256": sha256_bytes(canonical_json(files).encode("utf-8")),
    }
    _atomic_json(out / "report.json", manifest)
    return {**manifest, "out_dir": str(out)}


def validate_report(report_dir: str | os.PathLike[str]) -> dict[str, Any]:
    report = Path(report_dir).expanduser().resolve()
    manifest = _read_json(report / "report.json")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ReportError("report file manifest invalid")
    for relative, digest in files.items():
        path = report / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            raise ReportError("report artifact is missing or changed")
    if manifest.get("report_sha256") != sha256_bytes(canonical_json(files).encode("utf-8")):
        raise ReportError("report hash mismatch")
    return manifest
