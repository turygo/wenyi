"""Public report workflow facade and artifact publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from trans_novel.benchmark.artifacts import atomic_json, canonical_json, sha256_bytes
from trans_novel.benchmark.contracts import RUN_SCHEMA_VERSION
from trans_novel.benchmark.pricing import load_price_snapshot
from trans_novel.benchmark.report.metrics import aggregate_costs, aggregate_system, aggregate_usage
from trans_novel.benchmark.report.render import render_html
from trans_novel.benchmark.review import ReviewArtifactError, validate_review_artifacts

__all__ = ["ReportError", "build_report", "validate_report"]


class ReportError(ValueError):
    """Frozen benchmark facts cannot produce a valid report."""


def _read_report_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReportError(f"invalid JSON artifact {path}: {error}") from error


def _read_report_jsonl(path: Path) -> list[dict[str, Any]]:
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
        review_status = validate_review_artifacts(review)
    except ReviewArtifactError as error:
        raise ReportError(str(error)) from error
    if review_status["status"] != "complete":
        raise ReportError("report requires a completed review")
    run_manifest_path = run / "run.json"
    run_manifest = _read_report_json(run_manifest_path)
    run_state = _read_report_json(run / "run_state.json")
    if (
        run_manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or run_state.get("status") != "completed"
    ):
        raise ReportError("report requires a completed production benchmark")
    run_sha = sha256_bytes(run_manifest_path.read_bytes())
    review_manifest = _read_report_json(review / "review.json")
    if review_manifest.get("run_sha256") != run_sha or review_manifest.get(
        "benchmark_id"
    ) != run_manifest.get("benchmark_id"):
        raise ReportError("review does not belong to this benchmark run")
    rows = _read_report_json(run / "candidates.json")
    if not isinstance(rows, list) or not rows:
        raise ReportError("candidate artifact index is empty")
    comparison = _read_report_json(review / "comparison.json")
    review_summary = _read_report_json(review / "summary.json")
    if comparison.get("review_sha256") != review_status["review_sha256"]:
        raise ReportError("review comparison identity mismatch")
    run_candidates = {row.get("candidate_id") for row in rows}
    if run_candidates != set(comparison.get("candidates", {})):
        raise ReportError("review candidate set does not match the benchmark run")
    try:
        load_price_snapshot(price)
        aggregate_usage(rows)
    except ValueError as error:
        raise ReportError(str(error)) from error
    loaded_attempts: dict[int, list[dict[str, Any]]] = {}
    ordered_indexes = sorted(range(len(rows)), key=lambda index: rows[index]["candidate_id"])
    for index in ordered_indexes:
        artifact = rows[index]
        telemetry_path = run / str(artifact.get("telemetry_path"))
        expected_sha = artifact.get("telemetry_sha256")
        if (
            not telemetry_path.is_file()
            or not isinstance(expected_sha, str)
            or sha256_bytes(telemetry_path.read_bytes()) != expected_sha
        ):
            raise ReportError("candidate telemetry is missing or changed")
        loaded_attempts[index] = _read_report_jsonl(telemetry_path)
    telemetry_attempts = [loaded_attempts[index] for index in range(len(rows))]
    try:
        costs = aggregate_costs(rows, telemetry_attempts, price)
        system = aggregate_system(rows)
    except (TypeError, ValueError, KeyError) as error:
        raise ReportError(str(error)) from error
    integration = (
        _read_report_json(Path(integration_path).expanduser().resolve())
        if integration_path
        else None
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
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "comparison.json", comparison)
    atomic_json(out / "costs.json", costs)
    atomic_json(out / "system.json", system)
    if integration is not None:
        atomic_json(out / "integration.json", integration)
    (out / "findings.jsonl").write_bytes((review / "findings.jsonl").read_bytes())
    (out / "report.html").write_text(render_html(summary, comparison, costs), encoding="utf-8")
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
    atomic_json(out / "report.json", manifest)
    return {**manifest, "out_dir": str(out)}


def validate_report(report_dir: str | os.PathLike[str]) -> dict[str, Any]:
    report = Path(report_dir).expanduser().resolve()
    manifest = _read_report_json(report / "report.json")
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
