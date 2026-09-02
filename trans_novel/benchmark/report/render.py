"""Deterministic escaped HTML rendering for benchmark reports."""

from __future__ import annotations

import html
from typing import Any


def render_html(summary: dict[str, Any], comparison: dict[str, Any], costs: dict[str, Any]) -> str:
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


__all__ = ["render_html"]
