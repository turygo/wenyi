"""Phase 8 decoding and statistics for completed human evaluations."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from trans_novel.benchmark.corpus import canonical_json, sha256_bytes, validate_corpus
from trans_novel.benchmark.evaluation import (
    _pair_adjudication,
    _responses,
    build_units,
    validate_pack,
)
from trans_novel.benchmark.report_schema import ReportSpec
from trans_novel.benchmark.report_stats import (
    StatisticsError,
    bootstrap_bradley_terry,
    fit_bradley_terry,
    hierarchical_bootstrap,
    krippendorff_alpha,
    levenshtein_distance,
    levenshtein_ratio,
    wilson_upper95,
)
from trans_novel.benchmark.schema import EvaluationSpec

__all__ = ["HumanAnalysisError", "analyze_human"]

_DIMENSIONS = (
    "fidelity",
    "naturalness",
    "style_voice",
    "consistency",
    "context_handling",
    "readability",
    "format_integrity",
)
_WEIGHTS = {
    "fidelity": 0.30,
    "naturalness": 0.20,
    "style_voice": 0.15,
    "consistency": 0.15,
    "context_handling": 0.10,
    "readability": 0.05,
    "format_integrity": 0.05,
}
_SEVERITIES = ("critical", "major", "minor")
_TYPES = (
    "mistranslation",
    "omission",
    "addition",
    "hallucination",
    "terminology",
    "named_entity",
    "pronoun_reference",
    "style_register",
    "fluency",
    "formatting",
)
_STRENGTHS = ("first_much", "first_slight", "tie", "second_slight", "second_much")


class HumanAnalysisError(ValueError):
    """A completed evaluation violates the Phase 8 human report contract."""


def _fail(message: str) -> None:
    raise HumanAnalysisError(message)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid JSON: {path}")
        raise AssertionError from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _fail(f"invalid JSONL: {path}")
        raise AssertionError from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"invalid JSONL line {number}: {path}")
            raise AssertionError from exc
        if not isinstance(value, dict) or canonical_json(value) + "\n" != line + "\n":
            _fail(f"noncanonical JSONL: {path}:{number}")
        rows.append(value)
    return rows


def _raw(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        _fail(f"missing evaluation file: {path}")
        return b""


def _mean(values: list[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def _score(raw: float | None) -> float | None:
    return None if raw is None else (raw - 1.0) / 4.0 * 100.0


def _ci(
    rows: list[dict[str, Any]],
    value: Callable[[list[dict[str, Any]]], float],
    seed: int,
    replicates: int,
) -> tuple[float | None, float | None]:
    if not rows:
        return None, None
    try:
        # report_stats supplies one list per sampled book occurrence.
        def statistic(occurrences):
            return statistics.fmean(value(book_rows) for book_rows in occurrences)

        result = hierarchical_bootstrap(rows, seed=seed, replicates=replicates, statistic=statistic)
        return result.lower95, result.upper95
    except (StatisticsError, ValueError):
        return None, None


def _hier(
    rows: list[dict[str, Any]], field: str, seed: int, replicates: int
) -> tuple[float | None, float | None]:
    return _ci(
        rows,
        lambda occurrence: statistics.fmean(float(row[field]) for row in occurrence),
        seed,
        replicates,
    )


def _metric_scope(
    kind: str,
    candidate: str | None = None,
    surface: str | None = None,
    strategy: str | None = None,
    book: str | None = None,
) -> str:
    parts = [f"task={kind}"]
    if candidate is not None:
        parts.append(f"candidate={candidate}")
    if surface is not None:
        parts.append(f"surface={surface}")
    if strategy is not None:
        parts.append(f"strategy={strategy}")
    if book is not None:
        parts.append(f"book={book}")
    return "/".join(parts)


def _add_insufficient(out: set[tuple[str, str]], scope: str, reason: str) -> None:
    out.add((scope, reason))


def _norm_quote(value: Any) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).strip().split()).casefold()


def _relation_rows(
    relations: list[dict[str, Any]], book: str | None = None
) -> list[dict[str, Any]]:
    return [row for row in relations if book is None or row["book_id"] == book]


def _empty_composite() -> dict[str, Any]:
    return {
        "value": None,
        "lower95": None,
        "upper95": None,
        "n_ratings": 0,
        "n_units": 0,
        "source_words": 0,
    }


def _absolute(
    relations: list[dict[str, Any]],
    candidates: list[str],
    surfaces: list[str],
    books: list[str],
    insufficient: set[tuple[str, str]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in sorted(candidates):
        result[candidate] = {}
        for surface in sorted(surfaces):
            selected = [
                r for r in relations if r["candidate"] == candidate and r["surface"] == surface
            ]
            by_book: dict[str, Any] = {}
            available: list[dict[str, Any]] = []
            for book in books:
                rows = _relation_rows(selected, book)
                if not rows:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("absolute", candidate, surface, book=book),
                        "missing_scope",
                    )
                    _add_insufficient(
                        insufficient,
                        _metric_scope("absolute", candidate, surface, book=book),
                        "insufficient_book_sample",
                    )
                elif len(rows) < 10:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("absolute", candidate, surface, book=book),
                        "insufficient_book_sample",
                    )
                dims: dict[str, Any] = {}
                for dim in _DIMENSIONS:
                    vals = [float(resp[dim]) for row in rows for resp in row["responses"]]
                    raw = _mean(vals)
                    dims[dim] = {
                        "raw_mean": raw,
                        "score_100": _score(raw),
                        "n_ratings": len(vals),
                        "n_units": len(rows),
                        "source_words": sum(r["words"] for r in rows),
                    }
                composites = [float(resp["composite"]) for row in rows for resp in row["responses"]]
                comp = _empty_composite()
                comp.update(
                    {
                        "value": _mean(composites),
                        "n_ratings": len(composites),
                        "n_units": len(rows),
                        "source_words": sum(r["words"] for r in rows),
                    }
                )
                by_book[book] = {"dimensions": dims, "composite": comp}
                if rows:
                    available.extend(rows)
            macro_dims: dict[str, Any] = {}
            for dim in _DIMENSIONS:
                values = []
                ci_values = []
                for book in books:
                    rows = _relation_rows(selected, book)
                    if rows:
                        values.append(
                            {
                                "book_id": book,
                                "unit_id": f"{candidate}:{surface}:{book}:{dim}",
                                "value": statistics.fmean(
                                    float(resp[dim]) for row in rows for resp in row["responses"]
                                ),
                            }
                        )
                        ci_values.extend(
                            {
                                "book_id": row["book_id"],
                                "unit_id": row["unit_id"],
                                "value": statistics.fmean(
                                    float(resp[dim]) for resp in row["responses"]
                                ),
                            }
                            for row in rows
                        )
                raw = _mean([x["value"] for x in values])
                lower, upper = _hier(ci_values, "value", seed, replicates)
                macro_dims[dim] = {
                    "raw_mean": raw,
                    "score_100": _score(raw),
                    "lower95": _score(lower),
                    "upper95": _score(upper),
                    "n_ratings": sum(len(r["responses"]) for r in available),
                    "n_units": len(available),
                    "source_words": sum(r["words"] for r in available),
                    "n_books": len(values),
                }
            book_values = []
            relation_values = []
            for book in books:
                rows = _relation_rows(selected, book)
                if rows:
                    book_values.append(
                        {
                            "book_id": book,
                            "unit_id": f"{candidate}:{surface}:{book}:composite",
                            "value": statistics.fmean(
                                float(x["composite"]) for r in rows for x in r["responses"]
                            ),
                        }
                    )
                    relation_values.extend(
                        {
                            "book_id": r["book_id"],
                            "unit_id": r["unit_id"],
                            "value": statistics.fmean(
                                float(x["composite"]) for x in r["responses"]
                            ),
                        }
                        for r in rows
                    )
            value = _mean([x["value"] for x in book_values])
            low, high = _hier(relation_values, "value", seed, replicates)
            macro_comp = _empty_composite()
            macro_comp.update(
                {
                    "value": value,
                    "lower95": low,
                    "upper95": high,
                    "n_ratings": sum(len(r["responses"]) for r in available),
                    "n_units": len(available),
                    "source_words": sum(r["words"] for r in available),
                    "n_books": len(book_values),
                }
            )
            result[candidate][surface] = {
                "by_book": by_book,
                "macro": {"dimensions": macro_dims, "composite": macro_comp},
            }
    return result


def _mqm(
    relations: list[dict[str, Any]],
    candidates: list[str],
    surfaces: list[str],
    books: list[str],
    insufficient: set[tuple[str, str]],
    pending_units: set[tuple[str, str, str]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in sorted(candidates):
        output[candidate] = {}
        for surface in sorted(surfaces):
            selected = [
                r for r in relations if r["candidate"] == candidate and r["surface"] == surface
            ]
            by_book: dict[str, Any] = {}
            for book in books:
                rows = _relation_rows(selected, book)
                if not rows:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("mqm", candidate, surface, book=book),
                        "missing_scope",
                    )
                by_book[book] = _mqm_scope(rows)
            available = list(selected)
            if not available:
                _add_insufficient(
                    insufficient, _metric_scope("mqm", candidate, surface), "missing_scope"
                )
            macro = _mqm_scope(available)
            book_scopes = [
                _mqm_scope(_relation_rows(available, book))
                for book in sorted({r["book_id"] for r in available})
            ]
            for section in ("raw", "agreed"):
                for key in _SEVERITIES:
                    rates = [
                        scope[section]["severity"][key]["rate_per_10k"]
                        for scope in book_scopes
                        if scope[section]["severity"][key]["rate_per_10k"] is not None
                    ]
                    macro[section]["severity"][key]["rate_per_10k"] = _mean(rates)
                for key in _TYPES:
                    rates = [
                        scope[section]["type"][key]["rate_per_10k"]
                        for scope in book_scopes
                        if scope[section]["type"][key]["rate_per_10k"] is not None
                    ]
                    macro[section]["type"][key]["rate_per_10k"] = _mean(rates)
            macro["weighted_points_per_10k"] = _mean(
                [
                    scope["weighted_points_per_10k"]
                    for scope in book_scopes
                    if scope["weighted_points_per_10k"] is not None
                ]
            )
            low, high = _mqm_ci(available, "weighted_points_per_10k", seed, replicates)
            major_low, major_high = _mqm_ci(available, "major_rate", seed, replicates)
            macro["weighted_points_lower95"], macro["weighted_points_upper95"] = low, high
            macro["major_rate_lower95"], macro["major_rate_upper95"] = major_low, major_high
            output[candidate][surface] = {"by_book": by_book, "macro": macro}
            for row in selected:
                if row.get("discordant"):
                    pending_units.add((candidate, surface, row["unit_id"]))
    return output


def _mqm_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    words = sum(r["words"] for r in rows)
    units = len(rows)
    raw = dict.fromkeys(_SEVERITIES, 0)
    agreed = dict.fromkeys(_SEVERITIES, 0)
    raw_types = dict.fromkeys(_TYPES, 0)
    agreed_types = dict.fromkeys(_TYPES, 0)
    events = {
        severity: sum(1 for r in rows if r["agreed_severity"][severity] > 0)
        for severity in _SEVERITIES
    }
    for row in rows:
        for severity, count in row["raw_severity"].items():
            raw[severity] += count
        for severity, count in row["agreed_severity"].items():
            agreed[severity] += count
        for kind, count in row["raw_types"].items():
            raw_types[kind] += count
        for kind, count in row["agreed_types"].items():
            agreed_types[kind] += count

    def rates(counts: dict[str, int]) -> dict[str, dict[str, Any]]:
        return {
            key: {"count": value, "rate_per_10k": None if words == 0 else value / words * 10000.0}
            for key, value in counts.items()
        }

    weighted = 25 * agreed["critical"] + 5 * agreed["major"] + agreed["minor"]
    return {
        "source_words": words,
        "n_units": units,
        "n_books": len({r["book_id"] for r in rows}),
        "raw": {"severity": rates(raw), "type": rates(raw_types)},
        "agreed": {"severity": rates(agreed), "type": rates(agreed_types)},
        "weighted_points_per_10k": None if words == 0 else weighted / words * 10000.0,
        "event_wilson_upper95": {
            severity: wilson_upper95(events[severity], units) for severity in _SEVERITIES
        },
        "pending_adjudication_count": sum(1 for r in rows if r.get("discordant")),
    }


def _pairwise(
    relations: list[dict[str, Any]],
    candidates: list[str],
    surfaces: list[str],
    spec: ReportSpec,
    insufficient: set[tuple[str, str]],
    pending_pairs: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for surface in sorted(surfaces):
        rows = [r for r in relations if r["surface"] == surface]
        outcomes = [
            {
                "book_id": r["book_id"],
                "unit_id": r["unit_id"],
                "candidate_a": r["first"],
                "candidate_b": r["second"],
                "score_a": score,
            }
            for r in rows
            for score in r["scores"]
        ]
        strengths = dict.fromkeys(_STRENGTHS, 0)
        for row in rows:
            for strength in row["strengths"]:
                strengths[strength] += 1
            if row.get("discordant"):
                pending_pairs.add(row["relation_key"])
        comparisons = len(rows)
        books_count = len({r["book_id"] for r in rows})
        fit = None
        boot = None
        try:
            fit = fit_bradley_terry(
                candidates,
                [(r["first"], r["second"], score) for r in rows for score in r["scores"]],
            )
        except StatisticsError as exc:
            if "disconnected" in str(exc):
                _add_insufficient(
                    insufficient,
                    _metric_scope("pairwise", surface=surface),
                    "disconnected_pairwise",
                )
            else:
                raise HumanAnalysisError(f"Bradley-Terry point fit failed: {exc}") from exc
        if fit is not None:
            try:
                boot = bootstrap_bradley_terry(
                    outcomes,
                    candidates,
                    seed=spec.bootstrap_seed,
                    replicates=spec.bootstrap_replicates,
                )
            except StatisticsError as exc:
                raise HumanAnalysisError(f"Bradley-Terry bootstrap failed: {exc}") from exc
        candidate_values: dict[str, Any] = {}
        for candidate in sorted(candidates):
            ability = None if fit is None else fit.abilities[candidate]
            field = None if fit is None else fit.field_win_probability[candidate]
            aci = None if boot is None else boot.ability_ci[candidate]
            fci = None if boot is None else boot.field_win_ci[candidate]
            candidate_values[candidate] = {
                "ability": ability,
                "ability_lower95": None if aci is None else aci[0],
                "ability_upper95": None if aci is None else aci[1],
                "field_win": field,
                "field_win_lower95": None if fci is None else fci[0],
                "field_win_upper95": None if fci is None else fci[1],
            }
        result[surface] = {
            "candidates": candidate_values,
            "strength_counts": strengths,
            "comparisons": comparisons,
            "ratings": comparisons * 2,
            "units": len({r["unit_id"] for r in rows}),
            "books": books_count,
            "bootstrap": None
            if boot is None
            else {
                "requested": boot.requested,
                "attempted": boot.attempted,
                "discarded": boot.discarded,
            },
        }
    return result


def _mqm_ci(
    rows: list[dict[str, Any]], field: str, seed: int, replicates: int
) -> tuple[float | None, float | None]:
    def statistic(occurrences):
        values = []
        for occurrence in occurrences:
            scope = _mqm_scope(occurrence)
            values.append(
                scope["weighted_points_per_10k"]
                if field == "weighted_points_per_10k"
                else scope["agreed"]["severity"]["major"]["rate_per_10k"] or 0.0
            )
        return statistics.fmean(values)

    try:
        result = hierarchical_bootstrap(rows, seed=seed, replicates=replicates, statistic=statistic)
        return result.lower95, result.upper95
    except (StatisticsError, ValueError):
        return None, None


def _polish(
    relations: list[dict[str, Any]],
    candidates: list[str],
    books: list[str],
    insufficient: set[tuple[str, str]],
    mqm_relations: list[dict[str, Any]] | None = None,
    seed: int = 0,
    replicates: int = 1000,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    active_candidates = sorted({r["candidate"] for r in relations})
    for candidate in active_candidates:
        selected = [r for r in relations if r["candidate"] == candidate]
        by_book: dict[str, Any] = {}
        for book in books:
            rows = _relation_rows(selected, book)
            if not rows:
                _add_insufficient(
                    insufficient, _metric_scope("polish", candidate, book=book), "missing_scope"
                )
            by_book[book] = _polish_scope(rows)
        macro = _polish_scope(selected)
        nonempty_books = [metric for metric in by_book.values() if metric["total"]]
        for field in ("improved_rate", "neutral_rate", "harm_rate", "net"):
            macro[field] = _mean(
                [metric[field] for metric in nonempty_books if metric[field] is not None]
            )
        ci_rows = []
        for relation in selected:
            improved = sum(
                response["outcome"] in ("clearly_improved", "slightly_improved")
                for response in relation["responses"]
            )
            harm = sum(
                response["outcome"] in ("fluent_but_semantic_damage", "quality_declined")
                for response in relation["responses"]
            )
            ci_rows.append(
                {
                    "book_id": relation["book_id"],
                    "unit_id": relation["unit_id"],
                    "improved": improved / len(relation["responses"]),
                    "harm": harm / len(relation["responses"]),
                    "net": (improved - harm) / len(relation["responses"]),
                }
            )
        for field in ("improved", "harm", "net"):
            low, high = _hier(ci_rows, field, seed, replicates)
            macro[f"{field}_lower95"], macro[f"{field}_upper95"] = low, high
        output[candidate] = {"by_book": by_book, "macro": macro}
        semantic = [r for r in (mqm_relations or []) if r["candidate"] == candidate]
        output[candidate]["mqm_semantic_harm"] = {
            book: sum(
                1
                for r in semantic
                if r["book_id"] == book
                for key in r.get("agreed_keys", set())
                if key[1] in ("critical", "major")
                and key[2] in ("mistranslation", "omission", "addition", "hallucination")
            )
            for book in books
        }
    return output


def _polish_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"improved": 0, "neutral": 0, "harm": 0}
    split = {"accepted": [], "rejected": []}
    semantic = 0
    for row in rows:
        accepted = None
        if row.get("presentation_positions") and row["presentation_positions"][0]:
            accepted = row["presentation_positions"][0][0].get("polish_accepted")
        for response in row["responses"]:
            outcome = response["outcome"]
            if outcome in ("clearly_improved", "slightly_improved"):
                counts["improved"] += 1
            elif outcome == "no_material_change":
                counts["neutral"] += 1
            else:
                counts["harm"] += 1
            semantic += outcome == "fluent_but_semantic_damage"
            if accepted is not None:
                split["accepted" if accepted else "rejected"].append(outcome)
    total = sum(counts.values())
    result = {
        "improved": counts["improved"],
        "neutral": counts["neutral"],
        "harm": counts["harm"],
        "total": total,
        "improved_rate": None if total == 0 else counts["improved"] / total,
        "neutral_rate": None if total == 0 else counts["neutral"] / total,
        "harm_rate": None if total == 0 else counts["harm"] / total,
        "net": None if total == 0 else (counts["improved"] - counts["harm"]) / total,
        "harm_wilson_upper95": wilson_upper95(counts["harm"], total),
        "semantic_harm": semantic,
        "n_units": len(rows),
        "n_ratings": total,
        "source_words": sum(r["words"] for r in rows),
        "n_books": len({r["book_id"] for r in rows}),
    }
    for key, values in split.items():
        if values:
            result[key] = {
                "total": len(values),
                "improved": sum(v in ("clearly_improved", "slightly_improved") for v in values),
                "neutral": values.count("no_material_change"),
                "harm": sum(
                    v in ("fluent_but_semantic_damage", "quality_declined") for v in values
                ),
            }
    return result


def _context(
    relations: list[dict[str, Any]],
    candidates: list[str],
    books: list[str],
    insufficient: set[tuple[str, str]],
    pending: set[tuple[str, str]],
    seed: int = 0,
    replicates: int = 1000,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in sorted(candidates):
        selected = [r for r in relations if r["candidate"] == candidate]
        by_strategy: dict[str, Any] = {}
        for strategy in ("c0", "c1", "c2"):
            strategy_rows = [r for r in selected if r["strategy"] == strategy]
            by_book = {}
            for book in books:
                rows = _relation_rows(strategy_rows, book)
                if not rows:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("context", candidate, strategy=strategy, book=book),
                        "missing_scope",
                    )
                elif _context_scope(rows)["accuracy"] is None:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("context", candidate, strategy=strategy, book=book),
                        "all_uncertain_context",
                    )
                by_book[book] = _context_scope(rows)
            macro = _context_scope(strategy_rows)
            book_metrics = [
                by_book[book] for book in books if by_book[book]["accuracy"] is not None
            ]
            macro["accuracy"] = _mean([metric["accuracy"] for metric in book_metrics])
            macro["uncertain_rate"] = _mean([metric["uncertain_rate"] for metric in book_metrics])
            macro["n_books"] = len(book_metrics)
            macro["accuracy_lower95"], macro["accuracy_upper95"] = _context_accuracy_ci(
                strategy_rows, seed, replicates
            )
            by_strategy[strategy] = {"by_book": by_book, "macro": macro}
        lifts = {}
        for strategy in ("c1", "c2"):
            c0 = by_strategy["c0"]["macro"]["accuracy"]
            cx = by_strategy[strategy]["macro"]["accuracy"]
            value = None if c0 is None or cx is None else cx - c0
            lift_rows = []
            for unit in sorted({r["unit_id"] for r in selected}):
                left = next(
                    (r for r in selected if r["unit_id"] == unit and r["strategy"] == "c0"), None
                )
                right = next(
                    (r for r in selected if r["unit_id"] == unit and r["strategy"] == strategy),
                    None,
                )
                if left is not None and right is not None:
                    lift_rows.append(
                        {"book_id": left["book_id"], "unit_id": unit, "c0": left, "cx": right}
                    )
            low, high = _context_lift_ci(lift_rows, seed, replicates)
            per_book = {}
            for book in books:
                left = by_strategy["c0"]["by_book"][book]["accuracy"]
                right = by_strategy[strategy]["by_book"][book]["accuracy"]
                per_book[book] = None if left is None or right is None else right - left
            lifts[strategy] = {"value": value, "lower95": low, "upper95": high, "by_book": per_book}
        harm, denom = 0, 0
        for row in selected:
            if row["strategy"] != "c0":
                continue
            pair = next(
                (x for x in selected if x["strategy"] == "c2" and x["unit_id"] == row["unit_id"]),
                None,
            )
            if pair is None:
                continue
            a = _paired(row, pair)
            harm += a[0]
            denom += a[1]
            if a[2]:
                pending.add((candidate, row["unit_id"]))
        output[candidate] = {
            "by_strategy": by_strategy,
            "lift": lifts,
            "harm": {
                "events": harm,
                "total": denom,
                "rate": None if denom == 0 else harm / denom,
                "wilson_upper95": wilson_upper95(harm, denom),
            },
        }
    return output


def _context_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for r in rows for x in r["responses"] if x["judgment"] == "correct")
    incorrect = sum(1 for r in rows for x in r["responses"] if x["judgment"] == "incorrect")
    uncertain = sum(1 for r in rows for x in r["responses"] if x["judgment"] == "uncertain")

    total = correct + incorrect
    return {
        "correct": correct,
        "incorrect": incorrect,
        "uncertain": uncertain,
        "total": correct + incorrect + uncertain,
        "accuracy": None if total == 0 else correct / total,
        "uncertain_rate": None if not rows else uncertain / (correct + incorrect + uncertain),
        "n_units": len(rows),
        "n_ratings": sum(len(r["responses"]) for r in rows),
        "source_words": sum(r["words"] for r in rows),
        "n_books": len({r["book_id"] for r in rows}),
    }


def _context_accuracy_ci(
    rows: list[dict[str, Any]], seed: int, replicates: int
) -> tuple[float | None, float | None]:
    def statistic(occurrences):
        values = []
        for occurrence in occurrences:
            scope = _context_scope(occurrence)
            if scope["accuracy"] is not None:
                values.append(scope["accuracy"])
        return statistics.fmean(values)

    try:
        result = hierarchical_bootstrap(rows, seed=seed, replicates=replicates, statistic=statistic)
        return result.lower95, result.upper95
    except (StatisticsError, ValueError):
        return None, None


def _context_lift_ci(
    rows: list[dict[str, Any]], seed: int, replicates: int
) -> tuple[float | None, float | None]:
    def statistic(occurrences):
        values = []
        for occurrence in occurrences:
            c0 = _context_scope([row["c0"] for row in occurrence])
            cx = _context_scope([row["cx"] for row in occurrence])
            if c0["accuracy"] is not None and cx["accuracy"] is not None:
                values.append(cx["accuracy"] - c0["accuracy"])
        return statistics.fmean(values)

    try:
        result = hierarchical_bootstrap(rows, seed=seed, replicates=replicates, statistic=statistic)
        return result.lower95, result.upper95
    except (StatisticsError, ValueError):
        return None, None


def _paired(c0: dict[str, Any], c2: dict[str, Any]) -> tuple[int, int, bool]:
    by_rater0 = {x["rater_id"]: x["judgment"] for x in c0["responses"]}
    by_rater2 = {x["rater_id"]: x["judgment"] for x in c2["responses"]}
    usable = [
        (by_rater0[r], by_rater2[r])
        for r in sorted(set(by_rater0) & set(by_rater2))
        if by_rater0[r] != "uncertain" and by_rater2[r] != "uncertain"
    ]
    if usable:
        return sum(a == "correct" and b == "incorrect" for a, b in usable), len(usable), False

    def majority(values: list[str]) -> str | None:
        vals = [x for x in values if x in ("correct", "incorrect")]
        if not vals or vals.count("correct") == vals.count("incorrect"):
            return None
        return "correct" if vals.count("correct") > vals.count("incorrect") else "incorrect"

    left, right = majority(list(by_rater0.values())), majority(list(by_rater2.values()))
    return (
        (1 if left == "correct" and right == "incorrect" else 0),
        (1 if left and right else 0),
        left is None or right is None,
    )


def _postedit(
    relations: list[dict[str, Any]],
    candidates: list[str],
    surfaces: list[str],
    books: list[str],
    insufficient: set[tuple[str, str]],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in sorted(candidates):
        output[candidate] = {}
        for surface in sorted(surfaces):
            selected = [
                r for r in relations if r["candidate"] == candidate and r["surface"] == surface
            ]
            by_book = {}
            for book in books:
                rows = _relation_rows(selected, book)
                if not rows:
                    _add_insufficient(
                        insufficient,
                        _metric_scope("postedit", candidate, surface, book=book),
                        "missing_scope",
                    )
                by_book[book] = _postedit_scope(rows)
            macro = _postedit_scope(selected)
            scopes = [by_book[book] for book in books if by_book[book]["n_units"]]
            for field in (
                "active_minutes",
                "minutes_per_10k",
                "edit_distance_mean",
                "edit_ratio_mean",
            ):
                values = [scope[field] for scope in scopes if scope[field] is not None]
                macro[field] = _mean(values)
                low, high = _postedit_ci(selected, field, seed, replicates)
                macro[f"{field}_lower95"], macro[f"{field}_upper95"] = low, high
            output[candidate][surface] = {"by_book": by_book, "macro": macro}
    return output


def _postedit_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    words = sum(r["words"] for r in rows)
    minutes = sum(r["active_ms"] for r in rows) / 60000.0
    distances = [r["distance"] for r in rows]
    ratios = [r["ratio"] for r in rows]
    return {
        "active_minutes": minutes,
        "minutes_per_10k": None if words == 0 else minutes / words * 10000.0,
        "edit_distance_mean": _mean(distances),
        "edit_ratio_mean": _mean(ratios),
        "n_units": len(rows),
        "n_editors": len(rows),
        "source_words": words,
        "n_books": len({r["book_id"] for r in rows}),
    }


def _postedit_ci(
    rows: list[dict[str, Any]], field: str, seed: int, replicates: int
) -> tuple[float | None, float | None]:
    def statistic(occurrences):
        return statistics.fmean(_postedit_scope(occurrence)[field] for occurrence in occurrences)

    try:
        result = hierarchical_bootstrap(rows, seed=seed, replicates=replicates, statistic=statistic)
        return result.lower95, result.upper95
    except (StatisticsError, ValueError):
        return None, None


def _reliability(
    relations: dict[str, list[dict[str, Any]]],
    duplicates: list[tuple[str, str, str, dict[str, Any]]],
) -> dict[str, Any]:
    absolute: dict[str, Any] = {}
    for dim in _DIMENSIONS:
        units = {
            (r["candidate"], r["surface"], r["unit_id"]): {
                x["rater_id"]: x[dim] for x in r["responses"]
            }
            for r in relations.get("absolute", [])
        }
        absolute[dim] = krippendorff_alpha(units, level="ordinal", categories=[1, 2, 3, 4, 5])
    context_units = {
        (r["candidate"], r["strategy"], r["unit_id"]): {
            x["rater_id"]: x["judgment"] for x in r["responses"]
        }
        for r in relations.get("context", [])
    }
    context_values = {
        key: {
            rater: (0 if value == "incorrect" else 2 if value == "correct" else None)
            for rater, value in vals.items()
        }
        for key, vals in context_units.items()
    }
    context_alpha = krippendorff_alpha(context_values, level="ordinal", categories=[0, 2])
    pair_units = {
        r["relation_key"]: {x["rater_id"]: x.get("_winner") for x in r["responses"]}
        for r in relations.get("pairwise", [])
    }
    pair_alpha = krippendorff_alpha(
        pair_units, level="nominal", categories=["first", "tie", "second"]
    )
    mqm_severity_units: dict[Any, dict[str, str]] = {}
    mqm_type_units: dict[Any, dict[str, str]] = {}
    for relation in relations.get("mqm", []):
        for segment_id in relation.get("segment_ids", []):
            severity_row: dict[str, str] = {}
            type_row: dict[str, str] = {}
            for response in relation["responses"]:
                errors = [
                    e for e in response.get("errors", []) if e.get("segment_id") == segment_id
                ]
                severity_row[response["rater_id"]] = _highest(errors)
                type_row[response["rater_id"]] = (
                    "+".join(sorted({e["type"] for e in errors})) if errors else "none"
                )
            key = (relation["candidate"], relation["surface"], relation["unit_id"], segment_id)
            mqm_severity_units[key] = severity_row
            mqm_type_units[key] = type_row
    type_categories = sorted(
        {"none", *(value for row in mqm_type_units.values() for value in row.values())}
    )
    mqm_sev = krippendorff_alpha(
        mqm_severity_units, level="ordinal", categories=["none", "minor", "major", "critical"]
    )
    mqm_type = krippendorff_alpha(mqm_type_units, level="nominal", categories=type_categories)
    result: dict[str, Any] = {dim: absolute[dim] for dim in _DIMENSIONS}
    result.update(
        {
            "pairwise_winner": pair_alpha,
            "context_correctness": context_alpha,
            "mqm_severity": mqm_sev,
            "mqm_type": mqm_type,
        }
    )
    dup_out: dict[str, Any] = {}
    for kind, name, _, data in duplicates:
        exact = sum(1 for x, y in data["pairs"] if x == y)
        adjacent = sum(
            1
            for x, y in data["pairs"]
            if x == y
            or (isinstance(x, int | float) and isinstance(y, int | float) and abs(x - y) <= 1)
        )
        count = len(data["pairs"])
        dup_out.setdefault(kind, {})[name] = {
            "pairs": count,
            "exact": exact,
            "exact_rate": None if not count else exact / count,
            "adjacent": adjacent,
            "adjacent_rate": None if not count else adjacent / count,
        }
    result["hidden_duplicates"] = dup_out
    return result


def _mapping_relations(
    mapping: dict[str, Any],
    assignments: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    units: dict[str, dict[str, Any]],
    candidates: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple[str, str, str, dict[str, Any]]], set[str]]:
    groups: dict[tuple[Any, ...], list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(
        list
    )
    duplicate_links: list[tuple[str, str, str, dict[str, Any]]] = []
    primary_ids: set[str] = set()
    for aid, meta in mapping.items():
        if bool(meta.get("calibration")) != bool(assignments[aid].get("calibration")):
            _fail("calibration provenance mismatch")
        if meta.get("kind") != assignments[aid].get("kind") or meta.get("unit_id") != assignments[
            aid
        ].get("unit_id"):
            _fail("assignment provenance mismatch")
        if meta.get("unit_id") not in units:
            _fail("unknown evaluation unit")
        dup = meta.get("duplicate_of")
        if dup is not None:
            if (
                dup not in mapping
                or mapping[dup].get("duplicate_of") is not None
                or mapping[dup].get("calibration")
                or mapping[dup].get("unit_id") != meta.get("unit_id")
            ):
                _fail("duplicate chain or target invalid")
            if assignments[aid].get("rater_id") != assignments.get(dup, {}).get("rater_id"):
                _fail("duplicate owner mismatch")
            continue
        if meta.get("calibration"):
            continue
        primary_ids.add(aid)
        kind, unit, surface = meta.get("kind"), meta.get("unit_id"), meta.get("surface")
        positions = meta.get("positions")
        if not isinstance(positions, list) or not positions:
            _fail("mapping positions invalid")
        canonical = sorted((p.get("candidate_id"), p.get("replicate")) for p in positions)
        if any(candidate not in candidates for candidate, _ in canonical):
            _fail("unknown mapped candidate")
        strategy = meta.get("strategy")
        if kind != "context" and strategy not in (None, "c2"):
            _fail("unexpected non-context strategy")
        if kind == "context" and strategy not in ("c0", "c1", "c2"):
            _fail("invalid context strategy")
        key = (kind, unit, surface, strategy, tuple(canonical))
        groups[key].append((aid, meta, responses.get(aid, {})))
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_card = {
        "absolute": 3,
        "mqm": 2,
        "pairwise": 2,
        "polish": 3,
        "context": 3,
        "postedit": 1,
    }
    for (kind, unit_id, surface, strategy, canonical), items in sorted(
        groups.items(), key=lambda x: str(x[0])
    ):
        if len(items) != expected_card[kind]:
            _fail(f"{kind} relation cardinality invalid")
        unit = units[unit_id]
        row: dict[str, Any] = {
            "kind": kind,
            "candidate": canonical[0][0],
            "unit_id": unit_id,
            "book_id": unit["book_id"],
            "surface": surface,
            "strategy": strategy,
            "words": unit["word_count"],
            "assignment_ids": [x[0] for x in items],
            "responses": [x[2] for x in items],
            "positions": canonical,
            "segment_ids": list(unit["segment_ids"]),
            "presentation_positions": [x[1].get("positions", []) for x in items],
        }
        if kind == "absolute":
            for response in row["responses"]:
                response["composite"] = (
                    (sum(_WEIGHTS[d] * float(response[d]) for d in _DIMENSIONS) - 1.0) / 4.0 * 100.0
                )
        elif kind == "mqm":
            _prepare_mqm(row)
        elif kind == "pairwise":
            _prepare_pair(row, canonical)
        elif kind == "postedit":
            pass
        result[kind].append(row)
    # Duplicates are compared after primary rows are built, by same-rater target.
    for aid, meta in mapping.items():
        target = meta.get("duplicate_of")
        if target is None:
            continue
        src = responses.get(aid)
        dst = responses.get(target)
        if src is None or dst is None:
            _fail("missing duplicate response")
        kind = meta.get("kind")
        if kind == "absolute":
            pairs = [(src.get(d), dst.get(d)) for d in _DIMENSIONS]
        elif kind == "pairwise":
            rank = {
                "b_much_better": -2,
                "b_slightly_better": -1,
                "tie": 0,
                "a_slightly_better": 1,
                "a_much_better": 2,
            }
            reverse_pref = {
                "a_much_better": "b_much_better",
                "a_slightly_better": "b_slightly_better",
                "tie": "tie",
                "b_slightly_better": "a_slightly_better",
                "b_much_better": "a_much_better",
            }
            canonical_first = sorted(p["candidate_id"] for p in mapping[target]["positions"])[0]
            src_pref, dst_pref = src.get("preference"), dst.get("preference")
            if mapping[aid]["positions"][0]["candidate_id"] != canonical_first:
                src_pref = reverse_pref[src_pref]
            if mapping[target]["positions"][0]["candidate_id"] != canonical_first:
                dst_pref = reverse_pref[dst_pref]
            pairs = [(rank.get(src_pref), rank.get(dst_pref))]
        elif kind == "context":
            rank = {"incorrect": 0, "uncertain": 1, "correct": 2}
            pairs = [(rank.get(src.get("judgment")), rank.get(dst.get("judgment")))]
        elif kind == "polish":
            pairs = [(src.get("outcome"), dst.get("outcome"))]
        elif kind == "mqm":

            def projection(value: dict[str, Any]) -> tuple[Any, ...]:
                return tuple(
                    sorted(
                        (
                            e.get("segment_id"),
                            e.get("severity"),
                            e.get("type"),
                            _norm_quote(e.get("source_quote")),
                            _norm_quote(e.get("target_quote")),
                        )
                        for e in value.get("errors", [])
                    )
                )

            pairs = [(projection(src), projection(dst))]
        elif kind == "postedit":
            pairs = [(src.get("edited_target"), dst.get("edited_target"))]
        else:
            pairs = []
        duplicate_links.append((kind, meta.get("unit_id", ""), aid, {"pairs": pairs}))
    return result, duplicate_links, primary_ids


def _prepare_pair(row: dict[str, Any], canonical: list[tuple[str, int]]) -> None:
    first, second = canonical[0][0], canonical[1][0]
    scores: list[float] = []
    strengths: list[str] = []
    strength_map = {
        "a_much_better": "first_much",
        "a_slightly_better": "first_slight",
        "tie": "tie",
        "b_slightly_better": "second_slight",
        "b_much_better": "second_much",
    }
    score_map = {
        "a_much_better": 1.0,
        "a_slightly_better": 1.0,
        "tie": 0.5,
        "b_slightly_better": 0.0,
        "b_much_better": 0.0,
    }
    reverse = {
        "a_much_better": "b_much_better",
        "a_slightly_better": "b_slightly_better",
        "tie": "tie",
        "b_slightly_better": "a_slightly_better",
        "b_much_better": "a_much_better",
    }
    for response in row["responses"]:
        pref = response["preference"]
        if response.get("_mapped_first", first) != first:
            pref = reverse[pref]
        score = score_map[pref]
        scores.append(score)
        strengths.append(strength_map[pref])
        response["_winner"] = "tie" if score == 0.5 else ("first" if score == 1.0 else "second")
    row["first"], row["second"] = first, second
    row["scores"], row["strengths"] = scores, strengths
    row["score"] = statistics.fmean(scores)
    row["discordant"] = len(set(strengths)) > 1
    row["relation_key"] = sha256_bytes(
        canonical_json(
            {
                "surface": row["surface"],
                "unit_id": row["unit_id"],
                "positions": row["positions"],
            }
        ).encode()
    )
    row["winner"] = "tie" if row["score"] == 0.5 else ("first" if row["score"] > 0.5 else "second")


def _prepare_mqm(row: dict[str, Any]) -> None:
    keys: list[set[tuple[Any, ...]]] = []
    raw_sev = dict.fromkeys(_SEVERITIES, 0)
    raw_types = dict.fromkeys(_TYPES, 0)
    normalized_responses = []
    for response in row["responses"]:
        current: set[tuple[Any, ...]] = set()
        for error in response.get("errors", []):
            key = (
                error.get("segment_id"),
                error.get("severity"),
                error.get("type"),
                _norm_quote(error.get("source_quote")),
                _norm_quote(error.get("target_quote")),
            )
            current.add(key)
            raw_sev[error["severity"]] += 1
            raw_types[error["type"]] += 1
        keys.append(current)
        normalized_responses.append(current)
    intersection = set.intersection(*keys) if keys else set()
    agreed_sev = {x: sum(1 for k in intersection if k[1] == x) for x in _SEVERITIES}
    agreed_types = {x: sum(1 for k in intersection if k[2] == x) for x in _TYPES}
    row["agreed_keys"] = intersection
    row["raw_severity"], row["raw_types"] = raw_sev, raw_types
    row["agreed_severity"], row["agreed_types"] = agreed_sev, agreed_types
    row["discordant"] = any(keys[i] != keys[0] for i in range(1, len(keys)))
    row["responses"] = [
        {
            **response,
            "highest": _highest(response.get("errors", [])),
            "types": "+".join(sorted({e["type"] for e in response.get("errors", [])}))
            if response.get("errors")
            else "none",
        }
        for response in row["responses"]
    ]


def _highest(errors: list[dict[str, Any]]) -> str:
    order = {"critical": 3, "major": 2, "minor": 1}
    return max((e["severity"] for e in errors), key=lambda x: order[x], default="none")


def analyze_human(
    corpus_dir: Path, pack_dir: Path, evaluation_dir: Path, spec: ReportSpec
) -> dict[str, Any]:
    """Validate and decode a completed Phase 7 evaluation without returning content."""
    try:
        corpus_info = validate_corpus(corpus_dir)
        pack_info = validate_pack(pack_dir)
    except Exception as exc:
        if isinstance(exc, HumanAnalysisError):
            raise
        raise HumanAnalysisError(str(exc)) from exc
    corpus = _json(Path(corpus_dir) / "corpus.json")
    pack = _json(Path(pack_dir) / "pack.json")
    if (
        corpus_info.get("corpus_sha256") != spec.corpus_sha256
        or pack_info.get("pack_sha256") != spec.pack_sha256
    ):
        _fail("input hash mismatch")
    if (
        pack.get("benchmark_id") != spec.benchmark_id
        or pack.get("run_hash") != spec.run_hash
        or pack.get("corpus_sha256") != spec.corpus_sha256
    ):
        _fail("pack lineage mismatch")
    evaluation = Path(evaluation_dir)
    eval_raw = _raw(evaluation / "evaluation.json")
    complete_raw = _raw(evaluation / "evaluation_complete.json")
    if sha256_bytes(eval_raw) != spec.evaluation_sha256:
        _fail("evaluation hash mismatch")
    state = _json(evaluation / "import_state.json")
    immutable = json.loads(eval_raw.decode("utf-8"))
    completion = json.loads(complete_raw.decode("utf-8"))
    if (
        state.get("status") != "complete"
        or immutable.get("pack_sha256") != pack.get("pack_sha256")
        or immutable.get("pack_semantic_sha256") != pack_info.get("pack_semantic_sha256")
    ):
        _fail("evaluation is not complete")
    if completion.get("evaluation_sha256") != sha256_bytes(eval_raw) or completion.get(
        "import_state_sha256"
    ) != sha256_bytes(_raw(evaluation / "import_state.json")):
        _fail("evaluation completion hash mismatch")
    expected_raters = sorted(pack.get("raters", []))
    if (
        sorted(immutable.get("expected_raters", [])) != expected_raters
        or sorted(state.get("raters", {})) != expected_raters
    ):
        _fail("rater completion set mismatch")
    for rater in expected_raters:
        path = evaluation / "source_responses" / f"{rater}.json"
        digest = sha256_bytes(_raw(path))
        if (
            state["raters"][rater].get("response_sha256") != digest
            or completion.get("derived_files", {}).get(f"source_responses/{rater}.json") != digest
        ):
            _fail("source response hash mismatch")
    for name in (
        "responses.jsonl",
        "mqm_errors.jsonl",
        "post_edits.jsonl",
        "adjudication_needed.json",
    ):
        if completion.get("derived_files", {}).get(name) != sha256_bytes(_raw(evaluation / name)):
            _fail(f"derived evaluation hash mismatch: {name}")
    response_rows = _jsonl(evaluation / "responses.jsonl")
    mapping = _json(Path(pack_dir) / "secret_mapping.json").get("assignments", {})
    mqm_rows = _jsonl(evaluation / "mqm_errors.jsonl")
    postedit_rows = _jsonl(evaluation / "post_edits.jsonl")
    adjudication_rows = _json(evaluation / "adjudication_needed.json")
    if not isinstance(adjudication_rows, list) or any(
        not isinstance(row, dict) for row in adjudication_rows
    ):
        _fail("adjudication artifact invalid")
    if [(row.get("assignment_id"), row.get("rater_id")) for row in response_rows] != sorted(
        (row.get("assignment_id"), row.get("rater_id")) for row in response_rows
    ):
        _fail("responses are not deterministically ordered")
    for row in mqm_rows + postedit_rows:
        if not isinstance(row.get("assignment_id"), str):
            _fail("derived response join invalid")
    assignment_by_rater: dict[str, dict[str, Any]] = {}
    assignment_owner: dict[str, str] = {}
    expected_ids_by_rater: dict[str, set[str]] = {}
    pack_root = Path(pack_dir)
    if any(row["assignment_id"] not in mapping for row in mqm_rows + postedit_rows):
        _fail("derived response assignment mismatch")
    for rater in expected_raters:
        envelope = _json(pack_root / "raters" / rater / "assignments.json")
        expected_ids_by_rater[rater] = set()
        for item in envelope.get("assignments", []):
            aid = item.get("assignment_id")
            if not isinstance(aid, str) or aid in assignment_by_rater:
                _fail("assignment ownership mismatch")
            assignment_by_rater[aid] = item
            assignment_owner[aid] = rater
            expected_ids_by_rater[rater].add(aid)
    if set(mapping) != set(assignment_by_rater):
        _fail("mapping assignment set mismatch")
    reconstructed: list[dict[str, Any]] = []
    for rater in expected_raters:
        own = {
            aid: item for aid, item in assignment_by_rater.items() if assignment_owner[aid] == rater
        }
        try:
            rows, _ = _responses(
                evaluation / "source_responses" / f"{rater}.json",
                pack_info["pack_semantic_sha256"],
                rater,
                own,
            )
        except Exception as exc:
            raise HumanAnalysisError(f"source response validation failed: {rater}") from exc
        reconstructed.extend(rows)
    reconstructed.sort(key=lambda row: (row["assignment_id"], row["rater_id"]))
    if reconstructed != response_rows:
        _fail("derived responses do not match source envelopes")
    expected_mqm_rows = [
        {**error, "assignment_id": row["assignment_id"], "rater_id": row["rater_id"]}
        for row in reconstructed
        if row["kind"] == "mqm"
        for error in row["errors"]
    ]
    expected_postedit_rows = [row for row in reconstructed if row["kind"] == "postedit"]
    if expected_mqm_rows != mqm_rows or expected_postedit_rows != postedit_rows:
        _fail("derived MQM/postedit files do not match source envelopes")
    try:
        expected_adjudication = _pair_adjudication(reconstructed, mapping)
    except Exception as exc:
        raise HumanAnalysisError("adjudication reconstruction failed") from exc
    if expected_adjudication != adjudication_rows:
        _fail("derived adjudication does not match source envelopes")
    source_ids: set[str] = set()
    for rater in expected_raters:
        envelope = _json(evaluation / "source_responses" / f"{rater}.json")
        raw_responses = envelope.get("responses")
        if envelope.get("rater_id") != rater or not isinstance(raw_responses, list):
            _fail("source response envelope invalid")
        ids: set[str] = set()
        for raw_response in raw_responses:
            if not isinstance(raw_response, dict):
                _fail("source response row invalid")
            aid = raw_response.get("assignment_id")
            if (
                aid in ids
                or aid not in expected_ids_by_rater[rater]
                or raw_response.get("rater_id") != rater
            ):
                _fail("source response assignment set invalid")
            ids.add(aid)
        if ids != expected_ids_by_rater[rater]:
            _fail("source response set incomplete")
        source_ids.update(ids)
    if source_ids != set(assignment_by_rater):
        _fail("source response set mismatch")
    response_by_aid: dict[str, dict[str, Any]] = {}
    for row in response_rows:
        aid, rater = row.get("assignment_id"), row.get("rater_id")
        if aid in response_by_aid:
            _fail("duplicate response assignment")
        if aid not in mapping or assignment_owner.get(aid) != rater:
            _fail("response ownership mismatch")
        if assignment_by_rater[aid].get("kind") != row.get("kind"):
            _fail("response kind mismatch")
        response_by_aid[aid] = row
    if len(response_by_aid) != len(mapping):
        _fail("response set incomplete")
    # Add private orientation marker before grouping; it is never emitted.
    for aid, row in response_by_aid.items():
        meta = mapping[aid]
        if meta.get("kind") == "pairwise":
            row["_mapped_first"] = meta["positions"][0].get("candidate_id")
    runner = []
    for line in _raw(Path(corpus_dir) / "runner_segments.jsonl").decode("utf-8").splitlines():
        runner.append(json.loads(line))
    units_list = build_units(runner, spec.corpus_sha256)
    units = {u["unit_id"]: u for u in units_list}
    formal_books = sorted(
        {
            book["book_id"]
            for book in corpus.get("books", [])
            if (
                isinstance(book, dict)
                and book.get("split") == "formal"
                and isinstance(book.get("book_id"), str)
            )
        }
    )
    eval_spec = EvaluationSpec.model_validate(pack["evaluation_spec"])
    if sorted(eval_spec.candidate_ids) != sorted(pack["evaluation_spec"]["candidate_ids"]):
        _fail("evaluation candidate mismatch")
    groups, duplicates, primary_ids = _mapping_relations(
        mapping, assignment_by_rater, response_by_aid, units, eval_spec.candidate_ids
    )
    # Pairwise source orientation was private and response dicts are mutable; relation prep sees it.
    insufficient: set[tuple[str, str]] = set()
    pending_mqm: set[tuple[str, str, str]] = set()
    pending_pairs: set[str] = set()
    pending_context: set[tuple[str, str]] = set()
    absolute = _absolute(
        groups.get("absolute", []),
        eval_spec.candidate_ids,
        eval_spec.enabled_surfaces,
        formal_books,
        insufficient,
        spec.bootstrap_seed,
        spec.bootstrap_replicates,
    )
    mqm = _mqm(
        groups.get("mqm", []),
        eval_spec.candidate_ids,
        eval_spec.enabled_surfaces,
        formal_books,
        insufficient,
        pending_mqm,
        spec.bootstrap_seed,
        spec.bootstrap_replicates,
    )
    pairwise = _pairwise(
        groups.get("pairwise", []),
        eval_spec.candidate_ids,
        eval_spec.enabled_surfaces,
        spec,
        insufficient,
        pending_pairs,
    )
    polish = _polish(
        groups.get("polish", []),
        eval_spec.candidate_ids,
        formal_books,
        insufficient,
        groups.get("mqm", []),
        spec.bootstrap_seed,
        spec.bootstrap_replicates,
    )
    context = _context(
        groups.get("context", []),
        eval_spec.candidate_ids,
        formal_books,
        insufficient,
        pending_context,
        spec.bootstrap_seed,
        spec.bootstrap_replicates,
    )
    # Postedit text is read only long enough to derive numeric diagnostics.
    for row in groups.get("postedit", []):
        aid = row["assignment_ids"][0]
        response = row["responses"][0]
        target = assignment_by_rater[aid].get("target")
        edited = response.get("edited_target")
        if (
            not isinstance(target, str)
            or not target.strip()
            or not isinstance(edited, str)
            or not edited.strip()
        ):
            _fail("postedit target payload invalid")
        row["distance"] = levenshtein_distance(target, edited)
        row["ratio"] = levenshtein_ratio(target, edited)
        row["active_ms"] = response.get("active_ms", 0)
    postedit = _postedit(
        groups.get("postedit", []),
        eval_spec.candidate_ids,
        eval_spec.enabled_surfaces,
        formal_books,
        insufficient,
        spec.bootstrap_seed,
        spec.bootstrap_replicates,
    )
    reliability = _reliability(groups, duplicates)
    pending = len(pending_pairs) + len(pending_mqm) + len(pending_context)
    by_kind = {
        kind: sum(
            1
            for aid in mapping
            if mapping[aid].get("kind") == kind
            and not mapping[aid].get("calibration")
            and mapping[aid].get("duplicate_of") is None
        )
        for kind in ("absolute", "mqm", "pairwise", "polish", "context", "postedit")
    }
    denominators = {
        "responses": {
            "primary": len(primary_ids),
            "calibration": sum(1 for m in mapping.values() if m.get("calibration")),
            "duplicates": sum(1 for m in mapping.values() if m.get("duplicate_of") is not None),
            "by_kind": by_kind,
        },
        "books": formal_books,
        "primary_source_words_by_kind": {
            kind: sum(r["words"] for r in groups.get(kind, [])) for kind in by_kind
        },
    }
    insufficient_data = [
        {"scope": scope, "reason": reason} for scope, reason in sorted(insufficient)
    ]
    alpha_values = [
        reliability[key]
        for key in (
            (*_DIMENSIONS, "pairwise_winner", "context_correctness", "mqm_severity", "mqm_type")
        )
    ]
    needs_recalibration = any(
        value is None or value < float(spec.gates.krippendorff_alpha_min) for value in alpha_values
    )
    return {
        "schema_version": 1,
        "input_hashes": {
            "corpus_sha256": spec.corpus_sha256,
            "pack_sha256": pack_info["pack_sha256"],
            "pack_semantic_sha256": pack_info["pack_semantic_sha256"],
            "evaluation_sha256": sha256_bytes(eval_raw),
            "evaluation_complete_sha256": sha256_bytes(complete_raw),
        },
        "denominators": denominators,
        "absolute": absolute,
        "mqm": mqm,
        "pairwise": pairwise,
        "polish": polish,
        "context": context,
        "reliability": reliability,
        "postedit": postedit,
        "pending_adjudication_count": pending,
        "insufficient_data": insufficient_data,
        "needs_recalibration": needs_recalibration,
    }
