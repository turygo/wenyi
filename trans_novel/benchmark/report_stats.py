"""Deterministic, dependency-free statistics used by benchmark reports."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BootstrapResult",
    "BradleyTerryBootstrapResult",
    "BradleyTerryResult",
    "ParetoResult",
    "StatisticsError",
    "bootstrap_bradley_terry",
    "fit_bradley_terry",
    "hierarchical_bootstrap",
    "krippendorff_alpha",
    "levenshtein_distance",
    "levenshtein_ratio",
    "nearest_rank",
    "pareto_frontier",
    "percentile",
    "sequence_similarity",
    "wilson_upper95",
]


class StatisticsError(ValueError):
    """Raised when a statistic cannot be calculated from its input."""


def _number(value: object, *, name: str = "value") -> float:
    if type(value) not in (int, float):
        raise StatisticsError(f"{name} must be an exact int or float")
    try:
        result = float(value)
    except OverflowError as exc:
        raise StatisticsError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise StatisticsError(f"{name} must be finite")
    return result


def _seed(seed: object) -> int:
    if type(seed) is not int:
        raise StatisticsError("seed must be an exact int")
    return seed


def _replicates(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise StatisticsError("replicates must be a positive exact int")
    return value


def percentile(values: Iterable[object], p: object) -> float:
    """Return the type-7 percentile (linear interpolation) of *values*."""
    probability = _number(p, name="p")
    if not 0.0 <= probability <= 1.0:
        raise StatisticsError("p must be between 0 and 1")
    numbers = sorted(_number(value) for value in values)
    if not numbers:
        raise StatisticsError("values must be nonempty")
    if len(numbers) == 1:
        return numbers[0]
    h = (len(numbers) - 1) * probability
    lower = math.floor(h)
    upper = math.ceil(h)
    if lower == upper:
        return numbers[lower]
    return numbers[lower] + (h - lower) * (numbers[upper] - numbers[lower])


def nearest_rank(values: Iterable[object], p: object) -> float:
    """Return the nearest-rank percentile of *values*."""
    probability = _number(p, name="p")
    if not 0.0 < probability <= 1.0:
        raise StatisticsError("p must be greater than 0 and at most 1")
    numbers = sorted(_number(value) for value in values)
    if not numbers:
        raise StatisticsError("values must be nonempty")
    index = max(0, math.ceil(probability * len(numbers)) - 1)
    return numbers[index]


def wilson_upper95(events: object, total: object) -> float | None:
    """Return the exact 95% Wilson score upper bound, or ``None`` for n=0."""
    if type(events) is not int or type(total) is not int:
        raise StatisticsError("events and total must be exact ints")
    if events < 0 or total < 0 or events > total:
        raise StatisticsError("events must satisfy 0 <= events <= total")
    if total == 0:
        return None
    z = 1.959963984540054
    n = float(total)
    estimate = events / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = estimate + z2 / (2.0 * n)
    spread = z * math.sqrt(estimate * (1.0 - estimate) / n + z2 / (4.0 * n * n))
    return min(1.0, (centre + spread) / denominator)


def levenshtein_distance(a: str, b: str) -> int:
    """Return character Levenshtein distance using O(min(len(a), len(b))) space."""
    if not isinstance(a, str) or not isinstance(b, str):
        raise StatisticsError("a and b must be strings")
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    if not isinstance(a, str) or not isinstance(b, str):
        raise StatisticsError("a and b must be strings")
    maximum = max(len(a), len(b))
    if maximum == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / maximum


# The report contract calls this diagnostic by both names in different phases.
def sequence_similarity(a: str, b: str) -> float:
    return levenshtein_ratio(a, b)


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower95: float
    upper95: float
    samples: tuple[float, ...]


@dataclass(frozen=True)
class BradleyTerryResult:
    abilities: dict[str, float]
    field_win_probability: dict[str, float]
    iterations: int


@dataclass(frozen=True)
class BradleyTerryBootstrapResult:
    point: BradleyTerryResult
    ability_ci: dict[str, tuple[float, float]]
    field_win_ci: dict[str, tuple[float, float]]
    requested: int
    attempted: int
    discarded: dict[str, int]


@dataclass(frozen=True)
class ParetoResult:
    frontier: tuple[str, ...]
    excluded: dict[str, str]


def _checked_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[str], dict[str, dict[str, list[Mapping[str, Any]]]]]:
    if isinstance(rows, Mapping):
        raise StatisticsError("rows must be an iterable of mappings")
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise StatisticsError("rows must be an iterable of mappings") from exc
    count = 0
    for row in iterator:
        if not isinstance(row, Mapping):
            raise StatisticsError("every row must be a mapping")
        book_id = row.get("book_id")
        unit_id = row.get("unit_id")
        if not isinstance(book_id, str) or not book_id.strip():
            raise StatisticsError("book_id must be a nonblank string")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise StatisticsError("unit_id must be a nonblank string")
        grouped.setdefault(book_id, {}).setdefault(unit_id, []).append(row)
        count += 1
    if count == 0:
        raise StatisticsError("rows must be nonempty")
    return sorted(grouped), grouped


def _occurrences(
    books: Sequence[str],
    grouped: Mapping[str, Mapping[str, list[Mapping[str, Any]]]],
    rng: random.Random | None,
) -> list[list[Mapping[str, Any]]]:
    result: list[list[Mapping[str, Any]]] = []
    selected_books = books if rng is None else [rng.choice(books) for _ in books]
    for book_id in selected_books:
        units = sorted(grouped[book_id])
        selected_units = units if rng is None else [rng.choice(units) for _ in units]
        occurrence: list[Mapping[str, Any]] = []
        for unit_id in selected_units:
            occurrence.extend(grouped[book_id][unit_id])
        result.append(occurrence)
    return result


def _stat_value(value: object) -> float:
    return _number(value, name="statistic result")


def hierarchical_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
    statistic: Callable[[list[list[Mapping[str, Any]]]], object],
) -> BootstrapResult:
    """Bootstrap book occurrences, then units within each selected book occurrence."""
    requested = _replicates(replicates)
    rng = random.Random(_seed(seed))
    books, grouped = _checked_rows(rows)
    original = _occurrences(books, grouped, None)
    point = _stat_value(statistic(original))
    samples = tuple(
        _stat_value(statistic(_occurrences(books, grouped, rng))) for _ in range(requested)
    )
    return BootstrapResult(point, percentile(samples, 0.025), percentile(samples, 0.975), samples)


def _candidate_ids(candidates: Iterable[object]) -> list[str]:
    result: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            raise StatisticsError("candidate IDs must be nonblank strings")
        if candidate in result:
            raise StatisticsError("candidate IDs must be unique")
        result.append(candidate)
    if not result:
        raise StatisticsError("candidates must be nonempty")
    return sorted(result)


def _outcome_tuple(outcome: object) -> tuple[object, object, object]:
    if isinstance(outcome, str | bytes):
        raise StatisticsError("invalid_outcome: outcome must be a triple")
    try:
        values = tuple(outcome)  # type: ignore[arg-type]
    except TypeError as exc:
        raise StatisticsError("invalid_outcome: outcome must be a triple") from exc
    if len(values) != 3:
        raise StatisticsError("invalid_outcome: outcome must be a triple")
    return values  # type: ignore[return-value]


def _validated_outcomes(
    candidates: Sequence[str], outcomes: Iterable[object]
) -> tuple[list[tuple[str, str, float]], dict[tuple[str, str], int], dict[str, float]]:
    candidate_set = set(candidates)
    pair_counts: dict[tuple[str, str], int] = {}
    wins = dict.fromkeys(candidates, 0.0)
    seen: set[str] = set()
    try:
        iterator = iter(outcomes)
    except TypeError as exc:
        raise StatisticsError("invalid_outcome: outcomes must be iterable") from exc
    normalized: list[tuple[str, str, float]] = []
    for raw in iterator:
        left, right, score = _outcome_tuple(raw)
        if not isinstance(left, str) or not isinstance(right, str):
            raise StatisticsError("invalid_outcome: candidate IDs must be strings")
        if left not in candidate_set or right not in candidate_set or left == right:
            raise StatisticsError("invalid_outcome: unknown or self-comparison")
        if type(score) not in (int, float) or float(score) not in (0.0, 0.5, 1.0):
            raise StatisticsError("invalid_outcome: score must be exactly 0, 0.5, or 1")
        score_float = float(score)
        pair = tuple(sorted((left, right)))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        wins[left] += score_float
        wins[right] += 1.0 - score_float
        seen.update((left, right))
        normalized.append((left, right, score_float))
    if pair_counts and not _connected(candidates, pair_counts):
        raise StatisticsError("disconnected candidate graph")
    if seen != candidate_set:
        raise StatisticsError("invalid_outcome: every candidate must appear")
    return normalized, pair_counts, wins


def _connected(candidates: Sequence[str], pair_counts: Mapping[tuple[str, str], int]) -> bool:
    if not candidates:
        return False
    visited = {candidates[0]}
    while True:
        before = len(visited)
        for left, right in pair_counts:
            if left in visited:
                visited.add(right)
            if right in visited:
                visited.add(left)
        if len(visited) == before:
            break
    return len(visited) == len(candidates)


def fit_bradley_terry(
    candidates: Iterable[object],
    outcomes: Iterable[object],
    *,
    tolerance: object = 1e-10,
    max_iterations: int = 10000,
) -> BradleyTerryResult:
    candidate_list = _candidate_ids(candidates)
    if type(tolerance) is not float or not math.isfinite(tolerance) or tolerance <= 0:
        raise StatisticsError("tolerance must be a finite positive exact float")
    tol = tolerance
    if type(max_iterations) is not int or max_iterations <= 0:
        raise StatisticsError("max_iterations must be a positive exact int")
    _, pair_counts, wins = _validated_outcomes(candidate_list, outcomes)
    if not _connected(candidate_list, pair_counts):
        raise StatisticsError("disconnected candidate graph")

    strengths = dict.fromkeys(candidate_list, 1.0)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        updated: dict[str, float] = {}
        for candidate in candidate_list:
            denominator = 0.0
            for pair, count in pair_counts.items():
                if candidate in pair:
                    other = pair[1] if pair[0] == candidate else pair[0]
                    denominator += count / (strengths[candidate] + strengths[other])
            if denominator <= 0 or not math.isfinite(denominator) or wins[candidate] <= 0:
                raise StatisticsError("nonconverged Bradley-Terry fit")
            value = wins[candidate] / denominator
            if value <= 0 or not math.isfinite(value):
                raise StatisticsError("nonconverged Bradley-Terry fit")
            updated[candidate] = value
        logs = {candidate: math.log(updated[candidate]) for candidate in candidate_list}
        mean_log = sum(logs.values()) / len(candidate_list)
        normalized = {
            candidate: math.exp(logs[candidate] - mean_log) for candidate in candidate_list
        }
        if any(not math.isfinite(value) or value <= 0 for value in normalized.values()):
            raise StatisticsError("nonconverged Bradley-Terry fit")
        change = max(
            abs(math.log(normalized[candidate] / strengths[candidate]))
            for candidate in candidate_list
        )
        strengths = normalized
        iterations = iteration
        if change <= tol:
            abilities = {candidate: math.log(strengths[candidate]) for candidate in candidate_list}
            field = {
                candidate: sum(
                    strengths[candidate] / (strengths[candidate] + strengths[other])
                    for other in candidate_list
                    if other != candidate
                )
                / (len(candidate_list) - 1)
                for candidate in candidate_list
            }
            return BradleyTerryResult(abilities, field, iterations)
    raise StatisticsError("nonconverged Bradley-Terry fit")


def _bt_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[str], dict[str, dict[str, list[Mapping[str, Any]]]], list[tuple[str, str, float]]]:
    books, grouped = _checked_rows(rows)
    flattened: list[tuple[str, str, float]] = []
    for book in books:
        for unit in sorted(grouped[book]):
            for row in grouped[book][unit]:
                required = ("candidate_a", "candidate_b", "score_a")
                if any(key not in row for key in required):
                    raise StatisticsError("invalid_outcome: BT row is missing a required field")
                left, right, score = row["candidate_a"], row["candidate_b"], row["score_a"]
                if not isinstance(left, str) or not isinstance(right, str):
                    raise StatisticsError("invalid_outcome: candidate IDs must be strings")
                if type(score) not in (int, float) or float(score) not in (0.0, 0.5, 1.0):
                    raise StatisticsError("invalid_outcome: score must be exactly 0, 0.5, or 1")
                flattened.append((left, right, float(score)))
    return books, grouped, flattened


def bootstrap_bradley_terry(
    rows: Iterable[Mapping[str, Any]],
    candidates: Iterable[object],
    *,
    seed: int,
    replicates: int,
    tolerance: object = 1e-10,
    max_iterations: int = 10000,
) -> BradleyTerryBootstrapResult:
    requested = _replicates(replicates)
    candidate_list = _candidate_ids(candidates)
    books, grouped, point_outcomes = _bt_rows(rows)
    # The point fit is intentionally performed before consuming the bootstrap RNG.
    point = fit_bradley_terry(
        candidate_list, point_outcomes, tolerance=tolerance, max_iterations=max_iterations
    )
    rng = random.Random(_seed(seed))
    valid: list[BradleyTerryResult] = []
    discarded = {"disconnected": 0, "nonconverged": 0}
    attempted = 0
    while len(valid) < requested and attempted < requested * 10:
        attempted += 1
        occurrence_rows = _occurrences(books, grouped, rng)
        sampled_outcomes: list[tuple[object, object, object]] = []
        for occurrence in occurrence_rows:
            sampled_outcomes.extend(
                (row["candidate_a"], row["candidate_b"], row["score_a"]) for row in occurrence
            )
        try:
            valid.append(
                fit_bradley_terry(
                    candidate_list,
                    sampled_outcomes,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            )
        except StatisticsError as exc:
            message = str(exc)
            if "disconnected" in message or "every candidate" in message:
                discarded["disconnected"] += 1
            elif "nonconverged" in message:
                discarded["nonconverged"] += 1
            else:
                raise
    if len(valid) < requested:
        raise StatisticsError(
            f"insufficient valid BT bootstrap replicates: requested={requested}, "
            f"attempted={attempted}, disconnected={discarded['disconnected']}, "
            f"nonconverged={discarded['nonconverged']}"
        )
    ability_ci = {
        candidate: (
            percentile((fit.abilities[candidate] for fit in valid), 0.025),
            percentile((fit.abilities[candidate] for fit in valid), 0.975),
        )
        for candidate in candidate_list
    }
    field_ci = {
        candidate: (
            percentile((fit.field_win_probability[candidate] for fit in valid), 0.025),
            percentile((fit.field_win_probability[candidate] for fit in valid), 0.975),
        )
        for candidate in candidate_list
    }
    return BradleyTerryBootstrapResult(point, ability_ci, field_ci, requested, attempted, discarded)


def _category_index(categories: Sequence[object]) -> tuple[list[Hashable], dict[Hashable, int]]:
    values: list[Hashable] = []
    index: dict[Hashable, int] = {}
    if isinstance(categories, str | bytes | set | frozenset) or not isinstance(
        categories, Sequence
    ):
        raise StatisticsError("categories must be an ordered sequence")
    iterator = iter(categories)
    for category in iterator:
        if category is None:
            raise StatisticsError("categories cannot contain None")
        try:
            hash(category)
        except TypeError as exc:
            raise StatisticsError("categories must be hashable") from exc
        if category in index:
            raise StatisticsError("categories must be unique")
        index[category] = len(values)
        values.append(category)
    if not values:
        raise StatisticsError("categories must be nonempty")
    return values, index


def krippendorff_alpha(
    units: Mapping[object, object],
    *,
    level: str,
    categories: Sequence[object],
) -> float | None:
    if type(level) is not str or level not in ("nominal", "ordinal"):
        raise StatisticsError("level must be exactly nominal or ordinal")
    if not isinstance(units, Mapping):
        raise StatisticsError("units must be a mapping")
    category_list, category_index = _category_index(categories)
    unit_iter = units.values()
    coincidence = [[0.0 for _ in category_list] for _ in category_list]
    pooled = [0.0 for _ in category_list]
    comparable = False
    observations = 0
    for raw_unit in unit_iter:
        if isinstance(raw_unit, Mapping):
            ratings = list(raw_unit.values())
        else:
            try:
                ratings = list(raw_unit)
            except TypeError as exc:
                raise StatisticsError("each unit must be iterable") from exc
        present: list[int] = []
        for rating in ratings:
            if rating is None:
                continue
            try:
                position = category_index[rating]
            except (KeyError, TypeError) as exc:
                raise StatisticsError("unknown or unhashable rating") from exc
            present.append(position)
        if len(present) < 2:
            continue
        comparable = True
        observations += len(present)
        for position in present:
            pooled[position] += 1.0
        denominator = len(present) - 1
        for left_index, left in enumerate(present):
            for right_index, right in enumerate(present):
                if left_index != right_index:
                    coincidence[left][right] += 1.0 / denominator
    if observations < 2 or not comparable:
        return None
    total_coincidence = sum(map(sum, coincidence))
    if total_coincidence <= 0:
        return None

    def distance(left: int, right: int) -> float:
        if left == right:
            return 0.0
        if level == "nominal":
            return 1.0
        low, high = sorted((left, right))
        numerator = sum(pooled[low : high + 1]) - (pooled[low] + pooled[high]) / 2.0
        return (numerator / observations) ** 2

    observed_disagreement = (
        sum(
            coincidence[left][right] * distance(left, right)
            for left in range(len(category_list))
            for right in range(len(category_list))
        )
        / total_coincidence
    )
    expected = [[0.0 for _ in category_list] for _ in category_list]
    total_minus_one = observations - 1
    for left in range(len(category_list)):
        for right in range(len(category_list)):
            if left == right:
                expected[left][right] = pooled[left] * (pooled[left] - 1.0) / total_minus_one
            else:
                expected[left][right] = pooled[left] * pooled[right] / total_minus_one
    total_expected = sum(map(sum, expected))
    if total_expected <= 0:
        return None
    expected_disagreement = (
        sum(
            expected[left][right] * distance(left, right)
            for left in range(len(category_list))
            for right in range(len(category_list))
        )
        / total_expected
    )
    if expected_disagreement == 0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def pareto_frontier(
    rows: Iterable[Mapping[str, Any]],
    dimensions: Mapping[str, str],
) -> ParetoResult:
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise StatisticsError("dimensions must be a nonempty mapping")
    dimension_items: list[tuple[str, str]] = []
    for field, direction in dimensions.items():
        if not isinstance(field, str) or not field.strip():
            raise StatisticsError("dimension names must be nonblank strings")
        if direction not in ("min", "max"):
            raise StatisticsError("dimension direction must be min or max")
        dimension_items.append((field, direction))
    if isinstance(rows, Mapping):
        raise StatisticsError("rows must be an iterable of mappings")
    valid: dict[str, dict[str, float]] = {}
    excluded: dict[str, str] = {}
    try:
        row_iter = iter(rows)
    except TypeError as exc:
        raise StatisticsError("rows must be iterable") from exc
    for row in row_iter:
        if not isinstance(row, Mapping):
            raise StatisticsError("every row must be a mapping")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise StatisticsError("candidate_id must be a nonblank string")
        if candidate_id in valid or candidate_id in excluded:
            raise StatisticsError("candidate_id values must be unique")
        unknown_field: str | None = None
        values: dict[str, float] = {}
        for field, _ in dimension_items:
            value = row.get(field)
            if value is None or type(value) not in (int, float) or not math.isfinite(float(value)):
                unknown_field = field
                break
            values[field] = float(value)
        if unknown_field is not None:
            excluded[candidate_id] = f"unknown:{unknown_field}"
        else:
            valid[candidate_id] = values
    ids = sorted(valid)
    frontier: list[str] = []
    for candidate in ids:
        dominated = False
        for other in ids:
            if other == candidate:
                continue
            no_worse = True
            strictly_better = False
            for field, direction in dimension_items:
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
    return ParetoResult(tuple(frontier), dict(sorted(excluded.items())))
