from __future__ import annotations

import math
import unittest
from typing import ClassVar

from trans_novel.benchmark.report_stats import (
    StatisticsError,
    bootstrap_bradley_terry,
    fit_bradley_terry,
    hierarchical_bootstrap,
    krippendorff_alpha,
    levenshtein_distance,
    levenshtein_ratio,
    nearest_rank,
    pareto_frontier,
    percentile,
    wilson_upper95,
)


class BasicStatisticsTests(unittest.TestCase):
    def test_percentile_and_nearest_rank(self) -> None:
        self.assertEqual(percentile([4, 1, 2, 3], 0.5), 2.5)
        self.assertEqual(nearest_rank([4, 1, 2, 3], 0.5), 2.0)
        with self.assertRaises(StatisticsError):
            percentile([], 0.5)
        with self.assertRaises(StatisticsError):
            nearest_rank([1], 0)
        with self.assertRaises(StatisticsError):
            percentile([True], 0.5)

    def test_wilson_and_levenshtein(self) -> None:
        self.assertIsNone(wilson_upper95(0, 0))
        self.assertAlmostEqual(wilson_upper95(0, 10), 0.2775327998628892)
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("", ""), 0)
        self.assertEqual(levenshtein_ratio("", ""), 1.0)
        with self.assertRaises(StatisticsError):
            levenshtein_distance("a", 1)  # type: ignore[arg-type]


class BootstrapTests(unittest.TestCase):
    ROWS: ClassVar[list[dict[str, int]]] = [
        {"book_id": "b2", "unit_id": "u2", "value": 4},
        {"book_id": "b1", "unit_id": "u2", "value": 2},
        {"book_id": "b1", "unit_id": "u1", "value": 1},
        {"book_id": "b2", "unit_id": "u1", "value": 3},
    ]

    @staticmethod
    def mean_book_occurrences(occurrences: list[list[dict[str, int]]]) -> float:
        return sum(
            sum(row["value"] for row in occurrence) / len(occurrence) for occurrence in occurrences
        ) / len(occurrences)

    def test_hierarchical_is_deterministic_and_equal_book(self) -> None:
        one = hierarchical_bootstrap(
            self.ROWS, seed=7, replicates=8, statistic=self.mean_book_occurrences
        )
        two = hierarchical_bootstrap(
            self.ROWS, seed=7, replicates=8, statistic=self.mean_book_occurrences
        )
        self.assertEqual(one, two)
        self.assertEqual(one.point, 2.5)
        self.assertEqual(len(one.samples), 8)

    def test_hierarchical_validation_and_callback_errors(self) -> None:
        with self.assertRaises(StatisticsError):
            hierarchical_bootstrap([], seed=1, replicates=1, statistic=lambda _: 1)
        with self.assertRaises(RuntimeError):
            hierarchical_bootstrap(
                self.ROWS,
                seed=1,
                replicates=1,
                statistic=lambda _: (_ for _ in ()).throw(RuntimeError("x")),
            )


class BradleyTerryTests(unittest.TestCase):
    def test_ordering_ties_and_sorted_results(self) -> None:
        result = fit_bradley_terry(
            ["b", "a", "c"],
            [("a", "b", 1), ("a", "c", 1), ("b", "c", 1), ("c", "a", 0.5)],
        )
        self.assertEqual(list(result.abilities), ["a", "b", "c"])
        self.assertGreater(result.abilities["a"], result.abilities["b"])
        self.assertEqual(set(result.field_win_probability), {"a", "b", "c"})

    def test_disconnected_and_nonconverged(self) -> None:
        with self.assertRaisesRegex(StatisticsError, "disconnected"):
            fit_bradley_terry(["a", "b", "c"], [("a", "b", 1)])
        with self.assertRaisesRegex(StatisticsError, "nonconverged"):
            fit_bradley_terry(["a", "b"], [("a", "b", 1)], max_iterations=2)
        with self.assertRaises(StatisticsError):
            fit_bradley_terry(["a", "b"], [("a", "b", 0.5)], tolerance=1)
        with self.assertRaises(StatisticsError):
            fit_bradley_terry(["a", "b"], [("a", "b", 0.5)], tolerance=True)

    def test_bootstrap_reports_discard_cap(self) -> None:
        candidates = [f"c{i}" for i in range(21)]
        rows = [
            {
                "book_id": f"b{i:02d}",
                "unit_id": "unit",
                "candidate_a": f"c{i}",
                "candidate_b": f"c{i + 1}",
                "score_a": 0.5,
            }
            for i in range(20)
        ]
        with self.assertRaisesRegex(StatisticsError, "attempted=10"):
            bootstrap_bradley_terry(rows, candidates, seed=0, replicates=1)


class AlphaAndParetoTests(unittest.TestCase):
    def test_alpha_known_cases_and_validation(self) -> None:
        self.assertIsNone(
            krippendorff_alpha(
                {"u1": ["x", "x"], "u2": ["x", "x"]}, level="nominal", categories=["x", "y"]
            )
        )
        self.assertEqual(
            krippendorff_alpha(
                {"u1": ["x", "y"], "u2": ["y", "x"]}, level="nominal", categories=["x", "y"]
            ),
            -0.5,
        )
        self.assertIsNone(
            krippendorff_alpha(
                {"u1": ["x"], "u2": [None, "y"]}, level="nominal", categories=["x", "y"]
            )
        )
        self.assertEqual(
            krippendorff_alpha(
                {"u1": ["x", "x"], "u2": ["x", "y"]}, level="nominal", categories=["x", "y"]
            ),
            0.0,
        )
        with self.assertRaises(StatisticsError):
            krippendorff_alpha({"u1": ["unknown", "x"]}, level="nominal", categories=["x", "y"])
        with self.assertRaises(StatisticsError):
            krippendorff_alpha({"u1": ["x", "y"]}, level="nominal", categories={"x", "y"})
        with self.assertRaises(StatisticsError):
            krippendorff_alpha({"u1": ["x", "y"]}, level="Nominal", categories=["x", "y"])
        with self.assertRaises(StatisticsError):
            krippendorff_alpha({"u1": ["x", "y"]}, level=1, categories=["x", "y"])  # type: ignore[arg-type]

    def test_pareto_dominance_ties_and_unknown(self) -> None:
        result = pareto_frontier(
            [
                {"candidate_id": "dominated", "cost": 3, "quality": 1},
                {"candidate_id": "frontier", "cost": 2, "quality": 2},
                {"candidate_id": "tie", "cost": 2, "quality": 2},
                {"candidate_id": "unknown", "cost": math.nan, "quality": 3},
            ],
            {"cost": "min", "quality": "max"},
        )
        self.assertEqual(result.frontier, ("frontier", "tie"))
        self.assertEqual(result.excluded, {"unknown": "unknown:cost"})


if __name__ == "__main__":
    unittest.main()
