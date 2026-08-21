from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_phase7_evaluation import _make_pack, _write_responses
from trans_novel.benchmark.corpus import sha256_bytes
from trans_novel.benchmark.evaluation import import_responses
from trans_novel.benchmark.report_human import HumanAnalysisError, _paired, analyze_human
from trans_novel.benchmark.report_schema import ReportSpec


class Phase8HumanReportTests(unittest.TestCase):
    def _completed(self, mutate=None):
        root = Path(tempfile.mkdtemp())
        pack, evaluation_spec = _make_pack(root)
        corpus = root / "corpus"
        (corpus / "corpus.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "books": [
                        {"book_id": "book0", "split": "formal"},
                        {"book_id": "book1", "split": "formal"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        responses = root / "responses"
        _write_responses(pack, responses)
        if mutate:
            manifest = json.loads((pack / "pack.json").read_text())
            for rater in manifest["raters"]:
                path = responses / f"responses-{rater}.json"
                envelope = json.loads(path.read_text())
                assignments = {
                    x["assignment_id"]: x
                    for x in json.loads((pack / "raters" / rater / "assignments.json").read_text())[
                        "assignments"
                    ]
                }
                for row in envelope["responses"]:
                    mutate(row, assignments[row["assignment_id"]])
                path.write_text(json.dumps(envelope, ensure_ascii=False), encoding="utf-8")
        evaluation = root / "evaluation"
        import_responses(pack, responses, evaluation)
        manifest = json.loads((pack / "pack.json").read_text())
        spec = ReportSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": evaluation_spec["benchmark_id"],
                "corpus_sha256": evaluation_spec["corpus_sha256"],
                "run_hash": evaluation_spec["run_hash"],
                "preparation_sha256": "d" * 64,
                "pack_sha256": manifest["pack_sha256"],
                "evaluation_sha256": sha256_bytes((evaluation / "evaluation.json").read_bytes()),
                "price_snapshot_sha256": "f" * 64,
                "bootstrap_seed": 23,
                "bootstrap_replicates": 1000,
            }
        )
        return corpus, pack, evaluation, spec

    def _analyze(self, corpus, pack, evaluation, spec):
        with patch(
            "trans_novel.benchmark.report_human.validate_corpus",
            return_value={"corpus_sha256": spec.corpus_sha256},
        ):
            return analyze_human(corpus, pack, evaluation, spec)

    def test_completed_metrics_and_no_text(self):
        corpus, pack, evaluation, spec = self._completed()
        facts = self._analyze(corpus, pack, evaluation, spec)
        self.assertEqual(
            facts["absolute"]["c1"]["attribution_final"]["macro"]["dimensions"]["fidelity"][
                "score_100"
            ],
            50.0,
        )
        self.assertEqual(
            facts["mqm"]["c1"]["attribution_final"]["macro"]["agreed"]["severity"]["major"][
                "count"
            ],
            0,
        )
        self.assertEqual(facts["polish"]["c2"]["macro"]["improved_rate"], 1.0)
        self.assertEqual(facts["context"]["c1"]["by_strategy"]["c0"]["macro"]["accuracy"], 1.0)
        self.assertGreaterEqual(
            facts["postedit"]["c1"]["attribution_final"]["macro"]["edit_distance_mean"], 0
        )
        self.assertIn("pairwise_winner", facts["reliability"])
        self.assertIn("accuracy_lower95", facts["context"]["c1"]["by_strategy"]["c0"]["macro"])
        self.assertIn("by_book", facts["context"]["c1"]["lift"]["c1"])
        encoded = json.dumps(facts, ensure_ascii=False)
        for text in ("sourcezero", "translation-c1", "edited answer", "rationale"):
            self.assertNotIn(text, encoded)

    def test_deterministic_insufficient_and_pending(self):
        corpus, pack, evaluation, spec = self._completed()
        first, second = (
            self._analyze(corpus, pack, evaluation, spec),
            self._analyze(corpus, pack, evaluation, spec),
        )
        self.assertEqual(first, second)
        self.assertIsInstance(first["pending_adjudication_count"], int)
        self.assertEqual(
            first["insufficient_data"],
            sorted(first["insufficient_data"], key=lambda x: (x["scope"], x["reason"])),
        )

    def test_discordant_pair_and_mqm_pending(self):
        changed = {"pair": False, "mqm": False}

        def mutate(row, assignment):
            if row["kind"] == "pairwise" and not changed["pair"]:
                row["preference"] = "b_much_better"
                changed["pair"] = True
            if row["kind"] == "mqm" and not changed["mqm"]:
                row["errors"] = [
                    {
                        "segment_id": assignment["segment_ids"][0],
                        "severity": "major",
                        "type": "mistranslation",
                        "source_quote": None,
                        "target_quote": None,
                        "note": "disagreement",
                    }
                ]
                changed["mqm"] = True

        corpus, pack, evaluation, spec = self._completed(mutate)
        facts = self._analyze(corpus, pack, evaluation, spec)
        self.assertGreaterEqual(facts["pending_adjudication_count"], 2)

    def test_tamper_provenance_incomplete_and_derived_rejected(self):
        corpus, pack, evaluation, spec = self._completed()
        mapping_path = pack / "secret_mapping.json"
        mapping = json.loads(mapping_path.read_text())
        mapping["assignments"][sorted(mapping["assignments"])[0]]["unit_id"] = "unknown"
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
        with self.assertRaises(HumanAnalysisError):
            self._analyze(corpus, pack, evaluation, spec)
        corpus, pack, evaluation, spec = self._completed()
        path = evaluation / "responses.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            if row["kind"] == "absolute":
                row["fidelity"] = 5
                break
        path.write_text(
            "".join(
                json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                for x in rows
            ),
            encoding="utf-8",
        )
        complete = json.loads((evaluation / "evaluation_complete.json").read_text())
        complete["derived_files"]["responses.jsonl"] = sha256_bytes(path.read_bytes())
        (evaluation / "evaluation_complete.json").write_text(
            json.dumps(complete, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        with self.assertRaises(HumanAnalysisError):
            self._analyze(corpus, pack, evaluation, spec)

    def test_nonuniform_bootstrap_and_nonformal_book(self):
        from trans_novel.benchmark.report_human import _hier

        low, high = _hier(
            [
                {"book_id": "a", "unit_id": "a1", "value": 1.0},
                {"book_id": "a", "unit_id": "a2", "value": 9.0},
                {"book_id": "b", "unit_id": "b1", "value": 5.0},
            ],
            "value",
            19,
            1000,
        )
        self.assertLessEqual(low, high)
        corpus, pack, evaluation, spec = self._completed()
        data = json.loads((corpus / "corpus.json").read_text())
        data["books"].append({"book_id": "corpus-only", "split": "screen"})
        (corpus / "corpus.json").write_text(json.dumps(data), encoding="utf-8")
        facts = self._analyze(corpus, pack, evaluation, spec)
        by_book = facts["absolute"]["c1"]["attribution_final"]["by_book"]
        self.assertNotIn("corpus-only", by_book)
        self.assertFalse(
            any("book=corpus-only" in item["scope"] for item in facts["insufficient_data"])
        )

    def test_mixed_splits_use_only_formal_books_for_gates(self):
        corpus, pack, evaluation, spec = self._completed()
        data = json.loads((corpus / "corpus.json").read_text())
        data["books"] = [
            {"book_id": "book0", "split": "formal"},
            {"book_id": "book1", "split": "screen"},
            {"book_id": "screen-only", "split": "screen"},
            {"book_id": "hidden-only", "split": "hidden"},
        ]
        (corpus / "corpus.json").write_text(json.dumps(data), encoding="utf-8")
        facts = self._analyze(corpus, pack, evaluation, spec)

        self.assertEqual(facts["denominators"]["books"], ["book0"])
        for metric in ("absolute", "mqm", "postedit"):
            by_book = facts[metric]["c1"]["attribution_final"]["by_book"]
            self.assertEqual(set(by_book), {"book0"})
        self.assertFalse(
            any(
                "book=book1" in item["scope"]
                or "book=screen-only" in item["scope"]
                or "book=hidden-only" in item["scope"]
                for item in facts["insufficient_data"]
            )
        )

        data["books"].append({"book_id": "formal-missing", "split": "formal"})
        (corpus / "corpus.json").write_text(json.dumps(data), encoding="utf-8")
        facts = self._analyze(corpus, pack, evaluation, spec)
        self.assertTrue(
            any(
                "book=formal-missing" in item["scope"] and item["reason"] == "missing_scope"
                for item in facts["insufficient_data"]
            )
        )

    def test_context_majority_all_uncertain_and_tie_are_pending(self):
        c0 = {
            "responses": [
                {"rater_id": "r1", "judgment": "uncertain"},
                {"rater_id": "r2", "judgment": "uncertain"},
            ]
        }
        c2 = {
            "responses": [
                {"rater_id": "r1", "judgment": "correct"},
                {"rater_id": "r2", "judgment": "incorrect"},
            ]
        }
        self.assertEqual(_paired(c0, c2), (0, 0, True))
        c0["responses"] = [
            {"rater_id": "r1", "judgment": "correct"},
            {"rater_id": "r2", "judgment": "incorrect"},
        ]
        c2["responses"] = [
            {"rater_id": "r1", "judgment": "uncertain"},
            {"rater_id": "r2", "judgment": "uncertain"},
        ]
        self.assertEqual(_paired(c0, c2), (0, 0, True))


if __name__ == "__main__":
    unittest.main()
