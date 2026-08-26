from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trans_novel.benchmark.corpus import canonical_json, sha256_bytes
from trans_novel.benchmark.review import (
    ReviewArtifactError,
    finalize_review,
    prepare_review,
    validate_review,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_run(root: Path) -> tuple[Path, dict[str, object]]:
    run = root / "run"
    run.mkdir()
    run_manifest = {
        "schema_version": 2,
        "run_mode": "full",
        "benchmark_id": "review-test",
        "book_spec_sha256": "a" * 64,
        "candidate_spec_sha256": "b" * 64,
        "generation": {},
        "quality": "quality",
        "book_ids": [f"formal-{index:02d}" for index in range(1, 7)],
        "replicates": 1,
    }
    _write_json(run / "run.json", run_manifest)
    _write_json(
        run / "run_state.json", {"schema_version": 1, "status": "completed", "artifacts": {}}
    )
    artifacts = []
    sources = [
        '"You must leave now," Alice said.',
        "Professor Rowan carried the Atlas across London.",
        "This is a narrative sentence about a quiet evening.",
        "A" * 520,
        "She remembered the promise from the previous winter.",
    ]
    labels = {"candidate-a": "甲", "candidate-b": "乙", "candidate-c": "丙"}
    for candidate in labels:
        for book_index in range(1, 7):
            book_id = f"formal-{book_index:02d}"
            relative = Path("segments") / candidate / f"{book_id}.jsonl"
            rows = []
            for segment_index, source in enumerate(sources):
                segment_id = sha256_bytes(f"{book_id}:{segment_index}:{source}".encode())
                rows.append(
                    {
                        "segment_id": segment_id,
                        "chapter_index": 0,
                        "chapter_title": f"Chapter {book_index}",
                        "segment_index": segment_index,
                        "kind": "paragraph",
                        "source": source,
                        "target": f"{labels[candidate]}译文{book_index}-{segment_index}",
                        "review_findings": (
                            [{"index": segment_index}] if segment_index == 4 else []
                        ),
                        "backtranslation_findings": [],
                        "source_sha256": sha256_bytes(source.encode()),
                        "target_sha256": "c" * 64,
                    }
                )
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
            artifacts.append(
                {
                    "candidate_id": candidate,
                    "book_id": book_id,
                    "replicate": 1,
                    "segments_path": str(relative),
                }
            )
    _write_json(run / "candidates.json", artifacts)
    spec = {
        "schema_version": 1,
        "benchmark_id": "review-test",
        "run_sha256": sha256_bytes((run / "run.json").read_bytes()),
        "seed": 17,
        "segments_per_book": 4,
        "shard_count": 5,
    }
    return run, spec


def _write_results(review: Path, results: Path, *, all_ties: bool = False) -> None:
    manifest = json.loads((review / "review.json").read_text(encoding="utf-8"))
    results.mkdir(parents=True)
    for shard in manifest["shards"]:
        payload = json.loads(
            (review / "shards" / f"{shard['shard_id']}.json").read_text(encoding="utf-8")
        )
        reviews = []
        for index, unit in enumerate(payload["units"]):
            if all_ties:
                winner = "tie"
                findings = []
            else:
                winner = "A"
                findings = [
                    {
                        "finding_id": f"finding-{index}",
                        "side": "B",
                        "type": "mistranslation",
                        "severity": "major",
                        "source_quote": unit["source"][:1],
                        "target_quote": unit["targets"]["B"][:1],
                        "reason": "The selected translation loses the quoted source meaning.",
                    }
                ]
            reviews.append(
                {
                    "unit_id": unit["unit_id"],
                    "winner": winner,
                    "verdict_reason": "The evidence above determines the comparison.",
                    "findings": findings,
                }
            )
        _write_json(
            results / f"{shard['shard_id']}.json",
            {
                "schema_version": 1,
                "review_sha256": manifest["review_sha256"],
                "shard_id": shard["shard_id"],
                "reviews": reviews,
            },
        )


class TestBenchmarkReview(unittest.TestCase):
    def test_sampling_and_blinding_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            first = prepare_review(run, spec, root / "review-a")
            second = prepare_review(run, spec, root / "review-b")
            self.assertEqual(
                (first / "review.json").read_bytes(), (second / "review.json").read_bytes()
            )
            self.assertEqual(validate_review(first)["unit_count"], 72)
            manifest = json.loads((first / "review.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["shards"]), 5)
            secret = json.loads((first / "secret_mapping.json").read_text(encoding="utf-8"))
            self.assertTrue(all(set(mapping) == {"A", "B"} for mapping in secret["units"].values()))
            mappings = list(secret["units"].values())
            self.assertEqual(
                {frozenset(mapping.values()) for mapping in mappings},
                {
                    frozenset(("candidate-a", "candidate-b")),
                    frozenset(("candidate-a", "candidate-c")),
                    frozenset(("candidate-b", "candidate-c")),
                },
            )
            self.assertEqual(
                {
                    candidate: sum(candidate in mapping.values() for mapping in mappings)
                    for candidate in ("candidate-a", "candidate-b", "candidate-c")
                },
                {"candidate-a": 48, "candidate-b": 48, "candidate-c": 48},
            )

    def test_finalize_validates_quotes_and_aggregates_unblinded_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            finalize_review(review, results)
            comparison = json.loads((review / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(comparison["candidates"]),
                {"candidate-a", "candidate-b", "candidate-c"},
            )
            self.assertEqual(sum(row["wins"] for row in comparison["candidates"].values()), 72)
            self.assertEqual(validate_review(review)["status"], "complete")
            findings = (review / "findings.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(findings), 72)

    def test_finalize_rejects_a_target_quote_outside_the_selected_translation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            first = next(results.glob("*.json"))
            payload = json.loads(first.read_text(encoding="utf-8"))
            payload["reviews"][0]["findings"][0]["target_quote"] = "not present"
            _write_json(first, payload)
            with self.assertRaisesRegex(ReviewArtifactError, "target quote"):
                finalize_review(review, results)

    def test_all_ties_do_not_manufacture_a_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results, all_ties=True)
            finalize_review(review, results)
            comparison = json.loads((review / "comparison.json").read_text(encoding="utf-8"))
            self.assertIsNone(comparison["winner"])


if __name__ == "__main__":
    unittest.main()
