from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from trans_novel.benchmark.artifacts import canonical_json, sha256_bytes
from trans_novel.benchmark.report import ReportError, build_report, validate_report
from trans_novel.benchmark.review import finalize_review, prepare_review
from trans_novel.config import ModelRef
from trans_novel.llm.usage import UsageTracker


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_run(root: Path) -> tuple[Path, dict[str, object]]:
    run = root / "run"
    run.mkdir()
    run_manifest = {
        "schema_version": 3,
        "run_mode": "full",
        "benchmark_id": "review-test",
        "book_spec_sha256": "a" * 64,
        "candidate_spec_sha256": "b" * 64,
        "generation": {},
        "pipeline_variants": {
            "candidate-a": "minimal",
            "candidate-b": "minimal",
            "candidate-c": "polish",
        },
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
                        "lint_findings": [{"index": segment_index}] if segment_index == 4 else [],
                        "deterministic_findings": [],
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
                    "pipeline_variant": {
                        "candidate-a": "minimal",
                        "candidate-b": "minimal",
                        "candidate-c": "polish",
                    }[candidate],
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


def _usage() -> dict[str, object]:
    tracker = UsageTracker()
    tracker.record(
        agent="translator",
        operation="translate.batch",
        provider="test",
        model_ref=ModelRef("test", "model-a"),
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 25,
            "prompt_cache_miss_tokens": 75,
            "reasoning_tokens": 0,
        },
    )
    return tracker.summary()


def _complete_artifact_facts(run: Path) -> None:
    rows = json.loads((run / "candidates.json").read_text(encoding="utf-8"))
    for row in rows:
        row["usage"] = _usage()
        row["outputs"] = [f"outputs/{row['candidate_id']}/{row['book_id']}.epub"]
        row["output_hashes"] = {row["outputs"][0]: "a" * 64}
        relative = Path("telemetry") / row["candidate_id"] / f"{row['book_id']}.jsonl"
        telemetry = run / relative
        telemetry.parent.mkdir(parents=True, exist_ok=True)
        telemetry.write_text(
            canonical_json(
                {
                    "provider": "test",
                    "requested_model": "model-a",
                    "resolved_model": "model-a",
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cache_hit_tokens": 25,
                    "cache_miss_tokens": 75,
                    "reasoning_tokens": 0,
                    "billed_usage_unknown": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        row["telemetry_path"] = str(relative)
        row["telemetry_sha256"] = sha256_bytes(telemetry.read_bytes())
    _write_json(run / "candidates.json", rows)


def _write_prices(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "provider": "test",
                "region": "global",
                "currency": "USD",
                "retrieved_at": "2026-01-01T00:00:00Z",
                "source_urls": ["https://example.com/prices"],
                "models": {
                    "test:model-a": {
                        "model_id": "test:model-a",
                        "rules": [
                            {
                                "min_prompt_tokens": 0,
                                "max_prompt_tokens": None,
                                "time_band": "all",
                                "input_uncached_per_million": "1",
                                "input_cached_per_million": "0.5",
                                "output_per_million": "2",
                            }
                        ],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class TestBenchmarkReport(unittest.TestCase):
    def test_builds_quality_cost_and_evidence_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            _complete_artifact_facts(run)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            finalize_review(review, results)
            price = root / "prices.yaml"
            _write_prices(price)

            manifest = build_report(run, review, price, root / "report")
            self.assertEqual(manifest["status"], "final")
            self.assertEqual(
                validate_report(root / "report")["report_sha256"], manifest["report_sha256"]
            )
            costs = json.loads((root / "report" / "costs.json").read_text(encoding="utf-8"))
            self.assertTrue(
                all(value["api_cost"] is not None for value in costs["candidates"].values())
            )
            self.assertTrue((root / "report" / "findings.jsonl").read_text(encoding="utf-8"))
            self.assertIn(
                "Weighted errors / 10k",
                (root / "report" / "report.html").read_text(encoding="utf-8"),
            )

    def test_missing_price_keeps_report_provisional_without_a_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            _complete_artifact_facts(run)
            rows = json.loads((run / "candidates.json").read_text(encoding="utf-8"))
            for row in rows:
                telemetry = run / row["telemetry_path"]
                attempt = json.loads(telemetry.read_text(encoding="utf-8"))
                attempt["requested_model"] = "unknown"
                attempt["resolved_model"] = "unknown"
                telemetry.write_text(canonical_json(attempt) + "\n", encoding="utf-8")
                row["telemetry_sha256"] = sha256_bytes(telemetry.read_bytes())
            _write_json(run / "candidates.json", rows)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            finalize_review(review, results)
            price = root / "prices.yaml"
            _write_prices(price)

            build_report(run, review, price, root / "report")
            summary = json.loads((root / "report" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "provisional")
            self.assertIsNone(summary["winner"])

    def test_invalid_telemetry_errors_follow_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            _complete_artifact_facts(run)
            rows = json.loads((run / "candidates.json").read_text(encoding="utf-8"))
            for row in rows:
                if row["candidate_id"] in {"candidate-a", "candidate-b"}:
                    telemetry = run / row["telemetry_path"]
                    telemetry.write_text("{bad json\n", encoding="utf-8")
                    row["telemetry_sha256"] = sha256_bytes(telemetry.read_bytes())
            rows.reverse()
            _write_json(run / "candidates.json", rows)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            finalize_review(review, results)
            price = root / "prices.yaml"
            _write_prices(price)

            with self.assertRaisesRegex(ReportError, "candidate-a"):
                build_report(run, review, price, root / "report")

    def test_report_rejects_review_from_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, spec = _write_run(root)
            _complete_artifact_facts(run)
            review = prepare_review(run, spec, root / "review")
            results = root / "results"
            _write_results(review, results)
            finalize_review(review, results)
            price = root / "prices.yaml"
            _write_prices(price)
            manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
            manifest["benchmark_id"] = "different-run"
            _write_json(run / "run.json", manifest)

            with self.assertRaisesRegex(ReportError, "does not belong"):
                build_report(run, review, price, root / "report")


if __name__ == "__main__":
    unittest.main()
