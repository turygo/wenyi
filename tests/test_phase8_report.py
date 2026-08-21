from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

from trans_novel.benchmark.report import (
    ReportError,
    _pareto,
    _status,
    _validate_integration,
    build_report,
    reprice_report,
    validate_report,
)
from trans_novel.cli import app

HASHES = {
    "corpus_sha256": "a" * 64,
    "run_hash": "b" * 64,
    "preparation_sha256": "c" * 64,
    "pack_sha256": "d" * 64,
    "evaluation_sha256": "e" * 64,
    "price_snapshot_sha256": "f" * 64,
}
DIMENSIONS = (
    "fidelity",
    "naturalness",
    "style_voice",
    "consistency",
    "context_handling",
    "readability",
    "format_integrity",
)
RELIABILITY = (*DIMENSIONS, "pairwise_winner", "context_correctness", "mqm_severity", "mqm_type")
SYSTEM_FIELDS = (
    "protocol_errors",
    "json_errors",
    "alignment_errors",
    "required_node_failures",
    "resume_duplicate_operations",
    "reasoning_tokens",
    "resolved_model_mismatch",
)


def _dimension_values() -> dict[str, dict[str, float]]:
    return {name: {"raw_mean": 4.7, "lower95": 4.5} for name in DIMENSIONS}


def _mqm_scope() -> dict[str, object]:
    severity = {name: {"count": 0, "rate_per_10k": 0} for name in ("critical", "major", "minor")}
    return {
        "source_words": 1000,
        "agreed": {"severity": severity, "type": {}},
        "raw": {"severity": {}, "type": {}},
        "pending_adjudication_count": 0,
        "major_rate_upper95": 0,
        "major_rate_lower95": 0,
        "weighted_points_lower95": 0,
        "weighted_points_upper95": 0,
    }


def _human(
    candidates: tuple[str, ...] = ("a", "b"),
    books: tuple[str, ...] = ("book",),
    *,
    insufficient: list[dict[str, str]] | None = None,
    pending: int = 0,
) -> dict[str, object]:
    absolute: dict[str, object] = {}
    mqm: dict[str, object] = {}
    polish: dict[str, object] = {}
    postedit: dict[str, object] = {}
    pairwise_candidates: dict[str, object] = {}
    for index, candidate in enumerate(candidates):
        by_book = {
            book: {
                "dimensions": _dimension_values(),
                "composite": {
                    "value": 4.7 - index * 0.1,
                    "lower95": 4.5 - index * 0.1,
                    "n_units": 10,
                },
            }
            for book in books
        }
        absolute[candidate] = {
            "s1": {
                "by_book": by_book,
                "macro": {
                    "dimensions": _dimension_values(),
                    "composite": {
                        "value": 4.7 - index * 0.1,
                        "lower95": 4.5 - index * 0.1,
                        "n_units": 20,
                    },
                },
            }
        }
        mqm[candidate] = {
            "s1": {"by_book": {book: _mqm_scope() for book in books}, "macro": _mqm_scope()}
        }
        polish[candidate] = {
            "by_book": {book: {"harm": 0} for book in books},
            "macro": {"harm_wilson_upper95": 0},
            "mqm_semantic_harm": dict.fromkeys(books, 0),
        }
        postedit[candidate] = {"s1": {"macro": {"minutes_per_10k": 1}}}
        pairwise_candidates[candidate] = {
            "ability": 0.8 - index * 0.1,
            "ability_lower95": 0.7 - index * 0.1,
            "field_win": 0.8 - index * 0.1,
            "field_win_lower95": 0.7 - index * 0.1,
            "field_win_upper95": 0.9 - index * 0.1,
        }
    return {
        "schema_version": 1,
        "input_hashes": {
            **{key: HASHES[key] for key in ("corpus_sha256", "pack_sha256", "evaluation_sha256")},
            "evaluation_complete_sha256": "1" * 64,
        },
        "denominators": {"books": list(books)},
        "absolute": absolute,
        "mqm": mqm,
        "pairwise": {"s1": {"candidates": pairwise_candidates, "ratings": 20, "units": 10}},
        "polish": polish,
        "context": {candidate: {"by_strategy": {}, "lift": {}} for candidate in candidates},
        "reliability": dict.fromkeys(RELIABILITY, 0.9),
        "postedit": postedit,
        "pending_adjudication_count": pending,
        "insufficient_data": insufficient or [],
        "needs_recalibration": False,
    }


def _system(candidates: tuple[str, ...], books: tuple[str, ...]) -> dict[str, object]:
    def one() -> dict[str, object]:
        return {"completion_rate": "1", **dict.fromkeys(SYSTEM_FIELDS, 0), "latency_p95_ms": 10}

    return {
        candidate: {**one(), "by_book": {book: one() for book in books}} for candidate in candidates
    }


def _cost(
    candidates: tuple[str, ...] = ("a", "b"),
    books: tuple[str, ...] = ("book",),
    *,
    unknown: str | None = None,
    insufficient: list[dict[str, str]] | None = None,
    rates: tuple[str, ...] = ("50", "100"),
) -> dict[str, object]:
    candidate_costs: dict[str, object] = {}
    effective: dict[str, object] = {}
    for index, candidate in enumerate(candidates):
        candidate_costs[candidate] = {
            "api_cost": str(index + 1),
            "api_cost_lower_bound": str(index + 1),
            "unknown_count": 1 if candidate == unknown else 0,
            "validated_source_words": 1000,
            "cost_complete": candidate != unknown,
            "by_operation": {},
        }
        effective[candidate] = {
            rate: {"s1": {"value": str(index + 2 + int(rate) / 100)}} for rate in rates
        }
    return {
        "schema_version": 1,
        "input_hashes": {
            key: HASHES[key] for key in ("run_hash", "preparation_sha256", "price_snapshot_sha256")
        },
        "candidate_costs": candidate_costs,
        "physical_spend": {"api_cost_lower_bound": "3", "cost_complete": True, "unknown_count": 0},
        "million_word_estimate": {
            candidate: {
                "value": "1000",
                "lower95": "900",
                "upper95": "1100",
                "complete": True,
                "by_book": {book: {"complete": True, "value": "1000"} for book in books},
            }
            for candidate in candidates
        },
        "effective_costs": effective,
        "system_metrics": _system(candidates, books),
        "normalized_pricing_facts": {
            "schema_version": 1,
            "artifacts": [],
            "candidate_artifact_ids": [],
            "physical_artifact_ids": [],
            "preparation_artifact_ids_by_book": {},
            "candidate_source_words": {},
            "candidate_source_words_by_book": {},
            "system_facts": {},
            "initial_insufficient_data": [],
            "bootstrap_seed": 7,
            "bootstrap_replicates": 1000,
        },
        "insufficient_data": insufficient or [],
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def _spec_text(rates: tuple[str, ...] = ("50", "100")) -> str:
    lines = [
        "schema_version: 1",
        "benchmark_id: phase8",
        *[f"{key}: {value}" for key, value in HASHES.items()],
        "bootstrap_seed: 7",
        "bootstrap_replicates: 1000",
        "editor_hourly_rates:",
    ]
    lines.extend(f"  - '{rate}'" for rate in rates)
    return "\n".join(lines) + "\n"


def _write_integration(
    root: Path,
    candidates: tuple[str, ...] = ("a", "b"),
    *,
    failed: str | None = None,
    completion_status: str = "completed",
) -> Path:
    request = {
        "schema_version": 1,
        "benchmark_id": "phase8",
        "corpus_sha256": HASHES["corpus_sha256"],
        "book_spec_sha256": "b" * 64,
        "candidate_spec_sha256": "c" * 64,
        "integration_spec_sha256": "d" * 64,
        "book_id": "book",
        "source_sha256": "e" * 64,
        "source_language": "en",
        "target_language": "zh",
        "candidate_ids": list(candidates),
        "candidates": {
            candidate: {
                "provider": "bailian",
                "primary_model": (
                    "qwen3.8-max:off" if candidate == "a" else "deepseek-v4-flash:off"
                ),
                "editor_model": ("deepseek-v4-pro:off" if candidate == "a" else "qwen3.7-plus:off"),
                "fast_model": "qwen3.7-flash:off",
                "temperature": 0.1,
                "seed": None,
            }
            for candidate in candidates
        },
        "interrupt_after_committed_batches": 1,
        "output_mono": True,
        "output_bilingual": True,
        "bilingual_order": "target_first",
    }
    request_path = root / "integration_request.json"
    request_path.write_bytes(_canonical_json(request))
    request_hash = hashlib.sha256(request_path.read_bytes()).hexdigest()
    entries: dict[str, object] = {}
    complete_entries: dict[str, object] = {}
    for candidate in candidates:
        candidate_root = root / "candidates" / candidate
        candidate_root.mkdir(parents=True)
        mono = candidate_root / "mono.epub"
        bilingual = candidate_root / "bilingual.epub"
        mono.write_bytes(b"mono")
        bilingual.write_bytes(b"bilingual")
        telemetry = candidate_root / "telemetry.jsonl"
        first = candidate_root / "telemetry.first.jsonl"
        resume = candidate_root / "telemetry.resume.jsonl"
        telemetry.write_bytes(b"")
        first.write_bytes(b"")
        resume.write_bytes(b"")
        usage = candidate_root / "usage.json"
        usage.write_bytes(
            _canonical_json(
                {
                    "schema_version": 2,
                    "totals": {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cache_hit_tokens": 0,
                        "cache_miss_tokens": 0,
                    },
                    "by_agent": {},
                }
            )
        )
        failed_result = candidate == failed or completion_status == "failed"
        result = {
            "schema_version": 1,
            "candidate_id": candidate,
            "passed": not failed_result,
            "canary_passed": not failed_result,
            "expected_interruption_observed": not failed_result,
            "readiness_passed": not failed_result,
            "resume_duplicate_operations": 0,
            "reasoning_tokens": 0,
            "model_mismatch_count": 0,
            "unknown_required_usage_count": 0,
            "structural": {"structural_pass": not failed_result},
            "mono": {"structural_pass": not failed_result},
            "bilingual": {"structural_pass": not failed_result},
            "request_sha256": request_hash,
            "corpus_sha256": request["corpus_sha256"],
            "book_spec_sha256": request["book_spec_sha256"],
            "candidate_spec_sha256": request["candidate_spec_sha256"],
            "integration_spec_sha256": request["integration_spec_sha256"],
            "source_sha256": request["source_sha256"],
            "benchmark_id": request["benchmark_id"],
            "book_id": request["book_id"],
        }
        if not failed_result:
            result.update(
                {
                    "committed_batches": 0,
                    "skipped_batches": 0,
                    "remaining_batches": 0,
                    "repeated_batches": 0,
                    "phase_timings_ms": {
                        "prepare": 0,
                        "translate": 0,
                        "quality": 0,
                        "first_attempt": 1,
                        "resume": 1,
                        "total": 2,
                    },
                    "output_paths": {
                        "mono": f"candidates/{candidate}/mono.epub",
                        "bilingual": f"candidates/{candidate}/bilingual.epub",
                    },
                    "output_sha256": {
                        "mono": hashlib.sha256(mono.read_bytes()).hexdigest(),
                        "bilingual": hashlib.sha256(bilingual.read_bytes()).hexdigest(),
                    },
                    "telemetry_path": f"candidates/{candidate}/telemetry.jsonl",
                    "telemetry_sha256": hashlib.sha256(telemetry.read_bytes()).hexdigest(),
                    "first_telemetry_path": f"candidates/{candidate}/telemetry.first.jsonl",
                    "first_telemetry_sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                    "first_telemetry_count": 0,
                    "resume_telemetry_path": f"candidates/{candidate}/telemetry.resume.jsonl",
                    "resume_telemetry_sha256": hashlib.sha256(resume.read_bytes()).hexdigest(),
                    "resume_telemetry_count": 0,
                    "resume_attempt_telemetry_count": 0,
                    "telemetry_counts": {
                        "logical_call_count": 0,
                        "attempt_count": 0,
                        "operation_count": 0,
                        "agent_count": 0,
                        "retry_count": 0,
                        "translate_call_count": 0,
                    },
                    "usage_path": f"candidates/{candidate}/usage.json",
                    "usage_sha256": hashlib.sha256(usage.read_bytes()).hexdigest(),
                }
            )
        result_path = candidate_root / "result.json"
        result_path.write_bytes(_canonical_json(result))
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
        relative = f"candidates/{candidate}/result.json"
        entries[candidate] = {"result_path": relative, "result_sha256": digest}
        complete_entries[candidate] = {
            "result_path": relative,
            "result_sha256": digest,
            "status": "failed" if failed_result else "completed",
        }
    integration = {
        "schema_version": 1,
        "benchmark_id": request["benchmark_id"],
        "corpus_sha256": request["corpus_sha256"],
        "book_spec_sha256": request["book_spec_sha256"],
        "candidate_spec_sha256": request["candidate_spec_sha256"],
        "integration_spec_sha256": request["integration_spec_sha256"],
        "book_id": request["book_id"],
        "source_sha256": request["source_sha256"],
        "candidates": entries,
    }
    path = root / "integration.json"
    path.write_bytes(_canonical_json(integration))
    complete = {
        "schema_version": 1,
        "benchmark_id": request["benchmark_id"],
        "integration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "terminal": True,
        "candidates": complete_entries,
    }
    (root / "integration_complete.json").write_bytes(_canonical_json(complete))
    return path


class Phase8ReportContractTests(unittest.TestCase):
    def _build(
        self,
        root: Path,
        out: Path,
        human: dict[str, object] | None = None,
        cost: dict[str, object] | None = None,
        *,
        integration: Path | None = None,
        rates: tuple[str, ...] = ("50", "100"),
    ) -> dict[str, object]:
        spec = root / "REPORT_SPEC.yaml"
        spec.write_text(_spec_text(rates), encoding="utf-8")
        human = human or _human()
        cost = cost or _cost()
        with (
            patch("trans_novel.benchmark.report.analyze_human", return_value=human),
            patch("trans_novel.benchmark.report.analyze_cost_system", return_value=cost),
        ):
            return build_report(
                root / "corpus",
                root / "run",
                root / "preparation",
                root / "pack",
                root / "evaluation",
                root / "price.yaml",
                spec,
                out,
                integration_path=integration,
            )

    def test_evaluation_completion_hash_binds_request_identity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._build(root, root / "first", human=_human())
            changed = _human()
            changed["input_hashes"]["evaluation_complete_sha256"] = "2" * 64
            second = self._build(root, root / "second", human=changed)
            first_manifest = validate_report(Path(first["out_dir"]))
            second_manifest = validate_report(Path(second["out_dir"]))
            self.assertNotEqual(first["request_sha256"], second["request_sha256"])
            self.assertNotEqual(first_manifest["request_sha256"], second_manifest["request_sha256"])
            with self.assertRaises(ReportError):
                self._build(root, root / "first", human=changed)

    def test_existing_status_precedence(self) -> None:
        human = {
            "needs_recalibration": True,
            "insufficient_data": [{"scope": "global", "reason": "missing"}],
        }
        self.assertEqual(_status(human, {"insufficient_data": []}, None), "insufficient_data")

    def test_existing_surface_local_pareto(self) -> None:
        rows = [
            {
                "entity_id": "a@s1",
                "surface": "s1",
                "gate_pass": True,
                "gate_pass_without_integration": True,
                "api_cost": "1",
                "mqm_upper95": 1,
                "composite_lower95": 90,
                "bt_lower95": 0.8,
                "wall_p95_ms": 10,
                "effective_costs": {},
            },
            {
                "entity_id": "b@s2",
                "surface": "s2",
                "gate_pass": True,
                "gate_pass_without_integration": True,
                "api_cost": "2",
                "mqm_upper95": 2,
                "composite_lower95": 80,
                "bt_lower95": 0.7,
                "wall_p95_ms": 20,
                "effective_costs": {},
            },
        ]
        self.assertEqual(_pareto(rows, "final")["s1"]["api"], ["a@s1"])
        self.assertEqual(_pareto(rows, "final")["s2"]["api"], ["b@s2"])

    def test_existing_integration_shape_validation(self) -> None:
        class Spec:
            benchmark_id = "phase8"
            corpus_sha256 = HASHES["corpus_sha256"]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "integration.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ReportError):
                _validate_integration(path, Spec(), set())

    def test_public_validate_and_exact_noop(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report"
            first = self._build(root, out)
            second = self._build(root, out)
            manifest = validate_report(out)
            self.assertFalse(first["no_op"])
            self.assertTrue(second["no_op"])
            self.assertEqual(manifest["report_sha256"], first["report_sha256"])

    def test_absent_integration_is_provisional_public_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._build(root, root / "report")
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(result["status"], "provisional")
            self.assertEqual(summary["recommendations"], {})
            self.assertEqual(
                summary["pareto"], {"s1": {"api": [], "effective": {}, "excluded": {}}}
            )

    def test_valid_two_candidate_integration_is_final_public_workflow(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            result = self._build(root, root / "report", integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(result["status"], "final")
            self.assertEqual(summary["status"], "final")
            self.assertEqual(
                set(summary["recommendations"]["s1"]),
                {"cheapest", "effective_value", "highest_quality"},
            )

    def test_failed_completion_cannot_final(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root, completion_status="failed")
            result = self._build(root, root / "report", integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertNotEqual(result["status"], "final")
            self.assertTrue(
                any(
                    "integration_failed" in reason for reason in summary["candidates"][0]["reasons"]
                )
            )

    def test_quality_gate_failure_cannot_final(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            human = _human()
            for candidate in ("a", "b"):
                human["absolute"][candidate]["s1"]["macro"]["dimensions"]["fidelity"][
                    "raw_mean"
                ] = 4.0
            result = self._build(root, root / "report", human=human, integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertNotEqual(result["status"], "final")
            self.assertIn("fidelity", summary["withheld_reasons"]["a@s1"])
            self.assertEqual(summary["recommendations"], {})

    def test_exact_a_and_aa_scope_isolation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human(("a", "aa"))
            human["insufficient_data"] = [
                {"scope": "candidate=a/surface=s1", "reason": "missing_rating"}
            ]
            cost = _cost(
                ("a", "aa"),
                insufficient=[{"scope": "candidate=a/surface=s1", "reason": "missing_cost"}],
            )
            result = self._build(root, root / "report", human=human, cost=cost)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertIn(
                "candidate=a/surface=s1:missing_rating", summary["withheld_reasons"]["a@s1"]
            )
            self.assertIn(
                "candidate=a/surface=s1:missing_cost", summary["withheld_reasons"]["a@s1"]
            )
            self.assertNotIn("missing_rating", ";".join(summary["withheld_reasons"]["aa@s1"]))
            self.assertNotIn("missing_cost", ";".join(summary["withheld_reasons"]["aa@s1"]))
            self.assertEqual(result["status"], "provisional")

    def test_global_vs_entity_book_human_and_cost_insufficiency(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human(
                ("a", "b"),
                ("book1", "book2"),
                insufficient=[{"scope": "candidate=a/book=book1", "reason": "human_missing"}],
            )
            cost = _cost(
                ("a", "b"),
                ("book1", "book2"),
                insufficient=[{"scope": "candidate=b/book=book2", "reason": "cost_missing"}],
            )
            result = self._build(root, root / "report", human=human, cost=cost)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertIn(
                "book1:candidate=a/book=book1:human_missing", summary["withheld_reasons"]["a@s1"]
            )
            self.assertIn(
                "book2:candidate=b/book=book2:cost_missing", summary["withheld_reasons"]["b@s1"]
            )
            self.assertFalse(summary["candidates"][0]["gate_pass"])
            self.assertFalse(summary["candidates"][1]["gate_pass"])
            global_human = _human(
                insufficient=[{"scope": "global", "reason": "denominator_missing"}]
            )
            global_cost = _cost(insufficient=[{"scope": "global", "reason": "spend_missing"}])
            global_result = self._build(root, root / "global", human=global_human, cost=global_cost)
            global_summary = json.loads((root / "global" / "summary.json").read_text())
            self.assertEqual(global_result["status"], "insufficient_data")
            self.assertEqual(global_summary["recommendations"], {})
            self.assertEqual(result["status"], "provisional")

    def test_wilson_boundary_is_observable_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            human = _human()
            human["polish"]["a"]["macro"]["harm_wilson_upper95"] = 0.0101
            self._build(root, root / "report", human=human, integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            candidate = next(row for row in summary["candidates"] if row["entity_id"] == "a@s1")
            self.assertFalse(candidate["gate_pass"])
            self.assertIn("polish_harm_upper95", summary["withheld_reasons"]["a@s1"])
            self.assertNotIn("a@s1", summary["pareto"]["s1"]["api"])
            self.assertNotEqual(summary["recommendations"]["s1"]["cheapest"]["entity_id"], "a@s1")

    def test_configurable_rate_columns_and_recommendations(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            self._build(
                root,
                root / "report",
                cost=_cost(rates=("25", "75")),
                integration=integration,
                rates=("25", "75"),
            )
            header = (root / "report" / "candidates.csv").read_text().splitlines()[0]
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertIn("effective_cost_rate_25", header)
            self.assertIn("effective_cost_rate_75", header)
            self.assertNotIn("effective_cost_rate_50", header)
            self.assertEqual(set(summary["recommendations"]["s1"]["effective_value"]), {"25", "75"})

    def test_incomplete_existing_output_refuses_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report"
            out.mkdir()
            marker = out / "summary.json"
            marker.write_bytes(b"incomplete")
            before = marker.read_bytes()
            with self.assertRaises(ReportError):
                self._build(root, out)
            self.assertEqual(marker.read_bytes(), before)

    def test_corrupt_existing_output_refuses_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report"
            self._build(root, out)
            target = out / "summary.json"
            target.write_bytes(b"corrupt")
            before = target.read_bytes()
            with self.assertRaises(ReportError):
                self._build(root, out)
            self.assertEqual(target.read_bytes(), before)

    def test_file_tampered_destination_refuses_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report"
            self._build(root, out)
            target = out / "quality_by_book.csv"
            target.write_bytes(target.read_bytes() + b"tamper")
            before = target.read_bytes()
            with self.assertRaises(ReportError):
                self._build(root, out)
            self.assertEqual(target.read_bytes(), before)

    def test_request_mismatch_destination_refuses_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "report"
            self._build(root, out)
            before = {path.name: path.read_bytes() for path in out.iterdir()}
            with self.assertRaises(ReportError):
                self._build(root, out, rates=("75", "125"))
            self.assertEqual(before, {path.name: path.read_bytes() for path in out.iterdir()})

    def test_tampered_parent_blocks_reprice_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            target = report / "summary.json"
            target.write_bytes(target.read_bytes() + b"tamper")
            before = target.read_bytes()
            with (
                patch(
                    "trans_novel.benchmark.report.reprice_cost_system",
                    side_effect=AssertionError("repricer called"),
                ),
                self.assertRaises(ReportError),
            ):
                reprice_report(report, root / "new-price.yaml", root / "repriced")
            self.assertEqual(target.read_bytes(), before)
            self.assertFalse((root / "repriced").exists())

    def test_cli_build_writes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.yaml"
            spec.write_text(_spec_text(), encoding="utf-8")
            args = [
                "tools",
                "benchmark",
                "report",
                "build",
                str(root / "corpus"),
                str(root / "run"),
                str(root / "prep"),
                str(root / "pack"),
                str(root / "eval"),
                str(root / "price.yaml"),
                str(spec),
                "--out",
                str(root / "report"),
            ]
            with (
                patch("trans_novel.benchmark.report.analyze_human", return_value=_human()),
                patch("trans_novel.benchmark.report.analyze_cost_system", return_value=_cost()),
            ):
                result = CliRunner().invoke(app, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Report written", result.output)

    def test_cli_build_noop_message(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "spec.yaml"
            spec.write_text(_spec_text(), encoding="utf-8")
            args = [
                "tools",
                "benchmark",
                "report",
                "build",
                str(root / "corpus"),
                str(root / "run"),
                str(root / "prep"),
                str(root / "pack"),
                str(root / "eval"),
                str(root / "price.yaml"),
                str(spec),
                "--out",
                str(root / "report"),
            ]
            with (
                patch("trans_novel.benchmark.report.analyze_human", return_value=_human()),
                patch("trans_novel.benchmark.report.analyze_cost_system", return_value=_cost()),
            ):
                first = CliRunner().invoke(app, args)
                result = CliRunner().invoke(app, args)
            self.assertEqual(first.exit_code, 0, first.output)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Report no-op", result.output)

    def test_cli_reprice_noop_message(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            price = root / "new-price.yaml"
            price.write_text("price: 2\n", encoding="utf-8")
            new_cost = _cost()
            new_cost["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            args = [
                "tools",
                "benchmark",
                "report",
                "reprice",
                str(report),
                str(price),
                "--out",
                str(root / "repriced"),
            ]
            with (
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=new_cost),
            ):
                first = CliRunner().invoke(app, args)
                result = CliRunner().invoke(app, args)
            self.assertEqual(first.exit_code, 0, first.output)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Report reprice no-op", result.output)

    def test_cli_reprice_message(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            price = root / "new-price.yaml"
            price.write_text("price: 2\n", encoding="utf-8")
            new_cost = _cost()
            new_cost["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            with (
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=new_cost),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "tools",
                        "benchmark",
                        "report",
                        "reprice",
                        str(report),
                        str(price),
                        "--out",
                        str(root / "repriced"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("Report repriced", result.output)

    def test_cli_invalid_path_is_nonzero(self) -> None:
        with TemporaryDirectory() as directory:
            result = CliRunner().invoke(
                app, ["tools", "benchmark", "report", "build", str(Path(directory) / "missing")]
            )
            self.assertNotEqual(result.exit_code, 0)

    def test_html_has_no_raw_passage_or_external_url(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, root / "report")
            html = (root / "report" / "report.html").read_text()
            self.assertNotIn("http://", html)
            self.assertNotIn("https://", html)
            self.assertNotIn("raw fixture passage", html)

    def test_reprice_isolated_from_source_analyzers(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            new_cost = _cost()
            new_cost["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            with (
                patch(
                    "trans_novel.benchmark.report.analyze_human",
                    side_effect=AssertionError("human analyzer called"),
                ),
                patch(
                    "trans_novel.benchmark.report.analyze_cost_system",
                    side_effect=AssertionError("cost analyzer called"),
                ),
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=new_cost),
            ):
                result = reprice_report(report, root / "price.yaml", root / "repriced")
            self.assertFalse(result["no_op"])
            self.assertTrue((root / "repriced" / "summary.json").is_file())

    def test_chained_reprice_has_root_current_parent_lineage_in_html(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            prices = ["1" * 64, "2" * 64]
            parent = report
            for index, digest in enumerate(prices, 1):
                cost = _cost()
                cost["input_hashes"]["price_snapshot_sha256"] = digest
                destination = root / f"repriced{index}"
                with (
                    patch(
                        "trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()
                    ),
                    patch("trans_novel.benchmark.report_cost._price_hash", return_value=digest),
                    patch("trans_novel.benchmark.report.reprice_cost_system", return_value=cost),
                ):
                    reprice_report(parent, root / f"price{index}.yaml", destination)
                parent = destination
            first_manifest = validate_report(root / "repriced1")
            second_manifest = validate_report(root / "repriced2")
            html = (root / "repriced2" / "report.html").read_text()
            self.assertIn(first_manifest["report_semantic_sha256"], html)
            self.assertIn(second_manifest["original_price_snapshot_sha256"], html)
            self.assertIn(second_manifest["current_price_snapshot_sha256"], html)
            self.assertIn(second_manifest["parent_report_semantic_sha256"], html)

    def test_reprice_preserves_price_independent_bytes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            before = {
                name: (report / name).read_bytes()
                for name in (
                    "quality_by_book.csv",
                    "mqm_errors.csv",
                    "pairwise.csv",
                    "context_ablation.csv",
                    "polish_effect.csv",
                )
            }
            cost = _cost()
            cost["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            with (
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=cost),
            ):
                reprice_report(report, root / "price.yaml", root / "repriced")
            self.assertEqual(
                before, {name: (root / "repriced" / name).read_bytes() for name in before}
            )

    def test_final_recommendation_decimal_serialization_and_selected_subset(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root, ("a", "b"))
            result = self._build(
                root,
                root / "report",
                human=_human(("a", "b", "c")),
                cost=_cost(("a", "b", "c")),
                integration=integration,
            )
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(summary["recommendations"]["s1"]["cheapest"]["entity_id"], "a@s1")
            self.assertIsInstance(summary["recommendations"]["s1"]["cheapest"]["value"], str)
            self.assertEqual(set(summary["integration"]["candidates"]), {"a", "b"})
            selected = {
                item["entity_id"]
                for item in summary["recommendations"]["s1"]["effective_value"].values()
            }
            self.assertNotIn("c@s1", selected)
            self.assertEqual(result["status"], "final")

    def test_adjacent_decimal_costs_remain_distinct_in_pareto(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            integration = _write_integration(root)
            human["absolute"]["b"]["s1"] = json.loads(json.dumps(human["absolute"]["a"]["s1"]))
            human["mqm"]["b"]["s1"] = json.loads(json.dumps(human["mqm"]["a"]["s1"]))
            human["pairwise"]["s1"]["candidates"]["b"] = json.loads(
                json.dumps(human["pairwise"]["s1"]["candidates"]["a"])
            )
            cost = _cost()
            cost["candidate_costs"]["a"]["api_cost"] = "1.00000000000000001"
            cost["candidate_costs"]["b"]["api_cost"] = "1.00000000000000002"
            cost["effective_costs"]["a"]["50"]["s1"]["value"] = "1.00000000000000001"
            cost["effective_costs"]["b"]["50"]["s1"]["value"] = "1.00000000000000002"
            cost["effective_costs"]["a"]["100"]["s1"]["value"] = "1.00000000000000001"
            cost["effective_costs"]["b"]["100"]["s1"]["value"] = "1.00000000000000002"
            self._build(root, root / "report", human=human, cost=cost, integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(summary["pareto"]["s1"]["api"], ["a@s1"])
            self.assertEqual(summary["recommendations"]["s1"]["cheapest"]["entity_id"], "a@s1")

    def test_per_book_boundary_matrix_uses_book_facts(self) -> None:
        failing = ("completion", "critical", "major", "reasoning", "model", "resume")
        for case in failing:
            with self.subTest(case=case), TemporaryDirectory() as directory:
                root = Path(directory)
                human = _human()
                cost = _cost()
                integration = _write_integration(root)
                if case == "completion":
                    cost["system_metrics"]["a"]["by_book"]["book"]["completion_rate"] = "0.99"
                elif case == "critical":
                    human["mqm"]["a"]["s1"]["by_book"]["book"]["agreed"]["severity"]["critical"][
                        "count"
                    ] = 1
                elif case == "major":
                    human["mqm"]["a"]["s1"]["by_book"]["book"]["agreed"]["severity"]["major"][
                        "rate_per_10k"
                    ] = 2.01
                elif case == "reasoning":
                    cost["system_metrics"]["a"]["by_book"]["book"]["reasoning_tokens"] = 1
                elif case == "model":
                    cost["system_metrics"]["a"]["by_book"]["book"]["resolved_model_mismatch"] = 1
                else:
                    cost["system_metrics"]["a"]["by_book"]["book"][
                        "resume_duplicate_operations"
                    ] = 1
                self._build(root, root / "report", human=human, cost=cost, integration=integration)
                summary = json.loads((root / "report" / "summary.json").read_text())
                row = next(row for row in summary["candidates"] if row["entity_id"] == "a@s1")
                self.assertFalse(row["gate_pass"])
                expected = {
                    "completion": "book:completion",
                    "critical": "book:critical",
                    "major": "book:per_book_major",
                    "reasoning": "book:reasoning_tokens",
                    "model": "book:resolved_model_mismatch",
                    "resume": "book:resume_duplicate_operations",
                }[case]
                self.assertIn(expected, summary["withheld_reasons"]["a@s1"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["mqm"]["a"]["s1"]["by_book"]["book"]["agreed"]["severity"]["major"][
                "rate_per_10k"
            ] = 2.0
            result = self._build(
                root, root / "report", human=human, integration=_write_integration(root)
            )
            self.assertEqual(result["status"], "final")

    def test_cli_full_arity_invalid_path_and_schema(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing"
            spec = root / "bad-spec.yaml"
            spec.write_text("schema_version: 99\n", encoding="utf-8")
            build_args = [
                "tools",
                "benchmark",
                "report",
                "build",
                str(missing),
                str(missing),
                str(missing),
                str(missing),
                str(missing),
                str(missing),
                str(spec),
                "--out",
                str(root / "report"),
            ]
            reprice_args = [
                "tools",
                "benchmark",
                "report",
                "reprice",
                str(missing),
                str(root / "price.yaml"),
                "--out",
                str(root / "repriced"),
            ]
            invalid_build = CliRunner().invoke(app, build_args)
            invalid_reprice = CliRunner().invoke(app, reprice_args)
            self.assertNotEqual(invalid_build.exit_code, 0)
            self.assertNotEqual(invalid_reprice.exit_code, 0)
            self.assertTrue(invalid_build.output or invalid_reprice.output)

    def test_reprice_changes_pareto_and_recommendation_winners(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["absolute"]["b"]["s1"] = json.loads(json.dumps(human["absolute"]["a"]["s1"]))
            human["mqm"]["b"]["s1"] = json.loads(json.dumps(human["mqm"]["a"]["s1"]))
            human["pairwise"]["s1"]["candidates"]["b"] = json.loads(
                json.dumps(human["pairwise"]["s1"]["candidates"]["a"])
            )
            integration = _write_integration(root)
            self._build(root, root / "report", human=human, integration=integration)
            before = json.loads((root / "report" / "summary.json").read_text())
            changed = _cost()
            changed["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            changed["candidate_costs"]["a"]["api_cost"] = "9"
            changed["candidate_costs"]["b"]["api_cost"] = "1"
            changed["effective_costs"]["a"]["50"]["s1"]["value"] = "9"
            changed["effective_costs"]["b"]["50"]["s1"]["value"] = "1"
            changed["effective_costs"]["a"]["100"]["s1"]["value"] = "9"
            changed["effective_costs"]["b"]["100"]["s1"]["value"] = "1"
            with (
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=changed),
            ):
                reprice_report(root / "report", root / "price.yaml", root / "repriced1")
                reprice_report(root / "report", root / "price.yaml", root / "repriced2")
            after = json.loads((root / "repriced1" / "summary.json").read_text())
            self.assertEqual(before["pareto"]["s1"]["api"], ["a@s1"])
            self.assertEqual(before["recommendations"]["s1"]["cheapest"]["entity_id"], "a@s1")
            self.assertEqual(after["pareto"]["s1"]["api"], ["b@s1"])
            self.assertEqual(after["recommendations"]["s1"]["cheapest"]["entity_id"], "b@s1")
            self.assertEqual(
                (root / "repriced1" / "summary.json").read_bytes(),
                (root / "repriced2" / "summary.json").read_bytes(),
            )

    def test_fresh_output_directories_are_byte_identical(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._build(root, root / "report1")
            self._build(root, root / "report2")
            names = (
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
            self.assertEqual({path.name for path in (root / "report1").iterdir()}, set(names))
            self.assertEqual(
                {name: (root / "report1" / name).read_bytes() for name in names},
                {name: (root / "report2" / name).read_bytes() for name in names},
            )

    def test_noncanonical_integration_bytes_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            value = json.loads(integration.read_text())
            integration.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            complete = json.loads((root / "integration_complete.json").read_text())
            complete["integration_sha256"] = hashlib.sha256(integration.read_bytes()).hexdigest()
            (root / "integration_complete.json").write_bytes(_canonical_json(complete))
            with self.assertRaises(ReportError):
                self._build(root, root / "report", integration=integration)

    def test_missing_integration_completion_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            (root / "integration_complete.json").unlink()
            with self.assertRaises(ReportError):
                self._build(root, root / "report", integration=integration)

    def test_mismatched_integration_sha_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            complete_path = root / "integration_complete.json"
            complete = json.loads(complete_path.read_text())
            complete["integration_sha256"] = "0" * 64
            complete_path.write_bytes(_canonical_json(complete))
            with self.assertRaises(ReportError):
                self._build(root, root / "report", integration=integration)

    def test_tampered_candidate_result_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            result_path = root / "candidates" / "a" / "result.json"
            result = json.loads(result_path.read_text())
            result["passed"] = False
            result_path.write_bytes(_canonical_json(result))
            with self.assertRaises(ReportError):
                self._build(root, root / "report", integration=integration)

    def test_mismatched_candidate_result_hash_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            complete_path = root / "integration_complete.json"
            complete = json.loads(complete_path.read_text())
            complete["candidates"]["a"]["result_sha256"] = "0" * 64
            complete_path.write_bytes(_canonical_json(complete))
            with self.assertRaises(ReportError):
                self._build(root, root / "report", integration=integration)

    def test_missing_request_field_and_tampered_output_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            request_path = root / "integration_request.json"
            request = json.loads(request_path.read_text())
            request.pop("book_id")
            request_path.write_bytes(_canonical_json(request))
            with self.assertRaises(ReportError):
                self._build(root, root / "missing-request", integration=integration)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            output = root / "candidates" / "a" / "mono.epub"
            output.write_bytes(b"tampered")
            with self.assertRaises(ReportError):
                self._build(root, root / "tampered-output", integration=integration)

    def test_per_book_sample_nine_is_withheld(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["absolute"]["a"]["s1"]["by_book"]["book"]["composite"]["n_units"] = 9
            result = self._build(root, root / "report", human=human)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertFalse(
                next(row for row in summary["candidates"] if row["entity_id"] == "a@s1")[
                    "gate_pass"
                ]
            )
            self.assertIn("book:insufficient_book_sample", summary["withheld_reasons"]["a@s1"])
            self.assertEqual(result["status"], "provisional")

    def test_candidate_capability_and_control_negatives_are_rejected(self) -> None:
        cases = {
            "fake": lambda request: request["candidates"]["a"].update(provider="fake"),
            "control": lambda request: request["candidates"]["a"].update(editor_model=None),
            "no_off": lambda request: request["candidates"]["a"].update(
                primary_model="qwen3.8-max"
            ),
            "capability": lambda request: request["candidates"]["a"].update(
                primary_model="unknown-model:off"
            ),
            "temperature": lambda request: request["candidates"]["a"].update(temperature=0.2),
            "seed": lambda request: request["candidates"]["a"].update(seed=7),
            "casefold": lambda request: request.update(
                candidate_ids=["a", "A"],
                candidates={
                    **request["candidates"],
                    "A": request["candidates"]["b"],
                },
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory)
                integration = _write_integration(root)
                request_path = root / "integration_request.json"
                request = json.loads(request_path.read_text())
                mutate(request)
                request_path.write_bytes(_canonical_json(request))
                with self.assertRaises(ReportError):
                    self._build(root, root / "report", integration=integration)

    def test_per_book_sample_ten_passes_threshold(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            result = self._build(root, root / "report", integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertTrue(
                next(row for row in summary["candidates"] if row["entity_id"] == "a@s1")[
                    "gate_pass"
                ]
            )
            self.assertEqual(result["status"], "final")

    def test_pending_adjudication_withholds_recommendations(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            result = self._build(
                root, root / "report", human=_human(pending=1), integration=integration
            )
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(result["status"], "provisional")
            self.assertEqual(summary["recommendations"], {})

    def test_reliability_none_requires_recalibration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["reliability"]["fidelity"] = None
            result = self._build(root, root / "report", human=human)
            self.assertEqual(result["status"], "needs_recalibration")

    def test_reliability_below_threshold_requires_recalibration(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["reliability"]["fidelity"] = 0.66
            result = self._build(root, root / "report", human=human)
            self.assertEqual(result["status"], "needs_recalibration")

    def test_reliability_exact_threshold_allows_final(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["reliability"] = dict.fromkeys(RELIABILITY, 0.67)
            integration = _write_integration(root)
            result = self._build(root, root / "report", human=human, integration=integration)
            self.assertEqual(result["status"], "final")

    def test_individual_quality_and_system_boundaries_withhold_entity(self) -> None:
        cases = ("reasoning", "model", "resume", "completion", "critical", "major")
        for case in cases:
            with self.subTest(case=case), TemporaryDirectory() as directory:
                root = Path(directory)
                human = _human()
                cost = _cost()
                integration = _write_integration(root)
                if case == "reasoning":
                    cost["system_metrics"]["a"]["reasoning_tokens"] = 1
                elif case == "model":
                    cost["system_metrics"]["a"]["resolved_model_mismatch"] = 1
                elif case == "resume":
                    cost["system_metrics"]["a"]["resume_duplicate_operations"] = 1
                elif case == "completion":
                    cost["system_metrics"]["a"]["completion_rate"] = "0.99"
                elif case == "critical":
                    human["mqm"]["a"]["s1"]["macro"]["agreed"]["severity"]["critical"]["count"] = 1
                else:
                    human["mqm"]["a"]["s1"]["macro"]["major_rate_upper95"] = 1.01
                self._build(root, root / "report", human=human, cost=cost, integration=integration)
                summary = json.loads((root / "report" / "summary.json").read_text())
                self.assertFalse(
                    next(row for row in summary["candidates"] if row["entity_id"] == "a@s1")[
                        "gate_pass"
                    ]
                )
                self.assertNotEqual(
                    summary["recommendations"].get("s1", {}).get("cheapest", {}).get("entity_id"),
                    "a@s1",
                )

    def test_exact_major_boundary_allows_entity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            human["mqm"]["a"]["s1"]["macro"]["major_rate_upper95"] = 1.0
            integration = _write_integration(root)
            result = self._build(root, root / "report", human=human, integration=integration)
            self.assertEqual(result["status"], "final")

    def test_unknown_price_candidate_is_quality_visible_but_excluded(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            integration = _write_integration(root)
            self._build(root, root / "report", cost=_cost(unknown="a"), integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertIn("a@s1", {row["entity_id"] for row in summary["candidates"]})
            self.assertIn("unknown_cost", summary["withheld_reasons"]["a@s1"])
            self.assertNotIn("a@s1", summary["pareto"]["s1"]["api"])
            self.assertNotEqual(summary["recommendations"]["s1"]["cheapest"]["entity_id"], "a@s1")

    def test_conservative_endpoints_drive_quality_ranking(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            human = _human()
            integration = _write_integration(root)
            human["absolute"]["a"]["s1"]["macro"]["dimensions"]["fidelity"]["raw_mean"] = 4.9
            human["absolute"]["a"]["s1"]["macro"]["dimensions"]["fidelity"]["lower95"] = 4.3
            human["absolute"]["b"]["s1"]["macro"]["dimensions"]["fidelity"]["raw_mean"] = 4.5
            human["absolute"]["b"]["s1"]["macro"]["dimensions"]["fidelity"]["lower95"] = 4.6
            self._build(root, root / "report", human=human, integration=integration)
            summary = json.loads((root / "report" / "summary.json").read_text())
            self.assertEqual(
                summary["recommendations"]["s1"]["highest_quality"]["entity_id"], "b@s1"
            )

    def test_csv_escapes_delimited_reason_and_preserves_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            reason = 'bad,"reason' + chr(10) + "line"
            human = _human(insufficient=[{"scope": "candidate=a/surface=s1", "reason": reason}])
            result = self._build(root, root / "report", human=human)
            report_dir = Path(result["out_dir"])
            with (report_dir / "failures.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ["entity_id", "scope", "reason", "gate"])
            self.assertTrue(
                any(
                    row[0] == "a@s1" and row[2] == f"candidate=a/surface=s1:{reason}"
                    for row in rows[1:]
                )
            )

    def test_reprice_changes_price_outputs_and_repeats_deterministically(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report"
            self._build(root, report)
            changed = _cost()
            changed["input_hashes"]["price_snapshot_sha256"] = "1" * 64
            changed["candidate_costs"]["a"]["api_cost"] = "9"
            changed["candidate_costs"]["a"]["api_cost_lower_bound"] = "9"
            changed["candidate_costs"]["a"]["by_operation"] = {
                "translate": {
                    "api_cost_lower_bound": "9",
                    "cost_complete": True,
                    "unknown_count": 0,
                }
            }
            changed["physical_spend"]["api_cost_lower_bound"] = "11"
            changed["million_word_estimate"]["a"]["value"] = "2000"
            changed["million_word_estimate"]["a"]["lower95"] = "1900"
            changed["million_word_estimate"]["a"]["upper95"] = "2100"
            changed["million_word_estimate"]["a"]["by_book"]["book"]["value"] = "2000"
            changed["effective_costs"]["a"]["50"]["s1"]["value"] = "9"
            changed["effective_costs"]["a"]["100"]["s1"]["value"] = "10"
            with (
                patch("trans_novel.benchmark.pricing.load_price_snapshot", return_value=object()),
                patch("trans_novel.benchmark.report_cost._price_hash", return_value="1" * 64),
                patch("trans_novel.benchmark.report.reprice_cost_system", return_value=changed),
            ):
                reprice_report(report, root / "price.yaml", root / "repriced1")
                reprice_report(report, root / "price.yaml", root / "repriced2")
            self.assertNotEqual(
                (report / "summary.json").read_bytes(),
                (root / "repriced1" / "summary.json").read_bytes(),
            )
            self.assertNotEqual(
                (root / "repriced1" / "cost_by_operation.csv").read_bytes(),
                (report / "cost_by_operation.csv").read_bytes(),
            )
            self.assertEqual(
                (root / "repriced1" / "summary.json").read_bytes(),
                (root / "repriced2" / "summary.json").read_bytes(),
            )
            self.assertEqual(
                (root / "repriced1" / "report.html").read_bytes(),
                (root / "repriced2" / "report.html").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
