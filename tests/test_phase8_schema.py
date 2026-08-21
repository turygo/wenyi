from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import ValidationError

from trans_novel.benchmark.report_schema import PublicationGates, ReportSpec, load_report_spec

_HASHES = {
    "corpus_sha256": "a" * 64,
    "run_hash": "b" * 64,
    "preparation_sha256": "c" * 64,
    "pack_sha256": "d" * 64,
    "evaluation_sha256": "e" * 64,
    "price_snapshot_sha256": "f" * 64,
}


def _base_spec(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "benchmark_id": "phase8-main",
        **_HASHES,
        "bootstrap_seed": 17,
    }
    value.update(overrides)
    return value


class ReportSchemaTests(unittest.TestCase):
    def test_defaults_have_contract_values_and_stable_order(self) -> None:
        gates = PublicationGates()
        self.assertEqual(gates.critical_max, 0)
        self.assertEqual(gates.completion_min, Decimal("1.0"))
        self.assertEqual(gates.fidelity_mean_min, Decimal("4.3"))
        self.assertEqual(gates.krippendorff_alpha_min, Decimal("0.67"))

        spec = ReportSpec.model_validate(_base_spec())
        self.assertEqual(spec.bootstrap_replicates, 2000)
        self.assertEqual(spec.editor_hourly_rates, [Decimal("50"), Decimal("100"), Decimal("200")])
        self.assertEqual(
            list(ReportSpec.model_fields),
            [
                "schema_version",
                "benchmark_id",
                "corpus_sha256",
                "run_hash",
                "preparation_sha256",
                "pack_sha256",
                "evaluation_sha256",
                "price_snapshot_sha256",
                "bootstrap_seed",
                "bootstrap_replicates",
                "editor_hourly_rates",
                "gates",
            ],
        )

        first = ReportSpec.model_validate(_base_spec())
        second = ReportSpec.model_validate(_base_spec())
        self.assertIsNot(first.editor_hourly_rates, second.editor_hourly_rates)

    def test_external_models_are_strict_and_forbid_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PublicationGates.model_validate({"critical_max": "0"})
        with self.assertRaises(ValidationError):
            PublicationGates.model_validate({"unexpected": 1})
        with self.assertRaises(ValidationError):
            ReportSpec.model_validate({**_base_spec(), "unexpected": 1})
        with self.assertRaises(ValidationError):
            ReportSpec.model_validate({**_base_spec(), "editor_hourly_rates": (50, 100)})

    def test_hashes_benchmark_id_and_integer_boundaries(self) -> None:
        spec = ReportSpec.model_validate(_base_spec(benchmark_id="  id.with-space  "))
        self.assertEqual(spec.benchmark_id, "id.with-space")
        for field in _HASHES:
            with self.subTest(field=field):
                for value in ("A" * 64, "a" * 63, "a" * 65, "g" * 64):
                    with self.assertRaises(ValidationError):
                        ReportSpec.model_validate(_base_spec(**{field: value}))
        for value in (True, 1.5, "17"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReportSpec.model_validate(_base_spec(bootstrap_seed=value))
        for value in (999, 10001, True, 2000.0, "2000"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReportSpec.model_validate(_base_spec(bootstrap_replicates=value))

    def test_decimal_input_boundary_is_shared_by_direct_and_yaml_models(self) -> None:
        direct = ReportSpec.model_validate(
            _base_spec(
                editor_hourly_rates=["200.0", 50, Decimal("100")],
                gates={"completion_min": "0.95", "fidelity_mean_min": 4},
            )
        )
        self.assertEqual(
            direct.editor_hourly_rates, [Decimal("50"), Decimal("100"), Decimal("200.0")]
        )
        self.assertEqual(direct.gates.completion_min, Decimal("0.95"))
        self.assertEqual(direct.gates.fidelity_mean_min, Decimal("4"))

        for value in (1.25, True, "", " 1.25", "1.25 ", "NaN", "Infinity"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ReportSpec.model_validate(_base_spec(editor_hourly_rates=[value]))
            with self.subTest(gate_value=value), self.assertRaises(ValidationError):
                ReportSpec.model_validate(_base_spec(gates={"completion_min": value}))

    def test_rates_are_positive_unique_sorted_and_gates_respect_ranges(self) -> None:
        spec = ReportSpec.model_validate(_base_spec(editor_hourly_rates=[200, "50", 100]))
        self.assertEqual(spec.editor_hourly_rates, [Decimal("50"), Decimal("100"), Decimal("200")])
        for rates in ([], [0], [-1], [1, "1.0"]):
            with self.subTest(rates=rates), self.assertRaises(ValidationError):
                ReportSpec.model_validate(_base_spec(editor_hourly_rates=rates))

        decimal_limits = {
            "completion_min": (-1, 1.1),
            "major_per_10k_upper95_max": (-1, 1.1),
            "per_book_major_per_10k_max": (-1,),
            "fidelity_mean_min": (0.9, 5.1),
            "naturalness_mean_min": (0.9, 5.1),
            "polish_harm_rate_upper95_max": (-1, 1.1),
            "krippendorff_alpha_min": (-1, 1.1),
        }
        for field, values in decimal_limits.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    PublicationGates.model_validate({field: value})
        for field in (
            "critical_max",
            "structure_errors_max",
            "protocol_errors_max",
            "required_node_failures_max",
            "resume_duplicate_operations_max",
            "reasoning_tokens_max",
            "polish_major_semantic_harm_max",
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                PublicationGates.model_validate({field: -1})

    def test_yaml_loader_is_utf8_path_only_and_rejects_non_objects(self) -> None:
        yaml_spec = """\
 schema_version: 1
 benchmark_id: phase8-yaml
 corpus_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
 run_hash: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
 preparation_sha256: cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
 pack_sha256: dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
 evaluation_sha256: eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
 price_snapshot_sha256: ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
 bootstrap_seed: 17
 editor_hourly_rates: [200, 50, 100]
 gates:
   completion_min: '0.95'
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.yaml"
            path.write_text(yaml_spec, encoding="utf-8")
            loaded = load_report_spec(path)
        self.assertEqual(
            loaded.editor_hourly_rates, [Decimal("50"), Decimal("100"), Decimal("200")]
        )
        self.assertEqual(loaded.gates.completion_min, Decimal("0.95"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_report_spec(path)
            path.write_text("- not-an-object\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_report_spec(path)
            path.write_text(yaml_spec.replace("[200, 50, 100]", "[1.25]"), encoding="utf-8")
            with self.assertRaises(ValidationError):
                load_report_spec(path)
            path.write_text("!!python/object:object {}\n", encoding="utf-8")
            with self.assertRaises(yaml.constructor.ConstructorError):
                load_report_spec(path)
            with self.assertRaises(TypeError):
                load_report_spec(str(path))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
