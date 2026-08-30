"""Offline benchmark schema and minimal/polish arm contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from trans_novel.benchmark.runner import RUN_SCHEMA_VERSION, _target_hash, load_candidate_spec
from trans_novel.benchmark.schema import CandidateSpec
from trans_novel.pipeline.runstore import clone_closed_runstore


def _spec(*, candidates=None):
    return {
        "schema_version": 3,
        "benchmark_id": "offline",
        "temperature": 0.1,
        "seed": None,
        "replicates": 1,
        "candidates": candidates
        or [
            {
                "candidate_id": "same-minimal",
                "translator_model": "fake/same:high",
                "analyst_model": "fake/analysis:off",
                "editor_model": "fake/same:high",
                "fast_model": "fake/fast:off",
                "pipeline_variant": "minimal",
            },
            {
                "candidate_id": "same-polish",
                "translator_model": "fake/same:high",
                "analyst_model": "fake/analysis:off",
                "editor_model": "fake/same:high",
                "fast_model": "fake/fast:off",
                "pipeline_variant": "polish",
            },
        ],
    }


class TestBenchmarkSchema(unittest.TestCase):
    def test_candidate_spec_v3_allows_same_model_roles_across_variants(self):
        parsed = CandidateSpec.model_validate(_spec())
        self.assertEqual(parsed.schema_version, 3)
        self.assertEqual(
            {candidate.pipeline_variant for candidate in parsed.candidates}, {"minimal", "polish"}
        )

    def test_loader_rejects_missing_pipeline_variant(self):
        value = _spec(candidates=[{"candidate_id": "x"}])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "candidates.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_candidate_spec(path)

    def test_run_artifact_schema_is_three(self):
        self.assertEqual(RUN_SCHEMA_VERSION, 3)

    def test_initial_hash_is_canonical_and_clone_preserves_bytes(self):
        rows = [
            {"chapter_index": 0, "segment_index": 0, "target": "甲"},
            {"chapter_index": 0, "segment_index": 1, "target": "乙"},
        ]
        self.assertEqual(_target_hash(rows), _target_hash([dict(row) for row in rows]))
        changed = [*rows[:-1], {**rows[-1], "target": "丙"}]
        self.assertNotEqual(_target_hash(rows), _target_hash(changed))
        with tempfile.TemporaryDirectory() as d:
            source, destination = Path(d) / "source", Path(d) / "clone"
            source.mkdir()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            clone_closed_runstore(str(source), str(destination))
            self.assertEqual(
                (destination / "manifest.json").read_bytes(),
                (source / "manifest.json").read_bytes(),
            )
