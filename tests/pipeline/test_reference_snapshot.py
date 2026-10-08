from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.fixtures.migration_admission import admit_fixture, materialize_fixture
from tests.fixtures.migration_artifacts import sha256_file
from tests.fixtures.migration_bundle import (
    RecordingError,
    _envelope,
    _support_hashes,
    _validate_stage,
    assert_failed_recording_preserves_bundle,
    qualify_candidate_against_fixture,
    record_qualified_candidate,
)
from tests.fixtures.migration_epub import FIXTURE_ID
from tests.fixtures.migration_errors import ObservationError
from tests.fixtures.migration_observation import (
    assert_semantic_match,
    observe_run,
)
from tests.fixtures.migration_provenance import verify_baseline_inputs
from tests.fixtures.migration_reference import _assert_source_first_preservation, build_candidate

_TEST_IDENTIFIER = "tests.pipeline.test_reference_snapshot.TestReferenceSnapshot.test_full_flow_reference_candidate"
_RECORD_ENV = "WENYI_R0_RECORD"


class TestReferenceSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.temporary_directory = tempfile.TemporaryDirectory(prefix="wenyi-r0-reference-")
        cls.addClassCleanup(cls.temporary_directory.cleanup)
        cls.workspace = Path(cls.temporary_directory.name) / "candidate"
        cls.candidate = build_candidate(
            cls.workspace,
            cls.repo_root,
            test_identifier=_TEST_IDENTIFIER,
        )
        cls.primary_projection = observe_run(
            cls.workspace,
            cls.candidate.run_relative,
            cls.candidate.source_relative,
            list(cls.candidate.outputs),
        )

    def _copy_workspace(self, label: str) -> Path:
        destination = Path(self.temporary_directory.name) / f"corrupt-{label}"
        shutil.copytree(self.workspace, destination)
        return destination

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _observe(self, workspace: Path) -> dict[str, Any]:
        return observe_run(
            workspace,
            self.candidate.run_relative,
            self.candidate.source_relative,
            list(self.candidate.outputs),
            source_origin=self.candidate.projection["run"]["manifest"]["source_path"],
        )

    def test_full_flow_reference_candidate(self) -> None:
        candidate = self.candidate
        self.assertEqual(
            {output["role"] for output in candidate.projection["outputs"]},
            {"mono", "bilingual_target_first", "bilingual_source_first"},
        )
        self.assertEqual(len(candidate.projection["source"]["spine"]), 2)
        self.assertTrue(candidate.projection["source"]["nav"])
        self.assertTrue(candidate.projection["source"]["ncx"])
        self.assertEqual(candidate.comparison["model_calls"], 0)
        self.assertTrue(candidate.comparison["original_unchanged"])
        self.assertTrue(candidate.comparison["raw_usage_bytes_unchanged"])
        self.assertGreater(len(candidate.protocol["calls"]), 0)
        self.assertTrue(
            all(call["token_usage"]["total_tokens"] > 0 for call in candidate.protocol["calls"])
        )
        self.assertEqual(self.primary_projection["run"]["usage"]["schema_version"], 2)

        record_mode = os.environ.get(_RECORD_ENV)
        if record_mode not in {None, "fixtures/migration"}:
            self.fail(f"unsupported {_RECORD_ENV} value: {record_mode!r}")
        reference = self.repo_root / "fixtures/migration/state" / FIXTURE_ID / "reference.json"
        if reference.is_file():
            qualify_candidate_against_fixture(
                candidate,
                self.repo_root / "fixtures/migration",
            )
        elif record_mode == "fixtures/migration":
            record_qualified_candidate(self.repo_root, candidate)
            qualify_candidate_against_fixture(
                candidate,
                self.repo_root / "fixtures/migration",
            )
        if reference.is_file() or record_mode == "fixtures/migration":
            with patch.dict(os.environ, {_RECORD_ENV: "fixtures/migration"}):
                assert_failed_recording_preserves_bundle(self.repo_root, candidate)

    def test_valid_target_corruption_is_rejected(self) -> None:
        workspace = self._copy_workspace("target")
        run_dir = workspace / self.candidate.run_relative
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chapter_index = manifest["chapters"][0]["index"]
        chapter_path = run_dir / "chapters_v2" / f"ch{chapter_index}.json"
        chapter = json.loads(chapter_path.read_text(encoding="utf-8"))
        chapter["segments"][0]["target"] += "!"
        self._write_json(chapter_path, chapter)
        with self.assertRaises(ObservationError):
            self._observe(workspace)

    def test_accounting_corruption_changes_semantics(self) -> None:
        workspace = self._copy_workspace("accounting")
        usage_path = workspace / self.candidate.run_relative / "usage.json"
        usage = json.loads(usage_path.read_text(encoding="utf-8"))

        def change_total(value: Any) -> bool:
            if isinstance(value, dict):
                if type(value.get("total_tokens")) is int:
                    value["total_tokens"] += 1
                    return True
                return any(change_total(item) for item in value.values())
            if isinstance(value, list):
                return any(change_total(item) for item in value)
            return False

        self.assertTrue(change_total(usage), "persisted usage has no token accounting total")
        self._write_json(usage_path, usage)
        corrupted = self._observe(workspace)
        with self.assertRaises(ObservationError):
            assert_semantic_match(corrupted, self.primary_projection)

    def test_source_corruption_is_rejected(self) -> None:
        workspace = self._copy_workspace("source")
        source = workspace / self.candidate.source_relative
        with source.open("ab") as stream:
            stream.write(b"\0")
        with self.assertRaises(ObservationError):
            self._observe(workspace)

    def test_incomplete_lifecycle_is_rejected(self) -> None:
        workspace = self._copy_workspace("lifecycle")
        manifest_path = workspace / self.candidate.run_relative / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["nodes"]["assemble"]["status"] = "running"
        self._write_json(manifest_path, manifest)
        with self.assertRaises(ObservationError):
            self._observe(workspace)

    def test_pending_durable_journals_are_rejected(self) -> None:
        for name in ("journal.json", "usage.wal.json", "note_migration.json"):
            with self.subTest(journal=name):
                workspace = self._copy_workspace(name)
                self._write_json(workspace / self.candidate.run_relative / name, {})
                with self.assertRaises(ObservationError):
                    self._observe(workspace)

    def test_changed_source_origin_is_rejected_without_opening_it(self) -> None:
        workspace = self._copy_workspace("origin")
        path = workspace / self.candidate.run_relative / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["source_path"] = "/unrelated/source/book.epub"
        self._write_json(path, manifest)
        with self.assertRaises(ObservationError):
            self._observe(workspace)

    def test_source_first_content_fingerprint_changes_are_rejected(self) -> None:
        workspace = self._copy_workspace("fingerprints") / "assembly/source-first"
        run_relative = f"state/{self.candidate.run_slug}"
        path = workspace / run_relative / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["nodes"]["translate:0"]["input_fingerprint"] = "0" * 64
        self._write_json(path, manifest)
        copied = observe_run(
            workspace,
            run_relative,
            self.candidate.source_relative,
            [self.candidate.source_first_output],
            source_origin=self.candidate.projection["run"]["manifest"]["source_path"],
        )
        with self.assertRaises(RecordingError):
            _assert_source_first_preservation(self.primary_projection["run"], copied["run"])

    def test_independent_second_run_matches_the_frozen_semantic_contract(self) -> None:
        candidate = build_candidate(
            Path(self.temporary_directory.name) / "replay",
            self.repo_root,
            test_identifier=_TEST_IDENTIFIER,
        )
        assert_semantic_match(candidate.projection, self.candidate.projection)
        self.assertEqual(candidate.protocol, self.candidate.protocol)

    def test_corrupt_candidate_fails_staged_recording_without_publication(self) -> None:
        workspace = self._copy_workspace("recording")
        manifest = json.loads(
            (workspace / self.candidate.run_relative / "manifest.json").read_text(encoding="utf-8")
        )
        index = manifest["chapters"][0]["index"]
        path = workspace / self.candidate.run_relative / "chapters_v2" / f"ch{index}.json"
        chapter = json.loads(path.read_text(encoding="utf-8"))
        chapter["segments"][0]["target"] += "!"
        self._write_json(path, chapter)
        candidate = replace(self.candidate, workspace=workspace)
        fixture_root = self.repo_root / "fixtures/migration"
        before = {path: sha256_file(path) for path in fixture_root.rglob("*") if path.is_file()}
        with tempfile.TemporaryDirectory(prefix="wenyi-r0-rejected-stage-") as temporary:
            stage = Path(temporary)
            envelope = _envelope(
                candidate,
                self.repo_root,
                stage,
                verify_baseline_inputs(self.repo_root),
                _support_hashes(self.repo_root),
            )
            self._write_json(stage / f"state/{FIXTURE_ID}/reference.json", envelope)
            with self.assertRaises((ObservationError, RecordingError)):
                _validate_stage(stage, envelope, self.repo_root, candidate)
        self.assertEqual(
            {path: sha256_file(path) for path in fixture_root.rglob("*") if path.is_file()}, before
        )

    def test_missing_publication_evidence_is_rejected(self) -> None:
        workspace = self._copy_workspace("publication")
        output = self.candidate.outputs[0]
        epub = workspace / output["file"]
        self.assertTrue(epub.is_file())
        proof_path = workspace / output["proof"]
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        proof["published_outputs"].pop(epub.name)
        self._write_json(proof_path, proof)
        with self.assertRaises(ObservationError):
            self._observe(workspace)

    def test_fixture_admission_and_materialization(self) -> None:
        fixture_root = self.repo_root / "fixtures/migration"
        reference = fixture_root / "state" / FIXTURE_ID / "reference.json"
        if not reference.is_file():
            self.skipTest("the first qualified fixture has not been recorded yet")
        envelope = admit_fixture(fixture_root)
        with tempfile.TemporaryDirectory(prefix="wenyi-r0-materialized-") as temporary:
            workspace = Path(temporary) / "fixture"
            materialize_fixture(fixture_root, workspace)
            for record in envelope["files"]:
                if record["workspace_path"] is not None:
                    copied = workspace / record["workspace_path"]
                    self.assertEqual(copied.stat().st_size, record["size_bytes"])
                    self.assertEqual(sha256_file(copied), record["sha256"])
