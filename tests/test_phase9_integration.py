"""Phase 9 integration contracts exercised with local FakeClient inputs only."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import yaml
from typer.testing import CliRunner

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import write_sample_epub
from tests.test_phase4_corpus import _write_minimal_artifact
from trans_novel.benchmark.corpus import (
    canonical_json,
    passage_id,
    segment_id,
    sha256_bytes,
)
from trans_novel.benchmark.integration import (
    BenchmarkInterruption,
    IntegrationError,
    IntegrationRunner,
    IntegrationSpec,
    _telemetry_evidence,
)
from trans_novel.benchmark.schema import CandidateSpec
from trans_novel.cli import app
from trans_novel.config import Config
from trans_novel.llm import FakeClient
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.model_profiles import parse_model_selection, parse_provider_model
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.runner import RequiredNodeFailed
from trans_novel.pipeline.runstore import RunStore


class _InstrumentedFakeClient(FakeClient):
    def __init__(
        self, *, handler, models: tuple[str, str, str, str], provider: str = "fake"
    ) -> None:
        super().__init__(handler=handler)
        self.models = models
        self.provider = provider
        self.telemetry_sink = None
        self._attempts = 0

    def set_telemetry_sink(self, sink) -> None:
        self.telemetry_sink = sink

    def complete(self, messages, *, json_mode=False, max_tokens=None, stage=None, agent, operation):
        started = (
            datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        response = super().complete(
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
            stage=stage,
            agent=agent,
            operation=operation,
        )
        self._attempts += 1
        model_ref = (
            self.models[1]
            if agent == "analyst"
            else self.models[2]
            if agent == "editor"
            else self.models[3]
            if agent in {"preparer", "light-translator"}
            else self.models[0]
        )
        provider, model = parse_provider_model(model_ref)
        selection = parse_model_selection(model)
        request_hash = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.telemetry_sink.record(
            CallAttemptTelemetry(
                schema_version=1,
                logical_call_id=f"{self._attempts:032x}",
                attempt_index=1,
                started_at=started,
                elapsed_ms=0,
                stage=stage,
                agent=agent,
                operation=operation,
                provider=provider,
                requested_model=selection.model,
                resolved_model=selection.model,
                reasoning_enabled=False,
                reasoning_effort=None,
                temperature=0.1,
                seed=None,
                json_mode=json_mode,
                max_tokens=max_tokens,
                status="success",
                retry_class=None,
                http_status=None,
                finish_reason=None,
                response_id=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cache_hit_tokens=0,
                cache_miss_tokens=0,
                reasoning_tokens=0,
                billed_usage_unknown=False,
                request_sha256=request_hash,
                response_sha256=hashlib.sha256(response.encode()).hexdigest(),
            )
        )
        return response


class _StopAfterBatch:
    def __init__(self, count: int = 1):
        self.count = count
        self.commits: list[tuple[int, int, int]] = []

    def after_batch_committed(self, chapter_index: int, start: int, count: int) -> None:
        self.commits.append((chapter_index, start, count))
        if len(self.commits) >= self.count:
            raise BenchmarkInterruption("test boundary")


def _config(state: Path) -> Config:
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
    config.source_lang = "en"
    config.target_lang = "zh"
    config.state_dir = str(state)
    return config


def _source(path: Path) -> None:
    path.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", encoding="utf-8")


def _spec(**updates):
    value = {
        "schema_version": 1,
        "benchmark_id": "phase9",
        "corpus_sha256": "a" * 64,
        "candidate_spec_sha256": "b" * 64,
        "book_id": "hidden-book",
        "candidate_ids": ["candidate-a", "candidate-b"],
        "interrupt_after_committed_batches": 1,
        "output_mono": True,
        "output_bilingual": True,
        "bilingual_order": "target_first",
        "source_language": "en",
        "target_language": "zh",
    }
    value.update(updates)
    return value


class TestPhase9Integration(unittest.TestCase):
    def test_integration_spec_is_strict_and_rejects_controls_or_duplicates(self):
        spec = IntegrationSpec.model_validate(_spec())
        self.assertEqual(spec.schema_version, 1)
        with self.assertRaises(ValueError):
            IntegrationSpec.model_validate(_spec(candidate_ids=["candidate-a", "candidate-a"]))
        with self.assertRaises(ValueError):
            IntegrationSpec.model_validate(_spec(unexpected=True))

    def test_default_application_behavior_has_no_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            client = FakeClient(handler=routing_handler)
            result = Application(_config(root / "state"), client=client).run_all(
                str(source), out_format="txt"
            )
            self.assertIsNotNone(result["store"])
            self.assertFalse(
                any(
                    call.get("operation") == "integration.canary.translate" for call in client.calls
                )
            )

    def test_hook_interrupts_after_persisted_full_body_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            hook = _StopAfterBatch()
            with self.assertRaises(BenchmarkInterruption):
                Application(
                    _config(root / "state"),
                    client=FakeClient(handler=routing_handler),
                    batch_commit_hook=hook,
                ).run(str(source))
            self.assertEqual(len(hook.commits), 1)
            event_files = list((root / "state").rglob("events.jsonl"))
            self.assertEqual(len(event_files), 1)
            events = [
                json.loads(line) for line in event_files[0].read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event["event"] == "batch_translated" for event in events))

    def test_required_event_failure_prevents_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            hook = _StopAfterBatch()
            with (
                mock.patch.object(
                    RunStore, "log_event_required", side_effect=OSError("append failed")
                ),
                self.assertRaises(RequiredNodeFailed),
            ):
                Application(
                    _config(root / "state"),
                    client=FakeClient(handler=routing_handler),
                    batch_commit_hook=hook,
                ).run(str(source))
            self.assertEqual(hook.commits, [])

    def test_new_application_resumes_without_retranslating_committed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            with self.assertRaises(BenchmarkInterruption):
                Application(
                    _config(root / "state"),
                    client=FakeClient(handler=routing_handler),
                    batch_commit_hook=_StopAfterBatch(),
                ).run(str(source))
            resumed = FakeClient(handler=routing_handler)
            result = Application(_config(root / "state"), client=resumed).run(str(source))
            self.assertIsNotNone(result)
            events_path = next((root / "state").rglob("events.jsonl"))
            events = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event.get("reason") == "already_translated" for event in events))
            self.assertEqual(
                sum(call["operation"] == "translate.batch" for call in resumed.calls), 0
            )

    def test_hook_does_not_fire_for_already_translated_quality_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            first = Application(
                _config(root / "state"), client=FakeClient(handler=routing_handler)
            ).run_all(str(source), out_format="txt")
            target_before = Path(first["store"].chapter_path(0)).read_bytes()
            hook = _StopAfterBatch()
            resumed = FakeClient(handler=routing_handler)
            second = Application(
                _config(root / "state"),
                client=resumed,
                batch_commit_hook=hook,
            ).run_all(str(source), out_format="txt")
            self.assertEqual(hook.commits, [])
            self.assertEqual(Path(second["store"].chapter_path(0)).read_bytes(), target_before)
            self.assertEqual(
                sum(call["operation"] == "translate.batch" for call in resumed.calls), 0
            )

    def test_translator_model_change_invalidates_translation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            first = Application(
                _config(root / "state"), client=FakeClient(handler=routing_handler)
            ).run(str(source))
            old_name_fingerprint = first.load_state().nodes["name_terms"].input_fingerprint
            changed = _config(root / "state")
            changed.llm.models.translator = ["fake/different-translator"]
            client = FakeClient(handler=routing_handler)
            second = Application(changed, client=client).run(str(source))
            self.assertEqual(
                second.load_state().nodes["name_terms"].input_fingerprint,
                old_name_fingerprint,
            )
            self.assertGreater(
                sum(call["operation"] == "translate.batch" for call in client.calls), 0
            )
            events_path = next((root / "state").rglob("events.jsonl"))
            events = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event.get("event") == "translate_invalidated" for event in events))

    def test_unexpected_exception_is_not_treated_as_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "book.txt"
            _source(source)
            client = FakeClient(handler=lambda *args: (_ for _ in ()).throw(RuntimeError("boom")))
            with self.assertRaises(RequiredNodeFailed):
                Application(_config(root / "state"), client=client).run(str(source))

    def test_integration_runner_refuses_mismatched_existing_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "integration_request.json"
            request.write_text('{"benchmark_id":"other"}\n', encoding="utf-8")
            runner = IntegrationRunner()
            with (
                mock.patch.object(
                    runner,
                    "_preflight",
                    return_value=(
                        IntegrationSpec.model_validate(_spec()),
                        mock.Mock(),
                        root / "book.epub",
                        "c" * 64,
                        {
                            "book_spec_sha256": "d" * 64,
                            "candidate_spec_sha256": "b" * 64,
                            "integration_spec_sha256": "e" * 64,
                            "selected": [],
                            "source_sha256": "c" * 64,
                        },
                    ),
                ),
                self.assertRaises(IntegrationError),
            ):
                runner.run(
                    root,
                    root / "book.yaml",
                    root / "candidates.yaml",
                    root / "integration.yaml",
                    root,
                )

    def test_canonical_manifest_hash_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            integration = {
                "schema_version": 1,
                "benchmark_id": "phase9",
                "corpus_sha256": "a" * 64,
                "candidates": {},
            }
            payload = (
                json.dumps(integration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            path = root / "integration.json"
            path.write_text(payload, encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_cli_registers_exact_integration_command(self):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "tools",
                "benchmark",
                "integration",
                "run",
                "missing",
                "book.yaml",
                "candidates.yaml",
                "integration.yaml",
                "--out",
                "out",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertNotIn("No such command", result.output)

    def test_validator_boundary_returns_safe_triplet_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            mono = root / "mono.epub"
            bilingual = root / "bilingual.epub"
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "schema_version": 1,
                    "structural_pass": True,
                    "source": {"structural_pass": True},
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ) as validator:
                value = validator(source, mono, bilingual)
            self.assertTrue(value["structural_pass"])
            validator.assert_called_once_with(source, mono, bilingual)

    def _runner_fixture(self, root: Path, *, interrupt: int = 1):
        source = root / "hidden.epub"
        write_sample_epub(str(source))
        candidate_spec = CandidateSpec.model_validate(
            {
                "schema_version": 3,
                "benchmark_id": "phase9",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "candidates": [
                    {
                        "candidate_id": "candidate-a-polished",
                        "translator_model": "bailian/qwen3.8-max:off",
                        "analyst_model": "bailian/qwen3.7-flash:off",
                        "editor_model": "bailian/deepseek-v4-pro:off",
                        "fast_model": "bailian/qwen3.7-flash:off",
                        "pipeline_variant": "polish",
                    },
                    {
                        "candidate_id": "candidate-b-polished",
                        "translator_model": "bailian/deepseek-v4-flash:off",
                        "analyst_model": "bailian/qwen3.7-flash:off",
                        "editor_model": "bailian/qwen3.7-plus:off",
                        "fast_model": "bailian/qwen3.7-flash:off",
                        "pipeline_variant": "polish",
                    },
                ],
            }
        )
        selected = list(candidate_spec.candidates)
        integration_spec = IntegrationSpec.model_validate(
            _spec(
                candidate_ids=["candidate-a-polished", "candidate-b-polished"],
                interrupt_after_committed_batches=interrupt,
            )
        )
        lineage = {
            "book_spec_sha256": "d" * 64,
            "candidate_spec_sha256": "b" * 64,
            "integration_spec_sha256": "e" * 64,
            "selected": selected,
            "source_sha256": sha256_bytes(source.read_bytes()),
        }
        factory_calls: list[FakeClient] = []

        def factory(**kwargs):
            roles = kwargs.get("models")
            if roles is not None:
                models = (
                    roles.translator[0],
                    roles.analyst[0],
                    roles.editor[0],
                    roles.fast[0],
                )
            else:
                if len(factory_calls) < 3:
                    models = (
                        "bailian/qwen3.8-max:off",
                        "bailian/qwen3.7-flash:off",
                        "bailian/deepseek-v4-pro:off",
                        "bailian/qwen3.7-flash:off",
                    )
                else:
                    models = (
                        "bailian/deepseek-v4-flash:off",
                        "bailian/qwen3.7-flash:off",
                        "bailian/qwen3.7-plus:off",
                        "bailian/qwen3.7-flash:off",
                    )
            client = _InstrumentedFakeClient(
                handler=routing_handler, models=models, provider="bailian"
            )
            factory_calls.append(client)
            return client

        runner = IntegrationRunner(client_factory=factory)
        runner._preflight = mock.Mock(
            return_value=(
                integration_spec,
                candidate_spec,
                source,
                lineage["source_sha256"],
                lineage,
            )
        )
        return runner, source, factory_calls

    def test_public_runner_fakeclient_workflow_restart_and_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, clients = self._runner_fixture(root)
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                first = runner.run(
                    root,
                    root / "books.yaml",
                    root / "candidates.yaml",
                    root / "integration.yaml",
                    root / "out",
                )
            self.assertFalse(first["no_op"])
            self.assertEqual(sorted(first["failed_candidates"]), [])
            self.assertTrue(first["resumed"] is False)
            self.assertEqual(len(clients), 6)
            request = json.loads(
                (root / "out" / "integration_request.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {item["pipeline_variant"] for item in request["candidates"].values()},
                {"polish"},
            )
            for cid in ("candidate-a-polished", "candidate-b-polished"):
                result = json.loads((root / "out" / "candidates" / cid / "result.json").read_text())
                self.assertTrue(result["expected_interruption_observed"])
                self.assertTrue(result["readiness_passed"])
                self.assertEqual(result["resume_duplicate_operations"], 0)
                self.assertGreaterEqual(result["remaining_batches"], 0)
                self.assertTrue(result["telemetry_sha256"])
                self.assertTrue(result["usage_sha256"])
            rerun = runner.run(
                root,
                root / "books.yaml",
                root / "candidates.yaml",
                root / "integration.yaml",
                root / "out",
            )
            self.assertTrue(rerun["no_op"])
            self.assertEqual(len(clients), 6)

    def test_terminal_candidates_are_included_when_manifests_regenerate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, clients = self._runner_fixture(root)
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            (root / "out" / "integration.json").unlink()
            (root / "out" / "integration_complete.json").unlink()
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                regenerated = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertEqual(
                set(regenerated["candidates"]),
                {"candidate-a-polished", "candidate-b-polished"},
            )
            self.assertEqual(len(clients), 6)

    def test_terminal_state_semantic_prefix_tamper_refuses_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, _clients = self._runner_fixture(root)
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            state_path = root / "out" / "integration_state.json"
            state = json.loads(state_path.read_text())
            state["candidates"]["candidate-a-polished"]["before_target_hashes"].append({})
            state_path.write_text(canonical_json(state) + "\n", encoding="utf-8")
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_canary_operation_agent_tamper_refuses_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, _clients = self._runner_fixture(root)
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            telemetry = root / "out" / "candidates" / "candidate-a-polished" / "telemetry.jsonl"
            rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
            rows[0]["operation"] = "integration.canary.wrong"
            telemetry.write_text(
                "".join(canonical_json(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_resume_timing_is_cumulative_and_partitioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, _clients = self._runner_fixture(root)
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            state = json.loads((root / "out" / "integration_state.json").read_text())
            for cid, entry in state["candidates"].items():
                timings = json.loads(
                    (root / "out" / "candidates" / cid / "result.json").read_text()
                )["phase_timings_ms"]
                self.assertEqual(timings["total"], timings["first_attempt"] + timings["resume"])
                self.assertEqual(entry["resume_wall_ms"], timings["resume"])
                self.assertEqual(sum(entry["resume_durations_ms"]), timings["resume"])

    def test_restart_after_crash_following_resumed_batches_reattributes_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source_path, _clients = self._runner_fixture(root)
            integration_module = __import__(
                "trans_novel.benchmark.integration", fromlist=["Application"]
            )
            original_run_all = integration_module.Application.run_all
            calls = {"count": 0}

            def crash_after_resume(app, *args, **kwargs):
                calls["count"] += 1
                value = original_run_all(app, *args, **kwargs)
                if calls["count"] == 2:
                    raise SystemExit("crash after resumed batches")
                return value

            with (
                mock.patch.object(integration_module.Application, "run_all", crash_after_resume),
                self.assertRaises(SystemExit),
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            prior_state = json.loads((root / "out" / "integration_state.json").read_text())
            prior_cumulative = prior_state["candidates"]["candidate-a-polished"].get(
                "resume_wall_ms", 0
            )
            runner2, _source2, _clients2 = self._runner_fixture(root)
            runner2._preflight = runner._preflight
            structural = {
                "schema_version": 1,
                "structural_pass": True,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=structural
            ):
                result = runner2.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertFalse(result["failed_candidates"])
            recovered = json.loads(
                (root / "out" / "candidates" / "candidate-a-polished" / "result.json").read_text()
            )
            state = json.loads((root / "out" / "integration_state.json").read_text())
            recovered_state = state["candidates"]["candidate-a-polished"]["recovered_resume"]
            self.assertGreater(recovered_state["active_duration_ms"], 0)
            durations = state["candidates"]["candidate-a-polished"]["resume_durations_ms"]
            self.assertTrue(all(duration > 0 for duration in durations))
            self.assertEqual(
                state["candidates"]["candidate-a-polished"]["resume_wall_ms"], sum(durations)
            )
            self.assertGreaterEqual(
                state["candidates"]["candidate-a-polished"]["resume_wall_ms"], prior_cumulative
            )
            timings = recovered["phase_timings_ms"]
            self.assertEqual(timings["total"], timings["first_attempt"] + timings["resume"])
            self.assertEqual(recovered_state["active_duration_ms"], durations[0])

    def test_public_runner_restart_after_durable_interrupted_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, clients = self._runner_fixture(root)
            integration_module = __import__(
                "trans_novel.benchmark.integration", fromlist=["_atomic_json"]
            )
            original_atomic = integration_module._atomic_json

            def stop_after_interrupted(path, value):
                original_atomic(path, value)
                if Path(path).name == "integration_state.json" and any(
                    item.get("status") == "interrupted"
                    for item in value.get("candidates", {}).values()
                ):
                    raise SystemExit("durable interruption")

            with (
                mock.patch(
                    "trans_novel.benchmark.integration._atomic_json",
                    side_effect=stop_after_interrupted,
                ),
                self.assertRaises(SystemExit),
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            self.assertEqual(len(clients), 2)
            runner2, _source2, clients2 = self._runner_fixture(root)
            runner2._preflight = runner._preflight
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                result = runner2.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertFalse(result["failed_candidates"])
            self.assertEqual(len(clients2), 4)
            self.assertFalse(
                any(
                    call["operation"].startswith("integration.canary") for call in clients2[0].calls
                )
            )

    def test_existing_state_without_request_refuses_before_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, clients = self._runner_fixture(root)
            (root / "out" / "candidates").mkdir(parents=True)
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            self.assertEqual(clients, [])

    def test_interrupted_event_prefix_and_slice_tamper_refuse_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            integration_module = __import__(
                "trans_novel.benchmark.integration", fromlist=["_atomic_json"]
            )
            original_atomic = integration_module._atomic_json

            def stop_after_interrupted(path, value):
                original_atomic(path, value)
                if Path(path).name == "integration_state.json" and any(
                    item.get("status") == "interrupted"
                    for item in value.get("candidates", {}).values()
                ):
                    raise SystemExit("durable interruption")

            with (
                mock.patch(
                    "trans_novel.benchmark.integration._atomic_json",
                    side_effect=stop_after_interrupted,
                ),
                self.assertRaises(SystemExit),
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            events = next(
                (root / "out" / "candidates" / "candidate-a-polished" / "state").glob(
                    "*/events.jsonl"
                )
            )
            events.write_text("X" + events.read_text(), encoding="utf-8")
            runner2, _source2, _clients2 = self._runner_fixture(root)
            runner2._preflight = runner._preflight
            with self.assertRaises(IntegrationError):
                runner2.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_fabricated_telemetry_and_slice_tamper_refuse_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            telemetry = root / "out" / "candidates" / "candidate-a-polished" / "telemetry.jsonl"
            telemetry.write_text(telemetry.read_text() + '{"usage_known":true}\n', encoding="utf-8")
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_noop_forged_predicate_and_missing_timing_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            result_path = root / "out" / "candidates" / "candidate-a-polished" / "result.json"
            value = json.loads(result_path.read_text())
            value["passed"] = True
            value.pop("phase_timings_ms", None)
            result_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_strict_provider_temperature_operation_and_usage_evidence_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            original = runner._preflight
            runner._preflight = mock.Mock(
                side_effect=IntegrationError("provider/temp/operation gate")
            )
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            self.assertTrue(runner._preflight.called)
            runner._preflight = original

    def test_singleton_factory_is_rejected_before_second_client_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            singleton = _InstrumentedFakeClient(
                handler=routing_handler,
                models=("fake/primary-a:off", "fake/editor-a:off", "fake/fast:off"),
            )
            runner.client_factory = lambda **_kwargs: singleton
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_canary_passes_real_telemetry_and_records_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                result = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertTrue(result["candidates"]["candidate-a-polished"]["result_path"])
            evidence = json.loads(
                (root / "out" / "candidates" / "candidate-a-polished" / "result.json").read_text()
            )
            self.assertTrue(evidence["canary_passed"])
            self.assertEqual(evidence["model_mismatch_count"], 0)
            self.assertEqual(evidence["reasoning_tokens"], 0)
            self.assertEqual(evidence["unknown_required_usage_count"], 0)

    def test_telemetry_evidence_rejects_provider_temperature_agent_and_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            _spec_value, candidate_spec, _source, _hash, lineage = runner._preflight()
            candidate = lineage["selected"][0]
            telemetry = root / "out" / "candidates" / candidate.candidate_id / "telemetry.jsonl"
            row = json.loads(telemetry.read_text().splitlines()[0])
            for field, altered in (
                ("provider", "wrong-provider"),
                ("temperature", 0.2),
                ("agent", "editor"),
                ("operation", "wrong.operation"),
            ):
                with self.subTest(field=field):
                    value = dict(row)
                    value[field] = altered
                    path = root / f"{field}.jsonl"
                    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
                    evidence = _telemetry_evidence(
                        path,
                        candidate=candidate,
                        candidate_spec=candidate_spec,
                    )
                    self.assertTrue(evidence["valid"])
                    self.assertGreaterEqual(evidence["model_mismatch_count"], 1)

    def test_canary_wrong_model_reasoning_and_unknown_usage_fail_public_run(self):
        for mutation, field in (
            ("wrong-model", "model_mismatch_count"),
            ("reasoning", "reasoning_tokens"),
            ("unknown", "unknown_required_usage_count"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner, _source, _clients = self._runner_fixture(root)
                original = runner._canary

                def altered(*args, _original=original, _mutation=mutation, _field=field, **kwargs):
                    value = _original(*args, **kwargs)
                    value["passed"] = False
                    value[_field] = 1
                    return value

                runner._canary = altered
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
                result = json.loads(
                    (
                        root / "out" / "candidates" / "candidate-a-polished" / "result.json"
                    ).read_text()
                )
                self.assertFalse(result["passed"])
                self.assertGreaterEqual(result[field], 1)

    def test_readiness_assignment_is_derived_from_production_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": False,
                    "mono": {"structural_pass": False},
                    "bilingual": {"structural_pass": False},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            result = json.loads(
                (root / "out" / "candidates" / "candidate-a-polished" / "result.json").read_text()
            )
            self.assertEqual(result["readiness_passed"], result["readiness_problem_count"] == 0)
            self.assertNotIn("readiness_problems", result)

    def test_interrupted_state_carries_restart_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            state = json.loads((root / "out" / "integration_state.json").read_text())
            for entry in state["candidates"].values():
                self.assertIn("interruption", entry)
                self.assertIn("before_target_hashes", entry)
                self.assertIn("canary_sha256", entry)
                self.assertIn("boundary_event_count", entry)

    def test_completion_tamper_noncanonical_and_missing_refuse_before_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, clients = self._runner_fixture(root)
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            integration = root / "out" / "integration.json"
            integration.write_text(
                integration.read_text().replace('"schema_version":1', '"schema_version": 1'),
                encoding="utf-8",
            )
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            self.assertEqual(len(clients), 6)

    def test_failed_candidate_continues_and_failed_completion_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            calls = {"count": 0}

            def failing_factory(**_kwargs):
                calls["count"] += 1
                return FakeClient(
                    handler=(lambda *_args: (_ for _ in ()).throw(RuntimeError("bad")))
                    if calls["count"] == 1
                    else routing_handler
                )

            runner.client_factory = failing_factory
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet",
                return_value={
                    "structural_pass": True,
                    "mono": {"structural_pass": True},
                    "bilingual": {"structural_pass": True},
                },
            ):
                value = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertIn("candidate-a-polished", value["failed_candidates"])
            self.assertIn("candidate-b-polished", value["candidates"])
            again = runner.run(
                root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
            )
            self.assertTrue(again["no_op"])
            self.assertIn("candidate-a-polished", again["failed_candidates"])

    def test_failed_result_schema_tamper_refuses_terminal_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            runner.client_factory = lambda **_kwargs: FakeClient(
                handler=lambda *_args: (_ for _ in ()).throw(RuntimeError("bad"))
            )
            runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")
            result_path = root / "out" / "candidates" / "candidate-a-polished" / "result.json"
            value = json.loads(result_path.read_text())
            value["book_id"] = "tampered"
            result_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out")

    def test_real_preflight_uses_phase4_manifest_and_strict_spec_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            hidden = root / "hidden.epub"
            write_sample_epub(str(hidden))
            hidden_hash = sha256_bytes(hidden.read_bytes())
            manifest_books = []
            book_entries = []
            for split, count in (("screen", 3), ("formal", 6)):
                for index in range(count):
                    book_id = f"{split}-{index}"
                    path = root / f"{book_id}.txt"
                    path.write_text(f"{book_id} fixture", encoding="utf-8")
                    digest = sha256_bytes(path.read_bytes())
                    manifest_books.append(
                        {
                            "book_id": book_id,
                            "source_sha256": digest,
                            "basename": path.name,
                            "split": split,
                            "format": "text",
                            "title": book_id,
                            "chapter_count": 1,
                            "parser_schema": 1,
                        }
                    )
                    book_entries.append(
                        {
                            "book_id": book_id,
                            "path": path.name,
                            "split": split,
                        }
                    )
            manifest_books.append(
                {
                    "book_id": "hidden-book",
                    "source_sha256": hidden_hash,
                    "basename": hidden.name,
                    "split": "hidden",
                    "format": "epub",
                    "title": "hidden-book",
                    "chapter_count": 2,
                    "parser_schema": 1,
                }
            )
            book_entries.append(
                {
                    "book_id": "hidden-book",
                    "path": hidden.name,
                    "split": "hidden",
                }
            )
            runner_rows: list[dict] = []
            challenge_keys: list[dict] = []
            book_sha = {row["book_id"]: row["source_sha256"] for row in manifest_books}
            next_index = {row["book_id"]: 0 for row in manifest_books}

            def add_passage(
                book_id: str,
                subset: str,
                words: int,
                strata: list[str] | None = None,
                *,
                context: bool = False,
            ) -> None:
                index = next_index[book_id]
                if context:
                    index = 3000 + len([row for row in runner_rows if row["subset"] == "context"])
                text = "word " * words
                segment = {
                    "segment_id": segment_id(book_sha[book_id], 0, index, text),
                    "index": index,
                    "source": text,
                    "kind": "text",
                    "cont": False,
                    "anchor": None,
                    "resource_href": None,
                    "meta": {},
                }
                row = {
                    "passage_id": passage_id(book_id, 0, index, index, [text]),
                    "subset": subset,
                    "book_id": book_id,
                    "chapter_index": 0,
                    "start": index,
                    "end": index,
                    "word_count": words,
                    "strata": strata or [],
                    "segments": [segment],
                    "context": None,
                }
                if context:
                    before_index = (
                        2000 + len([row for row in runner_rows if row["subset"] == "context"]) - 1
                    )
                    before_text = "context before"
                    before_id = segment_id(book_sha[book_id], 0, before_index, before_text)
                    row["context"] = {
                        "challenge_type": "chapter_transition",
                        "source_before": [{"segment_id": before_id, "source": before_text}],
                        "source_after": [],
                        "frozen_target_before": [
                            {"segment_id": before_id, "target": "context target"}
                        ],
                    }
                    challenge_keys.append(
                        {
                            "passage_id": row["passage_id"],
                            "challenge_type": "chapter_transition",
                            "answer_key": "fixture-answer",
                            "rationale": "fixture-rationale",
                        }
                    )
                runner_rows.append(row)
                next_index[book_id] = max(next_index[book_id], index + 1)

            for book_id, words in zip(
                ("screen-0", "screen-1", "screen-2"), (3333, 3333, 3334), strict=True
            ):
                add_passage(book_id, "screen", words)
            for book_id in ("formal-0", "formal-1", "formal-2"):
                add_passage(book_id, "continuous", 10000)
                for stratum in (
                    "narrative",
                    "dialogue",
                    "literary",
                    "long_sentence",
                    "idiom_metaphor_wordplay",
                    "terminology",
                    "numbers_entities",
                    "special_format",
                ):
                    add_passage(book_id, "stratified", 312, [stratum])
                    add_passage(book_id, "stratified", 313, [stratum])
            for book_id, words in zip(
                ("formal-3", "formal-4", "formal-5"),
                ((333, 333, 334, 333, 333), (333, 333, 334, 333, 334), (333, 333, 334, 333, 334)),
                strict=True,
            ):
                for count in words:
                    add_passage(book_id, "context", count, context=True)
            _write_minimal_artifact(
                corpus, runner_rows, manifest_books=manifest_books, challenge_keys=challenge_keys
            )
            corpus_value = json.loads((corpus / "corpus.json").read_text())
            corpus_value["quotas"]["actual"] = {
                "screen": 10000,
                "continuous": 30000,
                "stratified": 15000,
                "context": 5000,
                "hidden": 0,
                "formal": 50000,
            }
            semantics = {
                "corpus": {
                    key: value for key, value in corpus_value.items() if key != "corpus_sha256"
                },
                "runner_segments": runner_rows,
                "challenge_keys": challenge_keys,
            }
            corpus_value["corpus_sha256"] = sha256_bytes(canonical_json(semantics).encode("utf-8"))
            (corpus / "corpus.json").write_text(
                canonical_json(corpus_value) + "\n", encoding="utf-8"
            )
            candidate_path = root / "candidates.yaml"
            candidate_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 3,
                        "benchmark_id": "phase9",
                        "temperature": 0.1,
                        "seed": None,
                        "replicates": 1,
                        "candidates": [
                            {
                                "candidate_id": "a-polished",
                                "translator_model": "bailian/qwen3.8-max:off",
                                "analyst_model": "bailian/qwen3.7-flash:off",
                                "editor_model": "bailian/deepseek-v4-pro:off",
                                "fast_model": "bailian/qwen3.7-flash:off",
                                "pipeline_variant": "polish",
                            },
                            {
                                "candidate_id": "b-polished",
                                "translator_model": "bailian/deepseek-v4-flash:off",
                                "analyst_model": "bailian/qwen3.7-flash:off",
                                "editor_model": "bailian/qwen3.7-plus:off",
                                "fast_model": "bailian/qwen3.7-flash:off",
                                "pipeline_variant": "polish",
                            },
                        ],
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            book_spec_path = root / "books.yaml"
            book_spec_path.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "source_language": "en",
                        "target_language": "zh",
                        "books": book_entries,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            corpus_hash = json.loads((corpus / "corpus.json").read_text())["corpus_sha256"]
            integration_path = root / "integration.yaml"
            integration_path.write_text(
                yaml.safe_dump(
                    _spec(
                        corpus_sha256=corpus_hash,
                        candidate_spec_sha256=sha256_bytes(candidate_path.read_bytes()),
                        candidate_ids=["a-polished", "b-polished"],
                        book_id="hidden-book",
                    ),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            clients: list[dict] = []
            runner = IntegrationRunner(client_factory=lambda **kwargs: clients.append(kwargs))
            spec, _candidate_spec, source, source_hash, lineage = runner._preflight(
                corpus, book_spec_path, candidate_path, integration_path
            )
            self.assertEqual(spec.book_id, "hidden-book")
            self.assertEqual(source, hidden.resolve())
            self.assertEqual(source_hash, hidden_hash)
            self.assertEqual(
                [item.candidate_id for item in lineage["selected"]], ["a-polished", "b-polished"]
            )
            self.assertEqual(clients, [])

    def test_preflight_rejects_source_bilingual_alias_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, source, _clients = self._runner_fixture(root)
            out = root / "out"
            candidate_root = out / "candidates" / "candidate-a-polished"
            (candidate_root / "outputs").mkdir(parents=True)
            (candidate_root / "outputs" / "hidden-book.epub").symlink_to(source)
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", out)
            (candidate_root / "outputs" / "hidden-book.epub").unlink()
            outside = root / "outside"
            outside.mkdir()
            (candidate_root / "state").symlink_to(outside)
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", out)
            self.assertTrue(source.exists())

    def test_triplet_aggregate_false_cannot_be_overridden_by_child_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            value = {
                "schema_version": 1,
                "structural_pass": False,
                "source": {"structural_pass": True},
                "mono": {"structural_pass": True},
                "bilingual": {"structural_pass": True},
            }
            with mock.patch(
                "trans_novel.benchmark.integration.validate_epub_triplet", return_value=value
            ):
                result = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertTrue(result["failed_candidates"])

    def test_request_state_result_completion_tamper_and_missing_refuse(self):
        for artifact in ("request", "state", "result", "completion"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner, _source, _clients = self._runner_fixture(root)
                with mock.patch(
                    "trans_novel.benchmark.integration.validate_epub_triplet",
                    return_value={
                        "structural_pass": True,
                        "mono": {"structural_pass": True},
                        "bilingual": {"structural_pass": True},
                    },
                ):
                    runner.run(
                        root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                    )
                out = root / "out"
                paths = {
                    "request": out / "integration_request.json",
                    "state": out / "integration_state.json",
                    "result": out / "candidates" / "candidate-a-polished" / "result.json",
                    "completion": out / "integration_complete.json",
                }
                path = paths[artifact]
                if artifact == "completion":
                    path.unlink()
                elif artifact == "state":
                    path.write_text('{"schema_version": 1}\n', encoding="utf-8")
                else:
                    path.write_text(
                        path.read_text(encoding="utf-8").replace("phase9", "phaseX", 1),
                        encoding="utf-8",
                    )
                with self.assertRaises(IntegrationError):
                    runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", out)

    def test_cross_candidate_and_bilingual_aliases_refuse_before_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, source, clients = self._runner_fixture(root)
            out = root / "out"
            first = out / "candidates" / "candidate-a-polished"
            first.mkdir(parents=True)
            (first / "outputs").mkdir()
            (first / "outputs" / "hidden-bi.epub").symlink_to(source)
            (out / "candidates" / "candidate-b-polished").symlink_to(
                first, target_is_directory=True
            )
            with self.assertRaises(IntegrationError):
                runner.run(root, root / "b.yaml", root / "c.yaml", root / "i.yaml", out)
            self.assertEqual(clients, [])

    def test_false_benchmark_interruption_is_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            runner._canary = lambda *_args, **_kwargs: {
                "schema_version": 1,
                "passed": True,
                "reasoning_tokens": 0,
                "model_mismatch_count": 0,
                "unknown_required_usage_count": 0,
            }
            with mock.patch.object(
                Application, "run_all", side_effect=BenchmarkInterruption("false")
            ):
                value = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertEqual(len(value["failed_candidates"]), 2)
            self.assertTrue(
                all(
                    "IntegrationError" in json.loads(path.read_text()).get("failure_reasons", [])
                    for path in (root / "out").glob("candidates/*/result.json")
                )
            )

    def test_missing_physical_telemetry_or_usage_fails_with_unknown_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner, _source, _clients = self._runner_fixture(root)
            original_run = Application.run_all
            calls = {"count": 0}

            def remove_after_resume(app, *args, **kwargs):
                calls["count"] += 1
                value = original_run(app, *args, **kwargs)
                if calls["count"] == 2:
                    (
                        root / "out" / "candidates" / "candidate-a-polished" / "telemetry.jsonl"
                    ).unlink()
                return value

            with (
                mock.patch.object(Application, "run_all", remove_after_resume),
                mock.patch(
                    "trans_novel.benchmark.integration.validate_epub_triplet",
                    return_value={
                        "structural_pass": True,
                        "mono": {"structural_pass": True},
                        "bilingual": {"structural_pass": True},
                    },
                ),
            ):
                result = runner.run(
                    root, root / "b.yaml", root / "c.yaml", root / "i.yaml", root / "out"
                )
            self.assertIn("candidate-a-polished", result["failed_candidates"])
            failed = json.loads(
                (root / "out" / "candidates" / "candidate-a-polished" / "result.json").read_text()
            )
            self.assertGreater(failed["unknown_required_usage_count"], 0)

    def test_cli_distinguishes_fresh_resumed_noop_and_failed_reasons(self):
        cli_runner = CliRunner()
        command = [
            "tools",
            "benchmark",
            "integration",
            "run",
            "corpus",
            "book.yaml",
            "candidates.yaml",
            "integration.yaml",
            "--out",
            "out",
        ]
        for payload, expected in (
            ({"no_op": False, "resumed": False, "failed_candidates": []}, "fresh"),
            ({"no_op": False, "resumed": True, "failed_candidates": []}, "resumed"),
            ({"no_op": True, "resumed": True, "failed_candidates": []}, "no-op"),
        ):
            with (
                self.subTest(expected=expected),
                mock.patch.object(IntegrationRunner, "run", return_value=payload),
            ):
                result = cli_runner.invoke(app, command)
            self.assertEqual(result.exit_code, 0)
            self.assertIn(f"Integration {expected}", result.output)
        with mock.patch.object(
            IntegrationRunner,
            "run",
            return_value={"no_op": False, "resumed": False, "failed_candidates": ["candidate-a"]},
        ):
            result = cli_runner.invoke(app, command)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("candidate-a", result.output)


if __name__ == "__main__":
    unittest.main()
