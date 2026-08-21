from __future__ import annotations

import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from trans_novel.agents.prompts import TranslationContextBundle
from trans_novel.assemble.translator import Translator
from trans_novel.benchmark.corpus import canonical_json, sha256_bytes
from trans_novel.benchmark.runner import (
    AttributionRunner,
    BenchmarkError,
    _JsonlTelemetrySink,
    _safe_id,
)
from trans_novel.benchmark.schema import (
    BookPreparation,
    CandidateSpec,
    ChapterSourceDigest,
    GlossaryPreparation,
    PreparationBundle,
    PreparationSpec,
)
from trans_novel.config import Config, LLMConfig, ModelRoles
from trans_novel.glossary.store import GlossaryTerm
from trans_novel.llm import FakeClient, GenerationOptions
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.model_profiles import validate_model_selection
from trans_novel.pipeline import lint


def _handler(messages, agent, operation, json_mode):
    user = messages[-1]["content"]
    count = len(re.findall(r"^\[\d+\] ", user, re.M))
    if agent == "translator":
        return '{"translations":[' + ",".join('"译文"' for _ in range(max(1, count))) + "]}"
    if agent == "editor":
        target_block = user.split("【待润色中文译文】", 1)[-1]
        count = len(re.findall(r"^\[\d+\] ", target_block, re.M))
        return '{"polished":[' + ",".join('"润色"' for _ in range(max(1, count))) + "]}"
    return "{}"


def _preparation_bundle(
    *,
    corpus_sha256: str = "0" * 64,
    book_id: str = "book",
    chapter_digests: dict[str, str] | None = None,
    glossary: list[dict] | None = None,
    style: str = "",
    synopsis: str = "",
    analysis: dict | None = None,
) -> PreparationBundle:
    """Build a strict, internally hashed fixture using production helpers."""
    chapter_digests = chapter_digests or {"0": ""}
    source_sha256 = sha256_bytes(f"{book_id}:source".encode())
    source_digests = [
        ChapterSourceDigest(
            chapter_index=int(index),
            source_sha256=sha256_bytes(f"{book_id}:chapter:{index}:{digest}".encode()),
        )
        for index, digest in sorted(chapter_digests.items(), key=lambda item: int(item[0]))
    ]
    book = BookPreparation(
        book_id=book_id,
        source_sha256=source_sha256,
        analysis=analysis or {},
        style=style,
        style_brief=style or "deterministic fixture style",
        book_synopsis=synopsis,
        chapter_digests=chapter_digests,
        source_digests=source_digests,
        glossary=[GlossaryPreparation.model_validate(row) for row in (glossary or [])],
        node_fingerprints={},
    )
    spec = PreparationSpec(
        schema_version=1,
        provider="fake",
        primary_model="fake:off",
        editor_model="fake:off",
        fast_model="fake:off",
        temperature=0.1,
        seed=None,
    )
    spec_hash = sha256_bytes(canonical_json(spec.model_dump(mode="python")).encode("utf-8"))
    provisional = PreparationBundle(
        schema_version=1,
        corpus_sha256=corpus_sha256,
        preparation_spec=spec,
        preparation_spec_sha256=spec_hash,
        preparation_sha256="0" * 64,
        books={book_id: book},
    )
    from trans_novel.benchmark.runner import _preparation_hash

    return provisional.model_copy(update={"preparation_sha256": _preparation_hash(provisional)})


class SinkFakeClient(FakeClient):
    telemetry_sink = None


class Phase5AttributionTests(unittest.TestCase):
    def test_strict_candidate_control_and_model_suffix(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "bailian",
                "fast_model": "qwen3.8-max:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "qwen3.8-max:off", "editor_model": None},
                    {
                        "candidate_id": "a-edit",
                        "primary_model": "qwen3.8-max:off",
                        "editor_model": "qwen3.8-max:off",
                    },
                ],
            }
        )
        self.assertEqual(spec.candidates[0].editor_model, None)
        with self.assertRaises(ValidationError):
            CandidateSpec.model_validate({**spec.model_dump(), "temperature": 0.2})

    def test_candidate_seed_must_be_exactly_null(self):
        value = {
            "schema_version": 1,
            "benchmark_id": "offline",
            "provider": "fake",
            "fast_model": "fake:off",
            "temperature": 0.1,
            "seed": 7,
            "replicates": 1,
            "default_context_strategy": "c2",
            "candidates": [
                {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
            ],
        }
        with self.assertRaises(ValidationError):
            CandidateSpec.model_validate(value)

    def test_polish_gate_rejects_locked_term_miss(self):
        locked = GlossaryTerm(source="Alice", target="爱丽丝", locked=True)
        result = lint.polish_gate(
            "I met Alice.", "我遇见了爱丽丝。", "我遇见了她。", locked_terms=[locked]
        )
        self.assertFalse(result.accepted)
        self.assertIn("term_miss", result.rejection_reasons)

    def test_benchmark_context_has_all_three_boundary_blocks(self):
        client = FakeClient(_handler)
        config = Config(
            llm=LLMConfig(
                provider="fake",
                models=ModelRoles(primary="fake:off", editor="fake:off", fast="fake:off"),
            ),
            source_lang="en",
            target_lang="zh",
        )
        translator = Translator(client, config)
        translator.translate_batch(
            ["Current"],
            agent="translator",
            context_bundle=TranslationContextBundle(
                "[before] Before", "[target] 之前", "[after] After"
            ),
        )
        user = client.calls[0]["messages"][-1]["content"]
        self.assertIn("[before] Before", user)
        self.assertIn("[target] 之前", user)
        self.assertIn("[after] After", user)
        self.assertIn("[0] Current", user)

    def test_polish_gate_rejects_new_lint_type_and_preserves_normalized_raw(self):
        result = lint.polish_gate("“Hi,” he said.", "他说:“嗨”。", "他说:嗨。", src_lang="en")
        self.assertFalse(result.accepted)
        self.assertEqual(result.selected, "他说：“嗨”。")
        self.assertIn("quote_loss", result.rejection_reasons)

    def test_polish_gate_accepts_proposal_without_new_lint_type(self):
        result = lint.polish_gate("“Hi,” he said.", "他说:“嗨”。", "他说：“嗨”！", src_lang="en")
        self.assertTrue(result.accepted)
        self.assertEqual(result.selected, "他说：“嗨”！")

    def test_shared_translation_artifact_resume_does_not_repeat_calls(self):
        client = SinkFakeClient(_handler)
        runner = AttributionRunner(client=client)
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        bundle = _preparation_bundle()
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "start": 0,
            "end": 0,
            "word_count": 1,
            "strata": [],
            "segments": [
                {
                    "segment_id": "s",
                    "index": 0,
                    "source": "Source",
                    "kind": "text",
                    "cont": False,
                    "anchor": None,
                    "resource_href": None,
                    "meta": {},
                }
            ],
            "context": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = GenerationOptions(temperature=0.1)
            runner._translate_artifact(
                root,
                spec,
                options,
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            first_calls = len(client.calls)
            runner._translate_artifact(
                root,
                spec,
                options,
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            self.assertEqual(len(client.calls), first_calls)

    def test_hidden_rows_are_ignored_and_empty_gender_is_valid(self):
        glossary = {
            "source": "Alice",
            "target": "爱丽丝",
            "type": "person",
            "gender": "",
        }
        bundle = _preparation_bundle(glossary=[glossary])
        self.assertEqual(bundle.books["book"].glossary[0].gender, "")
        invalid = bundle.model_dump(mode="python")
        invalid["books"]["book"]["glossary"][0]["source"] = ""
        with self.assertRaises(ValidationError):
            PreparationBundle.model_validate(invalid)
        rows = [
            {"passage_id": "hidden", "subset": "hidden"},
            {"passage_id": "screen", "subset": "screen"},
        ]
        runner = AttributionRunner()
        self.assertEqual(
            [row["passage_id"] for row in runner._passage_set(rows, "screen", "c2")], ["screen"]
        )

    def test_direct_and_compatibility_factory_clients_require_telemetry(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        options = GenerationOptions(temperature=0.1)

        class Unobservable:
            def __init__(self):
                self.calls = 0

            def complete(self, *args, **kwargs):
                self.calls += 1
                return "{}"

        direct = Unobservable()
        with tempfile.TemporaryDirectory() as directory:
            sink = _JsonlTelemetrySink(Path(directory) / "telemetry.jsonl")
            with self.assertRaises(BenchmarkError):
                AttributionRunner(client=direct)._client(
                    spec, "fake:off", "translator", options, sink
                )
            self.assertEqual(direct.calls, 0)

        class Compatible:
            telemetry_sink = None

        created = []

        def factory(model, role):
            client = Compatible()
            created.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory:
            sink = _JsonlTelemetrySink(Path(directory) / "telemetry.jsonl")
            client = AttributionRunner(client_factory=factory)._client(
                spec, "fake:off", "translator", options, sink
            )
            self.assertIs(client.telemetry_sink, sink)

    def test_canary_telemetry_requires_complete_typed_success(self):
        with tempfile.TemporaryDirectory() as directory:
            sink = _JsonlTelemetrySink(Path(directory) / "telemetry.jsonl")
            sink.records.append({"requested_model": "fake:off", "resolved_model": "fake:off"})
            with self.assertRaises(BenchmarkError):
                AttributionRunner._check_canary_telemetry(None, "fake:off", sink)

    def test_tampered_translation_journal_rejected_before_next_call(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        bundle = _preparation_bundle()
        rows = [
            {
                "passage_id": "p",
                "subset": "continuous",
                "book_id": "book",
                "chapter_index": 0,
                "segments": [
                    {"segment_id": "s0", "source": "a " * 1000},
                    {"segment_id": "s1", "source": "b " * 1000},
                ],
            }
        ]
        state = {"fail": True}

        def handler(messages, agent, operation, json_mode):
            if state["fail"] and len(client.calls) >= 2:
                raise RuntimeError("stop after first physical batch")
            count = len(re.findall(r"^\[\d+\] ", messages[-1]["content"], re.M))
            return '{"translations":[' + ",".join('"译文"' for _ in range(count)) + "]}"

        client = SinkFakeClient(handler)
        runner = AttributionRunner(client=client)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                runner._translate_artifact(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    "fake:off",
                    1,
                    "continuous",
                    "c2",
                    rows,
                    "key",
                )
            journal_path = root / "translation" / "key" / "journal.json"
            journal = json.loads(journal_path.read_text())
            journal["batches"][0]["segments"][0]["source"] = "tampered"
            journal_path.write_text(json.dumps(journal))
            calls = len(client.calls)
            state["fail"] = False
            with self.assertRaises(BenchmarkError):
                runner._translate_artifact(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    "fake:off",
                    1,
                    "continuous",
                    "c2",
                    rows,
                    "key",
                )
            self.assertEqual(len(client.calls), calls)
            journal["batches"][0]["segments"][0]["source"] = rows[0]["segments"][0]["source"]
            journal["batches"][0]["context"]["target_before"] = "tampered"
            journal_path.write_text(json.dumps(journal))
            with self.assertRaises(BenchmarkError):
                runner._translate_artifact(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    "fake:off",
                    1,
                    "continuous",
                    "c2",
                    rows,
                    "key",
                )
            self.assertEqual(len(client.calls), calls)

    def test_completed_translation_validation_is_read_only_and_fails_closed(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        bundle = _preparation_bundle()
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "segments": [{"segment_id": "s", "source": "Source"}],
        }
        runner = AttributionRunner(client=SinkFakeClient(_handler))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = runner._translate_artifact(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            files = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            runner._validate_translation_artifact(
                artifact,
                spec,
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            self.assertEqual(
                files,
                {
                    path.relative_to(root): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )
            usage_path = artifact / "usage.json"
            usage_path.write_text("{}")
            with self.assertRaises(BenchmarkError):
                runner._validate_translation_artifact(
                    artifact,
                    spec,
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    "fake:off",
                    1,
                    "screen",
                    "c2",
                    [row],
                    "key",
                )
            self.assertEqual(usage_path.read_text(), "{}")

    def test_editor_validation_recomputes_gate_and_lint_baseline(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "control", "primary_model": "fake:off", "editor_model": None},
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": "fake:off"},
                ],
            }
        )
        bundle = _preparation_bundle(style="", synopsis="")
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "segments": [{"segment_id": "s", "source": "Source"}],
        }
        runner = AttributionRunner(client=SinkFakeClient(_handler))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translation = runner._translate_artifact(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            editor = runner._editor_artifact(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                translation,
                "key",
            )
            passage_path = next((editor / "passages").glob("*.json"))
            passage = json.loads(passage_path.read_text())
            record = passage["segments"][0]
            record["translation_lint_issues"] = [{"type": "tampered", "detail": "tampered"}]
            record["final"] = "篡改"
            record["final_sha256"] = sha256_bytes(record["final"].encode())
            record["output_sha256"] = record["final_sha256"]
            passage_path.write_text(json.dumps(passage, ensure_ascii=False))
            with self.assertRaises(BenchmarkError):
                runner._validate_editor_artifact(
                    editor, spec, bundle, "0" * 64, "1" * 64, "fake:off", translation, "key"
                )

    def test_editor_two_batch_resume_does_not_repeat_completed_batch(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "control", "primary_model": "fake:off", "editor_model": None},
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": "fake:off"},
                ],
            }
        )
        bundle = _preparation_bundle()
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "segments": [
                {"segment_id": "s0", "source": "a " * 1000},
                {"segment_id": "s1", "source": "b " * 1000},
            ],
        }
        editor_calls = []
        state = {"fail": True}

        def handler(messages, agent, operation, json_mode):
            if agent == "editor":
                editor_calls.append(messages)
                if state["fail"] and len(editor_calls) >= 2:
                    raise RuntimeError("stop after first editor batch")
            target = messages[-1]["content"]
            if agent == "editor":
                target = target.split("【待润色中文译文】", 1)[-1]
            count = len(re.findall(r"^\[\d+\] ", target, re.M))
            key = "polished" if agent == "editor" else "translations"
            value = '"润色"' if agent == "editor" else '"译文"'
            return '{"' + key + '":[' + ",".join(value for _ in range(count)) + "]}"

        client = SinkFakeClient(handler)
        runner = AttributionRunner(client=client)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            translation = runner._translate_artifact(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                1,
                "screen",
                "c2",
                [row],
                "key",
            )
            with self.assertRaises(RuntimeError):
                runner._editor_artifact(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    "fake:off",
                    translation,
                    "key",
                )
            state["fail"] = False
            runner._editor_artifact(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                "fake:off",
                translation,
                "key",
            )
            self.assertEqual(len(editor_calls), 3)

    def test_completed_attribution_shape_and_manifest_tamper_fail_closed(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        bundle = _preparation_bundle()
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "segments": [{"segment_id": "s", "source": "Source"}],
        }
        runner = AttributionRunner(client=SinkFakeClient(_handler))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = runner._attribution(
                root, spec, GenerationOptions(temperature=0.1), bundle, "0" * 64, "1" * 64, [row]
            )
            files = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            calls = len(runner.client.calls)
            second = runner._attribution(
                root, spec, GenerationOptions(temperature=0.1), bundle, "0" * 64, "1" * 64, [row]
            )
            self.assertEqual(second["candidate_count"], result["candidate_count"])
            self.assertEqual(len(runner.client.calls), calls)
            self.assertEqual(
                files,
                {
                    path.relative_to(root): path.read_bytes()
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )
            manifest = root / "candidates.json"
            manifest.write_text("{}")
            with self.assertRaises(BenchmarkError):
                runner._validate_completed(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    [row],
                    "attribution",
                    None,
                )

    def test_completed_canary_recomputes_persisted_outputs_and_typed_telemetry(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "offline",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {"candidate_id": "a", "primary_model": "fake:off", "editor_model": None}
                ],
            }
        )
        bundle = _preparation_bundle()
        row = {
            "passage_id": "p",
            "subset": "screen",
            "book_id": "book",
            "chapter_index": 0,
            "segments": [{"segment_id": "s", "source": "Source"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            telemetry_path = root / "canary" / _safe_id("fake:off") / "telemetry.jsonl"
            sink = _JsonlTelemetrySink(telemetry_path)
            telemetry = CallAttemptTelemetry.model_validate(
                {
                    "schema_version": 1,
                    "logical_call_id": "call",
                    "attempt_index": 1,
                    "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "elapsed_ms": 0,
                    "stage": "Translator",
                    "agent": "translator",
                    "operation": "translate.batch",
                    "provider": "fake",
                    "requested_model": validate_model_selection("fake", "fake:off").model,
                    "resolved_model": validate_model_selection("fake", "fake:off").model,
                    "reasoning_enabled": False,
                    "reasoning_effort": None,
                    "temperature": 0.1,
                    "seed": None,
                    "json_mode": True,
                    "max_tokens": None,
                    "status": "success",
                    "retry_class": None,
                    "http_status": None,
                    "finish_reason": None,
                    "response_id": None,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "cache_hit_tokens": 0,
                    "cache_miss_tokens": 0,
                    "reasoning_tokens": 0,
                    "billed_usage_unknown": False,
                    "request_sha256": "0" * 64,
                    "response_sha256": "1" * 64,
                }
            )
            sink.record(telemetry)
            sink.records.insert(0, {"status": "error"})
            self.assertEqual(
                len(
                    AttributionRunner._check_canary_telemetry(
                        None,
                        validate_model_selection("fake", "fake:off").model,
                        sink,
                        start_index=1,
                    )
                ),
                1,
            )
            sink.records.append({**telemetry.model_dump(mode="python"), "status": None})
            with self.assertRaises(BenchmarkError):
                AttributionRunner._check_canary_telemetry(
                    None, validate_model_selection("fake", "fake:off").model, sink, start_index=2
                )
            outputs = ["译文"]
            current = [telemetry.model_dump(mode="python")]
            result = {
                "primary_model": "fake:off",
                "segments": 1,
                "outputs": outputs,
                "output_sha256": sha256_bytes(canonical_json(outputs).encode()),
                "telemetry_sha256": sha256_bytes(telemetry_path.read_bytes()),
                "telemetry_records_sha256": sha256_bytes(canonical_json(current).encode()),
                "attempt_count": 1,
                "editor_pairs": [],
            }
            canary = {
                "status": "passed",
                "sample_id": "p",
                "source_hash": sha256_bytes(
                    canonical_json([{"segment_id": "s", "source": "Source"}]).encode()
                ),
                "segments": [{"segment_id": "s", "source": "Source"}],
                "temperature": 0.1,
                "reasoning_tokens": 0,
                "results": [result],
            }
            (root / "canary.json").write_text(json.dumps(canary))
            runner = AttributionRunner()
            runner._validate_completed(
                root,
                spec,
                GenerationOptions(temperature=0.1),
                bundle,
                "0" * 64,
                "1" * 64,
                [row],
                "canary",
                "p",
            )
            result["outputs"] = ["tampered"]
            (root / "canary.json").write_text(json.dumps(canary))
            with self.assertRaises(BenchmarkError):
                runner._validate_completed(
                    root,
                    spec,
                    GenerationOptions(temperature=0.1),
                    bundle,
                    "0" * 64,
                    "1" * 64,
                    [row],
                    "canary",
                    "p",
                )


if __name__ == "__main__":
    unittest.main()
