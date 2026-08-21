from __future__ import annotations

import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from tests.fake_llm import routing_handler
from trans_novel.benchmark.corpus import canonical_json, sha256_bytes
from trans_novel.benchmark.runner import (
    BenchmarkError,
    FullRunner,
    _preparation_hash,
    _safe_book_id,
    build_continuous_document,
    freeze_preparation,
    preparation_source,
    validate_preparation,
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
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.llm import FakeClient, GenerationOptions
from trans_novel.llm.usage import usage_delta
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.contracts import GOAL_PREPARE
from trans_novel.pipeline.fingerprints import frozen_input_fingerprint
from trans_novel.pipeline.runstore import clone_closed_runstore


class SinkFakeClient(FakeClient):
    telemetry_sink = None


class Phase6FrozenPreparationTests(unittest.TestCase):
    def _completed_fixture(
        self,
        *,
        replicates: int = 1,
        handler=routing_handler,
        source: str = "Alice walks home.",
    ):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "corpus.json").write_text(
            json.dumps({"corpus_sha256": "d" * 64}), encoding="utf-8"
        )
        (corpus / "source_manifest.json").write_text(
            json.dumps({"books": [{"book_id": "formal-1", "source_sha256": "a" * 64}]}),
            encoding="utf-8",
        )
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "bench",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": replicates,
                "default_context_strategy": "c2",
                "candidates": [
                    {
                        "candidate_id": "control",
                        "primary_model": "primary:off",
                        "editor_model": None,
                    },
                    {
                        "candidate_id": "editor-a",
                        "primary_model": "primary:off",
                        "editor_model": "editor-a:off",
                    },
                    {
                        "candidate_id": "editor-b",
                        "primary_model": "primary:off",
                        "editor_model": "editor-b:off",
                    },
                ],
            }
        )
        rows = [
            {
                "passage_id": "p0",
                "subset": "continuous",
                "book_id": "formal-1",
                "chapter_index": 0,
                "segments": [
                    {
                        "segment_id": "s0",
                        "index": 0,
                        "source": source,
                        "kind": "text",
                        "cont": False,
                        "meta": {},
                    }
                ],
            }
        ]
        bundle = self._bundle()
        client = SinkFakeClient(handler=handler)
        with (
            patch(
                "trans_novel.benchmark.runner._load_corpus_rows",
                return_value=("d" * 64, rows),
            ),
            patch(
                "trans_novel.benchmark.runner.load_candidate_spec",
                return_value=spec,
            ),
            patch(
                "trans_novel.benchmark.runner.validate_candidate_capabilities",
                return_value=GenerationOptions(temperature=0.1),
            ),
            patch(
                "trans_novel.benchmark.runner.load_preparation_bundle",
                return_value=(bundle, bundle.preparation_sha256),
            ),
        ):
            result = FullRunner(client=client).run(
                corpus, "candidates.yaml", "preparation", root / "run"
            )
        self.assertEqual(result["status"], "completed")
        return temp, root, spec, bundle, rows, client

    def _resume_fixture(self, root, spec, bundle, rows):
        with (
            patch(
                "trans_novel.benchmark.runner._load_corpus_rows",
                return_value=("d" * 64, rows),
            ),
            patch(
                "trans_novel.benchmark.runner.load_candidate_spec",
                return_value=spec,
            ),
            patch(
                "trans_novel.benchmark.runner.validate_candidate_capabilities",
                return_value=GenerationOptions(temperature=0.1),
            ),
            patch(
                "trans_novel.benchmark.runner.load_preparation_bundle",
                return_value=(bundle, bundle.preparation_sha256),
            ),
        ):
            return FullRunner().run(root / "corpus", "candidates.yaml", "preparation", root / "run")

    def test_prepare_resume_retains_flushed_usage_without_model_calls(self):
        class FakeStore:
            def __init__(self, root: Path):
                self.root = root
                self.root.mkdir(parents=True, exist_ok=True)
                self.usage_path = str(self.root / "usage.json")
                self.glossary_path = str(self.root / "glossary.db")

            def load_usage(self):
                path = Path(self.usage_path)
                return json.loads(path.read_text()) if path.exists() else None

            def save_usage(self, data):
                Path(self.usage_path).write_text(json.dumps(data), encoding="utf-8")

            def log_event(self, *args, **kwargs):
                return None

            def load_analysis(self):
                return {"style": "plain", "book_synopsis": "synopsis"}

            def load_state(self):
                return SimpleNamespace(
                    chapters=[SimpleNamespace(index=0)],
                    nodes={"analyze": SimpleNamespace(input_fingerprint="c" * 64)},
                )

            def load_progress(self, chapter_index):
                return SimpleNamespace(source_digest="digest")

            def load_chapter(self, index):
                return SimpleNamespace(
                    text_segments=[
                        SimpleNamespace(
                            source="Alice walks home.",
                            target="",
                            meta={},
                        )
                    ]
                )

        class FakeApplication:
            configs: ClassVar[list[Config]] = []

            def __init__(self, config, client):
                self.config = config
                self.client = client
                self.store = FakeStore(Path(config.state_dir))
                self.configs.append(config)

            def prepare_for_translation(self, path):
                self.store.save_usage(self.client.usage_summary())
                return self.store

        class EmptyGlossary:
            def __init__(self, path):
                pass

            def all_terms(self):
                return []

            def close(self):
                pass

        client = SinkFakeClient(handler=lambda *args, **kwargs: "{}")
        client.usage.record(
            agent="preparer",
            operation="prepare.analysis",
            provider="fake",
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        )
        prep_spec = PreparationSpec(
            schema_version=1,
            provider="fake",
            primary_model="fake:off",
            editor_model="fake:off",
            fast_model="fake:off",
            temperature=0.1,
            seed=None,
        )
        book_entry = SimpleNamespace(
            book_id="formal-1",
            path="book.txt",
            split="formal",
        )
        book_spec = SimpleNamespace(books=[book_entry])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "corpus.json").write_text(
                json.dumps({"corpus_sha256": "d" * 64}), encoding="utf-8"
            )
            source = root / "book.txt"
            source.write_text("Alice walks home.", encoding="utf-8")
            (corpus / "source_manifest.json").write_text(
                json.dumps(
                    {
                        "books": [
                            {
                                "book_id": "formal-1",
                                "source_sha256": sha256_bytes(source.read_bytes()),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "preparation"
            with (
                patch("trans_novel.benchmark.runner.validate_corpus"),
                patch("trans_novel.benchmark.runner.load_book_spec", return_value=book_spec),
                patch(
                    "trans_novel.benchmark.runner.load_preparation_spec",
                    return_value=prep_spec,
                ),
                patch("trans_novel.benchmark.runner._validate_model"),
                patch("trans_novel.pipeline.bootstrap.Application", FakeApplication),
                patch("trans_novel.benchmark.runner.GlossaryStore", EmptyGlossary),
                patch(
                    "trans_novel.agents.analyzer.Analyzer.style_brief",
                    side_effect=[RuntimeError("crash after usage flush"), "plain"],
                ),
            ):
                with self.assertRaises(BenchmarkError):
                    freeze_preparation(
                        corpus,
                        root / "books.yaml",
                        root / "preparation.yaml",
                        output,
                        client=client,
                    )
                prepared = freeze_preparation(
                    corpus,
                    root / "books.yaml",
                    root / "preparation.yaml",
                    output,
                    client=client,
                )
            state_root = Path(FakeApplication.configs[-1].state_dir)
            self.assertEqual(state_root.parent.resolve(), (output / "work").resolve())
            self.assertEqual(state_root.name, _safe_book_id("formal-1"))
            self.assertNotEqual(state_root.name, "formal-1")
            persisted = json.loads((state_root / "usage.json").read_text())
            self.assertEqual(prepared["books"]["formal-1"]["usage"], persisted)
            telemetry_path = output / "telemetry" / f"{_safe_book_id('formal-1')}.jsonl"
            self.assertEqual(telemetry_path.read_bytes(), b"")
            self.assertEqual(
                prepared["books"]["formal-1"]["telemetry_path"],
                f"telemetry/{_safe_book_id('formal-1')}.jsonl",
            )
            self.assertEqual(
                prepared["books"]["formal-1"]["telemetry_sha256"],
                sha256_bytes(b""),
            )
            self.assertEqual(FakeApplication.configs[-1].state_dir, str(state_root))
            self.assertEqual(client.calls, [])

    def _bundle(self) -> PreparationBundle:
        spec = PreparationSpec(
            schema_version=1,
            provider="fake",
            primary_model="fake:off",
            editor_model="fake:off",
            fast_model="fake:off",
            temperature=0.1,
            seed=None,
        )
        book = BookPreparation(
            book_id="formal-1",
            source_sha256="a" * 64,
            analysis={"voice": "plain"},
            style="plain",
            style_brief="plain",
            book_synopsis="synopsis",
            chapter_digests={"0": "digest"},
            source_digests=[ChapterSourceDigest(chapter_index=0, source_sha256="b" * 64)],
            glossary=[
                GlossaryPreparation(
                    source="Alice",
                    target="爱丽丝",
                    reading="Alice",
                    type="人物",
                    gender="female",
                    aliases=["A"],
                    first_chapter=0,
                    note="locked",
                    confidence="high",
                    locked=True,
                    status="ok",
                )
            ],
            node_fingerprints={"analyze": "c" * 64},
        )
        book = book.model_copy(
            update={
                "usage": usage_delta(
                    {
                        "totals": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        }
                    },
                    {},
                )
            }
        )
        provisional = PreparationBundle(
            schema_version=1,
            corpus_sha256="d" * 64,
            preparation_spec=spec,
            preparation_spec_sha256=sha256_bytes(
                canonical_json(spec.model_dump(mode="python")).encode("utf-8")
            ),
            preparation_sha256="0" * 64,
            books={"formal-1": book},
        )
        return provisional.model_copy(update={"preparation_sha256": _preparation_hash(provisional)})

    def _bundle_for_identity(self, source_sha256: str) -> PreparationBundle:
        bundle = self._bundle()
        book = bundle.books["formal-1"].model_copy(update={"source_sha256": source_sha256})
        changed = bundle.model_copy(update={"books": {"formal-1": book}})
        return changed.model_copy(update={"preparation_sha256": _preparation_hash(changed)})

    def test_complete_glossary_and_bundle_hash_are_preserved(self):
        bundle = self._bundle()
        source = preparation_source(bundle)
        term = source.book_for(book_id="formal-1", source_sha256="a" * 64).glossary[0]
        self.assertEqual(term, GlossaryTerm(**bundle.books["formal-1"].glossary[0].model_dump()))
        self.assertEqual(source.preparation_sha256, bundle.preparation_sha256)

    def test_frozen_identity_mismatch_is_permanent(self):
        bundle = self._bundle()
        source = preparation_source(bundle)
        with self.assertRaises(ValueError):
            source.book_for(book_id="formal-1", source_sha256="e" * 64)

    def test_bundle_hash_is_deterministic_and_excludes_raw_evidence(self):
        first = self._bundle()
        second = self._bundle()
        self.assertEqual(first.preparation_sha256, second.preparation_sha256)
        self.assertEqual(_preparation_hash(first), first.preparation_sha256)

        def assert_no_raw_evidence(value):
            forbidden_fields = {
                "prompt",
                "response",
                "request",
                "request_body",
                "response_body",
                "api_key",
                "apikey",
                "authorization",
            }
            if isinstance(value, dict):
                for key, nested in value.items():
                    self.assertNotIn(str(key).lower(), forbidden_fields)
                    assert_no_raw_evidence(nested)
            elif isinstance(value, list):
                for nested in value:
                    assert_no_raw_evidence(nested)
            elif isinstance(value, str):
                self.assertFalse(Path(value).is_absolute())

        assert_no_raw_evidence(first.model_dump(mode="python"))

    def test_preparation_hash_excludes_physical_usage_evidence(self):
        first = self._bundle()
        book = first.books["formal-1"].model_copy(
            update={
                "usage": {"schema_version": 2, "totals": {"input_tokens": 10}},
                "telemetry_sha256": "e" * 64,
                "telemetry_path": "telemetry/formal.jsonl",
            }
        )
        second = first.model_copy(update={"books": {"formal-1": book}})
        second = second.model_copy(update={"preparation_sha256": _preparation_hash(second)})
        self.assertEqual(first.preparation_sha256, second.preparation_sha256)

    def test_planner_and_node_fingerprints_match_and_bundle_hash_invalidates(self):
        bundle = self._bundle()
        source = preparation_source(bundle)
        book = source.book_for(book_id="formal-1", source_sha256="a" * 64)
        planner_fp = source.node_fingerprint(
            book=book,
            node_id="analyze",
            source_mapping=book.book_id,
            content=book.analysis,
        )
        node_fp = frozen_input_fingerprint(
            bundle.preparation_sha256,
            "analyze",
            book.book_id,
            book.analysis,
        )
        self.assertEqual(planner_fp, node_fp)
        invalidated = bundle.model_copy(update={"preparation_sha256": "f" * 64})
        self.assertNotEqual(
            preparation_source(invalidated).node_fingerprint(
                book=book,
                node_id="analyze",
                source_mapping=book.book_id,
                content=book.analysis,
            ),
            planner_fp,
        )

    def test_validate_recomputes_bundle_and_rejects_tamper(self):
        bundle = self._bundle()
        empty_telemetry_hash = sha256_bytes(b"")
        book = bundle.books["formal-1"].model_copy(
            update={
                "telemetry_sha256": empty_telemetry_hash,
                "telemetry_path": f"telemetry/{_safe_book_id('formal-1')}.jsonl",
            }
        )
        bundle = bundle.model_copy(update={"books": {"formal-1": book}})
        bundle = bundle.model_copy(update={"preparation_sha256": _preparation_hash(bundle)})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            freeze = {
                "schema_version": 1,
                "corpus_sha256": bundle.corpus_sha256,
                "book_spec_sha256": "a" * 64,
                "preparation_spec": bundle.preparation_spec.model_dump(mode="python"),
            }
            freeze["immutable_sha256"] = sha256_bytes(canonical_json(freeze).encode("utf-8"))
            (root / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
            freeze_state = {
                "status": "completed",
                "books": {"formal-1": {"status": "completed"}},
            }
            (root / "freeze_state.json").write_text(
                json.dumps(freeze_state),
                encoding="utf-8",
            )
            telemetry = root / book.telemetry_path
            telemetry.parent.mkdir(parents=True)
            telemetry.write_bytes(b"")
            export = root / "books" / f"{_safe_book_id('formal-1')}.json"
            export.parent.mkdir(parents=True)
            export.write_text(json.dumps(book.model_dump(mode="python")), encoding="utf-8")
            (root / "preparation.json").write_text(
                json.dumps(bundle.model_dump(mode="python"), ensure_ascii=False),
                encoding="utf-8",
            )
            usage_file = root / "usage" / f"{_safe_book_id('formal-1')}.json"
            usage_file.parent.mkdir(parents=True)
            usage_file.write_text(json.dumps(book.usage), encoding="utf-8")
            aggregate_usage = root / "usage.json"
            aggregate_usage.write_text(
                json.dumps({"formal-1": book.usage}),
                encoding="utf-8",
            )
            completion = {
                "schema_version": 1,
                "preparation_sha256": bundle.preparation_sha256,
                "preparation_path": "preparation.json",
                "preparation_file_sha256": sha256_bytes((root / "preparation.json").read_bytes()),
                "usage_path": "usage.json",
                "usage_file_sha256": sha256_bytes(aggregate_usage.read_bytes()),
                "books": {
                    "formal-1": {
                        "export_path": f"books/{_safe_book_id('formal-1')}.json",
                        "export_sha256": sha256_bytes(export.read_bytes()),
                        "usage_path": f"usage/{_safe_book_id('formal-1')}.json",
                        "usage_sha256": sha256_bytes(usage_file.read_bytes()),
                        "telemetry_path": book.telemetry_path,
                        "telemetry_sha256": empty_telemetry_hash,
                    }
                },
            }
            completion["completion_sha256"] = sha256_bytes(
                canonical_json(completion).encode("utf-8")
            )
            (root / "preparation_complete.json").write_text(
                json.dumps(completion),
                encoding="utf-8",
            )
            freeze_state["completion_sha256"] = completion["completion_sha256"]
            (root / "freeze_state.json").write_text(
                json.dumps(freeze_state),
                encoding="utf-8",
            )
            self.assertEqual(
                validate_preparation(root)["preparation_sha256"],
                bundle.preparation_sha256,
            )
            payload = bundle.model_dump(mode="python")
            tampered_usage = {"schema_version": 2, "totals": {"input_tokens": 11}}
            tampered_telemetry = b"tampered evidence\n"
            payload["books"]["formal-1"]["usage"] = tampered_usage
            payload["books"]["formal-1"]["telemetry_sha256"] = sha256_bytes(tampered_telemetry)
            (root / "preparation.json").write_text(json.dumps(payload), encoding="utf-8")
            export.write_text(json.dumps(payload["books"]["formal-1"]), encoding="utf-8")
            telemetry.write_bytes(tampered_telemetry)
            usage_file.write_text(json.dumps(tampered_usage), encoding="utf-8")
            aggregate_usage.write_text(json.dumps({"formal-1": tampered_usage}), encoding="utf-8")
            completion["preparation_file_sha256"] = sha256_bytes(
                (root / "preparation.json").read_bytes()
            )
            completion["usage_file_sha256"] = sha256_bytes(aggregate_usage.read_bytes())
            completion["books"]["formal-1"].update(
                {
                    "export_sha256": sha256_bytes(export.read_bytes()),
                    "usage_sha256": sha256_bytes(usage_file.read_bytes()),
                    "telemetry_sha256": sha256_bytes(tampered_telemetry),
                }
            )
            completion["completion_sha256"] = sha256_bytes(
                canonical_json(
                    {key: value for key, value in completion.items() if key != "completion_sha256"}
                ).encode("utf-8")
            )
            (root / "preparation_complete.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            with self.assertRaises(BenchmarkError):
                validate_preparation(root)

    def test_frozen_prepare_uses_prebuilt_document_without_preparation_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = root / "identity.json"
            identity.write_text('{"schema_version":1}\n', encoding="utf-8")
            identity_sha = sha256_bytes(identity.read_bytes())
            bundle = self._bundle_for_identity(identity_sha)
            document = Document(
                title="formal-1",
                source_lang="en",
                target_lang="zh",
                fmt="text",
                source_path=str(identity),
                chapters=[
                    Chapter(
                        index=0,
                        title="Chapter",
                        segments=[Segment(index=0, source="Alice walks home.")],
                    )
                ],
                meta={"benchmark_book_id": "formal-1", "source_sha256": identity_sha},
            )

            def unexpected_call(*args, **kwargs):
                raise AssertionError("frozen preparation must not call an LLM")

            config = Config(
                llm=LLMConfig(
                    provider="fake",
                    models=ModelRoles(primary="fake:off", editor="fake:off", fast="fake:off"),
                ),
                source_lang="en",
                target_lang="zh",
                state_dir=str(root / "state"),
            )
            client = FakeClient(handler=unexpected_call)
            _, store = Application(
                config, client=client, frozen_preparation=preparation_source(bundle)
            ).run_document_goal(document, str(identity), GOAL_PREPARE)
            self.assertEqual(client.calls, [])
            self.assertEqual(store.load_analysis()["voice"], "plain")
            self.assertEqual(store.load_analysis()["book_synopsis"], "synopsis")
            self.assertEqual(store.load_progress(0).source_digest, "digest")
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)

    def test_none_frozen_path_remains_unconfigured(self):
        config = Config(
            llm=LLMConfig(
                provider="fake",
                models=ModelRoles(primary="fake:off", editor="fake:off", fast="fake:off"),
            ),
            source_lang="en",
            target_lang="zh",
        )
        self.assertIsNone(
            Application(config, client=FakeClient(handler=lambda *a, **k: "{}")).frozen_preparation
        )

    def test_continuous_document_mapping_and_identity_are_exact(self):
        rows = [
            {
                "passage_id": "p0",
                "subset": "continuous",
                "book_id": "formal-1",
                "chapter_index": 0,
                "segments": [
                    {
                        "segment_id": "s0",
                        "index": 3,
                        "source": "one",
                        "kind": "text",
                        "cont": False,
                        "meta": {},
                    },
                    {
                        "segment_id": "s1",
                        "index": 4,
                        "source": "two",
                        "kind": "text",
                        "cont": True,
                        "meta": {"x": 1},
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "corpus.json").write_text(
                json.dumps({"corpus_sha256": "d" * 64}), encoding="utf-8"
            )
            (root / "source_manifest.json").write_text(
                json.dumps({"books": [{"book_id": "formal-1", "source_sha256": "a" * 64}]}),
                encoding="utf-8",
            )
            with patch(
                "trans_novel.benchmark.runner._load_corpus_rows", return_value=("d" * 64, rows)
            ):
                document, identity_path, mapping = build_continuous_document(
                    root,
                    benchmark_id="bench",
                    book_id="formal-1",
                    replicate=2,
                    identity_dir=root / "identities",
                    preparation=self._bundle(),
                )
            self.assertEqual(mapping, {"0": "0"})
            self.assertEqual(document.title, "bench_formal-1_r2")
            self.assertEqual(document.chapters[0].title, "Benchmark 1")
            self.assertEqual(document.chapters[0].segments[0].index, 0)
            self.assertEqual(document.chapters[0].segments[1].meta["original_segment_id"], "s1")
            sidecar = json.loads(Path(identity_path).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["chapter_mapping"], {"0": "0"})

    def test_synthetic_chapter_uses_original_chapter_number(self):
        bundle = self._bundle()
        book = bundle.books["formal-1"].model_copy(
            update={
                "chapter_digests": {"12": "digest12"},
                "source_digests": [ChapterSourceDigest(chapter_index=12, source_sha256="b" * 64)],
            }
        )
        bundle = bundle.model_copy(update={"books": {"formal-1": book}})
        rows = [
            {
                "passage_id": "p12",
                "subset": "continuous",
                "book_id": "formal-1",
                "chapter_index": 12,
                "segments": [{"segment_id": "s12", "index": 0, "source": "chapter twelve"}],
            }
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "corpus.json").write_text(
                json.dumps({"corpus_sha256": "d" * 64}), encoding="utf-8"
            )
            (root / "source_manifest.json").write_text(
                json.dumps({"books": [{"book_id": "formal-1", "source_sha256": "a" * 64}]}),
                encoding="utf-8",
            )
            with patch(
                "trans_novel.benchmark.runner._load_corpus_rows", return_value=("d" * 64, rows)
            ):
                document, _, mapping = build_continuous_document(
                    root,
                    benchmark_id="bench",
                    book_id="formal-1",
                    replicate=1,
                    identity_dir=root / "identities",
                    preparation=bundle,
                )
        self.assertEqual(mapping, {"0": "12"})
        self.assertEqual(document.chapters[0].meta["original_chapter_index"], 12)

    def test_raw_key_shares_primary_translation_across_editor_roles(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "bench",
                "provider": "fake",
                "fast_model": "fast-a:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {
                        "candidate_id": "control",
                        "primary_model": "primary:off",
                        "editor_model": None,
                    },
                    {
                        "candidate_id": "a",
                        "primary_model": "primary:off",
                        "editor_model": "editor-a:off",
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp:
            identity = Path(temp) / "identity.json"
            identity.write_text("{}", encoding="utf-8")
            document = Document(
                title="bench_formal-1_r1", source_lang="en", target_lang="zh", fmt="text"
            )
            key_a = FullRunner._raw_key(
                spec, "primary:off", "d" * 64, "e" * 64, document, str(identity), "formal-1", 1
            )
            changed = spec.model_copy(update={"fast_model": "fast-b:off"})
            key_b = FullRunner._raw_key(
                changed, "primary:off", "d" * 64, "e" * 64, document, str(identity), "formal-1", 1
            )
            self.assertEqual(key_a, key_b)

    def test_quality_policy_and_usage_delta_are_downstream_only(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "bench",
                "provider": "fake",
                "fast_model": "fast:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {
                        "candidate_id": "control",
                        "primary_model": "primary:off",
                        "editor_model": None,
                    },
                    {
                        "candidate_id": "a",
                        "primary_model": "primary:off",
                        "editor_model": "editor:off",
                    },
                ],
            }
        )
        config = FullRunner._config(
            spec, "primary:off", "editor:off", quality=True, state_dir="state"
        )
        self.assertTrue(config.pipeline.polish)
        self.assertTrue(config.pipeline.naturalize)
        self.assertEqual(config.pipeline.backtranslate_sample, 0.05)
        delta = usage_delta({}, {})
        self.assertEqual(delta["schema_version"], 2)
        self.assertEqual(delta["totals"]["prompt_tokens"], 0)
        self.assertEqual(delta["totals"]["completion_tokens"], 0)

    def test_full_runner_shares_raw_translation_and_runs_quality_branches(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "bench",
                "provider": "fake",
                "fast_model": "fake:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {
                        "candidate_id": "control",
                        "primary_model": "primary:off",
                        "editor_model": None,
                    },
                    {
                        "candidate_id": "editor-a",
                        "primary_model": "primary:off",
                        "editor_model": "editor-a:off",
                    },
                    {
                        "candidate_id": "editor-b",
                        "primary_model": "primary:off",
                        "editor_model": "editor-b:off",
                    },
                ],
            }
        )
        rows = [
            {
                "passage_id": "p0",
                "subset": "continuous",
                "book_id": "formal-1",
                "chapter_index": 0,
                "segments": [
                    {
                        "segment_id": "s0",
                        "index": 0,
                        "source": "Alice walks home.",
                        "kind": "text",
                        "cont": False,
                        "meta": {},
                    }
                ],
            }
        ]

        class TelemetryFakeClient(FakeClient):
            telemetry_sink = None

        client = TelemetryFakeClient(handler=routing_handler)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "corpus.json").write_text(
                json.dumps({"corpus_sha256": "d" * 64}), encoding="utf-8"
            )
            (corpus / "source_manifest.json").write_text(
                json.dumps({"books": [{"book_id": "formal-1", "source_sha256": "a" * 64}]}),
                encoding="utf-8",
            )
            bundle = self._bundle()
            with (
                patch(
                    "trans_novel.benchmark.runner._load_corpus_rows",
                    return_value=("d" * 64, rows),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_candidate_spec",
                    return_value=spec,
                ),
                patch(
                    "trans_novel.benchmark.runner.validate_candidate_capabilities",
                    return_value=GenerationOptions(temperature=0.1),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_preparation_bundle",
                    return_value=(bundle, bundle.preparation_sha256),
                ),
            ):
                result = FullRunner(client=client).run(
                    corpus, "candidates.yaml", "preparation", root / "run"
                )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["branch_count"], 2)
            self.assertEqual(
                sum(call["operation"] == "translate.batch" for call in client.calls),
                1,
            )
            self.assertTrue(any(call["agent"] == "editor" for call in client.calls))
            candidates = json.loads((root / "run" / "candidates.json").read_text())
            self.assertEqual(len(candidates), 2)
            self.assertEqual(
                {row["raw_artifact_id"] for row in candidates},
                {candidates[0]["raw_artifact_id"]},
            )
            self.assertEqual(len({row["branch_artifact_id"] for row in candidates}), 2)
            for row in candidates:
                self.assertIn("preparation", row["allocated_usage"])
                self.assertIn("raw", row["allocated_usage"])
                self.assertIn("branch_increment", row["allocated_usage"])
            frozen_node_keys = (
                "prepare",
                "analyze",
                "digest:0",
                "mine_terms",
                "name_terms",
                "book_synopsis",
                "translate:0",
            )
            raw_id = candidates[0]["raw_artifact_id"]
            raw_state = json.loads(
                (root / "run" / "raw" / raw_id / "bench_formal-1_r1" / "manifest.json").read_text()
            )
            for key in frozen_node_keys:
                with self.subTest(scope="raw", key=key):
                    self.assertEqual(
                        raw_state["nodes"][key]["status"],
                        "succeeded",
                        msg=raw_state["nodes"][key],
                    )
            for row in candidates:
                branch_state = json.loads(
                    (
                        root
                        / "run"
                        / "branches"
                        / row["branch_artifact_id"]
                        / "bench_formal-1_r1"
                        / "manifest.json"
                    ).read_text()
                )
                for key in frozen_node_keys:
                    with self.subTest(scope="branch", artifact=row["branch_artifact_id"], key=key):
                        self.assertEqual(
                            branch_state["nodes"][key]["status"],
                            "succeeded",
                            msg=branch_state["nodes"][key],
                        )
            self.assertEqual(
                sum(call["operation"] == "translate.batch" for call in client.calls),
                1,
            )
            calls = len(client.calls)
            with (
                patch(
                    "trans_novel.benchmark.runner._load_corpus_rows",
                    return_value=("d" * 64, rows),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_candidate_spec",
                    return_value=spec,
                ),
                patch(
                    "trans_novel.benchmark.runner.validate_candidate_capabilities",
                    return_value=GenerationOptions(temperature=0.1),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_preparation_bundle",
                    return_value=(bundle, bundle.preparation_sha256),
                ),
            ):
                FullRunner(client=client).run(
                    corpus, "candidates.yaml", "preparation", root / "run"
                )
            self.assertEqual(len(client.calls), calls)

    def test_full_run_completed_hash_is_read_only_and_resumable(self):
        spec = CandidateSpec.model_validate(
            {
                "schema_version": 1,
                "benchmark_id": "bench",
                "provider": "fake",
                "fast_model": "fast:off",
                "temperature": 0.1,
                "seed": None,
                "replicates": 1,
                "default_context_strategy": "c2",
                "candidates": [
                    {
                        "candidate_id": "control",
                        "primary_model": "primary:off",
                        "editor_model": None,
                    },
                ],
            }
        )
        bundle = self._bundle()
        spec_hash = sha256_bytes(canonical_json(spec.model_dump(mode="python")).encode("utf-8"))
        immutable = {
            "schema_version": 1,
            "run_mode": "full",
            "corpus_sha256": bundle.corpus_sha256,
            "preparation_sha256": bundle.preparation_sha256,
            "benchmark_id": spec.benchmark_id,
            "spec_sha256": spec_hash,
            "replicates": 1,
        }
        rows = [{"subset": "continuous", "book_id": "formal-1"}]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "run.json").write_text(json.dumps(immutable), encoding="utf-8")
            (out / "run_state.json").write_text(
                json.dumps({"status": "completed", "artifacts": {}}), encoding="utf-8"
            )
            (out / "candidates.json").write_text("[]", encoding="utf-8")
            (out / "actual_usage.json").write_text("{}", encoding="utf-8")
            candidates_bytes = (out / "candidates.json").read_bytes()
            with (
                patch(
                    "trans_novel.benchmark.runner._load_corpus_rows",
                    return_value=("d" * 64, rows),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_candidate_spec",
                    return_value=spec,
                ),
                patch(
                    "trans_novel.benchmark.runner.validate_candidate_capabilities",
                    return_value=GenerationOptions(temperature=0.1),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_preparation_bundle",
                    return_value=(bundle, bundle.preparation_sha256),
                ),
            ):
                result = FullRunner().run("corpus", "candidates", "preparation", out)
            self.assertEqual(result["branch_count"], 0)
            self.assertEqual((out / "candidates.json").read_bytes(), candidates_bytes)
            self.assertEqual(result["status"], "completed")
            (out / "run.json").write_text(
                json.dumps({**immutable, "replicates": 2}), encoding="utf-8"
            )
            with (
                patch(
                    "trans_novel.benchmark.runner._load_corpus_rows",
                    return_value=("d" * 64, rows),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_candidate_spec",
                    return_value=spec,
                ),
                patch(
                    "trans_novel.benchmark.runner.validate_candidate_capabilities",
                    return_value=GenerationOptions(temperature=0.1),
                ),
                patch(
                    "trans_novel.benchmark.runner.load_preparation_bundle",
                    return_value=(bundle, bundle.preparation_sha256),
                ),
                self.assertRaises(BenchmarkError),
            ):
                FullRunner().run("corpus", "candidates", "preparation", out)
        for marker in ("journal.json", "work.pending", "nested.tmp"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as temp:
                source = Path(temp) / "source"
                destination = Path(temp) / "destination"
                source.mkdir()
                (source / marker).touch()
                with self.assertRaises(ValueError):
                    clone_closed_runstore(str(source), str(destination))
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            destination = Path(temp) / "destination"
            source.mkdir()
            (source / ".run.lock").touch()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            clone_closed_runstore(str(source), str(destination))
            self.assertFalse((destination / ".run.lock").exists())
            self.assertEqual((destination / "manifest.json").read_text(), "{}")

    def test_replicates_three_allocate_preparation_once_per_candidate_book(self):
        temp, root, _, bundle, _, _ = self._completed_fixture(replicates=3)
        self.addCleanup(temp.cleanup)
        candidates = json.loads((root / "run" / "candidates.json").read_text())
        self.assertEqual(len(candidates), 6)
        for candidate_id in ("editor-a", "editor-b"):
            owned = [row for row in candidates if row["candidate_id"] == candidate_id]
            self.assertEqual(
                {row["preparation_allocation_id"] for row in owned},
                {
                    FullRunner._preparation_allocation_id(
                        candidate_id,
                        "formal-1",
                        bundle.preparation_sha256,
                    )
                },
            )
            self.assertEqual(
                [row["replicate"] for row in owned if row["allocated_usage"]["preparation"]],
                [1],
            )
            self.assertEqual(
                owned[0]["allocated_usage"]["preparation"],
                bundle.books["formal-1"].usage,
            )
            self.assertEqual(
                [row["allocated_usage"]["preparation"] for row in owned[1:]],
                [{}, {}],
            )
            self.assertEqual(len({row["raw_artifact_id"] for row in owned}), 3)
            self.assertEqual(len({row["branch_artifact_id"] for row in owned}), 3)

    def test_completed_duplicate_missing_and_extra_tuples_rejected(self):
        temp, root, spec, bundle, rows, _ = self._completed_fixture()
        self.addCleanup(temp.cleanup)
        path = root / "run" / "candidates.json"
        original = json.loads(path.read_text())
        for mutation in ("duplicate", "missing", "extra"):
            payload = json.loads(json.dumps(original))
            if mutation in {"duplicate", "missing"}:
                payload[1] = json.loads(json.dumps(payload[0]))
            else:
                payload[1]["candidate_id"] = "extra-candidate"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(mutation=mutation), self.assertRaises(BenchmarkError):
                self._resume_fixture(root, spec, bundle, rows)
            path.write_text(json.dumps(original), encoding="utf-8")

    def test_completed_usage_and_actual_usage_recomputed_from_artifacts(self):
        temp, root, spec, bundle, rows, _ = self._completed_fixture()
        self.addCleanup(temp.cleanup)
        candidates_path = root / "run" / "candidates.json"
        original_candidates = json.loads(candidates_path.read_text())
        tampered_candidates = json.loads(json.dumps(original_candidates))
        tampered_candidates[0]["allocated_usage"]["raw"]["totals"]["calls"] += 1
        candidates_path.write_text(json.dumps(tampered_candidates), encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            self._resume_fixture(root, spec, bundle, rows)
        candidates_path.write_text(json.dumps(original_candidates), encoding="utf-8")
        actual_path = root / "run" / "actual_usage.json"
        actual = json.loads(actual_path.read_text())
        actual["totals"]["calls"] += 1
        actual_path.write_text(json.dumps(actual), encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            self._resume_fixture(root, spec, bundle, rows)

    def test_completed_raw_branch_manifests_bind_keys_models_and_linkage(self):
        temp, root, spec, bundle, rows, _ = self._completed_fixture()
        self.addCleanup(temp.cleanup)
        candidate = json.loads((root / "run" / "candidates.json").read_text())[0]
        raw_manifest_path = root / "run" / "raw" / candidate["raw_artifact_id"] / "manifest.json"
        branch_manifest_path = (
            root / "run" / "branches" / candidate["branch_artifact_id"] / "manifest.json"
        )
        raw_original = json.loads(raw_manifest_path.read_text())
        branch_original = json.loads(branch_manifest_path.read_text())
        mutations = (
            ("raw-empty", raw_manifest_path, {}),
            ("branch-empty", branch_manifest_path, {}),
            ("raw-key", raw_manifest_path, {**raw_original, "artifact_key": "0" * 64}),
            (
                "raw-model",
                raw_manifest_path,
                {**raw_original, "primary_model": "other:off"},
            ),
            (
                "branch-link",
                branch_manifest_path,
                {**branch_original, "raw_artifact": "0" * 64},
            ),
            (
                "branch-model",
                branch_manifest_path,
                {**branch_original, "editor_model": "other:off"},
            ),
        )
        for name, path, value in mutations:
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(mutation=name), self.assertRaises(BenchmarkError):
                self._resume_fixture(root, spec, bundle, rows)
            raw_manifest_path.write_text(json.dumps(raw_original), encoding="utf-8")
            branch_manifest_path.write_text(json.dumps(branch_original), encoding="utf-8")

    def test_quality_findings_are_stage_separated_and_tamper_bound(self):
        def findings_handler(messages, agent, operation, json_mode):
            system = messages[0]["content"]
            if "译文审校" in system:
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "mistranslation",
                                "detail": "review finding",
                                "suggestion": "review suggestion",
                            }
                        ],
                        "reviewed_segments": 1,
                        "complete": True,
                    }
                )
            if "回译译者" in system:
                return json.dumps({"backtranslations": ["Alice walks 3 miles."]})
            if "翻译保真度核查员" in system:
                return json.dumps({"issues": [{"index": 0, "detail": "backtranslation finding"}]})
            return routing_handler(messages, agent, operation, json_mode)

        def deterministic_lint(*args, **kwargs):
            return [
                SimpleNamespace(
                    index=0,
                    type="fixture_lint",
                    detail="lint finding",
                )
            ]

        with (
            patch(
                "trans_novel.pipeline.nodes.quality.BacktranslateNode._select_indices",
                return_value=[0],
            ),
            patch(
                "trans_novel.pipeline.nodes.translate.lint.lint_targets",
                side_effect=deterministic_lint,
            ),
            patch(
                "trans_novel.agents.consistency.ConsistencyChecker.check",
                return_value=[{"chapter": 0, "index": 0, "detail": "consistency finding"}],
            ),
        ):
            temp, root, spec, bundle, rows, _ = self._completed_fixture(
                handler=findings_handler, source="Alice walks 3 miles."
            )
        self.addCleanup(temp.cleanup)
        candidates_path = root / "run" / "candidates.json"
        candidates = json.loads(candidates_path.read_text())
        candidate = candidates[0]
        stage = candidate["stage"][0]
        self.assertEqual(stage["review_findings"][0]["detail"], "review finding")
        self.assertEqual(stage["lint_findings"][0]["detail"], "lint finding")
        self.assertEqual(
            stage["backtranslation_findings"][0]["detail"],
            "backtranslation finding",
        )
        report = json.loads(
            (
                root
                / "run"
                / "branches"
                / candidate["branch_artifact_id"]
                / "bench_formal-1_r1"
                / "report.json"
            ).read_text()
        )
        self.assertEqual(report["consistency_issues"][0]["detail"], "consistency finding")
        stage["review_findings"] = []
        candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            self._resume_fixture(root, spec, bundle, rows)

    def test_completed_branch_telemetry_checks_prior_history_and_allows_review_fix(self):
        temp, root, spec, bundle, rows, _ = self._completed_fixture()
        self.addCleanup(temp.cleanup)
        candidates_path = root / "run" / "candidates.json"
        candidates = json.loads(candidates_path.read_text())
        candidate = candidates[0]
        telemetry_path = (
            root / "run" / "branches" / candidate["branch_artifact_id"] / "telemetry.jsonl"
        )
        original = telemetry_path.read_bytes()
        telemetry_path.write_bytes(original + b'{"operation":"translate.review_fix"}\n')
        candidate["branch_telemetry_sha256"] = sha256_bytes(telemetry_path.read_bytes())
        candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
        self._resume_fixture(root, spec, bundle, rows)
        telemetry_path.write_bytes(
            telemetry_path.read_bytes() + b'{"operation":"translate.batch"}\n'
        )
        candidate["branch_telemetry_sha256"] = sha256_bytes(telemetry_path.read_bytes())
        candidates_path.write_text(json.dumps(candidates), encoding="utf-8")
        with self.assertRaises(BenchmarkError):
            self._resume_fixture(root, spec, bundle, rows)

    def test_clone_refuses_held_stable_lock_without_replacing_inode(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            destination = Path(temp) / "destination"
            source.mkdir()
            (source / "manifest.json").write_text("{}", encoding="utf-8")
            lock_path = source / ".run.lock"
            lock_path.touch()
            inode = lock_path.stat().st_ino
            with lock_path.open("a+b") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(ValueError):
                    clone_closed_runstore(str(source), str(destination))
                fcntl.flock(held.fileno(), fcntl.LOCK_UN)
            self.assertEqual(lock_path.stat().st_ino, inode)
            self.assertFalse(destination.exists())

    def test_nonempty_completed_resume_is_byte_read_only_and_never_repairs_corruption(self):
        temp, root, spec, bundle, rows, _ = self._completed_fixture()
        self.addCleanup(temp.cleanup)
        run = root / "run"
        before = {
            path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()
        }
        self._resume_fixture(root, spec, bundle, rows)
        after = {
            path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()
        }
        self.assertEqual(after, before)
        actual_path = run / "actual_usage.json"
        actual_path.write_bytes(b"{}")
        with self.assertRaises(BenchmarkError):
            self._resume_fixture(root, spec, bundle, rows)
        self.assertEqual(actual_path.read_bytes(), b"{}")


if __name__ == "__main__":
    unittest.main()
