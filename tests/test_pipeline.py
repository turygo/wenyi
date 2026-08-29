"""Offline regressions for the minimal translation pipeline."""

from __future__ import annotations

import tempfile
import unittest

from tests.fake_llm import fake_llm_dict, routing_handler
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.llm import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.contracts import GOAL_RUN_ALL, ExecutionGoal, NodeRequest
from trans_novel.pipeline.nodes.finish import DeterministicQANode
from trans_novel.pipeline.runstore import RunStore
from trans_novel.pipeline.state import (
    NODE_DETERMINISTIC_QA,
    NODE_REPORT,
    NODE_SUCCEEDED,
    RunIdentity,
)


def _config(state_dir: str, *, quality: str = "balanced") -> Config:
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": quality})
    config.source_lang = "en"
    config.target_lang = "zh"
    config.state_dir = state_dir
    return config


def _document() -> Document:
    return Document(
        title="Book",
        fmt="text",
        source_lang="en",
        target_lang="zh",
        source_path="book.txt",
        chapters=[
            Chapter(
                index=0,
                title="Chapter",
                segments=[
                    Segment(index=0, source="A complete sentence."),
                    Segment(index=1, source="An interior segment."),
                ],
            )
        ],
    )


class TestMinimalPipeline(unittest.TestCase):
    def test_balanced_emits_no_polish_or_legacy_operations(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient(handler=routing_handler)
            result = Application(_config(d), client=client).run_all(
                _write_source(d), out_format="txt"
            )
            operations = {call["operation"] for call in client.calls}
            self.assertNotIn("polish.batch", operations)
            self.assertTrue(
                operations
                <= {
                    "language.detect",
                    "analyzer.analyze",
                    "prescan.term_mine",
                    "prescan.name_terms",
                    "translate.batch",
                    "translate.lint_fix",
                    "translate.back_matter",
                    "title.translate",
                    "report",
                }
            )
            state = result["store"].load_state()
            for node_id in (
                "prepare",
                "analyze",
                "mine_terms",
                "name_terms",
                "translate:0",
                "titles",
                "deterministic_qa",
                "report",
                "assemble",
            ):
                self.assertIsNotNone(state.nodes[node_id].started_at)
                self.assertIsNotNone(state.nodes[node_id].finished_at)

    def test_identical_quality_run_keeps_qa_and_report_fingerprints(self):
        with tempfile.TemporaryDirectory() as d:
            source = _write_source(d)
            app = Application(_config(d), client=FakeClient(handler=routing_handler))
            goal = ExecutionGoal(name="run_all", phases=GOAL_RUN_ALL.phases, out_format="txt")
            doc = _document()
            _, store = app.run_document_goal(doc, source, goal)
            first = store.load_state()
            first_fingerprints = {
                node_id: first.nodes[node_id].input_fingerprint
                for node_id in (NODE_DETERMINISTIC_QA, NODE_REPORT)
            }
            _, store = app.run_document_goal(doc, source, goal)
            second = store.load_state()
            for node_id, fingerprint in first_fingerprints.items():
                self.assertEqual(second.nodes[node_id].status, NODE_SUCCEEDED)
                self.assertEqual(second.nodes[node_id].input_fingerprint, fingerprint)

    def test_two_segment_light_backmatter_resume_keeps_targets(self):
        with tempfile.TemporaryDirectory() as d:
            source = _write_source(d)
            doc = Document(
                title="Backmatter",
                fmt="text",
                source_lang="en",
                target_lang="zh",
                source_path="book.txt",
                chapters=[
                    Chapter(
                        index=0,
                        title="Index",
                        segments=[
                            Segment(index=0, source="First note."),
                            Segment(index=1, source="Second note."),
                        ],
                    )
                ],
            )
            app = Application(
                _config(d, quality="economy"), client=FakeClient(handler=routing_handler)
            )
            goal = ExecutionGoal(name="run_all", phases=GOAL_RUN_ALL.phases, out_format="txt")
            _, store = app.run_document_goal(doc, source, goal)
            first_targets = [segment.target for segment in store.load_chapter(0).segments]
            with open(store.event_log_path, encoding="utf-8") as stream:
                event_start = stream.read()
            _, store = app.run_document_goal(doc, source, goal)
            second_targets = [segment.target for segment in store.load_chapter(0).segments]
            with open(store.event_log_path, encoding="utf-8") as stream:
                event_end = stream.read()
            self.assertEqual(second_targets, first_targets)
            self.assertNotIn("translate_invalidated", event_end[len(event_start) :])

    def test_quality_adds_only_polish(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient(handler=routing_handler)
            Application(_config(d, quality="quality"), client=client).run_all(
                _write_source(d), out_format="txt"
            )
            operations = [call["operation"] for call in client.calls]
            self.assertIn("polish.batch", operations)
            self.assertTrue(
                set(operations)
                <= {
                    "language.detect",
                    "analyzer.analyze",
                    "prescan.term_mine",
                    "prescan.name_terms",
                    "translate.batch",
                    "translate.lint_fix",
                    "translate.back_matter",
                    "title.translate",
                    "polish.batch",
                    "polish.segment",
                    "report",
                }
            )

    def test_polish_disabled_body_chapter_reaches_done(self):
        with tempfile.TemporaryDirectory() as d:
            store = Application(_config(d), client=FakeClient(handler=routing_handler)).run(
                _write_source(d)
            )
            self.assertEqual(store.load_progress(0).status, "done")
            self.assertEqual(store.load_chapter(0).segments[0].target, "译0")

    def test_term_mining_and_naming_have_no_summary_operation(self):
        with tempfile.TemporaryDirectory() as d:
            client = FakeClient(handler=routing_handler)
            Application(_config(d), client=client).prepare_for_translation(_write_source(d))
            operations = {call["operation"] for call in client.calls}
            self.assertTrue({"prescan.term_mine", "prescan.name_terms"} <= operations)
            self.assertTrue(
                all(op.startswith(("language.", "analyzer.", "prescan.")) for op in operations)
            )

    def test_deterministic_qa_scans_interior_without_llm_or_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(d)
            doc = _document()
            state = store.stage_document(
                doc, RunIdentity(source_bytes_sha256="x", source_lang="en", target_lang="zh")
            )
            state["initialized"] = True
            store.save_manifest(state)
            progress = store.load_progress(0)
            progress.status = "done"
            store.save_progress(0, progress)
            chapter = store.load_chapter(0)
            chapter.segments[0].target = "A complete sentence."
            chapter.segments[1].target = "wrong"
            store.save_chapter(chapter)
            before = [s.target for s in chapter.segments]
            glossary = GlossaryStore(store.glossary_path)
            glossary.upsert_term(GlossaryTerm(source="interior", target="内部", locked=True))
            try:
                client = FakeClient(handler=routing_handler)
                node = DeterministicQANode(glossary=glossary)
                outcome = node.execute(
                    NodeRequest(
                        store=store,
                        node_id=NODE_DETERMINISTIC_QA,
                        key=NODE_DETERMINISTIC_QA,
                        ci=None,
                        scope="book",
                        input_path="book.txt",
                    )
                )
                self.assertTrue(
                    any(
                        issue["chapter"] == 0 and issue["index"] == 1
                        for issue in outcome.artifacts["issues"]
                    )
                )
                self.assertEqual([s.target for s in store.load_chapter(0).segments], before)
                self.assertEqual(client.calls, [])
            finally:
                glossary.close()


def _write_source(directory: str) -> str:
    path = f"{directory}/book.txt"
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("Chapter\n\nA complete sentence.\n\nAn interior segment.")
    return path
