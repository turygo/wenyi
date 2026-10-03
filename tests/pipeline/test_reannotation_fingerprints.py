"""显式富文本迁移接受当前内容依赖，恢复全文时不重复付费。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from tests.fixtures.fake_llm import fake_llm_dict, routing_handler
from tests.pipeline.nodes import test_reannotate
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.llm import FakeClient
from trans_novel.pipeline import Application
from trans_novel.pipeline.composition.agents import AgentBundle
from trans_novel.pipeline.composition.context import RunContext
from trans_novel.pipeline.contracts import GOAL_RUN_ALL
from trans_novel.pipeline.planning import (
    WorkflowPolicy,
    build_prescan_inputs,
    content_fingerprints,
)
from trans_novel.pipeline.state import NODE_SUCCEEDED, NodeState


def _context(application, store):
    return RunContext(
        store=store,
        config=application.config,
        doc=None,
        agent_builder=lambda src, tgt: AgentBundle(
            client=application.client, config=application.config, src=src, tgt=tgt
        ),
    )


class TestReannotationFingerprints(unittest.TestCase):
    def _fixture(self, directory):
        path, store = test_reannotate.TestReannotateStore()._fixture(directory)
        state = store.load_state()
        state.title = "Book"
        state.identity.source_lang = state.source_lang = "en"
        state.identity.target_lang = state.target_lang = "zh"
        keys = ["prepare", "analyze", "mine_terms", "name_terms", "titles", "deterministic_qa"]
        keys += [
            f"{base}:{chapter.index}"
            for chapter in state.chapters
            for base in ("translate", "polish")
        ]
        keys += ["layout", "report", "assemble"]
        state.nodes = {
            key: NodeState(
                node_id=key.partition(":")[0],
                status=NODE_SUCCEEDED,
                input_fingerprint="policy3:" + key,
                output={"accepted": key},
            )
            for key in keys
        }
        state.nodes["polish:0"].status = "skipped"
        state.nodes["polish:0"].input_fingerprint = ""
        store.save_state(state)
        store.save_analysis({"genre": "finance", "tone": "plain", "style_guide": "concise"})
        GlossaryStore(store.glossary_path).close()
        config = Config.from_dict({"llm": fake_llm_dict(), "quality": "balanced"})
        config.source_lang = "en"
        config.target_lang = "zh"
        client = FakeClient(handler=routing_handler)
        application = Application(config, client=client)
        return path, store, application, os.path.join(directory, "migrated")

    def test_clone_rebases_only_successful_content_and_preserves_paid_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path, origin, application, destination = self._fixture(directory)
            original_state = origin.load_state().model_dump()
            original_targets = [
                segment.target
                for chapter in origin.load_state().chapters
                for segment in origin.load_chapter(chapter.index).segments
            ]
            analysis = origin.load_analysis()
            glossary = Path(origin.glossary_path).read_bytes()
            migrated = application.reannotate(origin, path, state_copy=destination)
            state = migrated.load_state()
            self.assertEqual(origin.load_state().model_dump(), original_state)
            self.assertEqual(migrated.load_analysis(), analysis)
            self.assertEqual(Path(migrated.glossary_path).read_bytes(), glossary)
            self.assertEqual(
                [
                    segment.target
                    for chapter in state.chapters
                    for segment in migrated.load_chapter(chapter.index).segments
                ],
                original_targets,
            )
            self.assertEqual(state.identity.translation_policy_version, 4)
            self.assertTrue(application.client.calls)
            self.assertEqual(
                {call["operation"] for call in application.client.calls}, {"richtext.annotate"}
            )
            self.assertEqual({call["agent"] for call in application.client.calls}, {"analyst"})
            context = _context(application, migrated)
            try:
                values = content_fingerprints(application.config, migrated, context)
                audit = state.meta["richtext_migration"]["fingerprint_rebase"]
                self.assertEqual(set(audit), set(values))
                for key, value in values.items():
                    self.assertEqual(state.nodes[key].input_fingerprint, value)
                    self.assertEqual(
                        state.nodes[key].status, original_state["nodes"][key]["status"]
                    )
                    self.assertEqual(
                        state.nodes[key].output, original_state["nodes"][key]["output"]
                    )
                    self.assertEqual(
                        audit[key],
                        {
                            "old": "policy3:" + key,
                            "new": value,
                            "audit": [{"old": "policy3:" + key, "new": value}],
                        },
                    )
                for key in ("layout", "report", "polish:0"):
                    self.assertEqual(state.nodes[key].model_dump(), original_state["nodes"][key])
                self.assertEqual(state.nodes["assemble"].status, "pending")
                policy = WorkflowPolicy.from_config(application.config)
                inputs = build_prescan_inputs(
                    application.config, migrated, policy, context, GOAL_RUN_ALL
                )
                self.assertEqual(
                    inputs.prepare_fingerprint(), state.nodes["prepare"].input_fingerprint
                )
                for chapter in state.chapters:
                    self.assertEqual(
                        inputs.translate_fingerprint(chapter.index),
                        state.nodes[f"translate:{chapter.index}"].input_fingerprint,
                    )
                plan = application.planner.build_plan(
                    goal=GOAL_RUN_ALL, store=migrated, policy=policy, prescan=inputs
                )
                self.assertFalse(
                    any(
                        entry.action == "run"
                        and entry.node_id
                        in {
                            "prepare",
                            "analyze",
                            "mine_terms",
                            "name_terms",
                            "translate",
                            "polish",
                            "titles",
                        }
                        for stage in plan.stages
                        for entry in stage.entries
                    )
                )
            finally:
                context.close()

    def test_repeat_reuses_annotations_and_does_not_append_fingerprint_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path, origin, application, destination = self._fixture(directory)
            migrated = application.reannotate(origin, path, state_copy=destination)
            before = migrated.load_state().meta["richtext_migration"]["fingerprint_rebase"]
            chapters = [
                migrated.load_chapter(chapter.index).to_dict()
                for chapter in migrated.load_state().chapters
            ]
            application.client.calls.clear()
            application.reannotate(origin, path, state_copy=destination)
            self.assertEqual(application.client.calls, [])
            self.assertEqual(
                migrated.load_state().meta["richtext_migration"]["fingerprint_rebase"], before
            )
            self.assertEqual(
                [
                    migrated.load_chapter(chapter.index).to_dict()
                    for chapter in migrated.load_state().chapters
                ],
                chapters,
            )

    def test_future_analyst_change_still_invalidates_normal_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path, origin, application, destination = self._fixture(directory)
            migrated = application.reannotate(origin, path, state_copy=destination)
            application.config.llm.models.analyst = ["fake/changed"]
            context = _context(application, migrated)
            try:
                policy = WorkflowPolicy.from_config(application.config)
                inputs = build_prescan_inputs(
                    application.config, migrated, policy, context, GOAL_RUN_ALL
                )
                self.assertNotEqual(
                    inputs.analyze_fingerprint(),
                    migrated.load_state().nodes["analyze"].input_fingerprint,
                )
                plan = application.planner.build_plan(
                    goal=GOAL_RUN_ALL, store=migrated, policy=policy, prescan=inputs
                )
                self.assertTrue(
                    any(
                        entry.action == "run" and entry.node_id == "analyze"
                        for stage in plan.stages
                        for entry in stage.entries
                    )
                )
                self.assertTrue(migrated.pending_chapters())
                self.assertEqual(
                    {call["operation"] for call in application.client.calls}, {"richtext.annotate"}
                )
            finally:
                context.close()

    def test_explicit_reacceptance_keeps_original_fingerprint_and_audits_each_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path, origin, application, destination = self._fixture(directory)
            migrated = application.reannotate(origin, path, state_copy=destination)
            first = migrated.load_state().nodes["analyze"].input_fingerprint
            application.config.llm.models.analyst = ["fake/accepted"]
            application.client.calls.clear()
            application.reannotate(origin, path, state_copy=destination)
            state = migrated.load_state()
            latest = state.nodes["analyze"].input_fingerprint
            record = state.meta["richtext_migration"]["fingerprint_rebase"]["analyze"]
            self.assertNotEqual(first, latest)
            self.assertEqual(application.client.calls, [])
            self.assertEqual(record["old"], "policy3:analyze")
            self.assertEqual(record["new"], latest)
            self.assertEqual(
                record["audit"],
                [{"old": "policy3:analyze", "new": first}, {"old": first, "new": latest}],
            )
            self.assertEqual(state.nodes["analyze"].status, NODE_SUCCEEDED)
