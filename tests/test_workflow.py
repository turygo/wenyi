"""Minimal pipeline planning and preset contracts (offline)."""

from __future__ import annotations

import tempfile
import unittest

from trans_novel.config import PipelineConfig
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.pipeline.bootstrap import build_workflow_definition
from trans_novel.pipeline.contracts import GOAL_RUN_ALL
from trans_novel.pipeline.planner import Planner, PrescanInputs, WorkflowPolicy
from trans_novel.pipeline.runstore import RunStore
from trans_novel.pipeline.state import (
    NODE_ASSEMBLE,
    NODE_DETERMINISTIC_QA,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REPORT,
    NODE_TITLES,
    NODE_TRANSLATE,
    RunIdentity,
)


class TestPresets(unittest.TestCase):
    def test_exact_quality_contract(self):
        expected = {
            "economy": (False, "light"),
            "balanced": (False, "full"),
            "quality": (True, "full"),
        }
        for name, (polish, back_matter) in expected.items():
            policy = PipelineConfig.for_quality(name)
            self.assertEqual((policy.polish, policy.back_matter), (polish, back_matter))


class TestWorkflowDefinition(unittest.TestCase):
    def test_only_minimal_nodes_are_registered(self):
        self.assertEqual(
            set(build_workflow_definition().node_ids),
            {
                NODE_PREPARE,
                "analyze",
                NODE_MINE_TERMS,
                NODE_NAME_TERMS,
                NODE_TRANSLATE,
                NODE_POLISH,
                NODE_TITLES,
                NODE_DETERMINISTIC_QA,
                NODE_REPORT,
                NODE_ASSEMBLE,
            },
        )

    def test_body_chain_is_translate_then_optional_polish(self):
        definition = build_workflow_definition()
        self.assertEqual(definition.depends_on(NODE_POLISH), (NODE_TRANSLATE,))
        self.assertEqual(definition.depends_on(NODE_TITLES), (NODE_TRANSLATE, NODE_POLISH))


class TestPlanner(unittest.TestCase):
    @staticmethod
    def _store(tmp: str) -> RunStore:
        store = RunStore(tmp)
        doc = Document(
            title="Book",
            fmt="text",
            source_lang="en",
            target_lang="zh",
            source_path="source.txt",
            chapters=[
                Chapter(
                    index=0, title="Chapter", segments=[Segment(index=0, source="A paragraph.")]
                ),
                Chapter(index=1, title="Notes", segments=[Segment(index=0, source="Note.")]),
            ],
        )
        state = store.stage_document(
            doc, RunIdentity(source_bytes_sha256="source", source_lang="en", target_lang="zh")
        )
        state["initialized"] = True
        store.save_manifest(state)
        return store

    def test_disabled_polish_skip_is_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_RUN_ALL,
                store=store,
                policy=WorkflowPolicy(polish=False, back_matter="full"),
                prescan=PrescanInputs(),
            )
            body = [e for stage in plan.stages for e in stage.entries if e.ci == 0]
            self.assertEqual(
                [(e.node_id, e.action, e.finalize_chapter) for e in body],
                [(NODE_TRANSLATE, "run", False), (NODE_POLISH, "skip", True)],
            )

    def test_quality_adds_polish_after_translation(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_RUN_ALL,
                store=store,
                policy=WorkflowPolicy(polish=True, back_matter="full"),
                prescan=PrescanInputs(),
            )
            body = [e.node_id for stage in plan.stages for e in stage.entries if e.ci == 0]
            self.assertEqual(body, [NODE_TRANSLATE, NODE_POLISH])

    def test_plan_uses_only_registered_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_RUN_ALL, store=store, policy=WorkflowPolicy(), prescan=PrescanInputs()
            )
            keys = plan.entry_keys()
            allowed = {
                "prepare",
                "analyze",
                "mine_terms",
                "name_terms",
                "translate",
                "polish",
                "titles",
                "deterministic_qa",
                "report",
                "assemble",
            }
            self.assertTrue(all(key.split(":", 1)[0] in allowed for key in keys))
