"""Back-matter policy regressions for the minimal pipeline."""

from __future__ import annotations

import tempfile
import unittest

from trans_novel.config import PipelineConfig
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.bootstrap import build_workflow_definition
from trans_novel.pipeline.contracts import GOAL_RUN_ALL
from trans_novel.pipeline.planner import Planner, PrescanInputs, WorkflowPolicy
from trans_novel.pipeline.runstore import RunStore
from trans_novel.pipeline.state import NODE_TRANSLATE, RunIdentity


class TestBackMatterPolicy(unittest.TestCase):
    def test_economy_defaults_light(self):
        self.assertEqual(PipelineConfig.for_quality("economy").back_matter, "light")

    def test_classifier_only_marks_obvious_back_matter(self):
        self.assertTrue(is_back_matter("索引", index=1, total=2))
        self.assertFalse(is_back_matter("第一章", index=0, total=2))

    def test_light_backmatter_uses_translate_without_polish(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(d)
            doc = Document(
                title="Book",
                fmt="text",
                source_lang="en",
                target_lang="zh",
                source_path="book.txt",
                chapters=[
                    Chapter(index=0, title="Index", segments=[Segment(index=0, source="Note.")]),
                ],
            )
            state = store.stage_document(
                doc, RunIdentity(source_bytes_sha256="x", source_lang="en", target_lang="zh")
            )
            state["initialized"] = True
            store.save_manifest(state)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_RUN_ALL,
                store=store,
                policy=WorkflowPolicy(polish=False, back_matter="light"),
                prescan=PrescanInputs(),
            )
            entries = [e for stage in plan.stages for e in stage.entries if e.ci == 0]
            self.assertEqual([e.node_id for e in entries], [NODE_TRANSLATE])
