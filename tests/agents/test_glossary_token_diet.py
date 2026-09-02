"""Offline glossary mining and naming regressions."""

from __future__ import annotations

import json
import tempfile
import unittest

from tests.fixtures.fake_llm import fake_llm_dict
from trans_novel.agents.namer import CastNamer
from trans_novel.config import Config
from trans_novel.glossary.miner import Candidate
from trans_novel.glossary.store import GlossaryStore
from trans_novel.llm import FakeClient


class TestGlossaryAgents(unittest.TestCase):
    def test_name_terms_uses_candidate_context_without_summary(self):
        seen = []

        def handler(messages, agent, operation, json_mode):
            seen.append((agent, operation, messages[-1]["content"]))
            return json.dumps({"terms": [{"source": "Alice", "target": "爱丽丝", "type": "人物"}]})

        with tempfile.TemporaryDirectory() as d:
            cfg = Config.from_dict({"llm": fake_llm_dict()})
            glossary = GlossaryStore(f"{d}/glossary.db")
            try:
                out = CastNamer(FakeClient(handler=handler), cfg).name_terms(
                    [Candidate(surface="Alice", count=2, contexts=["Alice entered the room."])],
                    analysis_brief="简洁",
                    existing=[],
                )
                self.assertEqual(out[0].target, "爱丽丝")
                self.assertEqual(seen[0][:2], ("analyst", "prescan.name_terms"))
                self.assertTrue(all(item[1].startswith("prescan.") for item in seen))
            finally:
                glossary.close()
