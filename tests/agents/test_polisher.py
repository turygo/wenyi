"""Polisher behavior tests (offline)."""

from __future__ import annotations

import json
import unittest

from tests.fixtures.fake_llm import fake_llm_dict
from trans_novel.agents.polisher import Polisher
from trans_novel.config import Config
from trans_novel.llm import FakeClient


def _cfg():
    config = Config.from_dict({"llm": fake_llm_dict()})
    config.source_lang = "ja"
    return config


class TestPolisher(unittest.TestCase):
    def test_polishes_each_segment_independently(self):
        responses = iter(["润色甲", "润色乙"])
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(
                {"polished": [next(responses)]}, ensure_ascii=False
            )
        )

        out = Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"])

        self.assertEqual(out, ["润色甲", "润色乙"])
        self.assertEqual([call["operation"] for call in client.calls], ["polish.segment"] * 2)
        self.assertTrue(all(call["agent"] == "editor" for call in client.calls))

    def test_machine_literal_is_exact_and_skips_editor(self):
        client = FakeClient(handler=lambda *args: self.fail("machine literals must not call LLM"))

        literal = "{var=a--b}"
        self.assertEqual(
            Polisher(client, _cfg()).polish([literal], [literal], strict=True),
            [literal],
        )
        self.assertEqual(client.calls, [])

    def test_polish_uses_plain_text_contract(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps({"polished": ["润甲乙"]}, ensure_ascii=False)
        )
        polished = Polisher(client, _cfg()).polish(
            ["甲乙"],
            ["Alpha Beta"],
            strict=True,
        )
        self.assertEqual(polished, ["润甲乙"])
        self.assertNotIn("EPUB", client.calls[0]["messages"][-1]["content"])

    def test_invalid_non_strict_item_keeps_only_its_original(self):
        responses = iter([{"polished": []}, {"polished": ["润色乙"]}])
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(next(responses), ensure_ascii=False)
        )

        out = Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"])

        self.assertEqual(out, ["甲", "润色乙"])

    def test_invalid_strict_item_fails(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps({"polished": []}, ensure_ascii=False)
        )

        with self.assertRaisesRegex(Exception, "polish_item_invalid"):
            Polisher(client, _cfg()).polish(["甲"], ["a"], strict=True)

    def test_each_prompt_contains_only_its_source(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps({"polished": ["润色"]}, ensure_ascii=False)
        )

        Polisher(client, _cfg()).polish(["甲", "乙"], sources=["ALPHA_SRC", "BETA_SRC"], style="S")

        first = client.calls[0]["messages"][-1]["content"]
        second = client.calls[1]["messages"][-1]["content"]
        self.assertIn("ALPHA_SRC", first)
        self.assertNotIn("BETA_SRC", first)
        self.assertIn("BETA_SRC", second)
        self.assertNotIn("ALPHA_SRC", second)
