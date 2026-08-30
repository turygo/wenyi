"""Polisher behavior tests (offline)."""

from __future__ import annotations

import json
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.agents.polisher import Polisher
from trans_novel.config import Config
from trans_novel.llm import FakeClient


def _cfg():
    config = Config.from_dict({"llm": fake_llm_dict()})
    config.source_lang = "ja"
    return config


class TestPolisher(unittest.TestCase):
    def test_polish_ok(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(
                {"polished": ["润色甲", "润色乙"]}, ensure_ascii=False
            )
        )
        out = Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"])
        self.assertEqual(out, ["润色甲", "润色乙"])
        self.assertEqual(client.calls[-1]["operation"], "polish.batch")
        self.assertEqual(client.calls[-1]["agent"], "editor")

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

    def test_polish_mismatch_keeps_original(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps({"polished": ["只有一段"]}, ensure_ascii=False)
        )
        self.assertEqual(Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"]), ["甲", "乙"])

    def test_polish_strict_retries_count_mismatch(self):
        responses = iter([{"polished": ["只有一段"]}, {"polished": ["润色甲", "润色乙"]}])
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(next(responses), ensure_ascii=False)
        )
        out = Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"], strict=True)
        self.assertEqual(out, ["润色甲", "润色乙"])
        self.assertEqual(len(client.calls), 2)
        self.assertIn("exactly 2 strings", client.calls[-1]["messages"][-1]["content"])

    def test_polish_strict_recovers_each_segment(self):
        responses = iter(
            [
                {"polished": ["批次段数错误"]},
                {"polished": ["重试仍错误"]},
                {"polished": ["润色甲"]},
                {"polished": ["润色乙"]},
            ]
        )
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(next(responses), ensure_ascii=False)
        )
        out = Polisher(client, _cfg()).polish(["甲", "乙"], ["a", "b"], strict=True)
        self.assertEqual(out, ["润色甲", "润色乙"])
        self.assertEqual(
            [c["operation"] for c in client.calls],
            ["polish.batch", "polish.batch", "polish.segment", "polish.segment"],
        )

    def test_polish_prompt_includes_source_for_fidelity(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(
                {"polished": ["润色甲", "润色乙"]}, ensure_ascii=False
            )
        )
        Polisher(client, _cfg()).polish(["甲", "乙"], sources=["ALPHA_SRC", "BETA_SRC"], style="S")
        messages = client.calls[-1]["messages"]
        user = messages[-1]["content"]
        i_src, i_tgt = user.index("【源文对照】"), user.index("【待润色中文译文】")
        for token in ("ALPHA_SRC", "BETA_SRC"):
            self.assertLess(i_src, user.index(token))
            self.assertLess(user.index(token), i_tgt)
        system = messages[0]["content"]
        self.assertIn("源文", system)
        self.assertTrue("不得遗漏" in system or "增改" in system)
