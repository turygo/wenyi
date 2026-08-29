"""Polisher behavior tests (offline)."""

from __future__ import annotations

import json
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.agents.polisher import Polisher
from trans_novel.config import Config
from trans_novel.ingest.models import (
    EpubSegmentState,
    EpubTextSlot,
    Segment,
    assign_segment_translation,
)
from trans_novel.llm import FakeClient


def _cfg():
    config = Config.from_dict({"llm": fake_llm_dict()})
    config.source_lang = "ja"
    return config


def _epub_segment() -> Segment:
    slots = [
        EpubTextSlot(
            id="slot-a",
            element_path=(0,),
            field="text",
            source_value="Alpha ",
            trailing_whitespace=" ",
            source_core="Alpha",
        ),
        EpubTextSlot(
            id="slot-b", element_path=(0,), field="tail", source_value="Beta", source_core="Beta"
        ),
    ]
    return Segment(
        index=0,
        source="Alpha Beta",
        target="甲 乙",
        resource_href="OEBPS/ch.xhtml",
        epub_state=EpubSegmentState(
            resource_href="OEBPS/ch.xhtml",
            resource_sha256="resource",
            block_path=(0,),
            block_fingerprint="block",
            parse_mode="xml",
            slots=slots,
            slot_contract_sha256="contract",
        ),
    )


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

    def test_polish_preserves_ordered_epub_slots_and_target(self):
        client = FakeClient(
            handler=lambda m, a, o, j: json.dumps(
                {
                    "polished": [
                        {
                            "slots": [
                                {"id": "slot-a", "core": "润甲"},
                                {"id": "slot-b", "core": "润乙"},
                            ]
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        segment = _epub_segment()
        polished = Polisher(client, _cfg()).polish(
            [[{"id": "slot-a", "core": "甲"}, {"id": "slot-b", "core": "乙"}]],
            [segment.source],
            segments=[segment],
            strict=True,
        )
        assign_segment_translation(segment, polished[0])
        self.assertEqual(polished[0], [("slot-a", "润甲"), ("slot-b", "润乙")])
        self.assertEqual(segment.target, "润甲 润乙")

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
