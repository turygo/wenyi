"""审校 / 润色 / 回译抽检 测试（离线）。"""

from __future__ import annotations

import json
import unittest

from trans_novel.agents.polisher import Polisher
from trans_novel.agents.reviewer import BackTranslator, Reviewer
from trans_novel.config import Config
from trans_novel.llm.base import FakeClient


def _cfg():
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
        }
    )


class TestReviewer(unittest.TestCase):
    def test_review_reports_issues(self):
        issues = {
            "issues": [
                {"index": 0, "type": "missing", "detail": "漏了后半句"},
                {"index": 1, "type": "terminology", "detail": "人名译法不符"},
            ]
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(issues, ensure_ascii=False))
        r = Reviewer(client, _cfg())
        out = r.review(["あ", "い"], ["甲", "乙"])
        self.assertEqual(len(out), 2)
        self.assertEqual(client.calls[-1]["tier"], "cheap")  # 审校走廉价档


class TestPolisher(unittest.TestCase):
    def test_polish_ok(self):
        client = FakeClient(
            handler=lambda m, t, j: json.dumps(
                {"polished": ["润色甲", "润色乙"]}, ensure_ascii=False
            )
        )
        p = Polisher(client, _cfg())
        out = p.polish(["甲", "乙"], ["a", "b"])
        self.assertEqual(out, ["润色甲", "润色乙"])
        self.assertEqual(client.calls[-1]["tier"], "strong")

    def test_polish_mismatch_keeps_original(self):
        client = FakeClient(
            handler=lambda m, t, j: json.dumps({"polished": ["只有一段"]}, ensure_ascii=False)
        )
        p = Polisher(client, _cfg())
        out = p.polish(["甲", "乙"], ["a", "b"])
        self.assertEqual(out, ["甲", "乙"])  # 段数不符 → 保守保留原译

    def test_polish_prompt_includes_source_for_fidelity(self):
        # 契约：润色必须把源文作为忠实度参照注入 prompt，且落在【源文对照】块内。
        client = FakeClient(
            handler=lambda m, t, j: json.dumps(
                {"polished": ["润色甲", "润色乙"]}, ensure_ascii=False
            )
        )
        p = Polisher(client, _cfg())
        # sources 用独特英文串，便于在 prompt 中定位
        p.polish(["甲", "乙"], sources=["ALPHA_SRC", "BETA_SRC"], style="S")

        messages = client.calls[-1]["messages"]
        user = messages[-1]["content"]
        i_src = user.index("【源文对照】")
        i_tgt = user.index("【待润色中文译文】")
        # 源文进了源文对照块：出现在【源文对照】之后、【待润色中文译文】之前
        for token in ("ALPHA_SRC", "BETA_SRC"):
            self.assertIn(token, user)
            self.assertLess(i_src, user.index(token))
            self.assertLess(user.index(token), i_tgt)
        # system 含逐段对照源文的忠实度铁律
        system = messages[0]["content"]
        self.assertIn("源文", system)
        self.assertTrue("不得遗漏" in system or "增改" in system)


class TestBackTranslator(unittest.TestCase):
    def test_check(self):
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            if "回译译者" in system:
                return json.dumps({"backtranslations": ["あ", "い"]}, ensure_ascii=False)
            if "保真度" in system:
                return json.dumps(
                    {"issues": [{"index": 1, "detail": "含义改变"}]}, ensure_ascii=False
                )
            return "{}"

        bt = BackTranslator(FakeClient(handler=handler), _cfg())
        issues = bt.check(["あ", "い"], ["甲", "乙"])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["index"], 1)


if __name__ == "__main__":
    unittest.main()
