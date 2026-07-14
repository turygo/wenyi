"""翻译 agent 的对齐保证测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import re
import unittest

from trans_novel.agents import prompts
from trans_novel.agents.translator import Translator
from trans_novel.config import Config
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.checks import count_aligned, length_flags


def _count_segments(user_content: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", user_content, re.M))


class TestTranslatorAlignment(unittest.TestCase):
    def _config(self):
        return Config.from_dict(
            {
                "language": {"source": "ja", "target": "zh"},
                "llm": {
                    "provider": "fake",
                    "tiers": {
                        "strong": {"model": "deepseek-v4-pro"},
                        "cheap": {"model": "deepseek-v4-flash"},
                    },
                },
                "pipeline": {"align_retry_limit": 1},
            }
        )

    def test_happy_path_aligned(self):
        def handler(messages, tier, json_mode):
            n = _count_segments(messages[-1]["content"])
            return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

        t = Translator(FakeClient(handler=handler), self._config())
        out = t.translate_batch(["あ", "い", "う"])
        self.assertEqual(len(out), 3)
        self.assertEqual(out, ["译0", "译1", "译2"])

    def test_fallback_to_per_segment_on_mismatch(self):
        # 多段批次故意少返回一段；单段调用正常 → 触发逐段兜底
        def handler(messages, tier, json_mode):
            n = _count_segments(messages[-1]["content"])
            trans = [f"译{i}" for i in range(n)]
            if n > 1:
                trans = trans[:-1]  # 故意制造段数不符
            return json.dumps({"translations": trans}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        t = Translator(client, self._config())
        out = t.translate_batch(["あ", "い", "う"])
        self.assertEqual(len(out), 3)  # 兜底后仍保证 1:1
        # 验证确实回退到了逐段（出现过 n==1 的调用）
        single_calls = [
            c for c in client.calls if _count_segments(c["messages"][-1]["content"]) == 1
        ]
        self.assertGreaterEqual(len(single_calls), 3)

    def test_empty_per_segment_fallback_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, tier, json_mode: json.dumps({"translations": []})
        )
        translator = Translator(client, self._config())

        with self.assertRaisesRegex(Exception, "第 0 段失败"):
            translator.translate_batch(["あ", "い"])

    def test_non_string_translation_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, tier, json_mode: json.dumps({"translations": [None]})
        )
        translator = Translator(client, self._config())

        with self.assertRaisesRegex(Exception, "第 0 段失败"):
            translator.translate_batch(["あ"])


class TestTranslatorPromptOrder(unittest.TestCase):
    def test_static_chapter_digest_precedes_dynamic_glossary(self):
        for template in (prompts.TRANSLATOR_USER, prompts.TRANSLATOR_FIX_USER):
            self.assertLess(
                template.template.index("【本章梗概】"),
                template.template.index("【专有名词对照表】"),
            )


class TestPromptBlockOrder(unittest.TestCase):
    """翻译提示词块序契约：恒定块（风格→全书概览→本章梗概）在前，
    易变块（术语表→前文译文→待译段）在后。顺序被无意打破会破坏
    provider 侧前缀缓存命中（恒定前缀必须逐字节一致且位于开头）。"""

    # 按契约顺序排列的块标题（前缀匹配：fix 模板的前文块标题无「（最近）」后缀）
    BLOCKS = [
        "【角色信息 / 风格指南】",
        "【全书概览】",
        "【本章梗概】",
        "【专有名词对照表】",
        "【前文译文",
    ]

    def _assert_block_order(self, rendered: str):
        for b in self.BLOCKS:
            self.assertIn(b, rendered, f"缺少块标题：{b}")
        for a, b in zip(self.BLOCKS, self.BLOCKS[1:]):
            self.assertLess(
                rendered.index(a), rendered.index(b), f"块序逆转：{a} 必须出现在 {b} 之前"
            )

    def test_translator_user_block_order(self):
        out = prompts.render(
            "translator_user",
            src="ja",
            tgt="zh",
            style="克制冷峻",
            book_synopsis="主线与人物关系。",
            chapter_digest="人物登场，情节推进。",
            glossary="- 綾小路 → 绫小路",
            context="上一批译文。",
            n=1,
            n_minus_1=0,
            numbered_source="[0] 原文",
        )
        self._assert_block_order(out)

    def test_translator_fix_user_block_order(self):
        out = prompts.render(
            "translator_fix_user",
            src="ja",
            tgt="zh",
            style="克制冷峻",
            book_synopsis="主线与人物关系。",
            chapter_digest="人物登场，情节推进。",
            glossary="- 綾小路 → 绫小路",
            context_before="前文译文。",
            context_after="后文译文。",
            feedback="漏了一句",
            source="原文",
        )
        self._assert_block_order(out)

    # 第一个易变块标题：恒定前缀 = 它之前的全部内容
    VOLATILE_HDR = "【专有名词对照表】"
    FIRST_HDR = "【角色信息 / 风格指南】"

    def test_translator_user_constant_prefix_byte_identical(self):
        # 前缀缓存契约：仅易变输入（术语表/前文/待译段）变化时，恒定前缀
        # （风格→全书概览→本章梗概）必须逐字节一致且位于最开头——这才是
        # provider 前缀缓存命中的前提。相对块序正确并不保证前缀逐字节稳定：
        # 若把任一易变块挪到恒定块之前，两次渲染的前缀就会因易变输入不同而不等。
        common = dict(
            src="ja",
            tgt="zh",
            style="克制冷峻",
            book_synopsis="主线与人物关系。",
            chapter_digest="人物登场，情节推进。",
        )
        a = prompts.render(
            "translator_user",
            **common,
            glossary="- 綾小路 → 绫小路",
            context="上一批译文。",
            n=1,
            n_minus_1=0,
            numbered_source="[0] 原文A",
        )
        b = prompts.render(
            "translator_user",
            **common,
            glossary="- 堀北 → 堀北\n- 一之瀬 → 一之濑",
            context="完全不同的前文批次。",
            n=2,
            n_minus_1=1,
            numbered_source="[0] 原文B\n[1] 原文C",
        )

        pa = a[: a.index(self.VOLATILE_HDR)]
        pb = b[: b.index(self.VOLATILE_HDR)]
        # 载荷断言：易变输入全变、恒定输入不变时，前缀仍逐字节一致
        self.assertEqual(pa, pb, "易变输入变化时恒定前缀必须逐字节一致（前缀缓存命中）")

        # 前缀确实携带三段恒定内容
        for content in ("克制冷峻", "主线与人物关系。", "人物登场，情节推进。"):
            self.assertIn(content, pa, f"恒定前缀应含：{content}")
        # 第一块标题就在最开头，且领先所有易变内容
        self.assertEqual(a.index(self.FIRST_HDR), 0, "第一块标题必须位于提示词最开头")
        self.assertLess(
            a.index(self.FIRST_HDR), a.index(self.VOLATILE_HDR), "恒定首块必须领先所有易变块"
        )

    def test_translator_fix_user_constant_prefix_byte_identical(self):
        # fix 模板同理：恒定块（风格/概览/梗概）在前，易变块（术语表/前后文/
        # 审校意见/待重译段）在后；仅易变输入变化时恒定前缀必须逐字节一致。
        common = dict(
            src="ja",
            tgt="zh",
            style="克制冷峻",
            book_synopsis="主线与人物关系。",
            chapter_digest="人物登场，情节推进。",
        )
        a = prompts.render(
            "translator_fix_user",
            **common,
            glossary="- 綾小路 → 绫小路",
            context_before="前文A。",
            context_after="后文A。",
            feedback="漏了一句",
            source="原文A",
        )
        b = prompts.render(
            "translator_fix_user",
            **common,
            glossary="- 堀北 → 堀北",
            context_before="前文B完全不同。",
            context_after="后文B完全不同。",
            feedback="人称错了",
            source="原文B",
        )

        pa = a[: a.index(self.VOLATILE_HDR)]
        pb = b[: b.index(self.VOLATILE_HDR)]
        self.assertEqual(pa, pb, "fix 模板同样要求恒定前缀逐字节一致")

        for content in ("克制冷峻", "主线与人物关系。", "人物登场，情节推进。"):
            self.assertIn(content, pa, f"恒定前缀应含：{content}")
        self.assertEqual(a.index(self.FIRST_HDR), 0, "第一块标题必须位于提示词最开头")
        self.assertLess(
            a.index(self.FIRST_HDR), a.index(self.VOLATILE_HDR), "恒定首块必须领先所有易变块"
        )


class TestChecks(unittest.TestCase):
    def test_count_aligned(self):
        self.assertTrue(count_aligned(["a", "b"], ["甲", "乙"]))
        self.assertFalse(count_aligned(["a", "b"], ["甲"]))

    def test_length_flags(self):
        sources = ["これは長い日本語の文章です。" * 3, "短い", "x" * 10]
        targets = ["", "短い但正常的中文译文内容", "x" * 40]
        flags = length_flags(sources, targets)
        kinds = {f.index: f.reason for f in flags}
        self.assertEqual(kinds.get(0), "empty")  # 译文为空
        self.assertEqual(kinds.get(2), "too_long")  # 比值过大


if __name__ == "__main__":
    unittest.main()
