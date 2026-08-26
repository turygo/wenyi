"""翻译 agent 的对齐保证测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import re
import unittest
from typing import ClassVar

from tests.fake_llm import fake_llm_dict
from trans_novel.agents import prompts
from trans_novel.agents.translator import AlignmentError, Translator
from trans_novel.config import Config, ModelRef
from trans_novel.ingest.models import (
    EpubSegmentState,
    EpubTextSlot,
    Segment,
    assign_segment_translation,
)
from trans_novel.llm import FakeClient
from trans_novel.llm.errors import AllModelsFailedError
from trans_novel.pipeline.checks import count_aligned, length_flags


def _count_segments(user_content: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", user_content, re.M))


def _epub_segment() -> Segment:
    slots = [
        EpubTextSlot(
            id="slot-a",
            element_path=(0,),
            field="text",
            source_value="Alpha ",
            leading_whitespace="",
            trailing_whitespace=" ",
            source_core="Alpha",
        ),
        EpubTextSlot(
            id="slot-b",
            element_path=(0,),
            field="tail",
            source_value="Beta",
            source_core="Beta",
        ),
    ]
    state = EpubSegmentState(
        resource_href="OEBPS/ch.xhtml",
        resource_sha256="resource",
        block_path=(0,),
        block_fingerprint="block",
        parse_mode="xml",
        slots=slots,
        slot_contract_sha256="contract",
    )
    return Segment(
        index=0,
        source="Alpha Beta",
        resource_href="OEBPS/ch.xhtml",
        epub_state=state,
    )


class TestTranslatorEpubSlots(unittest.TestCase):
    def _config(self):
        config = Config.from_dict({"llm": fake_llm_dict()})
        config.source_lang = "en"
        config.pipeline.align_retry_limit = 1
        return config

    def test_translation_and_retry_return_ordered_slots(self):
        responses = iter(
            [
                {
                    "translations": [
                        {"slots": [{"id": "slot-b", "core": "乙"}, {"id": "slot-a", "core": "甲"}]}
                    ]
                },
                {
                    "translations": [
                        {"slots": [{"id": "slot-a", "core": "甲"}, {"id": "slot-b", "core": "乙"}]}
                    ]
                },
                {
                    "translations": [
                        {
                            "slots": [
                                {"id": "slot-a", "core": "改甲"},
                                {"id": "slot-b", "core": "改乙"},
                            ]
                        }
                    ]
                },
            ]
        )
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                next(responses), ensure_ascii=False
            )
        )
        segment = _epub_segment()
        translator = Translator(client, self._config())

        translated = translator.translate_batch(
            [segment.source],
            agent="translator",
            segments=[segment],
        )
        assign_segment_translation(segment, translated[0])
        self.assertEqual(
            translated[0],
            [("slot-a", "甲"), ("slot-b", "乙")],
        )
        self.assertEqual(segment.target, "甲 乙")

        retried = translator.retranslate_with_feedback(
            segment.source,
            feedback="保持槽位顺序",
            operation="translate.review_fix",
            segment=segment,
        )
        assign_segment_translation(segment, retried)
        self.assertEqual(
            retried,
            [("slot-a", "改甲"), ("slot-b", "改乙")],
        )
        self.assertEqual(segment.target, "改甲 改乙")


class TestTranslatorAlignment(unittest.TestCase):
    def _config(self):
        config = Config.from_dict({"llm": fake_llm_dict()})
        config.source_lang = "ja"
        config.pipeline.align_retry_limit = 1
        return config

    def test_happy_path_aligned(self):
        def handler(messages, agent, operation, json_mode):
            n = _count_segments(messages[-1]["content"])
            return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        t = Translator(client, self._config())
        out = t.translate_batch(["あ", "い", "う"], agent="translator")
        self.assertEqual(len(out), 3)
        self.assertEqual(out, ["译0", "译1", "译2"])
        # 路由归因：正文翻译显式走 translator Agent（translate.batch）
        self.assertTrue(client.calls)
        self.assertTrue(all(c["agent"] == "translator" for c in client.calls))
        self.assertTrue(all(c["operation"] == "translate.batch" for c in client.calls))

    def test_back_matter_light_agent_routing(self):
        """light 旁路调用显式传 light-translator Agent，agent 原样透传到调用记录。"""

        def handler(messages, agent, operation, json_mode):
            n = _count_segments(messages[-1]["content"])
            return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        t = Translator(client, self._config())
        out = t.translate_batch(
            ["あ", "い"], agent="light-translator", operation="translate.back_matter"
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(client.calls[0]["agent"], "light-translator")
        self.assertEqual(client.calls[0]["operation"], "translate.back_matter")

    def test_fallback_to_per_segment_on_mismatch(self):
        # 多段批次故意少返回一段；单段调用正常 → 触发逐段兜底
        def handler(messages, agent, operation, json_mode):
            n = _count_segments(messages[-1]["content"])
            trans = [f"译{i}" for i in range(n)]
            if n > 1:
                trans = trans[:-1]  # 故意制造段数不符
            return json.dumps({"translations": trans}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        t = Translator(client, self._config())
        out = t.translate_batch(["あ", "い", "う"], agent="translator")
        self.assertEqual(len(out), 3)  # 兜底后仍保证 1:1
        # 验证确实回退到了逐段（出现过 n==1 的调用）
        single_calls = [
            c for c in client.calls if _count_segments(c["messages"][-1]["content"]) == 1
        ]
        self.assertGreaterEqual(len(single_calls), 3)

    def test_empty_per_segment_fallback_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps({"translations": []})
        )
        translator = Translator(client, self._config())

        with self.assertRaisesRegex(Exception, "索引为 0 的段落"):
            translator.translate_batch(["あ", "い"], agent="translator")

    def test_non_string_translation_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"translations": [None]}
            )
        )
        translator = Translator(client, self._config())

        with self.assertRaisesRegex(Exception, "索引为 0 的段落"):
            translator.translate_batch(["あ"], agent="translator")

    def test_retranslate_batch_with_feedback_rejects_non_string_element(self):
        """translations 数组中包含 dict 元素时应视为无效并返回 []，不应经 str() 强制转换后采纳。"""
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"translations": ["译0", {"text": "译1"}]}, ensure_ascii=False
            )
        )
        translator = Translator(client, self._config())
        out = translator.retranslate_batch_with_feedback(
            [(0, "あ", "意见0"), (1, "い", "意见1")],
            ["旧译0", "旧译1"],
            operation="translate.review_fix",
        )
        self.assertEqual(out, [])

    def test_retranslate_batch_with_feedback_rejects_blank_element(self):
        """translations 数组含空串元素时视为无效，返回 []。"""
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"translations": ["译0", "  "]}, ensure_ascii=False
            )
        )
        translator = Translator(client, self._config())
        out = translator.retranslate_batch_with_feedback(
            [(0, "あ", "意见0"), (1, "い", "意见1")],
            ["旧译0", "旧译1"],
            operation="translate.review_fix",
        )
        self.assertEqual(out, [])

    def test_malformed_json_recovers_via_per_segment_fallback(self):
        """整批翻译返回无法解析的内容 → JSONParseError 触发整批重试，再由逐段兜底恢复一一对应。"""

        def handler(messages, agent, operation, json_mode):
            n = _count_segments(messages[-1]["content"])
            if n > 1:
                return "模型输出了不可解析的内容"
            return json.dumps({"translations": [f"译{i}" for i in range(n)]}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        out = translator.translate_batch(["あ", "い"], agent="translator")
        self.assertEqual(out, ["译0", "译0"])
        # align_retry_limit=1 → 2 次整批 + 逐段 2 次，全部计入逻辑调用
        self.assertEqual(len(client.calls), 4)

    def test_malformed_json_in_fallback_raises_alignment_with_stable_reason(self):
        """逐段兜底时仍返回无法解析的内容 → 包装为 AlignmentError，提供稳定 reason 并保留中文消息。"""
        client = FakeClient(handler=lambda messages, agent, operation, json_mode: "不可解析")
        translator = Translator(client, self._config())
        with self.assertRaises(AlignmentError) as caught:
            translator.translate_batch(["あ"], agent="translator")
        self.assertEqual(caught.exception.reason, "translation_segment_fallback_failed")
        self.assertRegex(str(caught.exception), "索引为 0 的段落")
        self.assertEqual(len(client.calls), 3)  # 2 次整批 + 1 次单段兜底

    def test_provider_permanent_error_bubbles_immediately(self):
        provider_error = RuntimeError("provider permanent")

        def handler(messages, agent, operation, json_mode):
            raise provider_error

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        with self.assertRaises(RuntimeError) as caught:
            translator.translate_batch(["あ", "い"], agent="translator")
        self.assertIs(caught.exception, provider_error, "必须原样向上抛出异常对象")
        self.assertEqual(len(client.calls), 1, "Provider 异常不得触发整批重试")

    def test_provider_retryable_error_bubbles_immediately(self):
        provider_error = TimeoutError("provider timeout")

        def handler(messages, agent, operation, json_mode):
            raise provider_error

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        with self.assertRaises(TimeoutError) as caught:
            translator.translate_batch(["あ", "い"], agent="translator")
        self.assertIs(caught.exception, provider_error, "必须原样向上抛出异常对象")
        self.assertEqual(len(client.calls), 1)

    def test_retry_exhaustion_bubbles_immediately(self):
        provider_error = AllModelsFailedError(((ModelRef("provider", "model"), "server_error"),))

        def handler(messages, agent, operation, json_mode):
            raise provider_error

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        with self.assertRaises(AllModelsFailedError) as caught:
            translator.translate_batch(["あ", "い"], agent="translator")
        self.assertIs(caught.exception, provider_error, "必须原样向上抛出异常对象")
        self.assertEqual(len(client.calls), 1)

    def test_ordinary_value_error_bubbles_immediately(self):
        business_error = ValueError("业务拒绝")

        def handler(messages, agent, operation, json_mode):
            raise business_error

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        with self.assertRaises(ValueError) as caught:
            translator.translate_batch(["あ", "い"], agent="translator")
        self.assertIs(caught.exception, business_error, "业务错误不得被吞掉，也不得触发重试")
        self.assertEqual(len(client.calls), 1)

    def test_provider_failure_during_fallback_bubbles_immediately(self):
        provider_error = AllModelsFailedError(((ModelRef("provider", "model"), "server_error"),))

        def handler(messages, agent, operation, json_mode):
            n = _count_segments(messages[-1]["content"])
            if n > 1:
                # 整批翻译故意少返回一段译文 → AlignmentError 触发整批重试和逐段兜底
                return json.dumps(
                    {"translations": [f"译{i}" for i in range(n - 1)]}, ensure_ascii=False
                )
            raise provider_error  # 第一个单段兜底调用：Provider 失败

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())
        with self.assertRaises(AllModelsFailedError) as caught:
            translator.translate_batch(["あ", "い", "う"], agent="translator")
        self.assertIs(caught.exception, provider_error, "兜底阶段的 Provider 异常必须原样向上抛出")
        # 2 次整批（少一段）+ 第 1 个单段调用即失败，不再继续兜底
        self.assertEqual(len(client.calls), 3)


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
    BLOCKS: ClassVar[list[str]] = [
        "【角色信息 / 风格指南】",
        "【全书概览】",
        "【本章梗概】",
        "【专有名词对照表】",
        "【前文译文",
    ]

    def _assert_block_order(self, rendered: str):
        for b in self.BLOCKS:
            self.assertIn(b, rendered, f"缺少块标题：{b}")
        for a, b in zip(self.BLOCKS, self.BLOCKS[1:], strict=False):
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
        common = {
            "src": "ja",
            "tgt": "zh",
            "style": "克制冷峻",
            "book_synopsis": "主线与人物关系。",
            "chapter_digest": "人物登场，情节推进。",
        }
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
        common = {
            "src": "ja",
            "tgt": "zh",
            "style": "克制冷峻",
            "book_synopsis": "主线与人物关系。",
            "chapter_digest": "人物登场，情节推进。",
        }
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
