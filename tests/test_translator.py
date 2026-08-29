"""翻译 agent 的对齐保证测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import re
import unittest

from tests.fake_llm import fake_llm_dict
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
            operation="translate.lint_fix",
            segment=segment,
        )
        assign_segment_translation(segment, retried)
        self.assertEqual(
            retried,
            [("slot-a", "改甲"), ("slot-b", "改乙")],
        )
        self.assertEqual(segment.target, "改甲 改乙")

    def test_non_linguistic_epub_segment_preserves_source_slots_without_call(self):
        slot = EpubTextSlot(
            id="slot-page",
            field="text",
            source_value="123",
            source_core="123",
        )
        segment = Segment(
            index=0,
            source="123",
            resource_href="OEBPS/ch.xhtml",
            epub_state=EpubSegmentState(
                resource_href="OEBPS/ch.xhtml",
                resource_sha256="resource",
                block_fingerprint="block",
                parse_mode="xml",
                slots=[slot],
                slot_contract_sha256="contract",
            ),
        )
        client = FakeClient(handler=lambda *args: self.fail("LLM must not be called"))

        translated = Translator(client, self._config()).translate_batch(
            [segment.source],
            agent="translator",
            segments=[segment],
        )

        self.assertEqual(translated, [[{"id": "slot-page", "core": "123"}]])
        self.assertEqual(client.calls, [])


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

    def test_non_linguistic_segments_are_preserved_and_excluded_from_prompt(self):
        def handler(messages, agent, operation, json_mode):
            self.assertEqual(_count_segments(messages[-1]["content"]), 1)
            return json.dumps({"translations": ["你好"]}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        translator = Translator(client, self._config())

        translated = translator.translate_batch(["-", "123", "Hello"], agent="translator")
        untouched = translator.translate_batch(["—", "2026"], agent="translator")

        self.assertEqual(translated, ["-", "123", "你好"])
        self.assertEqual(untouched, ["—", "2026"])
        self.assertEqual(len(client.calls), 1)

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
            operation="translate.lint_fix",
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
            operation="translate.lint_fix",
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
