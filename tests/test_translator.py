"""翻译 agent 的对齐保证测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import re
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.agents.translator import AlignmentError, Translator
from trans_novel.config import Config, ModelRef
from trans_novel.llm import FakeClient
from trans_novel.llm.errors import AllModelsFailedError
from trans_novel.pipeline.checks import count_aligned, length_flags


def _count_segments(user_content: str) -> int:
    return len(re.findall(r"^\[(\d+)\]", user_content, re.M))


class TestTranslatorPlainContract(unittest.TestCase):
    def _config(self):
        config = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
        config.source_lang = "en"
        config.pipeline.align_retry_limit = 1
        return config

    def test_translation_and_feedback_return_plain_strings(self):
        responses = iter(
            [
                {"translations": ["甲乙"]},
                {"translations": ["改甲乙"]},
            ]
        )
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                next(responses), ensure_ascii=False
            )
        )
        translator = Translator(client, self._config())

        self.assertEqual(
            translator.translate_batch(["Alpha Beta"], agent="translator"),
            ["甲乙"],
        )
        self.assertEqual(
            translator.retranslate_with_feedback(
                "Alpha Beta",
                feedback="修正译文",
                operation="translate.lint_fix",
            ),
            "改甲乙",
        )
        self.assertNotIn("EPUB", client.calls[0]["messages"][-1]["content"])

    def test_non_linguistic_segment_preserves_source_without_call(self):
        client = FakeClient(handler=lambda *args: self.fail("LLM must not be called"))
        self.assertEqual(
            Translator(client, self._config()).translate_batch(["123"], agent="translator"),
            ["123"],
        )
        self.assertEqual(client.calls, [])


class TestTranslatorAlignment(unittest.TestCase):
    def _config(self):
        config = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
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
            if not json_mode:
                return "译文"
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
        single_calls = [call for call in client.calls if not call["json_mode"]]
        self.assertEqual(len(single_calls), 3)

    def test_empty_per_segment_fallback_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: (
                json.dumps({"translations": []}) if json_mode else ""
            )
        )
        translator = Translator(client, self._config())

        with self.assertRaisesRegex(Exception, "索引为 0 的段落"):
            translator.translate_batch(["あ", "い"], agent="translator")

    def test_abnormally_long_per_segment_fallback_is_rejected(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: (
                json.dumps({"translations": [None]}) if json_mode else "译" * 300
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
            if not json_mode:
                return "译0"
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
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: "不可解析" if json_mode else ""
        )
        translator = Translator(client, self._config())
        with self.assertRaises(AlignmentError) as caught:
            translator.translate_batch(["あ"], agent="translator")
        self.assertEqual(caught.exception.reason, "translation_segment_fallback_failed")
        self.assertRegex(str(caught.exception), "索引为 0 的段落")
        self.assertEqual(len(client.calls), 4)  # 2 次整批 + 2 次单段兜底

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


class TestTranslatorStrictSegmentContract(unittest.TestCase):
    def _config(self):
        config = Config.from_dict({"llm": fake_llm_dict(), "quality": "balanced"})
        config.source_lang = "en"
        config.pipeline.align_retry_limit = 1
        return config

    def test_each_source_has_its_own_strict_json_call(self):
        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            self.assertTrue(json_mode)
            self.assertNotEqual("Alpha" in user, "Beta" in user)
            return json.dumps(
                {"translation": "甲" if "Alpha" in user else "乙"},
                ensure_ascii=False,
            )

        client = FakeClient(handler=handler)
        result = Translator(client, self._config()).translate_batch(
            ["Alpha", "Beta", "123", "{var=dc:http_errors:rate10m,job=webserver}"],
            agent="translator",
        )

        self.assertEqual(
            result,
            ["甲", "乙", "123", "{var=dc:http_errors:rate10m,job=webserver}"],
        )
        self.assertEqual(len(client.calls), 2)

    def test_invalid_single_value_schema_retries(self):
        responses = iter(
            [
                {"translations": ["错误结构"]},
                {"translation": "正确译文", "explanation": "多余字段"},
                {"translation": "正确译文"},
            ]
        )
        config = self._config()
        config.pipeline.align_retry_limit = 2
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                next(responses), ensure_ascii=False
            )
        )

        self.assertEqual(
            Translator(client, config).translate_batch(["Alpha"], agent="translator"),
            ["正确译文"],
        )
        self.assertEqual(len(client.calls), 3)

    def test_contract_exhaustion_fails_closed(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"translations": ["错误结构"]}, ensure_ascii=False
            )
        )

        with self.assertRaises(AlignmentError) as caught:
            Translator(client, self._config()).translate_batch(["Alpha"], agent="translator")

        self.assertEqual(caught.exception.reason, "translation_segment_contract_failed")
        self.assertEqual(len(client.calls), 2)


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
