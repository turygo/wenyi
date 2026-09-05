"""模型语言检测测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.fixtures.books import write_sample_txt
from tests.fixtures.fake_llm import fake_llm_dict, routing_handler
from trans_novel.config import Config
from trans_novel.llm import FakeClient
from trans_novel.pipeline import Application


class TestModelLanguageDetection(unittest.TestCase):
    def _cfg(self, state: str) -> Config:
        config = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
        config.state_dir = state
        return config

    def test_auto_uses_model_detection(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = self._cfg(os.path.join(d, "state"))

            captured = {}

            def handler(messages, agent, operation, json_mode):
                if "语言识别器" in messages[0]["content"]:
                    captured["agent"] = agent
                    captured["operation"] = operation
                    return json.dumps({"language": "russian"}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            client = FakeClient(handler=handler)
            store = Application(cfg, client=client).prepare(txt)
            # 解析后的源语言以运行状态（manifest/identity）为权威，不再改写全局 config
            self.assertEqual(store.load_manifest()["source_lang"], "ru")
            self.assertEqual(captured["agent"], "preparer")
            self.assertEqual(captured["operation"], "language.detect")

    def test_auto_detection_failure_requires_user_source(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = self._cfg(os.path.join(d, "state"))

            def handler(messages, agent, operation, json_mode):
                if "语言识别器" in messages[0]["content"]:
                    return json.dumps({"language": ""}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            with self.assertRaisesRegex(RuntimeError, "--source-language"):
                Application(cfg, client=FakeClient(handler=handler)).prepare(txt)

    def test_invalid_language_response_retries_automatically(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = self._cfg(os.path.join(d, "state"))
            detection_calls = 0

            def handler(messages, agent, operation, json_mode):
                nonlocal detection_calls
                if "语言识别器" in messages[0]["content"]:
                    detection_calls += 1
                    language = "" if detection_calls == 1 else "ja"
                    return json.dumps({"language": language}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            store = Application(cfg, client=FakeClient(handler=handler)).prepare(txt)

            self.assertEqual(detection_calls, 2)
            self.assertEqual(store.load_state().identity.source_lang, "ja")

    def test_auto_detection_request_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = self._cfg(os.path.join(d, "state"))

            def handler(messages, agent, operation, json_mode):
                if "语言识别器" in messages[0]["content"]:
                    raise RuntimeError("missing provider credential")
                return routing_handler(messages, agent, operation, json_mode)

            with self.assertRaisesRegex(RuntimeError, "missing provider credential"):
                Application(cfg, client=FakeClient(handler=handler)).prepare(txt)

    def test_explicit_same_source_and_target_stops_before_model_calls(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = Config.from_dict({"llm": fake_llm_dict()})
            cfg.source_lang = "ja"
            cfg.target_lang = "ja-JP"
            cfg.state_dir = os.path.join(d, "state")
            client = FakeClient(handler=routing_handler)

            with self.assertRaisesRegex(ValueError, "源语言与目标语言相同（ja）"):
                Application(cfg, client=client).prepare(txt)

            self.assertEqual(client.calls, [])

    def test_auto_detected_source_matching_target_stops_before_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = self._cfg(os.path.join(d, "state"))

            def handler(messages, agent, operation, json_mode):
                if "语言识别器" in messages[0]["content"]:
                    return json.dumps({"language": "chinese"}, ensure_ascii=False)
                raise AssertionError("相同语言不应继续进入分析或翻译")

            with self.assertRaisesRegex(ValueError, "源语言与目标语言相同（zh）"):
                Application(cfg, client=FakeClient(handler=handler)).prepare(txt)


if __name__ == "__main__":
    unittest.main()
