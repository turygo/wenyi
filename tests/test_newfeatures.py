"""新功能测试（离线）：模型语言检测、标点规范化、术语 AI 审计统一、连续全流程。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import write_sample_txt
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.llm import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.postprocess.punct import normalize_heading_numbering, normalize_zh


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
            self.assertEqual(captured["agent"], "reviewer")
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


class TestPunct(unittest.TestCase):
    def test_japanese_quotes(self):
        self.assertEqual(normalize_zh("「你好」"), "“你好”")
        self.assertEqual(normalize_zh("『书名』"), "‘书名’")

    def test_halfwidth_to_full_in_cjk(self):
        self.assertEqual(normalize_zh("他说,真的吗?"), "他说，真的吗？")

    def test_no_harm_to_english_numbers(self):
        self.assertEqual(normalize_zh("9.11 vs 9.8"), "9.11 vs 9.8")

    def test_ellipsis_and_dash(self):
        self.assertEqual(normalize_zh("等等...走了--他笑了"), "等等……走了——他笑了")


class TestHeadingNumbering(unittest.TestCase):
    def test_boundary_numbers(self):
        self.assertEqual(normalize_heading_numbering("第5章 迫击炮"), "第五章 迫击炮")
        self.assertEqual(normalize_heading_numbering("第10章"), "第十章")
        self.assertEqual(normalize_heading_numbering("第22章"), "第二十二章")
        self.assertEqual(normalize_heading_numbering("第100章"), "第一百章")
        self.assertEqual(normalize_heading_numbering("第105章"), "第一百零五章")
        self.assertEqual(normalize_heading_numbering("第110章"), "第一百一十章")
        self.assertEqual(normalize_heading_numbering("第1024章"), "第一千零二十四章")

    def test_fullwidth_digits(self):
        self.assertEqual(normalize_heading_numbering("第５章 全角"), "第五章 全角")

    def test_idempotent(self):
        once = normalize_heading_numbering("第5章 迫击炮")
        self.assertEqual(normalize_heading_numbering(once), once)

    def test_already_hanzi_unchanged(self):
        self.assertEqual(normalize_heading_numbering("第五章 迫击炮"), "第五章 迫击炮")

    def test_non_matching_text_unchanged(self):
        self.assertEqual(normalize_heading_numbering("迫击炮与大规模生产"), "迫击炮与大规模生产")

    def test_quantifier_variants(self):
        self.assertEqual(normalize_heading_numbering("第3部 序曲"), "第三部 序曲")
        self.assertEqual(normalize_heading_numbering("第7节"), "第七节")
        self.assertEqual(normalize_heading_numbering("第2卷"), "第二卷")
        self.assertEqual(normalize_heading_numbering("第9回"), "第九回")

    def test_mid_string_number_not_touched(self):
        self.assertEqual(
            normalize_heading_numbering("番外：来自第5章的回忆"),
            "番外：来自第5章的回忆",
        )

    def test_out_of_range_unchanged(self):
        self.assertEqual(normalize_heading_numbering("第0章"), "第0章")
        self.assertEqual(normalize_heading_numbering("第10000章"), "第10000章")

    def test_empty_and_none_like(self):
        self.assertEqual(normalize_heading_numbering(""), "")


class TestGlossaryAudit(unittest.TestCase):
    def test_unify_variants_and_rewrite_targets(self):
        from trans_novel.agents.glossary_auditor import GlossaryAuditor

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            state = os.path.join(d, "state")
            cfg = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
            cfg.source_lang = "ja"
            cfg.state_dir = state
            orch = Application(cfg, client=FakeClient(handler=routing_handler))
            store = orch.run(txt)

            # 人为制造译法漂移：术语表写入 佳穂子，章节正文里混入 佳穗子（3字，避开防线2的2字上限）
            g = GlossaryStore(store.glossary_path)
            g.upsert_term(GlossaryTerm(source="カホ", target="佳穂子", type="人物"), chapter=0)
            g.close()
            ch = store.load_chapter(0)
            ch.segments[1].target = "佳穂子和佳穗子在一起。"  # 同名两种写法
            store.save_chapter(ch)

            def handler(messages, agent, operation, json_mode):
                if "术语一致性审计员" in messages[0]["content"]:
                    return json.dumps(
                        {
                            "unifications": [
                                {
                                    "source": "カホ",
                                    "canonical": "佳穂子",
                                    "variants": ["佳穗子"],
                                    "reason": "统一为佳穂子",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                return "{}"

            g = GlossaryStore(store.glossary_path)
            client = FakeClient(handler=handler)
            applied = GlossaryAuditor(client, cfg).audit(store, g)
            self.assertEqual(client.calls[0]["agent"], "analyst")
            self.assertEqual(client.calls[0]["operation"], "glossary.audit")
            self.assertEqual(len(applied), 1)
            term = g.get_term("カホ")
            self.assertTrue(term.locked)
            self.assertIn("佳穗子", term.aliases)
            g.close()

            # 正文里的 佳穗子 应已被改写为 佳穂子
            ch2 = store.load_chapter(0)
            self.assertEqual(ch2.segments[1].target, "佳穂子和佳穂子在一起。")


class TestRunAll(unittest.TestCase):
    def test_continuous_pipeline_outputs_epub(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            state = os.path.join(d, "state")
            cfg = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
            cfg.state_dir = state
            cfg.pipeline.backtranslate_sample = 0
            seen = []
            orch = Application(cfg, client=FakeClient(handler=routing_handler))
            result = orch.run_all(
                txt,
                progress=lambda done, total, label: seen.append((done, total)),
                out_format="epub",
            )
            self.assertTrue(result["output"].endswith(".epub"))
            self.assertTrue(zipfile.is_zipfile(result["output"]))
            # 进度回调被触发，且最终 done==total
            self.assertTrue(seen)
            self.assertEqual(seen[-1][0], seen[-1][1])
            # auto 通过模型检测把源语言定为 ja（以运行状态为权威，不改写全局 config）
            self.assertEqual(result["store"].load_manifest()["source_lang"], "ja")
            # 报告含一致性字段；术语审计不在连续流程中自动运行
            self.assertIn("consistency_issues", result["report"])
            with open(result["store"].event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            event_names = [e["event"] for e in events]
            self.assertIn("run_initialized", event_names)
            self.assertIn("batch_translated", event_names)
            self.assertIn("report_saved", event_names)
            self.assertIn("assembled", event_names)
            self.assertNotIn("glossary_audit_finished", event_names)
            translated = next(e for e in events if e["event"] == "batch_translated")
            self.assertTrue(translated["segments"])
            self.assertIn("source", translated["segments"][0])
            self.assertIn("target", translated["segments"][0])


if __name__ == "__main__":
    unittest.main()
