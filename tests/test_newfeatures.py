"""新功能测试（离线）：模型语言检测、标点规范化、术语 AI 审计统一、连续全流程。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import write_sample_txt
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.ingest.models import Chapter, Segment
from trans_novel.llm import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.runstore import RunStore
from trans_novel.pipeline.state import ChapterIndex, ChapterProgress, RunIdentity, RunState
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
            # glossary_rewrite_applied 不复述全局 replace_map、不带 source 明文，
            # 只带实际命中的替换元数据与 before/after
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            rewrites = [e for e in events if e["event"] == "glossary_rewrite_applied"]
            self.assertTrue(rewrites, "正文改写应发 glossary_rewrite_applied")
            for e in rewrites:
                self.assertNotIn("source", e, "事件不带 source 明文")
                self.assertNotIn("replace_map", e, "事件不复述全局 replace_map")
                self.assertIn("before", e)
                self.assertIn("after", e)
                self.assertEqual(e["replacements"], [{"variant": "佳穗子", "canonical": "佳穂子"}])

    def test_rewrite_replacements_follow_actual_sequential_execution(self):
        """glossary_rewrite_applied 的 replacements 必须来自真实替换过程：
        重叠变体被先执行的长变体吞掉时不误报；后执行替换引入的新变体随后
        命中并执行时如实按序上报；同一章多段改写只整章保存一次。"""
        from trans_novel.agents.glossary_auditor import GlossaryAuditor

        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state"))
            store.save_state(
                RunState(
                    identity=RunIdentity(
                        source_bytes_sha256="test-hash",
                        run_input_schema_version=1,
                        source_lang="en",
                        target_lang="zh",
                    ),
                    title="T",
                    fmt="text",
                    source_path="",
                    source_lang="en",
                    target_lang="zh",
                    chapters=[ChapterIndex(index=0, title="第一章", href=None)],
                    progress={0: ChapterProgress()},
                )
            )
            chapter = Chapter(
                index=0,
                title="第一章",
                segments=[
                    # 段0：BC 被先执行的长变体 ABC 整段吞掉 → 只报 ABC
                    Segment(index=0, source="a", target="内容ABC结尾。"),
                    # 段1：AB→CD 引入了变体 CD，随后 CD→E 真实执行 → 按序报 AB、CD
                    Segment(index=1, source="b", target="前缀AB后缀。"),
                    Segment(index=2, source="c", target="无变体段落。"),
                ],
            )
            store.save_chapter(chapter)
            g = GlossaryStore(store.glossary_path)
            saves: list[int] = []
            real_save = store.save_chapter

            def counting_save(ch):
                saves.append(ch.index)
                return real_save(ch)

            store.save_chapter = counting_save
            changed = GlossaryAuditor._rewrite_targets(
                store, g, {"ABC": "X", "BC": "Y", "AB": "CD", "CD": "E"}
            )
            g.close()

            self.assertEqual(changed, 2)
            self.assertEqual(saves, [0], "同一章的多段改写应只整章保存一次")
            reloaded = store.load_chapter(0)
            self.assertEqual(reloaded.segments[0].target, "内容X结尾。")
            self.assertEqual(reloaded.segments[1].target, "前缀E后缀。")
            self.assertEqual(reloaded.segments[2].target, "无变体段落。")
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            rewrites = {e["index"]: e for e in events if e["event"] == "glossary_rewrite_applied"}
            self.assertEqual(
                rewrites[0]["replacements"],
                [{"variant": "ABC", "canonical": "X"}],
                "被长变体吞掉的 BC 不得误报",
            )
            self.assertEqual(
                rewrites[1]["replacements"],
                [{"variant": "AB", "canonical": "CD"}, {"variant": "CD", "canonical": "E"}],
                "替换引入的新变体随后执行时应按序上报",
            )

    def test_latin_residue_coalesces_one_save_per_chapter(self):
        """多个锁定拉丁术语命中同一章：每章只保存一次、每个命中术语一条 applied 记录；
        glossary_latin_residue_fixed 事件紧凑（无 source 明文）且按章落盘后发出。"""
        from trans_novel.agents.glossary_auditor import GlossaryAuditor

        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state"))
            store.save_state(
                RunState(
                    identity=RunIdentity(
                        source_bytes_sha256="test-hash",
                        run_input_schema_version=1,
                        source_lang="en",
                        target_lang="zh",
                    ),
                    title="T",
                    fmt="text",
                    source_path="",
                    source_lang="en",
                    target_lang="zh",
                    chapters=[ChapterIndex(index=0, title="第一章", href=None)],
                    progress={0: ChapterProgress()},
                )
            )
            chapter = Chapter(
                index=0,
                title="第一章",
                segments=[
                    Segment(index=0, source="a", target="Ayanokoji 走进了教室。"),
                    Segment(index=1, source="b", target="Kaho 笑了笑。Ayanokoji 点头。"),
                ],
            )
            store.save_chapter(chapter)
            g = GlossaryStore(store.glossary_path)
            g.upsert_term(
                GlossaryTerm(source="Kaho", target="佳穂", confidence="high", locked=True)
            )
            g.upsert_term(
                GlossaryTerm(source="Ayanokoji", target="綾小路", confidence="high", locked=True)
            )

            cfg = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
            cfg.source_lang = "en"
            cfg.state_dir = os.path.join(d, "state")
            client = FakeClient(handler=lambda m, a, o, j: "{}")

            saves: list[int] = []
            real_save = store.save_chapter

            def counting_save(ch):
                saves.append(ch.index)
                return real_save(ch)

            store.save_chapter = counting_save
            applied = GlossaryAuditor(client, cfg).audit(store, g)
            g.close()

            self.assertEqual(saves, [0], "多个术语命中同一章应只整章写一次")
            reloaded = store.load_chapter(0)
            self.assertEqual(reloaded.segments[0].target, "綾小路走进了教室。")
            self.assertEqual(reloaded.segments[1].target, "佳穂笑了笑。綾小路点头。")
            self.assertEqual(len(applied), 2, "每个命中术语一条 applied 记录")
            self.assertEqual({a["source"] for a in applied}, {"Kaho", "Ayanokoji"})
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            fixed = [e for e in events if e["event"] == "glossary_latin_residue_fixed"]
            self.assertEqual(len(fixed), 3)  # 段0 一处 + 段1 两处
            for e in fixed:
                self.assertNotIn("source", e, "事件不带 source 明文")
                self.assertNotIn("replace_map", e)
                self.assertIn("before", e)
                self.assertIn("after", e)
                self.assertIn("term_source", e)
                self.assertIn("term_target", e)


if __name__ == "__main__":
    unittest.main()
