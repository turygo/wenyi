"""编排器端到端 + 断点续跑测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from trans_novel.config import Config
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.orchestrator import Orchestrator, _normalize_lang
from trans_novel.pipeline.runstore import STATUS_DONE, STATUS_PENDING
from trans_novel.postprocess.punct import normalize_zh
from tests.sample_data import write_sample_txt
from tests.fake_llm import routing_handler


def _translated_para_count(calls) -> int:
    """统计送进翻译模型的源段总数（按编号行计）。"""
    n = 0
    for c in calls:
        if "文学翻译" in c["messages"][0]["content"]:
            n += len(re.findall(r"^\[(\d+)\]", c["messages"][-1]["content"], re.M))
    return n


def _config(state_dir: str):
    return Config.from_dict({
        "language": {"source": "ja", "target": "zh"},
        "llm": {"provider": "fake", "tiers": {
            "strong": {"model": "p"}, "cheap": {"model": "f"}}},
        "segment": {"max_chars_per_batch": 1800},
        "pipeline": {"review": True, "polish": True,
                     "backtranslate_sample": 0.0, "consistency_qa": True},
        "paths": {"state_dir": state_dir},
    })


class TestOrchestrator(unittest.TestCase):
    def test_full_run_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            state = os.path.join(d, "state")
            cfg = _config(state)

            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.run(txt)

            # 全部章节标记 done
            m = store.load_manifest()
            self.assertEqual(len(m["chapters"]), 2)
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m["chapters"]))

            # 每段都有译文（润色后为 "润{i}"）
            ch0 = store.load_chapter(0)
            self.assertTrue(all(s.target for s in ch0.text_segments))

            # 术语抽取写入了「堀北」；分析器种入了「绫小路」
            from trans_novel.glossary.store import GlossaryStore
            g = GlossaryStore(store.glossary_path)
            self.assertIsNotNone(g.get_term("綾小路"))
            self.assertIsNotNone(g.get_term("堀北"))
            self.assertGreater(g.stats()["tm_entries"], 0)  # 翻译记忆库已写入
            g.close()

            # ── 续跑：所有章已 done，不应再产生翻译调用 ──
            client2 = FakeClient(handler=routing_handler)
            orch2 = Orchestrator(cfg, client=client2)
            orch2.run(txt)  # resume 语义
            translate_calls = [c for c in client2.calls
                               if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(len(translate_calls), 0)

    def test_resume_after_partial(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            state = os.path.join(d, "state")
            cfg = _config(state)

            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            # 只翻第 0 章
            store = orch.run(txt, only_chapter=0)
            m = store.load_manifest()
            self.assertEqual(m["chapters"][0]["status"], STATUS_DONE)
            self.assertNotEqual(m["chapters"][1]["status"], STATUS_DONE)

            # 续跑应只补翻第 1 章
            client2 = FakeClient(handler=routing_handler)
            orch2 = Orchestrator(cfg, client=client2)
            store2 = orch2.run(txt)
            m2 = store2.load_manifest()
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m2["chapters"]))


class TestSegmentLevelResume(unittest.TestCase):
    def _tr_handler(self, tag):
        """返回带标记的翻译 handler（译文形如 {tag}译{i}），其余走默认路由。"""
        def handler(messages, tier, json_mode):
            if "文学翻译" in messages[0]["content"]:
                n = len(re.findall(r"^\[(\d+)\]", messages[-1]["content"], re.M))
                return json.dumps({"translations": [f"{tag}译{i}" for i in range(n)]},
                                  ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)
        return handler

    def test_resume_skips_done_segments_keeps_their_text(self):
        """中断后续跑：已译完的段原样保留、不重翻；只补译未完成的段。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 8     # 每段≈独立批，便于精确续跑
            cfg.pipeline.polish = False             # 保留翻译标记，便于断言（与续跑无关）

            # 第一次：用 R1 译完第 0 章
            c1 = FakeClient(handler=self._tr_handler("R1"))
            store = Orchestrator(cfg, client=c1).run(txt, only_chapter=0)
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target and s.target.startswith("R1") for s in ch.text_segments))

            # 模拟中断：清空最后一段译文、章状态改回 pending
            ch.segments[-1].target = ""
            store.save_chapter(ch)
            store.set_chapter_status(0, STATUS_PENDING)

            # 第二次：用 R2 续跑——只应补译被清空的那 1 段
            c2 = FakeClient(handler=self._tr_handler("R2"))
            Orchestrator(cfg, client=c2).run(txt, only_chapter=0)
            self.assertEqual(_translated_para_count(c2.calls), 1)   # 仅 1 段被重翻

            ch2 = store.load_chapter(0)
            # 之前已译的段仍是 R1（未被跨位置复用、也未重翻），补译段是 R2
            self.assertTrue(ch2.text_segments[0].target.startswith("R1"))
            self.assertTrue(ch2.text_segments[-1].target.startswith("R2"))


class TestBookUnderstanding(unittest.TestCase):
    def _translate_user(self, calls) -> str:
        """返回最后一次翻译调用送进模型的 user 文本。"""
        for c in reversed(calls):
            if "文学翻译" in c["messages"][0]["content"]:
                return c["messages"][-1]["content"]
        return ""

    def test_prepass_builds_and_injects(self):
        """预扫产出逐章梗概+全书概览，并注入翻译 prompt。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            # 逐章梗概落盘到 chapter.meta
            self.assertTrue(store.load_chapter(0).meta.get("source_digest"))
            # 全书概览落盘到 analysis
            self.assertTrue((store.load_analysis() or {}).get("book_synopsis"))

            # 翻译 prompt 注入了全书概览 / 本章梗概块（且非「（无）」占位）
            user = self._translate_user(client.calls)
            self.assertIn("【全书概览】", user)
            self.assertIn("【本章梗概】", user)
            self.assertIn("全书概览", user)   # fake 概览正文
            self.assertIn("本章梗概", user)   # fake 逐章梗概正文

    def test_prescan_parallel(self):
        """并行预扫：多线程 digest 后各章梗概按章序落盘，翻译注入正常。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.prescan_concurrency = 3

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            m = store.load_manifest()
            for c in m["chapters"]:
                self.assertTrue(store.load_chapter(c["index"]).meta.get("source_digest"))
            self.assertTrue((store.load_analysis() or {}).get("book_synopsis"))
            user = self._translate_user(client.calls)
            self.assertIn("【本章梗概】", user)

    def test_resume_skips_prepass(self):
        """续跑：梗概/概览已落盘，不再产生预扫调用。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            c2 = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=c2).run(txt)
            prepass = [c for c in c2.calls
                       if "梗概员" in c["messages"][0]["content"]
                       or "概览员" in c["messages"][0]["content"]]
            self.assertEqual(len(prepass), 0)

    def test_toggle_off(self):
        """关闭 book_understanding：不预扫，prompt 用「（无）」占位。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.book_understanding = False

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            self.assertFalse(store.load_chapter(0).meta.get("source_digest"))
            self.assertFalse((store.load_analysis() or {}).get("book_synopsis"))
            prepass = [c for c in client.calls
                       if "梗概员" in c["messages"][0]["content"]
                       or "概览员" in c["messages"][0]["content"]]
            self.assertEqual(len(prepass), 0)


class TestRunSteps(unittest.TestCase):
    def test_subset_only_assemble(self):
        """run_steps 步骤子集：仅回填时不应再产生翻译调用（幂等）。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt"); write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            orch.run_steps(txt, {"translate"})
            # 仅回填，不应再翻译
            client2 = FakeClient(handler=routing_handler)
            res = Orchestrator(cfg, client=client2).run_steps(txt, {"assemble"})
            self.assertTrue(res["output"].endswith(".epub"))
            self.assertTrue(os.path.isfile(res["output"]))
            translate_calls = [c for c in client2.calls
                               if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(len(translate_calls), 0)


class TestReviewReporting(unittest.TestCase):
    """章末审校 + 严重项自动重译（autofix_severe）。"""

    # 样例首段「第一章　出会い」7 字；fix 需在 3-21 字间（比值 0.3-3.0）方可通过长度校验
    FIX_TEXT = "第一章 邂逅"   # 7 字，比值 1.0

    def _handler(self, fix_text):
        """审校每块报 index 0 漏译；带【审校意见】的翻译调用返回定向重译文。"""
        def handler(messages, tier, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "译文审校" in sys:
                return json.dumps({"issues": [
                    {"index": 0, "type": "missing", "detail": "漏了一句", "suggestion": "补上"}
                ]}, ensure_ascii=False)
            if "文学翻译" in sys and "【审校意见】" in user:
                return json.dumps({"translations": [fix_text]}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)
        return handler

    def _run(self, d, *, autofix, fix_text=None):
        txt = os.path.join(d, "novel.txt"); write_sample_txt(txt)
        cfg = _config(os.path.join(d, "state"))
        cfg.pipeline.autofix_severe = autofix
        handler = self._handler(fix_text or self.FIX_TEXT)
        return Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)

    def test_autofix_adopts_retranslation(self):
        """autofix 开：严重项定向重译被采纳 → target 更新、fixed=True。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=True)
            ch = store.load_chapter(0)
            flagged = [i for i in ch.meta["review_issues"] if i.get("type") == "missing"]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is True for i in flagged))
            self.assertTrue(all(i.get("stage") == "review" for i in flagged))
            self.assertTrue(all("chapter" in i for i in flagged))
            self.assertEqual(ch.text_segments[0].target, self.FIX_TEXT)

    def test_autofix_off_reports_only(self):
        """autofix 关：仅上报 fixed=False，正文不动。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=False)
            ch = store.load_chapter(0)
            flagged = [i for i in ch.meta["review_issues"] if i.get("type") == "missing"]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is False for i in flagged))
            self.assertNotEqual(ch.text_segments[0].target, self.FIX_TEXT)

    def test_autofix_rejects_short_retranslation(self):
        """重译结果过短（疑漏译）→ 不采纳，fixed=False，保留原译。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=True, fix_text="短")
            ch = store.load_chapter(0)
            flagged = [i for i in ch.meta["review_issues"] if i.get("type") == "missing"]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is False for i in flagged))
            self.assertNotEqual(ch.text_segments[0].target, "短")

    def test_review_index_mapping(self):
        """整章多块审校时，块内 index 正确映射回章内段号。"""
        def handler(messages, tier, json_mode):
            if "译文审校" in messages[0]["content"]:
                return json.dumps({"issues": [
                    {"index": 0, "type": "missing", "detail": "x", "suggestion": ""}
                ]}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt"); write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 8   # 审校块预算=24 → 每段自成一块
            cfg.pipeline.autofix_severe = False
            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)
            ch = store.load_chapter(0)
            idxs = sorted(i["index"] for i in ch.meta["review_issues"]
                          if i.get("type") == "missing")
            # 每块报 index 0 → 映射后应为各块首段的章内段号（0,1,2,...互不相同）
            self.assertEqual(idxs, list(range(len(ch.text_segments))))


class TestStyleAnalysis(unittest.TestCase):
    def _long_doc(self, d):
        from trans_novel.ingest.segmenter import load_document
        txt = os.path.join(d, "long.txt")
        chapters = []
        for i in range(3):
            # 段落勿以「第N章」开头，避免被 TXT reader 的章标题启发式误判
            body = "\n\n".join(f"章{i}の段落{j}です。" + "あ" * 60 for j in range(8))
            chapters.append(f"# 第{i}章\n\n{body}")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n\n".join(chapters))
        return load_document(txt, "ja", "zh")

    def test_sample_text_multipoint(self):
        """labeled=True 多点采样带三个标注；labeled=False 为纯源文单段。"""
        with tempfile.TemporaryDirectory() as d:
            doc = self._long_doc(d)
            labeled = Orchestrator._sample_text(doc)
            for tag in ("【开头样章】", "【中部样章】", "【结尾样章】"):
                self.assertIn(tag, labeled)
            plain = Orchestrator._sample_text(doc, labeled=False)
            self.assertNotIn("样章】", plain)
            self.assertIn("章0の段落0です", plain)

    def test_sample_text_short_book_dedup(self):
        """单章书：三个采样点重合，只取一次、不重复。"""
        with tempfile.TemporaryDirectory() as d:
            from trans_novel.ingest.segmenter import load_document
            txt = os.path.join(d, "short.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write("# 唯一章\n\n" + "长段落。" + "あ" * 300)
            doc = load_document(txt, "ja", "zh")
            sample = Orchestrator._sample_text(doc)
            self.assertEqual(sample.count("【开头样章】"), 1)
            self.assertNotIn("【中部样章】", sample)
            self.assertNotIn("【结尾样章】", sample)

    def test_style_brief_new_fields(self):
        """style_brief 渲染新风格维度；旧 analysis（缺新字段）不报错不输出。"""
        from trans_novel.agents.analyzer import Analyzer
        from trans_novel.llm.base import FakeClient as FC

        cfg = _config("state")
        ana = Analyzer(FC(), cfg)
        brief = ana.style_brief({
            "genre": "校园", "pacing": "短句为主", "register": "口语",
            "dialogue_style": "语气词丰富", "narration": "第一人称",
        })
        self.assertIn("句式节奏：短句为主", brief)
        self.assertIn("语域：口语", brief)
        self.assertIn("对话风格：语气词丰富", brief)
        self.assertIn("叙事：第一人称", brief)
        # 旧格式：只有老字段
        old = ana.style_brief({"genre": "校园", "tone": "冷峻"})
        self.assertIn("体裁：校园", old)
        self.assertNotIn("句式节奏", old)


class TestGlossaryScope(unittest.TestCase):
    def _run_with_terms(self, d, scope):
        from trans_novel.glossary.store import GlossaryStore, GlossaryTerm

        txt = os.path.join(d, "novel.txt")
        write_sample_txt(txt)
        cfg = _config(os.path.join(d, "state"))
        cfg.pipeline.glossary_scope = scope

        orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
        store = orch.prepare(txt)
        g = GlossaryStore(store.glossary_path)
        # ①锁定人物（source 不在正文）②无关术语（source/alias 均不在正文）③alias 在正文出现
        g.upsert_term(GlossaryTerm(source="外部人物X", target="外部译名",
                                   type="人物", locked=True))
        g.upsert_term(GlossaryTerm(source="無関係用語", target="无关术语", type="术语"))
        g.upsert_term(GlossaryTerm(source="ホリキタ", target="堀北译名",
                                   aliases=["堀北"], type="术语"))
        g.close()

        client = FakeClient(handler=routing_handler)
        Orchestrator(cfg, client=client).run(txt)
        return ["\n".join(m["content"] for m in c["messages"])
                for c in client.calls
                if "文学翻译" in c["messages"][0]["content"]]

    def test_chapter_scope_prunes(self):
        """chapter：锁定人物保留、无关术语剔除、alias 命中保留。"""
        with tempfile.TemporaryDirectory() as d:
            translate_prompts = self._run_with_terms(d, "chapter")
            self.assertTrue(translate_prompts)
            for p in translate_prompts:
                self.assertIn("外部人物X", p)     # 锁定人物：始终保留
                self.assertNotIn("無関係用語", p)  # 本章未出现：剔除
                self.assertIn("ホリキタ", p)      # 别名「堀北」在正文：保留

    def test_full_scope_keeps_all(self):
        with tempfile.TemporaryDirectory() as d:
            translate_prompts = self._run_with_terms(d, "full")
            self.assertTrue(translate_prompts)
            for p in translate_prompts:
                self.assertIn("外部人物X", p)
                self.assertIn("無関係用語", p)
                self.assertIn("ホリキタ", p)

    def test_batch_glossary_refreshes_following_prompts(self):
        """批次翻译后实时抽取术语，后续批次 prompt 立即带上新称谓。"""
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in system:
                n = len(re.findall(r"^\[(\d+)\]", user, re.M))
                return json.dumps({"translations": ["小夏帆" for _ in range(n)]},
                                  ensure_ascii=False)
            if "术语" in system and "抽取器" in system and "夏帆ちゃん" in user and "小夏帆" in user:
                return json.dumps({"terms": [
                    {"source": "夏帆ちゃん", "target": "小夏帆",
                     "type": "称谓", "aliases": ["夏帆"], "note": "亲昵称呼"}
                ]}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(
                    "# 第一章\n\n"
                    "「夏帆ちゃん」と母親が言った。\n\n"
                    "夏帆ちゃんは窓の外を見た。\n"
                )
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            cfg.segment.max_chars_per_batch = 10

            client = FakeClient(handler=handler)
            Orchestrator(cfg, client=client).run(txt)

            translate_prompts = [
                "\n".join(m["content"] for m in c["messages"])
                for c in client.calls
                if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertGreaterEqual(len(translate_prompts), 3)
            self.assertIn("夏帆ちゃん → 小夏帆", translate_prompts[-1])

    def test_chapter_glossary_refreshes_review_prompt(self):
        """全章兜底术语抽取在 review 前执行，章末审校能看到新称谓。"""
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in system:
                n = len(re.findall(r"^\[(\d+)\]", user, re.M))
                return json.dumps({"translations": ["小夏帆" for _ in range(n)]},
                                  ensure_ascii=False)
            if "术语" in system and "抽取器" in system and "夏帆ちゃん" in user:
                return json.dumps({"terms": [
                    {"source": "夏帆ちゃん", "target": "小夏帆",
                     "type": "称谓", "aliases": ["夏帆"], "note": "亲昵称呼"}
                ]}, ensure_ascii=False)
            if "译文审校" in system:
                self.assertIn("夏帆ちゃん → 小夏帆", user)
                return json.dumps({"issues": []}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write("# 第一章\n\n「夏帆ちゃん」と母親が言った。\n")
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            cfg.segment.max_chars_per_batch = 200

            Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)


class TestTierRouting(unittest.TestCase):
    def test_task_tiers(self):
        """机械任务走 fast 档、判断类走 cheap、翻译走 strong；梗概带 max_tokens 上限。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.backtranslate_sample = 1.0  # 强制触发回译

            client = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=client).run(txt)

            expect = {
                "章节梗概员": "fast", "全书概览员": "fast",
                "术语与称呼抽取器": "fast", "回译译者": "fast",
                "译文审校": "cheap", "保真度": "cheap",
                "文学翻译": "strong",
            }
            seen = set()
            for c in client.calls:
                system = c["messages"][0]["content"]
                for marker, tier in expect.items():
                    if marker in system:
                        self.assertEqual(c["tier"], tier, f"{marker} 应走 {tier} 档")
                        seen.add(marker)
                        if marker == "章节梗概员":
                            self.assertEqual(c["max_tokens"], 600)
                        if marker == "全书概览员":
                            self.assertEqual(c["max_tokens"], 1200)
            self.assertEqual(seen, set(expect), "各类调用都应出现")


class TestLangNormalize(unittest.TestCase):
    def test_normalize_lang(self):
        self.assertEqual(_normalize_lang("Japanese"), "ja")
        self.assertEqual(_normalize_lang("日语"), "ja")
        self.assertEqual(_normalize_lang("RU"), "ru")
        self.assertEqual(_normalize_lang("russian"), "ru")
        self.assertEqual(_normalize_lang("fr"), "fr")
        self.assertEqual(_normalize_lang("unknown"), "")
        self.assertEqual(_normalize_lang(""), "")


class TestPolishAsync(unittest.TestCase):
    def test_batch_translated_then_batch_polished_events(self):
        """polish 开启：batch_translated 先发（polished=False，segments 为 raw 译文），
        章末排干润色后再发 batch_polished（segments 为最终润色文本），pending_polish 清空。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(
                txt, only_chapter=0)

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            translated = [e for e in events
                         if e["event"] == "batch_translated" and e["chapter"] == 0]
            polished = [e for e in events
                       if e["event"] == "batch_polished" and e["chapter"] == 0]
            self.assertTrue(translated)
            self.assertTrue(polished)
            # batch_translated 触发时尚未润色：polished=False，segments 记 raw 译文
            for e in translated:
                self.assertFalse(e["polished"])
                for seg in e["segments"]:
                    self.assertTrue(seg["target"].startswith("译"))
            # 章末排干后 batch_polished 携带最终润色文本
            for e in polished:
                for seg in e["segments"]:
                    self.assertTrue(seg["target"].startswith("润"))
            # run() 返回时排干已完成：正文与 meta 均为最终态，无残留 pending 标记
            ch = store.load_chapter(0)
            self.assertFalse(ch.meta.get("pending_polish"))
            self.assertTrue(all(s.target.startswith("润") for s in ch.text_segments))


class TestPendingPolishResume(unittest.TestCase):
    def test_resume_repolishes_leftover_pending_batches(self):
        """续跑：章末未排干完的 pending_polish 批次，续跑时重新提交润色并写回，
        不静默丢失（不变量 b）；该批本身因已有译文，不会被重翻。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(
                txt, only_chapter=0)
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target and s.target.startswith("润") for s in ch.text_segments))
            self.assertFalse(ch.meta.get("pending_polish"))

            # 模拟"批已落盘但章末排干润色前中断"：把最后一段的译文改回未润色的 raw
            # （"译{i}"），补回 pending_polish 标记，章状态改回 pending。
            last_idx = len(ch.text_segments) - 1
            ch.segments[last_idx].target = f"译{last_idx}"
            ch.meta["pending_polish"] = [{"start": last_idx, "count": 1}]
            store.save_chapter(ch)
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=client2).run(txt, only_chapter=0)
            translate_calls = [c for c in client2.calls
                               if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(len(translate_calls), 0)   # 已有译文，批跳过，未重翻

            ch2 = store.load_chapter(0)
            # routing_handler 的润色输出按"本次调用内"的局部下标编号：该批只含 1 段
            # （原始的第 last_idx 段），单独重新提交润色后局部下标为 0 → "润0"。
            self.assertEqual(ch2.text_segments[last_idx].target, "润0")
            self.assertFalse(ch2.meta.get("pending_polish"))

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(any(
                e["event"] == "batch_polished" and e["chapter"] == 0
                and e["start_index"] == last_idx for e in events))


class TestReviewAsync(unittest.TestCase):
    """review=true 且 autofix_severe=false：章末审校提交共享线程池异步跑，
    run() 返回前必须排干——issues 合并写入 chapter.meta["review_issues"]
    并发 chapter_reviewed 事件；review worker 出错不得中断 run。"""

    @staticmethod
    def _issue_handler(messages, tier, json_mode):
        # 无共享可变状态：每次调用构造新 dict，可被线程池并发调用
        if "译文审校" in messages[0]["content"]:
            return json.dumps({"issues": [
                {"index": 0, "type": "terminology", "detail": "术语不一致",
                 "suggestion": "改用对照表"}
            ]}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    def test_async_review_issues_persisted_before_run_returns(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Orchestrator(
                cfg, client=FakeClient(handler=self._issue_handler)).run(txt)

            m = store.load_manifest()
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m["chapters"]))
            for ci in range(len(m["chapters"])):
                ch = store.load_chapter(ci)
                found = [i for i in ch.meta.get("review_issues", [])
                         if i.get("type") == "terminology"]
                self.assertTrue(found, f"第 {ci} 章异步审校结果未写回 meta")
                for it in found:
                    self.assertEqual(it.get("chapter"), ci)
                    self.assertEqual(it.get("stage"), "review")
                    self.assertIs(it.get("fixed"), False)

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            reviewed = {e["chapter"] for e in events
                        if e["event"] == "chapter_reviewed"}
            self.assertEqual(reviewed, set(range(len(m["chapters"]))),
                             "每章都应发 chapter_reviewed 事件")

    def test_review_worker_failure_does_not_break_run(self):
        # review 未来（_review_chapter）本身抛异常 → 触发 _drain_ready_reviews 的
        # except 分支（记 chapter_review_failed 后 continue，不中断 run）。
        # 注意：不能靠 handler 对 '译文审校' 抛异常来验证——Reviewer.review 内部
        # _ask_json(..., default=[]) 会吞掉 LLM 异常返回 []，future 正常完成、照常
        # 发 chapter_reviewed，except 分支永不执行（旧版删掉错误处理测试仍会通过）。
        # 故直接以实例属性遮蔽绑定方法 _review_chapter，让提交到线程池的 future 真抛。
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False  # 异步审校路径

            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))

            def _boom(*_a, **_k):
                raise RuntimeError("审校崩")
            orch._review_chapter = _boom  # 遮蔽类方法：future 执行即抛

            store = orch.run(txt)

            m = store.load_manifest()
            chapters = set(range(len(m["chapters"])))

            # (a) 审校 future 全崩，run 仍走完，每章保持 DONE（未被异常中断）
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m["chapters"]),
                            "审校 worker 抛异常不得阻断整章完成")

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            failed = {e["chapter"] for e in events
                      if e["event"] == "chapter_review_failed"}
            reviewed = {e["chapter"] for e in events
                        if e["event"] == "chapter_reviewed"}

            # (b) 载荷断言：每个审校崩溃的章都记了 chapter_review_failed——这是唯一能
            # 证明 except→chapter_review_failed 分支真的执行过的证据。若删掉该错误处理
            # （让异常穿透），异常会在机会性/收尾 drain 里抛出，run() 直接崩、拿不到
            # store，本断言必失败。
            self.assertEqual(failed, chapters,
                             "每个审校崩溃的章都必须记 chapter_review_failed")
            # (c) 崩溃章不得发 chapter_reviewed，且 review_issues 未被写回（保持空）
            self.assertEqual(reviewed, set(),
                             "审校失败的章不得发 chapter_reviewed")
            for ci in chapters:
                self.assertEqual(
                    store.load_chapter(ci).meta.get("review_issues", []), [],
                    f"第 {ci} 章审校失败，review_issues 不得被写入")

    def test_crash_resume_reruns_pending_review(self):
        # review 断点续跑不变量（异步审校版）：章已标 DONE 但异步审校结果还没写回
        # 就宕机时，靠 manifest 里的 review_pending 持久标记 + run() 开头的
        # _resume_pending_reviews 补跑，审校结果不静默丢失。没有标记或补跑逻辑，
        # 崩溃后该章审校结果永久缺失，本测试必失败。
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            # (1) 正常跑一遍：审校结果写回、标记清空（前置条件）
            store = Orchestrator(
                cfg, client=FakeClient(handler=self._issue_handler)).run(txt)
            self.assertEqual(store.review_pending_chapters(), [],
                             "正常收尾后不应残留任何 review_pending 标记")

            # (2) 模拟崩溃窗口：章 0 已 DONE，但标记残留且审校结果被抹掉
            store.set_review_pending(0, True)
            ch = store.load_chapter(0)
            ch.meta["review_issues"] = []
            store.save_chapter(ch)
            self.assertIn(0, store.review_pending_chapters(),
                          "崩溃模拟：章 0 应带 review_pending 标记")

            with open(store.event_log_path, encoding="utf-8") as f:
                events_before = sum(1 for line in f if line.strip())

            # (3) 续跑：所有章已 DONE → targets 为空，补跑只能来自 _resume_pending_reviews
            client2 = FakeClient(handler=self._issue_handler)
            Orchestrator(cfg, client=client2).run(txt)

            # (4a) 载荷断言：章 0 审校结果被重新写回（术语项，字段完整）
            issues = store.load_chapter(0).meta.get("review_issues", [])
            found = [i for i in issues if i.get("type") == "terminology"]
            self.assertTrue(found, "续跑必须重跑章 0 审校并写回 review_issues")
            for it in found:
                self.assertEqual(it.get("chapter"), 0)
                self.assertEqual(it.get("stage"), "review")
                self.assertIs(it.get("fixed"), False)

            # (4b) 补跑成功后标记被清空
            self.assertEqual(store.review_pending_chapters(), [],
                             "续跑写回后 review_pending 标记必须清空")

            # (4c) 第二次 run 的事件里有章 0 的 chapter_reviewed
            with open(store.event_log_path, encoding="utf-8") as f:
                all_events = [json.loads(line) for line in f if line.strip()]
            second_run = all_events[events_before:]
            reviewed = {e["chapter"] for e in second_run
                        if e["event"] == "chapter_reviewed"}
            self.assertIn(0, reviewed, "续跑应为章 0 补发 chapter_reviewed 事件")

            # (4d) 续跑只补审校，绝不重译（无 '文学翻译' 调用）
            translate_calls = [c for c in client2.calls
                               if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(len(translate_calls), 0, "续跑只补审校，绝不重译")


class TestPolishFailureFallback(unittest.TestCase):
    def test_polish_failure_falls_back_to_raw_translation(self):
        """润色调用失败（handler 抛异常）：该批最终 target 回退为未润色译文
        （经标点规范化），run() 正常完成，无 pending_polish 残留。"""
        def handler(messages, tier, json_mode):
            if "中文润色编辑" in messages[0]["content"]:
                raise RuntimeError("润色模型宕机")
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(
                txt, only_chapter=0)

            m = store.load_manifest()
            self.assertEqual(m["chapters"][0]["status"], STATUS_DONE,
                             "润色失败不得阻断整章完成")
            ch = store.load_chapter(0)
            # 单批：routing_handler 译文按批内下标编号 → 段 i 的 raw 译文为 "译{i}"
            expected = [normalize_zh(f"译{i}") for i in range(len(ch.text_segments))]
            self.assertEqual([s.target for s in ch.text_segments], expected)
            self.assertFalse(ch.meta.get("pending_polish"),
                             "润色失败的批次也必须清掉 pending_polish 标记")


if __name__ == "__main__":
    unittest.main()
