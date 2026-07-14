"""编排器端到端 + 断点续跑测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unittest

from tests.fake_llm import routing_handler
from tests.sample_data import write_sample_txt
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.orchestrator import Orchestrator, _normalize_lang
from trans_novel.pipeline.runstore import STATUS_DONE, STATUS_PENDING
from trans_novel.postprocess.punct import normalize_zh


def _translated_para_count(calls) -> int:
    """统计送进翻译模型的源段总数（按编号行计）。"""
    n = 0
    for c in calls:
        if "文学翻译" in c["messages"][0]["content"]:
            n += len(re.findall(r"^\[(\d+)\]", c["messages"][-1]["content"], re.M))
    return n


def _config(state_dir: str):
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
            "segment": {"max_chars_per_batch": 1800},
            "pipeline": {
                "review": True,
                "polish": True,
                "backtranslate_sample": 0.0,
                "consistency_qa": True,
            },
            "paths": {"state_dir": state_dir},
        }
    )


class TestOrchestrator(unittest.TestCase):
    def test_prepare_retries_after_analysis_failure(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            def fail_analysis(messages, tier, json_mode):
                raise RuntimeError("temporary model failure")

            with self.assertRaisesRegex(RuntimeError, "temporary model failure"):
                Orchestrator(cfg, client=FakeClient(handler=fail_analysis)).prepare(txt)

            run_dirs = [os.path.join(cfg.state_dir, name) for name in os.listdir(cfg.state_dir)]
            self.assertEqual(len(run_dirs), 1)
            self.assertFalse(os.path.isfile(os.path.join(run_dirs[0], "manifest.json")))

            store = Orchestrator(cfg, client=FakeClient(handler=routing_handler)).prepare(txt)
            self.assertTrue(store.exists())
            self.assertTrue(store.load_manifest()["initialized"])
            self.assertIsNotNone(store.load_analysis())

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

            # 默认配置（inflight_glossary=False）：术语库有 namer 一次性定名种入的条目
            # （fake 全书定名路由把候选原样定名，type=人物）；分析器种入了「绫小路」；
            # 全程不应向 FakeClient 发出旧版"抽取器" system 请求。
            from trans_novel.glossary.store import GlossaryStore

            g = GlossaryStore(store.glossary_path)
            self.assertIsNotNone(g.get_term("綾小路"))
            self.assertIsNotNone(g.get_term("堀北"))
            self.assertGreater(g.stats()["tm_entries"], 0)  # 翻译记忆库已写入
            g.close()
            extractor_calls = [
                c
                for c in client.calls
                if "术语" in c["messages"][0]["content"] and "抽取器" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(extractor_calls), 0, "默认路径不得调用旧版抽取器")
            analysis = store.load_analysis() or {}
            self.assertTrue(analysis.get("term_mining_done"))

            # ── 续跑：所有章已 done，不应再产生翻译调用；也不应重复定名 ──
            client2 = FakeClient(handler=routing_handler)
            orch2 = Orchestrator(cfg, client=client2)
            orch2.run(txt)  # resume 语义
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0)
            naming_calls = [c for c in client2.calls if "全书定名" in c["messages"][0]["content"]]
            self.assertEqual(len(naming_calls), 0, "续跑不应重复定名")

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
        """返回带标记的翻译 handler（译文形如 {tag}译{i}），其余走默认路由。
        用原文长度补齐译文（填充字符），避免触发新增确定性 lint 的 too_short
        判定——本类只测续跑/段级幂等，不是 lint 的测试范围。"""

        def handler(messages, tier, json_mode):
            if "文学翻译" in messages[0]["content"]:
                user = messages[-1]["content"]
                pairs = re.findall(r"^\[(\d+)\] (.*)$", user, re.M)
                out = []
                for i, src in pairs:
                    base = f"{tag}译{i}"
                    out.append(base + "文" * max(0, len(src) - len(base)))
                return json.dumps({"translations": out}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        return handler

    def test_resume_skips_done_segments_keeps_their_text(self):
        """中断后续跑：已译完的段原样保留、不重翻；只补译未完成的段。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 8  # 每段≈独立批，便于精确续跑
            cfg.pipeline.polish = False  # 保留翻译标记，便于断言（与续跑无关）

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
            self.assertEqual(_translated_para_count(c2.calls), 1)  # 仅 1 段被重翻

            ch2 = store.load_chapter(0)
            # 之前已译的段仍是 R1（未被跨位置复用、也未重翻），补译段是 R2
            self.assertTrue(ch2.text_segments[0].target.startswith("R1"))
            self.assertTrue(ch2.text_segments[-1].target.startswith("R2"))

    def test_resume_splits_mixed_batch_after_budget_change(self):
        """大批次内只缺一段时，也不能覆盖同批已有译文。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 100_000
            cfg.pipeline.polish = False

            first_client = FakeClient(handler=self._tr_handler("R1"))
            store = Orchestrator(cfg, client=first_client).run(txt, only_chapter=0)
            chapter = store.load_chapter(0)
            chapter.text_segments[-1].target = ""
            store.save_chapter(chapter)
            store.set_chapter_status(0, STATUS_PENDING)

            # 改变预算后，新分批仍可能把已完成段与空段放在一起。
            cfg.segment.max_chars_per_batch = 50_000
            second_client = FakeClient(handler=self._tr_handler("R2"))
            Orchestrator(cfg, client=second_client).run(txt, only_chapter=0)

            self.assertEqual(_translated_para_count(second_client.calls), 1)
            resumed = store.load_chapter(0).text_segments
            self.assertTrue(
                all((segment.target or "").startswith("R1") for segment in resumed[:-1])
            )
            self.assertTrue((resumed[-1].target or "").startswith("R2"))


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
            self.assertIn("全书概览", user)  # fake 概览正文
            self.assertIn("本章梗概", user)  # fake 逐章梗概正文

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
            prepass = [
                c
                for c in c2.calls
                if "梗概员" in c["messages"][0]["content"]
                or "概览员" in c["messages"][0]["content"]
            ]
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
            prepass = [
                c
                for c in client.calls
                if "梗概员" in c["messages"][0]["content"]
                or "概览员" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(prepass), 0)


class TestTermMiningRobustness(unittest.TestCase):
    """reviewer 三个 major 缺陷的回归：定名失败不落幂等标记、既有人物确认后升级锁定、
    预扫挖掘输入用 is_back_matter 排除附属章。"""

    @staticmethod
    def _events(store):
        with open(store.event_log_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_naming_failure_does_not_set_flag_and_retries_on_resume(self):
        """一次强档定名异常：term_mining_done 不落盘，不静默永久跳过；续跑重试并成功。"""

        def failing_handler(messages, tier, json_mode):
            if "全书定名" in messages[0]["content"]:
                raise RuntimeError("strong tier timeout")
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=failing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            analysis = store.load_analysis() or {}
            self.assertFalse(
                analysis.get("term_mining_done"), "定名异常时不得落盘 term_mining_done"
            )
            g = GlossaryStore(store.glossary_path)
            self.assertIsNone(g.get_term("堀北"))
            g.close()
            failed = [e for e in self._events(store) if e["event"] == "cast_naming_failed"]
            self.assertTrue(failed, "应记录 cast_naming_failed 事件")

            # 续跑：换正常 handler，应重试挖掘/定名并成功落盘（不是静默永久跳过）
            client2 = FakeClient(handler=routing_handler)
            store2 = Orchestrator(cfg, client=client2).run(txt)
            analysis2 = store2.load_analysis() or {}
            self.assertTrue(analysis2.get("term_mining_done"))
            g2 = GlossaryStore(store2.glossary_path)
            self.assertIsNotNone(g2.get_term("堀北"))
            g2.close()

    def test_namer_confirmed_person_gets_locked(self):
        """seed_glossary 先种入的未锁定人物，被 namer 确认沿用译法后应升级为 locked+高置信度。"""
        from trans_novel.glossary.store import TYPE_PERSON, GlossaryTerm

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            store = orch.prepare(txt)
            g = GlossaryStore(store.glossary_path)
            # 模拟 seed_glossary 种入的未锁定人物：source 与 mining 固定候选「堀北」同名，
            # target 与 fake 全书定名路由的原样定名结果一致，用于验证确认升级逻辑。
            g.upsert_term(
                GlossaryTerm(
                    source="堀北",
                    target="堀北",
                    type=TYPE_PERSON,
                    confidence="medium",
                    locked=False,
                ),
                chapter=1,
            )
            g.close()

            Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            g2 = GlossaryStore(store.glossary_path)
            term = g2.get_term("堀北")
            self.assertTrue(term.locked, "namer 确认沿用后应升级为锁定")
            self.assertEqual(term.confidence, "high")
            g2.close()

    def test_full_back_matter_excluded_from_mining_input(self):
        """back_matter=full 时 _back_matter_mode 恒不旁路，但挖掘输入仍须用 is_back_matter
        排除 Notes 等附属章——引文人名/书目标题不得混入候选。"""
        marker = "ZZQ_NOTES_MINING_MARKER"
        body = "綾小路は教室の窓際に座っていた。空はどこまでも青く鳥が鳴いていた。" + "あ" * 220
        dialog = "「おはよう、綾小路くん」と堀北が声をかけた。彼女はいつも通り無表情だった。"
        notes = f"1. Endnote {marker} on chapter one, page 12.\n\n2. Bibliography entry."
        doc = f"# 第一章 出会い\n\n{body}\n\n{dialog}\n\n# Notes\n\n{notes}\n"

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(doc)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.back_matter = "full"

            client = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=client).run(txt)

            mining_calls = [
                c for c in client.calls if "术语候选挖掘" in c["messages"][0]["content"]
            ]
            self.assertTrue(mining_calls, "应产生挖掘调用（正文章）")
            for c in mining_calls:
                self.assertNotIn(
                    marker,
                    c["messages"][-1]["content"],
                    "back_matter=full 时 Notes 章不得进入挖掘候选输入",
                )

    def test_mining_input_chapters_match_pre_overlap_semantics(self):
        """本批只把 digest/term-mining 改成重叠调度，term mining 的章节输入集合本身
        （哪些章、按什么顺序、is_back_matter 排除口径）必须与改动前完全一致——
        直接拦截 mine_candidates 的真实调用参数比对，而不仅凭候选词是否漏出判断。"""
        from unittest.mock import patch

        import trans_novel.pipeline.orchestrator as orchestrator_module

        marker = "ZZQ_NOTES_MINING_MARKER"
        body = "綾小路は教室の窓際に座っていた。空はどこまでも青く鳥が鳴いていた。" + "あ" * 220
        dialog = "「おはよう、綾小路くん」と堀北が声をかけた。彼女はいつも通り無表情だった。"
        notes = f"1. Endnote {marker} on chapter one, page 12.\n\n2. Bibliography entry."
        doc = f"# 第一章 出会い\n\n{body}\n\n{dialog}\n\n# Notes\n\n{notes}\n"

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(doc)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.back_matter = "full"  # _back_matter_mode 恒不旁路，正文照常入 digest

            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.prepare(txt)
            manifest = store.load_manifest()
            chapters = manifest["chapters"]

            from trans_novel.pipeline.backmatter import is_back_matter

            # 改动前的推导逻辑（与 orchestrator._build_understanding 里未变的过滤条件
            # 完全一致）：只用 is_back_matter 排除，不受 back_matter=full 的
            # _back_matter_mode 影响；顺序=manifest 章序。
            expected_chapter_indices = [
                c["index"]
                for c in chapters
                if not is_back_matter(
                    store.load_chapter(c["index"]).title, index=c["index"], total=len(chapters)
                )
            ]

            captured = {}
            real_mine_candidates = orchestrator_module.mine_candidates

            def _spy(src_lang, chapters_arg, agent, **kwargs):
                captured["chapters"] = list(chapters_arg)
                return real_mine_candidates(src_lang, chapters_arg, agent, **kwargs)

            with patch.object(orchestrator_module, "mine_candidates", _spy):
                orch.run(txt)

            self.assertIn("chapters", captured, "mine_candidates 必须被真实调用一次")
            actual_indices = [ci for ci, _ in captured["chapters"]]
            self.assertEqual(
                actual_indices,
                expected_chapter_indices,
                "term mining 的章节输入集合/顺序必须与改动前一致（本批只重叠调度，不新增排除）",
            )
            # 每章喂入的文本也必须是该章全部源文段落拼接（未经额外裁剪）
            for ci, text in captured["chapters"]:
                ch = store.load_chapter(ci)
                self.assertEqual(text, "\n".join(s.source for s in ch.text_segments))


class TestTitleReuse(unittest.TestCase):
    """标题复用（正文 heading 段优先）+ 标题 prompt 注入全书概览。"""

    def _title_calls(self, calls):
        return [c for c in calls if "标题翻译" in c["messages"][0]["content"]]

    def test_heading_titles_reused_no_llm_call(self):
        """两章标题都来自已译 heading 段：title_translated 取自正文，零标题 LLM 请求。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            m = store.load_manifest()
            for c in m["chapters"]:
                heading = store.load_chapter(c["index"]).segments[0]
                self.assertEqual(heading.kind, "heading")
                self.assertEqual(c["title_translated"], " ".join(heading.target.split()))
            # 全部复用，标题 agent 一次都不该被调用
            self.assertEqual(len(self._title_calls(client.calls)), 0)

    def test_non_heading_title_falls_back_to_llm_with_synopsis(self):
        """无可复用 heading 段的章 + toc_entries 仍走标题 agent；user prompt 含全书概览块，
        且已复用章的标题不重复进入 numbered 列表。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            store = Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            # 模拟章 0 无可复用 heading（如非文本源、或首段未译）：清空首段译文
            ch0 = store.load_chapter(0)
            ch0.segments[0].target = ""
            store.save_chapter(ch0)
            m = store.load_manifest()
            m["chapters"][0]["title_translated"] = None
            meta = m.setdefault("meta", {})
            meta["toc_entries"] = [{"href": "extra.xhtml", "title": "特別編"}]
            store.save_manifest(m)

            captured = {}

            def handler(messages, tier, json_mode):
                if "标题翻译" in messages[0]["content"]:
                    captured["user"] = messages[-1]["content"]
                return routing_handler(messages, tier, json_mode)

            glossary = GlossaryStore(store.glossary_path)
            client2 = FakeClient(handler=handler)
            Orchestrator(cfg, client=client2)._translate_titles(store, glossary)
            glossary.close()

            self.assertIn("user", captured)
            self.assertIn("【全书概览】", captured["user"])
            # 只有章0 + toc 条目共 2 条进入 LLM 列表（章1 已复用，不重复发送）
            self.assertEqual(len(re.findall(r"^\[(\d+)\]", captured["user"], re.M)), 2)

            m2 = store.load_manifest()
            self.assertTrue(m2["chapters"][0]["title_translated"])
            self.assertTrue(m2["meta"]["toc_entries"][0]["title_translated"])


class TestRunSteps(unittest.TestCase):
    def test_subset_only_assemble(self):
        """run_steps 步骤子集：仅回填时不应再产生翻译调用（幂等）。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            orch.run_steps(txt, {"translate"})
            # 仅回填，不应再翻译
            client2 = FakeClient(handler=routing_handler)
            res = Orchestrator(cfg, client=client2).run_steps(txt, {"assemble"})
            self.assertTrue(res["output"].endswith(".epub"))
            self.assertTrue(os.path.isfile(res["output"]))
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0)


class TestReviewReporting(unittest.TestCase):
    """章末审校 + 严重项自动重译（autofix_severe）。"""

    # 样例首段「第一章　出会い」7 字；fix 需在 3-21 字间（比值 0.3-3.0）方可通过长度校验
    FIX_TEXT = "第一章 邂逅"  # 7 字，比值 1.0

    def _handler(self, fix_text):
        """审校每块报 index 0 漏译；带【审校意见】的翻译调用返回定向重译文。"""

        def handler(messages, tier, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "译文审校" in sys:
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "missing",
                                "detail": "漏了一句",
                                "suggestion": "补上",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            if "文学翻译" in sys and "【审校意见】" in user:
                return json.dumps({"translations": [fix_text]}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        return handler

    def _run(self, d, *, autofix, fix_text=None):
        txt = os.path.join(d, "novel.txt")
        write_sample_txt(txt)
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
        """autofix 关：审校严重项仅上报 fixed=False，审校通道本身不动正文
        （lint 层独立于 autofix_severe 常开，可能另行修正与本用例无关的段落，
        不在此断言范围——只验证 review 通道未触发 autofix_applied）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=False)
            ch = store.load_chapter(0)
            flagged = [i for i in ch.meta["review_issues"] if i.get("type") == "missing"]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is False for i in flagged))
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertFalse(
                any(e["event"] == "autofix_applied" for e in events),
                "autofix 关闭时，审校严重项通道不得写回正文",
            )

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
                return json.dumps(
                    {"issues": [{"index": 0, "type": "missing", "detail": "x", "suggestion": ""}]},
                    ensure_ascii=False,
                )
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 8  # 审校块预算=24 → 每段自成一块
            cfg.pipeline.autofix_severe = False
            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)
            ch = store.load_chapter(0)
            idxs = sorted(
                i["index"] for i in ch.meta["review_issues"] if i.get("type") == "missing"
            )
            # 每块报 index 0 → 映射后应为各块首段的章内段号（0,1,2,...互不相同）
            self.assertEqual(idxs, list(range(len(ch.text_segments))))

    def test_review_accepts_numeric_string_index(self):
        def handler(messages, tier, json_mode):
            if "译文审校" in messages[0]["content"]:
                return json.dumps(
                    {
                        "issues": [
                            {"index": "0", "type": "missing", "detail": "x", "suggestion": ""}
                        ]
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)

            issues = store.load_chapter(0).meta["review_issues"]
            self.assertTrue(issues)
            self.assertEqual(issues[0]["index"], 0)

    def test_review_warns_when_index_is_invalid(self):
        def handler(messages, tier, json_mode):
            if "译文审校" in messages[0]["content"]:
                return json.dumps(
                    {
                        "issues": [
                            {"index": "unknown", "type": "missing", "detail": "x", "suggestion": ""}
                        ]
                    }
                )
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            with self.assertWarnsRegex(RuntimeWarning, "无效审校索引"):
                store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(
                    txt, only_chapter=0
                )

            # 本地 lint 层（确定性检查）也会写入 review_issues；此处只关心审校通道
            review_issues = [
                i for i in store.load_chapter(0).meta["review_issues"] if i.get("stage") != "lint"
            ]
            self.assertEqual(review_issues, [])


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
        brief = ana.style_brief(
            {
                "genre": "校园",
                "pacing": "短句为主",
                "register": "口语",
                "dialogue_style": "语气词丰富",
                "narration": "第一人称",
            }
        )
        self.assertIn("句式节奏：短句为主", brief)
        self.assertIn("语域：口语", brief)
        self.assertIn("对话风格：语气词丰富", brief)
        self.assertIn("叙事：第一人称", brief)
        # 格式约定（年代/星期/度量单位）渲染为独立行
        conv = ana.style_brief({"genre": "校园", "conventions": "年代统一用'20世纪90年代'。"})
        self.assertIn("格式约定：年代统一用'20世纪90年代'。", conv)
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
        # ①锁定人物（全章无任何形式出现）②无关术语 ③alias 在正文出现
        # ④锁定人物全名不在正文、但姓氏前缀「堀北」在正文（无空格汉字名）
        # ⑤锁定人物空格分词名，其中「綾小路」一词在正文
        g.upsert_term(GlossaryTerm(source="外部人物X", target="外部译名", type="人物", locked=True))
        g.upsert_term(GlossaryTerm(source="無関係用語", target="无关术语", type="术语"))
        g.upsert_term(
            GlossaryTerm(source="ホリキタ", target="堀北译名", aliases=["堀北"], type="术语")
        )
        g.upsert_term(GlossaryTerm(source="堀北鈴音", target="堀北铃音", type="人物", locked=True))
        g.upsert_term(
            GlossaryTerm(source="綾小路 清隆", target="绫小路清隆", type="人物", locked=True)
        )
        g.close()

        client = FakeClient(handler=routing_handler)
        Orchestrator(cfg, client=client).run(txt)
        return [
            "\n".join(m["content"] for m in c["messages"])
            for c in client.calls
            if "文学翻译" in c["messages"][0]["content"]
        ]

    def test_chapter_scope_prunes(self):
        """chapter：本章无关的锁定人物剔除；部分称呼（姓氏/分词）命中的人物保留。"""
        with tempfile.TemporaryDirectory() as d:
            translate_prompts = self._run_with_terms(d, "chapter")
            self.assertTrue(translate_prompts)
            for p in translate_prompts:
                self.assertNotIn("外部人物X", p)  # 锁定人物但全章无任何形式出现：剔除
                self.assertNotIn("無関係用語", p)  # 本章未出现：剔除
                self.assertIn("ホリキタ", p)  # 别名「堀北」在两章正文均出现：保留
                self.assertIn("堀北鈴音", p)  # 姓氏前缀「堀北」在两章正文均出现：保留
            # 「綾小路」只在第一章正文出现：分词命中该章保留，第二章（放課後）剔除
            ch1 = [p for p in translate_prompts if "綾小路は教室" in p]
            ch2 = [p for p in translate_prompts if "放課後、二人は" in p]
            self.assertTrue(ch1)
            self.assertTrue(ch2)
            for p in ch1:
                self.assertIn("綾小路 清隆", p)
            for p in ch2:
                self.assertNotIn("綾小路 清隆", p)

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
                return json.dumps(
                    {"translations": ["小夏帆" for _ in range(n)]}, ensure_ascii=False
                )
            if (
                "术语" in system
                and "抽取器" in system
                and "夏帆ちゃん" in user
                and "小夏帆" in user
            ):
                return json.dumps(
                    {
                        "terms": [
                            {
                                "source": "夏帆ちゃん",
                                "target": "小夏帆",
                                "type": "称谓",
                                "aliases": ["夏帆"],
                                "note": "亲昵称呼",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(
                    "# 第一章\n\n「夏帆ちゃん」と母親が言った。\n\n夏帆ちゃんは窓の外を見た。\n"
                )
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            cfg.pipeline.inflight_glossary = True
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
                return json.dumps(
                    {"translations": ["小夏帆" for _ in range(n)]}, ensure_ascii=False
                )
            if "术语" in system and "抽取器" in system and "夏帆ちゃん" in user:
                return json.dumps(
                    {
                        "terms": [
                            {
                                "source": "夏帆ちゃん",
                                "target": "小夏帆",
                                "type": "称谓",
                                "aliases": ["夏帆"],
                                "note": "亲昵称呼",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
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
            cfg.pipeline.inflight_glossary = True
            cfg.segment.max_chars_per_batch = 200

            Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)


class TestInflightGlossary(unittest.TestCase):
    """inflight_glossary=True：旧版"译后逐批+章末抽取"路径原样保留（日文轻小说场景）。"""

    def test_legacy_extraction_path_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.inflight_glossary = True

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            extractor_calls = [
                c
                for c in client.calls
                if "术语" in c["messages"][0]["content"] and "抽取器" in c["messages"][0]["content"]
            ]
            self.assertTrue(extractor_calls, "inflight_glossary=True 时应调用旧版抽取器")

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(any(e["event"] == "batch_glossary_extracted" for e in events))
            self.assertTrue(any(e["event"] == "chapter_glossary_extracted" for e in events))

            from trans_novel.glossary.store import GlossaryStore

            g = GlossaryStore(store.glossary_path)
            self.assertIsNotNone(g.get_term("堀北"))
            g.close()


class TestNaturalizePipeline(unittest.TestCase):
    """去翻译腔升级为章级流水线环节（config.pipeline.naturalize，默认开）。"""

    NATURALIZE_MARKERS = ("书稿的母语审读编辑", "改写编辑", "两个版本", "双语翻译审核员")

    @staticmethod
    def _naturalize_handler(messages, tier, json_mode):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "书稿的母语审读编辑" in system:
            return json.dumps(
                {"issues": [{"index": 0, "quote": "别扭", "reason": "翻译腔"}]}, ensure_ascii=False
            )
        if "改写编辑" in system:
            return json.dumps({"rewritten": "这是更自然的表达"}, ensure_ascii=False)
        if "双语翻译审核员" in system:
            return json.dumps({"faithful": True, "detail": ""}, ensure_ascii=False)
        if "两个版本" in system:
            m = re.search(r"【版本 A】\n(.*?)\n\n【版本 B】\n(.*?)\n\n请判断", user, re.S)
            winner = "A" if "更自然" in m.group(1) else "B"
            return json.dumps({"winner": winner}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    def _naturalize_calls(self, calls):
        return [
            c
            for c in calls
            if any(marker in c["messages"][0]["content"] for marker in self.NATURALIZE_MARKERS)
        ]

    def test_naturalize_applied_and_meta_flag_set(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=self._naturalize_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            m = store.load_manifest()
            for ci in range(len(m["chapters"])):
                ch = store.load_chapter(ci)
                self.assertTrue(
                    ch.meta.get("naturalized"),
                    f"第 {ci} 章 naturalize 后应标记 meta['naturalized']",
                )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            applied = [e for e in events if e["event"] == "naturalize_applied"]
            self.assertTrue(applied, "嫌疑段应走完三道关卡闭环并采纳写回")

    def test_naturalize_disabled_zero_calls(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.naturalize = False

            client = FakeClient(handler=self._naturalize_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            self.assertEqual(
                self._naturalize_calls(client.calls),
                [],
                "naturalize=False 时不应发生任何 naturalize 相关调用",
            )
            m = store.load_manifest()
            for ci in range(len(m["chapters"])):
                self.assertFalse(store.load_chapter(ci).meta.get("naturalized"))

    def test_naturalize_idempotent_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            store = Orchestrator(cfg, client=FakeClient(handler=self._naturalize_handler)).run(txt)
            self.assertTrue(store.load_chapter(0).meta.get("naturalized"))

            # 模拟"naturalize 已完成、meta 已落盘，但章末 DONE 标记前中断"续跑：
            # 章状态改回 pending，meta["naturalized"] 保持 True（幂等标记未丢）。
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=self._naturalize_handler)
            Orchestrator(cfg, client=client2).run(txt, only_chapter=0)

            self.assertEqual(
                self._naturalize_calls(client2.calls),
                [],
                "meta 标记已置位，续跑不应重复 naturalize",
            )


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
                "章节梗概员": "fast",
                "全书概览员": "fast",
                "术语候选挖掘": "fast",
                "全书定名": "strong",
                "回译译者": "fast",
                "译文审校": "cheap",
                "保真度": "cheap",
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
                txt, only_chapter=0
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            translated = [
                e for e in events if e["event"] == "batch_translated" and e["chapter"] == 0
            ]
            polished = [e for e in events if e["event"] == "batch_polished" and e["chapter"] == 0]
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
                txt, only_chapter=0
            )
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
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0)  # 已有译文，批跳过，未重翻

            ch2 = store.load_chapter(0)
            # routing_handler 的润色输出按"本次调用内"的局部下标编号：该批只含 1 段
            # （原始的第 last_idx 段），单独重新提交润色后局部下标为 0 → "润0"。
            self.assertEqual(ch2.text_segments[last_idx].target, "润0")
            self.assertFalse(ch2.meta.get("pending_polish"))

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(
                any(
                    e["event"] == "batch_polished"
                    and e["chapter"] == 0
                    and e["start_index"] == last_idx
                    for e in events
                )
            )


class TestReviewAsync(unittest.TestCase):
    """review=true 且 autofix_severe=false：章末审校提交共享线程池异步跑，
    run() 返回前必须排干——issues 合并写入 chapter.meta["review_issues"]
    并发 chapter_reviewed 事件；review worker 出错不得中断 run。"""

    @staticmethod
    def _issue_handler(messages, tier, json_mode):
        # 无共享可变状态：每次调用构造新 dict，可被线程池并发调用
        if "译文审校" in messages[0]["content"]:
            return json.dumps(
                {
                    "issues": [
                        {
                            "index": 0,
                            "type": "terminology",
                            "detail": "术语不一致",
                            "suggestion": "改用对照表",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        return routing_handler(messages, tier, json_mode)

    def test_async_review_issues_persisted_before_run_returns(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Orchestrator(cfg, client=FakeClient(handler=self._issue_handler)).run(txt)

            m = store.load_manifest()
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m["chapters"]))
            for ci in range(len(m["chapters"])):
                ch = store.load_chapter(ci)
                found = [
                    i for i in ch.meta.get("review_issues", []) if i.get("type") == "terminology"
                ]
                self.assertTrue(found, f"第 {ci} 章异步审校结果未写回 meta")
                for it in found:
                    self.assertEqual(it.get("chapter"), ci)
                    self.assertEqual(it.get("stage"), "review")
                    self.assertIs(it.get("fixed"), False)

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            reviewed = {e["chapter"] for e in events if e["event"] == "chapter_reviewed"}
            self.assertEqual(
                reviewed, set(range(len(m["chapters"]))), "每章都应发 chapter_reviewed 事件"
            )

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
            self.assertTrue(
                all(c["status"] == STATUS_DONE for c in m["chapters"]),
                "审校 worker 抛异常不得阻断整章完成",
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            failed = {e["chapter"] for e in events if e["event"] == "chapter_review_failed"}
            reviewed = {e["chapter"] for e in events if e["event"] == "chapter_reviewed"}

            # (b) 载荷断言：每个审校崩溃的章都记了 chapter_review_failed——这是唯一能
            # 证明 except→chapter_review_failed 分支真的执行过的证据。若删掉该错误处理
            # （让异常穿透），异常会在机会性/收尾 drain 里抛出，run() 直接崩、拿不到
            # store，本断言必失败。
            self.assertEqual(failed, chapters, "每个审校崩溃的章都必须记 chapter_review_failed")
            # (c) 崩溃章不得发 chapter_reviewed，且 review 通道的 review_issues 未被
            # 写回（保持空）；lint 层独立于异步审校常开，可能另行写入 stage="lint" 的
            # 条目，不属于本用例断言范围。
            self.assertEqual(reviewed, set(), "审校失败的章不得发 chapter_reviewed")
            for ci in chapters:
                review_stage_issues = [
                    i
                    for i in store.load_chapter(ci).meta.get("review_issues", [])
                    if i.get("stage") == "review"
                ]
                self.assertEqual(
                    review_stage_issues,
                    [],
                    f"第 {ci} 章审校失败，review 阶段的 review_issues 不得被写入",
                )

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
            store = Orchestrator(cfg, client=FakeClient(handler=self._issue_handler)).run(txt)
            self.assertEqual(
                store.review_pending_chapters(), [], "正常收尾后不应残留任何 review_pending 标记"
            )

            # (2) 模拟崩溃窗口：章 0 已 DONE，但标记残留且审校结果被抹掉
            store.set_review_pending(0, True)
            ch = store.load_chapter(0)
            ch.meta["review_issues"] = []
            store.save_chapter(ch)
            self.assertIn(
                0, store.review_pending_chapters(), "崩溃模拟：章 0 应带 review_pending 标记"
            )

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
            self.assertEqual(
                store.review_pending_chapters(), [], "续跑写回后 review_pending 标记必须清空"
            )

            # (4c) 第二次 run 的事件里有章 0 的 chapter_reviewed
            with open(store.event_log_path, encoding="utf-8") as f:
                all_events = [json.loads(line) for line in f if line.strip()]
            second_run = all_events[events_before:]
            reviewed = {e["chapter"] for e in second_run if e["event"] == "chapter_reviewed"}
            self.assertIn(0, reviewed, "续跑应为章 0 补发 chapter_reviewed 事件")

            # (4d) 续跑只补审校，绝不重译（无 '文学翻译' 调用）
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0, "续跑只补审校，绝不重译")


class TestReviewChunkConcurrency(unittest.TestCase):
    """Reviewer chunk 并发：有界并发真实发生，且合并结果严格保持原 chunk 顺序。"""

    def test_chunks_run_concurrently_in_bounded_pool_and_merge_in_order(self):
        from concurrent.futures import ThreadPoolExecutor

        n_chunks = 3
        barrier = threading.Barrier(n_chunks, timeout=5)
        order_lock = threading.Lock()
        completion_order: list[int] = []

        def handler(messages, tier, json_mode):
            user = messages[-1]["content"]
            m = re.search(r"MARK(\d+)", user)
            assert m, "chunk marker missing from review prompt"
            c = int(m.group(1))
            # 所有 chunk 必须同时在飞才能通过 barrier——若并发退化为串行，该 wait 会
            # 一直阻塞直到 barrier 超时并抛出 BrokenBarrierError，测试失败。
            barrier.wait()
            # 提交顺序 c=0,1,2；故意让完成顺序反转（c 越大睡得越少），验证结果
            # 合并顺序仍按 chunk 原始顺序而非完成顺序。
            time.sleep((n_chunks - c) * 0.03)
            with order_lock:
                completion_order.append(c)
            return json.dumps(
                {"issues": [{"index": 0, "type": "missing", "detail": f"chunk{c}"}]},
                ensure_ascii=False,
            )

        cfg = _config("state")
        cfg.segment.max_chars_per_batch = 1  # budget=3，强制每对独立成块
        client = FakeClient(handler=handler)
        orch = Orchestrator(cfg, client=client)
        pairs = [(f"MARK{c} " + "源文" * 10, f"译文{c}") for c in range(n_chunks)]

        with ThreadPoolExecutor(max_workers=4) as review_executor:
            issues = orch._review_chapter(pairs, [], review_executor)

        self.assertEqual([it["detail"] for it in issues], [f"chunk{c}" for c in range(n_chunks)])
        # 完成顺序确实被反转了（证明真的并发，且合并未按完成顺序）
        self.assertEqual(completion_order, list(reversed(range(n_chunks))))

    def test_book_wide_pool_bounds_concurrency_across_chapters_without_deadlock(self):
        """异步路径（review=true, autofix_severe=false）多章并发提交审校：
        book-wide review_executor 上限生效、且不因"任务等自己"而死锁。"""
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def handler(messages, tier, json_mode):
            nonlocal max_concurrent, current
            if "译文审校" in messages[0]["content"]:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.01)
                with lock:
                    current -= 1
                return json.dumps({"issues": []}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)
            # 未死锁即已经证明池分离设计成立；再断言并发确实发生过且没有超过硬上限 4。
            self.assertGreaterEqual(max_concurrent, 1)
            self.assertLessEqual(max_concurrent, 4)
            m = store.load_manifest()
            self.assertTrue(all(c["status"] == STATUS_DONE for c in m["chapters"]))


class TestPrescanOverlap(unittest.TestCase):
    """digest 与 term mining 真正并发；naming 等待两者收尾；异常语义不变。"""

    def test_digest_and_mining_genuinely_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            digest_reached = threading.Event()
            mining_reached = threading.Event()
            overlapped = threading.Event()

            def handler(messages, tier, json_mode):
                system = messages[0]["content"]
                if "梗概员" in system:
                    digest_reached.set()
                    if mining_reached.wait(timeout=2):
                        overlapped.set()
                    return "本章梗概：人物登场，情节推进。"
                if "术语候选挖掘" in system:
                    mining_reached.set()
                    if digest_reached.wait(timeout=2):
                        overlapped.set()
                    return json.dumps({"candidates": ["堀北"]}, ensure_ascii=False)
                return routing_handler(messages, tier, json_mode)

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)

            self.assertTrue(
                overlapped.is_set(),
                "digest 与 term mining 必须真正同时在跑，而非先后串行执行",
            )
            self.assertTrue((store.load_analysis() or {}).get("term_mining_done"))

    def test_naming_waits_for_slower_mining_branch(self):
        """mining 分支人为拖慢：naming（全书定名）调用必须等它彻底收尾才发生。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            lock = threading.Lock()
            mining_finished_at: list[float] = []
            naming_started_at: list[float] = []

            def handler(messages, tier, json_mode):
                system = messages[0]["content"]
                if "术语候选挖掘" in system:
                    time.sleep(0.1)  # 故意拖慢挖掘分支
                    with lock:
                        mining_finished_at.append(time.monotonic())
                    return json.dumps({"candidates": ["堀北"]}, ensure_ascii=False)
                if "全书定名" in system:
                    with lock:
                        naming_started_at.append(time.monotonic())
                return routing_handler(messages, tier, json_mode)

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt)

            self.assertTrue(mining_finished_at and naming_started_at)
            self.assertGreaterEqual(
                min(naming_started_at),
                max(mining_finished_at),
                "naming 必须等挖掘分支（含人为拖慢的每一章）全部收尾才能开始",
            )
            self.assertTrue((store.load_analysis() or {}).get("term_mining_done"))
            g = GlossaryStore(store.glossary_path)
            self.assertIsNotNone(g.get_term("堀北"), "naming 必须拿到挖掘分支的完整候选")
            g.close()

    def test_digest_exception_precedence_after_draining_mining_branch(self):
        """digest 分支异常整体冒泡（旧同步语义），但挖掘分支必须先被排干；
        term_mining_done 不得因此被落盘。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            store = orch.prepare(txt)

            def _boom(source_text):
                raise RuntimeError("digest 崩")

            orch.synopsizer.digest_chapter = _boom  # 遮蔽实例方法，绕过 _ask_text 的吞异常

            with self.assertRaises(RuntimeError):
                orch.run(txt)

            reloaded_analysis = store.load_analysis() or {}
            self.assertFalse(
                reloaded_analysis.get("term_mining_done"),
                "digest 异常时挖掘/定名结果不得被落盘为完成",
            )


class TestOperationOutcomeAccounting(unittest.TestCase):
    """业务采纳结果写入对应 operation 槽位的 accepted/rejected（decision 19/53）。"""

    def test_polish_batch_accepted_when_no_new_lint_issue(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))  # polish=True
            client = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["polish.batch"]
            self.assertGreater(op["accepted"], 0)
            self.assertEqual(op["rejected"], 0)

    def test_polish_batch_rejected_when_introduces_new_lint_issue(self):
        src = "「おはようございます」と彼は静かな声で言った。"

        def handler(messages, tier, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in sys:
                n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
                return json.dumps(
                    {"translations": ["“早上好”他轻声说道" for _ in range(n)]}, ensure_ascii=False
                )
            if "中文润色编辑" in sys:
                target_block = user.split("【待润色中文译文】", 1)[-1]
                n = len(re.findall(r"^\[(\d+)\] ", target_block, re.M))
                return json.dumps(
                    {"polished": ["早上好他轻声说道" for _ in range(n)]}, ensure_ascii=False
                )
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(f"# 第一章\n\n{src}\n")
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            client = FakeClient(handler=handler)
            Orchestrator(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["polish.batch"]
            self.assertGreaterEqual(op["rejected"], 1)

    def test_lint_fix_accepted_when_retranslation_reduces_issues(self):
        src = "「おはようございます」と彼は静かな声で言った。窓の外には青い空が広がっていた。"

        def handler(messages, tier, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in sys:
                n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
                if "【审校意见】" in user:
                    out = ["“早上好”他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
                else:
                    out = ["早上好他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
                return json.dumps({"translations": out}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(f"# 第一章\n\n{src}\n")
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            client = FakeClient(handler=handler)
            Orchestrator(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["translate.lint_fix"]
            self.assertEqual(op["accepted"], 1)
            self.assertEqual(op["rejected"], 0)

    def test_review_fix_accepted_and_rejected_recorded(self):
        FIX_TEXT = "第一章 邂逅"  # 7 字，比值 1.0，通过长度校验 → accepted

        def handler(fix_text):
            def h(messages, tier, json_mode):
                sys = messages[0]["content"]
                user = messages[-1]["content"]
                if "译文审校" in sys:
                    return json.dumps(
                        {
                            "issues": [
                                {
                                    "index": 0,
                                    "type": "missing",
                                    "detail": "漏了一句",
                                    "suggestion": "补上",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                if "文学翻译" in sys and "【审校意见】" in user:
                    return json.dumps({"translations": [fix_text]}, ensure_ascii=False)
                return routing_handler(messages, tier, json_mode)

            return h

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)

            cfg_accept = _config(os.path.join(d, "state_accept"))
            cfg_accept.pipeline.autofix_severe = True
            accepted_client = FakeClient(handler=handler(FIX_TEXT))
            Orchestrator(cfg_accept, client=accepted_client).run(txt)
            op = accepted_client.usage_summary()["by_operation"]["translate.review_fix"]
            self.assertGreaterEqual(op["accepted"], 1)
            self.assertEqual(op["rejected"], 0)

            cfg_reject = _config(os.path.join(d, "state_reject"))
            cfg_reject.pipeline.autofix_severe = True
            rejected_client = FakeClient(handler=handler("短"))  # 过短，长度校验不通过
            Orchestrator(cfg_reject, client=rejected_client).run(txt)
            op2 = rejected_client.usage_summary()["by_operation"]["translate.review_fix"]
            self.assertEqual(op2["accepted"], 0)
            self.assertGreaterEqual(op2["rejected"], 1)


class TestOperationLabelCompleteness(unittest.TestCase):
    """production 调用一律显式标注 operation，不留 class-name-only 空档（decision 44/59）。"""

    def test_no_production_call_has_blank_operation(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.backtranslate_sample = 1.0  # 强制触发回译抽检

            client = FakeClient(handler=routing_handler)
            orch = Orchestrator(cfg, client=client)
            orch.run_steps(txt, {"translate", "qa", "report"})

            blank = [c for c in client.calls if not c.get("operation")]
            self.assertEqual(
                blank,
                [],
                f"生产调用不得省略 operation 标签，缺失的 system 前缀："
                f"{[c['messages'][0]['content'][:30] for c in blank]}",
            )


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

            store = Orchestrator(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)

            m = store.load_manifest()
            self.assertEqual(m["chapters"][0]["status"], STATUS_DONE, "润色失败不得阻断整章完成")
            ch = store.load_chapter(0)
            # 单批：routing_handler 译文按批内下标编号 → 段 i 的 raw 译文为 "译{i}"
            expected = [normalize_zh(f"译{i}") for i in range(len(ch.text_segments))]
            self.assertEqual([s.target for s in ch.text_segments], expected)
            self.assertFalse(
                ch.meta.get("pending_polish"), "润色失败的批次也必须清掉 pending_polish 标记"
            )


class TestLintQuoteRefix(unittest.TestCase):
    """批循环内 lint：首译丢引号 → 定向重译修复（事件 batch_linted + lint_refixed）。"""

    SRC = "「おはようございます」と彼は静かな声で言った。窓の外には青い空が広がっていた。"

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}\n")

    @staticmethod
    def _handler(messages, tier, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            if "【审校意见】" in user:
                # 定向重译：修复丢引号问题，补回成对引号
                out = ["“早上好”他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
            else:
                # 首译：丢引号（触发 quote_loss）
                out = ["早上好他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
            return json.dumps({"translations": out}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    def test_quote_loss_caught_and_refixed(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=self._handler)).run(txt)
            ch = store.load_chapter(0)
            target = ch.text_segments[1].target
            # 最终译文带引号（定向重译采纳，替换了丢引号的首译）
            self.assertTrue(
                any(q in target for q in "“”「」『』"), f"最终译文应保留引号：{target!r}"
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            linted = [e for e in events if e["event"] == "batch_linted" and e["chapter"] == 0]
            refixed = [e for e in events if e["event"] == "lint_refixed" and e["chapter"] == 0]
            self.assertTrue(linted, "丢引号首译应触发 batch_linted")
            self.assertTrue(
                any(i["type"] == "quote_loss" for e in linted for i in e["issues"]),
                "batch_linted 摘要应含 quote_loss",
            )
            self.assertTrue(refixed, "重译修复后应发 lint_refixed")
            self.assertEqual(refixed[0]["index"], 1)
            self.assertNotIn("“", refixed[0]["before"])
            self.assertIn("“", refixed[0]["after"])


class TestPolishQuoteRejection(unittest.TestCase):
    """章末排干润色：润色剥掉引号（引入新 lint issue）→ polish_rejected，保留润色前译文。"""

    SRC = "「おはようございます」と彼は静かな声で言った。"

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}\n")

    @staticmethod
    def _handler(messages, tier, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            return json.dumps(
                {"translations": ["“早上好”他轻声说道" for _ in range(n)]}, ensure_ascii=False
            )
        if "中文润色编辑" in sys:
            target_block = user.split("【待润色中文译文】", 1)[-1]
            n = len(re.findall(r"^\[(\d+)\] ", target_block, re.M))
            # 润色把引号剥掉了——引入新的 quote_loss，应被回退拒绝
            return json.dumps(
                {"polished": ["早上好他轻声说道" for _ in range(n)]}, ensure_ascii=False
            )
        return routing_handler(messages, tier, json_mode)

    def test_polish_stripping_quotes_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = True
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=self._handler)).run(txt)
            ch = store.load_chapter(0)
            target = ch.text_segments[1].target
            # 润色剥引号被拒：保留润色前（带引号）译文，而非润色后的无引号文本
            self.assertTrue(
                any(q in target for q in "“”「」『』"), f"应保留润色前带引号的译文：{target!r}"
            )
            self.assertIn("早上好", target)

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            rejected = [e for e in events if e["event"] == "polish_rejected"]
            self.assertTrue(rejected, "润色剥引号应触发 polish_rejected")
            self.assertEqual(rejected[0]["chapter"], 0)
            self.assertEqual(rejected[0]["index"], 1)
            self.assertIn("quote_loss", rejected[0]["reason"])
            self.assertNotIn("“", rejected[0]["polished"])


class TestLintTooShortReportOnly(unittest.TestCase):
    """too_short/too_long 降为 report-only：批循环 lint 发现但不触发定向重译。"""

    SRC = (
        "This is a sufficiently long English source sentence, padded with "
        "extra descriptive words, written specifically so its character "
        "count comfortably clears the one-twenty threshold for this test."
    )

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Chapter One\n\n{self.SRC}\n")

    @staticmethod
    def _handler(messages, tier, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            return json.dumps({"translations": ["short" for _ in range(n)]}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    def test_too_short_reported_but_not_retranslated(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = Config.from_dict(
                {
                    "language": {"source": "en", "target": "zh"},
                    "llm": {
                        "provider": "fake",
                        "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
                    },
                    "segment": {"max_chars_per_batch": 1800},
                    "pipeline": {
                        "review": False,
                        "polish": False,
                        "backtranslate_sample": 0.0,
                        "consistency_qa": False,
                        "book_understanding": False,
                    },
                    "paths": {"state_dir": os.path.join(d, "state")},
                }
            )
            client = FakeClient(handler=self._handler)
            store = Orchestrator(cfg, client=client).run(txt)

            translate_calls = [c for c in client.calls if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(
                len(translate_calls), 1, "too_short 不属于可重译类型，不该触发定向重译"
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertFalse(any(e["event"] == "lint_refixed" for e in events))
            linted = [e for e in events if e["event"] == "batch_linted"]
            self.assertTrue(
                any(i["type"] == "too_short" for e in linted for i in e["issues"]),
                "过短译文应被 lint 发现并记入 batch_linted",
            )

            ch = store.load_chapter(0)
            recorded = [
                i
                for i in ch.meta["review_issues"]
                if i.get("type") == "too_short" and i.get("stage") == "lint"
            ]
            self.assertTrue(recorded, "too_short 应作为 report-only 记入 review_issues")
            self.assertTrue(all(i.get("fixed") is False for i in recorded))


class TestLintSkipBranchRecordsIssue(unittest.TestCase):
    """崩溃续跑：已译批次走跳过分支时，确定性 lint 仍复检一遍记录未修复项（不重译）。"""

    SRC = "「おはよう」と彼は言った。"

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}\n")

    @staticmethod
    def _handler(messages, tier, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            # 首译、定向重译都返回丢引号译文（模拟修复失败，issue 保持未解决）
            return json.dumps(
                {"translations": ["早上好他说道" for _ in range(n)]}, ensure_ascii=False
            )
        return routing_handler(messages, tier, json_mode)

    def test_skip_branch_relints_without_retranslating(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Orchestrator(cfg, client=FakeClient(handler=self._handler)).run(txt)
            ch = store.load_chapter(0)
            unresolved = [
                i
                for i in ch.meta["review_issues"]
                if i.get("type") == "quote_loss" and i.get("stage") == "lint"
            ]
            self.assertTrue(unresolved, "首译丢引号且重译未修复，应作为未解决 lint issue 记录")

            # 模拟崩溃窗口：章已 DONE 但 review_issues 被清空、状态改回 pending，
            # 段译文原样保留（已落盘、未变）——续跑应走批跳过分支，不重译。
            ch.meta["review_issues"] = []
            store.save_chapter(ch)
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=self._handler)
            store2 = Orchestrator(cfg, client=client2).run(txt)

            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0, "跳过分支不得重译")

            ch2 = store2.load_chapter(0)
            recovered = [
                i
                for i in ch2.meta["review_issues"]
                if i.get("type") == "quote_loss" and i.get("stage") == "lint"
            ]
            self.assertTrue(recovered, "跳过分支应重新记录未修复的 lint issue，不静默丢失")


class TestProgressLabels(unittest.TestCase):
    """进度回调覆盖译前/译中/译后全阶段——防止新增阶段静默停在"准备中"。

    digest（通读全书章节…）与 term mining（查找专有名词…）现在真正并发跑，二者的
    进度标签彼此交织、相对顺序不确定；只断言宏观阶段边界仍严格有序（各自都晚于
    分析全书风格，且都在纳入"统一译名…"（naming 等待两者）之前收尾）。
    """

    def test_stage_labels_appear_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            labels: list[str] = []
            orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
            orch.run_steps(
                txt,
                {"translate", "qa", "report", "assemble"},
                progress=lambda done, total, label: labels.append(label),
            )
            for label in (
                "读取原书…",
                "分析全书风格…",
                "通读全书章节…",
                "查找专有名词…",
                "统一译名…",
                "生成全书概览…",
                "翻译完成",
                "检查全书一致性…",
                "生成报告…",
                "生成译文文件…",
            ):
                self.assertIn(label, labels, f"缺失阶段标签：{label}；实际序列={labels}")

            def first(lbl):
                return labels.index(lbl)

            def last(lbl):
                return len(labels) - 1 - labels[::-1].index(lbl)

            self.assertLess(first("读取原书…"), first("分析全书风格…"))
            self.assertLess(first("分析全书风格…"), first("通读全书章节…"))
            self.assertLess(first("分析全书风格…"), first("查找专有名词…"))
            # naming（统一译名…）必须等 digest 与 mining 两条并发分支都收尾——
            # 二者各自的最后一次进度回调都要早于统一译名的首次回调。
            self.assertLess(last("通读全书章节…"), first("统一译名…"))
            self.assertLess(last("查找专有名词…"), first("统一译名…"))
            self.assertLess(last("统一译名…"), first("生成全书概览…"))
            self.assertLess(last("生成全书概览…"), first("翻译完成"))
            self.assertLess(first("翻译完成"), first("检查全书一致性…"))
            self.assertLess(first("检查全书一致性…"), first("生成报告…"))
            self.assertLess(first("生成报告…"), first("生成译文文件…"))


if __name__ == "__main__":
    unittest.main()
