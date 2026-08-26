"""工作流端到端 + 断点续跑测试（离线 FakeClient）。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import unittest
import warnings
from unittest import mock

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import (
    write_grouped_nav_epub,
    write_nested_toc_epub,
    write_sample_epub,
    write_sample_txt,
)
from trans_novel.agents.reviewer import ReviewOutputError
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import Chapter, Segment
from trans_novel.llm import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.contracts import ExecutionGoal, assemble_goal
from trans_novel.pipeline.nodes.prepare import _normalize_lang
from trans_novel.pipeline.runner import RequiredNodeFailed
from trans_novel.pipeline.runstore import (
    STATUS_DONE,
    STATUS_PENDING,
    RunStore,
    slugify,
    stable_digest,
)
from trans_novel.pipeline.state import (
    NODE_ANALYZE,
    NODE_BACKTRANSLATE,
    NODE_BOOK_SYNOPSIS,
    NODE_DIGEST,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_NATURALIZE,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REVIEW,
    NODE_TITLES,
    NODE_TRANSLATE,
    RUN_STATE_SCHEMA_VERSION,
    ChapterIndex,
    ChapterProgress,
    NodeState,
    PolishBatch,
    RunIdentity,
    RunState,
    chapter_node_key,
)


def _stamp_completed_store(store, *, chapters: int) -> None:
    """模拟“已完整翻译的书”：为手工构造的 RunStore 补全节点成功态。

    服务目标（translate_titles 等）经依赖闭包只信任持久化 succeeded 状态；手工
    fixture 缺节点状态会被闭包误判为“未翻译”而重新规划整条链，进而触发身份校验
    （合成状态没有可读源文件）。补全后闭包直接满足，只跑目标节点。
    """
    state = store.load_state()
    for node_id in (
        NODE_PREPARE,
        NODE_ANALYZE,
        NODE_MINE_TERMS,
        NODE_NAME_TERMS,
        NODE_BOOK_SYNOPSIS,
    ):
        state.nodes[node_id] = NodeState(node_id=node_id, status="succeeded")
    # 不补 NODE_TITLES：标题测试需要 titles 节点真的执行（补上会被闭包判为已满足）。
    for ci in range(chapters):
        for node_id in (
            NODE_DIGEST,
            NODE_TRANSLATE,
            NODE_POLISH,
            NODE_NATURALIZE,
            NODE_REVIEW,
            NODE_BACKTRANSLATE,
        ):
            key = chapter_node_key(node_id, ci)
            state.nodes[key] = NodeState(node_id=key, status="succeeded")
    store.save_state(state)


def _translated_para_count(calls) -> int:
    """统计送进翻译模型的源段总数（按编号行计）。"""
    n = 0
    for c in calls:
        if "文学翻译" in c["messages"][0]["content"]:
            n += len(re.findall(r"^\[(\d+)\]", c["messages"][-1]["content"], re.M))
    return n


def _config(state_dir: str):
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
    config.source_lang = "ja"
    config.state_dir = state_dir
    config.pipeline.backtranslate_sample = 0
    return config


def _epub_config(state_dir: str):
    """供英文源 EPUB 样书使用（write_nested_toc_epub / write_grouped_nav_epub 生成的内容均为英文）。"""
    config = _config(state_dir)
    config.source_lang = "en"
    return config


def _title_calls(calls):
    return [c for c in calls if "标题翻译" in c["messages"][0]["content"]]


def _review_node(cfg, client):
    """构造一个可直接调用 review_chapter 的 ReviewNode（绕过 runner）。"""
    import tempfile

    from trans_novel.agents.reviewer import Reviewer
    from trans_novel.agents.translator import Translator
    from trans_novel.glossary.store import GlossaryStore
    from trans_novel.pipeline.nodes.quality import ReviewNode

    glossary = GlossaryStore(os.path.join(tempfile.mkdtemp(), "glossary.db"))
    return ReviewNode(
        reviewer=Reviewer(client, cfg),
        translator=Translator(client, cfg),
        glossary=glossary,
        config=cfg,
        style_brief="",
    )


class TestEpubResumeSchemaGate(unittest.TestCase):
    def test_old_schema_rejected_before_provider_calls(self):
        from trans_novel.ingest.segmenter import load_document

        for schema in (None, 1, 2):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                epub = os.path.join(directory, "book.epub")
                write_sample_epub(epub)
                document = load_document(epub, "ja", "zh")
                state_dir = os.path.join(directory, "state")
                run_dir = os.path.join(state_dir, slugify(document.title))
                os.makedirs(run_dir, exist_ok=True)
                meta = {} if schema is None else {"epub_schema": schema}
                with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "title": document.title,
                            "fmt": "epub",
                            "source_path": epub,
                            "source_lang": "ja",
                            "target_lang": "zh",
                            "meta": meta,
                            "chapters": [],
                        },
                        stream,
                        ensure_ascii=False,
                    )

                client = FakeClient(handler=routing_handler)
                with self.assertRaisesRegex(ValueError, "fresh translation"):
                    Application(_config(state_dir), client=client).run(epub)
                self.assertEqual(client.calls, [])


class TestApplication(unittest.TestCase):
    def test_prepare_retries_after_analysis_failure(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            def fail_analysis(messages, agent, operation, json_mode):
                raise RuntimeError("temporary model failure")

            with self.assertRaisesRegex(RuntimeError, "temporary model failure"):
                Application(cfg, client=FakeClient(handler=fail_analysis)).prepare(txt)

            run_dirs = [os.path.join(cfg.state_dir, name) for name in os.listdir(cfg.state_dir)]
            self.assertEqual(len(run_dirs), 1)
            self.assertFalse(os.path.isfile(os.path.join(run_dirs[0], "manifest.json")))

            store = Application(cfg, client=FakeClient(handler=routing_handler)).prepare(txt)
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
            orch = Application(cfg, client=client)
            store = orch.run(txt)

            # 全部章节标记 done
            m = store.load_manifest()
            self.assertEqual(len(m["chapters"]), 2)
            self.assertTrue(
                all(store.chapter_status(c["index"]) == STATUS_DONE for c in m["chapters"])
            )

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
            self.assertGreater(g.stats()["terms"], 0)  # 术语库已写入
            g.close()
            extractor_calls = [
                c
                for c in client.calls
                if "术语" in c["messages"][0]["content"] and "抽取器" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(extractor_calls), 0, "默认路径不得调用旧版抽取器")
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)

            # ── 续跑：所有章已 done，不应再产生翻译调用；也不应重复定名 ──
            client2 = FakeClient(handler=routing_handler)
            orch2 = Application(cfg, client=client2)
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
            orch = Application(cfg, client=client)
            # 只翻第 0 章
            store = orch.run(txt, only_chapter=0)
            self.assertEqual(store.chapter_status(0), STATUS_DONE)
            self.assertNotEqual(store.chapter_status(1), STATUS_DONE)

            # 续跑应只补翻第 1 章
            client2 = FakeClient(handler=routing_handler)
            orch2 = Application(cfg, client=client2)
            store2 = orch2.run(txt)
            m2 = store2.load_manifest()
            self.assertTrue(
                all(store2.chapter_status(c["index"]) == STATUS_DONE for c in m2["chapters"])
            )


class TestSegmentLevelResume(unittest.TestCase):
    def _tr_handler(self, tag):
        """返回带标记的翻译 handler（译文形如 {tag}译{i}），其余走默认路由。
        用原文长度补齐译文（填充字符），避免触发新增确定性 lint 的 too_short
        判定——本类只测续跑/段级幂等，不是 lint 的测试范围。"""

        def handler(messages, agent, operation, json_mode):
            if "文学翻译" in messages[0]["content"]:
                user = messages[-1]["content"]
                pairs = re.findall(r"^\[(\d+)\] (.*)$", user, re.M)
                out = []
                for i, src in pairs:
                    base = f"{tag}译{i}"
                    out.append(base + "文" * max(0, len(src) - len(base)))
                return json.dumps({"translations": out}, ensure_ascii=False)
            return routing_handler(messages, agent, operation, json_mode)

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
            store = Application(cfg, client=c1).run(txt, only_chapter=0)
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target and s.target.startswith("R1") for s in ch.text_segments))

            # 模拟中断：清空最后一段译文、章状态改回 pending
            ch.segments[-1].target = ""
            store.save_chapter(ch)
            store.set_chapter_status(0, STATUS_PENDING)

            # 第二次：用 R2 续跑——只应补译被清空的那 1 段
            c2 = FakeClient(handler=self._tr_handler("R2"))
            Application(cfg, client=c2).run(txt, only_chapter=0)
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
            store = Application(cfg, client=first_client).run(txt, only_chapter=0)
            chapter = store.load_chapter(0)
            chapter.text_segments[-1].target = ""
            store.save_chapter(chapter)
            store.set_chapter_status(0, STATUS_PENDING)

            # 改变预算后，新分批仍可能把已完成的段与待翻译的段放在一起。
            cfg.segment.max_chars_per_batch = 50_000
            second_client = FakeClient(handler=self._tr_handler("R2"))
            Application(cfg, client=second_client).run(txt, only_chapter=0)

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
            store = Application(cfg, client=client).run(txt)

            # 逐章梗概落盘到 ChapterProgress
            self.assertTrue(store.load_progress(0).source_digest)
            # 全书概览落盘到 analysis
            self.assertTrue((store.load_analysis() or {}).get("book_synopsis"))

            # 翻译 prompt 注入了全书概览 / 本章梗概块（且非「（无）」占位）
            user = self._translate_user(client.calls)
            self.assertIn("【全书概览】", user)
            self.assertIn("【本章梗概】", user)
            self.assertIn("全书概览", user)  # fake 概览正文
            self.assertIn("本章梗概", user)  # fake 逐章梗概正文

    def test_prepare_for_translation_stops_before_body_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            text_path = os.path.join(directory, "novel.txt")
            write_sample_txt(text_path)
            config = _config(os.path.join(directory, "state"))
            client = FakeClient(handler=routing_handler)

            store = Application(config, client=client).prepare_for_translation(text_path)

            manifest = store.load_manifest()
            self.assertTrue((store.load_analysis() or {}).get("book_synopsis"))
            glossary = GlossaryStore(store.glossary_path)
            try:
                self.assertGreater(glossary.stats()["terms"], 0)
            finally:
                glossary.close()
            for item in manifest["chapters"]:
                chapter = store.load_chapter(item["index"])
                self.assertTrue(store.load_progress(item["index"]).source_digest)
                self.assertTrue(all(segment.target is None for segment in chapter.segments))
            self.assertEqual(_translated_para_count(client.calls), 0)

    def test_prescan_parallel(self):
        """并行预扫：多线程 digest 后各章梗概按章序落盘，翻译注入正常。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.prescan_concurrency = 3

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(txt)

            m = store.load_manifest()
            for c in m["chapters"]:
                self.assertTrue(store.load_progress(c["index"]).source_digest)
            self.assertTrue((store.load_analysis() or {}).get("book_synopsis"))
            user = self._translate_user(client.calls)
            self.assertIn("【本章梗概】", user)

    def test_resume_skips_prepass(self):
        """续跑：梗概/概览已落盘，不再产生预扫调用。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            c2 = FakeClient(handler=routing_handler)
            Application(cfg, client=c2).run(txt)
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
            store = Application(cfg, client=client).run(txt)

            self.assertFalse(store.load_progress(0).source_digest)
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

        def failing_handler(messages, agent, operation, json_mode):
            if "全书定名" in messages[0]["content"]:
                raise RuntimeError("routed model timeout")
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=failing_handler)
            store = Application(cfg, client=client).run(txt)

            self.assertFalse(
                store.load_state().analysis_flags.term_mining_done,
                "定名异常时不得落盘 term_mining_done",
            )
            g = GlossaryStore(store.glossary_path)
            self.assertIsNone(g.get_term("堀北"))
            g.close()
            failed = [e for e in self._events(store) if e["event"] == "cast_naming_failed"]
            self.assertTrue(failed, "应记录 cast_naming_failed 事件")

            # 续跑：换正常 handler，应重试挖掘/定名并成功落盘（不是静默永久跳过）
            client2 = FakeClient(handler=routing_handler)
            store2 = Application(cfg, client=client2).run(txt)
            self.assertTrue(store2.load_state().analysis_flags.term_mining_done)
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

            orch = Application(cfg, client=FakeClient(handler=routing_handler))
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

            Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

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
            Application(cfg, client=client).run(txt)

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

        import trans_novel.pipeline.nodes.prescan as prescan_module

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
            orch = Application(cfg, client=client)
            store = orch.prepare(txt)
            manifest = store.load_manifest()
            chapters = manifest["chapters"]

            from trans_novel.pipeline.backmatter import is_back_matter

            # 改动前的推导逻辑（与迁移前 _build_understanding 里未变的过滤条件
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
            real_mine_candidates = prescan_module.mine_candidates

            def _spy(src_lang, chapters_arg, agent, **kwargs):
                captured["chapters"] = list(chapters_arg)
                return real_mine_candidates(src_lang, chapters_arg, agent, **kwargs)

            with patch.object(prescan_module, "mine_candidates", _spy):
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

    def test_heading_titles_reused_no_llm_call(self):
        """两章标题都来自已译 heading 段：title_translated 取自正文，零标题 LLM 请求。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(txt)

            m = store.load_manifest()
            for c in m["chapters"]:
                heading = store.load_chapter(c["index"]).segments[0]
                self.assertEqual(heading.kind, "heading")
                self.assertEqual(c["title_translated"], " ".join(heading.target.split()))
            # 全部复用，标题 agent 一次都不该被调用
            self.assertEqual(len(_title_calls(client.calls)), 0)

    def test_non_heading_title_falls_back_to_llm_with_synopsis(self):
        """无可复用 heading 段的章 + toc_entries 仍走标题 agent；user prompt 含全书概览块，
        且已复用章的标题不重复进入 numbered 列表。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            # 模拟章 0 无可复用 heading（如非文本源、或首段未译）：清空首段译文
            ch0 = store.load_chapter(0)
            ch0.segments[0].target = ""
            store.save_chapter(ch0)
            m = store.load_manifest()
            m["chapters"][0]["title_translated"] = None
            meta = m.setdefault("meta", {})
            # 额外构造一条未映射到任何章节的 toc entry（如 depth > 0 的子节点），字段遵循 schema 2。
            meta["toc_entries"] = [
                {"entry_id": "extra.ncx:0", "title": "特別編", "external": False}
            ]
            store.save_manifest(m)

            captured = {}

            def handler(messages, agent, operation, json_mode):
                if "标题翻译" in messages[0]["content"]:
                    captured["user"] = messages[-1]["content"]
                    captured["agent"] = agent
                    captured["operation"] = operation
                return routing_handler(messages, agent, operation, json_mode)

            glossary = GlossaryStore(store.glossary_path)
            client2 = FakeClient(handler=handler)
            Application(cfg, client=client2).translate_titles(store)
            glossary.close()

            self.assertIn("user", captured)
            self.assertEqual(captured["agent"], "translator")
            self.assertEqual(captured["operation"], "title.translate")
            self.assertIn("【全书概览】", captured["user"])
            # 只有章0 + toc 条目共 2 条进入 LLM 列表（章1 已复用，不重复发送）
            self.assertEqual(len(re.findall(r"^\[(\d+)\]", captured["user"], re.M)), 2)

            m2 = store.load_manifest()
            self.assertTrue(m2["chapters"][0]["title_translated"])
            self.assertTrue(m2["meta"]["toc_entries"][0]["title_translated"])


class TestEpubTitleTranslation(unittest.TestCase):
    """schema 2 下，TOC entry 驱动标题翻译：先翻译 entry，再按 toc_entry_id 回填章节。"""

    def test_toc_boundary_titles_synced_from_entry(self):
        """所有章节均由 TOC 边界定义：每个 toc entry（包括不作为边界的子节点）都会写入
        title_translated；边界章节的 title_translated 与对应 entry 完全一致。"""
        with tempfile.TemporaryDirectory() as d:
            epub = os.path.join(d, "nested.epub")
            write_nested_toc_epub(epub)
            cfg = _epub_config(os.path.join(d, "state"))

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(epub)

            m = store.load_manifest()
            entries = m["meta"]["toc_entries"]
            self.assertEqual(len(entries), 4)  # PART I / Section 1 / PART II / Section 2
            for entry in entries:
                self.assertTrue(entry.get("title_translated"))

            entries_by_id = {e["entry_id"]: e for e in entries}
            chapters = m["chapters"]
            self.assertEqual(len(chapters), 2)
            for c in chapters:
                self.assertTrue(c["toc_entry_id"])
                self.assertEqual(
                    c["title_translated"], entries_by_id[c["toc_entry_id"]]["title_translated"]
                )

    def test_grouped_part_title_not_confused_with_child_heading(self):
        """无 href 的“部”级 entry 会继承子节点边界，但保留自己的标题。即使章节首段是子标题
        （文本与“部”标题不同），该章节的 title_translated 仍来自对应 entry 的翻译，
        不会误用正文 heading 的译文。"""
        with tempfile.TemporaryDirectory() as d:
            epub = os.path.join(d, "grouped.epub")
            write_grouped_nav_epub(epub)
            cfg = _epub_config(os.path.join(d, "state"))

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(epub)

            m = store.load_manifest()
            entries_by_id = {e["entry_id"]: e for e in m["meta"]["toc_entries"]}
            for entry in entries_by_id.values():
                self.assertTrue(entry.get("title_translated"))

            chapters = m["chapters"]
            self.assertEqual([c["title"] for c in chapters], ["PART I", "PART II"])
            for c in chapters:
                entry = entries_by_id[c["toc_entry_id"]]
                self.assertEqual(entry["title"], c["title"])
                self.assertEqual(c["title_translated"], entry["title_translated"])
                # 章首段实为子章节标题（"Section N"），其正文译文不应等于部标题译文
                # ——证明部标题没有被误当成正文 heading 复用。
                first_seg = store.load_chapter(c["index"]).segments[0]
                self.assertNotEqual(first_seg.source, c["title"])
                self.assertNotEqual(c["title_translated"], first_seg.target)

    def test_spine_fallback_reuses_heading_no_extra_llm_call(self):
        """无目录的 spine-fallback EPUB：章标题与首个 heading 源文一致时直接复用
        译文，标题 agent 零调用。"""
        with tempfile.TemporaryDirectory() as d:
            epub = os.path.join(d, "sample.epub")
            write_sample_epub(epub)
            cfg = _config(os.path.join(d, "state"))  # write_sample_epub 内容为日文

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(epub)

            m = store.load_manifest()
            self.assertEqual(len(m["chapters"]), 2)
            for c in m["chapters"]:
                self.assertFalse(c.get("toc_entry_id"))
                heading = store.load_chapter(c["index"]).segments[0]
                self.assertEqual(heading.kind, "heading")
                self.assertEqual(c["title_translated"], " ".join(heading.target.split()))
            self.assertEqual(len(_title_calls(client.calls)), 0)

    def test_title_translation_idempotent_on_rerun(self):
        """全部标题已译（含 entry 同步）后重复调用标题翻译：零新增标题
        请求，manifest 内容不变。"""
        with tempfile.TemporaryDirectory() as d:
            epub = os.path.join(d, "nested.epub")
            write_nested_toc_epub(epub)
            cfg = _epub_config(os.path.join(d, "state"))

            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(epub)
            m1 = store.load_manifest()
            self.assertTrue(all(c.get("title_translated") for c in m1["chapters"]))
            self.assertTrue(all(e.get("title_translated") for e in m1["meta"]["toc_entries"]))
            chapters_snapshot = {c["index"]: c["title_translated"] for c in m1["chapters"]}
            entries_snapshot = {
                e["entry_id"]: e["title_translated"] for e in m1["meta"]["toc_entries"]
            }

            glossary = GlossaryStore(store.glossary_path)
            client2 = FakeClient(handler=routing_handler)
            Application(cfg, client=client2).translate_titles(store)
            glossary.close()

            self.assertEqual(len(_title_calls(client2.calls)), 0)
            m2 = store.load_manifest()
            self.assertEqual(
                {c["index"]: c["title_translated"] for c in m2["chapters"]}, chapters_snapshot
            )
            self.assertEqual(
                {e["entry_id"]: e["title_translated"] for e in m2["meta"]["toc_entries"]},
                entries_snapshot,
            )

    def test_dedup_shared_source_title_translated_once(self):
        """两章 + 一个 toc entry 共享同一源标题文本：LLM 只收到一条唯一标题，
        三者最终拿到同一份译文。"""
        with tempfile.TemporaryDirectory() as d:
            state_dir = os.path.join(d, "state")
            cfg = _epub_config(state_dir)
            store = RunStore(os.path.join(state_dir, "book"))

            store.save_chapter(
                Chapter(
                    index=0, title="Same Title", segments=[Segment(index=0, source="Body one.")]
                )
            )
            store.save_chapter(
                Chapter(
                    index=1, title="Same Title", segments=[Segment(index=0, source="Body two.")]
                )
            )
            store.save_state(
                RunState(
                    identity=RunIdentity(
                        source_bytes_sha256="test-hash",
                        run_input_schema_version=1,
                        source_lang="en",
                        target_lang="zh",
                    ),
                    title="Book",
                    fmt="epub",
                    source_path="",
                    source_lang="en",
                    target_lang="zh",
                    meta={
                        "toc_entries": [
                            {"entry_id": "toc.ncx:0", "title": "Same Title", "external": False}
                        ]
                    },
                    chapters=[
                        ChapterIndex(index=0, title="Same Title", href="a.xhtml"),
                        ChapterIndex(index=1, title="Same Title", href="b.xhtml"),
                    ],
                    progress={
                        0: ChapterProgress(status=STATUS_DONE),
                        1: ChapterProgress(status=STATUS_DONE),
                    },
                )
            )
            _stamp_completed_store(store, chapters=2)

            captured = {}

            def handler(messages, agent, operation, json_mode):
                if "标题翻译" in messages[0]["content"]:
                    captured["user"] = messages[-1]["content"]
                return routing_handler(messages, agent, operation, json_mode)

            glossary = GlossaryStore(store.glossary_path)
            Application(cfg, client=FakeClient(handler=handler)).translate_titles(store)
            glossary.close()

            self.assertIn("user", captured)
            # 三处共享同一源标题，只应发一条
            self.assertEqual(len(re.findall(r"^\[(\d+)\]", captured["user"], re.M)), 1)

            m = store.load_manifest()
            translated = {c["title_translated"] for c in m["chapters"]}
            translated.add(m["meta"]["toc_entries"][0]["title_translated"])
            self.assertEqual(len(translated), 1)


class TestRunSteps(unittest.TestCase):
    def test_subset_only_assemble(self):
        """步骤目标：仅回填时不应再产生翻译调用（幂等）。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            orch = Application(cfg, client=FakeClient(handler=routing_handler))
            orch.run(txt)
            # 仅回填，不应再翻译
            client2 = FakeClient(handler=routing_handler)
            res = Application(cfg, client=client2).run_goal_result(txt, assemble_goal())
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
        """审校每块报 index 0 漏译；带审校意见的定向重译调用返回定向重译文。"""

        def handler(messages, agent, operation, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "译文审校" in sys:
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "missing",
                                "detail": "漏了一句",
                                "suggestion": "补上",
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            if "文学翻译" in sys and "审校意见" in user:
                return json.dumps({"translations": [fix_text]}, ensure_ascii=False)
            return routing_handler(messages, agent, operation, json_mode)

        return handler

    def _run(self, d, *, autofix, fix_text=None):
        txt = os.path.join(d, "novel.txt")
        write_sample_txt(txt)
        cfg = _config(os.path.join(d, "state"))
        cfg.pipeline.autofix_severe = autofix
        handler = self._handler(fix_text or self.FIX_TEXT)
        return Application(cfg, client=FakeClient(handler=handler)).run(txt)

    def test_autofix_adopts_retranslation(self):
        """autofix 开：严重项定向重译被采纳 → target 更新、fixed=True。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=True)
            ch = store.load_chapter(0)
            flagged = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("type") == "missing"
            ]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is True for i in flagged))
            self.assertTrue(all(i.get("stage") == "review" for i in flagged))
            self.assertTrue(all("chapter" in i for i in flagged))
            self.assertEqual(ch.text_segments[0].target, self.FIX_TEXT)
            # chapter_reviewed 是在 chapter/stage/fixed 全部补齐且进度落盘后发出的
            # 同步事件，其摘要必须与持久化的最终审校项一致。
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            reviewed = [e for e in events if e["event"] == "chapter_reviewed" and e["chapter"] == 0]
            self.assertTrue(reviewed, "autofix 路径应发 chapter_reviewed 事件")
            persisted_review = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("stage") == "review"
            ]
            self.assertTrue(persisted_review)
            for e in reviewed:
                self.assertEqual(
                    e["issues_sha256"],
                    stable_digest(persisted_review),
                    "同步 autofix 路径的 chapter_reviewed 摘要必须对应当前持久化的审校项",
                )

    def test_autofix_review_digest_matches_normalized_persisted_issues(self):
        """载荷中的 fixed 为整数 0 时，经过 ReviewIssue 模型转换后会归一化为
        False；chapter_reviewed 的 issue_count/issues_sha256 必须基于归一化后的
        持久化审校项计算，而非原始模型载荷（否则无法根据落盘的审校项复现摘要）。"""

        def handler(messages, agent, operation, json_mode):
            if "译文审校" in messages[0]["content"]:
                user = messages[-1]["content"]
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "terminology",  # 非严重项，不触发定向重译
                                "detail": "术语不一致",
                                "suggestion": "改用对照表",
                                "fixed": 0,  # 0 经 ReviewIssue 归一化为 False
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = True  # 同步审校路径
            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)

            persisted_review = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("stage") == "review"
            ]
            self.assertTrue(persisted_review, "审校项应已落盘")
            self.assertIs(persisted_review[0]["fixed"], False, "fixed 应被归一化为布尔 False")
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            reviewed = [e for e in events if e["event"] == "chapter_reviewed" and e["chapter"] == 0]
            self.assertTrue(reviewed, "autofix 路径应发 chapter_reviewed 事件")
            for e in reviewed:
                self.assertEqual(e["issue_count"], len(persisted_review))
                self.assertEqual(
                    e["issues_sha256"],
                    stable_digest(persisted_review),
                    "fixed 归一化后，同步路径摘要必须等于持久化审校项数据的摘要",
                )

    def test_autofix_off_reports_only(self):
        """autofix 关：审校严重项仅上报 fixed=False，审校通道本身不动正文
        （lint 层独立于 autofix_severe 常开，可能另行修正与本用例无关的段落，
        不在此断言范围——只验证 review 通道未触发 autofix_applied）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._run(d, autofix=False)
            flagged = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("type") == "missing"
            ]
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
            flagged = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("type") == "missing"
            ]
            self.assertTrue(flagged)
            self.assertTrue(all(i.get("fixed") is False for i in flagged))
            self.assertNotEqual(ch.text_segments[0].target, "短")
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            rejected = [e for e in events if e["event"] == "autofix_rejected"]
            self.assertTrue(rejected, "过短重译应发 autofix_rejected")
            for e in rejected:
                self.assertNotIn("source", e, "autofix_rejected 不携带 source 明文")
                self.assertNotIn("before", e, "autofix_rejected 不携带 before 明文")
                self.assertNotIn("proposed", e, "autofix_rejected 不携带提案明文")
                self.assertNotIn("issues", e, "autofix_rejected 不携带完整 issue 明文")
                self.assertIn("chapter", e)
                self.assertIn("index", e)
                self.assertIn("reason", e)
                self.assertIn("issues_sha256", e)
                self.assertIn("proposal_sha256", e)

    def test_autofix_no_accept_skips_redundant_chapter_save(self):
        """无采纳 autofix 时 review 阶段不冗余整章写：与审校关闭基线相比，各章
        保存次数完全一致；采纳场景则每章恰好多一次。"""

        def run_once(autofix: bool, fix_text: str) -> dict[int, int]:
            with tempfile.TemporaryDirectory() as d:
                txt = os.path.join(d, "novel.txt")
                write_sample_txt(txt)
                cfg = _config(os.path.join(d, "state"))
                cfg.pipeline.autofix_severe = autofix
                handler = self._handler(fix_text)
                real_save = RunStore.save_chapter
                saves: dict[int, int] = {}

                def counting_save(self, chapter):
                    saves[chapter.index] = saves.get(chapter.index, 0) + 1
                    return real_save(self, chapter)

                with mock.patch.object(RunStore, "save_chapter", counting_save):
                    Application(cfg, client=FakeClient(handler=handler)).run(txt)
                return saves

        baseline = run_once(autofix=False, fix_text="短")
        rejected = run_once(autofix=True, fix_text="短")
        self.assertEqual(rejected, baseline, "无采纳 autofix 时 review 阶段不得整章写")
        accepted = run_once(autofix=True, fix_text=self.FIX_TEXT)
        self.assertEqual(
            {ci: n + 1 for ci, n in rejected.items()},
            {ci: accepted[ci] for ci in rejected},
            "采纳 autofix 时 review 每章恰好多一次整章写",
        )

    def test_review_index_mapping(self):
        """整章多块审校时，块内 index 正确映射回章内段号。"""

        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            if "译文审校" in messages[0]["content"]:
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "missing",
                                "detail": "x",
                                "suggestion": "修正",
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.segment.max_chars_per_batch = 8  # 审校块预算=24 → 每段自成一块
            cfg.pipeline.autofix_severe = False
            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)
            ch = store.load_chapter(0)
            idxs = sorted(
                i["index"]
                for i in store.load_progress(0).review_issue_dicts()
                if i.get("type") == "missing"
            )
            # 每块报 index 0 → 映射后应为各块首段的章内段号（0,1,2,...互不相同）
            self.assertEqual(idxs, list(range(len(ch.text_segments))))

    def test_review_accepts_numeric_string_index(self):
        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            if "译文审校" in messages[0]["content"]:
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": "0",
                                "type": "missing",
                                "detail": "x",
                                "suggestion": "修正",
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)

            issues = store.load_progress(0).review_issue_dicts()
            self.assertTrue(issues)
            self.assertEqual(issues[0]["index"], 0)

    def test_review_rejects_invalid_index_and_keeps_pending(self):
        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            if "译文审校" in messages[0]["content"]:
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": "unknown",
                                "type": "missing",
                                "detail": "x",
                                "suggestion": "修正",
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False
            cfg.pipeline.review_output_retries = 0
            store = Application(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)

            self.assertIn(0, store.review_pending_chapters())
            review_issues = [
                i for i in store.load_progress(0).review_issue_dicts() if i.get("stage") == "review"
            ]
            self.assertEqual(review_issues, [])
            with open(store.event_log_path, encoding="utf-8") as f:
                failed = [
                    json.loads(line)
                    for line in f
                    if line.strip() and json.loads(line).get("event") == "chapter_review_failed"
                ]
            self.assertTrue(failed)
            self.assertEqual(failed[-1]["reason"], "invalid_issue_index")


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
        from trans_novel.pipeline.nodes.common import sample_text

        with tempfile.TemporaryDirectory() as d:
            doc = self._long_doc(d)
            labeled = sample_text(doc)
            for tag in ("【开头样章】", "【中部样章】", "【结尾样章】"):
                self.assertIn(tag, labeled)
            plain = sample_text(doc, labeled=False)
            self.assertNotIn("样章】", plain)
            self.assertIn("章0の段落0です", plain)

    def test_sample_text_short_book_dedup(self):
        """单章书：三个采样点重合，只取一次、不重复。"""
        with tempfile.TemporaryDirectory() as d:
            from trans_novel.ingest.segmenter import load_document
            from trans_novel.pipeline.nodes.common import sample_text

            txt = os.path.join(d, "short.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write("# 唯一章\n\n" + "长段落。" + "あ" * 300)
            doc = load_document(txt, "ja", "zh")
            sample = sample_text(doc)
            self.assertEqual(sample.count("【开头样章】"), 1)
            self.assertNotIn("【中部样章】", sample)
            self.assertNotIn("【结尾样章】", sample)

    def test_style_brief_new_fields(self):
        """style_brief 渲染新风格维度；旧 analysis（缺新字段）不报错不输出。"""
        from trans_novel.agents.analyzer import Analyzer
        from trans_novel.llm import FakeClient as FC

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

        orch = Application(cfg, client=FakeClient(handler=routing_handler))
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
        Application(cfg, client=client).run(txt)
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

        def handler(messages, agent, operation, json_mode):
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
            return routing_handler(messages, agent, operation, json_mode)

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
            Application(cfg, client=client).run(txt)

            translate_prompts = [
                "\n".join(m["content"] for m in c["messages"])
                for c in client.calls
                if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertGreaterEqual(len(translate_prompts), 3)
            self.assertIn("夏帆ちゃん → 小夏帆", translate_prompts[-1])

    def test_chapter_glossary_refreshes_review_prompt(self):
        """全章兜底术语抽取在 review 前执行，章末审校能看到新称谓。"""

        def handler(messages, agent, operation, json_mode):
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
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {"issues": [], "reviewed_segments": n, "complete": True},
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

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

            Application(cfg, client=FakeClient(handler=handler)).run(txt)


class TestInflightGlossary(unittest.TestCase):
    """inflight_glossary=True：旧版"译后逐批+章末抽取"路径原样保留（日文轻小说场景）。"""

    def test_legacy_extraction_path_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.inflight_glossary = True

            client = FakeClient(handler=routing_handler)
            store = Application(cfg, client=client).run(txt)

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
    def _naturalize_handler(messages, agent, operation, json_mode):
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
        return routing_handler(messages, agent, operation, json_mode)

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
            store = Application(cfg, client=client).run(txt)

            m = store.load_manifest()
            for ci in range(len(m["chapters"])):
                self.assertTrue(
                    store.load_progress(ci).naturalized,
                    f"第 {ci} 章 naturalize 后应标记进度 naturalized",
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
            store = Application(cfg, client=client).run(txt)

            self.assertEqual(
                self._naturalize_calls(client.calls),
                [],
                "naturalize=False 时不应发生任何 naturalize 相关调用",
            )
            m = store.load_manifest()
            for ci in range(len(m["chapters"])):
                self.assertFalse(store.load_progress(ci).naturalized)

    def test_naturalize_idempotent_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            store = Application(cfg, client=FakeClient(handler=self._naturalize_handler)).run(txt)
            self.assertTrue(store.load_progress(0).naturalized)

            # 模拟"naturalize 已完成、进度已落盘，但章末 DONE 标记前中断"续跑：
            # 章状态改回 pending，进度 naturalized 保持 True（幂等标记未丢）。
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=self._naturalize_handler)
            Application(cfg, client=client2).run(txt, only_chapter=0)

            self.assertEqual(
                self._naturalize_calls(client2.calls),
                [],
                "进度标记已置位，续跑不应重复 naturalize",
            )


class TestOperationRouting(unittest.TestCase):
    def test_task_operations(self):
        """每个生产 LLM 调用都带稳定 operation（内部业务标签）与 agent（功能路由键）；
        梗概带 max_tokens 上限。

        翻译类三个 operation（translate.batch / translate.lint_fix / translate.review_fix）
        共用同一系统前缀「文学翻译」，不能依据共享 marker 做一一映射；这里根据
        用户 prompt 模板的结构逐一精确判定，并分别覆盖三条路由。
        """
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.backtranslate_sample = 1.0  # 强制触发回译

            def handler(messages, agent, operation, json_mode):
                # 审校报一个严重项（missing，默认 autofix_severe=True），
                # 驱动章末 _autofix_severe → translate.review_fix 路由。
                if "译文审校" in messages[0]["content"]:
                    n = len(re.findall(r"^\[(\d+)\]", messages[-1]["content"], re.M))
                    return json.dumps(
                        {
                            "issues": [
                                {
                                    "index": 0,
                                    "type": "missing",
                                    "detail": "漏译",
                                    "suggestion": "补充全文",
                                }
                            ],
                            "reviewed_segments": n,
                            "complete": True,
                        },
                        ensure_ascii=False,
                    )
                return routing_handler(messages, agent, operation, json_mode)

            client = FakeClient(handler=handler)
            Application(cfg, client=client).run(txt)

            expect = {
                "章节梗概员": "prescan.digest",
                "全书概览员": "prescan.book_synopsis",
                "术语候选挖掘": "prescan.term_mine",
                "全书定名": "prescan.name_terms",
                "回译译者": "backtranslate.translate",
                "译文审校": "review.chapter",
                "保真度": "backtranslate.check",
                "中文润色编辑": "polish.batch",
                "中文书稿的母语审读编辑": "naturalize.screen",
            }
            expect_agent = {
                "章节梗概员": "preparer",
                "全书概览员": "preparer",
                "术语候选挖掘": "preparer",
                "全书定名": "analyst",
                "回译译者": "light-translator",
                "译文审校": "reviewer",
                "保真度": "reviewer",
                "中文润色编辑": "editor",
                "中文书稿的母语审读编辑": "reviewer",
            }
            seen = set()
            translator_calls = []
            for c in client.calls:
                system = c["messages"][0]["content"]
                if "文学翻译" in system:
                    self.assertEqual(c["agent"], "translator", "翻译调用固定走 translator Agent")
                    translator_calls.append(c)
                    continue
                for marker, operation in expect.items():
                    if marker in system:
                        self.assertEqual(c["operation"], operation, f"{marker} 应走 {operation}")
                        self.assertEqual(
                            c["agent"],
                            expect_agent[marker],
                            f"{marker} 应走 {expect_agent[marker]}",
                        )
                        seen.add(marker)
                        if marker == "章节梗概员":
                            self.assertEqual(c["max_tokens"], 600)
                        if marker == "全书概览员":
                            self.assertEqual(c["max_tokens"], 1200)
            self.assertEqual(seen, set(expect), "各类调用都应出现")

            # 翻译类：按可观察的用户 prompt 模板区分三条路由（系统前缀共享，不可作键）。
            # 批译模板含「请翻译以上每一段」；定向重译模板必有「重译」；其中仅章末严重项
            # 自动重译（_autofix_severe）把反馈拼成「…（建议：…）」，据此区分 lint/review 修复。
            routes = {"translate.batch": 0, "translate.lint_fix": 0, "translate.review_fix": 0}
            for c in translator_calls:
                user = c["messages"][-1]["content"]
                if "请翻译以上每一段" in user:
                    expected = "translate.batch"
                else:
                    self.assertIn("重译", user, "翻译调用须命中批译或定向重译模板之一")
                    expected = (
                        "translate.review_fix" if "（建议：" in user else "translate.lint_fix"
                    )
                self.assertEqual(c["operation"], expected, f"{expected} 被错误标注")
                routes[expected] += 1
            self.assertGreater(routes["translate.batch"], 0, "普通批翻译必须走 translate.batch")
            self.assertGreater(
                routes["translate.lint_fix"], 0, "lint 修复必须走 translate.lint_fix"
            )
            self.assertGreater(
                routes["translate.review_fix"], 0, "review 修复必须走 translate.review_fix"
            )


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
        """polish 开启：batch_translated 先发（polished=False，V2 无明文、只带 raw 译文摘要），
        章末排干润色后再发 batch_polished（仅记录实际改动 + 最终 target 摘要），
        pending_polish 清空。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(
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
            # batch_translated 触发时尚未润色：polished=False，只带摘要，无任何明文载荷
            for e in translated:
                self.assertFalse(e["polished"])
                self.assertEqual(e.get("event_schema"), 2)
                self.assertNotIn("segments", e)
                self.assertNotIn("source", e)
                self.assertNotIn("target", e)
                self.assertIn("target_sha256", e)
            # 章末排干后 batch_polished 仅记录实际改动（稳定段号），并带最终 target 摘要
            ch = store.load_chapter(0)
            segs = ch.text_segments
            for e in polished:
                self.assertEqual(e.get("event_schema"), 2)
                self.assertNotIn("segments", e)
                self.assertEqual(e["changed_count"], len(e["changes"]))
                batch_segs = segs[e["start_index"] : e["start_index"] + e["count"]]
                self.assertEqual(
                    e["target_sha256"],
                    stable_digest([{"index": s.index, "target": s.target} for s in batch_segs]),
                    "batch_polished 摘要必须对应当前落盘的 target",
                )
                for c in e["changes"]:
                    self.assertIn("index", c)
                    self.assertNotEqual(c["before"], c["after"])
            # 每段都被润色改写（译→润），逐段出现在 changes 里，无遗漏
            all_changed = {c["index"] for e in polished for c in e["changes"]}
            self.assertEqual(all_changed, {s.index for s in segs})
            # run() 返回时排干已完成：正文与 meta 均为最终态，无残留 pending 标记
            self.assertFalse(store.load_progress(0).pending_polish)
            self.assertTrue(all(s.target.startswith("润") for s in segs))

    def test_noop_polish_batch_still_emits_compact_event(self):
        """润色不改动任何文本（no-op 批次）：仍发一条 batch_polished，changed_count=0、
        changes=[]，并带最终 target 摘要。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            def handler(messages, agent, operation, json_mode):
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if "中文润色编辑" in system:
                    # 原样返回待润色译文：润色不产生任何改动
                    target_block = user.split("【待润色中文译文】", 1)[-1]
                    polished = re.findall(r"^\[\d+\] (.*)$", target_block, re.M)
                    return json.dumps({"polished": polished}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            polished = [e for e in events if e["event"] == "batch_polished" and e["chapter"] == 0]
            self.assertTrue(polished, "no-op 润色批次仍应发 batch_polished")
            ch = store.load_chapter(0)
            segs = ch.text_segments
            for e in polished:
                self.assertEqual(e["changed_count"], 0)
                self.assertEqual(e["changes"], [])
                batch_segs = segs[e["start_index"] : e["start_index"] + e["count"]]
                self.assertEqual(
                    e["target_sha256"],
                    stable_digest([{"index": s.index, "target": s.target} for s in batch_segs]),
                )

    def test_punctuation_only_change_is_audited(self):
        """润色器输出与 raw 完全一致、仅标点规范化造成落盘差异：审计仅记录实际
        改动，并以持久化前的 raw 作为 before 基线，规范化差异也要如实上报。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            def handler(messages, agent, operation, json_mode):
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if "文学翻译" in system:
                    n = len(re.findall(r"^\[\d+\]", user, re.M))
                    return json.dumps({"translations": ["你好,世界。"] * n}, ensure_ascii=False)
                if "中文润色编辑" in system:
                    # 原样返回待润色译文：润色器本身不改任何字
                    target_block = user.split("【待润色中文译文】", 1)[-1]
                    polished = re.findall(r"^\[\d+\] (.*)$", target_block, re.M)
                    return json.dumps({"polished": polished}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target == "你好，世界。" for s in ch.text_segments))
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            polished = [e for e in events if e["event"] == "batch_polished" and e["chapter"] == 0]
            self.assertTrue(polished, "应发 batch_polished 事件")
            for e in polished:
                self.assertGreater(e["changed_count"], 0, "标点规范化差异应计入 changed_count")
                for c in e["changes"]:
                    self.assertEqual(c["before"], "你好,世界。", "before 必须取持久化前的 raw")
                    self.assertEqual(c["after"], "你好，世界。")


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

            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(
                txt, only_chapter=0
            )
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target and s.target.startswith("润") for s in ch.text_segments))
            self.assertFalse(store.load_progress(0).pending_polish)

            # 模拟"批已落盘但章末排干润色前中断"：把最后一段的译文改回未润色的 raw
            # （"译{i}"），补回 pending_polish 标记，章状态改回 pending。
            last_idx = len(ch.text_segments) - 1
            ch.segments[last_idx].target = f"译{last_idx}"
            pg = store.load_progress(0)
            pg.pending_polish = [PolishBatch(start=last_idx, count=1)]
            store.save_progress(0, pg)
            store.save_chapter(ch)
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=routing_handler)
            Application(cfg, client=client2).run(txt, only_chapter=0)
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0)  # 已有译文，批跳过，未重翻

            ch2 = store.load_chapter(0)
            # routing_handler 的润色输出按"本次调用内"的局部下标编号：该批只含 1 段
            # （原始的第 last_idx 段），单独重新提交润色后局部下标为 0 → "润0"。
            self.assertEqual(ch2.text_segments[last_idx].target, "润0")
            self.assertFalse(store.load_progress(0).pending_polish)

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


class TestAcceptedEventCommitOrdering(unittest.TestCase):
    """采纳改写事件必须在正文成功持久化后发出；正文保存失败时不得记录事件。"""

    def test_batch_translated_absent_when_chapter_save_raises(self):
        """译文保存抛出 OSError 时，节点应失败，且不得出现 batch_translated 或 lint_refixed 事件。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            cfg.pipeline.polish = False

            real_save = RunStore.save_chapter

            def flaky_save(self, chapter):
                if any((s.target or "").strip() for s in chapter.segments):
                    raise OSError("disk full")
                return real_save(self, chapter)

            with (
                mock.patch.object(RunStore, "save_chapter", flaky_save),
                self.assertRaises(RequiredNodeFailed),
            ):
                Application(cfg, client=FakeClient(handler=routing_handler)).run(
                    txt, only_chapter=0
                )
            store = RunStore(cfg.state_dir)
            # 保存失败时不记录任何事件，因此可能不会创建 events.jsonl
            events = []
            if os.path.exists(store.event_log_path):
                with open(store.event_log_path, encoding="utf-8") as f:
                    events = [json.loads(line) for line in f if line.strip()]
            self.assertFalse(
                any(e["event"] == "batch_translated" for e in events),
                "译文保存失败时不得发出 batch_translated 采纳事件",
            )
            self.assertFalse(
                any(e["event"] == "lint_refixed" for e in events),
                "译文保存失败时不得发出 lint_refixed 采纳事件",
            )


class TestEventLogBestEffort(unittest.TestCase):
    """事件日志追加失败（仅 OSError）→ RuntimeWarning，流程不失败、不触发重跑。"""

    def test_oserror_warns_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state"))
            store.ensure_dirs()
            # 用目录占位事件文件：open(..., "a") 抛 IsADirectoryError（OSError 子类）
            os.makedirs(store.event_log_path)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.log_event("probe", payload="x")  # 不得抛异常
            self.assertTrue(
                any(issubclass(w.category, RuntimeWarning) for w in caught),
                "事件追加失败应发 RuntimeWarning",
            )
            # 恢复为文件后正常追加 V2 行
            os.rmdir(store.event_log_path)
            store.log_event("probe", payload="x")
            with open(store.event_log_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(rows[-1]["event"], "probe")
            self.assertEqual(rows[-1]["event_schema"], 2)

    def test_oserror_from_dir_setup_warns_and_does_not_raise(self):
        """目录创建抛出 OSError 时也按尽力写入策略处理：发出 RuntimeWarning，
        不向外抛出异常。

        ensure_dirs() 和文件追加必须处于同一 OSError 捕获范围内——用普通文件
        占位 chapters_v2 目录，令 makedirs(exist_ok=True) 抛 FileExistsError。
        """
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state"))
            os.rmdir(store.chapters_v2_dir)
            with open(store.chapters_v2_dir, "w", encoding="utf-8") as f:
                f.write("blocker")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.log_event("probe", payload="x")  # 目录创建失败也不得抛异常
            self.assertTrue(
                any(issubclass(w.category, RuntimeWarning) for w in caught),
                "目录创建阶段的 OSError 也应发 RuntimeWarning",
            )

    def test_non_oserror_propagates(self):
        """序列化错误和编程错误不属于 OSError，仍应向外抛出，不能被尽力写入逻辑忽略。"""
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state"))
            store.ensure_dirs()
            with self.assertRaises(TypeError):
                store.log_event("probe", payload=object())

    def test_event_failure_does_not_rerun_committed_translation(self):
        """事件写失败只告警：续跑会通过跳过分支复用已提交的译文，绝不重新翻译。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            cfg.pipeline.polish = False

            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            # 把事件文件替换成目录：此后任何事件追加都抛 OSError
            os.remove(store.event_log_path)
            os.makedirs(store.event_log_path)
            # 模拟崩溃窗口：章回 pending、译文保留 → 续跑走批跳过分支
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=routing_handler)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                Application(cfg, client=client2).run(txt, only_chapter=0)
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(translate_calls, [], "事件写失败不得触发已提交译文重翻")
            self.assertTrue(
                any(issubclass(w.category, RuntimeWarning) for w in caught),
                "续跑的事件追加失败应发 RuntimeWarning",
            )


class TestReviewAsync(unittest.TestCase):
    """review=true 且 autofix_severe=false：章末审校提交共享线程池异步跑，
    run() 返回前必须排干——issues 合并写入 ChapterProgress.review_issues
    并发 chapter_reviewed 事件；review worker 出错不得中断 run。"""

    @staticmethod
    def _issue_handler(messages, agent, operation, json_mode):
        # 无共享可变状态：每次调用构造新 dict，可被线程池并发调用
        if "译文审校" in messages[0]["content"]:
            user = messages[-1]["content"]
            n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
            return json.dumps(
                {
                    "issues": [
                        {
                            "index": 0,
                            "type": "terminology",
                            "detail": "术语不一致",
                            "suggestion": "改用对照表",
                        }
                    ],
                    "reviewed_segments": n,
                    "complete": True,
                },
                ensure_ascii=False,
            )
        return routing_handler(messages, agent, operation, json_mode)

    def test_async_review_issues_persisted_before_run_returns(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Application(cfg, client=FakeClient(handler=self._issue_handler)).run(txt)

            m = store.load_manifest()
            self.assertTrue(
                all(store.chapter_status(c["index"]) == STATUS_DONE for c in m["chapters"])
            )
            for ci in range(len(m["chapters"])):
                found = [
                    i
                    for i in store.load_progress(ci).review_issue_dicts()
                    if i.get("type") == "terminology"
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
            for e in events:
                if e["event"] != "chapter_reviewed":
                    continue
                self.assertEqual(e.get("event_schema"), 2)
                self.assertNotIn("issues", e, "chapter_reviewed 不再携带完整 issue 明文")
                self.assertIn("issue_count", e)
                self.assertIn("issues_sha256", e)

    def test_async_review_digest_matches_normalized_persisted_issues(self):
        """异步 finish 路径与同步路径语义一致：若载荷中的 fixed 为整数 0，
        ReviewIssue 模型会将其归一化为 False；事件摘要必须与持久化审校项数据的
        摘要一致。"""

        def handler(messages, agent, operation, json_mode):
            if "译文审校" in messages[0]["content"]:
                user = messages[-1]["content"]
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "terminology",
                                "detail": "术语不一致",
                                "suggestion": "改用对照表",
                                "fixed": 0,  # 0 经 ReviewIssue 归一化为 False
                            }
                        ],
                        "reviewed_segments": n,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False  # 异步审校路径

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)

            m = store.load_manifest()
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            for ci in range(len(m["chapters"])):
                persisted_review = [
                    i
                    for i in store.load_progress(ci).review_issue_dicts()
                    if i.get("stage") == "review"
                ]
                self.assertTrue(persisted_review, f"第 {ci} 章审校项应已落盘")
                self.assertIs(persisted_review[0]["fixed"], False, "fixed 应被归一化为布尔 False")
                reviewed = [
                    e for e in events if e["event"] == "chapter_reviewed" and e["chapter"] == ci
                ]
                self.assertTrue(reviewed, f"第 {ci} 章应发 chapter_reviewed 事件")
                for e in reviewed:
                    self.assertEqual(e["issue_count"], len(persisted_review))
                    self.assertEqual(
                        e["issues_sha256"],
                        stable_digest(persisted_review),
                        "fixed 归一化后，异步路径摘要必须等于持久化审校项数据的摘要",
                    )

    def test_review_worker_failure_does_not_break_run(self):
        # review 未来（review_chapter）本身抛异常 → 触发 runner 异步排干的
        # except 分支（记 chapter_review_failed 后 continue，不中断 run）。
        # 注意：不能靠 handler 对 '译文审校' 抛异常来验证——Reviewer.review 内部
        # _ask_json(..., default=[]) 会吞掉 LLM 异常返回 []，future 正常完成、照常
        # 发 chapter_reviewed，except 分支永不执行（旧版删掉错误处理测试仍会通过）。
        # 故直接遮蔽 ReviewNode.review_chapter 类方法，让提交到线程池的 future 真抛。
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False  # 异步审校路径

            from unittest.mock import patch

            from trans_novel.pipeline.nodes.quality import ReviewNode

            app = Application(cfg, client=FakeClient(handler=routing_handler))

            def _boom(*_a, **_k):
                raise RuntimeError("审校崩")

            with patch.object(ReviewNode, "review_chapter", _boom):
                store = app.run(txt)

            m = store.load_manifest()
            chapters = set(range(len(m["chapters"])))

            # (a) 审校 future 全崩，run 仍走完，每章保持 DONE（未被异常中断）
            self.assertTrue(
                all(store.chapter_status(c["index"]) == STATUS_DONE for c in m["chapters"]),
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
                    for i in store.load_progress(ci).review_issue_dicts()
                    if i.get("stage") == "review"
                ]
                self.assertEqual(
                    review_stage_issues,
                    [],
                    f"第 {ci} 章审校失败，review 阶段的 review_issues 不得被写入",
                )

    def test_crash_resume_reruns_pending_review(self):
        # review 断点续跑不变量（异步审校版）：章已标 DONE 但异步审校结果还没写回
        # 就宕机时，靠进度里的 review_pending 持久标记 + run() 开头的
        # _resume_pending_reviews 补跑，审校结果不静默丢失。没有标记或补跑逻辑，
        # 崩溃后该章审校结果永久缺失，本测试必失败。
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            # (1) 正常跑一遍：审校结果写回、标记清空（前置条件）
            store = Application(cfg, client=FakeClient(handler=self._issue_handler)).run(txt)
            self.assertEqual(
                store.review_pending_chapters(), [], "正常收尾后不应残留任何 review_pending 标记"
            )

            # (2) 模拟崩溃窗口：章 0 已 DONE，但标记残留且审校结果被抹掉
            store.set_review_pending(0, True)
            pg = store.load_progress(0)
            pg.set_review_issue_dicts([])
            store.save_progress(0, pg)
            self.assertIn(
                0, store.review_pending_chapters(), "崩溃模拟：章 0 应带 review_pending 标记"
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events_before = sum(1 for line in f if line.strip())

            # (3) 续跑：所有章已 DONE → targets 为空，补跑只能来自 _resume_pending_reviews
            client2 = FakeClient(handler=self._issue_handler)
            Application(cfg, client=client2).run(txt)

            # (4a) 载荷断言：章 0 审校结果被重新写回（术语项，字段完整）
            issues = store.load_progress(0).review_issue_dicts()
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


class TestReviewRecovery(unittest.TestCase):
    def test_malformed_multi_segment_chunk_splits_and_maps_indexes(self):
        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
            if n > 1:
                return '{"issues":['
            match = re.search(r"^\[0\] 原文：SRC(\d+)$", user, re.M)
            assert match
            i = int(match.group(1))
            return json.dumps(
                {
                    "issues": [
                        {
                            "index": 0,
                            "type": "missing",
                            "detail": f"segment{i}",
                            "suggestion": "修正",
                        }
                    ],
                    "reviewed_segments": 1,
                    "complete": True,
                },
                ensure_ascii=False,
            )

        cfg = _config("state")
        cfg.segment.max_chars_per_batch = 100
        cfg.pipeline.review_output_retries = 0
        node = _review_node(cfg, FakeClient(handler=handler))
        pairs = [(f"SRC{i}", f"TGT{i}") for i in range(4)]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as review_executor:
            issues = node.review_chapter(pairs, [], review_executor)
        self.assertEqual([it["index"] for it in issues], list(range(4)))
        self.assertEqual([it["detail"] for it in issues], [f"segment{i}" for i in range(4)])

    def test_singleton_retries_after_protocol_error(self):
        calls = 0

        def handler(messages, agent, operation, json_mode):
            nonlocal calls
            calls += 1
            if calls == 1:
                return '{"issues":['
            return json.dumps(
                {"issues": [], "reviewed_segments": 1, "complete": True},
                ensure_ascii=False,
            )

        cfg = _config("state")
        cfg.pipeline.review_output_retries = 2
        node = _review_node(cfg, FakeClient(handler=handler))
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as review_executor:
            self.assertEqual(
                node.review_chapter([("SRC", "TGT")], [], review_executor),
                [],
            )
        self.assertEqual(calls, 2)

    def test_singleton_exhaustion_raises_after_bounded_attempts(self):
        calls = 0

        def handler(messages, agent, operation, json_mode):
            nonlocal calls
            calls += 1
            return '{"issues":['

        cfg = _config("state")
        cfg.pipeline.review_output_retries = 2
        node = _review_node(cfg, FakeClient(handler=handler))
        from concurrent.futures import ThreadPoolExecutor

        with (
            ThreadPoolExecutor(max_workers=2) as review_executor,
            self.assertRaises(ReviewOutputError),
        ):
            node.review_chapter([("SRC", "TGT")], [], review_executor)
        self.assertEqual(calls, 3)

    def test_provider_exception_propagates_without_retry(self):
        calls = 0

        def handler(messages, agent, operation, json_mode):
            nonlocal calls
            calls += 1
            raise RuntimeError("provider down")

        cfg = _config("state")
        cfg.pipeline.review_output_retries = 2
        node = _review_node(cfg, FakeClient(handler=handler))
        from concurrent.futures import ThreadPoolExecutor

        with (
            ThreadPoolExecutor(max_workers=2) as review_executor,
            self.assertRaisesRegex(RuntimeError, "provider down"),
        ):
            node.review_chapter([("SRC", "TGT")], [], review_executor)
        self.assertEqual(calls, 1)


class TestReviewChunkConcurrency(unittest.TestCase):
    """Reviewer chunk 预热 + 并发：chunk 0 先单独完成（预热 provider 前缀缓存），
    其余 chunk 随后真正并发；合并结果严格保持原 chunk 顺序。"""

    def test_chunk0_warms_up_then_rest_run_concurrently_and_merge_in_order(self):
        from concurrent.futures import ThreadPoolExecutor

        n_chunks = 3
        rest_barrier = threading.Barrier(n_chunks - 1, timeout=5)
        order_lock = threading.Lock()
        completion_order: list[int] = []
        chunk0_done = threading.Event()

        def handler(messages, agent, operation, json_mode):
            user = messages[-1]["content"]
            m = re.search(r"MARK(\d+)", user)
            assert m, "chunk marker missing from review prompt"
            c = int(m.group(1))
            if c == 0:
                # chunk 0 必须独自先完成（预热前缀缓存），此时其余 chunk 尚未提交。
                with order_lock:
                    completion_order.append(c)
                chunk0_done.set()
                return json.dumps(
                    {
                        "issues": [
                            {
                                "index": 0,
                                "type": "missing",
                                "detail": "chunk0",
                                "suggestion": "修正",
                            }
                        ],
                        "reviewed_segments": 1,
                        "complete": True,
                    },
                    ensure_ascii=False,
                )
            assert chunk0_done.is_set(), "chunk 0 应先完成预热，其余 chunk 才可提交"
            # 其余 chunk 必须同时在飞才能通过 barrier——若退化为串行，该 wait 会
            # 一直阻塞直到 barrier 超时并抛出 BrokenBarrierError，测试失败。
            rest_barrier.wait()
            # 提交顺序 c=1,2；故意让完成顺序反转（c 越大睡得越少），验证结果
            # 合并顺序仍按 chunk 原始顺序而非完成顺序。
            time.sleep((n_chunks - c) * 0.03)
            with order_lock:
                completion_order.append(c)
            return json.dumps(
                {
                    "issues": [
                        {
                            "index": 0,
                            "type": "missing",
                            "detail": f"chunk{c}",
                            "suggestion": "修正",
                        }
                    ],
                    "reviewed_segments": 1,
                    "complete": True,
                },
                ensure_ascii=False,
            )

        cfg = _config("state")
        cfg.segment.max_chars_per_batch = 1  # budget=3，强制每对独立成块
        client = FakeClient(handler=handler)
        node = _review_node(cfg, client)
        pairs = [(f"MARK{c} " + "源文" * 10, f"译文{c}") for c in range(n_chunks)]

        with ThreadPoolExecutor(max_workers=4) as review_executor:
            issues = node.review_chapter(pairs, [], review_executor)

        self.assertEqual([it["detail"] for it in issues], [f"chunk{c}" for c in range(n_chunks)])
        # chunk 0 确实先于其余 chunk 完成；其余 chunk 之间完成顺序被反转（证明真并发）。
        self.assertEqual(completion_order[0], 0)
        self.assertEqual(completion_order[1:], [2, 1])

    def test_book_wide_pool_bounds_concurrency_across_chapters_without_deadlock(self):
        """异步路径（review=true, autofix_severe=false）多章并发提交审校：
        book-wide review_executor 上限生效、且不因"任务等自己"而死锁。"""
        max_concurrent = 0
        current = 0
        lock = threading.Lock()

        def handler(messages, agent, operation, json_mode):
            nonlocal max_concurrent, current
            if "译文审校" in messages[0]["content"]:
                with lock:
                    current += 1
                    max_concurrent = max(max_concurrent, current)
                time.sleep(0.01)
                with lock:
                    current -= 1
                user = messages[-1]["content"]
                n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                return json.dumps(
                    {"issues": [], "reviewed_segments": n, "complete": True},
                    ensure_ascii=False,
                )
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.autofix_severe = False

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)
            # 未死锁即已经证明池分离设计成立；再断言并发确实发生过且没有超过硬上限 4。
            self.assertGreaterEqual(max_concurrent, 1)
            self.assertLessEqual(max_concurrent, 4)
            m = store.load_manifest()
            self.assertTrue(
                all(store.chapter_status(c["index"]) == STATUS_DONE for c in m["chapters"])
            )


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

            def handler(messages, agent, operation, json_mode):
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
                return routing_handler(messages, agent, operation, json_mode)

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)

            self.assertTrue(
                overlapped.is_set(),
                "digest 与 term mining 必须真正同时在跑，而非先后串行执行",
            )
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)

    def test_naming_waits_for_slower_mining_branch(self):
        """mining 分支人为拖慢：naming（全书定名）调用必须等它彻底收尾才发生。"""
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))

            lock = threading.Lock()
            mining_finished_at: list[float] = []
            naming_started_at: list[float] = []

            def handler(messages, agent, operation, json_mode):
                system = messages[0]["content"]
                if "术语候选挖掘" in system:
                    time.sleep(0.1)  # 故意拖慢挖掘分支
                    with lock:
                        mining_finished_at.append(time.monotonic())
                    return json.dumps({"candidates": ["堀北"]}, ensure_ascii=False)
                if "全书定名" in system:
                    with lock:
                        naming_started_at.append(time.monotonic())
                return routing_handler(messages, agent, operation, json_mode)

            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)

            self.assertTrue(mining_finished_at and naming_started_at)
            self.assertGreaterEqual(
                min(naming_started_at),
                max(mining_finished_at),
                "naming 必须等挖掘分支（含人为拖慢的每一章）全部收尾才能开始",
            )
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)
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
            from unittest.mock import patch

            from trans_novel.agents.synopsis import Synopsizer

            client = FakeClient(handler=routing_handler)
            app = Application(cfg, client=client)
            store = app.prepare(txt)

            def _boom(source_text):
                raise RuntimeError("digest 崩")

            # 遮蔽 Synopsizer.digest_chapter_strict（节点实际使用的 strict 调用），
            # 绕过 _ask_text 的吞异常；并行层 join 后排干挖掘分支再整体冒泡。
            with (
                patch.object(Synopsizer, "digest_chapter_strict", _boom),
                self.assertRaises(RuntimeError),
            ):
                app.run(txt)

            self.assertFalse(
                store.load_state().analysis_flags.term_mining_done,
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
            Application(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["polish.batch"]
            self.assertGreater(op["accepted"], 0)
            self.assertEqual(op["rejected"], 0)

    def test_polish_batch_rejected_when_introduces_new_lint_issue(self):
        src = "「おはようございます」と彼は静かな声で言った。"

        def handler(messages, agent, operation, json_mode):
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
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            with open(txt, "w", encoding="utf-8") as f:
                f.write(f"# 第一章\n\n{src}\n")
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False
            client = FakeClient(handler=handler)
            Application(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["polish.batch"]
            self.assertGreaterEqual(op["rejected"], 1)

    def test_lint_fix_accepted_when_retranslation_reduces_issues(self):
        src = "「おはようございます」と彼は静かな声で言った。窓の外には青い空が広がっていた。"

        def handler(messages, agent, operation, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in sys:
                n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
                if "审校意见" in user:
                    out = ["“早上好”他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
                else:
                    out = ["早上好他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
                return json.dumps({"translations": out}, ensure_ascii=False)
            return routing_handler(messages, agent, operation, json_mode)

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
            Application(cfg, client=client).run(txt)

            op = client.usage_summary()["by_operation"]["translate.lint_fix"]
            self.assertEqual(op["accepted"], 1)
            self.assertEqual(op["rejected"], 0)

    def test_review_fix_accepted_and_rejected_recorded(self):
        FIX_TEXT = "第一章 邂逅"  # 7 字，比值 1.0，通过长度校验 → accepted

        def handler(fix_text):
            def h(messages, agent, operation, json_mode):
                sys = messages[0]["content"]
                user = messages[-1]["content"]
                if "译文审校" in sys:
                    n = len(re.findall(r"^\[\d+\] 原文：", user, re.M))
                    return json.dumps(
                        {
                            "issues": [
                                {
                                    "index": 0,
                                    "type": "missing",
                                    "detail": "漏了一句",
                                    "suggestion": "补上",
                                }
                            ],
                            "reviewed_segments": n,
                            "complete": True,
                        },
                        ensure_ascii=False,
                    )
                if "文学翻译" in sys and "审校意见" in user:
                    return json.dumps({"translations": [fix_text]}, ensure_ascii=False)
                return routing_handler(messages, agent, operation, json_mode)

            return h

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)

            cfg_accept = _config(os.path.join(d, "state_accept"))
            cfg_accept.pipeline.autofix_severe = True
            accepted_client = FakeClient(handler=handler(FIX_TEXT))
            Application(cfg_accept, client=accepted_client).run(txt)
            op = accepted_client.usage_summary()["by_operation"]["translate.review_fix"]
            self.assertGreaterEqual(op["accepted"], 1)
            self.assertEqual(op["rejected"], 0)

            cfg_reject = _config(os.path.join(d, "state_reject"))
            cfg_reject.pipeline.autofix_severe = True
            rejected_client = FakeClient(handler=handler("短"))  # 过短，长度校验不通过
            Application(cfg_reject, client=rejected_client).run(txt)
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
            orch = Application(cfg, client=client)
            orch.run_goal_result(
                txt,
                ExecutionGoal(
                    name="translate-qa-report",
                    phases=("prepare", "prescan", "translate", "titles", "qa", "report"),
                    do_qa=True,
                ),
            )

            blank = [c for c in client.calls if not c.get("operation")]
            self.assertEqual(
                blank,
                [],
                f"生产调用不得省略 operation 标签，缺失的 system 前缀："
                f"{[c['messages'][0]['content'][:30] for c in blank]}",
            )


class TestPolishFailureFallback(unittest.TestCase):
    def test_polish_failure_falls_back_to_raw_translation(self):
        """workflow 必需润色路径：provider 失败必须传播（runner 落失败态并重试），
        不得伪装成“润色成功”；pending_polish 保留供续跑重试。"""
        from trans_novel.ingest.segmenter import load_document
        from trans_novel.pipeline.runner import RequiredNodeFailed
        from trans_novel.pipeline.state import NODE_POLISH

        def handler(messages, agent, operation, json_mode):
            if "中文润色编辑" in messages[0]["content"]:
                raise RuntimeError("润色模型宕机")
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            doc = load_document(txt, "ja", "zh")
            store = RunStore(os.path.join(cfg.state_dir, slugify(doc.title)))

            with self.assertRaises(RequiredNodeFailed):
                Application(cfg, client=FakeClient(handler=handler)).run(txt, only_chapter=0)
            node = store.load_state().nodes[chapter_node_key(NODE_POLISH, 0)]
            self.assertEqual(node.status, "failed_permanent")
            self.assertEqual(node.failure.kind, "provider_permanent")
            self.assertTrue(
                store.load_progress(0).pending_polish,
                "provider 失败必须保留 pending_polish 供续跑重试（不得清空伪装成功）",
            )
            ch = store.load_chapter(0)
            self.assertTrue(
                all(s.target and s.target.startswith("译") for s in ch.text_segments),
                "失败批次译文保持未润色原文",
            )


class TestLintQuoteRefix(unittest.TestCase):
    """批循环内 lint：首译丢引号 → 定向重译修复（事件 batch_linted + lint_refixed）。"""

    SRC = "「おはようございます」と彼は静かな声で言った。窓の外には青い空が広がっていた。"

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}\n")

    @staticmethod
    def _handler(messages, agent, operation, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            if "审校意见" in user:
                # 定向重译：修复丢引号问题，补回成对引号
                out = ["“早上好”他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
            else:
                # 首译：丢引号（触发 quote_loss）
                out = ["早上好他轻声说道窗外是一片蔚蓝的天空" for _ in range(n)]
            return json.dumps({"translations": out}, ensure_ascii=False)
        return routing_handler(messages, agent, operation, json_mode)

    def test_quote_loss_caught_and_refixed(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Application(cfg, client=FakeClient(handler=self._handler)).run(txt)
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
            for e in linted:
                self.assertEqual(e.get("event_schema"), 2)
                self.assertNotIn("issues", e, "batch_linted 不再携带完整 issue 明文")
                self.assertEqual(e["issue_count"], sum(e["by_type"].values()))
                self.assertIn("quote_loss", e["by_type"])
                self.assertIn("issues_sha256", e)
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
    def _handler(messages, agent, operation, json_mode):
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
        return routing_handler(messages, agent, operation, json_mode)

    def test_polish_stripping_quotes_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = True
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Application(cfg, client=FakeClient(handler=self._handler)).run(txt)
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
            self.assertNotIn("polished", rejected[0], "被拒润色候选不再写明文")
            self.assertEqual(
                rejected[0]["proposal_sha256"],
                stable_digest("早上好他轻声说道"),
                "被拒候选以稳定摘要形式审计",
            )


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
    def _handler(messages, agent, operation, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            return json.dumps({"translations": ["short" for _ in range(n)]}, ensure_ascii=False)
        return routing_handler(messages, agent, operation, json_mode)

    def test_too_short_reported_but_not_retranslated(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = Config.from_dict({"llm": fake_llm_dict(), "quality": "economy"})
            cfg.source_lang = "en"
            cfg.state_dir = os.path.join(d, "state")
            client = FakeClient(handler=self._handler)
            store = Application(cfg, client=client).run(txt)

            translate_calls = [c for c in client.calls if "文学翻译" in c["messages"][0]["content"]]
            self.assertEqual(
                len(translate_calls), 1, "too_short 不属于可重译类型，不该触发定向重译"
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            self.assertFalse(any(e["event"] == "lint_refixed" for e in events))
            linted = [e for e in events if e["event"] == "batch_linted"]
            self.assertTrue(
                any("too_short" in e["by_type"] for e in linted),
                "过短译文应被 lint 发现并记入 batch_linted 的类型计数",
            )

            recorded = [
                i
                for i in store.load_progress(0).review_issue_dicts()
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
    def _handler(messages, agent, operation, json_mode):
        sys = messages[0]["content"]
        user = messages[-1]["content"]
        if "文学翻译" in sys:
            n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
            # 首译、定向重译都返回丢引号译文（模拟修复失败，issue 保持未解决）
            return json.dumps(
                {"translations": ["早上好他说道" for _ in range(n)]}, ensure_ascii=False
            )
        return routing_handler(messages, agent, operation, json_mode)

    def test_skip_branch_relints_without_retranslating(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            store = Application(cfg, client=FakeClient(handler=self._handler)).run(txt)
            unresolved = [
                i
                for i in store.load_progress(0).review_issue_dicts()
                if i.get("type") == "quote_loss" and i.get("stage") == "lint"
            ]
            self.assertTrue(unresolved, "首译丢引号且重译未修复，应作为未解决 lint issue 记录")

            # 模拟崩溃窗口：章已 DONE 但 review_issues 被清空、状态改回 pending，
            # 段译文原样保留（已落盘、未变）——续跑应走批跳过分支，不重译。
            pg = store.load_progress(0)
            pg.set_review_issue_dicts([])
            store.save_progress(0, pg)
            store.set_chapter_status(0, STATUS_PENDING)

            client2 = FakeClient(handler=self._handler)
            Application(cfg, client=client2).run(txt)

            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0, "跳过分支不得重译")

            recovered = [
                i
                for i in store.load_progress(0).review_issue_dicts()
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
            orch = Application(cfg, client=FakeClient(handler=routing_handler))
            orch.run_goal_result(
                txt,
                ExecutionGoal(
                    name="full",
                    phases=(
                        "prepare",
                        "prescan",
                        "translate",
                        "titles",
                        "qa",
                        "report",
                        "assemble",
                    ),
                    do_qa=True,
                ),
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


class TestLintFixMerged(unittest.TestCase):
    """同一批次内的多个待修复段落合并为一次定向重译调用，并逐段独立决定采纳或拒收。"""

    SRC = (
        "「一」と彼は静かな声で言った。\n"
        "\n"
        "「二」と彼女は静かな声で言った。\n"
        "\n"
        "「三」と先生は静かな声で言った。\n"
    )

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}")

    def _handler(self, fix_calls):
        def handler(messages, agent, operation, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in sys:
                if "审校意见" in user:
                    fix_calls.append(user)
                    items_block = user.split("待重译", 1)[-1]
                    n = len(re.findall(r"^\[(\d+)\] ", items_block, re.M))
                    # 前两段修复成功（补回引号），第三段修复仍未带引号 → 应被独立拒收
                    out = [f"“修复{i}”他说道" if i != 2 else "仍未带引号" for i in range(n)]
                    return json.dumps({"translations": out}, ensure_ascii=False)
                n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
                return json.dumps(
                    {"translations": [f"首译{i}丢引号" for i in range(n)]}, ensure_ascii=False
                )
            return routing_handler(messages, agent, operation, json_mode)

        return handler

    def test_three_flagged_segments_merge_into_one_call_with_independent_accept(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            fix_calls: list[str] = []
            client = FakeClient(handler=self._handler(fix_calls))
            store = Application(cfg, client=client).run(txt)

            self.assertEqual(
                len(fix_calls), 1, "3 个待修复段落应合并为一次调用，而不是分别发起调用。"
            )
            self.assertIn("【本批当前译文】", fix_calls[0], "多段合并 prompt 必须含整批当前译文块")
            items_block = fix_calls[0].split("待重译", 1)[-1]
            self.assertEqual(len(re.findall(r"^\[(\d+)\] ", items_block, re.M)), 3)
            item_indices = {int(m) for m in re.findall(r"^\[(\d+)\] ", items_block, re.M)}
            self.assertEqual(
                item_indices,
                {1, 2, 3},
                "待重译项编号必须是批内段号（与【本批当前译文】的下标一致），不能重新按 0 至 N-1 编号。",
            )

            ch = store.load_chapter(0)
            # text_segments[0] 是标题段，正文从 [1] 开始。
            self.assertIn("“", ch.text_segments[1].target, "第一段修复应被采纳")
            self.assertIn("“", ch.text_segments[2].target, "第二段修复应被采纳")
            self.assertNotIn(
                "“", ch.text_segments[3].target, "第三段修复未解决问题，应被拒收保留首译"
            )

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            refixed = {
                e["index"]: e for e in events if e["event"] == "lint_refixed" and e["chapter"] == 0
            }
            self.assertEqual(set(refixed), {1, 2}, "只有独立采纳的段才应产生 lint_refixed 事件")

            unresolved = [
                i
                for i in store.load_progress(0).review_issue_dicts()
                if i.get("type") == "quote_loss" and i.get("stage") == "lint"
            ]
            self.assertEqual(
                {i["index"] for i in unresolved}, {3}, "被拒收的第三段应作为未解决 lint issue 记录"
            )


class TestLintFixMergeFallback(unittest.TestCase):
    """合并调用返回数量不符时，回退到原有的逐段定向重译。"""

    SRC = "「一」と彼は静かな声で言った。\n\n「二」と彼女は静かな声で言った。\n"

    def _write(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# 第一章\n\n{self.SRC}")

    def _handler(self, fix_calls):
        def handler(messages, agent, operation, json_mode):
            sys = messages[0]["content"]
            user = messages[-1]["content"]
            if "文学翻译" in sys:
                if "审校意见" in user:
                    fix_calls.append(user)
                    items_block = user.split("待重译", 1)[-1]
                    n = len(re.findall(r"^\[(\d+)\] ", items_block, re.M))
                    if n > 1:
                        # 合并调用故意少返回一段：长度不符，触发回退
                        return json.dumps(
                            {"translations": ["“修复”他说道" for _ in range(n - 1)]},
                            ensure_ascii=False,
                        )
                    return json.dumps({"translations": ["“修复”他说道"]}, ensure_ascii=False)
                n = len(re.findall(r"^\[(\d+)\] ", user, re.M))
                return json.dumps(
                    {"translations": [f"首译{i}丢引号" for i in range(n)]}, ensure_ascii=False
                )
            return routing_handler(messages, agent, operation, json_mode)

        return handler

    def test_merge_length_mismatch_falls_back_to_per_segment_calls(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            self._write(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            cfg.pipeline.review = False
            cfg.pipeline.consistency_qa = False
            cfg.pipeline.book_understanding = False

            fix_calls: list[str] = []
            client = FakeClient(handler=self._handler(fix_calls))
            store = Application(cfg, client=client).run(txt)

            # 1 次合并调用（长度不符失败）+ 2 次逐段兜底调用 = 共 3 次带审校意见的调用
            self.assertEqual(len(fix_calls), 3)
            self.assertIn("【本批当前译文】", fix_calls[0], "首次合并调用仍应走多段模板")
            merged_items = fix_calls[0].split("待重译", 1)[-1]
            self.assertEqual(len(re.findall(r"^\[(\d+)\] ", merged_items, re.M)), 2)
            for single_call in fix_calls[1:]:
                single_items = single_call.split("待重译", 1)[-1]
                self.assertEqual(len(re.findall(r"^\[(\d+)\] ", single_items, re.M)), 1)
                # 回退单段路径必须走恢复后的单段模板：前文/后文译文独立成块，
                # 而不是合并路径使用的、无法映射段号的单一上下文块。
                self.assertIn("【前文译文】", single_call)
                self.assertIn("【后文译文】", single_call)
                self.assertLess(
                    single_call.index("【前文译文】"), single_call.index("【后文译文】")
                )

            ch = store.load_chapter(0)
            self.assertIn("“", ch.text_segments[1].target, "回退逐段调用应成功修复第一段")
            self.assertIn("“", ch.text_segments[2].target, "回退逐段调用应成功修复第二段")


class TestTitleTermTrim(unittest.TestCase):
    """标题翻译术语裁剪：仅注入标题文本命中的词条 + 锁定人物部分姓名命中；
    标题裁剪无条件生效，不受 glossary_scope 配置影响。"""

    def _run(self, d, scope):
        from trans_novel.glossary.store import GlossaryTerm

        state_dir = os.path.join(d, "state")
        cfg = _epub_config(state_dir)
        cfg.pipeline.glossary_scope = scope
        store = RunStore(os.path.join(state_dir, "book"))
        store.save_chapter(
            Chapter(
                index=0,
                title="The Duel of Vane",
                segments=[Segment(index=0, source="Body one.")],
            )
        )
        store.save_manifest(
            {
                "title": "Book",
                "fmt": "epub",
                "source_path": "",
                "source_lang": "en",
                "target_lang": "zh",
                "meta": {},
                "chapters": [
                    {
                        "index": 0,
                        "title": "The Duel of Vane",
                        "href": "a.xhtml",
                        "toc_entry_id": None,
                        "status": "done",
                    }
                ],
            }
        )
        _stamp_completed_store(store, chapters=1)
        glossary = GlossaryStore(store.glossary_path)
        # 锁定人物：标题里只出现「Vane」这个部分称呼（非全名），仍应命中保留。
        glossary.upsert_term(
            GlossaryTerm(source="Alden Vane", target="奥尔登·韦恩", type="人物", locked=True)
        )
        # source 直接在标题文本里出现：命中保留。
        glossary.upsert_term(GlossaryTerm(source="Duel", target="决斗", type="术语"))
        # 标题里完全未出现：应被裁剪掉。
        glossary.upsert_term(GlossaryTerm(source="Excalibur", target="王者之剑", type="术语"))
        glossary.close()

        captured = {}

        def handler(messages, agent, operation, json_mode):
            if "标题翻译" in messages[0]["content"]:
                captured["user"] = messages[-1]["content"]
            return routing_handler(messages, agent, operation, json_mode)

        glossary2 = GlossaryStore(store.glossary_path)
        Application(cfg, client=FakeClient(handler=handler)).translate_titles(store)
        glossary2.close()
        self.assertIn("user", captured)
        return captured["user"]

    def test_title_prompt_only_includes_title_hits(self):
        with tempfile.TemporaryDirectory() as d:
            user = self._run(d, "chapter")
            self.assertIn("Duel → 决斗", user)
            self.assertIn("Alden Vane → 奥尔登·韦恩", user)
            self.assertNotIn("Excalibur", user)

    def test_full_scope_does_not_bypass_title_trim(self):
        """glossary_scope=full 只影响章节翻译路径，标题裁剪始终无条件生效。"""
        with tempfile.TemporaryDirectory() as d:
            user = self._run(d, "full")
            self.assertIn("Duel → 决斗", user)
            self.assertIn("Alden Vane → 奥尔登·韦恩", user)
            self.assertNotIn("Excalibur", user)


class TestStrictFailurePropagation(unittest.TestCase):
    """必需节点的 provider 失败必须冒泡（失败态落盘 + 计划中止/应用边界抛出），
    不得伪装成成功空结果（reviewer finding 10）。"""

    @staticmethod
    def _boom(exc):
        def handler(messages, agent, operation, json_mode):
            raise exc

        return handler

    def test_digest_strict_propagates(self):
        from trans_novel.agents.synopsis import Synopsizer

        with tempfile.TemporaryDirectory() as d:
            cfg = _config(os.path.join(d, "state"))
            client = FakeClient(handler=self._boom(RuntimeError("digest down")))
            agent = Synopsizer(client, cfg)
            agent.src, agent.tgt = "ja", "zh"
            with self.assertRaisesRegex(RuntimeError, "digest down"):
                agent.digest_chapter_strict("テキスト")
            # 非 strict 变体保留旧回退语义（空串）
            self.assertEqual(agent.digest_chapter("テキスト"), "")

    def test_book_synopsis_strict_propagates(self):
        from trans_novel.agents.synopsis import Synopsizer

        with tempfile.TemporaryDirectory() as d:
            cfg = _config(os.path.join(d, "state"))
            client = FakeClient(handler=self._boom(RuntimeError("synopsis down")))
            agent = Synopsizer(client, cfg)
            agent.src, agent.tgt = "ja", "zh"
            with self.assertRaisesRegex(RuntimeError, "synopsis down"):
                agent.book_synopsis_strict(["梗概一", "梗概二"], "风格")
            self.assertEqual(agent.book_synopsis(["梗概一", "梗概二"], "风格"), "")

    def _complete_store(self, d: str) -> RunStore:
        txt = os.path.join(d, "novel.txt")
        write_sample_txt(txt)
        cfg = _config(os.path.join(d, "state"))
        store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
        return cfg, store

    def test_consistency_strict_propagates(self):
        from trans_novel.agents.consistency import ConsistencyChecker
        from trans_novel.glossary.store import GlossaryStore

        with tempfile.TemporaryDirectory() as d:
            cfg, store = self._complete_store(d)
            glossary = GlossaryStore(store.glossary_path)
            try:
                checker = ConsistencyChecker(
                    FakeClient(handler=self._boom(RuntimeError("qa down"))), cfg
                )
                checker.src, checker.tgt = "ja", "zh"
                with self.assertRaisesRegex(RuntimeError, "qa down"):
                    checker.check(store, glossary, strict=True)
                # 非 strict 变体保持旧回退语义（空 issue 列表）
                self.assertEqual(checker.check(store, glossary), [])
            finally:
                glossary.close()

    def test_backtranslate_strict_propagates(self):
        from trans_novel.agents.reviewer import BackTranslator

        with tempfile.TemporaryDirectory() as d:
            cfg = _config(os.path.join(d, "state"))
            agent = BackTranslator(FakeClient(handler=self._boom(RuntimeError("bt down"))), cfg)
            agent.src, agent.tgt = "ja", "zh"
            with self.assertRaisesRegex(RuntimeError, "bt down"):
                agent.check(["源一", "源二"], ["译一", "译二"], strict=True)
            # 回译数量对不齐是窄协议恢复：跳过样本、不抛
            agent2 = BackTranslator(FakeClient(handler=routing_handler), cfg)
            agent2.src, agent2.tgt = "ja", "zh"
            self.assertEqual(agent2.check(["源一"], ["译一", "译二"], strict=True), [])

    def test_titles_provider_failure_fails_node(self):
        from trans_novel.pipeline.runner import RequiredNodeFailed

        with tempfile.TemporaryDirectory() as d:
            cfg, store = self._complete_store(d)
            # 完整 run 已把 titles 标 succeeded 且标题已译 → 闭包判已满足、titles
            # 幂等跳过，provider 失败不会触发。清掉 titles 状态，并把章标题改成与
            # 正文 heading 源文不匹配的独立标题（绕过 heading 复用缓存），确保
            # titles 节点真正发起 LLM 调用。
            state = store.load_state()
            state.nodes.pop(NODE_TITLES, None)
            for ci, c in enumerate(state.chapters):
                c.title = f"待译标题{ci}"
                c.title_translated = None
            raw_toc = state.meta.get("toc_entries") if isinstance(state.meta, dict) else None
            if isinstance(raw_toc, list):
                for e in raw_toc:
                    if isinstance(e, dict):
                        e.pop("title_translated", None)
            store.save_state(state)

            def handler(messages, agent, operation, json_mode):
                if operation == "title.translate":
                    raise RuntimeError("title provider down")
                return routing_handler(messages, agent, operation, json_mode)

            with self.assertRaisesRegex(RequiredNodeFailed, "title provider down"):
                Application(cfg, client=FakeClient(handler=handler)).translate_titles(store)
            node = store.load_state().nodes[NODE_TITLES]
            self.assertEqual(node.status, "failed_permanent")
            self.assertEqual(node.failure.kind, "provider_permanent")

    def test_naturalize_screen_failure_not_marked(self):
        from trans_novel.agents.naturalizer import Naturalizer, naturalize_chapter
        from trans_novel.glossary.store import GlossaryStore
        from trans_novel.ingest.models import Chapter, Segment

        with tempfile.TemporaryDirectory() as d:
            cfg = _config(os.path.join(d, "state"))
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store = RunStore(os.path.join(d, "run"))
            store.save_state(
                RunState(
                    run_state_schema=RUN_STATE_SCHEMA_VERSION,
                    identity=RunIdentity(source_lang="ja", target_lang="zh"),
                    title="T",
                    fmt="text",
                    source_lang="ja",
                    target_lang="zh",
                    chapters=[ChapterIndex(index=0, title="第一章")],
                    progress={0: ChapterProgress()},
                )
            )
            chapter = Chapter(
                index=0,
                title="第一章",
                segments=[Segment(index=0, source="S0", target="翻译腔很重的句子。")],
            )
            store.save_chapter(chapter)
            glossary = GlossaryStore(store.glossary_path)
            try:
                agent = Naturalizer(
                    FakeClient(handler=self._boom(RuntimeError("screen down"))), cfg
                )
                agent.src, agent.tgt = "ja", "zh"
                with self.assertRaisesRegex(RuntimeError, "screen down"):
                    naturalize_chapter(
                        agent,
                        chapter,
                        0,
                        1,
                        [],
                        cfg,
                        store,
                        dry_run=False,
                        remaining=None,
                        strict_screen=True,
                    )
                # 筛查失败不得把整章标记为已自然化
                self.assertFalse(store.load_progress(0).naturalized)
            finally:
                glossary.close()


class TestGlossaryAfterPolish(unittest.TestCase):
    """inflight_glossary + polish：章级术语兜底抽取必须发生在润色落盘之后
    （润色前的译文可能含被润色修正的术语变体，reviewer finding 17）。"""

    def test_chapter_glossary_extracted_after_polish(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = True
            cfg.pipeline.inflight_glossary = True
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            polished = [e for e in events if e["event"] == "batch_polished"]
            fallback = [e for e in events if e["event"] == "chapter_glossary_extracted"]
            self.assertTrue(polished, "润色应产生 batch_polished")
            self.assertTrue(fallback, "章级兜底抽取应运行")
            # 逐章比较：每章的兜底抽取必须发生在该章最后一次 batch_polished 之后
            for ci in range(len(store.load_manifest()["chapters"])):
                ch_polished = [
                    i
                    for i, e in enumerate(events)
                    if e["event"] == "batch_polished" and e["chapter"] == ci
                ]
                ch_fallback = [
                    i
                    for i, e in enumerate(events)
                    if e["event"] == "chapter_glossary_extracted" and e["chapter"] == ci
                ]
                if not ch_fallback:
                    continue
                self.assertTrue(ch_polished, f"第 {ci} 章应先润色")
                self.assertGreater(
                    ch_fallback[0],
                    ch_polished[-1],
                    f"第 {ci} 章兜底抽取必须发生在润色落盘之后（否则锁死润色前的术语变体）",
                )


class TestTranslateRunBookkeeping(unittest.TestCase):
    """翻译边界的进度计数与生命周期事件（reviewer finding 18）。"""

    def test_progress_totals_initialized_and_run_events(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            seen: list[tuple[int, int, str]] = []
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(
                txt, progress=lambda done, total, label: seen.append((done, total, label))
            )

            mid = [(d, t) for d, t, _ in seen if t > 0]
            self.assertTrue(mid, "翻译中回调必须携带真实 total（不得恒为 0）")
            self.assertEqual(seen[-1][2], "翻译完成")
            self.assertEqual(seen[-1][0], seen[-1][1])

            with open(store.event_log_path, encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            names = [e["event"] for e in events]
            self.assertIn("translate_run_started", names)
            self.assertIn("translate_run_finished", names)
            started = next(e for e in events if e["event"] == "translate_run_started")
            self.assertGreater(started["total_segments"], 0)
            finished = next(e for e in events if e["event"] == "translate_run_finished")
            self.assertGreaterEqual(finished["total_segments"], 0)


class TestInitCrashRecovery(unittest.TestCase):
    """初始化崩溃窗口：analysis.json 已落盘但 manifest 未落盘 → 重试必须补完初始化。"""

    def test_analysis_without_manifest_recovers_on_retry(self):
        from trans_novel.ingest.segmenter import load_document

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            doc = load_document(txt, "ja", "zh")
            # 模拟 save_analysis 之后、save_manifest 之前的崩溃：analysis.json 存在、
            # 章节文件存在、manifest 不存在。
            store = RunStore(os.path.join(d, "state", slugify(doc.title)))
            store.ensure_dirs()
            store.save_analysis({"style": {"tone": "cool"}})
            for ci, title in ((0, "第一章　出会い"), (1, "第二章　放課後")):
                store.save_chapter(
                    Chapter(
                        index=ci,
                        title=title,
                        segments=[Segment(index=0, source="源文", kind="heading")],
                    )
                )
            self.assertFalse(store.exists())

            retried = Application(cfg, client=FakeClient(handler=routing_handler)).prepare(txt)
            self.assertTrue(retried.exists(), "重试必须原子补完初始化 manifest")
            self.assertTrue(retried.load_manifest()["initialized"])
            self.assertIsNotNone(retried.load_analysis())


class TestTitlesMismatchRetry(unittest.TestCase):
    """标题数量不符是协议错误：节点失败可重试；随后正常返回时成功。"""

    def _handler(self, fail_first):
        state = {"fail": fail_first}

        def handler(messages, agent, operation, json_mode):
            if operation == "title.translate":
                if state["fail"]:
                    state["fail"] = False
                    return json.dumps({"titles": ["只有一条"]}, ensure_ascii=False)  # 数量不符
                return routing_handler(messages, agent, operation, json_mode)
            return routing_handler(messages, agent, operation, json_mode)

        return handler

    def test_mismatch_then_valid_resume(self):
        from trans_novel.pipeline.runner import RequiredNodeFailed
        from trans_novel.pipeline.state import NODE_TITLES

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            handler = self._handler(fail_first=True)
            app = Application(cfg, client=FakeClient(handler=handler))
            state = store.load_state()
            state.nodes.pop(NODE_TITLES, None)
            for ci, c in enumerate(state.chapters):
                c.title = f"待译标题{ci}"
                c.title_translated = None
            store.save_state(state)
            with self.assertRaises(RequiredNodeFailed):
                app.translate_titles(store)
            node = store.load_state().nodes[NODE_TITLES]
            self.assertEqual(node.status, "failed_retryable")
            self.assertEqual(node.failure.kind, "protocol")

            # 恢复：再次 translate_titles → 正常数量 → 成功且标题落盘
            client2 = FakeClient(handler=self._handler(fail_first=False))
            Application(cfg, client=client2).translate_titles(store)
            self.assertEqual(store.load_state().nodes[NODE_TITLES].status, "succeeded")
            m = store.load_manifest()
            self.assertTrue(
                all(c.get("title_translated") for c in m["chapters"]),
                "重试成功后标题必须落盘",
            )


class TestQaPersistence(unittest.TestCase):
    """一致性 QA 产物跨调用持久化：tools qa 后 tools report 不丢问题。"""

    @staticmethod
    def _issue_handler(messages, agent, operation, json_mode):
        if operation == "consistency.check":
            return json.dumps(
                {"issues": [{"type": "terminology", "detail": "译法漂移"}]},
                ensure_ascii=False,
            )
        return routing_handler(messages, agent, operation, json_mode)

    def test_qa_then_report_across_invocations_keeps_issues(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            issues = Application(cfg, client=FakeClient(handler=self._issue_handler)).qa(store)
            self.assertTrue(issues, "tools qa 应返回问题")
            # 换新 client（模拟新进程）跑 tools report：仍应拿到持久化的问题
            report = Application(cfg, client=FakeClient(handler=routing_handler)).report(store)
            self.assertEqual(
                report["consistency_issues"],
                issues,
                "跨调用 report 不得丢 QA 问题（回退为 []）",
            )


class TestStrictSchemaValidation(unittest.TestCase):
    """必需节点的严格 schema 校验：缺失键/错误形状 = 协议错误，不是成功空结果。"""

    def test_consistency_missing_key_is_protocol_error(self):
        from trans_novel.agents.consistency import ConsistencyChecker
        from trans_novel.glossary.store import GlossaryStore
        from trans_novel.pipeline.contracts import classify_failure

        def handler(messages, agent, operation, json_mode):
            if operation == "consistency.check":
                return json.dumps({"wrong_key": []}, ensure_ascii=False)
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            glossary = GlossaryStore(store.glossary_path)
            try:
                checker = ConsistencyChecker(FakeClient(handler=handler), cfg)
                checker.src, checker.tgt = "ja", "zh"
                with self.assertRaises(Exception) as ctx:
                    checker.check(store, glossary, strict=True)
                self.assertEqual(
                    classify_failure(ctx.exception),
                    "protocol",
                    "缺失键必须是协议错误（可重试），不是成功空结果",
                )
            finally:
                glossary.close()


class TestPolishEnablement(unittest.TestCase):
    """polish 从禁用切到启用：已译章一次性补润色（不清除/不重译）。"""

    def test_polish_enabled_later_polishes_existing_translations(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            cfg.pipeline.polish = False
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            ch = store.load_chapter(0)
            self.assertTrue(all(s.target and s.target.startswith("译") for s in ch.text_segments))

            cfg.pipeline.polish = True
            client2 = FakeClient(handler=routing_handler)
            Application(cfg, client=client2).run(txt)
            ch2 = store.load_chapter(0)
            self.assertTrue(
                all(s.target and s.target.startswith("润") for s in ch2.text_segments),
                "启用 polish 后已译章必须被润色",
            )
            self.assertFalse(store.load_progress(0).pending_polish)
            # 未重译：续跑只补润色
            translate_calls = [
                c for c in client2.calls if "文学翻译" in c["messages"][0]["content"]
            ]
            self.assertEqual(len(translate_calls), 0)


class TestReportFailedNodes(unittest.TestCase):
    """尽力而为节点失败必须出现在 QA 报告里（不能呈现一切正常）。"""

    def test_failed_best_effort_node_visible_in_report(self):
        from trans_novel.assemble.report import build_report
        from trans_novel.glossary.store import GlossaryStore
        from trans_novel.pipeline.state import NODE_MINE_TERMS

        def handler(messages, agent, operation, json_mode):
            if operation == "prescan.mine_terms" or "术语候选挖掘" in messages[0]["content"]:
                raise RuntimeError("mining down")
            return routing_handler(messages, agent, operation, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=handler)).run(txt)
            state = store.load_state()
            self.assertEqual(state.nodes[NODE_MINE_TERMS].status, "failed_permanent")
            glossary = GlossaryStore(store.glossary_path)
            try:
                report = build_report(store, glossary)
            finally:
                glossary.close()
            self.assertGreater(report["summary"]["failed_nodes"], 0)
            self.assertTrue(
                any(n["node"] == NODE_MINE_TERMS for n in report["failed_nodes"]),
                "报告必须列出失败的尽力而为节点",
            )


if __name__ == "__main__":
    unittest.main()
