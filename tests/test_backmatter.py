"""附属章节（back matter）旁路测试（离线 FakeClient）。

守护两层契约：
1. trans_novel/pipeline/backmatter.py 的标题识别 is_back_matter；
2. orchestrator 在 config.pipeline.back_matter=skip/light/full 三档下对附属章的旁路行为：
   - light：fast 档粗翻，跳过润色/审校/术语/回译/预扫梗概；
   - skip：原文直通，附属章不发任何翻译调用（seg.target==seg.source）；
   - full：逃生舱（对照组），附属章回落到与正文相同的完整流水线。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trans_novel.config import Config
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.orchestrator import Orchestrator
from trans_novel.pipeline.runstore import STATUS_DONE
from tests.fake_llm import routing_handler

# Notes 正文里的独特标记：一旦出现在任何翻译调用的 user prompt 里，
# 就说明附属章源文被送进了翻译模型——skip 档下这是契约违背。
_BM_MARKER = "ZZQ_BACKMATTER_MARKER"


def _write_doc(path: str) -> None:
    """一篇正文章（日文、长段避开风格采样边界）+ 一个 '# Notes' 附属章（英文/数字）。"""
    # >200 字符长段，避免风格采样取到过短片段的边界。
    body_para = "綾小路は教室の窓際に座っていた。空はどこまでも青く鳥が鳴いていた。" + "あ" * 220
    dialog = "「おはよう、綾小路くん」と堀北が声をかけた。彼女はいつも通り無表情だった。"
    notes = (
        f"1. Endnote {_BM_MARKER} on chapter one, page 12.\n\n"
        "2. Bibliography entry: Doe, J. (2020). A Study. Some Press."
    )
    content = f"# 第一章 出会い\n\n{body_para}\n\n{dialog}\n\n# Notes\n\n{notes}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _config(state_dir: str, back_matter: str) -> Config:
    return Config.from_dict({
        "language": {"source": "ja", "target": "zh"},
        "llm": {"provider": "fake", "tiers": {
            "strong": {"model": "p"}, "cheap": {"model": "f"}}},
        "segment": {"max_chars_per_batch": 1800},
        # 打开 review/polish/backtranslate，让「附属章没有这些事件」成为有信号的断言
        # ——正文章会产生它们，附属章旁路后才不产生。
        "pipeline": {"review": True, "polish": True,
                     "backtranslate_sample": 1.0, "consistency_qa": True,
                     "back_matter": back_matter},
        "paths": {"state_dir": state_dir},
    })


def _events(store) -> list[dict]:
    with open(store.event_log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _bm_index(store) -> int:
    """从 manifest 里定位附属章的章序号（不硬编码，跟随 is_back_matter）。"""
    m = store.load_manifest()
    return next(c["index"] for c in m["chapters"] if is_back_matter(c["title"]))


def _lit_calls(calls, tier=None):
    """翻译调用（system 含「文学翻译」），可按 tier 过滤。"""
    return [c for c in calls
            if "文学翻译" in c["messages"][0]["content"]
            and (tier is None or c["tier"] == tier)]


class TestIsBackMatter(unittest.TestCase):
    """标题识别：附属章关键词命中，普通章/前言/序章不命中。"""

    def test_positive_titles(self):
        # 英文词边界（大小写不敏感）+ 短语，中文子串
        positives = [
            "Notes", "NOTES", "Endnotes", "Index",
            "Bibliography", "Selected Bibliography", "References",
            "Acknowledgments", "Acknowledgements", "Copyright",
            "Works Cited", "About the Author", "About the Authors",
            "注释", "索引", "参考文献", "致谢", "版权", "关于作者",
        ]
        for title in positives:
            with self.subTest(title=title):
                self.assertTrue(is_back_matter(title))

    def test_negative_titles(self):
        # 空串、正文章、前言/序章等叙事内容不得误判为附属章
        negatives = [
            "", "Chapter 1", "第一章 出会い",
            "The Beginning", "Introduction", "Prologue",
        ]
        for title in negatives:
            with self.subTest(title=title):
                self.assertFalse(is_back_matter(title))


class TestBackMatterLight(unittest.TestCase):
    """light 档：附属章走 fast 档粗翻，跳过润色/审校/术语/回译，正文仍走强档。"""

    def test_light_bypass(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write_doc(txt)
            cfg = _config(os.path.join(d, "state"), "light")

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            bm = _bm_index(store)
            bm_events = [e for e in _events(store) if e.get("chapter") == bm]

            # (a) 旁路专属事件：chapter_back_matter(mode=light) + light 批次翻译（fast 档）
            cbm = [e for e in bm_events if e["event"] == "chapter_back_matter"]
            self.assertEqual(len(cbm), 1)
            self.assertEqual(cbm[0]["mode"], "light")
            bts = [e for e in bm_events if e["event"] == "batch_translated"]
            self.assertTrue(bts, "light 档附属章应产生 batch_translated")
            for e in bts:
                self.assertTrue(e.get("back_matter"))
                self.assertEqual(e.get("tier"), "fast")

            # (b) 完整流水线的重活一律不落到附属章
            for ev in ("batch_polished", "chapter_reviewed",
                       "batch_glossary_extracted", "chapter_glossary_extracted",
                       "chapter_backtranslation_checked"):
                self.assertFalse(
                    any(e["event"] == ev for e in bm_events),
                    f"附属章不应产生 {ev}")
            # 预扫梗概亦跳过附属章
            self.assertFalse(
                any(e["event"] == "book_understanding_chapter_digest_saved"
                    for e in bm_events),
                "预扫逐章梗概应跳过附属章")

            # (c) 附属章走 fast 档、正文走 strong 档，两类翻译调用都在
            self.assertTrue(_lit_calls(client.calls, tier="fast"),
                            "附属章 light 翻译应为 fast 档")
            self.assertTrue(_lit_calls(client.calls, tier="strong"),
                            "正文翻译应为 strong 档")

            # (d) 附属章每段都有译文
            ch = store.load_chapter(bm)
            self.assertTrue(ch.text_segments)
            self.assertTrue(all(s.target for s in ch.text_segments))

            # (e) 附属章在 manifest 标 done
            status = next(c["status"] for c in store.load_manifest()["chapters"]
                         if c["index"] == bm)
            self.assertEqual(status, STATUS_DONE)


class TestBackMatterSkip(unittest.TestCase):
    """skip 档：附属章原文直通，零翻译调用，target 等于 source。"""

    def test_skip_bypass(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write_doc(txt)
            cfg = _config(os.path.join(d, "state"), "skip")

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            bm = _bm_index(store)

            # (a) 原文直通：每段译文即原文
            ch = store.load_chapter(bm)
            self.assertTrue(ch.text_segments)
            self.assertTrue(all(s.target == s.source for s in ch.text_segments))

            # (b) 事件：chapter_back_matter(skip) + chapter_done(back_matter,skip)；
            #     没有 batch_translated，也没有润色/审校/术语/回译
            bm_events = [e for e in _events(store) if e.get("chapter") == bm]
            cbm = [e for e in bm_events if e["event"] == "chapter_back_matter"]
            self.assertEqual(len(cbm), 1)
            self.assertEqual(cbm[0]["mode"], "skip")
            done = [e for e in bm_events if e["event"] == "chapter_done"]
            self.assertEqual(len(done), 1)
            self.assertTrue(done[0].get("back_matter"))
            self.assertEqual(done[0]["mode"], "skip")
            for ev in ("batch_translated", "batch_polished", "chapter_reviewed",
                       "batch_glossary_extracted", "chapter_glossary_extracted",
                       "chapter_backtranslation_checked"):
                self.assertFalse(
                    any(e["event"] == ev for e in bm_events),
                    f"skip 档附属章不应产生 {ev}")

            # (c) 附属章源文从未进入任何翻译调用
            self.assertTrue(_lit_calls(client.calls), "正文仍应被翻译")
            for c in _lit_calls(client.calls):
                self.assertNotIn(_BM_MARKER, c["messages"][-1]["content"],
                                 "附属章源文不得被送进翻译模型")


class TestBackMatterFull(unittest.TestCase):
    """full 档（逆向对照）：附属章无旁路，回落到完整流水线并正常抽词。"""

    def test_full_runs_normal_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write_doc(txt)
            cfg = _config(os.path.join(d, "state"), "full")

            client = FakeClient(handler=routing_handler)
            store = Orchestrator(cfg, client=client).run(txt)

            bm = _bm_index(store)
            events = _events(store)

            # 无旁路：全程不产生 chapter_back_matter
            self.assertFalse(any(e["event"] == "chapter_back_matter" for e in events),
                             "full 档不得旁路附属章")
            # 逃生舱：附属章走完整流水线 → 产生抽词事件
            bm_events = [e for e in events if e.get("chapter") == bm]
            self.assertTrue(
                any(e["event"] in ("chapter_glossary_extracted",
                                   "batch_glossary_extracted") for e in bm_events),
                "full 档附属章应走完整流水线并抽取术语")


class TestBackMatterResume(unittest.TestCase):
    """断点续跑：附属章已 done 后再 run，不得重翻。"""

    def test_resume_skips_translated_back_matter(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write_doc(txt)
            cfg = _config(os.path.join(d, "state"), "light")

            Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            # 换新 client 续跑：全部章已 done，不应再发任何翻译调用
            client2 = FakeClient(handler=routing_handler)
            Orchestrator(cfg, client=client2).run(txt)
            self.assertEqual(len(_lit_calls(client2.calls)), 0,
                             "续跑不得重翻任何章（含附属章 fast 档粗翻）")


if __name__ == "__main__":
    unittest.main()
