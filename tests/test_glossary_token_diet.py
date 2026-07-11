"""术语表 token 瘦身三项功能的验收测试（离线 FakeClient，不发网络）。

覆盖三条契约（均针对 `_translate_chapter` 与 `GlossaryExtractor.store_terms`）：

(2) full 档附属章：走完整翻译→润色→审校→回译流水线，但所有抽取点被 `if not bm` 守住
    → 零 batch_glossary_extracted / chapter_glossary_extracted；正文章照常抽词。
(3) 批内内联抽取的『已有对照表』按本批源文裁剪（terms_in(snapshot, batch_src)），
    别的批才出现的历史词条不进本批抽取 prompt。
(4) 章级快照条件刷新（不变量 d）：批内新词命中「剩余源文（当前批之后）」才刷新快照——
    不命中则复用旧对象，相邻批翻译 prompt 的术语表块逐字节相同（保 DeepSeek 前缀缓存）；
    命中则刷新，下一批 prompt 立即出现新词。翻译走章级冻结快照、不按批裁剪，故只有
    「是否新增新词」会改变术语块。

既有测试 `test_batch_glossary_refreshes_following_prompts` 覆盖了 (4) 命中侧的另一样例，
`TestBackMatterFull` 覆盖了 (2) 的粗粒度断言；这里聚焦增量：(2) 的完整流水线阶段、
(3) 的按批裁剪、(4) 的不命中逐字节冻结 + 命中刷新的两侧闭环。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.fake_llm import routing_handler
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.orchestrator import Orchestrator


def _cfg(
    state_dir,
    *,
    back_matter="full",
    max_chars_per_batch=1800,
    review=True,
    polish=True,
    backtranslate=1.0,
    book_understanding=True,
    consistency_qa=True,
    inflight_glossary=True,
):
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
            "segment": {"max_chars_per_batch": max_chars_per_batch, "max_chars_per_segment": 1200},
            "pipeline": {
                "review": review,
                "polish": polish,
                "backtranslate_sample": backtranslate,
                "consistency_qa": consistency_qa,
                "book_understanding": book_understanding,
                "back_matter": back_matter,
                "inflight_glossary": inflight_glossary,
            },
            "paths": {"state_dir": state_dir},
        }
    )


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _events(store) -> list[dict]:
    with open(store.event_log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _lit_prompts(client) -> list[str]:
    """所有翻译（文学翻译）调用的 user prompt。"""
    return [
        c["messages"][-1]["content"]
        for c in client.calls
        if "文学翻译" in c["messages"][0]["content"]
    ]


def _extract_prompts(client) -> list[str]:
    """所有术语抽取调用的 user prompt。"""
    return [
        c["messages"][-1]["content"]
        for c in client.calls
        if "术语" in c["messages"][0]["content"] and "抽取器" in c["messages"][0]["content"]
    ]


def _translate_glossary_block(prompt: str) -> str:
    """截取翻译 prompt 里的【专有名词对照表】块（到下一个【为止）。"""
    return prompt.split("【专有名词对照表】", 1)[1].split("【", 1)[0]


def _extract_existing_block(prompt: str) -> str:
    """截取抽取 prompt 里的【已有对照表…】块（到【原文为止）。"""
    return prompt.split("【已有对照表", 1)[1].split("【原文", 1)[0]


def _extract_source_block(prompt: str) -> str:
    """截取抽取 prompt 里的【原文…】块（到【译文为止）。"""
    return prompt.split("【原文", 1)[1].split("【译文", 1)[0]


def _translate_source_block(prompt: str) -> str:
    """截取翻译 prompt 里的待译源文（到「请翻译」为止）。"""
    return prompt.split("段落】", 1)[1].split("请翻译", 1)[0]


class TestBackMatterFullNoExtraction(unittest.TestCase):
    """(2) full 档附属章：完整流水线照跑，但零术语抽取；正文章仍抽词。"""

    def test_full_backmatter_runs_pipeline_without_extraction(self):
        """full 档下 is_back_matter 命中的附属章不再旁路：翻译/润色/审校/回译事件齐全，
        但 `if not bm` 守住全部抽取点 → 无任何抽词事件；正文章照常抽词。"""
        body = "綾小路は教室の窓際に座っていた。空はどこまでも青かった。" + "あ" * 220
        dialog = "「おはよう」と堀北が声をかけた。彼女は無表情だった。"
        notes = "1. Endnote on chapter one, page 12.\n\n2. Bibliography: Doe, J. (2020). A Study."
        doc = f"# 第一章 出会い\n\n{body}\n\n{dialog}\n\n# Notes\n\n{notes}\n"
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write(txt, doc)
            cfg = _cfg(os.path.join(d, "state"), back_matter="full")
            store = Orchestrator(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            m = store.load_manifest()
            bm = next(c["index"] for c in m["chapters"] if is_back_matter(c["title"]))
            body_ci = next(c["index"] for c in m["chapters"] if not is_back_matter(c["title"]))
            events = _events(store)
            bm_ev = {e["event"] for e in events if e.get("chapter") == bm}
            body_ev = {e["event"] for e in events if e.get("chapter") == body_ci}

            # 未旁路：全程不产生 chapter_back_matter
            self.assertFalse(
                any(e["event"] == "chapter_back_matter" for e in events), "full 档不得旁路附属章"
            )
            # 附属章跑了完整流水线（翻译 + 润色 + 审校 + 回译，不止首阶段翻译）
            self.assertTrue(
                {
                    "batch_translated",
                    "batch_polished",
                    "chapter_reviewed",
                    "chapter_backtranslation_checked",
                }.issubset(bm_ev),
                f"full 档附属章应走完整流水线，实际事件={sorted(bm_ev)}",
            )
            # 但零抽词事件（所有抽取点被 not bm 守住）
            self.assertNotIn("batch_glossary_extracted", bm_ev, "full 档附属章不得批内抽词")
            self.assertNotIn("chapter_glossary_extracted", bm_ev, "full 档附属章不得章末兜底抽词")
            # 正文章不受影响，仍抽词
            self.assertIn("batch_glossary_extracted", body_ev, "正文章应批内抽词")
            self.assertIn("chapter_glossary_extracted", body_ev, "正文章应章末兜底抽词")


class TestBatchExtractionPruning(unittest.TestCase):
    """(3) 抽取按批裁剪：批内内联抽取的『已有对照表』只含本批源文命中的词条。"""

    @staticmethod
    def _extractor_returns_nothing(messages, tier, json_mode):
        system = messages[0]["content"]
        if "术语" in system and "抽取器" in system:
            return json.dumps({"terms": []}, ensure_ascii=False)
        return routing_handler(messages, tier, json_mode)

    def test_inline_extraction_existing_pruned_to_batch(self):
        """预种两个 source 分处不同批的词条：批内抽取的 existing 只保留本批命中的那个，
        别的批的词条不进本批抽取 prompt（章末兜底抽取按整章裁剪、含两者，不在本契约内）。"""
        doc = "# 章\n\n红猫睡。\n\n蓝狗吠。\n"
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write(txt, doc)
            cfg = _cfg(
                os.path.join(d, "state"),
                max_chars_per_batch=8,
                review=False,
                polish=False,
                backtranslate=0.0,
                book_understanding=False,
                consistency_qa=False,
            )
            orch = Orchestrator(cfg, client=FakeClient(handler=self._extractor_returns_nothing))
            store = orch.prepare(txt)
            g = GlossaryStore(store.glossary_path)
            g.upsert_term(GlossaryTerm(source="红猫", target="红色猫", type="术语"))
            g.upsert_term(GlossaryTerm(source="蓝狗", target="蓝色狗", type="术语"))
            g.close()

            client = FakeClient(handler=self._extractor_returns_nothing)
            Orchestrator(cfg, client=client).run(txt)

            # 内联抽取：原文只含单个词条那批 → 分别定位；章末兜底原文含两者 → 天然排除。
            cat_only = dog_only = None
            for prompt in _extract_prompts(client):
                src = _extract_source_block(prompt)
                if "红猫" in src and "蓝狗" not in src:
                    cat_only = prompt
                elif "蓝狗" in src and "红猫" not in src:
                    dog_only = prompt
            self.assertIsNotNone(cat_only, "缺少红猫批的内联抽取调用")
            self.assertIsNotNone(dog_only, "缺少蓝狗批的内联抽取调用")

            cat_existing = _extract_existing_block(cat_only)
            self.assertIn("红猫", cat_existing)  # 本批命中 → 保留
            self.assertNotIn("蓝狗", cat_existing)  # 非本批 → 裁掉

            dog_existing = _extract_existing_block(dog_only)
            self.assertIn("蓝狗", dog_existing)
            self.assertNotIn("红猫", dog_existing)


class TestBatchSnapshotFreeze(unittest.TestCase):
    """(4) 章级快照条件刷新（不变量 d）：新词命中剩余源文才刷新，否则术语块逐字节冻结。"""

    def _run(self, doc: str, new_source: str, new_target: str) -> dict[str, str]:
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if "术语" in system and "抽取器" in system:
                # 仅当本批相关内容里出现新词时确认它（模拟“该批抽出新词”）。
                if new_source in user:
                    return json.dumps(
                        {"terms": [{"source": new_source, "target": new_target, "type": "称谓"}]},
                        ensure_ascii=False,
                    )
                return json.dumps({"terms": []}, ensure_ascii=False)
            return routing_handler(messages, tier, json_mode)

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            _write(txt, doc)
            cfg = _cfg(
                os.path.join(d, "state"),
                max_chars_per_batch=6,
                review=False,
                polish=False,
                backtranslate=0.0,
                book_understanding=False,
                consistency_qa=False,
            )
            orch = Orchestrator(cfg, client=FakeClient(handler=handler))
            store = orch.prepare(txt)
            g = GlossaryStore(store.glossary_path)
            # 章内每段都含的基准词：让术语块非空、可作逐字节比较的稳定内容。
            g.upsert_term(GlossaryTerm(source="基準詞", target="基准译", type="术语"))
            g.close()

            client = FakeClient(handler=handler)
            Orchestrator(cfg, client=client).run(txt)

            blocks: dict[str, str] = {}
            for prompt in _lit_prompts(client):
                src = _translate_source_block(prompt)
                for mk in ("アルファ", "ベータ", "ガンマ"):
                    if mk in src:
                        blocks[mk] = _translate_glossary_block(prompt)
            return blocks

    def test_miss_freezes_block_byte_for_byte(self):
        """新词只在首段源文、不在其后续段 → 不刷新：随后各批翻译 prompt 的术语块逐字节相同，
        且冻结的快照里始终不含该新词。"""
        doc = "# 章\n\n基準詞と甲詞アルファ。\n\n基準詞ベータ。\n\n基準詞ガンマ。\n"
        blocks = self._run(doc, "甲詞", "甲译")
        self.assertEqual(set(blocks), {"アルファ", "ベータ", "ガンマ"}, "三段都应各自成批并被翻译")
        self.assertEqual(blocks["アルファ"], blocks["ベータ"], "不命中 → 术语块逐字节不变")
        self.assertEqual(blocks["ベータ"], blocks["ガンマ"], "不命中 → 术语块逐字节不变")
        for mk, b in blocks.items():
            self.assertNotIn("甲詞", b, f"{mk} 批的冻结快照不应含未命中剩余源文的新词")

    def test_hit_refreshes_and_exposes_new_term(self):
        """新词在首段与后续段均出现 → 命中剩余源文 → 刷新：下一批翻译 prompt 立即出现新词，
        且术语块相对首批发生变化（不变量 d 的命中侧）。"""
        doc = "# 章\n\n基準詞と触発詞アルファ。\n\n基準詞と触発詞ベータ。\n\n基準詞ガンマ。\n"
        blocks = self._run(doc, "触発詞", "触发译")
        self.assertEqual(set(blocks), {"アルファ", "ベータ", "ガンマ"})
        self.assertNotIn("触発詞", blocks["アルファ"], "首批翻译发生在抽取之前，尚不含新词")
        self.assertIn("触発詞", blocks["ベータ"], "命中剩余源文 → 下一批可见新词")
        self.assertNotEqual(blocks["アルファ"], blocks["ベータ"], "命中 → 术语块应刷新变化")


if __name__ == "__main__":
    unittest.main()
