"""分析器 / 术语抽取 / 滚动上下文 的测试（离线）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trans_novel.config import Config
from trans_novel.llm.base import FakeClient
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm, TYPE_PERSON
from trans_novel.glossary.extractor import GlossaryExtractor
from trans_novel.agents.analyzer import Analyzer
from trans_novel.pipeline.context import RollingContext


def _cfg():
    return Config.from_dict({
        "language": {"source": "ja", "target": "zh"},
        "llm": {"provider": "fake", "tiers": {
            "strong": {"model": "p"}, "cheap": {"model": "f"}}},
    })


class TestAnalyzer(unittest.TestCase):
    def test_analyze_and_seed(self):
        analysis = {
            "genre": "校园", "tone": "冷峻第三人称",
            "style_guide": "保持克制",
            "characters": [{"source": "綾小路", "target": "绫小路",
                            "gender": "男", "reading": "あやのこうじ", "note": "第一人称用俺"}],
            "terms": [{"source": "高度育成高校", "target": "高度育成高中", "type": "组织"}],
        }
        client = FakeClient(handler=lambda m, t, j: json.dumps(analysis, ensure_ascii=False))
        a = Analyzer(client, _cfg())
        result = a.analyze("……样章……")
        self.assertEqual(result["genre"], "校园")

        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            n = a.seed_glossary(store, result)
            self.assertEqual(n, 2)
            self.assertEqual(store.get_term("綾小路").gender, "男")
            self.assertEqual(store.get_term("高度育成高校").type, "组织")
            store.close()

        brief = a.style_brief(result)
        self.assertIn("绫小路", brief)


class TestExtractor(unittest.TestCase):
    def test_extract_and_store(self):
        terms = {"terms": [
            {"source": "堀北", "target": "堀北", "type": "人物", "gender": "女",
             "aliases": ["堀北さん"]},
            {"source": "屋上", "target": "天台", "type": "地名", "gender": "未知"},
        ]}
        client = FakeClient(handler=lambda m, t, j: json.dumps(terms, ensure_ascii=False))
        ext = GlossaryExtractor(client, _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            summary, _ = ext.extract_and_store(store, "原文", "译文", chapter=1)
            self.assertEqual(summary["inserted"], 2)
            horikita = store.get_term("堀北")
            self.assertEqual(horikita.gender, "女")
            self.assertEqual(horikita.aliases, ["堀北さん"])
            self.assertEqual(horikita.first_chapter, 1)
            self.assertEqual(store.get_term("屋上").gender, "")
            store.close()

    def test_store_terms_independent_db_write(self):
        """store_terms 应只做入库，不触发任何 LLM 调用。"""
        client = FakeClient()
        ext = GlossaryExtractor(client, _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            terms = [GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON, gender="女")]
            summary, changed = ext.store_terms(store, terms, chapter=3)
            self.assertEqual(summary, {"inserted": 1, "updated": 0, "conflict": 0, "unchanged": 0})
            self.assertEqual([t.source for t in changed], ["堀北"])
            self.assertEqual(store.get_term("堀北").first_chapter, 3)
            self.assertEqual(client.calls, [])
            store.close()

    def test_store_terms_changed_tracks_inserted_and_updated_only(self):
        """changed 只含 inserted/updated；unchanged/conflict 不进——批内条件刷新据此决定
        是否重建章级快照，故语义必须精确。"""
        client = FakeClient()
        ext = GlossaryExtractor(client, _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            # 预置：高置信词条（制造 conflict）+ 低置信词条（制造 updated）。
            store.upsert_term(
                GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON,
                             confidence="high"), chapter=1)
            store.upsert_term(
                GlossaryTerm(source="屋上", target="天台", type="地名",
                             confidence="low"), chapter=1)
            terms = [
                GlossaryTerm(source="龙园", target="龙园", type=TYPE_PERSON),          # inserted
                GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON),          # unchanged
                GlossaryTerm(source="屋上", target="屋顶", type="地名",
                             confidence="high"),                                       # updated（新胜出）
                GlossaryTerm(source="堀北", target="北堀", type=TYPE_PERSON,
                             confidence="low"),                                        # conflict（现有胜出）
            ]
            summary, changed = ext.store_terms(store, terms, chapter=2)
            self.assertEqual(
                summary, {"inserted": 1, "updated": 1, "conflict": 1, "unchanged": 1})
            self.assertEqual({t.source for t in changed}, {"龙园", "屋上"})
            store.close()

    def test_extract_and_store_trims_existing_for_prompt(self):
        """existing 只保留本次原文命中的词条 + 锁定人物，其余不进 prompt。"""
        captured = {}

        def handler(messages, tier, json_mode):
            captured["user"] = messages[1]["content"]
            return json.dumps({"terms": []}, ensure_ascii=False)

        client = FakeClient(handler=handler)
        ext = GlossaryExtractor(client, _cfg())
        with tempfile.TemporaryDirectory() as d:
            store = GlossaryStore(os.path.join(d, "g.db"))
            store.upsert_term(GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON), chapter=1)
            store.upsert_term(
                GlossaryTerm(source="龙园", target="龙园", type=TYPE_PERSON, locked=True), chapter=1)
            store.upsert_term(GlossaryTerm(source="屋上", target="天台", type="地名"), chapter=1)
            ext.extract_and_store(store, "堀北在教室", "堀北在教室的翻译", chapter=2)
            prompt = captured["user"]
            self.assertIn("堀北", prompt)  # 本章原文命中
            self.assertIn("龙园", prompt)  # 未命中但锁定人物，兜底保留
            self.assertNotIn("屋上", prompt)  # 未命中且非锁定人物，裁掉
            store.close()


class TestRollingContext(unittest.TestCase):
    def test_render_and_bound(self):
        ctx = RollingContext(max_recent_keep=3)
        ctx.add_targets(["a", "b", "c", "d", "e"])
        self.assertEqual(ctx.recent_targets, ["c", "d", "e"])  # 限长
        rendered = ctx.render(n_recent=2)  # 只取最近两段
        self.assertIn("d", rendered)
        self.assertIn("e", rendered)
        self.assertNotIn("c", rendered)

    def test_roundtrip(self):
        ctx = RollingContext(recent_targets=["x", "y"])
        ctx2 = RollingContext.from_dict(ctx.to_dict())
        self.assertEqual(ctx2.recent_targets, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
