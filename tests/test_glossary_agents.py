"""分析器 / 术语抽取 / 滚动上下文 的测试（离线）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from trans_novel.agents.analyzer import Analyzer
from trans_novel.agents.glossary_auditor import GlossaryAuditor
from trans_novel.config import Config
from trans_novel.glossary.extractor import GlossaryExtractor
from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore, GlossaryTerm
from trans_novel.ingest.models import Chapter, Segment
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.context import RollingContext
from trans_novel.pipeline.runstore import RunStore


def _cfg():
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
        }
    )


class TestAnalyzer(unittest.TestCase):
    def test_analyze_and_seed(self):
        analysis = {
            "genre": "校园",
            "tone": "冷峻第三人称",
            "style_guide": "保持克制",
            "characters": [
                {
                    "source": "綾小路",
                    "target": "绫小路",
                    "gender": "男",
                    "reading": "あやのこうじ",
                    "note": "第一人称用俺",
                }
            ],
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
        terms = {
            "terms": [
                {
                    "source": "堀北",
                    "target": "堀北",
                    "type": "人物",
                    "gender": "女",
                    "aliases": ["堀北さん"],
                },
                {"source": "屋上", "target": "天台", "type": "地名", "gender": "未知"},
            ]
        }
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
                GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON, confidence="high"),
                chapter=1,
            )
            store.upsert_term(
                GlossaryTerm(source="屋上", target="天台", type="地名", confidence="low"), chapter=1
            )
            terms = [
                GlossaryTerm(source="龙园", target="龙园", type=TYPE_PERSON),  # inserted
                GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON),  # unchanged
                GlossaryTerm(
                    source="屋上", target="屋顶", type="地名", confidence="high"
                ),  # updated（新胜出）
                GlossaryTerm(
                    source="堀北", target="北堀", type=TYPE_PERSON, confidence="low"
                ),  # conflict（现有胜出）
            ]
            summary, changed = ext.store_terms(store, terms, chapter=2)
            self.assertEqual(summary, {"inserted": 1, "updated": 1, "conflict": 1, "unchanged": 1})
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
            store.upsert_term(
                GlossaryTerm(source="堀北", target="堀北", type=TYPE_PERSON), chapter=1
            )
            store.upsert_term(
                GlossaryTerm(source="龙园", target="龙园", type=TYPE_PERSON, locked=True), chapter=1
            )
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


class TestLatinResidueFix(unittest.TestCase):
    """确定性拉丁残留修复 pass（GlossaryAuditor._fix_latin_residue / audit）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(os.path.join(self.tmp.name, "run"))
        self.store.save_manifest({"chapters": [{"index": 0}]})
        segments = [
            Segment(index=0, source="To Liya.", target="致 Liya"),
            Segment(index=1, source="Liya (again).", target="利亚(Liya)"),
            Segment(index=2, source="Dear Liya,", target="Dear Liya,"),
            Segment(index=3, source="He is Mark.", target="他是Mark。"),
        ]
        self.store.save_chapter(Chapter(index=0, segments=segments))
        self.glossary = GlossaryStore(self.store.glossary_path)
        # 已锁定的拉丁人名术语：应被修复
        self.glossary.upsert_term(
            GlossaryTerm(
                source="Liya", target="利亚", type=TYPE_PERSON, confidence="high", locked=True
            ),
        )
        # 未锁定术语：即使正文含 Mark 也不得替换
        self.glossary.upsert_term(GlossaryTerm(source="Mark", target="马克", confidence="medium"))

    def tearDown(self):
        self.glossary.close()
        self.tmp.cleanup()

    def _client(self):
        return FakeClient(handler=lambda messages, tier, json_mode: "{}")

    def test_fixes_locked_latin_residue_and_squeezes_space(self):
        applied = GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        rec = next(a for a in applied if a["source"] == "Liya")
        self.assertEqual(rec["canonical"], "利亚")
        self.assertEqual(rec["variants"], ["Liya"])
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[0].target, "致利亚")

    def test_bracketed_gloss_left_untouched(self):
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[1].target, "利亚(Liya)")

    def test_pure_latin_segment_without_cjk_untouched(self):
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[2].target, "Dear Liya,")

    def test_unlocked_term_not_replaced(self):
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[3].target, "他是Mark。")

    def test_word_boundary_does_not_match_inside_longer_name(self):
        self.glossary.upsert_term(
            GlossaryTerm(source="Li", target="李", confidence="high", locked=True),
        )
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        # "Liya" 不应被 source="Li" 的术语误伤
        self.assertEqual(ch.segments[0].target, "致利亚")

    def test_idempotent_rerun_yields_no_further_changes(self):
        auditor = GlossaryAuditor(self._client(), _cfg())
        auditor.audit(self.store, self.glossary)
        applied2 = auditor.audit(self.store, self.glossary)
        latin_fixes = [a for a in applied2 if a["source"] == "Liya"]
        self.assertEqual(latin_fixes, [])
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[0].target, "致利亚")

    def test_tm_synced_for_fixed_segment(self):
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        self.assertEqual(self.glossary.tm_lookup("To Liya."), "致利亚")


class TestGlossaryAuditGuards(unittest.TestCase):
    """五道确定性防线单测（2026-07-11 事故后新增）：
    防线1/2/3 = _candidates 的候选收紧；防线4 = _decide 的裁定过滤；
    防线5 = _fix_latin_residue 的逐命中 CJK 近邻门控。
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = RunStore(os.path.join(self.tmp.name, "run"))
        self.glossary = GlossaryStore(self.store.glossary_path)

    def tearDown(self):
        self.glossary.close()
        self.tmp.cleanup()

    def _seed_chapter(self, targets):
        self.store.save_manifest({"chapters": [{"index": 0}]})
        segments = [Segment(index=i, source=f"s{i}", target=t) for i, t in enumerate(targets)]
        self.store.save_chapter(Chapter(index=0, segments=segments))

    def _client(self, handler=None):
        return FakeClient(handler=handler or (lambda m, t, j: "{}"))

    def test_two_char_target_no_hamming_candidates(self):
        """防线2：2字译名（len<3）直接返回空，不产生 hamming 候选。"""
        self.glossary.upsert_term(GlossaryTerm(source="Liya", target="利亚", type=TYPE_PERSON))
        self._seed_chapter(
            ["利亚在东京。", "东亚经济增长。", "南亚气候炎热。", "利用工具。", "利益相关。"]
        )
        cand = GlossaryAuditor(self._client(), _cfg())._candidates(self.store, self.glossary)
        self.assertNotIn("Liya", cand)

    def test_non_person_term_no_hamming_candidates(self):
        """防线1：非 TYPE_PERSON 术语不扫描 hamming，即使正文里有形近词。"""
        self.glossary.upsert_term(
            GlossaryTerm(source="supply chain", target="供应链条", type="术语")
        )
        self._seed_chapter(["供应链条很重要。", "供应连条断裂了。"])
        cand = GlossaryAuditor(self._client(), _cfg())._candidates(self.store, self.glossary)
        self.assertNotIn("supply chain", cand)

    def test_person_three_char_name_variant_still_candidate(self):
        """回归保护：TYPE_PERSON 3字名的 1 个形近变体仍走原流程产出候选。"""
        self.glossary.upsert_term(GlossaryTerm(source="Kaho", target="佳穂子", type=TYPE_PERSON))
        self._seed_chapter(["佳穂子和佳穗子在一起。"])
        cand = GlossaryAuditor(self._client(), _cfg())._candidates(self.store, self.glossary)
        self.assertIn("Kaho", cand)
        self.assertEqual(cand["Kaho"]["variants"], ["佳穗子"])

    def test_excess_variants_term_discarded(self):
        """防线3：单术语变体数 >8（模式噪声签名）时整体丢弃该术语候选。"""
        self.glossary.upsert_term(GlossaryTerm(source="Kaho", target="佳穂子", type=TYPE_PERSON))
        variants = [f"佳{c}子" for c in "穗和平安宁静祥瑞康"]  # 9 个形近变体
        self.assertEqual(len(variants), 9)
        self._seed_chapter(["".join(variants)])
        cand = GlossaryAuditor(self._client(), _cfg())._candidates(self.store, self.glossary)
        self.assertNotIn("Kaho", cand)

    def test_decide_drops_variants_outside_candidates(self):
        """防线4：LLM 返回的变体必须 ⊆ 提交候选集合，超集部分静默丢弃，不进 replace_map/别名。"""
        self.glossary.upsert_term(GlossaryTerm(source="Kaho", target="佳穂子", type=TYPE_PERSON))
        self._seed_chapter(["佳穂子和佳穗子在一起。"])

        def handler(messages, tier, json_mode):
            return json.dumps(
                {
                    "unifications": [
                        {
                            "source": "Kaho",
                            "canonical": "佳穂子",
                            "variants": ["佳穗子", "幻觉变体"],
                            "reason": "统一",
                        },
                    ]
                },
                ensure_ascii=False,
            )

        auditor = GlossaryAuditor(self._client(handler), _cfg())
        applied = auditor.audit(self.store, self.glossary)
        rec = next(a for a in applied if a["source"] == "Kaho")
        self.assertEqual(rec["variants"], ["佳穗子"])
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[0].target, "佳穂子和佳穂子在一起。")
        term = self.glossary.get_term("Kaho")
        self.assertNotIn("幻觉变体", term.aliases)

    def test_latin_residue_skips_hit_embedded_in_all_latin_quote(self):
        """防线5：命中点前后各12字符内两侧全拉丁/标点（英文引文里的人名）时跳过该次命中。"""
        self.glossary.upsert_term(
            GlossaryTerm(
                source="Samsung", target="三星", type=TYPE_PERSON, confidence="high", locked=True
            ),
        )
        self._seed_chapter(["供应35%的市场：Ken Koyanagi, \u201cSamsung Deal\u2026\u201d"])
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        self.assertEqual(
            ch.segments[0].target,
            "供应35%的市场：Ken Koyanagi, \u201cSamsung Deal\u2026\u201d",
        )

    def test_latin_residue_replaces_hit_near_cjk(self):
        """防线5 正向用例：命中点近邻有 CJK 时仍替换（不误伤真实场景）。"""
        self.glossary.upsert_term(
            GlossaryTerm(
                source="Samsung", target="三星", type=TYPE_PERSON, confidence="high", locked=True
            ),
        )
        self._seed_chapter(["他说 Samsung 是巨头。"])
        GlossaryAuditor(self._client(), _cfg()).audit(self.store, self.glossary)
        ch = self.store.load_chapter(0)
        self.assertEqual(ch.segments[0].target, "他说三星是巨头。")


if __name__ == "__main__":
    unittest.main()
