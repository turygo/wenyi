"""去翻译腔闭环测试（离线 FakeClient，不发网络请求）。"""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from trans_novel.agents.naturalizer import (
    Naturalizer,
    candidate_segments,
    naturalize_chapter,
    run_naturalize,
)
from trans_novel.config import Config
from trans_novel.ingest.models import KIND_HEADING, KIND_TEXT, Chapter, Segment
from trans_novel.llm.base import FakeClient
from trans_novel.pipeline.runstore import RunStore
from trans_novel.postprocess.punct import normalize_zh


def _config(state_dir: str = "state") -> Config:
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
            "paths": {"state_dir": state_dir},
        }
    )


def _seg(
    i: int, source: str, target: str | None, *, kind: str = KIND_TEXT, cont: bool = False
) -> Segment:
    return Segment(index=i, source=source, target=target, kind=kind, cont=cont)


def _make_store(d: str, chapters: list[Chapter]) -> RunStore:
    store = RunStore(os.path.join(d, "state"))
    manifest = {
        "title": "T",
        "fmt": "text",
        "source_path": "",
        "meta": {},
        "source_lang": "en",
        "target_lang": "zh",
        "chapters": [
            {"index": c.index, "title": c.title, "href": None, "status": "pending"}
            for c in chapters
        ],
    }
    store.save_manifest(manifest)
    for c in chapters:
        store.save_chapter(c)
    return store


def _events(store: RunStore) -> list[dict]:
    if not os.path.isfile(store.event_log_path):
        return []
    with open(store.event_log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


ORIG5 = "这是一段读起来别扭的翻译腔文字，因为存在生硬的欧化句式。"
REWRITE5 = "这段文字明显带有翻译腔，因为句式过于欧化。"
ORIG1 = "400名工程师建造了这座桥。"
REWRITE1_DROPS_NUMBER = "许多工程师建造了这座桥。"


def _combined_handler(messages, tier, json_mode):
    system = messages[0]["content"]
    user = messages[-1]["content"]
    if "书稿的母语审读编辑" in system:
        return json.dumps(
            {
                "issues": [
                    {"index": 0, "quote": "400名工程师", "reason": "数字堆叠翻译腔"},
                    {"index": 1, "quote": "读起来别扭", "reason": "欧化句式"},
                ]
            },
            ensure_ascii=False,
        )
    if "改写编辑" in system:
        if ORIG1 in user:
            return json.dumps({"rewritten": REWRITE1_DROPS_NUMBER}, ensure_ascii=False)
        return json.dumps({"rewritten": REWRITE5}, ensure_ascii=False)
    if "两个版本" in system:
        m = re.search(r"【版本 A】\n(.*?)\n\n【版本 B】\n(.*?)\n\n请判断", user, re.S)
        a = m.group(1)
        winner = "B" if a == ORIG5 else "A"
        return json.dumps({"winner": winner, "reason": "更自然"}, ensure_ascii=False)
    if "双语翻译审核员" in system:
        return json.dumps({"faithful": True, "detail": ""}, ensure_ascii=False)
    return "{}"


class TestCandidateSegments(unittest.TestCase):
    """段候选规则：heading / cont 跨段落 / 空 target / 非中文段 全部排除。"""

    def test_filters(self):
        chapter = Chapter(
            index=0,
            title="第一章",
            segments=[
                _seg(0, "Chapter One", "第一章", kind=KIND_HEADING),
                _seg(1, "A long paragraph split.", "被拆分的段落前半", cont=False),
                _seg(2, "continuation.", "续段部分", cont=True),
                _seg(3, "", None),  # 空 target
                _seg(4, "OK.", "OK"),  # 汉字占比 0
                _seg(5, "Awkward literal sentence.", ORIG5),
            ],
        )
        cands = candidate_segments(chapter)
        self.assertEqual([s.index for s in cands], [5])


class TestPairwiseAccept(unittest.TestCase):
    """正反两序判断采纳逻辑：双胜→接受；一胜一负→拒；tie→拒。"""

    def test_both_orders_win_accepts(self):
        def handler(messages, tier, json_mode):
            user = messages[-1]["content"]
            m = re.search(r"【版本 A】\n(.*?)\n\n【版本 B】\n(.*?)\n\n请判断", user, re.S)
            a = m.group(1)
            winner = "B" if a == "原译" else "A"
            return json.dumps({"winner": winner}, ensure_ascii=False)

        agent = Naturalizer(FakeClient(handler=handler), _config())
        self.assertTrue(agent.pairwise_accept("原译", "改写"))

    def test_one_win_one_loss_rejects(self):
        # 位置偏好（恒选 A 位置），忽略内容：第一序 A=原译 赢，第二序 A=改写 也赢
        # → 改写只赢第二序，第一序告负，须拒。
        def handler(messages, tier, json_mode):
            return json.dumps({"winner": "A"}, ensure_ascii=False)

        agent = Naturalizer(FakeClient(handler=handler), _config())
        self.assertFalse(agent.pairwise_accept("原译", "改写"))

    def test_tie_rejects(self):
        def handler(messages, tier, json_mode):
            return json.dumps({"winner": "tie"}, ensure_ascii=False)

        agent = Naturalizer(FakeClient(handler=handler), _config())
        self.assertFalse(agent.pairwise_accept("原译", "改写"))


class TestNaturalizeChapterFlow(unittest.TestCase):
    """完整闭环：审读→改写→关卡①拒绝(丢数字)→关卡②接受→写回，事件含 before/after。"""

    def _build(self, d: str) -> tuple[RunStore, Chapter]:
        chapter = Chapter(
            index=0,
            title="第一章",
            segments=[
                _seg(0, "Chapter One", "第一章", kind=KIND_HEADING),
                _seg(1, "400 engineers built this bridge.", ORIG1),
                _seg(5, "Awkward literal sentence.", ORIG5),
            ],
        )
        back_matter = Chapter(
            index=1,
            title="Notes",
            segments=[
                _seg(0, "Some awkward literal note text here.", ORIG5),
            ],
        )
        store = _make_store(d, [chapter, back_matter])
        return store, chapter

    def test_full_flow(self):
        with tempfile.TemporaryDirectory() as d:
            store, _ = self._build(d)
            config = _config(os.path.join(d, "state"))
            agent = Naturalizer(FakeClient(handler=_combined_handler), config)
            stats = run_naturalize(agent, store, _FakeGlossary(), config)

            self.assertEqual(stats["screened"], 2)
            self.assertEqual(stats["suspects"], 2)
            self.assertEqual(stats["rewritten"], 2)
            self.assertEqual(stats["lint_rejected"], 1)
            self.assertEqual(stats["pairwise_rejected"], 0)
            self.assertEqual(stats["applied"], 1)

            reloaded = store.load_chapter(0)
            seg1 = next(s for s in reloaded.segments if s.index == 1)
            seg5 = next(s for s in reloaded.segments if s.index == 5)
            self.assertEqual(seg1.target, ORIG1, "lint 拒绝后原译必须保留")
            self.assertEqual(seg5.target, normalize_zh(REWRITE5))
            self.assertTrue(
                reloaded.meta.get("naturalized"),
                "非 dry_run 后应由 naturalize_chapter 自行落盘 naturalized 标记，"
                "不依赖 caller 二次保存",
            )

            events = _events(store)
            applied = [e for e in events if e["event"] == "naturalize_applied"]
            rejected = [e for e in events if e["event"] == "naturalize_rejected"]
            self.assertEqual(len(applied), 1)
            self.assertEqual(applied[0]["chapter"], 0)
            self.assertEqual(applied[0]["index"], 5)
            self.assertEqual(applied[0]["before"], ORIG5)
            self.assertEqual(applied[0]["after"], normalize_zh(REWRITE5))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["gate"], "lint")
            self.assertEqual(rejected[0]["index"], 1)

            # back matter 章节完全不参与（不贡献 screened 计数、无对该章事件）
            self.assertFalse(any(e.get("chapter") == 1 for e in events))

    def test_dry_run_no_writeback(self):
        with tempfile.TemporaryDirectory() as d:
            store, _ = self._build(d)
            config = _config(os.path.join(d, "state"))
            agent = Naturalizer(FakeClient(handler=_combined_handler), config)
            stats = run_naturalize(agent, store, _FakeGlossary(), config, dry_run=True)

            self.assertEqual(stats["applied"], 1)  # 仍统计"将采纳"用于打印
            reloaded = store.load_chapter(0)
            seg5 = next(s for s in reloaded.segments if s.index == 5)
            seg1 = next(s for s in reloaded.segments if s.index == 1)
            self.assertEqual(seg5.target, ORIG5, "dry-run 不落盘")
            self.assertEqual(seg1.target, ORIG1)
            self.assertFalse(reloaded.meta.get("naturalized"), "dry_run 不置 naturalized 标记")
            events = _events(store)
            self.assertFalse(any(e["event"] == "naturalize_applied" for e in events))
            self.assertFalse(any(e["event"] == "naturalize_rejected" for e in events))
            # dry-run 仍应把待写回内容暴露给调用方打印
            self.assertEqual(len(stats["applied_entries"]), 1)
            self.assertEqual(stats["applied_entries"][0]["before"], ORIG5)

    def test_limit_caps_applied(self):
        with tempfile.TemporaryDirectory() as d:
            # 两段都会通过两道关卡：用只认内容差异的判断器，两者皆可各自双胜。
            def handler(messages, tier, json_mode):
                system = messages[0]["content"]
                user = messages[-1]["content"]
                if "书稿的母语审读编辑" in system:
                    return json.dumps(
                        {
                            "issues": [
                                {"index": 0, "quote": "别扭1", "reason": "翻译腔"},
                                {"index": 1, "quote": "别扭2", "reason": "翻译腔"},
                            ]
                        },
                        ensure_ascii=False,
                    )
                if "改写编辑" in system:
                    if "第一段翻译腔原文" in user:
                        return json.dumps({"rewritten": "第一段改写后文字"}, ensure_ascii=False)
                    return json.dumps({"rewritten": "第二段改写后文字"}, ensure_ascii=False)
                if "两个版本" in system:
                    m = re.search(r"【版本 A】\n(.*?)\n\n【版本 B】\n(.*?)\n\n请判断", user, re.S)
                    a = m.group(1)
                    winner = "B" if "原文" in a else "A"
                    return json.dumps({"winner": winner}, ensure_ascii=False)
                if "双语翻译审核员" in system:
                    return json.dumps({"faithful": True}, ensure_ascii=False)
                return "{}"

            chapter = Chapter(
                index=0,
                title="第一章",
                segments=[
                    _seg(0, "src a", "第一段翻译腔原文，读起来很别扭。"),
                    _seg(1, "src b", "第二段翻译腔原文，同样很别扭。"),
                ],
            )
            store = _make_store(d, [chapter])
            config = _config(os.path.join(d, "state"))
            agent = Naturalizer(FakeClient(handler=handler), config)
            stats = run_naturalize(agent, store, _FakeGlossary(), config, limit=1)

            self.assertEqual(stats["applied"], 1)
            events = _events(store)
            applied = [e for e in events if e["event"] == "naturalize_applied"]
            self.assertEqual(len(applied), 1)
            reloaded = store.load_chapter(0)
            seg0 = next(s for s in reloaded.segments if s.index == 0)
            seg1 = next(s for s in reloaded.segments if s.index == 1)
            # 只有一段被采纳写回，另一段保留原译
            changed = [
                s
                for s in (seg0, seg1)
                if s.target
                not in ("第一段翻译腔原文，读起来很别扭。", "第二段翻译腔原文，同样很别扭。")
            ]
            self.assertEqual(len(changed), 1)


class TestFidelityGate(unittest.TestCase):
    """关卡③忠实度判断：不通过→拒(gate=fidelity)且不再跑成对判断；解析失败/缺字段按拒处理。"""

    def _chapter(self) -> Chapter:
        return Chapter(
            index=0,
            title="第一章",
            segments=[
                _seg(0, "Only every witness saw it happen.", ORIG5),
            ],
        )

    def _handler(self, fidelity_response: str):
        def handler(messages, tier, json_mode):
            system = messages[0]["content"]
            user = messages[-1]["content"]
            if "书稿的母语审读编辑" in system:
                return json.dumps(
                    {
                        "issues": [
                            {"index": 0, "quote": "别扭", "reason": "翻译腔"},
                        ]
                    },
                    ensure_ascii=False,
                )
            if "改写编辑" in system:
                return json.dumps({"rewritten": REWRITE5}, ensure_ascii=False)
            if "双语翻译审核员" in system:
                return fidelity_response
            if "两个版本" in system:
                m = re.search(r"【版本 A】\n(.*?)\n\n【版本 B】\n(.*?)\n\n请判断", user, re.S)
                a = m.group(1)
                winner = "B" if a == ORIG5 else "A"
                return json.dumps({"winner": winner}, ensure_ascii=False)
            return "{}"

        return handler

    def test_unfaithful_rejects_and_skips_pairwise(self):
        with tempfile.TemporaryDirectory() as d:
            chapter = self._chapter()
            store = _make_store(d, [chapter])
            config = _config(os.path.join(d, "state"))
            client = FakeClient(
                handler=self._handler(
                    json.dumps(
                        {"faithful": False, "detail": "丢失了 every 的全称限定"}, ensure_ascii=False
                    )
                )
            )
            agent = Naturalizer(client, config)
            stats = naturalize_chapter(
                agent, chapter, 0, 1, [], config, store, dry_run=False, remaining=None
            )

            self.assertEqual(stats["fidelity_rejected"], 1)
            self.assertEqual(stats["applied"], 0)
            self.assertFalse(
                any("两个版本" in c["messages"][0]["content"] for c in client.calls),
                "忠实度不通过时不应发生成对判断调用",
            )

            reloaded = store.load_chapter(0)
            self.assertEqual(reloaded.segments[0].target, ORIG5, "拒绝后原译必须保留")
            self.assertTrue(
                reloaded.meta.get("naturalized"),
                "即使无改写被采纳，非 dry_run 也应置 naturalized 标记并落盘",
            )
            events = _events(store)
            rejected = [e for e in events if e["event"] == "naturalize_rejected"]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["gate"], "fidelity")

    def test_faithful_passes_to_pairwise(self):
        with tempfile.TemporaryDirectory() as d:
            chapter = self._chapter()
            store = _make_store(d, [chapter])
            config = _config(os.path.join(d, "state"))
            client = FakeClient(
                handler=self._handler(
                    json.dumps({"faithful": True, "detail": ""}, ensure_ascii=False)
                )
            )
            agent = Naturalizer(client, config)
            stats = naturalize_chapter(
                agent, chapter, 0, 1, [], config, store, dry_run=False, remaining=None
            )

            self.assertEqual(stats["fidelity_rejected"], 0)
            self.assertTrue(
                any("两个版本" in c["messages"][0]["content"] for c in client.calls),
                "忠实度通过后应发生成对判断调用",
            )
            self.assertEqual(stats["applied"], 1)

    def test_malformed_json_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            chapter = self._chapter()
            store = _make_store(d, [chapter])
            config = _config(os.path.join(d, "state"))
            client = FakeClient(handler=self._handler("不是合法 JSON"))
            agent = Naturalizer(client, config)
            stats = naturalize_chapter(
                agent, chapter, 0, 1, [], config, store, dry_run=False, remaining=None
            )

            self.assertEqual(stats["fidelity_rejected"], 1)
            self.assertEqual(stats["applied"], 0)
            self.assertTrue(store.load_chapter(0).meta.get("naturalized"))

    def test_missing_field_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            chapter = self._chapter()
            store = _make_store(d, [chapter])
            config = _config(os.path.join(d, "state"))
            client = FakeClient(
                handler=self._handler(
                    json.dumps({"detail": "缺少 faithful 字段"}, ensure_ascii=False)
                )
            )
            agent = Naturalizer(client, config)
            stats = naturalize_chapter(
                agent, chapter, 0, 1, [], config, store, dry_run=False, remaining=None
            )

            self.assertEqual(stats["fidelity_rejected"], 1)
            self.assertEqual(stats["applied"], 0)
            self.assertTrue(store.load_chapter(0).meta.get("naturalized"))


class _FakeGlossary:
    """最小 GlossaryStore 替身：all_terms() 返回空列表（无锁定术语）。"""

    def all_terms(self):
        return []


if __name__ == "__main__":
    unittest.main()
