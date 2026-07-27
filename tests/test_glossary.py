"""术语库测试。"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from trans_novel.glossary.resolver import resolve
from trans_novel.glossary.store import (
    TYPE_APPELLATION,
    TYPE_PERSON,
    GlossaryStore,
    GlossaryTerm,
)


class TestGlossary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = GlossaryStore(os.path.join(self.tmp.name, "g.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_insert_and_lookup(self):
        r = self.store.upsert_term(
            GlossaryTerm(
                source="綾小路",
                target="绫小路",
                type=TYPE_PERSON,
                gender="男",
                aliases=["綾小路くん"],
                reading="あやのこうじ",
            ),
            chapter=0,
        )
        self.assertEqual(r, "inserted")
        t = self.store.get_term("綾小路")
        assert t is not None
        self.assertEqual(t.target, "绫小路")
        self.assertEqual(t.gender, "男")

    def test_terms_in_text_matches_alias(self):
        self.store.upsert_term(
            GlossaryTerm(source="綾小路", target="绫小路", aliases=["綾小路くん"])
        )
        hits = self.store.terms_in_text("「おはよう、綾小路くん」と堀北が言った。")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "綾小路")

    def test_terms_in_text_normalizes_case_and_character_width(self):
        self.store.upsert_term(GlossaryTerm(source="OpenAI", target="开放人工智能"))
        self.store.upsert_term(GlossaryTerm(source="ＡＢＣ", target="ABC 组织"))

        hits = self.store.terms_in_text("openai 与 ABC")

        self.assertEqual(
            {term.source for term in hits},
            {"OpenAI", "ＡＢＣ"},
        )

    def test_terms_in_text_matches_ascii_term_at_word_boundary(self):
        self.store.upsert_term(GlossaryTerm(source="Ann", target="安"))

        self.assertEqual(
            [term.source for term in self.store.terms_in_text("ann opened.")],
            ["Ann"],
        )
        self.assertEqual(
            [term.source for term in self.store.terms_in_text("(ANN), she said.")],
            ["Ann"],
        )
        self.assertEqual(self.store.terms_in_text("Anna opened."), [])

    def test_terms_in_text_respects_punctuation_at_source_edges(self):
        self.store.upsert_term(GlossaryTerm(source="Mr.", target="先生"))
        self.store.upsert_term(GlossaryTerm(source="@Ann", target="@安"))

        self.assertEqual(
            [term.source for term in self.store.terms_in_text("Mr.Smith arrived.")],
            ["Mr."],
        )
        self.assertEqual(
            [term.source for term in self.store.terms_in_text("foo@Ann")],
            ["@Ann"],
        )

    def test_terms_in_text_handles_combining_mark_at_source_edge(self):
        self.store.upsert_term(GlossaryTerm(source="İ", target="İ"))
        self.store.upsert_term(GlossaryTerm(source="i", target="i"))

        self.assertEqual(self.store.terms_in_text("İstanbul"), [])
        self.assertEqual(
            {term.source for term in self.store.terms_in_text("İ arrived.")},
            {"İ", "i"},
        )

    def test_terms_in_text_does_not_match_inside_cyrillic_word(self):
        self.store.upsert_term(GlossaryTerm(source="гад", target="混蛋"))

        self.assertEqual(
            [term.source for term in self.store.terms_in_text("Этот гад пришёл.")],
            ["гад"],
        )
        self.assertEqual(self.store.terms_in_text("гадкий человек"), [])

    def test_terms_in_text_keeps_cjk_substring_matching(self):
        self.store.upsert_term(GlossaryTerm(source="東京", target="东京"))

        self.assertEqual(
            [term.source for term in self.store.terms_in_text("東京都へ行く。")],
            ["東京"],
        )

    def test_terms_in_text_applies_boundaries_to_aliases(self):
        self.store.upsert_term(GlossaryTerm(source="Elizabeth", target="伊丽莎白", aliases=["Liz"]))

        self.assertEqual(
            [term.source for term in self.store.terms_in_text("Liz arrived.")],
            ["Elizabeth"],
        )
        self.assertEqual(self.store.terms_in_text("Blitz arrived."), [])

    def test_appellation_does_not_match_bare_name_alias(self):
        self.store.upsert_term(
            GlossaryTerm(
                source="夏帆ちゃん",
                target="小夏帆",
                type=TYPE_APPELLATION,
                aliases=["夏帆"],
            )
        )
        self.assertEqual(self.store.terms_in_text("夏帆は窓の外を見た。"), [])
        hits = self.store.terms_in_text("「夏帆ちゃん」と母親が言った。")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "夏帆ちゃん")

    def test_conflict_keeps_locked(self):
        self.store.upsert_term(
            GlossaryTerm(source="堀北", target="堀北", confidence="high"), chapter=0
        )
        self.store.lock_term("堀北")
        # 提出不同译法 → 应保留锁定译法并记冲突
        r = self.store.upsert_term(
            GlossaryTerm(source="堀北", target="掘北", confidence="medium"), chapter=1
        )
        self.assertEqual(r, "conflict")
        term = self.store.get_term("堀北")
        assert term is not None
        self.assertEqual(term.target, "堀北")
        self.assertEqual(len(self.store.open_conflicts()), 1)

    def test_concurrent_upserts_make_one_atomic_conflict_decision(self):
        path = os.path.join(self.tmp.name, "concurrent.db")
        initial = GlossaryStore(path)
        initial.close()
        barrier = threading.Barrier(2)

        def write(target: str) -> str:
            store = GlossaryStore(path)
            try:
                barrier.wait()
                return store.upsert_term(GlossaryTerm(source="Name", target=target), chapter=1)
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(write, ["译名甲", "译名乙"]))

        check = GlossaryStore(path)
        try:
            self.assertCountEqual(results, ["inserted", "conflict"])
            self.assertEqual(len(check.all_terms()), 1)
            self.assertEqual(len(check.open_conflicts()), 1)
        finally:
            check.close()

    def test_conflict_overrides_low_confidence(self):
        self.store.upsert_term(GlossaryTerm(source="X", target="旧译", confidence="low"), chapter=0)
        r = self.store.upsert_term(
            GlossaryTerm(source="X", target="新译", confidence="high"), chapter=1
        )
        self.assertEqual(r, "updated")
        term = self.store.get_term("X")
        assert term is not None
        self.assertEqual(term.target, "新译")

    def test_stats(self):
        self.store.upsert_term(GlossaryTerm(source="A", target="甲"))
        s = self.store.stats()
        self.assertEqual(s["terms"], 1)
        self.assertEqual(s["open_conflicts"], 0)
        self.assertEqual(set(s), {"terms", "open_conflicts"})

    def test_fresh_schema_omits_translation_memory(self):
        tables = {
            row[0]
            for row in self.store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        self.assertNotIn("translation_memory", tables)

    def test_legacy_translation_memory_is_preserved(self):
        path = os.path.join(self.tmp.name, "legacy.db")
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE translation_memory (
                source_hash TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                target_text TEXT NOT NULL,
                chapter INTEGER,
                updated_at REAL
            )"""
        )
        conn.execute(
            "INSERT INTO translation_memory "
            "(source_hash, source_text, target_text, chapter, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-hash", "風が強かった。", "风很大。", 1, 123.0),
        )
        conn.commit()
        conn.close()

        store = GlossaryStore(path)
        try:
            self.assertIsNotNone(
                store.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='translation_memory'"
                ).fetchone()
            )
            row = store.conn.execute(
                "SELECT source_hash, source_text, target_text, chapter, updated_at "
                "FROM translation_memory"
            ).fetchone()
            self.assertEqual(
                tuple(row),
                ("legacy-hash", "風が強かった。", "风很大。", 1, 123.0),
            )
            self.assertEqual(
                store.upsert_term(GlossaryTerm(source="A", target="甲")),
                "inserted",
            )
            self.assertIsNotNone(store.get_term("A"))
            preserved = store.conn.execute(
                "SELECT source_hash, source_text, target_text, chapter, updated_at "
                "FROM translation_memory"
            ).fetchone()
            self.assertEqual(tuple(preserved), tuple(row))
        finally:
            store.close()


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = GlossaryStore(os.path.join(self.tmp.name, "g.db"))

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_resolve_creates_and_locks_missing_term(self):
        self.assertIsNone(self.store.get_term("Liya"))
        resolve(self.store, "Liya", "利亚")
        t = self.store.get_term("Liya")
        assert t is not None
        self.assertTrue(t.locked)
        self.assertEqual(t.target, "利亚")
        self.assertEqual(t.confidence, "high")

    def test_resolve_overwrites_and_locks_existing_term(self):
        self.store.upsert_term(GlossaryTerm(source="Liya", target="莉雅", confidence="low"))
        resolve(self.store, "Liya", "利亚")
        t = self.store.get_term("Liya")
        assert t is not None
        self.assertTrue(t.locked)
        self.assertEqual(t.target, "利亚")

    def test_resolve_clears_conflict_flags(self):
        self.store.upsert_term(
            GlossaryTerm(source="Liya", target="莉雅", locked=True, confidence="high")
        )
        self.store.upsert_term(GlossaryTerm(source="Liya", target="丽雅"))  # 触发冲突记录
        self.assertEqual(len(self.store.open_conflicts()), 1)
        resolve(self.store, "Liya", "利亚")
        self.assertEqual(len(self.store.open_conflicts()), 0)


if __name__ == "__main__":
    unittest.main()
