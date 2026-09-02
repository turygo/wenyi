"""SQLite 术语库。

两张表：
- glossary：专有名词对照表（source 唯一）。冲突检测：同 source 出现不同 target 时，
  若现有条目已锁定/高置信度则保留并记入 term_conflicts，否则更新。
- term_conflicts：待裁决的译法冲突日志，供人工复核。
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# 术语类型
TYPE_PERSON = "人物"
TYPE_PLACE = "地名"
TYPE_ORG = "组织"
TYPE_TERM = "术语"
TYPE_SKILL = "招式"
TYPE_APPELLATION = "称谓"
TYPE_HONORIFIC = "敬称"
TYPE_SPEECH = "口癖"
TYPE_FIXED_EXPR = "固定表达"
TYPE_ONOMATOPOEIA = "拟声词"

_SOURCE_ONLY_TYPES = {TYPE_APPELLATION, TYPE_HONORIFIC, TYPE_SPEECH, TYPE_FIXED_EXPR}

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class GlossaryTerm:
    source: str
    target: str
    reading: str = ""
    type: str = TYPE_TERM
    gender: str = ""
    aliases: list[str] = field(default_factory=list)
    first_chapter: int | None = None
    note: str = ""
    confidence: str = "medium"
    locked: bool = False
    status: str = "ok"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> GlossaryTerm:
        return cls(
            source=row["source"],
            target=row["target"],
            reading=row["reading"] or "",
            type=row["type"] or TYPE_TERM,
            gender=row["gender"] or "",
            aliases=json.loads(row["aliases"] or "[]"),
            first_chapter=row["first_chapter"],
            note=row["note"] or "",
            confidence=row["confidence"] or "medium",
            locked=bool(row["locked"]),
            status=row["status"] or "ok",
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (
    source        TEXT PRIMARY KEY,
    target        TEXT NOT NULL,
    reading       TEXT,
    type          TEXT,
    gender        TEXT,
    aliases       TEXT,
    first_chapter INTEGER,
    note          TEXT,
    confidence    TEXT DEFAULT 'medium',
    locked        INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'ok',
    updated_at    REAL
);
CREATE TABLE IF NOT EXISTS term_conflicts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    existing_target TEXT,
    proposed_target TEXT,
    chapter         INTEGER,
    note            TEXT,
    resolved        INTEGER DEFAULT 0,
    created_at      REAL
);
"""


def _match_text(text: str) -> str:
    """Normalize width/compatibility forms and case for glossary matching."""
    return unicodedata.normalize("NFKC", text).casefold()


def _source_edge_is_word(source: str, index: int, step: int) -> bool:
    """Treat combining marks at a source edge as part of the nearest base."""
    while 0 <= index < len(source) and unicodedata.category(source[index]).startswith("M"):
        index += step
    if not 0 <= index < len(source):
        return False
    char = source[index]
    return char.isalnum() or char == "_"


def _matches_source(normalized_text: str, source: str) -> bool:
    """Match a normalized source term with script-aware word boundaries."""
    normalized_source = _match_text(source)
    if not normalized_source:
        return False

    alphabetic = [char for char in normalized_source if char.isalpha()]
    has_word_boundaries = all(ord(char) < 128 for char in normalized_source) or (
        bool(alphabetic)
        and all(
            unicodedata.name(char, "").startswith(("LATIN ", "GREEK ", "CYRILLIC "))
            for char in alphabetic
        )
    )
    if not has_word_boundaries:
        return normalized_source in normalized_text

    source_starts_word = _source_edge_is_word(normalized_source, 0, 1)
    source_ends_word = _source_edge_is_word(normalized_source, len(normalized_source) - 1, -1)
    start = normalized_text.find(normalized_source)
    while start >= 0:
        end = start + len(normalized_source)
        before_is_word = _source_edge_is_word(normalized_text, start - 1, -1)
        after_is_word = _source_edge_is_word(normalized_text, end, 1)
        before_is_boundary = not source_starts_word or not before_is_word
        after_is_boundary = not source_ends_word or not after_is_word
        if before_is_boundary and after_is_boundary:
            return True
        start = normalized_text.find(normalized_source, start + 1)


_WORD_RE = re.compile(r"[^\W\d_]+")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _person_mentioned(term, text: str, words: set[str]) -> bool:
    """锁定人物是否以「部分形式」出现在文本里（全名/别名的整体匹配由 terms_in 负责）。"""
    for name in (term.source, *(term.aliases or [])):
        parts = [p for p in _WORD_RE.findall(name) if len(p) >= 2]
        if len(parts) >= 2:
            for part in parts:
                if _HAN_RE.search(part):
                    if part in text:
                        return True
                elif part[0].isupper() and part in words:
                    return True
        elif _HAN_RE.search(name):
            for plen in (2, 3):
                if plen < len(name) and name[:plen] in text:
                    return True
    return False


def terms_matching_text(terms: list, text: str) -> list:
    """按纯文本裁剪术语表：source/alias 命中 + 锁定人物以「部分形式」出现。"""
    hit = {t.source for t in GlossaryStore.terms_in(terms, text)}
    words = set(_WORD_RE.findall(text))
    return [
        t
        for t in terms
        if t.source in hit
        or (t.type == TYPE_PERSON and t.locked and _person_mentioned(t, text, words))
    ]
    return False


class GlossaryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        # 并发写等待，避免 Web 编辑与翻译 worker 同写时报 "database is locked"
        self.conn.execute("PRAGMA busy_timeout = 5000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── 术语 ──────────────────────────────────────────────────────────────
    def get_term(self, source: str) -> GlossaryTerm | None:
        row = self.conn.execute("SELECT * FROM glossary WHERE source = ?", (source,)).fetchone()
        return GlossaryTerm.from_row(row) if row else None

    def upsert_term(self, term: GlossaryTerm, chapter: int | None = None) -> str:
        """插入或更新术语，返回 'inserted'|'updated'|'unchanged'|'conflict'。

        冲突规则：同 source 已存在且 target 不同时——
          现有条目 locked 或置信度更高 → 保留现有，记冲突，返回 'conflict'；
          否则用新条目覆盖，返回 'updated'。
        """
        try:
            # 读取 existing 前先加锁，避免两个连接同时依据旧快照作出判断。
            self.conn.execute("BEGIN IMMEDIATE")
            existing = self.get_term(term.source)
            now = time.time()
            if existing is None:
                self.conn.execute(
                    """INSERT INTO glossary
                       (source,target,reading,type,gender,aliases,first_chapter,note,
                        confidence,locked,status,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        term.source,
                        term.target,
                        term.reading,
                        term.type,
                        term.gender,
                        json.dumps(term.aliases, ensure_ascii=False),
                        term.first_chapter if term.first_chapter is not None else chapter,
                        term.note,
                        term.confidence,
                        int(term.locked),
                        term.status,
                        now,
                    ),
                )
                result = "inserted"
            elif existing.target == term.target:
                # 合并别名 / 补全字段，不算冲突
                merged_aliases = sorted(set(existing.aliases) | set(term.aliases))
                self.conn.execute(
                    """UPDATE glossary SET reading=COALESCE(NULLIF(?,''),reading),
                       gender=COALESCE(NULLIF(?,''),gender), aliases=?,
                       note=COALESCE(NULLIF(?,''),note), updated_at=? WHERE source=?""",
                    (
                        term.reading,
                        term.gender,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        term.note,
                        now,
                        term.source,
                    ),
                )
                result = "unchanged"
            else:
                # target 不同 → 冲突判定
                existing_priority = (existing.locked, CONFIDENCE_ORDER.get(existing.confidence, 1))
                new_priority = (term.locked, CONFIDENCE_ORDER.get(term.confidence, 1))
                self._log_conflict(term.source, existing.target, term.target, chapter)
                if existing_priority >= new_priority:
                    self.conn.execute(
                        "UPDATE glossary SET status='conflict', updated_at=? WHERE source=?",
                        (now, term.source),
                    )
                    result = "conflict"
                else:
                    self.conn.execute(
                        """UPDATE glossary SET target=?, reading=COALESCE(NULLIF(?,''),reading),
                           gender=COALESCE(NULLIF(?,''),gender), confidence=?, status='conflict',
                           updated_at=? WHERE source=?""",
                        (term.target, term.reading, term.gender, term.confidence, now, term.source),
                    )
                    result = "updated"
            self.conn.commit()
            return result
        except Exception:
            self.conn.rollback()
            raise

    def _log_conflict(self, source, existing_target, proposed_target, chapter):
        self.conn.execute(
            """INSERT INTO term_conflicts
               (source,existing_target,proposed_target,chapter,created_at)
               VALUES (?,?,?,?,?)""",
            (source, existing_target, proposed_target, chapter, time.time()),
        )

    def delete_term(self, source: str) -> bool:
        """删除一个术语条目（前端编辑用）。返回是否确有删除。"""
        cur = self.conn.execute("DELETE FROM glossary WHERE source = ?", (source,))
        self.conn.commit()
        return cur.rowcount > 0

    def lock_term(self, source: str, target: str | None = None) -> None:
        if target is not None:
            self.conn.execute(
                "UPDATE glossary SET target=?, locked=1, confidence='high', status='ok' WHERE source=?",
                (target, source),
            )
        else:
            self.conn.execute(
                "UPDATE glossary SET locked=1, confidence='high', status='ok' WHERE source=?",
                (source,),
            )
        self.conn.commit()

    def confirm_locked(self, source: str, target: str) -> bool:
        """namer 一次性定名确认沿用某已有译法时调用：把该条目升级为 locked+高置信度。

        seed_glossary 先种入的角色（medium/未锁）光靠 upsert_term 的同译法分支升不了
        locked（该分支只合并别名/补字段，不动 locked/confidence），term_miss 硬校验因此
        形同虚设。仅当当前 target 与确认值完全一致才生效，防止把错误译法锁死。
        返回是否执行了升级（未命中/已是最高状态时返回 False，避免多余 UPDATE）。
        """
        existing = self.get_term(source)
        if existing is None or existing.target != target:
            return False
        if existing.locked and existing.confidence == "high":
            return False
        self.conn.execute(
            "UPDATE glossary SET locked=1, confidence='high', status='ok', updated_at=? WHERE source=?",
            (time.time(), source),
        )
        self.conn.commit()
        return True

    def all_terms(self) -> list[GlossaryTerm]:
        rows = self.conn.execute("SELECT * FROM glossary ORDER BY type, source").fetchall()
        return [GlossaryTerm.from_row(r) for r in rows]

    @staticmethod
    def terms_in(terms: list[GlossaryTerm], text: str) -> list[GlossaryTerm]:
        """从给定术语列表里筛出 source 或任一别名在 text 中出现的项。

        与 terms_in_text 同义，但接受预取的术语快照，避免逐批重复查库（章内术语表不变）。
        """
        out: list[GlossaryTerm] = []
        normalized_text = _match_text(text)
        for term in terms:
            # 称谓/口癖/固定表达是带语气或场景的派生写法，不能因为 alias
            # 命中裸名就把派生译法注入到普通称呼处。
            keys = (
                [term.source] if term.type in _SOURCE_ONLY_TYPES else [term.source, *term.aliases]
            )
            if any(k and _matches_source(normalized_text, k) for k in keys):
                out.append(term)
        return out

    def terms_in_text(self, text: str) -> list[GlossaryTerm]:
        """返回 source 或任一别名在 text 中出现的术语（注入翻译 prompt 用）。"""
        return self.terms_in(self.all_terms(), text)

    def mark_conflicts_resolved(self, source: str) -> None:
        self.conn.execute("UPDATE term_conflicts SET resolved=1 WHERE source=?", (source,))
        self.conn.commit()

    def open_conflicts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM term_conflicts WHERE resolved=0 ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def low_confidence_terms(self) -> list[GlossaryTerm]:
        rows = self.conn.execute(
            "SELECT * FROM glossary WHERE confidence='low' OR status='conflict' ORDER BY source"
        ).fetchall()
        return [GlossaryTerm.from_row(r) for r in rows]

    def stats(self) -> dict[str, int]:
        g = self.conn.execute("SELECT COUNT(*) FROM glossary").fetchone()[0]
        c = self.conn.execute("SELECT COUNT(*) FROM term_conflicts WHERE resolved=0").fetchone()[0]
        return {"terms": g, "open_conflicts": c}
