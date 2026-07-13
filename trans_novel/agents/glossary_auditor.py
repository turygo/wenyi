"""术语 AI 审计与统一（收尾自动 pass）。

目标：消除同一专名的译法漂移（如 "Kaho" 在正文里 佳穂/佳穗 混用）。

流程：
1. 候选侦测：取术语表 + 已记录的译法冲突 + 在各章译文里扫描"与术语译法形近(汉字编辑距离 1)"的变体；
2. 强档模型裁定每个原文词的【规范译法 canonical】与应被替换的变体；
3. 落地：锁定术语表 canonical、变体并入别名、标记冲突已解决；
   并**改写各章已译正文**（变体→canonical），同步翻译记忆库。
"""

from __future__ import annotations

import re
from typing import Any

from ..glossary.store import TYPE_PERSON, GlossaryStore, GlossaryTerm
from ..pipeline.runstore import RunStore
from . import prompts
from .base import Agent


def _is_cjk(s: str) -> bool:
    return bool(s) and all("一" <= c <= "鿿" for c in s)


def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


_LATIN_SOURCE_RE = re.compile(r"[A-Za-z][A-Za-z .\-]*")


def _is_latin_source(s: str) -> bool:
    """source 是否为拉丁人名/术语串（ASCII 字母，可含空格/./-）。"""
    return bool(s) and _LATIN_SOURCE_RE.fullmatch(s) is not None


_CJK_SPACE_GAP_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")


def _hamming1_variants(target: str, corpus: str) -> set[str]:
    """在 corpus 中找与 target 等长、仅差 1 个汉字、且确为汉字串的形近变体。"""
    L = len(target)
    if L < 3 or not _is_cjk(target):  # 防线2: 2字串组合爆炸(如 利亚→东亚/南亚…)，直接不扫
        return set()
    found: set[str] = set()
    seen: set[str] = set()
    for i in range(len(corpus) - L + 1):
        w = corpus[i : i + L]
        if w in seen or w == target:
            continue
        seen.add(w)
        if not _is_cjk(w):
            continue
        if sum(1 for a, b in zip(w, target) if a != b) == 1:
            found.add(w)
    return found


class GlossaryAuditor(Agent):
    # ── 候选侦测 ────────────────────────────────────────────────────────────
    def _candidates(self, store: RunStore, glossary: GlossaryStore) -> dict[str, dict[str, Any]]:
        terms = glossary.all_terms()
        corpus = self._target_corpus(store)
        cand: dict[str, dict[str, Any]] = {}
        for t in terms:
            variants: set[str] = set()
            # 防线1: 形近变体扫描仅对人名启用（领域名词的 hamming 邻居多为真实词，噪声源）
            if t.type == TYPE_PERSON:
                variants |= _hamming1_variants(t.target, corpus)
            if len(variants) > 8:
                # 防线3: 海量变体=模式噪声签名（真实漂移通常 1-3 个），整体丢弃该术语候选
                continue
            if variants:
                cand[t.source] = {
                    "source": t.source,
                    "current": t.target,
                    "type": t.type,
                    "variants": sorted(variants),
                }
        # 已记录的译法冲突也并入候选
        for c in glossary.open_conflicts():
            src = c["source"]
            entry = cand.setdefault(
                src,
                {
                    "source": src,
                    "current": c.get("existing_target", ""),
                    "type": "",
                    "variants": [],
                },
            )
            for v in (c.get("existing_target"), c.get("proposed_target")):
                if v and v != entry["current"] and v not in entry["variants"]:
                    entry["variants"].append(v)
        return cand

    @staticmethod
    def _target_corpus(store: RunStore) -> str:
        m = store.load_manifest()
        parts: list[str] = []
        for c in m["chapters"]:
            ch = store.load_chapter(c["index"])
            parts.extend(s.target or "" for s in ch.text_segments)
        return "\n".join(parts)

    # ── 模型裁定 ────────────────────────────────────────────────────────────
    def _decide(self, candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        lines = []
        for c in candidates.values():
            allv = [c["current"]] + [v for v in c["variants"] if v != c["current"]]
            lines.append(
                f"- {c['source']}（{c['type'] or '?'}）: 现有译法/变体 = {', '.join(allv)}"
            )
        user = (
            "下列原文词在术语表或正文里出现了多种译法/形近变体，请为每个裁定唯一规范译法：\n"
            + "\n".join(lines)
            + '\n\n输出 JSON：{"unifications":[{"source":"...","canonical":"...","variants":["..."],"reason":"..."}]}'
        )
        system = prompts.render("glossary_audit_system", src=self.src, tgt=self.tgt)
        uni = self._ask_json(
            system, user, tier="strong", key="unifications", default=[], operation="glossary.audit"
        )
        result: list[dict[str, Any]] = []
        for u in self.dict_items(uni):
            if not u.get("source") or not u.get("canonical"):
                continue
            # 防线4: LLM 返回的变体必须 ⊆ 裁定时刻提交的候选集合，超集部分（幻觉发明）静默丢弃
            allowed = set(candidates.get(str(u["source"]), {}).get("variants", []))
            u = dict(u)
            u["variants"] = [v for v in u.get("variants", []) if str(v).strip() in allowed]
            result.append(u)
        return result

    # ── 落地 ────────────────────────────────────────────────────────────────
    def audit(self, store: RunStore, glossary: GlossaryStore) -> list[dict[str, Any]]:
        """返回已应用的统一记录列表。"""
        candidates = self._candidates(store, glossary)
        unifications = self._decide(candidates)
        applied: list[dict[str, Any]] = []
        # 收集 变体→规范 的全局替换表（仅汉字变体，避免误伤）
        replace_map: dict[str, str] = {}
        for u in unifications:
            src = str(u["source"])
            canonical = str(u["canonical"]).strip()
            variants = [str(v).strip() for v in u.get("variants", []) if str(v).strip()]
            variants = [v for v in variants if v and v != canonical]
            if not canonical:
                continue
            # 术语表：锁定 canonical，变体并入别名，标记冲突已解决
            glossary.lock_term(src, canonical)
            if variants:
                glossary.upsert_term(
                    GlossaryTerm(
                        source=src,
                        target=canonical,
                        aliases=variants,
                        confidence="high",
                        locked=True,
                    ),
                )
            glossary.mark_conflicts_resolved(src)
            for v in variants:
                if _is_cjk(v):  # 仅对汉字变体做正文替换，安全
                    replace_map[v] = canonical
            applied.append(
                {
                    "source": src,
                    "canonical": canonical,
                    "variants": variants,
                    "reason": u.get("reason", ""),
                }
            )

        if replace_map:
            self._rewrite_targets(store, glossary, replace_map)

        applied.extend(self._fix_latin_residue(store, glossary))
        return applied

    @staticmethod
    def _rewrite_targets(
        store: RunStore, glossary: GlossaryStore, replace_map: dict[str, str]
    ) -> int:
        """把各章 target 里的变体替换为规范译法，并同步 TM。返回改动段数。"""
        # 长变体优先替换，避免短串先替导致嵌套问题
        variants_sorted = sorted(replace_map, key=len, reverse=True)

        def _apply(text: str) -> str:
            if not text:
                return text
            for v in variants_sorted:
                if v in text:
                    text = text.replace(v, replace_map[v])
            return text

        m = store.load_manifest()
        changed = 0
        for c in m["chapters"]:
            ch = store.load_chapter(c["index"])
            dirty = False
            for idx, seg in enumerate(ch.segments):
                if not seg.target:
                    continue
                new = _apply(seg.target)
                if new != seg.target:
                    old = seg.target
                    seg.target = new
                    dirty = True
                    changed += 1
                    glossary.add_tm(seg.source, new, c["index"])
                    store.log_event(
                        "glossary_rewrite_applied",
                        chapter=c["index"],
                        index=idx,
                        source=seg.source,
                        before=old,
                        after=new,
                        replace_map=replace_map,
                    )
            if dirty:
                store.save_chapter(ch)

        # 同步改写已译的章节标题，保持目录一致；书名保持原文，清理旧译名字段。
        man_dirty = False
        if "title_translated" in m:
            old_title = m.pop("title_translated")
            man_dirty = True
            store.log_event(
                "glossary_book_title_translation_removed",
                title=True,
                before=old_title,
                replace_map=replace_map,
            )
        for c in m["chapters"]:
            old_title = c.get("title_translated")
            ct = _apply(old_title) if isinstance(old_title, str) else old_title
            if ct != old_title:
                c["title_translated"] = ct
                man_dirty = True
                store.log_event(
                    "glossary_title_rewrite_applied",
                    chapter=c["index"],
                    before=old_title,
                    after=ct,
                    replace_map=replace_map,
                )
        if man_dirty:
            store.save_manifest(m)
        return changed

    @staticmethod
    def _fix_latin_residue(store: RunStore, glossary: GlossaryStore) -> list[dict[str, Any]]:
        """确定性修复：锁定术语的拉丁 source 残留在已译正文里的段落，逐段条件替换。

        零 LLM，与 _decide 的候选/裁定流程无关，即使模型未产生任何统一建议也执行。
        替换条件（三者同时满足才动该段，且第 4 条逐命中判断）：
        a. 段 target 含 CJK 汉字（英文直通段——如 back_matter skip——天然不含 CJK，自动跳过）；
        b. `\\b<source>\\b` 命中该段 target（词边界，避免 "Li" 误伤 "Liya"）；
        c. 段内尚未出现该术语的 target 译名（"利亚(Liya)" 这类括注格式视为已译，跳过）；
        d. 防线5——命中点前后各 12 个字符内至少一侧含 CJK 才替换该次命中；两侧全是拉丁/
           标点（人名嵌在整句英文引文里，如脚注引题）则跳过这次命中，逐命中而非逐段判断。
        """
        terms = [t for t in glossary.all_terms() if t.locked and _is_latin_source(t.source)]
        if not terms:
            return []
        applied: list[dict[str, Any]] = []
        m = store.load_manifest()
        for t in terms:
            pattern = re.compile(r"\b" + re.escape(t.source) + r"\b")
            touched = False
            for c in m["chapters"]:
                ch = store.load_chapter(c["index"])
                dirty = False
                for idx, seg in enumerate(ch.segments):
                    text = seg.target
                    if not text or not _has_cjk(text) or t.target in text:
                        continue
                    matches = list(pattern.finditer(text))
                    if not matches:
                        continue
                    parts: list[str] = []
                    last = 0
                    replaced = False
                    for mo in matches:
                        left = text[max(0, mo.start() - 12) : mo.start()]
                        right = text[mo.end() : mo.end() + 12]
                        if not (_has_cjk(left) or _has_cjk(right)):
                            continue
                        parts.append(text[last : mo.start()])
                        parts.append(t.target)
                        last = mo.end()
                        replaced = True
                    if not replaced:
                        continue
                    parts.append(text[last:])
                    sub = "".join(parts)
                    new = _CJK_SPACE_GAP_RE.sub("", sub)
                    old = text
                    seg.target = new
                    dirty = True
                    touched = True
                    glossary.add_tm(seg.source, new, c["index"])
                    store.log_event(
                        "glossary_latin_residue_fixed",
                        chapter=c["index"],
                        index=idx,
                        source=seg.source,
                        before=old,
                        after=new,
                        term_source=t.source,
                        term_target=t.target,
                    )
                if dirty:
                    store.save_chapter(ch)
            if touched:
                applied.append(
                    {
                        "source": t.source,
                        "canonical": t.target,
                        "variants": [t.source],
                        "reason": "锁定术语拉丁残留替换",
                    }
                )
        return applied
