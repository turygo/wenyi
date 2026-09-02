"""Pure glossary candidate and unification policy."""

from __future__ import annotations

import re
from typing import Any

from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore, GlossaryTerm


def _is_cjk(value: str) -> bool:
    return bool(value) and all("一" <= character <= "鿿" for character in value)


def has_cjk(value: str) -> bool:
    return any("一" <= character <= "鿿" for character in value)


_LATIN_SOURCE_RE = re.compile(r"[A-Za-z][A-Za-z .\-]*")


def is_latin_source(value: str) -> bool:
    """source 是否为拉丁人名/术语串（ASCII 字母，可含空格/./-）。"""
    return bool(value) and _LATIN_SOURCE_RE.fullmatch(value) is not None


CJK_SPACE_GAP_RE = re.compile(r"(?<=[\u4e00-\u9fff])[ \t]+(?=[\u4e00-\u9fff])")


def _hamming1_variants(target: str, corpus: str) -> set[str]:
    """在 corpus 中找与 target 等长、仅差 1 个汉字、且确为汉字串的形近变体。"""
    length = len(target)
    if length < 3 or not _is_cjk(target):
        return set()
    found: set[str] = set()
    seen: set[str] = set()
    for index in range(len(corpus) - length + 1):
        word = corpus[index : index + length]
        if word in seen or word == target:
            continue
        seen.add(word)
        if not _is_cjk(word):
            continue
        if sum(1 for left, right in zip(word, target, strict=False) if left != right) == 1:
            found.add(word)
    return found


def find_candidates(
    terms: list[GlossaryTerm], conflicts: list[dict[str, Any]], corpus: str
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for term in terms:
        variants: set[str] = set()
        if term.type == TYPE_PERSON:
            variants |= _hamming1_variants(term.target, corpus)
        if len(variants) > 8:
            continue
        if variants:
            candidates[term.source] = {
                "source": term.source,
                "current": term.target,
                "type": term.type,
                "variants": sorted(variants),
            }
    for conflict in conflicts:
        source = conflict["source"]
        entry = candidates.setdefault(
            source,
            {
                "source": source,
                "current": conflict.get("existing_target", ""),
                "type": "",
                "variants": [],
            },
        )
        for variant in (conflict.get("existing_target"), conflict.get("proposed_target")):
            if variant and variant != entry["current"] and variant not in entry["variants"]:
                entry["variants"].append(variant)
    return candidates


def apply_unifications(
    glossary: GlossaryStore, unifications: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    applied: list[dict[str, Any]] = []
    replace_map: dict[str, str] = {}
    for unification in unifications:
        source = str(unification["source"])
        canonical = str(unification["canonical"]).strip()
        variants = [
            str(value).strip() for value in unification.get("variants", []) if str(value).strip()
        ]
        variants = [variant for variant in variants if variant and variant != canonical]
        if not canonical:
            continue
        glossary.lock_term(source, canonical)
        if variants:
            glossary.upsert_term(
                GlossaryTerm(
                    source=source,
                    target=canonical,
                    aliases=variants,
                    confidence="high",
                    locked=True,
                )
            )
        glossary.mark_conflicts_resolved(source)
        for variant in variants:
            if _is_cjk(variant):
                replace_map[variant] = canonical
        applied.append(
            {
                "source": source,
                "canonical": canonical,
                "variants": variants,
                "reason": unification.get("reason", ""),
            }
        )
    return applied, replace_map
