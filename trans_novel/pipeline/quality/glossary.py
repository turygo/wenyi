"""RunStore/chapter orchestration for glossary auditing."""

from __future__ import annotations

import re
from typing import Any

from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.epub.richtext_edits import apply_rich_edits, replace_rich_terms
from trans_novel.glossary.audit import (
    CJK_SPACE_GAP_RE,
    apply_unifications,
    find_candidates,
    has_cjk,
    is_latin_source,
)
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest import segment_preserves_source
from trans_novel.postprocess.text_edits import TextEdit, apply_text_edits


def target_corpus(store) -> str:
    manifest = store.load_manifest()
    parts: list[str] = []
    for chapter in manifest["chapters"]:
        loaded = store.load_chapter(chapter["index"])
        parts.extend(
            segment.target or ""
            for segment in loaded.text_segments
            if not segment_preserves_source(segment)
        )
    return "\n".join(parts)


def rewrite_targets(store, glossary: GlossaryStore, replace_map: dict[str, str]) -> int:
    """把各章 target 里的变体替换为规范译法。返回改动段数。"""
    variants_sorted = sorted(replace_map, key=len, reverse=True)

    def apply(text: str, executed: list[dict[str, str]]) -> str:
        if not text:
            return text
        current = RichTarget(runs=[InlineRun(text=text)])
        for variant in variants_sorted:
            before = current.text
            current, _conflicts = replace_rich_terms(current, {variant: replace_map[variant]})
            if current.text != before:
                executed.append({"variant": variant, "canonical": replace_map[variant]})
        text = current.text
        return text

    manifest = store.load_manifest()
    changed = 0
    for chapter in manifest["chapters"]:
        loaded = store.load_chapter(chapter["index"])
        dirty = False
        entries: list[dict[str, Any]] = []
        for index, segment in enumerate(loaded.segments):
            if segment_preserves_source(segment):
                continue
            if segment.target is None:
                continue
            executed: list[dict[str, str]] = []
            old = segment.target
            if segment.rich_target is not None:
                candidate = segment.rich_target
                for variant in variants_sorted:
                    before = candidate.text
                    candidate, conflicts = replace_rich_terms(
                        candidate,
                        {variant: replace_map[variant]},
                        source=segment.epub_state.rich_source,
                    )
                    if conflicts:
                        store.log_event(
                            "glossary_rewrite_skipped",
                            chapter=chapter["index"],
                            index=index,
                            conflicts=conflicts,
                        )
                    if candidate.text != before:
                        executed.append({"variant": variant, "canonical": replace_map[variant]})
                if candidate.text == old:
                    continue
                segment.assign_translation(candidate)
            elif segment.epub_state is not None:
                if any(variant in old for variant in variants_sorted):
                    store.log_event(
                        "glossary_rewrite_skipped",
                        chapter=chapter["index"],
                        index=index,
                        conflicts=["EPUB target needs trusted rich annotation"],
                    )
                continue
            else:
                new = apply(old, executed)
                if new == old:
                    continue
                segment.assign_translation(new)
            dirty = True
            changed += 1
            entries.append(
                {
                    "chapter": chapter["index"],
                    "index": index,
                    "before": old,
                    "after": segment.target,
                    "replacements": executed,
                }
            )
        if dirty:
            store.save_chapter(loaded)
            for entry in entries:
                store.log_event("glossary_rewrite_applied", **entry)

    manifest_dirty = False

    def rewrite_title(owner: dict[str, Any], *, event: dict[str, Any]) -> None:
        nonlocal manifest_dirty
        old_title = owner.get("title_translated")
        new_title = apply(old_title, []) if isinstance(old_title, str) else old_title
        if new_title == old_title:
            return
        owner["title_translated"] = new_title
        manifest_dirty = True
        store.log_event(
            "glossary_title_rewrite_applied",
            **event,
            before=old_title,
            after=new_title,
            replace_map=replace_map,
        )

    rewrite_title(manifest, event={"title": True})
    for chapter in manifest["chapters"]:
        rewrite_title(chapter, event={"chapter": chapter["index"]})
    meta = manifest.get("meta")
    toc_entries = meta.get("toc_entries", []) if isinstance(meta, dict) else []
    for entry in toc_entries:
        if isinstance(entry, dict):
            rewrite_title(entry, event={"toc_entry_id": entry.get("entry_id")})
    if manifest_dirty:
        store.save_manifest(manifest)
    return changed


def _assign_local_edits(segment, edits: list[TextEdit]) -> list[str]:
    """保留安全命中，返回未执行编辑的明确原因。"""
    if segment.rich_target is None:
        if segment.epub_state is not None:
            return ["EPUB target needs trusted rich annotation"]
        target = RichTarget(runs=[InlineRun(text=segment.target or "")])
        safe = []
        conflicts = []
        for edit in edits:
            try:
                apply_rich_edits(target, [edit])
            except ValueError as exc:
                conflicts.append(f"{edit.start}:{edit.end}: {exc}")
            else:
                safe.append(edit)
        segment.assign_translation(apply_text_edits(segment.target or "", safe))
        return conflicts
    safe: list[TextEdit] = []
    conflicts: list[str] = []
    for edit in edits:
        try:
            apply_rich_edits(segment.rich_target, [edit], source=segment.epub_state.rich_source)
        except ValueError as exc:
            conflicts.append(f"{edit.start}:{edit.end}: {exc}")
        else:
            safe.append(edit)
    if safe:
        segment.assign_translation(
            apply_rich_edits(segment.rich_target, safe, source=segment.epub_state.rich_source)
        )
    return conflicts


def _latin_edits(text: str, pattern: re.Pattern[str], replacement: str) -> list[TextEdit]:
    """仅替换中文上下文附近的完整拉丁术语。"""
    return [
        TextEdit(match.start(), match.end(), replacement)
        for match in pattern.finditer(text)
        if has_cjk(text[max(0, match.start() - 12) : match.start()])
        or has_cjk(text[match.end() : match.end() + 12])
    ]


def fix_latin_residue(store, glossary: GlossaryStore) -> list[dict[str, Any]]:
    """确定性修复锁定术语残留，局部保留格式并记录跨格式冲突。"""
    terms = [term for term in glossary.all_terms() if term.locked and is_latin_source(term.source)]
    compiled = [
        (term, re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term.source) + r"(?![A-Za-z0-9_])"))
        for term in terms
    ]
    touched_sources: set[str] = set()
    for chapter in store.load_manifest()["chapters"] if terms else []:
        loaded = store.load_chapter(chapter["index"])
        if loaded.preserve_source:
            continue
        entries: list[dict[str, Any]] = []
        for index, segment in enumerate(loaded.segments):
            if segment_preserves_source(segment):
                continue
            for term, pattern in compiled:
                old = segment.target
                if not old or not has_cjk(old) or term.target in old:
                    continue
                edits = _latin_edits(old, pattern, term.target)
                if not edits:
                    continue
                conflicts = _assign_local_edits(segment, edits)
                if conflicts:
                    store.log_event(
                        "glossary_latin_residue_skipped",
                        chapter=chapter["index"],
                        index=index,
                        term_source=term.source,
                        conflicts=conflicts,
                    )
                if segment.target == old:
                    continue
                spaces = [
                    TextEdit(match.start(), match.end(), "")
                    for match in CJK_SPACE_GAP_RE.finditer(segment.target or "")
                ]
                conflicts = _assign_local_edits(segment, spaces) if spaces else []
                if conflicts:
                    store.log_event(
                        "glossary_whitespace_skipped",
                        chapter=chapter["index"],
                        index=index,
                        conflicts=conflicts,
                    )
                touched_sources.add(term.source)
                entries.append(
                    {
                        "chapter": chapter["index"],
                        "index": index,
                        "before": old,
                        "after": segment.target,
                        "term_source": term.source,
                        "term_target": term.target,
                    }
                )
        if entries:
            store.save_chapter(loaded)
            for entry in entries:
                store.log_event("glossary_latin_residue_fixed", **entry)
    return [
        {
            "source": term.source,
            "canonical": term.target,
            "variants": [term.source],
            "reason": "锁定术语拉丁残留替换",
        }
        for term in terms
        if term.source in touched_sources
    ]


def audit_glossary(store, glossary: GlossaryStore, auditor) -> list[dict[str, Any]]:
    candidates = find_candidates(
        glossary.all_terms(), glossary.open_conflicts(), target_corpus(store)
    )
    unifications = auditor.decide(candidates)
    applied, replace_map = apply_unifications(glossary, unifications)
    if replace_map:
        rewrite_targets(store, glossary, replace_map)
    applied.extend(fix_latin_residue(store, glossary))
    return applied
