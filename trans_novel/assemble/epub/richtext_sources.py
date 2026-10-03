"""为已有章节从可信原书重新建立源内联清单。"""

from __future__ import annotations

import hashlib
import zipfile
from collections import defaultdict

from lxml import etree

from trans_novel.assemble.epub.legacy_whitespace import restore_legacy_whitespace_fields
from trans_novel.assemble.epub.rendering.note_recovery import bind_note_references
from trans_novel.assemble.epub.richtext_styles import enrich_rich_sources
from trans_novel.epub.archive import preflight_zip, read_member
from trans_novel.epub.markup import resource_parser
from trans_novel.epub.slots import slot_contract_digest
from trans_novel.ingest.epub.markup import designated_slot_values
from trans_novel.ingest.epub.richtext import extract_rich_sources
from trans_novel.ingest.models import Chapter


def _resolve(root: etree._Element, path: tuple[int, ...]) -> etree._Element:
    """依据原书元素坐标定位，忽略注释等非元素节点。"""
    for index in path:
        children = [child for child in root if isinstance(child.tag, str)]
        if not 0 <= index < len(children):
            raise ValueError("rich source block locator mismatch")
        root = children[index]
    return root


def _hydrate_resource(archive, href, entries, protected, restore_legacy_whitespace) -> list[dict]:
    data = read_member(archive, archive.getinfo(href))
    digest = hashlib.sha256(data).hexdigest()
    tree, mode, _ = resource_parser(data)
    root = tree.getroot()
    groups = defaultdict(list)
    records = []
    for state, owner in entries:
        if state.resource_sha256 != digest or state.parse_mode != mode:
            raise ValueError("rich source resource mismatch")
        if state.slot_contract_sha256 != slot_contract_digest(state.slots):
            raise ValueError("rich source slot contract mismatch")
        groups[state.block_path].append((state, owner))
    for path, group in groups.items():
        block = _resolve(root, path)
        fingerprint = hashlib.sha256(
            etree.tostring(block, encoding="utf-8", with_tail=False)
        ).hexdigest()
        if any(state.block_fingerprint != fingerprint for state, _ in group):
            raise ValueError("rich source block fingerprint mismatch")
        fields = designated_slot_values(block, root=root, protected_paths=protected)
        expected = set(fields)
        actual = {
            (slot.element_path, slot.field, slot.source_value)
            for state, _ in group
            for slot in state.slots
        }
        if actual != expected and restore_legacy_whitespace:
            records.extend(
                restore_legacy_whitespace_fields(block, group, fields, resource_href=href)
            )
        unique = {}
        keys = []
        for state, _ in group:
            key = tuple((slot.element_path, slot.field, slot.source_value) for slot in state.slots)
            keys.append(key)
            unique.setdefault(key, state)
        actual = {field for key in unique for field in key}
        if actual != expected:
            raise ValueError("rich source block slot coverage mismatch")
        sources = extract_rich_sources(
            root, block, list(unique.values()), protected_paths=protected
        )
        by_key = dict(zip(unique, sources, strict=True))
        for (state, _), key in zip(group, keys, strict=True):
            source = by_key[key].model_copy(deep=True)
            original = unique[key]
            slot_ids = {
                old.id: new.id for old, new in zip(original.slots, state.slots, strict=True)
            }
            for run in source.runs:
                if run.slot_id is not None:
                    run.slot_id = slot_ids[run.slot_id]
            state.rich_source = source
    return records


def hydrate_rich_sources(
    source_path: str,
    chapters: list[Chapter],
    *,
    protected_paths: dict[str, set[tuple[int, ...]]] | None = None,
    restore_legacy_whitespace: bool = False,
) -> list[dict]:
    """只从原书和原始源坐标提取清单，绝不将旧译文槽位视为格式证据。"""
    by_resource = defaultdict(list)
    for chapter in chapters:
        for segment in chapter.segments:
            if segment.epub_state is None:
                raise ValueError("rich source hydration requires EPUB source coordinates")
            by_resource[segment.epub_state.resource_href].append(
                (
                    segment.epub_state,
                    {
                        "chapter_index": chapter.index,
                        "segment_index": segment.index,
                        "anchor": segment.anchor,
                    },
                )
            )
    if not by_resource:
        return []
    records = []
    with zipfile.ZipFile(source_path) as archive:
        preflight_zip(archive)
        for href, entries in by_resource.items():
            records.extend(
                _hydrate_resource(
                    archive,
                    href,
                    entries,
                    (protected_paths or {}).get(href, set()),
                    restore_legacy_whitespace,
                )
            )
    enrich_rich_sources(source_path, chapters)
    bind_note_references(source_path, chapters)
    return records


__all__ = ["hydrate_rich_sources"]
