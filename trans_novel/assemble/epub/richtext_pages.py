"""按可信原书页码身份将中文页码锚点移到最近句末。"""

from __future__ import annotations

import hashlib
import zipfile
from collections import defaultdict

from lxml import etree

from trans_novel.epub.archive import preflight_zip, read_member
from trans_novel.epub.markup import resource_parser
from trans_novel.epub.richtext import validate_rich_target
from trans_novel.epub.richtext_edits import place_boundary_atoms
from trans_novel.ingest.epub.markup import element_path
from trans_novel.ingest.epub.richtext import extract_rich_sources
from trans_novel.ingest.models import Chapter, segment_preserves_source

_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


def _resolve(root: etree._Element, path: tuple[int, ...]) -> etree._Element:
    """仅接受源元素坐标，不把其他对象推断成页码。"""
    for index in path:
        children = [child for child in root if isinstance(child.tag, str)]
        if not 0 <= index < len(children):
            raise ValueError("page anchor source locator mismatch")
        root = children[index]
    return root


def _geometry(state):
    return tuple((slot.element_path, slot.field, slot.source_value) for slot in state.slots)


def _verified_anchors(root, block, states):
    """重抽原子身份，拒绝将伪造坐标重新指向同块内另一个真实页码。"""
    unique = {}
    protected = set()
    for state in states:
        unique.setdefault(_geometry(state), state)
        for atom in state.rich_source.atoms:
            if atom.kind != "note":
                continue
            owner = _resolve(block, atom.path)
            for node in owner.iter():
                if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1].lower() == "a":
                    protected.add(element_path(root, node))
    fresh = extract_rich_sources(root, block, list(unique.values()), protected_paths=protected)
    inventories = dict(zip(unique, fresh, strict=True))
    for state in states:
        actual = [atom for atom in inventories[_geometry(state)].atoms if atom.kind == "anchor"]
        saved = [atom for atom in state.rich_source.atoms if atom.kind == "anchor"]
        if saved != actual:
            raise ValueError("page anchor source inventory mismatch")


def _settle_block(root, block, segments) -> int:
    states = [segment.epub_state for segment in segments]
    _verified_anchors(root, block, states)
    changes = 0
    for segment in segments:
        if segment_preserves_source(segment) or segment.rich_target is None:
            continue
        state = segment.epub_state
        ids = set()
        for atom in state.rich_source.atoms:
            if atom.kind != "anchor":
                continue
            node = _resolve(block, atom.path)
            types = node.get(_EPUB_TYPE, node.get("epub:type", "")).split()
            if "pagebreak" in types or "doc-pagebreak" in node.get("role", "").split():
                ids.add(atom.id)
        before = segment.target
        updated, count = place_boundary_atoms(segment.rich_target, ids, source=state.rich_source)
        validate_rich_target(state.rich_source, updated, expected_text=before)
        if count:
            segment.assign_translation(updated)
            changes += count
    return changes


def settle_page_anchors(source_path: str, chapters: list[Chapter]) -> int:
    """页码来源及对象身份保持不变，只更新已接受中文中的目标位置。"""
    grouped = defaultdict(list)
    for chapter in chapters:
        if chapter.preserve_source:
            continue
        for segment in chapter.segments:
            state = segment.epub_state
            if state is not None and state.rich_source is not None:
                grouped[state.resource_href].append(segment)
    resources = {
        href: segments
        for href, segments in grouped.items()
        if any(
            segment.rich_target is not None
            and not segment_preserves_source(segment)
            and any(atom.kind == "anchor" for atom in segment.epub_state.rich_source.atoms)
            for segment in segments
        )
    }
    if not resources:
        return 0
    changes = 0
    with zipfile.ZipFile(source_path) as archive:
        preflight_zip(archive)
        for href, segments in resources.items():
            data = read_member(archive, archive.getinfo(href))
            digest = hashlib.sha256(data).hexdigest()
            tree, mode, _ = resource_parser(data)
            root = tree.getroot()
            by_block = defaultdict(list)
            for segment in segments:
                state = segment.epub_state
                if state.resource_sha256 != digest or state.parse_mode != mode:
                    raise ValueError("page anchor source resource mismatch")
                by_block[state.block_path].append(segment)
            for path, block_segments in by_block.items():
                if not any(
                    segment.rich_target is not None
                    and any(atom.kind == "anchor" for atom in segment.epub_state.rich_source.atoms)
                    for segment in block_segments
                ):
                    continue
                block = _resolve(root, path)
                fingerprint = hashlib.sha256(
                    etree.tostring(block, encoding="utf-8", with_tail=False)
                ).hexdigest()
                if any(
                    segment.epub_state.block_fingerprint != fingerprint
                    for segment in block_segments
                ):
                    raise ValueError("page anchor source block fingerprint mismatch")
                changes += _settle_block(root, block, block_segments)
    return changes


__all__ = ["settle_page_anchors"]
