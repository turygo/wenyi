"""Safely identify reader-visible XHTML mirrors of the canonical EPUB TOC."""

from __future__ import annotations

import re
from typing import Any

from lxml import etree

from trans_novel.epub.navigation import resolve_epub_href
from trans_novel.ingest.models import KIND_HEADING, Segment

_WS_RE = re.compile(r"[ \t\r\n\f\v]+")


def _local(node: etree._Element) -> str:
    return node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""


def _normalized_title(value: str) -> str:
    return " ".join(value.split()).casefold()


def _allows_root_omission(document_title: str, root_title: str) -> bool:
    normalized_document_title = _normalized_title(document_title)
    normalized_root_title = _normalized_title(root_title)
    return bool(normalized_root_title) and (
        normalized_document_title == normalized_root_title
        or normalized_document_title.startswith(f"{normalized_root_title}:")
    )


def _owns_list_anchor(anchor: etree._Element) -> bool:
    owner = anchor.getparent()
    if owner is None or _local(owner) != "li":
        return False
    children = [child for child in owner if isinstance(child.tag, str)]
    if sum(_local(child) == "a" for child in children) != 1:
        return False
    if any(_local(child) not in {"a", "ol", "ul"} for child in children):
        return False
    if owner.text and owner.text.strip():
        return False
    return all(not child.tail or not child.tail.strip() for child in owner)


def _resolve(root: etree._Element, path: tuple[int, ...]) -> etree._Element | None:
    node = root
    try:
        for index in path:
            node = [child for child in node if isinstance(child.tag, str)][index]
    except IndexError:
        return None
    return node


def _direct_link(block: etree._Element) -> etree._Element | None:
    if _local(block) == "a":
        return block if _owns_list_anchor(block) else None
    children = [child for child in block if isinstance(child.tag, str)]
    if len(children) != 1 or _local(children[0]) != "a":
        return None
    anchor = children[0]
    block_text = _WS_RE.sub(" ", "".join(block.itertext())).strip()
    anchor_text = _WS_RE.sub(" ", "".join(anchor.itertext())).strip()
    return anchor if block_text and block_text == anchor_text else None


def mark_mirrored_toc_segments(
    resources: list[dict[str, object]],
    trees: dict[str, etree._Element],
    toc_entries: list[dict[str, Any]],
    toc_paths: list[str],
    canonical_toc_path: str,
    document_title: str,
) -> None:
    """Persist exact canonical title IDs only for structurally complete TOC mirrors."""
    if not canonical_toc_path:
        canonical_toc_path = next(
            (
                path
                for entry in toc_entries
                if isinstance((path := entry.get("toc_path")), str) and path
            ),
            "",
        )
    canonical = [
        entry
        for entry in toc_entries
        if entry.get("toc_path") == canonical_toc_path
        and isinstance(entry.get("entry_id"), str)
        and isinstance(entry.get("target_key"), str)
        and entry.get("target_key")
        and not entry.get("external")
        and isinstance(entry.get("title"), str)
        and bool(entry["title"].strip())
    ]
    by_target: dict[str, list[str]] = {}
    for entry in canonical:
        by_target.setdefault(entry["target_key"], []).append(entry["entry_id"])
    unique_targets = {target: ids[0] for target, ids in by_target.items() if len(ids) == 1}
    expected_order = [
        entry["target_key"] for entry in canonical if entry["target_key"] in unique_targets
    ]
    if len(expected_order) < 2 or len(expected_order) != len(set(expected_order)):
        return
    accepted_orders = [expected_order]
    if canonical[0]["target_key"] == expected_order[0] and _allows_root_omission(
        document_title, canonical[0]["title"]
    ):
        accepted_orders.append(expected_order[1:])

    declared = set(toc_paths)
    for resource in resources:
        href = resource.get("href")
        raw_segments = resource.get("segments")
        if not isinstance(href, str) or href in declared or not isinstance(raw_segments, list):
            continue
        root = trees.get(href)
        if root is None:
            continue
        segments = [segment for segment in raw_segments if isinstance(segment, Segment)]
        headings = [segment for segment in segments if segment.kind == KIND_HEADING]
        if len(headings) > 1:
            continue
        links: list[tuple[Segment, str]] = []
        rejected = False
        for segment in segments:
            if segment.kind == KIND_HEADING:
                continue
            state = segment.epub_state
            block = _resolve(root, state.block_path) if state is not None else None
            anchor = _direct_link(block) if block is not None else None
            raw_href = anchor.get("href") if anchor is not None else None
            if not isinstance(raw_href, str) or not raw_href:
                rejected = True
                break
            try:
                resolved = resolve_epub_href(href, raw_href)
            except ValueError:
                rejected = True
                break
            if resolved.external or resolved.target_key not in unique_targets:
                rejected = True
                break
            links.append((segment, resolved.target_key))
        target_order = [target for _segment, target in links]
        if (
            rejected
            or len(links) < 2
            or len(target_order) != len(set(target_order))
            or target_order not in accepted_orders
        ):
            continue
        for segment, target in links:
            segment.meta["mirrored_toc_entry_id"] = unique_targets[target]


__all__ = ["mark_mirrored_toc_segments"]
