"""确定性识别 EPUB 注释关系。"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from lxml import etree

from trans_novel.epub.archive import safe_name
from trans_novel.epub.navigation import resolve_epub_href


class NoteMarker(TypedDict):
    resource_href: str
    path: list[int]
    kind: Literal["noteref", "backlink"]
    label: str
    target_resource: str
    target_path: list[int]


class NoteTarget(TypedDict):
    resource_href: str
    path: list[int]
    kind: Literal["footnote", "endnote"]


class NoteRelations(TypedDict):
    version: Literal[1]
    markers: list[NoteMarker]
    targets: list[NoteTarget]


_NOTE_BLOCK_TAGS = {"p", "li", "aside"}
_PROTECTED_TAGS = {"script", "style", "code", "math", "svg", "form", "nav"}
_SYMBOL_MARKER_RE = re.compile(r"^(?:\*{1,3}|†{1,3}|‡{1,3}|§{1,3}|¶{1,3})$")
_NUMBER_MARKER_RE = re.compile(r"^(?:[0-9]{1,4}|\[[0-9]{1,4}\]|\([0-9]{1,4}\))$")
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")
_RECOVERY_NUMBER_RE = re.compile(r"^(?:[0-9]{1,4}|\[[0-9]{1,4}\]|\([0-9]{1,4}\))\.?$")


def _local_name(element: etree._Element) -> str:
    return element.tag.rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""


def _tokens(element: etree._Element) -> tuple[set[str], set[str]]:
    epub_type = element.get("{http://www.idpf.org/2007/ops}type", element.get("epub:type", ""))
    return set(epub_type.split()), set(element.get("role", "").split())


def _marker_kinds(
    element: etree._Element,
) -> set[Literal["noteref", "backlink"]]:
    epub_types, roles = _tokens(element)
    kinds: set[Literal["noteref", "backlink"]] = set()
    if "noteref" in epub_types or "doc-noteref" in roles:
        kinds.add("noteref")
    if "backlink" in epub_types or "doc-backlink" in roles:
        kinds.add("backlink")
    return kinds


def _marker_kind(element: etree._Element) -> Literal["noteref", "backlink"] | None:
    kinds = _marker_kinds(element)
    return next(iter(kinds)) if len(kinds) == 1 else None


def _target_kinds(
    element: etree._Element,
) -> set[Literal["footnote", "endnote"]]:
    epub_types, roles = _tokens(element)
    kinds: set[Literal["footnote", "endnote"]] = set()
    if "footnote" in epub_types or "doc-footnote" in roles:
        kinds.add("footnote")
    if "endnote" in epub_types or "doc-endnote" in roles:
        kinds.add("endnote")
    return kinds


def _target_kind(element: etree._Element) -> Literal["footnote", "endnote"] | None:
    kinds = _target_kinds(element)
    return next(iter(kinds)) if len(kinds) == 1 else None


def _element_children(element: etree._Element) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str)]


def _element_path(root: etree._Element, element: etree._Element) -> list[int]:
    path: list[int] = []
    current = element
    while current is not root:
        parent = current.getparent()
        if parent is None:
            raise ValueError("EPUB note element is detached from its resource root")
        path.append(_element_children(parent).index(current))
        current = parent
    return list(reversed(path))


def _is_protected(element: etree._Element) -> bool:
    return any(
        _local_name(candidate) in _PROTECTED_TAGS
        for candidate in (element, *element.iterancestors())
    )


def _anchor_label(anchor: etree._Element) -> str:
    return "".join(anchor.itertext())


def _normalized_label(anchor: etree._Element) -> str:
    return _WS_RE.sub(" ", _anchor_label(anchor)).strip()


def _short_marker_kind(label: str) -> Literal["symbol", "number"] | None:
    if _SYMBOL_MARKER_RE.fullmatch(label):
        return "symbol"
    if _NUMBER_MARKER_RE.fullmatch(label):
        return "number"
    return None


def _labels_correspond(left: str, right: str) -> bool:
    if _NUMBER_MARKER_RE.fullmatch(left) and _NUMBER_MARKER_RE.fullmatch(right):
        return left.strip("[]()") == right.strip("[]()")
    return left == right


def _identifier_indexes(
    resources: Mapping[str, etree._Element], excluded: set[str]
) -> dict[str, dict[str, etree._Element | None]]:
    indexes: dict[str, dict[str, etree._Element | None]] = {}
    for href, root in resources.items():
        if href in excluded:
            continue
        index: dict[str, etree._Element | None] = {}
        for element in root.iter():
            if not isinstance(element.tag, str):
                continue
            for identifier in dict.fromkeys((element.get("id"), element.get("name"))):
                if not identifier:
                    continue
                previous = index.get(identifier)
                index[identifier] = (
                    element if previous is None and identifier not in index else None
                )
        indexes[href] = index
    return indexes


def _resolve_target(
    resource_href: str,
    anchor: etree._Element,
    resources: Mapping[str, etree._Element],
    indexes: Mapping[str, Mapping[str, etree._Element | None]],
) -> tuple[str, etree._Element] | None:
    raw_href = anchor.get("href", "")
    if not raw_href:
        return None
    try:
        parsed = urlsplit(raw_href)
        resolved = resolve_epub_href(resource_href, raw_href)
    except ValueError:
        return None
    if parsed.query:
        return None
    if (
        resolved.external
        or not resolved.resource_href
        or not resolved.fragment
        or not safe_name(resolved.resource_href)
        or resolved.resource_href not in resources
    ):
        return None
    target = indexes.get(resolved.resource_href, {}).get(resolved.fragment)
    return (
        (resolved.resource_href, target)
        if target is not None and not _is_protected(target)
        else None
    )


def _resolved_anchors(
    resources: Mapping[str, etree._Element],
    excluded: set[str],
    indexes: Mapping[str, Mapping[str, etree._Element | None]],
) -> tuple[
    list[tuple[str, etree._Element]],
    dict[int, tuple[str, etree._Element] | None],
]:
    anchors = [
        (href, element)
        for href, root in resources.items()
        if href not in excluded
        for element in root.iter()
        if _local_name(element) == "a" and not _is_protected(element)
    ]
    return anchors, {
        id(anchor): _resolve_target(href, anchor, resources, indexes) for href, anchor in anchors
    }


def _note_block(element: etree._Element) -> etree._Element | None:
    fallback: etree._Element | None = None
    for candidate in (element, *element.iterancestors()):
        kinds = _target_kinds(candidate)
        if len(kinds) > 1:
            return None
        if len(kinds) == 1:
            return candidate
        if fallback is None and _local_name(candidate) in _NOTE_BLOCK_TAGS:
            fallback = candidate
    return fallback


def _has_note_semantics(element: etree._Element) -> bool:
    return any(
        _target_kind(candidate) is not None for candidate in (element, *element.iterancestors())
    )


def _text_before(root: etree._Element, target: etree._Element) -> str:
    parts: list[str] = []

    def walk(node: etree._Element) -> bool:
        if node is target:
            return True
        if node.text:
            parts.append(node.text)
        for child in node:
            if not isinstance(child.tag, str):
                continue
            if walk(child):
                return True
            if child.tail:
                parts.append(child.tail)
        return False

    walk(root)
    return "".join(parts)


def _at_note_start(block: etree._Element, anchor: etree._Element) -> bool:
    prefix = _WS_RE.sub("", _text_before(block, anchor))
    if not prefix:
        return True
    while prefix:
        match = re.match(
            r"(?:\*{1,3}|†{1,3}|‡{1,3}|§{1,3}|¶{1,3}|[0-9]{1,4}|\[[0-9]{1,4}\]|\([0-9]{1,4}\))",
            prefix,
        )
        if match is None:
            return False
        prefix = prefix[match.end() :]
    return True


def _preceding_empty_identifier(anchor: etree._Element) -> etree._Element | None:
    top = anchor
    while top.getparent() is not None and _local_name(top.getparent()) in {"sup", "span"}:
        parent = top.getparent()
        if (
            (parent.text or "").strip()
            or (top.tail or "").strip()
            or len(_element_children(parent)) != 1
        ):
            break
        top = parent
    previous = top.getprevious()
    if (
        previous is not None
        and _local_name(previous) == "a"
        and (previous.get("id") or previous.get("name"))
        and not _anchor_label(previous).strip()
        and not (previous.tail or "").strip()
    ):
        return previous
    return None


_InferredPair = tuple[
    str,
    etree._Element,
    str,
    etree._Element,
    etree._Element,
    str,
    etree._Element,
    etree._Element,
]


def _inferred_note_pairs(
    anchors: list[tuple[str, etree._Element]],
    resolved: Mapping[int, tuple[str, etree._Element] | None],
    resource_kinds: Mapping[str, Literal["footnote", "endnote"]],
) -> list[_InferredPair]:
    pairs: list[_InferredPair] = []
    for ref_href, ref_anchor in anchors:
        ref_kinds = _marker_kinds(ref_anchor)
        if ref_kinds and ref_kinds != {"noteref"}:
            continue
        ref_semantic = ref_kinds == {"noteref"}
        ref_label = _normalized_label(ref_anchor)
        marker_type = _short_marker_kind(ref_label)
        if not ref_semantic and marker_type is None:
            continue
        linked_note = resolved[id(ref_anchor)]
        if linked_note is None:
            continue
        note_href, note_target = linked_note
        note_block = _note_block(note_target)
        if note_block is None:
            continue
        accepted_target = _preceding_empty_identifier(ref_anchor)
        for backlink_href, backlink in anchors:
            if backlink_href != note_href or not (
                backlink is note_block
                or any(backlink is node for node in note_block.iterdescendants())
            ):
                continue
            backlink_kinds = _marker_kinds(backlink)
            if backlink_kinds and backlink_kinds != {"backlink"}:
                continue
            backlink_semantic = backlink_kinds == {"backlink"}
            backlink_label = _normalized_label(backlink)
            if not backlink_semantic and _short_marker_kind(backlink_label) is None:
                continue
            linked_ref = resolved[id(backlink)]
            if linked_ref is None or linked_ref[0] != ref_href:
                continue
            if linked_ref[1] is not ref_anchor and linked_ref[1] is not accepted_target:
                continue
            if not _at_note_start(note_block, backlink):
                continue
            if not (
                ref_semantic or backlink_semantic or _labels_correspond(ref_label, backlink_label)
            ):
                continue
            if (
                not ref_semantic
                and marker_type == "number"
                and not _has_note_semantics(note_block)
                and note_href not in resource_kinds
            ):
                continue
            pairs.append(
                (
                    ref_href,
                    ref_anchor,
                    note_href,
                    note_target,
                    note_block,
                    backlink_href,
                    backlink,
                    linked_ref[1],
                )
            )
    return pairs


def detect_note_relations(
    resources: Mapping[str, etree._Element],
    *,
    excluded_resources: Collection[str] = (),
    note_resources: Mapping[str, Literal["footnote", "endnote"]] | None = None,
) -> NoteRelations:
    """返回经源文验证的注释标记及正文目标位置，不修改 DOM。"""
    excluded = set(excluded_resources)
    indexes = _identifier_indexes(resources, excluded)
    resource_kinds = note_resources or {}
    anchors, resolved = _resolved_anchors(resources, excluded, indexes)
    marker_records: dict[tuple[str, tuple[int, ...], str], NoteMarker] = {}
    target_records: dict[tuple[str, tuple[int, ...]], NoteTarget] = {}

    def add_marker(
        href: str,
        anchor: etree._Element,
        kind: Literal["noteref", "backlink"],
        target_href: str,
        target: etree._Element,
    ) -> None:
        path = _element_path(resources[href], anchor)
        key = (href, tuple(path), kind)
        marker_records.setdefault(
            key,
            {
                "resource_href": href,
                "path": path,
                "kind": kind,
                "label": _anchor_label(anchor),
                "target_resource": target_href,
                "target_path": _element_path(resources[target_href], target),
            },
        )

    def add_target(href: str, block: etree._Element) -> None:
        path = _element_path(resources[href], block)
        key = (href, tuple(path))
        target_records.setdefault(
            key,
            {
                "resource_href": href,
                "path": path,
                "kind": _target_kind(block) or resource_kinds.get(href, "footnote"),
            },
        )

    for href, anchor in anchors:
        explicit_kind = _marker_kind(anchor)
        linked = resolved[id(anchor)]
        if explicit_kind is None or linked is None:
            continue
        target_href, target = linked
        add_marker(href, anchor, explicit_kind, target_href, target)
        note_block = _note_block(target) if explicit_kind == "noteref" else _note_block(anchor)
        if note_block is not None:
            add_target(target_href if explicit_kind == "noteref" else href, note_block)

    inferred_pairs = _inferred_note_pairs(anchors, resolved, resource_kinds)
    inferred_kinds: dict[tuple[str, tuple[int, ...]], set[Literal["noteref", "backlink"]]] = {}
    for ref_href, ref, _, _, _, backlink_href, backlink, _ in inferred_pairs:
        ref_key = (ref_href, tuple(_element_path(resources[ref_href], ref)))
        backlink_key = (
            backlink_href,
            tuple(_element_path(resources[backlink_href], backlink)),
        )
        inferred_kinds.setdefault(ref_key, set()).add("noteref")
        inferred_kinds.setdefault(backlink_key, set()).add("backlink")
    conflicts = {key for key, kinds in inferred_kinds.items() if len(kinds) > 1}
    for (
        ref_href,
        ref,
        note_href,
        note_target,
        note_block,
        backlink_href,
        backlink,
        linked_ref,
    ) in inferred_pairs:
        ref_key = (ref_href, tuple(_element_path(resources[ref_href], ref)))
        backlink_key = (
            backlink_href,
            tuple(_element_path(resources[backlink_href], backlink)),
        )
        if ref_key in conflicts or backlink_key in conflicts:
            continue
        add_marker(ref_href, ref, "noteref", note_href, note_target)
        add_marker(backlink_href, backlink, "backlink", ref_href, linked_ref)
        add_target(note_href, note_block)

    return {
        "version": 1,
        "markers": list(marker_records.values()),
        "targets": list(target_records.values()),
    }


def note_resources_by_semantics(
    entries: list[dict],
) -> dict[str, Literal["footnote", "endnote"]]:
    """按 guide/landmark 的显式语义返回无冲突的注释资源。"""
    evidence: dict[str, set[Literal["footnote", "endnote"]]] = {}
    for entry in entries:
        href = entry.get("resource_href")
        if not isinstance(href, str) or not href:
            continue
        tokens = {
            token
            for name in ("type", "nav_type", "role")
            if isinstance(entry.get(name), str)
            for token in entry[name].split()
        }
        if tokens & {"notes", "footnotes", "doc-footnote"}:
            evidence.setdefault(href, set()).add("footnote")
        if tokens & {"endnotes", "doc-endnotes"}:
            evidence.setdefault(href, set()).add("endnote")
    return {href: next(iter(kinds)) for href, kinds in evidence.items() if len(kinds) == 1}


def _recovery_marker_kind(label: str) -> Literal["symbol", "number"] | None:
    if _SYMBOL_MARKER_RE.fullmatch(label):
        return "symbol"
    if _RECOVERY_NUMBER_RE.fullmatch(label):
        return "number"
    return None


def _recovery_labels_correspond(left: str, right: str) -> bool:
    if _RECOVERY_NUMBER_RE.fullmatch(left) and _RECOVERY_NUMBER_RE.fullmatch(right):
        return left.removesuffix(".").strip("[]()") == right.removesuffix(".").strip("[]()")
    return left == right


def _recovery_ref_targets(anchor: etree._Element) -> list[etree._Element]:
    targets = [anchor]
    preceding = _preceding_empty_identifier(anchor)
    if preceding is not None:
        targets.append(preceding)
    top = anchor
    while top.getparent() is not None and _local_name(top.getparent()) in {"span", "sup"}:
        parent = top.getparent()
        if (
            (parent.text or "").strip()
            or (top.tail or "").strip()
            or len(_element_children(parent)) != 1
        ):
            break
        targets.append(parent)
        top = parent
    return targets


def _recovery_has_body(block: etree._Element, backlink: etree._Element) -> bool:
    parts: list[str] = []
    for element in block.iter():
        if not isinstance(element.tag, str):
            continue
        within_backlink = element is backlink or any(
            ancestor is backlink for ancestor in element.iterancestors()
        )
        if not within_backlink and element.text:
            parts.append(element.text)
        if element is not block and element.tail and (element is backlink or not within_backlink):
            parts.append(element.tail)
    return bool("".join(parts).strip())


def _recovery_pair_allowed(
    ref: etree._Element,
    backlink: etree._Element,
    note_href: str,
    note_block: etree._Element,
    resource_kinds: Mapping[str, Literal["footnote", "endnote"]],
) -> bool:
    ref_label = _normalized_label(ref)
    backlink_label = _normalized_label(backlink)
    if not _recovery_labels_correspond(ref_label, backlink_label):
        return False
    if not _at_note_start(note_block, backlink):
        return False
    if not _recovery_has_body(note_block, backlink):
        return False
    if (
        _recovery_marker_kind(ref_label) == "number"
        and _marker_kinds(ref) != {"noteref"}
        and _marker_kinds(backlink) != {"backlink"}
        and not _has_note_semantics(note_block)
        and note_href not in resource_kinds
    ):
        body_block = _note_block(ref)
        return (
            _local_name(note_block) in _NOTE_BLOCK_TAGS
            and body_block is not None
            and not _at_note_start(body_block, ref)
        )
    return True


def _recoverable_pairs(
    anchors: list[tuple[str, etree._Element]],
    resolved: Mapping[int, tuple[str, etree._Element] | None],
    resource_kinds: Mapping[str, Literal["footnote", "endnote"]],
) -> list[_InferredPair]:
    pairs: list[_InferredPair] = []
    for ref_href, ref in anchors:
        if (
            _marker_kinds(ref) - {"noteref"}
            or _recovery_marker_kind(_normalized_label(ref)) is None
        ):
            continue
        linked_note = resolved[id(ref)]
        if linked_note is None:
            continue
        note_href, note_target = linked_note
        note_block = _note_block(note_target)
        if note_block is None:
            continue
        accepted_targets = _recovery_ref_targets(ref)
        backlinks = [
            (href, anchor)
            for href, anchor in anchors
            if href == note_href
            and (anchor is note_block or any(node is note_block for node in anchor.iterancestors()))
            and resolved[id(anchor)] is not None
            and resolved[id(anchor)][0] == ref_href
            and any(resolved[id(anchor)][1] is target for target in accepted_targets)
        ]
        if len(backlinks) != 1:
            continue
        backlink_href, backlink = backlinks[0]
        if (
            _marker_kinds(backlink) - {"backlink"}
            or _recovery_marker_kind(_normalized_label(backlink)) is None
            or not _recovery_pair_allowed(ref, backlink, note_href, note_block, resource_kinds)
        ):
            continue
        linked_ref = resolved[id(backlink)]
        if linked_ref is None:
            continue
        pairs.append(
            (
                ref_href,
                ref,
                note_href,
                note_target,
                note_block,
                backlink_href,
                backlink,
                linked_ref[1],
            )
        )
    return pairs


def _unique_recovery_pairs(
    pairs: list[_InferredPair],
    anchors: list[tuple[str, etree._Element]],
    resolved: Mapping[int, tuple[str, etree._Element] | None],
) -> list[_InferredPair]:
    roles: dict[int, set[str]] = {}
    inbound: dict[tuple[str, int], int] = {}
    for _, ref, _, _, _, _, backlink, _ in pairs:
        roles.setdefault(id(ref), set()).add("noteref")
        roles.setdefault(id(backlink), set()).add("backlink")
    for _, anchor in anchors:
        if _marker_kinds(anchor) - {"noteref"}:
            continue
        if _recovery_marker_kind(_normalized_label(anchor)) is None:
            continue
        linked = resolved[id(anchor)]
        if linked is None:
            continue
        block = _note_block(linked[1])
        if block is not None:
            key = (linked[0], id(block))
            inbound[key] = inbound.get(key, 0) + 1
    return [
        pair
        for pair in pairs
        if roles[id(pair[1])] == {"noteref"}
        and roles[id(pair[6])] == {"backlink"}
        and inbound.get((pair[2], id(pair[4]))) == 1
    ]


def detect_recoverable_note_relations(
    resources: Mapping[str, etree._Element],
    *,
    excluded_resources: Collection[str] = (),
    note_resources: Mapping[str, Literal["footnote", "endnote"]] | None = None,
) -> NoteRelations:
    """返回满足双向唯一证明的注释对象关系，保持源 DOM 和原始标签不变。"""
    excluded = set(excluded_resources)
    indexes = _identifier_indexes(resources, excluded)
    anchors, resolved = _resolved_anchors(resources, excluded, indexes)
    resource_kinds = note_resources or {}
    pairs = _recoverable_pairs(anchors, resolved, resource_kinds)
    markers: list[NoteMarker] = []
    targets: list[NoteTarget] = []
    for (
        ref_href,
        ref,
        note_href,
        note_target,
        block,
        backlink_href,
        backlink,
        linked_ref,
    ) in _unique_recovery_pairs(pairs, anchors, resolved):
        for href, anchor, kind, target_href, target in (
            (ref_href, ref, "noteref", note_href, note_target),
            (backlink_href, backlink, "backlink", ref_href, linked_ref),
        ):
            markers.append(
                {
                    "resource_href": href,
                    "path": _element_path(resources[href], anchor),
                    "kind": kind,
                    "label": _anchor_label(anchor),
                    "target_resource": target_href,
                    "target_path": _element_path(resources[target_href], target),
                }
            )
        targets.append(
            {
                "resource_href": note_href,
                "path": _element_path(resources[note_href], block),
                "kind": _target_kind(block) or resource_kinds.get(note_href, "footnote"),
            }
        )
    return {"version": 1, "markers": markers, "targets": targets}


__all__ = [
    "NoteMarker",
    "NoteRelations",
    "NoteTarget",
    "detect_note_relations",
    "detect_recoverable_note_relations",
    "note_resources_by_semantics",
]
