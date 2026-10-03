"""按独立的译文内联序列重建源文块。"""

from __future__ import annotations

from copy import deepcopy

from lxml import etree

from trans_novel.assemble.epub.rendering.source_dom import resolve_element_path
from trans_novel.epub.richtext import RichTarget, validate_rich_target
from trans_novel.ingest import Segment, segment_preserves_source


def rich_block_segments(segments: list[Segment]) -> dict[tuple[int, ...], list[Segment]]:
    """按原文块聚合已明确标注的译文，拒绝局部覆盖。"""
    grouped: dict[tuple[int, ...], list[Segment]] = {}
    for segment in segments:
        if segment.epub_state is not None:
            grouped.setdefault(segment.epub_state.block_path, []).append(segment)
    result: dict[tuple[int, ...], list[Segment]] = {}
    for path, items in grouped.items():
        decisions = {segment_preserves_source(item) for item in items}
        if len(decisions) > 1:
            raise ValueError("EPUB preserve range is ambiguous within one block")
        translated = [
            item
            for item in items
            if item.target is not None
            and (item.rich_target is not None or item.target != item.source)
            and not segment_preserves_source(item)
        ]
        if not translated:
            continue
        if any(item.epub_state.rich_source is not None for item in translated):
            prepared = []
            for item in items:
                source = item.epub_state.rich_source
                if item.rich_target is None and item.target == item.source and source is not None:
                    target = RichTarget(
                        runs=[run.model_copy(update={"slot_id": None}) for run in source.runs]
                    )
                    item = item.model_copy(update={"rich_target": target, "target": target.text})
                if (
                    item.rich_target is None
                    or item.target is None
                    or segment_preserves_source(item)
                ):
                    raise ValueError("EPUB translated rich block requires complete trusted targets")
                prepared.append(item)
            result[path] = prepared
    return result


def render_rich_block(
    block: etree._Element, segments: list[Segment]
) -> dict[etree._Element, etree._Element]:
    """只重建块内内容，并返回原文节点到译文节点的身份映射。"""
    marks, atoms = {}, {}
    runs = []
    for segment in segments:
        state, target = segment.epub_state, segment.rich_target
        if state is None or state.rich_source is None or target is None:
            raise ValueError("EPUB rich target contract is missing")
        validate_rich_target(state.rich_source, target, expected_text=segment.target)
        for inventory, descriptors in (
            (marks, state.rich_source.marks),
            (atoms, state.rich_source.atoms),
        ):
            for descriptor in descriptors:
                prior = inventory.get(descriptor.id)
                if prior is not None and prior != descriptor:
                    raise ValueError("EPUB conflicting inline source descriptors")
                inventory[descriptor.id] = descriptor
        runs.extend(target.runs)
    meaningful_marks = {
        key for run in runs if run.text or run.atom is not None for key in run.marks
    }
    source_marks = {key: resolve_element_path(block, mark.path) for key, mark in marks.items()}
    source_atoms = {key: _atom_owner(block, atom.path) for key, atom in atoms.items()}
    for key, original in source_marks.items():
        mark = marks[key]
        if (
            original.tag.rsplit("}", 1)[-1].lower() != mark.tag.rsplit("}", 1)[-1].lower()
            or dict(original.attrib) != mark.attributes
        ):
            raise ValueError("EPUB inline descriptor does not match source DOM")
    replacements = {block: block}
    staging = etree.Element(block.tag, nsmap=block.nsmap)
    active_ids: tuple[str, ...] = ()
    active_nodes = [staging]
    used_marks: set[str] = set()
    for run in runs:
        if not run.text and run.atom is None and set(run.marks) <= meaningful_marks:
            continue
        common = 0
        while (
            common < min(len(active_ids), len(run.marks))
            and active_ids[common] == run.marks[common]
        ):
            common += 1
        active_nodes = active_nodes[: common + 1]
        for key in run.marks[common:]:
            original = source_marks[key]
            attrs = dict(original.attrib)
            if key in used_marks:
                attrs.pop("id", None)
                attrs.pop("name", None)
            node = etree.SubElement(
                active_nodes[-1], original.tag, attrib=attrs, nsmap=original.nsmap
            )
            replacements.setdefault(original, node)
            used_marks.add(key)
            active_nodes.append(node)
        active_ids = run.marks
        parent = active_nodes[-1]
        if run.atom is not None:
            original = source_atoms[run.atom]
            copied = deepcopy(original)
            if isinstance(original.tag, str):
                copied.tail = None
            parent.append(copied)
            for before, after in zip(original.iter(), copied.iter(), strict=True):
                replacements[before] = after
        elif len(parent) and run.text:
            previous = parent[-1]
            if not isinstance(previous.tag, str) and previous.tail:
                namespace = block.tag.split("}", 1)[0] + "}" if block.tag.startswith("{") else ""
                text_node = etree.SubElement(
                    parent, namespace + "span", attrib={"data-tn-richtext": "text"}
                )
                text_node.text = run.text
            elif dict(previous.attrib) == {"data-tn-richtext": "text"}:
                previous.text = (previous.text or "") + run.text
            else:
                previous.tail = (previous.tail or "") + run.text
        else:
            parent.text = (parent.text or "") + run.text
    for child in list(block):
        block.remove(child)
    block.text = staging.text
    for child in list(staging):
        block.append(child)
    return replacements


def _atom_owner(block: etree._Element, path: tuple[int, ...]) -> etree._Element:
    if path and path[-1] < 0:
        parent = resolve_element_path(block, path[:-1])
        node = list(parent)[-1 - path[-1]]
        if isinstance(node.tag, str):
            raise ValueError("EPUB opaque atom locator points to an element")
        return node
    return resolve_element_path(block, path)
