"""独立核对译文文字、格式范围与原子节点后恢复源树供外层验证。"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from lxml import etree

from trans_novel.assemble.epub.rendering.bilingual import (
    BILINGUAL_SOURCE_CLASS,
    is_bilingual_container_tag,
    segment_needs_source,
)
from trans_novel.assemble.epub.rendering.source_dom import bilingual_source_copy
from trans_novel.assemble.epub.verification import dom
from trans_novel.assemble.epub.verification import source as source_proof
from trans_novel.epub.richtext import validate_rich_target
from trans_novel.ingest.epub.markup import designated_slot_values
from trans_novel.ingest.epub.richtext import extract_rich_sources


def _signature(node: etree._Element, *, tail: bool = False) -> tuple:
    return (
        (node.tag, getattr(node, "target", None)),
        tuple(sorted(node.attrib.items())),
        node.text or "",
        tuple(_signature(child, tail=True) for child in node),
        (node.tail or "") if tail else "",
    )


def _scope(mark, tag) -> tuple:
    return (
        tag,
        tuple(
            sorted(
                (key, value) for key, value in mark.attributes.items() if key not in {"id", "name"}
            )
        ),
    )


def _source_atom(block, path):
    if path and path[-1] < 0:
        parent = dom.resolve_path_lxml(block, path[:-1])
        node = list(parent)[-1 - path[-1]]
        if isinstance(node.tag, str):
            raise ValueError("EPUB opaque atom locator is invalid")
        return node
    return dom.resolve_path_lxml(block, path)


def _append_token(tokens: list, kind: str, value, scopes: tuple) -> None:
    if kind == "text" and not value:
        return
    if kind == "text" and tokens and tokens[-1][0] == kind and tokens[-1][2] == scopes:
        prior = tokens.pop()
        value = prior[1] + value
    tokens.append((kind, value, scopes))


def _slot_geometry(state):
    return tuple((slot.element_path, slot.field) for slot in state.slots)


def _complete_source_states(block, states, protected):
    owned = {}
    unique = {}
    for state in states:
        for slot in state.slots:
            location = (slot.element_path, slot.field)
            if location in owned and owned[location] != slot.source_value:
                raise ValueError("EPUB conflicting source field ownership")
            owned[location] = slot.source_value
        unique.setdefault(_slot_geometry(state), state)
    required = {
        (path, field): value
        for path, field, value in designated_slot_values(
            block, protected_paths=protected, root=block.getroottree().getroot()
        )
    }
    if owned != required:
        raise ValueError("EPUB rich source field coverage mismatch")
    return list(unique.values())


def _source_inventory(block, segments) -> tuple[dict, dict]:
    states = [segment.epub_state for segment in segments]
    protected = {
        (*state.block_path, *atom.path)
        for state in states
        for atom in state.rich_source.atoms
        if atom.kind == "note"
    }
    for state in states:
        for atom in state.rich_source.atoms:
            if atom.kind != "note":
                continue
            owner = dom.resolve_path_lxml(block, atom.path)
            for node in owner.iter():
                if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1] == "a":
                    protected.add(dom.element_path_lxml(block.getroottree().getroot(), node))
    unique_states = _complete_source_states(block, states, protected)
    fresh_sources = extract_rich_sources(
        block.getroottree().getroot(), block, unique_states, protected_paths=protected
    )
    expected = {
        _slot_geometry(state): fresh
        for state, fresh in zip(unique_states, fresh_sources, strict=True)
    }
    marks, atoms = {}, {}
    for segment in segments:
        fresh = expected[_slot_geometry(segment.epub_state)]
        saved = segment.epub_state.rich_source
        if saved is None or saved.runs != fresh.runs or saved.atoms != fresh.atoms:
            raise ValueError("EPUB rich source inventory mismatch")
        actual_marks = {mark.id: mark for mark in saved.marks}
        if set(actual_marks) != {mark.id for mark in fresh.marks}:
            raise ValueError("EPUB rich source marks mismatch")
        for mark in fresh.marks:
            actual = actual_marks[mark.id]
            if any(
                getattr(actual, field) != getattr(mark, field)
                for field in ("path", "tag", "attributes", "source_text")
            ):
                raise ValueError("EPUB rich source mark identity mismatch")
        validate_rich_target(saved, segment.rich_target, expected_text=segment.target)
        for inventory, descriptors in ((marks, saved.marks), (atoms, saved.atoms)):
            for descriptor in descriptors:
                if descriptor.id in inventory and inventory[descriptor.id] != descriptor:
                    raise ValueError("EPUB conflicting rich source descriptors")
                inventory[descriptor.id] = descriptor
    return marks, atoms


def prove_rich_target(source_block, output_block, segments) -> None:
    """按实际输出遍历核对文字与格式范围，不调用渲染器。"""
    marks, atoms = _source_inventory(source_block, segments)
    mark_scopes = {
        key: _scope(mark, dom.resolve_path_lxml(source_block, mark.path).tag)
        for key, mark in marks.items()
    }
    atom_signatures = {
        key: _signature(
            _source_atom(source_block, atom.path), tail=bool(atom.path and atom.path[-1] < 0)
        )
        for key, atom in atoms.items()
    }
    expected = []
    for segment in segments:
        for run in segment.rich_target.runs:
            scopes = tuple(mark_scopes[key] for key in run.marks)
            if run.atom is not None:
                _append_token(expected, "atom", atom_signatures[run.atom], scopes)
            else:
                _append_token(expected, "text", run.text, scopes)
    allowed_scopes = set(mark_scopes.values())
    identities = Counter(
        (key, value)
        for mark in marks.values()
        for key, value in mark.attributes.items()
        if key in {"id", "name"}
    )
    identity_scopes = {
        (key, value): mark_scopes[mark.id]
        for mark in marks.values()
        for key, value in mark.attributes.items()
        if key in {"id", "name"}
    }
    found_identities: Counter = Counter()
    actual = []

    def walk(node, scopes=()):
        _append_token(actual, "text", node.text or "", scopes)
        for child in node:
            signature = _signature(child, tail=not isinstance(child.tag, str))
            owns_tail = not isinstance(child.tag, str)
            if owns_tail and signature not in atom_signatures.values():
                untailed = _signature(child)
                if untailed in atom_signatures.values():
                    signature = untailed
                    owns_tail = False
            if signature in atom_signatures.values():
                _append_token(actual, "atom", signature, scopes)
                if owns_tail:
                    continue
            else:
                if not isinstance(child.tag, str):
                    raise ValueError("EPUB unexpected rich target node")
                namespace = (
                    output_block.tag.split("}", 1)[0] + "}"
                    if output_block.tag.startswith("{")
                    else ""
                )
                if child.tag == namespace + "span" and dict(child.attrib) == {
                    "data-tn-richtext": "text"
                }:
                    previous = child.getprevious()
                    if (
                        previous is None
                        or isinstance(previous.tag, str)
                        or not previous.tail
                        or len(child)
                    ):
                        raise ValueError("EPUB invalid opaque-tail text wrapper")
                    _append_token(actual, "text", child.text or "", scopes)
                    _append_token(actual, "text", child.tail or "", scopes)
                    continue
                scope = (
                    child.tag,
                    tuple(
                        sorted(
                            (key, value)
                            for key, value in child.attrib.items()
                            if key not in {"id", "name"}
                        )
                    ),
                )
                if scope not in allowed_scopes:
                    raise ValueError("EPUB unexpected rich target format")
                for key, value in child.attrib.items():
                    if key in {"id", "name"}:
                        if identity_scopes.get((key, value)) != scope:
                            raise ValueError("EPUB fabricated rich target identity")
                        found_identities[(key, value)] += 1
                walk(child, (*scopes, scope))
            _append_token(actual, "text", child.tail or "", scopes)

    walk(output_block)
    if actual != expected:
        raise ValueError("EPUB rich target prose, scopes or atoms mismatch")
    if found_identities != identities:
        raise ValueError("EPUB rich target inline identity missing or duplicated")


def _structural_block(root, path):
    current = root
    for index in path:
        children = [
            child
            for child in current
            if isinstance(child.tag, str) and "tn-source" not in child.get("class", "").split()
        ]
        if index >= len(children):
            raise ValueError("EPUB rich target block missing")
        current = children[index]
    return current


def _remove_bilingual_copy(source_block, target_block, segments, source_lang, order):
    if not any(segment_needs_source(segment) for segment in segments):
        return 0
    container = is_bilingual_container_tag(source_block.tag)
    candidates = list(target_block) if container else list(target_block.getparent())
    copies = [node for node in candidates if "tn-source" in node.get("class", "").split()]
    if container:
        matched = copies
    else:
        anchor = target_block.getprevious() if order == "source_first" else target_block.getnext()
        matched = [anchor] if anchor in copies else []
    if len(matched) != 1:
        raise ValueError("EPUB rich bilingual source pairing mismatch")
    copied = matched[0]
    expected = bilingual_source_copy(
        source_block, target_block, source_lang=source_lang, source_tag=copied.tag
    )
    expected.set("class", BILINGUAL_SOURCE_CLASS)
    if _signature(expected) != _signature(copied):
        raise ValueError("EPUB rich bilingual source subtree mismatch")
    if container:
        index = target_block.index(copied)
        if (order == "source_first" and (index != 0 or (target_block.text or "").strip())) or (
            order == "target_first"
            and (index != len(target_block) - 1 or (copied.tail or "").strip())
        ):
            raise ValueError("EPUB rich bilingual source order mismatch")
    source_proof.remove_preserving_tail(copied)
    return 1


def prove_and_restore_rich_blocks(
    root_source, root_output, groups, *, bilingual, source_lang, order
) -> int:
    """先独立证明重建块，之后恢复源内容以验证其他所有节点。"""
    added = 0
    for path, segments in groups.items():
        source_block = dom.resolve_path_lxml(root_source, path)
        target_block = _structural_block(root_output, path)
        if source_block is None or target_block.tag != source_block.tag:
            raise ValueError("EPUB rich target block identity mismatch")
        if dict(source_block.attrib) != dict(target_block.attrib):
            raise ValueError("EPUB rich target block attributes mismatch")
        if bilingual:
            added += _remove_bilingual_copy(
                source_block, target_block, segments, source_lang, order
            )
        prove_rich_target(source_block, target_block, segments)
        target_block.text = source_block.text
        for child in list(target_block):
            target_block.remove(child)
        for child in source_block:
            target_block.append(deepcopy(child))
    return added
