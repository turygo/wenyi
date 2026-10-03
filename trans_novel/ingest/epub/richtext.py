"""从源文坐标提取格式证据和不可拆分对象。"""

from __future__ import annotations

from collections.abc import Collection

from lxml import etree

from trans_novel.epub.markup import is_backlink, is_noteref
from trans_novel.epub.richtext import InlineAtom, InlineMark, InlineRun, RichSource, tag_semantics
from trans_novel.epub.slots import EpubSegmentState

_OPAQUE = {"script", "style", "rt", "rp"}
_MEDIA = {"audio", "canvas", "embed", "hr", "iframe", "img", "math", "object", "svg", "video"}
_EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"


def _name(node: etree._Element) -> str:
    return node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""


def _children(node: etree._Element) -> list[etree._Element]:
    return [child for child in node if isinstance(child.tag, str)]


def _resolve(root: etree._Element, path: tuple[int, ...]) -> etree._Element:
    for index in path:
        root = _children(root)[index]
    return root


def _identifier(prefix: str, path: tuple[int, ...]) -> str:
    return prefix + ":" + ".".join(str(index) for index in path)


def _note(node: etree._Element, path: tuple[int, ...], protected: Collection) -> bool:
    epub_type = node.get(_EPUB_TYPE, node.get("epub:type", ""))
    role = node.get("role", "")
    return path in protected or is_noteref(epub_type, role) or is_backlink(epub_type, role)


def _atom_kind(node: etree._Element, path: tuple[int, ...], protected: Collection) -> str | None:
    if not isinstance(node.tag, str):
        return "opaque"
    name = _name(node)
    if name == "a" and _note(node, path, protected):
        return "note"
    children = _children(node)
    if (
        name == "sup"
        and len(children) == 1
        and node.text is None
        and children[0].tail is None
        and _name(children[0]) == "a"
        and _note(children[0], (*path, 0), protected)
    ):
        return "note"
    if name == "br":
        return "linebreak"
    if name in _MEDIA:
        return "media"
    if name in _OPAQUE:
        return "opaque"
    if not children and not node.text:
        return "anchor"
    return None


def extract_rich_sources(
    root: etree._Element,
    block: etree._Element,
    states: list[EpubSegmentState],
    *,
    protected_paths: Collection[tuple[int, ...]] = (),
) -> list[RichSource]:
    """按精确源字段归属分配证据；格式 ID 在同一个源块内稳定。"""
    if not states:
        return []
    base = states[0].block_path
    if _resolve(root, base) is not block or any(state.block_path != base for state in states):
        raise ValueError("rich extraction requires one original source block")
    owners = {}
    for index, state in enumerate(states):
        for slot in state.slots:
            key = (slot.element_path, slot.field)
            if key in owners:
                raise ValueError("rich extraction source slots overlap")
            owners[key] = (index, slot)
    events: list[tuple[int | None, InlineRun]] = []
    marks: dict[str, InlineMark] = {}
    atoms: dict[str, InlineAtom] = {}

    def add(path, field, text, active):
        owned = owners.get((path, field))
        if owned is None:
            return
        index, slot = owned
        if text != slot.source_value:
            raise ValueError("rich extraction source value mismatch")
        events.append((index, InlineRun(text=text, marks=active, slot_id=slot.id)))

    def walk(node, path, active):
        kind = _atom_kind(node, (*base, *path), protected_paths) if path else None
        if kind is not None:
            atom_id = _identifier("a", path)
            atoms[atom_id] = InlineAtom(
                id=atom_id,
                path=path,
                kind=kind,
                label=(
                    "".join(node.itertext()) if isinstance(node.tag, str) else (node.text or "")
                ),
            )
            events.append((None, InlineRun(atom=atom_id, marks=active)))
            return
        if path:
            mark_id = _identifier("m", path)
            name = _name(node)
            mark_kind = (
                "bold"
                if name in {"b", "strong"}
                else "italic"
                if name in {"i", "em"}
                else "link"
                if name == "a"
                else "style"
            )
            marks[mark_id] = InlineMark(
                id=mark_id,
                path=path,
                tag=node.tag,
                attributes=dict(node.attrib),
                source_text="",
                kind=mark_kind,
                semantics=tag_semantics(node.tag),
            )
            active = (*active, mark_id)
        add(path, "text", node.text, active)
        element_index = 0
        for child_index, child in enumerate(node):
            if not isinstance(child.tag, str):
                walk(child, (*path, -1 - child_index), active)
                continue
            walk(child, (*path, element_index), active)
            add((*path, element_index), "tail", child.tail, active)
            element_index += 1

    walk(block, (), ())
    for mark in marks.values():
        mark.source_text = "".join(run.text for _, run in events if mark.id in run.marks)
    assigned: list[list[InlineRun]] = [[] for _ in states]
    for position, (owner, run) in enumerate(events):
        if owner is None:
            preceding = next((x for x, _ in reversed(events[:position]) if x is not None), None)
            following = next((x for x, _ in events[position + 1 :] if x is not None), None)
            atom = atoms[run.atom]
            owner = following if atom.kind == "linebreak" else preceding
            if owner is None:
                owner = following if following is not None else preceding
        if owner is None:
            raise ValueError("rich atom has no source text owner")
        assigned[owner].append(run)
    result = []
    for state, runs in zip(states, assigned, strict=True):
        used_marks = {mark_id for run in runs for mark_id in run.marks}
        used_atoms = {run.atom for run in runs if run.atom is not None}
        source = RichSource(
            marks=[mark for mark in marks.values() if mark.id in used_marks],
            atoms=[atom for atom in atoms.values() if atom.id in used_atoms],
            runs=runs,
        )
        if source.text != "".join(slot.source_value for slot in state.slots):
            raise ValueError("rich source does not cover the exact source slots")
        result.append(source)
    return result


__all__ = ["extract_rich_sources"]
