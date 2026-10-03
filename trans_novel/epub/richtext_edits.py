"""在译文自己的区间上执行编辑，保持格式与原子对象的所有权。"""

from __future__ import annotations

import re
from itertools import pairwise

from trans_novel.epub.richtext import InlineRun, RichSource, RichTarget
from trans_novel.postprocess.text_edits import (
    TextEdit,
    literal_ranges,
    punctuation_edits,
    validate_text_edits,
)

_LITERAL_TAGS = frozenset({"code", "kbd", "samp", "pre"})


def _edit_marks(target: RichTarget, edit: TextEdit) -> tuple[str, ...]:
    offset = 0
    consumed: set[tuple[str, ...]] = set()
    for run in target.runs:
        end = offset + len(run.text)
        if run.atom is not None:
            if edit.start < offset < edit.end or edit.start == edit.end == offset:
                raise ValueError("target edit crosses an immutable atom")
        elif max(offset, edit.start) < min(end, edit.end) or (
            edit.start == edit.end and offset <= edit.start <= end and run.text
        ):
            consumed.add(tuple(run.marks))
        offset = end
    if len(consumed) != 1:
        raise ValueError("target edit crosses different mark contexts")
    return next(iter(consumed))


def protected_rich_ranges(
    target: RichTarget, source: RichSource | None = None
) -> list[tuple[int, int]]:
    """代码范围及 URL 属于字面量，确定性编辑不得改变其内容。"""
    literal_ids = (
        {mark.id for mark in source.marks if mark.tag.rsplit("}", 1)[-1].lower() in _LITERAL_TAGS}
        if source is not None
        else set()
    )
    ranges = literal_ranges(target.text)
    offset = 0
    for run in target.runs:
        end = offset + len(run.text)
        if literal_ids.intersection(run.marks):
            ranges.append((offset, end))
        offset = end
    return ranges


def apply_rich_edits(
    target: RichTarget, edits: list[TextEdit], *, source: RichSource | None = None
) -> RichTarget:
    """原子地应用目标区间编辑；跨格式或原子对象的替换明确拒绝。"""
    validate_text_edits(target.text, edits)
    protected = protected_rich_ranges(target, source)
    if any(
        max(start, edit.start) < min(end, edit.end) for edit in edits for start, end in protected
    ):
        raise ValueError("target edit touches a protected literal")
    replacements = [_edit_marks(target, edit) for edit in edits]
    runs: list[InlineRun] = []
    offset = 0
    emitted: set[int] = set()
    for run in target.runs:
        end = offset + len(run.text)
        if run.atom is not None or not run.text:
            runs.append(run.model_copy(deep=True))
            continue
        cursor = offset
        for index, edit in enumerate(edits):
            if edit.end <= offset and not (edit.start == edit.end == offset):
                continue
            if edit.start >= end:
                continue
            before = max(cursor, min(end, edit.start))
            if before > cursor:
                runs.append(
                    InlineRun(text=run.text[cursor - offset : before - offset], marks=run.marks)
                )
            if offset <= edit.start < end and index not in emitted:
                if edit.replacement:
                    runs.append(InlineRun(text=edit.replacement, marks=replacements[index]))
                emitted.add(index)
            cursor = max(cursor, min(end, edit.end))
        if cursor < end:
            runs.append(InlineRun(text=run.text[cursor - offset :], marks=run.marks))
        offset = end
    for index, edit in enumerate(edits):
        if index not in emitted and edit.start == edit.end == len(target.text) and edit.replacement:
            runs.append(InlineRun(text=edit.replacement, marks=replacements[index]))
    return RichTarget(runs=runs)


def normalize_rich_target(target: RichTarget, *, source: RichSource | None = None) -> RichTarget:
    """整段识别标点，跳过字面量和有歧义的跨格式折叠。"""
    if source is not None:
        target = omit_decorations(target, source)
    literal_ranges = protected_rich_ranges(target, source)
    safe: list[TextEdit] = []
    for edit in punctuation_edits(target.text, protected_ranges=literal_ranges):
        if any(max(start, edit.start) < min(end, edit.end) for start, end in literal_ranges):
            continue
        try:
            _edit_marks(target, edit)
        except ValueError:
            continue
        safe.append(edit)
    return apply_rich_edits(target, safe)


def omit_decorations(target: RichTarget, source: RichSource) -> RichTarget:
    """取消中文首字装饰范围；原节点身份通过无文字节点保存。"""
    decorations = {mark.id: mark for mark in source.marks if mark.kind == "decoration"}
    identities: set[str] = set()
    runs = []
    for run in target.runs:
        removed = set(run.marks).intersection(decorations)
        for index, mark_id in enumerate(run.marks):
            if mark_id not in removed or mark_id in identities:
                continue
            mark = decorations[mark_id]
            if mark.attributes.get("id") or mark.attributes.get("name"):
                runs.append(InlineRun(marks=run.marks[: index + 1]))
                identities.add(mark_id)
        retained = tuple(mark for mark in run.marks if mark not in decorations)
        if run.text or run.atom is not None or retained:
            runs.append(run.model_copy(update={"marks": retained}))
    for mark_id, mark in decorations.items():
        if mark_id not in identities and (mark.attributes.get("id") or mark.attributes.get("name")):
            runs.append(InlineRun(marks=(mark_id,)))
    result = RichTarget(runs=runs)
    if result.text != target.text:
        raise ValueError("decoration removal changed target prose")
    return result


def replace_rich_terms(
    target: RichTarget, replacements: dict[str, str], *, source: RichSource | None = None
) -> tuple[RichTarget, list[str]]:
    """按变体长度依次替换，保留安全命中并报告跳过的格式冲突。"""
    current = target
    conflicts: list[str] = []
    for variant in sorted(replacements, key=len, reverse=True):
        if not variant:
            raise ValueError("term replacement variant must not be empty")
        edits: list[TextEdit] = []
        for match in re.finditer(re.escape(variant), current.text):
            edit = TextEdit(match.start(), match.end(), replacements[variant])
            try:
                apply_rich_edits(current, [edit], source=source)
            except ValueError as exc:
                conflicts.append(f"{variant}@{edit.start}:{edit.end}: {exc}")
            else:
                edits.append(edit)
        if edits:
            current = apply_rich_edits(current, edits, source=source)
    return current, conflicts


def _sentence_boundaries(text: str, protected: list[tuple[int, int]]) -> list[int]:
    """中文终止符及闭引号构成句界；英文句点只在段末生效。"""
    boundaries = {len(text)}
    closing = "”’\"'）)]】》」』〉}"
    for match in re.finditer(r"[。！？!?]+|…{2,}", text):
        end = match.end()
        while end < len(text) and text[end] in closing:
            end += 1
        if any(max(start, match.start()) < min(stop, end) for start, stop in protected):
            continue
        if any(start < end < stop for start, stop in protected):
            continue
        boundaries.add(end)
    return sorted(boundaries)


def place_boundary_atoms(
    target: RichTarget, atom_ids: set[str], *, source: RichSource | None = None
) -> tuple[RichTarget, int]:
    """只移动指定页码对象到最近中文句末，等距向后，保持文字和强调范围。"""
    if not atom_ids:
        return target, 0
    boundaries = _sentence_boundaries(target.text, protected_rich_ranges(target, source))
    destinations: dict[int, list[InlineRun]] = {}
    seen: set[str] = set()
    changed = 0
    offset = 0
    for run in target.runs:
        if run.atom in atom_ids:
            if run.atom in seen:
                raise ValueError("page anchor occurs more than once in target")
            seen.add(run.atom)
            destination = (
                0 if offset == 0 else min(boundaries, key=lambda end: (abs(end - offset), -end))
            )
            destinations.setdefault(destination, []).append(run.model_copy(deep=True))
            changed += destination != offset
        offset += len(run.text)
    if seen != atom_ids:
        raise ValueError("page anchor is missing from target")
    if not changed:
        return target, 0
    result: list[InlineRun] = []
    emitted: set[int] = set()

    def emit(position: int) -> None:
        if position not in emitted:
            result.extend(destinations.get(position, ()))
            emitted.add(position)

    offset = 0
    for run in target.runs:
        if run.atom in atom_ids:
            continue
        end = offset + len(run.text)
        emit(offset)
        if not run.text:
            result.append(run.model_copy(deep=True))
        else:
            cuts = [offset, *(point for point in destinations if offset < point < end), end]
            cuts.sort()
            for start, stop in pairwise(cuts):
                emit(start)
                result.append(
                    run.model_copy(update={"text": run.text[start - offset : stop - offset]})
                )
        offset = end
    emit(offset)
    updated = RichTarget(runs=result)
    if updated.text != target.text:
        raise ValueError("page anchor relocation changed target prose")
    return updated, changed
