"""用旧版 BR 分组的可重放证据恢复已丢弃的纯 XML 空白源字段。"""

from __future__ import annotations

from lxml import etree

from trans_novel.epub.slots import EpubTextSlot, slot_contract_digest


def _legacy_runs(block, fields):
    """重放 3367d77 的直接 BR 分组，不能用译文长度推断归属。"""
    children = [child for child in block if isinstance(child.tag, str)]
    runs = [[]]
    for field in fields:
        path = field[0]
        index = (
            0
            if not path
            else sum(
                child.tag.rsplit("}", 1)[-1].lower() == "br" for child in children[: path[0] + 1]
            )
        )
        while len(runs) <= index:
            runs.append([])
        runs[index].append(field)
    return runs


def _restoration_plan(block, entries, fields):
    runs = _legacy_runs(block, fields)
    retained = {
        tuple(run): index
        for index, run in enumerate(runs)
        if any(value.strip() for _, _, value in run)
    }
    actual = {
        tuple((slot.element_path, slot.field, slot.source_value) for slot in state.slots)
        for state, _ in entries
    }
    if actual != set(retained):
        raise ValueError("legacy whitespace restoration requires all retained BR runs")
    discarded = [
        (index, run) for index, run in enumerate(runs) if run and tuple(run) not in retained
    ]
    if not discarded or not retained:
        raise ValueError("legacy whitespace restoration has no proven discarded run")
    if any(set(value) - set(" \t\r\n") for _, run in discarded for _, _, value in run):
        raise ValueError("legacy whitespace restoration requires pure XML whitespace")
    by_index = {index: key for key, index in retained.items()}
    plan = {key: [] for key in retained}
    for index, run in discarded:
        preceding = [candidate for candidate in by_index if candidate < index]
        owner = (
            max(preceding)
            if preceding
            else min(candidate for candidate in by_index if candidate > index)
        )
        plan[by_index[owner]].extend(run)
    return plan


def restore_legacy_whitespace_fields(
    block: etree._Element, entries, fields, *, resource_href: str
) -> list[dict]:
    """调用方已核对原书哈希与坐标；恢复字段但不新增译文段落或文字。"""
    plan = _restoration_plan(block, entries, fields)
    order = {field: index for index, field in enumerate(fields)}
    records = []
    for state, owner in entries:
        geometry = tuple((slot.element_path, slot.field, slot.source_value) for slot in state.slots)
        used_ids = {slot.id for slot in state.slots}
        anchor = owner.get("anchor") or f"legacy:{owner['chapter_index']}:{owner['segment_index']}"
        for path, field, value in plan[geometry]:
            sequence = 1
            slot_id = f"{anchor}:restored_ws{sequence}"
            while slot_id in used_ids:
                sequence += 1
                slot_id = f"{anchor}:restored_ws{sequence}"
            used_ids.add(slot_id)
            state.slots.append(
                EpubTextSlot(
                    id=slot_id, element_path=path, field=field, source_value=value, target_value=""
                )
            )
            records.append(
                {
                    "resource_href": resource_href,
                    "block_path": list(state.block_path),
                    **owner,
                    "slot_id": slot_id,
                    "element_path": list(path),
                    "field": field,
                    "source_value": value,
                }
            )
        state.slots.sort(key=lambda slot: order[(slot.element_path, slot.field, slot.source_value)])
        state.slot_contract_sha256 = slot_contract_digest(state.slots)
    return records


__all__ = ["restore_legacy_whitespace_fields"]
