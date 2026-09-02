"""Direct-run source/target pairing proof stages."""

from __future__ import annotations

from typing import Any

from lxml import etree

from trans_novel.assemble.epub.rendering import (
    BILINGUAL_DIRECT_TARGET_ATTRS,
    direct_run_add_whitespace,
    direct_run_has_active_ancestor,
    direct_run_is_active,
    direct_run_source_copy,
)
from trans_novel.assemble.epub.verification import archive_model, dom


def _find_direct_target(
    owner: etree._Element,
    boundary: etree._Element,
    slot: Any,
    expected_target: str,
    target_block: etree._Element,
) -> tuple[etree._Element | None, etree._Element | None]:
    if slot.field == "text":
        target_candidates = [
            child
            for child in dom.element_children_lxml(owner)
            if dict(child.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS
        ]
        target_node = next(
            (child for child in target_candidates if (child.text or "") == expected_target),
            None,
        )
        source_parent = (
            owner if boundary is owner and not direct_run_is_active(owner) else boundary.getparent()
        )
        return target_node, source_parent
    source_parent = boundary if boundary is target_block else boundary.getparent()
    target_parent = owner.getparent()
    target_node = None
    if target_parent is not None:
        siblings = dom.element_children_lxml(target_parent)
        try:
            owner_index = siblings.index(owner)
        except ValueError:
            owner_index = -1
        if owner_index >= 0:
            target_node = next(
                (
                    child
                    for child in siblings[owner_index + 1 :]
                    if dict(child.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS
                    and (child.text or "") == expected_target
                ),
                None,
            )
    return target_node, source_parent


def _find_direct_source_node(
    root_output: etree._Element,
    source_block: etree._Element,
    original_owner: etree._Element,
    source_parent: etree._Element,
    source_lang: str,
    slot: Any,
    grouped_ruby: bool,
    source_prefix: str,
    source_suffix: str,
    target_node: etree._Element | None,
    boundary: etree._Element,
    owner: etree._Element,
    target_block: etree._Element,
    order: str,
    assigned: set[tuple[int, ...]],
) -> etree._Element | None:
    expected_source_node = direct_run_source_copy(
        source_block,
        original_owner,
        source_lang=source_lang,
        source_tag="span",
        source_value=slot.source_value,
        ruby_source=grouped_ruby,
    )
    direct_run_add_whitespace(
        expected_source_node,
        prefix=source_prefix,
        suffix=source_suffix,
    )
    source_match_value = dom.source_node_visible_text(expected_source_node)
    candidates = [
        node
        for node in dom.element_children_lxml(source_parent)
        if "tn-source" in str(node.get("class", "")).split()
        and dom.element_path_lxml(root_output, node) not in assigned
        and dom.source_node_visible_text(node) == source_match_value
    ]
    if (boundary is owner and slot.field == "text" and not direct_run_is_active(owner)) or (
        slot.field == "tail"
        and boundary is owner
        and not direct_run_is_active(owner)
        and target_node is not None
    ):
        siblings = dom.element_children_lxml(source_parent)
        target_index = siblings.index(target_node)
        return next(
            (
                node
                for node in candidates
                if (siblings.index(node) < target_index) == (order == "source_first")
            ),
            None,
        )
    if (
        slot.field == "tail"
        and boundary is owner
        and target_node is not None
        and target_node.getparent() is source_parent
    ):
        siblings = dom.element_children_lxml(source_parent)
        target_index = siblings.index(target_node)
        boundary_index = siblings.index(boundary) if boundary in siblings else -1
        return next(
            (
                node
                for node in candidates
                if boundary_index >= 0
                and ((siblings.index(node) < target_index) == (order == "source_first"))
                and ((siblings.index(node) < boundary_index) == (order == "source_first"))
            ),
            None,
        )
    if boundary is not target_block:
        siblings = dom.element_children_lxml(source_parent)
        boundary_index = siblings.index(boundary) if boundary in siblings else -1
        return next(
            (
                node
                for node in candidates
                if boundary_index >= 0
                and ((siblings.index(node) < boundary_index) == (order == "source_first"))
            ),
            None,
        )
    return candidates[0] if candidates else None


def _record_direct_source_match(
    root_source: etree._Element,
    root_output: etree._Element,
    source_block: etree._Element,
    original_owner: etree._Element,
    source_node: etree._Element,
    source_lang: str,
    slot: Any,
    grouped_ruby: bool,
    source_prefix: str,
    source_suffix: str,
    target_block: etree._Element,
    resource: str,
    failures: list[dict[str, str]],
    direct_source_paths: set[tuple[int, ...]],
    direct_source_object_ids: set[int],
    assigned: set[tuple[int, ...]],
    source_parent: etree._Element,
    last_source_index: dict[int, int],
) -> None:
    source_path = dom.element_path_lxml(root_output, source_node)
    if source_path is None:
        failures.append(
            archive_model.item("bilingual_source", "source_node_order", resource, "pair_mismatch")
        )
        return
    siblings = dom.element_children_lxml(source_parent)
    source_index = siblings.index(source_node)
    parent_key = id(source_parent)
    prior_index = last_source_index.get(parent_key)
    if prior_index is not None and source_index <= prior_index:
        failures.append(
            archive_model.item("bilingual_source", "source_node_order", resource, "pair_mismatch")
        )
    last_source_index[parent_key] = source_index
    direct_source_object_ids.add(id(source_node))
    assigned.add(source_path)
    direct_source_paths.add(source_path)
    if direct_run_has_active_ancestor(target_block, source_node):
        failures.append(
            archive_model.item(
                "bilingual_source", "source_node_active_ancestor", resource, "active"
            )
        )
    expected_source = direct_run_source_copy(
        source_block,
        original_owner,
        source_lang=source_lang,
        source_tag=source_node.tag,
        source_value=slot.source_value,
        ruby_source=grouped_ruby,
    )
    direct_run_add_whitespace(
        expected_source,
        prefix=source_prefix,
        suffix=source_suffix,
    )
    if dom.source_subtree_signature(expected_source) != dom.source_subtree_signature(source_node):
        failures.append(
            archive_model.item(
                "bilingual_source", "source_node_subtree_mismatch", resource, "invalid"
            )
        )


def _match_direct_slot(
    root_source: etree._Element,
    root_output: etree._Element,
    source_block: etree._Element,
    target_block: etree._Element,
    source_lang: str,
    order: str,
    resource: str,
    failures: list[dict[str, str]],
    record: tuple[Any, ...],
    source_prefix: str,
    source_suffix: str,
    assigned: set[tuple[int, ...]],
    last_source_index: dict[int, int],
    direct_source_paths: set[tuple[int, ...]],
    direct_source_object_ids: set[int],
) -> int:
    (
        _slot_index,
        slot,
        owner,
        original_owner,
        _ruby_path,
        grouped_ruby,
        source_duplicate,
        boundary,
    ) = record
    if owner is None or original_owner is None:
        failures.append(
            archive_model.item(
                "bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch"
            )
        )
        return 0
    if not slot.source_value.strip():
        return 0
    expected_target = slot.target_value if slot.target_value is not None else slot.source_value
    target_node, source_parent = _find_direct_target(
        owner, boundary, slot, expected_target, target_block
    )
    direct_target_used = 0
    if target_node is None or dict(target_node.attrib) != BILINGUAL_DIRECT_TARGET_ATTRS:
        failures.append(
            archive_model.item(
                "bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch"
            )
        )
    else:
        direct_target_used = 1
    if source_duplicate:
        return direct_target_used
    if source_parent is None:
        failures.append(
            archive_model.item(
                "bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch"
            )
        )
        return direct_target_used
    source_node = _find_direct_source_node(
        root_output,
        source_block,
        original_owner,
        source_parent,
        source_lang,
        slot,
        grouped_ruby,
        source_prefix,
        source_suffix,
        target_node,
        boundary,
        owner,
        target_block,
        order,
        assigned,
    )
    if source_node is None:
        failures.append(
            archive_model.item("bilingual_source", "source_node_order", resource, "pair_mismatch")
        )
        return direct_target_used
    _record_direct_source_match(
        root_source,
        root_output,
        source_block,
        original_owner,
        source_node,
        source_lang,
        slot,
        grouped_ruby,
        source_prefix,
        source_suffix,
        target_block,
        resource,
        failures,
        direct_source_paths,
        direct_source_object_ids,
        assigned,
        source_parent,
        last_source_index,
    )
    return direct_target_used


def match_direct_runs(
    root_source: etree._Element,
    root_output: etree._Element,
    source_nodes: list[etree._Element],
    source_block: etree._Element,
    target_block: etree._Element,
    block_path: tuple[int, ...],
    direct_segments: list[Any],
    run_plans: dict[int, Any],
    source_lang: str,
    order: str,
    resource: str,
    failures: list[dict[str, str]],
    direct_target_used: int,
    direct_source_paths: set[tuple[int, ...]],
    direct_source_object_ids: set[int],
):
    assigned: set[tuple[int, ...]] = set()
    last_source_index: dict[int, int] = {}
    for segment in direct_segments:
        records, _, source_prefixes, source_suffixes = run_plans[id(segment)]
        for record in records:
            slot_index = record[0]
            direct_target_used += _match_direct_slot(
                root_source,
                root_output,
                source_block,
                target_block,
                source_lang,
                order,
                resource,
                failures,
                record,
                source_prefixes.get(slot_index, ""),
                source_suffixes.get(slot_index, ""),
                assigned,
                last_source_index,
                direct_source_paths,
                direct_source_object_ids,
            )
    return direct_target_used, assigned
