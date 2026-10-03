"""仅在实际 DOM 证明规则无法生效后投影校验副本，不改原 CSS。"""

from __future__ import annotations

from collections.abc import Mapping

import tinycss2
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.css import is_animation_property
from trans_novel.assemble.epub.rendering.theme.source_selectors import (
    source_declarations,
    source_node_map,
    source_selector_branches,
    source_selector_may_match,
)


def _missing_original_identity(selector, soup):
    """正向原书 ID/类名缺失才稳定；生成节点只可能新增保留的 tn- 身份。"""
    tokens = tinycss2.parse_component_value_list(selector.base)
    for index, token in enumerate(tokens):
        if token.type == "hash" and token.is_identifier:
            if not token.value.startswith("tn-") and not soup.select(tinycss2.serialize([token])):
                return True
        elif token.type == "literal" and token.value == "." and index + 1 < len(tokens):
            name = tokens[index + 1]
            if (
                name.type == "ident"
                and not name.value.startswith("tn-")
                and not soup.select(tinycss2.serialize([token, name]))
            ):
                return True
    return False


def _project_rules(rules, root, view, resource, require_missing_identity, depth=0):
    if depth > 64:
        raise ThemeError("theme_css", "invalid_source_css", resource=resource)
    for rule in rules:
        if rule.type == "qualified-rule":
            declarations = source_declarations(rule.content, resource=resource)
            if not any(is_animation_property(item.name) for item in declarations):
                continue
            if view[0] is None:
                view[0] = source_node_map(root, resource=resource)[0]
            selectors = source_selector_branches(rule.prelude, resource=resource)
            if any(
                source_selector_may_match(view[0], item, resource=resource) for item in selectors
            ):
                continue
            if require_missing_identity and not all(
                _missing_original_identity(item, view[0]) for item in selectors
            ):
                continue
            parsed = tinycss2.parse_declaration_list(rule.content)
            retained = [
                item
                for item in parsed
                if not (item.type == "declaration" and is_animation_property(item.lower_name))
            ]
            rule.content = tinycss2.parse_component_value_list(tinycss2.serialize(retained))
        elif (
            rule.type == "at-rule"
            and rule.lower_at_keyword in {"media", "supports"}
            and rule.content is not None
        ):
            nested = tinycss2.parse_rule_list(rule.content)
            _project_rules(nested, root, view, resource, require_missing_identity, depth + 1)
            rule.content = tinycss2.parse_component_value_list(tinycss2.serialize(nested))


def project_nonmatching_animations(
    stylesheets: Mapping[str, bytes],
    root: etree._Element,
    *,
    resource: str,
    require_missing_identity: bool = False,
) -> dict[str, bytes]:
    """只剔除已证明零潜在匹配的动画声明供门禁检查，保留所有其他规则。"""
    projected = {}
    view = [None]
    for key, data in stylesheets.items():
        rules, _ = tinycss2.parse_stylesheet_bytes(data)
        try:
            _project_rules(rules, root, view, resource, require_missing_identity)
        except RecursionError:
            raise ThemeError("theme_css", "invalid_source_css", resource=resource) from None
        projected[key] = tinycss2.serialize(rules).encode("utf-8")
    return projected


def reject_inline_animations(root: etree._Element, *, resource: str) -> None:
    """源语义不猜真实节点上的动画时序；主题自身的内联处理仍由原门禁负责。"""
    for node in root.iter():
        style = node.get("style") if isinstance(node.tag, str) else None
        if style is None:
            continue
        try:
            declarations = source_declarations(
                tinycss2.parse_component_value_list(style), resource=resource
            )
        except (ThemeError, RecursionError):
            raise ThemeError("theme_css", "invalid_inline_css", resource=resource) from None
        if any(
            is_animation_property(item.name) and item.value.lower() != "none"
            for item in declarations
        ):
            raise ThemeError("theme_css", "source_animation", resource=resource)
