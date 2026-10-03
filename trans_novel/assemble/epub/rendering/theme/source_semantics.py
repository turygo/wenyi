"""有限且可证明的源文字样式级联，以及中文首字装饰覆盖。"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import tinycss2
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.cascade import (
    normalize_inline,
    source_specificity_bound,
)
from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.css import CssDeclaration
from trans_novel.assemble.epub.rendering.theme.source_applicability import (
    project_nonmatching_animations,
    reject_inline_animations,
)
from trans_novel.assemble.epub.rendering.theme.source_selectors import (
    SEMANTIC_PROPERTIES,
    semantic_properties,
    source_declarations,
    source_node_map,
    source_rules,
    source_selector_applies,
    supported_source_selector,
)

_UNKNOWN = "<undetermined>"
_INITIAL = {"font-weight": 400.0, "font-style": "normal", "vertical-align": "baseline"}
_LENGTH = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:em|rem|ex|ch|px|pt|pc|in|cm|mm|%)$")


def _fail(resource: str) -> ThemeError:
    return ThemeError("theme_css", "ambiguous_source_semantics", resource=resource)


def _relative_weight(value: float, direction: str) -> float:
    if direction == "bolder":
        return 400.0 if value < 350 else 700.0 if value < 550 else 900.0 if value < 900 else value
    return value if value < 100 else 100.0 if value < 550 else 400.0 if value < 750 else 700.0


def _value(prop: str, text: str, parent):
    text = text.strip().lower()
    if text == "inherit" or (text == "unset" and prop != "vertical-align"):
        return parent
    if text == "initial" or (text == "unset" and prop == "vertical-align"):
        return _INITIAL[prop]
    if prop == "font-weight":
        if text in {"normal", "bold"}:
            return 400.0 if text == "normal" else 700.0
        if text in {"bolder", "lighter"}:
            return _UNKNOWN if parent == _UNKNOWN else _relative_weight(parent, text)
        try:
            number = float(text)
        except ValueError:
            return _UNKNOWN
        return number if 1 <= number <= 1000 else _UNKNOWN
    if prop == "font-style":
        if text in {"normal", "italic", "oblique"}:
            return text
        if re.fullmatch(r"oblique\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)deg", text):
            angle = float(text.split()[1][:-3])
            return "oblique" if -90 <= angle <= 90 else _UNKNOWN
        return _UNKNOWN
    if text in {"sub", "super"}:
        return text
    if text in {"baseline", "middle", "top", "bottom", "text-top", "text-bottom"}:
        return "baseline"
    if text == "0":
        return "baseline"
    match = _LENGTH.fullmatch(text)
    if match:
        value = float(match.group(1))
        return "super" if value > 0 else "sub" if value < 0 else "baseline"
    return _UNKNOWN


def _expanded(declarations: tuple[CssDeclaration, ...]):
    for index, declaration in enumerate(declarations):
        properties = semantic_properties((declaration,))
        for prop in properties:
            text = declaration.value
            if declaration.name not in SEMANTIC_PROPERTIES and text.strip().lower() not in {
                "inherit",
                "initial",
                "unset",
            }:
                text = _UNKNOWN
            yield prop, text, declaration.important, index


def _candidates(root, stylesheets, resource):
    soup, reverse = source_node_map(root, resource=resource)
    matched = {}
    for order, rule in enumerate(source_rules(stylesheets, resource=resource)):
        if not semantic_properties(rule.declarations):
            continue
        selected = {}
        for selector in rule.selectors:
            if selector.pseudo is not None:
                continue
            if not source_selector_applies(soup, selector, resource=resource, semantic=True):
                continue
            for tag in soup.select(selector.base):
                node = reverse[id(tag)]
                selected[node] = max(selected.get(node, (0, 0, 0)), selector.specificity)
        for node, specificity in selected.items():
            values = matched.setdefault(node, {})
            for prop, text, important, index in _expanded(rule.declarations):
                rank = (important, False, specificity, order, index)
                values.setdefault(prop, []).append((rank, text, bool(rule.conditions)))
    for node in reverse.values():
        inline = node.get("style")
        if inline is None:
            continue
        normalize_inline(inline, (), resource=resource)
        declarations = source_declarations(
            tinycss2.parse_component_value_list(inline), resource=resource
        )
        for prop, text, important, index in _expanded(declarations):
            rank = (important, True, (0, 0, 0), 0, index)
            matched.setdefault(node, {}).setdefault(prop, []).append((rank, text, False))
    return matched


def _native_text(node, prop):
    name = node.tag.rsplit("}", 1)[-1].lower()
    if prop == "font-weight":
        return "bolder" if name in {"b", "strong"} else "inherit"
    if prop == "font-style":
        return "italic" if name in {"i", "em"} else "inherit"
    return "super" if name == "sup" else "sub" if name == "sub" else "initial"


def _computed_property(node, prop, parent_values, candidates):
    unconditional = [item for item in candidates if not item[2]]
    winner = max(unconditional, default=None, key=lambda item: item[0])
    texts = [_native_text(node, prop) if winner is None else winner[1]]
    texts.extend(
        item[1] for item in candidates if item[2] and (winner is None or item[0] > winner[0])
    )
    pairs = {(parent, _value(prop, text, parent)) for parent in parent_values for text in texts}
    return frozenset(value for _, value in pairs), pairs


def _roles(computed, weight_pairs, resource):
    if any(_UNKNOWN in values for values in computed.values()):
        raise _fail(resource)
    bold = {value > parent for parent, value in weight_pairs if parent != _UNKNOWN}
    if any(parent == _UNKNOWN for parent, _ in weight_pairs) or len(bold) != 1:
        raise _fail(resource)
    italic = {value != "normal" for value in computed["font-style"]}
    vertical = {
        "superscript" if value == "super" else "subscript" if value == "sub" else None
        for value in computed["vertical-align"]
    }
    if len(italic) != 1 or len(vertical) != 1:
        raise _fail(resource)
    result = []
    if True in bold:
        result.append("bold")
    if True in italic:
        result.append("italic")
    result.extend(value for value in vertical if value is not None)
    return tuple(result)


def _source_styles(
    root: etree._Element,
    nodes: Sequence[etree._Element],
    stylesheets: Mapping[str, bytes],
    *,
    resource: str,
):
    source_specificity_bound(
        project_nonmatching_animations(stylesheets, root, resource=resource), resource=resource
    )
    reject_inline_animations(root, resource=resource)
    candidates = _candidates(root, stylesheets, resource)
    requested = set(nodes)
    relevant = {ancestor for node in nodes for ancestor in (node, *node.iterancestors())}
    actual_nodes = {node for node in root.iter() if isinstance(node.tag, str)}
    if not requested <= actual_nodes:
        raise ThemeError("theme_css", "invalid_markup", resource=resource)
    initial = {prop: frozenset((value,)) for prop, value in _INITIAL.items()}
    computed, roles, requirements = {}, {}, {}
    for node in root.iter():
        if node not in relevant or not isinstance(node.tag, str):
            continue
        parent = computed.get(node.getparent(), initial)
        values = {}
        weight_pairs = set()
        style_pairs = set()
        for prop in _INITIAL:
            values[prop], pairs = _computed_property(
                node, prop, parent[prop], candidates.get(node, {}).get(prop, [])
            )
            if prop == "font-weight":
                weight_pairs = pairs
            elif prop == "font-style":
                style_pairs = pairs
        computed[node] = values
        if node in requested:
            roles[node] = _roles(values, weight_pairs, resource)
            decisions = set()
            for inherited_weight, own_weight in weight_pairs:
                for inherited_style, own_style in style_pairs:
                    decisions.add(own_weight != inherited_weight or own_style != inherited_style)
            if len(decisions) != 1:
                raise _fail(resource)
            explicit_vertical = bool(candidates.get(node, {}).get("vertical-align"))
            native_vertical = node.tag.rsplit("}", 1)[-1].lower() in {"sub", "sup"}
            requirements[node] = True in decisions or explicit_vertical or native_vertical
    return tuple(roles[node] for node in nodes), tuple(requirements[node] for node in nodes)


def source_semantic_styles(
    root: etree._Element,
    nodes: Sequence[etree._Element],
    stylesheets: Mapping[str, bytes],
    *,
    resource: str,
) -> tuple[tuple[str, ...], ...]:
    """只报告所有可行级联结果都一致的必要语义，不猜条件是否生效。"""
    return _source_styles(root, nodes, stylesheets, resource=resource)[0]


def source_style_requirements(
    root: etree._Element,
    nodes: Sequence[etree._Element],
    stylesheets: Mapping[str, bytes],
    *,
    resource: str,
) -> tuple[bool, ...]:
    """保留改变或取消继承样式的节点，不把 normal 重置当普通装饰。"""
    return _source_styles(root, nodes, stylesheets, resource=resource)[1]


def decoration_declarations(declarations: tuple[CssDeclaration, ...]):
    """展开首字装饰属性与会覆盖它们的 font/all 重置。"""
    for index, declaration in enumerate(declarations):
        properties = (
            ("font-size", "float", "initial-letter")
            if declaration.name == "all"
            else ("font-size",)
            if declaration.name == "font"
            else (declaration.name,)
        )
        for prop in properties:
            if prop not in {"font-size", "float", "initial-letter"}:
                continue
            text = declaration.value.strip().lower() if declaration.name == prop else _UNKNOWN
            yield prop, text, declaration.important, index


def dropcap_style_proven(declarations: tuple[CssDeclaration, ...]) -> bool:
    """仅证明相对放大、浮动或 initial-letter，不靠字体加重猜装饰。"""
    winners = {}
    for prop, text, important, index in decoration_declarations(declarations):
        rank = (important, index)
        prior = winners.get(prop)
        if prior is None or rank > prior[0]:
            winners[prop] = (rank, text)
    size = winners.get("font-size", (None, ""))[1]
    match = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)(em|%)", size)
    enlarged = bool(match and float(match.group(1)) > (1 if match.group(2) == "em" else 100))
    initial = winners.get("initial-letter", (None, ""))[1]
    match = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)(?:\s+\d+)?", initial)
    enlarged = enlarged or bool(match and float(match.group(1)) > 1)
    floating = winners.get("float", (None, ""))[1] in {
        "left",
        "right",
        "inline-start",
        "inline-end",
    }
    return enlarged or floating


def _decoration_error(resource):
    return ThemeError("theme_css", "ambiguous_source_decoration", resource=resource)


def _dropcap_entries(rules, root, resource):
    soup = source_node_map(root, resource=resource)[0] if root is not None else None
    entries = []
    for order, rule in enumerate(rules):
        declarations = tuple(decoration_declarations(rule.declarations))
        if not declarations:
            continue
        for selector in rule.selectors:
            if selector.pseudo != "first-letter":
                continue
            matched = None
            if soup is not None:
                try:
                    if not source_selector_applies(
                        soup, selector, resource=resource, semantic=True
                    ):
                        continue
                    matched = frozenset(id(node) for node in soup.select(selector.base))
                except ThemeError:
                    raise _decoration_error(resource) from None
            entries.append((order, rule, selector, declarations, matched))
    return entries


def _prove_dropcap_candidate(candidate, entries, resource):
    _, candidate_rule, candidate_selector, _, matched = candidate
    nodes = (None,) if matched is None else matched
    for node in nodes:
        values = {prop: [] for prop in ("font-size", "float", "initial-letter")}
        for order, rule, selector, declarations, selected in entries:
            if selected is not None and node not in selected:
                continue
            forced = set(rule.conditions) <= set(candidate_rule.conditions) and (
                matched is not None or selector.base == candidate_selector.base
            )
            for prop, text, important, index in declarations:
                rank = (important, selector.specificity, order, index)
                values[prop].append((rank, text, forced))
        proven = False
        for prop, candidates in values.items():
            certain = [item for item in candidates if item[2]]
            winner = max(certain, default=None, key=lambda item: item[0])
            possible = [_UNKNOWN if winner is None else winner[1]]
            possible.extend(
                item[1]
                for item in candidates
                if not item[2] and (winner is None or item[0] > winner[0])
            )
            proven = proven or all(
                dropcap_style_proven((CssDeclaration(prop, text),)) for text in possible
            )
        if not proven:
            raise _decoration_error(resource)


def chinese_dropcap_overrides(
    stylesheets: Mapping[str, bytes], *, resource: str, root: etree._Element | None = None
) -> str:
    """只取消已证明的中文首字装饰，保留原条件和首行规则。"""
    if root is None:
        source_specificity_bound(stylesheets, resource=resource)
    else:
        source_specificity_bound(
            project_nonmatching_animations(stylesheets, root, resource=resource), resource=resource
        )
        reject_inline_animations(root, resource=resource)
    output = []
    declarations = (
        "all:unset!important;"
        "font-size:inherit!important;font-style:inherit!important;font-weight:inherit!important;"
        "line-height:inherit!important;float:none!important;initial-letter:normal!important"
    )
    rules = source_rules(stylesheets, resource=resource)
    entries = _dropcap_entries(rules, root, resource)
    for candidate in entries:
        _, rule, selector, _, matched = candidate
        if not dropcap_style_proven(rule.declarations):
            continue
        if matched is not None and not matched:
            continue
        _prove_dropcap_candidate(candidate, entries, resource)
        if not supported_source_selector(selector, resource=resource, semantic=False):
            continue
        value = f"{selector.base}:lang(zh):not(.tn-source):not(.tn-source *)::first-letter{{{declarations}}}"
        for condition in reversed(rule.conditions):
            value = f"{condition}{{{value}}}"
        output.append(value)
    return "\n".join(dict.fromkeys(output))
