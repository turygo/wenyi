"""源选择器分支、终端伪元素和可确定的选择器优先级。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import soupsieve
import tinycss2
from bs4 import BeautifulSoup
from bs4.element import Tag
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.css import CssDeclaration, declaration_properties

_LEGACY_ELEMENTS = {"before", "after", "first-letter", "first-line"}
_DYNAMIC = {
    "active",
    "any-link",
    "checked",
    "disabled",
    "enabled",
    "focus",
    "focus-visible",
    "focus-within",
    "hover",
    "link",
    "target",
    "visited",
}
_STATIC = {
    "root",
    "scope",
    "empty",
    "first-child",
    "last-child",
    "only-child",
    "nth-child",
    "nth-last-child",
    "first-of-type",
    "last-of-type",
    "only-of-type",
    "nth-of-type",
    "nth-last-of-type",
    "not",
    "is",
    "where",
    "has",
    "lang",
    "dir",
}
SEMANTIC_PROPERTIES = frozenset({"font-weight", "font-style", "vertical-align"})


@dataclass(frozen=True)
class SourceSelector:
    selector: str
    base: str
    pseudo: str | None
    specificity: tuple[int, int, int]


@dataclass(frozen=True)
class SourceRule:
    selectors: tuple[SourceSelector, ...]
    declarations: tuple[CssDeclaration, ...]
    conditions: tuple[str, ...] = ()
    selector_text: str = ""
    declaration_text: str = ""


def _fail(detail: str, resource: str) -> ThemeError:
    return ThemeError("theme_css", detail, resource=resource)


def _clean(tokens: list) -> list:
    return [token for token in tokens if token.type not in {"whitespace", "comment"}]


def _check_tokens(tokens: list, resource: str) -> None:
    for token in tokens:
        if token.type == "error":
            raise _fail("invalid_source_css", resource)
        if token.type == "literal" and token.value == "|":
            raise _fail("unsupported_source_selector", resource)
        nested = getattr(token, "arguments", getattr(token, "content", None))
        if nested is not None:
            _check_tokens(nested, resource)


def _branches(tokens: list, resource: str) -> list[list]:
    result, current = [], []
    for token in tokens:
        if token.type == "literal" and token.value == ",":
            if not _clean(current):
                raise _fail("invalid_source_css", resource)
            result.append(current)
            current = []
        else:
            current.append(token)
    if not _clean(current):
        raise _fail("invalid_source_css", resource)
    return [*result, current]


def _specificity(tokens: list, resource: str) -> tuple[int, int, int]:
    tokens = _clean(tokens)
    total = [0, 0, 0]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "hash" and token.is_identifier:
            total[0] += 1
        elif token.type == "[] block":
            total[1] += 1
        elif token.type == "literal" and token.value == ".":
            total[1] += 1
            index += 1
        elif token.type == "literal" and token.value == ":":
            index += 1
            if index >= len(tokens):
                raise _fail("invalid_source_css", resource)
            pseudo = tokens[index]
            name = getattr(pseudo, "lower_name", getattr(pseudo, "lower_value", ""))
            if name in {"is", "not", "has", "where"} and pseudo.type == "function":
                weights = [
                    _specificity(branch, resource)
                    for branch in _branches(pseudo.arguments, resource)
                ]
                weight = (0, 0, 0) if name == "where" else max(weights)
                total = [value + extra for value, extra in zip(total, weight, strict=True)]
            else:
                total[1] += 1
                if pseudo.type == "function" and name.startswith("nth-"):
                    args = pseudo.arguments
                    for offset, item in enumerate(args):
                        if item.type == "ident" and item.lower_value == "of":
                            weight = max(
                                _specificity(branch, resource)
                                for branch in _branches(args[offset + 1 :], resource)
                            )
                            total = [
                                value + extra for value, extra in zip(total, weight, strict=True)
                            ]
                            break
        elif token.type == "ident":
            total[2] += 1
        index += 1
    return tuple(total)


def source_selector_branches(tokens: list, *, resource: str) -> tuple[SourceSelector, ...]:
    """按顶层逗号拆分，不把属性或函数中的逗号误当分支。"""
    _check_tokens(tokens, resource)
    result = []
    for branch in _branches(tokens, resource):
        significant = _clean(branch)
        pseudo, cut = None, None
        for index, token in enumerate(significant):
            if token.type != "literal" or token.value != ":":
                continue
            offset = index + 1
            doubled = (
                offset < len(significant)
                and significant[offset].type == "literal"
                and significant[offset].value == ":"
            )
            if doubled:
                offset += 1
            if offset >= len(significant):
                raise _fail("invalid_source_css", resource)
            item = significant[offset]
            name = getattr(item, "lower_name", getattr(item, "lower_value", ""))
            if doubled or name in _LEGACY_ELEMENTS:
                if offset != len(significant) - 1 or item.type not in {"ident", "function"}:
                    raise _fail("unsupported_source_selector", resource)
                pseudo, cut = name, index
                break
        selector = tinycss2.serialize(branch).strip()
        base = selector if cut is None else tinycss2.serialize(significant[:cut]).strip() or "*"
        # 删除空白会改变后代组合器，因此基底从原始 token 截取。
        if cut is not None:
            marker = significant[cut]
            base = tinycss2.serialize(branch[: branch.index(marker)]).strip() or "*"
        weight = _specificity(tinycss2.parse_component_value_list(base), resource)
        if pseudo is not None:
            weight = (weight[0], weight[1], weight[2] + 1)
        result.append(SourceSelector(selector, base, pseudo, weight))
    return tuple(result)


def semantic_properties(declarations: tuple[CssDeclaration, ...]) -> frozenset[str]:
    """包含 font 和 all 重置，但不把无关排版声明当语义。"""
    result = set()
    for declaration in declarations:
        result.update(
            SEMANTIC_PROPERTIES
            if declaration.name == "all"
            else declaration_properties(declaration.name) & SEMANTIC_PROPERTIES
        )
    return frozenset(result)


def _unknown_pseudo(tokens: list) -> bool:
    tokens = _clean(tokens)
    for index, token in enumerate(tokens):
        if token.type != "literal" or token.value != ":" or index + 1 >= len(tokens):
            continue
        pseudo = tokens[index + 1]
        name = getattr(pseudo, "lower_name", getattr(pseudo, "lower_value", ""))
        if name not in _STATIC:
            return True
        if pseudo.type == "function" and _unknown_pseudo(pseudo.arguments):
            return True
    return False


def _syntax_view(tokens: list) -> str:
    """未知伪类仅在语法探针中替成合法占位，不用于正文匹配。"""
    pieces = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "literal" and token.value == ":" and index + 1 < len(tokens):
            pseudo = tokens[index + 1]
            name = getattr(pseudo, "lower_name", getattr(pseudo, "lower_value", ""))
            if name not in _STATIC:
                pieces.append(":not(*)")
                index += 2
                continue
            if pseudo.type == "function" and name in {"not", "is", "where", "has"}:
                pieces.append(f":{name}({_syntax_view(pseudo.arguments)})")
                index += 2
                continue
        pieces.append(tinycss2.serialize([token]))
        index += 1
    return "".join(pieces)


def supported_source_selector(selector: SourceSelector, *, resource: str, semantic: bool) -> bool:
    """必要语义不接受依赖交互状态或未知伪类的普通选择器。"""
    unknown = _unknown_pseudo(tinycss2.parse_component_value_list(selector.base))
    if unknown and semantic:
        raise _fail("unsupported_source_semantic_selector", resource)
    try:
        soupsieve.compile(selector.base)
    except (soupsieve.SelectorSyntaxError, NotImplementedError, ValueError):
        if unknown and not semantic:
            try:
                soupsieve.compile(_syntax_view(tinycss2.parse_component_value_list(selector.base)))
            except (soupsieve.SelectorSyntaxError, NotImplementedError, ValueError):
                raise _fail("unsupported_source_selector", resource) from None
            return False
        raise _fail("unsupported_source_selector", resource) from None
    return not unknown


def _dynamic_upper_bound(tokens: list) -> str | None:
    """只给平面已知动态伪类构造匹配上界，不把否定或未知函数当静态事实。"""
    pieces = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type == "literal" and token.value == ":" and index + 1 < len(tokens):
            pseudo = tokens[index + 1]
            name = getattr(pseudo, "lower_name", getattr(pseudo, "lower_value", ""))
            if pseudo.type == "ident" and name in _DYNAMIC:
                pieces.append(":is(*)")
                index += 2
                continue
            if name not in _STATIC or (
                pseudo.type == "function" and _unknown_pseudo(pseudo.arguments)
            ):
                return None
        pieces.append(tinycss2.serialize([token]))
        index += 1
    return "".join(pieces)


def source_selector_may_match(
    soup: BeautifulSoup, selector: SourceSelector, *, resource: str
) -> bool:
    """返回保守匹配上界，未知函数或否定条件不能被误判为空集。"""
    if supported_source_selector(selector, resource=resource, semantic=False):
        return bool(soup.select(selector.base))
    upper = _dynamic_upper_bound(tinycss2.parse_component_value_list(selector.base))
    return upper is None or bool(soup.select(upper))


def source_selector_applies(
    soup: BeautifulSoup, selector: SourceSelector, *, resource: str, semantic: bool
) -> bool:
    """动态样式仅在潜在匹配为空时跳过，有可能影响正文仍明确拒绝。"""
    if supported_source_selector(selector, resource=resource, semantic=False):
        return True
    if not semantic:
        return False
    if not source_selector_may_match(soup, selector, resource=resource):
        return False
    raise _fail("unsupported_source_semantic_selector", resource)


def source_declarations(tokens: list, *, resource: str) -> tuple[CssDeclaration, ...]:
    result = []
    for item in tinycss2.parse_declaration_list(tokens, skip_whitespace=True, skip_comments=True):
        if item.type != "declaration":
            raise _fail("invalid_source_css", resource)
        _check_tokens(item.value, resource)
        result.append(
            CssDeclaration(item.lower_name, tinycss2.serialize(item.value).strip(), item.important)
        )
    return tuple(result)


def _rules(rules: list, resource: str, conditions: tuple[str, ...] = ()) -> list[SourceRule]:
    result = []
    for rule in rules:
        if rule.type == "qualified-rule":
            result.append(
                SourceRule(
                    source_selector_branches(rule.prelude, resource=resource),
                    source_declarations(rule.content, resource=resource),
                    conditions,
                    tinycss2.serialize(rule.prelude).strip(),
                    tinycss2.serialize(rule.content).strip(),
                )
            )
        elif (
            rule.type == "at-rule"
            and rule.lower_at_keyword in {"media", "supports"}
            and rule.content is not None
        ):
            condition = f"@{rule.lower_at_keyword} {tinycss2.serialize(rule.prelude).strip()}"
            result.extend(
                _rules(
                    tinycss2.parse_rule_list(
                        rule.content, skip_whitespace=True, skip_comments=True
                    ),
                    resource,
                    (*conditions, condition),
                )
            )
        elif rule.type == "error":
            raise _fail("invalid_source_css", resource)
    return result


def source_rules(stylesheets: Mapping[str, bytes], *, resource: str) -> tuple[SourceRule, ...]:
    """读取已通过源 CSS 安全门禁的普通及条件规则。"""
    result = []
    for data in stylesheets.values():
        parsed, _ = tinycss2.parse_stylesheet_bytes(data, skip_whitespace=True, skip_comments=True)
        result.extend(_rules(parsed, resource))
    return tuple(result)


def source_node_map(root: etree._Element, *, resource: str) -> tuple[BeautifulSoup, dict]:
    """证明 CSS 匹配视图的元素顺序与原书元素一致。"""
    nodes = [node for node in root.iter() if isinstance(node.tag, str)]
    # 脱离 DTD 文档上下文，防止序列化为匹配视图时凭空插入 Content-Type meta。
    soup = BeautifulSoup(etree.tostring(deepcopy(root), method="xml"), "xml")
    parsed = [node for node in soup.find_all(True) if isinstance(node, Tag)]
    if len(nodes) != len(parsed) or any(
        node.tag.rsplit("}", 1)[-1].lower() != str(tag.name).split(":")[-1].lower()
        for node, tag in zip(nodes, parsed, strict=True)
    ):
        raise _fail("invalid_markup", resource)
    return soup, {id(tag): node for node, tag in zip(nodes, parsed, strict=True)}
