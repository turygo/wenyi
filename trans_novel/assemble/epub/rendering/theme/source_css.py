from __future__ import annotations

import zipfile
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

import tinycss2
from bs4 import BeautifulSoup
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.cascade import (
    normalize_inline,
    source_specificity_bound,
)
from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.source_applicability import (
    project_nonmatching_animations,
    reject_inline_animations,
)
from trans_novel.assemble.epub.rendering.theme.source_selectors import (
    SourceRule,
    semantic_properties,
    source_node_map,
    source_rules,
    source_selector_applies,
)
from trans_novel.assemble.epub.rendering.theme.source_semantics import (
    chinese_dropcap_overrides,
    source_semantic_styles,
    source_style_requirements,
)
from trans_novel.epub.archive import ZipSafetyError, read_member, safe_name
from trans_novel.epub.navigation import resolve_epub_href

_IGNORED_TOKENS = {"whitespace", "comment"}

__all__ = [
    "chinese_dropcap_overrides",
    "collect_source_stylesheets",
    "source_semantic_styles",
    "source_style_evidence",
    "source_style_requirements",
]


def _fail(resource: str, detail: str = "source_import_failed") -> ThemeError:
    return ThemeError("theme_css", detail, resource=resource)


def _nested_tokens(token: object) -> list[object] | None:
    nested = getattr(token, "arguments", None)
    if nested is None:
        nested = getattr(token, "content", None)
    return nested


def _has_error(tokens: list[object]) -> bool:
    for token in tokens:
        if getattr(token, "type", None) == "error":
            return True
        nested = _nested_tokens(token)
        if nested is not None and _has_error(nested):
            return True
    return False


def _has_layer(tokens: list[object]) -> bool:
    for token in tokens:
        if (
            getattr(token, "type", None) == "ident"
            and getattr(token, "lower_value", None) == "layer"
        ) or (
            getattr(token, "type", None) == "function"
            and getattr(token, "lower_name", None) == "layer"
        ):
            return True
        nested = _nested_tokens(token)
        if nested is not None and _has_layer(nested):
            return True
    return False


def _media_wrappers(value: str) -> tuple[str, ...]:
    tokens = tinycss2.parse_component_value_list(value)
    if _has_error(tokens):
        raise ValueError
    media = tinycss2.serialize(tokens).strip()
    return () if not media or media.lower() == "all" else (f"@media {media}",)


def _import_wrappers(conditions: list[object]) -> tuple[str, ...]:
    wrappers = []
    significant = [
        token for token in conditions if getattr(token, "type", None) not in _IGNORED_TOKENS
    ]
    if significant and getattr(significant[0], "lower_name", None) == "supports":
        support = significant[0]
        arguments = [
            token
            for token in support.arguments
            if getattr(token, "type", None) not in _IGNORED_TOKENS
        ]
        if not arguments:
            raise ValueError
        text = tinycss2.serialize(support.arguments).strip()
        if len(arguments) != 1 or getattr(arguments[0], "type", None) != "function":
            text = f"({text})"
        wrappers.append(f"@supports {text}")
        conditions = conditions[conditions.index(support) + 1 :]
    wrappers.extend(_media_wrappers(tinycss2.serialize(conditions)))
    return tuple(wrappers)


def _import_href(prelude: list[object], resource: str) -> tuple[str, tuple[str, ...]]:
    if _has_error(prelude):
        raise ValueError
    tokens = [token for token in prelude if getattr(token, "type", None) not in _IGNORED_TOKENS]
    if not tokens:
        raise ValueError
    source = tokens[0]
    conditions = prelude[prelude.index(source) + 1 :]
    token_type = getattr(source, "type", None)
    if token_type in {"string", "url"}:
        href = source.value
    elif token_type == "function" and getattr(source, "lower_name", None) == "url":
        arguments = [
            token
            for token in source.arguments
            if getattr(token, "type", None) not in _IGNORED_TOKENS
        ]
        if len(arguments) != 1 or getattr(arguments[0], "type", None) != "string":
            raise ValueError
        href = arguments[0].value
    else:
        raise ValueError
    if not href:
        raise ValueError
    if _has_layer(conditions):
        raise _fail(resource, "source_layers")
    return href, _import_wrappers(conditions)


def _resolve_stylesheet(base_path: str, raw_href: str) -> str:
    if not raw_href:
        raise ValueError
    parsed = urlsplit(raw_href)
    resolved = resolve_epub_href(base_path, raw_href)
    if (
        parsed.query
        or parsed.fragment
        or resolved.external
        or not safe_name(resolved.resource_href)
    ):
        raise ValueError
    return resolved.resource_href


def _is_css(node: etree._Element) -> bool:
    media_type = node.get("type")
    return media_type is None or not media_type.strip() or media_type.strip().lower() == "text/css"


def _stylesheet_nodes(
    root: etree._Element, resource: str
) -> Iterator[tuple[etree._Element, int | None]]:
    style_ordinal = 0
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        name = node.tag.rsplit("}", 1)[-1].lower()
        if "{http://www.w3.org/XML/1998/namespace}base" in node.attrib or (
            name == "base" and node.get("href") is not None
        ):
            raise _fail(resource, "source_base_unsupported")
        if name == "style":
            ordinal = style_ordinal
            style_ordinal += 1
            if _is_css(node):
                yield node, ordinal
        elif (
            name == "link"
            and _is_css(node)
            and "stylesheet" in {token.lower() for token in (node.get("rel") or "").split()}
        ):
            yield node, None


def collect_source_stylesheets(
    archive: zipfile.ZipFile,
    root: etree._Element,
    resource_href: str,
) -> dict[str, bytes]:
    """按原顺序展开每次导入，保留条件并返回 UTF-8 副本。"""
    collected: dict[str, bytes] = {}
    active: set[str] = set()
    occurrences: dict[str, int] = {}

    def collect_bytes(
        data: bytes,
        *,
        key: str,
        base_path: str,
        protocol_encoding: str | None = None,
        environment_encoding: Any = None,
        wrappers: tuple[str, ...] = (),
    ) -> None:
        rules, encoding = tinycss2.parse_stylesheet_bytes(
            data,
            protocol_encoding=protocol_encoding,
            environment_encoding=environment_encoding,
            skip_comments=False,
            skip_whitespace=False,
        )
        retained: list[object] = []
        for rule in rules:
            if rule.type == "error":
                raise _fail(resource_href)
            if getattr(rule, "type", None) == "at-rule":
                keyword = getattr(rule, "lower_at_keyword", None)
                if keyword == "charset":
                    continue
                if keyword == "import":
                    href, conditions = _import_href(rule.prelude, resource_href)
                    collect_file(
                        _resolve_stylesheet(base_path, href),
                        environment_encoding=encoding,
                        wrappers=(*wrappers, *conditions),
                    )
                    continue
            retained.append(rule)
        css = tinycss2.serialize(retained)
        for condition in reversed(wrappers):
            css = f"{condition}{{{css}}}"
        occurrences[key] = occurrences.get(key, 0) + 1
        occurrence = occurrences[key]
        actual_key = key if occurrence == 1 else f"{key}#occurrence:{occurrence}"
        collected[actual_key] = css.encode("utf-8")

    def collect_file(
        member: str, *, environment_encoding: Any = None, wrappers: tuple[str, ...] = ()
    ) -> None:
        if member in active:
            raise _fail(resource_href, "source_import_cycle")
        active.add(member)
        try:
            info = archive.getinfo(member)
            collect_bytes(
                read_member(archive, info),
                key=f"file:{member}",
                base_path=member,
                environment_encoding=environment_encoding,
                wrappers=wrappers,
            )
        finally:
            active.remove(member)

    try:
        if not safe_name(resource_href):
            raise ValueError
        for node, ordinal in _stylesheet_nodes(root, resource_href):
            if ordinal is not None:
                collect_bytes(
                    "".join(node.itertext()).encode("utf-8"),
                    key=f"inline:{ordinal}",
                    base_path=resource_href,
                    protocol_encoding="utf-8",
                    wrappers=_media_wrappers(node.get("media") or ""),
                )
            else:
                collect_file(
                    _resolve_stylesheet(resource_href, node.get("href") or ""),
                    wrappers=_media_wrappers(node.get("media") or ""),
                )
    except ThemeError:
        raise
    except (
        KeyError,
        OSError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        zipfile.BadZipFile,
        ZipSafetyError,
    ):
        raise _fail(resource_href) from None
    return collected


def _wrap_evidence(selector: str, declarations: str, media: tuple[str, ...]) -> str:
    value = f"{selector}{{{declarations}}}"
    for condition in reversed(media):
        value = f"{condition}{{{value}}}"
    return value


def _rule_evidence(
    rule: SourceRule,
    soup: BeautifulSoup,
    reverse: Mapping[int, etree._Element],
    *,
    resource: str,
    preserve_rule_text: bool,
) -> list[tuple[frozenset[etree._Element], str]]:
    declarations = (
        rule.declaration_text
        if preserve_rule_text
        else ";".join(
            f"{item.name}:{item.value}{' !important' if item.important else ''}"
            for item in rule.declarations
        )
    )
    matched = []
    all_admitted = True
    for selector in rule.selectors:
        if selector.pseudo is not None or not source_selector_applies(
            soup,
            selector,
            resource=resource,
            semantic=bool(semantic_properties(rule.declarations)),
        ):
            all_admitted = False
            continue
        try:
            selected = frozenset(
                reverse[id(tag)] for tag in soup.select(selector.base) if id(tag) in reverse
            )
            if selected:
                matched.append(
                    (selected, _wrap_evidence(selector.selector, declarations, rule.conditions))
                )
        except (RecursionError, NotImplementedError, TypeError, ValueError):
            raise _fail(resource, "unsupported_source_selector") from None
    if preserve_rule_text and all_admitted and matched:
        # 整条普通规则只输出一次，但不将拒绝的动态分支或伪元素带回证据。
        union = frozenset(node for selected, _ in matched for node in selected)
        return [(union, _wrap_evidence(rule.selector_text, declarations, rule.conditions))]
    return matched


def source_style_evidence(
    root: etree._Element,
    nodes: Sequence[etree._Element],
    stylesheets: Mapping[str, bytes],
    *,
    resource: str,
    preserve_rule_text: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """返回每个节点及其祖先匹配的原始 CSS 证据，并保留条件规则上下文。"""
    source_specificity_bound(
        project_nonmatching_animations(stylesheets, root, resource=resource), resource=resource
    )
    reject_inline_animations(root, resource=resource)
    try:
        soup, reverse = source_node_map(root, resource=resource)
    except (TypeError, ValueError, etree.LxmlError):
        raise _fail(resource, "invalid_markup") from None
    matched: list[tuple[frozenset[etree._Element], str]] = []
    for rule in source_rules(stylesheets, resource=resource):
        matched.extend(
            _rule_evidence(
                rule,
                soup,
                reverse,
                resource=resource,
                preserve_rule_text=preserve_rule_text,
            )
        )

    node_set = set(reverse.values())
    output: list[tuple[str, ...]] = []
    for node in nodes:
        if node not in node_set:
            raise _fail(resource, "invalid_markup")
        ancestry = (node, *node.iterancestors())
        evidence = [
            value
            for selected, value in matched
            if any(ancestor in selected for ancestor in ancestry)
        ]
        for ancestor in reversed(ancestry):
            inline = ancestor.get("style")
            if inline is None:
                continue
            normalize_inline(inline, (), resource=resource)
            name = ancestor.tag.rsplit("}", 1)[-1].lower()
            evidence.append(f"@inline {name}{{{inline}}}")
        output.append(tuple(evidence))
    return tuple(output)
