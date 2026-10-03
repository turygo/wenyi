"""为可信源内联清单补充 CSS 证据，识别已授权取消的首字装饰。"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import defaultdict

import tinycss2
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.css import CssDeclaration
from trans_novel.assemble.epub.rendering.theme.source_css import (
    collect_source_stylesheets,
    source_semantic_styles,
    source_style_evidence,
    source_style_requirements,
)
from trans_novel.assemble.epub.rendering.theme.source_selectors import (
    source_declarations,
    source_selector_branches,
)
from trans_novel.assemble.epub.rendering.theme.source_semantics import (
    decoration_declarations,
    dropcap_style_proven,
)
from trans_novel.epub.archive import preflight_zip, read_member
from trans_novel.epub.markup import resource_parser
from trans_novel.epub.richtext import InlineMark, RichSource, tag_semantics
from trans_novel.ingest.models import Chapter


def _resolve(root: etree._Element, path: tuple[int, ...]) -> etree._Element:
    """使用源元素坐标，不把注释节点计入定位。"""
    for index in path:
        children = [child for child in root if isinstance(child.tag, str)]
        if not 0 <= index < len(children):
            raise ValueError("rich style source locator mismatch")
        root = children[index]
    return root


def _added_dropcap_style(evidence: list[str]) -> bool:
    """仅计算新增无条件装饰证据的获胜属性，不以加重本身判断装饰。"""
    winners = {}
    for order, value in enumerate(evidence):
        if value.startswith(("@media", "@supports")):
            continue
        if "{" not in value or not value.endswith("}"):
            continue
        selector, body = value.split("{", 1)
        inline = selector.startswith("@inline ")
        specificity = (
            (0, 0, 0)
            if inline
            else max(
                item.specificity
                for item in source_selector_branches(
                    tinycss2.parse_component_value_list(selector), resource="<style>"
                )
            )
        )
        declarations = source_declarations(
            tinycss2.parse_component_value_list(body[:-1]), resource="<style>"
        )
        for prop, text, important, index in decoration_declarations(declarations):
            rank = (important, inline, specificity, order, index)
            if prop not in winners or rank > winners[prop][0]:
                winners[prop] = (rank, text)
    return dropcap_style_proven(
        tuple(CssDeclaration(prop, text) for prop, (_, text) in winners.items())
    )


def _opening_letter(mark: InlineMark, sources: list[RichSource]) -> bool:
    """首字可带开引号或独立成词，但不能把完整短语或后续字母当装饰。"""
    if mark.kind != "style" or tag_semantics(mark.tag):
        return False
    if re.fullmatch(r"[“‘\"']?[A-Za-z]", mark.source_text) is None:
        return False
    characters = [
        (char, run.marks) for source in sources for run in source.runs for char in run.text
    ]
    opening_quotes = "“‘\"'"
    for index, (char, marks) in enumerate(characters):
        if char.isspace() or char in opening_quotes:
            continue
        remainder = "".join(value for value, _ in characters[index + 1 :]).strip()
        return char == mark.source_text[-1] and mark.id in marks and bool(remainder)
    return False


def enrich_rich_sources(source_path: str, chapters: list[Chapter]) -> None:
    """批量读取每个资源的样式证据，不改变源槽位几何或原书文件。"""
    by_resource: dict[str, list] = defaultdict(list)
    for chapter in chapters:
        for segment in chapter.segments:
            state = segment.epub_state
            if state is not None and state.rich_source is not None and state.rich_source.marks:
                by_resource[state.resource_href].append(state)
    if not by_resource:
        return
    with zipfile.ZipFile(source_path) as archive:
        preflight_zip(archive)
        for href, states in by_resource.items():
            data = read_member(archive, archive.getinfo(href))
            digest = hashlib.sha256(data).hexdigest()
            tree, mode, _ = resource_parser(data)
            root = tree.getroot()
            by_block: dict[tuple[int, ...], list[RichSource]] = defaultdict(list)
            records: list[tuple[InlineMark, etree._Element, tuple[int, ...]]] = []
            nodes: list[etree._Element] = []
            for state in states:
                if state.resource_sha256 != digest or state.parse_mode != mode:
                    raise ValueError("rich style source resource mismatch")
                block = _resolve(root, state.block_path)
                actual = hashlib.sha256(
                    etree.tostring(block, encoding="utf-8", with_tail=False)
                ).hexdigest()
                if actual != state.block_fingerprint:
                    raise ValueError("rich style source block mismatch")
                by_block[state.block_path].append(state.rich_source)
                for mark in state.rich_source.marks:
                    node = _resolve(block, mark.path)
                    if node.tag != mark.tag or dict(node.attrib) != mark.attributes:
                        raise ValueError("rich style source descriptor mismatch")
                    records.append((mark, node, state.block_path))
                    for candidate in (node, node.getparent()):
                        if candidate is not None and candidate not in nodes:
                            nodes.append(candidate)
            stylesheets = collect_source_stylesheets(archive, root, href)
            evidence = source_style_evidence(root, nodes, stylesheets, resource=href)
            by_node = dict(zip(nodes, evidence, strict=True))
            semantics = dict(
                zip(
                    nodes,
                    source_semantic_styles(root, nodes, stylesheets, resource=href),
                    strict=True,
                )
            )
            requirements = dict(
                zip(
                    nodes,
                    source_style_requirements(root, nodes, stylesheets, resource=href),
                    strict=True,
                )
            )
            for mark, node, block_path in records:
                mark.style_evidence = list(by_node[node])
                mark.semantics = tuple(dict.fromkeys((*tag_semantics(mark.tag), *semantics[node])))
                mark.required = bool(tag_semantics(mark.tag)) or requirements[node]
                parent_evidence = set(by_node.get(node.getparent(), ()))
                added = [value for value in mark.style_evidence if value not in parent_evidence]
                block_sources = by_block[block_path]
                if _opening_letter(mark, block_sources) and _added_dropcap_style(added):
                    mark.kind = "decoration"
                    mark.semantics = ()
                    mark.required = False


__all__ = ["enrich_rich_sources"]
