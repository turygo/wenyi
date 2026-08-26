"""EPUB 读取器（纯标准库 + BeautifulSoup）。

EPUB 即一个 zip：
  META-INF/container.xml → 指向 OPF
  OPF → manifest（资源清单）+ spine（阅读顺序）

读取时先按 spine 逐个物理 XHTML 标注 Segment（锚点按物理资源序号生成，
与逻辑章号无关），再根据切片粒度自动选择 NCX/NAV 的目录层级（见
``select_boundaries``），将整书的 Segment 流切分为逻辑 Chapter。因此 Chapter 与
XHTML 不再是一对一：切章之后，每个
Segment 的 ``resource_href`` 仍记录它所属的物理资源，写回时据此按
物理文件聚合。标注模板不再随 Chapter 持久化，而是统一放进
``Document.meta["epub_resource_templates"]``（键为物理资源 href），
由 RunStore 写入独立状态文件，与频繁重写的 manifest 解耦。
"""

from __future__ import annotations

import hashlib
import io
import os
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, UnicodeDammit
from lxml import etree

from trans_novel.ingest.epub_toc import (
    nav_root_list,
    nav_toc_scopes,
    parse_toc_entries,
    resolve_epub_href,
    select_boundaries,
)
from trans_novel.ingest.models import (
    KIND_HEADING,
    KIND_TEXT,
    Chapter,
    Document,
    EpubSegmentState,
    EpubTextSlot,
    Segment,
    _slot_contract_digest,
)

_CONTAINER = "META-INF/container.xml"
_HTML_EXTS = (".xhtml", ".html", ".htm")
_BLOCK_TAGS = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "td",
    "th",
    "dt",
    "dd",
}
_BLOCK_CANDIDATE_TAGS = _BLOCK_TAGS | {"div"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_INLINE_META_KEY = "epub_inline"
_INLINE_ID_ATTR = "data-tn-inline-id"
_ATOMIC_INLINE_TAGS = {
    "audio",
    "canvas",
    "embed",
    "hr",
    "iframe",
    "img",
    "math",
    "object",
    "svg",
    "video",
}
_LINE_WRAPPER_ATTR = "data-tn-line"
_STRATEGY_SPINE_FALLBACK = "spine-fallback"

_FOOTNOTE_CLASS_TOKENS = {"sup", "super", "superscript", "sub", "subscript"}
_FOOTNOTE_HINT_ATTRS = ("id", "name", "class", "title", "aria-label")
_FOOTNOTE_HINT_RE = re.compile(
    r"(?:^|[^a-z])(?:footnote|endnote|notes|note|fn)(?=$|[^a-z])",
    re.IGNORECASE,
)
_SHORT_MARKER_RE = re.compile(r"[A-Za-z]?[-_]?\d+")


def _is_internal_footnote_link(link: Tag) -> bool:
    """仅接受带 fragment 的 EPUB 内部链接，保留原始 href 不做规范化。"""
    href = link.get("href")
    if not isinstance(href, str):
        return False
    parts = urlsplit(href)
    return not parts.scheme and not parts.netloc and bool(parts.fragment)


def _is_semantic_footnote_wrapper(node: Tag) -> bool:
    """判断节点是否显式表达上标/下标脚注语义。"""
    if node.name in {"sup", "sub"}:
        return True
    if node.name != "span":
        return False
    class_values = node.get("class", [])
    if isinstance(class_values, str):
        class_values = class_values.split()
    if isinstance(class_values, list) and any(
        token.lower() in _FOOTNOTE_CLASS_TOKENS
        for value in class_values
        if isinstance(value, str)
        for token in value.split()
    ):
        return True
    style = node.get("style")
    if not isinstance(style, str):
        return False
    for declaration in style.split(";"):
        property_name, separator, value = declaration.partition(":")
        if (
            separator
            and property_name.strip().lower() == "vertical-align"
            and value.strip().lower() in {"super", "sub"}
        ):
            return True
    return False


def _has_footnote_hint(link: Tag, wrapper: Tag) -> bool:
    """判断内部链接及其语义包装是否带有脚注标识。"""
    for node in (link, wrapper):
        for attr in _FOOTNOTE_HINT_ATTRS:
            value = node.get(attr)
            values = value if isinstance(value, list) else [value]
            if any(isinstance(item, str) and _FOOTNOTE_HINT_RE.search(item) for item in values):
                return True

    href = link.get("href")
    if isinstance(href, str):
        parts = urlsplit(href)
        if _FOOTNOTE_HINT_RE.search(parts.path) or _FOOTNOTE_HINT_RE.search(parts.fragment):
            return True
    return any(
        _SHORT_MARKER_RE.fullmatch(node.get_text(strip=True) or "") is not None
        for node in (link, wrapper)
    )


def _footnote_marker_roots(block: Tag) -> list[Tag]:
    """找出块内需要作为原子内联节点保留的脚注标记包装节点。"""
    roots: list[Tag] = []
    for link in block.find_all("a", href=True):
        if not _is_internal_footnote_link(link):
            continue
        wrapper: Tag | None = link if _is_semantic_footnote_wrapper(link) else None
        parent = link.parent
        while wrapper is None and isinstance(parent, Tag) and parent is not block:
            if _is_semantic_footnote_wrapper(parent):
                wrapper = parent
                break
            parent = parent.parent
        if wrapper is not None and _has_footnote_hint(link, wrapper):
            roots.append(wrapper)
    return roots


def _outermost_nodes(block: Tag, candidates: list[Tag]) -> list[Tag]:
    """按 DOM 顺序去重，并丢弃已包含在其他候选根中的内层节点。"""
    candidate_ids = {id(node) for node in candidates}
    roots: list[Tag] = []
    for node in block.find_all(True):
        if id(node) not in candidate_ids:
            continue
        if any(id(parent) in candidate_ids for parent in node.parents if isinstance(parent, Tag)):
            continue
        roots.append(node)
    return roots


def _inside_roots(node: Tag, roots: list[Tag]) -> bool:
    """判断节点是否位于任一已识别的原子根内。"""
    return any(node is root or any(parent is root for parent in node.parents) for root in roots)


def _preserved_inline_roots(block: Tag) -> list[Tag]:
    """返回需要原样回填的非文本节点及带语义的脚注标记。"""
    candidates = _footnote_marker_roots(block)
    for candidate in block.find_all(True):
        is_atomic = candidate.name in _ATOMIC_INLINE_TAGS
        is_empty_anchor = (
            candidate.name == "a"
            and not candidate.get_text(strip=True)
            and (candidate.has_attr("id") or candidate.has_attr("name"))
        )
        if not is_atomic and not is_empty_anchor:
            continue

        root = candidate
        parent = root.parent
        while (
            isinstance(parent, Tag)
            and parent is not block
            and parent.name not in _BLOCK_TAGS
            and not parent.get_text(strip=True)
        ):
            root = parent
            parent = root.parent
        candidates.append(root)
    return _outermost_nodes(block, candidates)


def _segment_content(block: Tag, anchor: str) -> tuple[str, dict[str, object]]:
    """提取可翻译文本，并给内联非文本节点写入稳定 ID 和位置元数据。"""
    roots = _preserved_inline_roots(block)
    root_ids = {id(node) for node in roots}
    text_parts: list[str] = []
    preserved_nodes: list[tuple[str, Tag]] = []

    def walk(parent: Tag) -> None:
        for child in parent.children:
            if isinstance(child, Tag):
                if child.name in {"rt", "rp"}:
                    # 振假名与不支持 ruby 时显示的备用括号都不是正文；
                    # 保留在模板中，但不要把 ``漢字（かんじ）`` 拆成
                    # 可翻译源文里的 ``漢字（）``。
                    continue
                if id(child) in root_ids:
                    marker = f"\ue000tn-inline-{len(preserved_nodes)}\ue001"
                    preserved_nodes.append((marker, child))
                    text_parts.append(marker)
                else:
                    walk(child)
            elif isinstance(child, NavigableString) and not isinstance(child, Comment):
                text_parts.append(str(child))

    walk(block)
    raw_text = re.sub(r"[ \t\r\n\f\v]+", " ", "".join(text_parts))

    node_offsets: list[tuple[Tag, int]] = []
    for marker, node in preserved_nodes:
        offset = raw_text.find(marker)
        if offset < 0:  # pragma: no cover - marker 由本函数写入
            continue
        raw_text = raw_text[:offset] + raw_text[offset + len(marker) :]
        node_offsets.append((node, offset))

    text = raw_text.strip()
    if not text:
        return "", {}

    leading = len(raw_text) - len(raw_text.lstrip())
    source_length = len(text)
    nodes: list[dict[str, object]] = []
    for index, (node, raw_offset) in enumerate(node_offsets):
        inline_id = f"{anchor}_inline_{index}"
        offset = min(max(raw_offset - leading, 0), source_length)
        placement = "before" if offset == 0 else "after" if offset == source_length else "inline"
        node[_INLINE_ID_ATTR] = inline_id
        nodes.append(
            {
                "id": inline_id,
                "tag": node.name,
                "placement": placement,
                "offset": offset,
            }
        )

    meta: dict[str, object] = {}
    if nodes:
        meta[_INLINE_META_KEY] = {
            "version": 1,
            "source_length": source_length,
            "nodes": nodes,
        }
    return text, meta


def _has_meaningful_descendant_block(element: Tag) -> bool:
    """块内若已有更细粒度的正文块，则外层只作为布局容器保留。"""
    return any(
        descendant.get_text(strip=True) for descendant in element.find_all(_BLOCK_CANDIDATE_TAGS)
    )


def _list_item_link_target(element: Tag) -> Tag | None:
    """返回列表项自己的直接链接标签，避免回填时清空 ``li`` 和子列表。"""
    link = element.find("a", recursive=False)
    return link if isinstance(link, Tag) and link.get_text(strip=True) else None


def _split_direct_break_lines(element: Tag, soup: BeautifulSoup) -> list[Tag]:
    """把直接 ``br`` 分隔的可见行包装为独立翻译目标，原 ``br`` 不动。"""
    children = list(element.children)
    if not any(isinstance(child, Tag) and child.name == "br" for child in children):
        return [element]

    runs: list[list[Tag | NavigableString]] = [[]]
    for child in children:
        if isinstance(child, Tag) and child.name == "br":
            runs.append([])
        elif isinstance(child, Tag | NavigableString):
            runs[-1].append(child)

    targets: list[Tag] = []
    for run in runs:
        has_text = any(
            node.get_text(strip=True)
            if isinstance(node, Tag)
            else not isinstance(node, Comment) and bool(str(node).strip())
            for node in run
        )
        if not has_text:
            continue
        wrapper = soup.new_tag("span")
        wrapper[_LINE_WRAPPER_ATTR] = "true"
        run[0].insert_before(wrapper)
        for node in run:
            wrapper.append(node.extract())
        targets.append(wrapper)
    return targets


def _translation_targets(
    soup: BeautifulSoup,
    *,
    skip_navigation: bool,
    toc_lists: list[Tag] | None = None,
) -> list[Tag]:
    """按文档顺序选择可安全替换内容的最细粒度 EPUB 节点。

    含子正文块的 ``div``/``blockquote`` 等仅作为容器保留；``li`` 的
    直接链接文字单独成为翻译目标，从而同时保留列表层级和 ``href``。
    """
    active_toc_lists = toc_lists or (
        [ol for scope in nav_toc_scopes(soup) if (ol := nav_root_list(scope)) is not None]
        if skip_navigation
        else []
    )
    targets: list[Tag] = []
    for element in soup.find_all(_BLOCK_CANDIDATE_TAGS):
        if skip_navigation and _inside_navigation_list(element, active_toc_lists):
            continue

        has_descendant_block = _has_meaningful_descendant_block(element)
        if element.name == "li":
            link = _list_item_link_target(element)
            if link is not None and not _has_meaningful_descendant_block(link):
                targets.extend(_split_direct_break_lines(link, soup))
            if link is not None or has_descendant_block:
                continue

        if has_descendant_block:
            continue
        targets.extend(_split_direct_break_lines(element, soup))
    return targets


def _find_opf_path(zf: zipfile.ZipFile) -> str:
    data = zf.read(_CONTAINER)
    root = ET.fromstring(data)
    # container.xml 用了默认命名空间，按 localname 匹配
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "rootfile":
            path = el.attrib.get("full-path", "").strip()
            if path:
                return path
    raise ValueError("EPUB 损坏：container.xml 未找到有效的 rootfile full-path")


def _zip_href(base_path: str, href: str) -> str:
    """Resolve an EPUB-relative href to a normalized zip member path."""
    return resolve_epub_href(base_path, href).resource_href


def _parse_opf(zf: zipfile.ZipFile, opf_path: str) -> tuple[str, list[str], list[str]]:
    """返回 (书名, spine 顺序的 XHTML zip 路径列表, TOC/NAV 文件路径列表)。

    多份目录时 NAV 排在最前：EPUB3 NAV 是主目录，spine.toc 指定的
    EPUB2 NCX 次之，其余声明为 NCX 媒体类型的条目殿后。切章阶段按
    此顺序逐份尝试，取第一份能产出边界的目录。
    """
    root = ET.fromstring(zf.read(opf_path))

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    title = ""
    manifest: dict[str, tuple[str, str, str]] = {}  # id -> (href, media-type, properties)
    spine_ids: list[str] = []
    toc_ids: list[str] = []

    for el in root.iter():
        name = local(el.tag)
        if name == "title" and not title and el.text:
            title = el.text.strip()
        elif name == "item":
            item_id = el.attrib.get("id", "").strip()
            if not item_id:
                continue
            manifest[item_id] = (
                el.attrib.get("href", ""),
                el.attrib.get("media-type", ""),
                el.attrib.get("properties", ""),
            )
        elif name == "itemref":
            idref = el.attrib.get("idref", "").strip()
            if idref:
                spine_ids.append(idref)
        elif name == "spine":
            toc = el.attrib.get("toc")
            if toc:
                toc_ids.append(toc)

    hrefs: list[str] = []
    for sid in spine_ids:
        if sid not in manifest:
            continue
        href, media, _props = manifest[sid]
        if "html" not in media and not href.endswith((".xhtml", ".html", ".htm")):
            continue
        resolved_href = _zip_href(opf_path, href)
        if resolved_href and resolved_href not in hrefs:
            # 同一物理资源可被 spine 重复引用，但 zip 中仍只有一份 XHTML；
            # 只标注一次，避免生成无法回填的第二套锚点。
            hrefs.append(resolved_href)

    nav_ids = [
        item_id for item_id, (_href, _media, props) in manifest.items() if "nav" in props.split()
    ]
    ncx_ids = [
        item_id
        for item_id, (_href, media, _props) in manifest.items()
        if media == "application/x-dtbncx+xml"
    ]
    ordered_toc_ids = nav_ids + toc_ids + ncx_ids
    toc_paths: list[str] = []
    for item_id in ordered_toc_ids:
        if item_id not in manifest:
            continue
        href = _zip_href(opf_path, manifest[item_id][0])
        if href and href not in toc_paths:
            toc_paths.append(href)
    return title, hrefs, toc_paths


def _decode_markup(data: bytes) -> str:
    """按 XML/HTML 声明和字节特征解码 XHTML；都无法识别时，才用 UTF-8 解码并替换无效字节。"""
    decoded = UnicodeDammit(data).unicode_markup
    return decoded if decoded is not None else data.decode("utf-8", errors="replace")


def _looks_like_internal_title(title: str, href: str, book_title: str = "") -> bool:
    base = posixpath.basename(href).rsplit(".", 1)[0]
    stripped = title.strip()
    return (bool(base) and stripped == base) or (
        bool(book_title) and stripped == book_title.strip()
    )


_IMMUTABLE_TEXT_TAGS = {"script", "style", "rt", "rp"}
_ATOMIC_TEXT_TAGS = {
    "audio",
    "canvas",
    "embed",
    "hr",
    "iframe",
    "img",
    "math",
    "object",
    "svg",
    "video",
}
_WS_RE = re.compile(r"[ \t\r\n\f\v]+")


def _element_children(element: etree._Element) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str)]


def _element_path(root: etree._Element, element: etree._Element) -> tuple[int, ...]:
    if element is root:
        return ()
    path: list[int] = []
    current = element
    while current is not root:
        parent = current.getparent()
        if parent is None:
            raise ValueError("EPUB element is detached from its resource root")
        children = _element_children(parent)
        try:
            path.append(children.index(current))
        except ValueError as error:
            raise ValueError("EPUB element locator is not an element-index path") from error
        current = parent
    return tuple(reversed(path))


def _attr_local(element: etree._Element, name: str) -> str:
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].split(":", 1)[-1] == name:
            return value
    return ""


def _nav_toc_roots(root: etree._Element) -> list[etree._Element]:
    """Match ``epub_toc.nav_toc_scopes``/``nav_root_list`` in lxml."""
    navs = [
        node
        for node in root.iter()
        if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1].lower() == "nav"
    ]
    typed = [node for node in navs if "toc" in _attr_local(node, "type").split()]
    scopes = typed or navs[:1] or [root]
    roots: list[etree._Element] = []
    for scope in scopes:
        direct = next(
            (
                child
                for child in scope
                if isinstance(child.tag, str) and child.tag.rsplit("}", 1)[-1].lower() == "ol"
            ),
            None,
        )
        if direct is None:
            direct = next(
                (
                    node
                    for node in scope.iter()
                    if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1].lower() == "ol"
                ),
                None,
            )
        if direct is not None:
            roots.append(direct)
    return roots


def _visible_text(element: etree._Element) -> str:
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
        if tag in _IMMUTABLE_TEXT_TAGS or tag in _ATOMIC_TEXT_TAGS or _is_footnote_marker(node):
            return
        if node.text:
            parts.append(node.text)
        for child in node:
            if isinstance(child.tag, str):
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(element)
    return _WS_RE.sub(" ", "".join(parts)).strip()


def _is_semantic_footnote_wrapper_lxml(node: etree._Element) -> bool:
    tag = node.tag.rsplit("}", 1)[-1].lower() if isinstance(node.tag, str) else ""
    if tag in {"sup", "sub"}:
        return True
    if tag != "span":
        return False
    classes = str(node.get("class", "")).split()
    if any(token.lower() in _FOOTNOTE_CLASS_TOKENS for token in classes):
        return True
    for declaration in str(node.get("style", "")).split(";"):
        name, separator, value = declaration.partition(":")
        if (
            separator
            and name.strip().lower() == "vertical-align"
            and value.strip().lower() in {"super", "sub"}
        ):
            return True
    return False


def _is_footnote_marker(element: etree._Element) -> bool:
    tag = element.tag.rsplit("}", 1)[-1].lower() if isinstance(element.tag, str) else ""
    if tag not in {"sup", "sub", "span", "a"}:
        return False
    wrapper = element if _is_semantic_footnote_wrapper_lxml(element) else None
    parent = element.getparent()
    while wrapper is None and parent is not None:
        parent_tag = parent.tag.rsplit("}", 1)[-1].lower() if isinstance(parent.tag, str) else ""
        if parent_tag in _BLOCK_TAGS:
            break
        if _is_semantic_footnote_wrapper_lxml(parent):
            wrapper = parent
            break
        parent = parent.getparent()
    if wrapper is None:
        return False
    for link in element.iter():
        if not isinstance(link.tag, str) or link.tag.rsplit("}", 1)[-1].lower() != "a":
            continue
        href = _attr_local(link, "href")
        if not href:
            continue
        parts = urlsplit(href)
        if parts.scheme or parts.netloc or not parts.fragment:
            continue
        epub_type = _attr_local(link, "type")
        hints = " ".join(
            str(node.get(key, "")) for node in (link, wrapper) for key in _FOOTNOTE_HINT_ATTRS
        )
        href_hint = bool(
            _FOOTNOTE_HINT_RE.search(parts.path) or _FOOTNOTE_HINT_RE.search(parts.fragment)
        )
        short_marker = any(
            _SHORT_MARKER_RE.fullmatch("".join(node.itertext()).strip()) is not None
            for node in (link, wrapper)
        )
        if "noteref" in epub_type.split() or _FOOTNOTE_HINT_RE.search(hints):
            return True
        if href_hint or short_marker:
            return True
    return False


def _slot_parts(value: str) -> tuple[str, str, str]:
    leading_match = re.match(r"[ \t\r\n\f\v]*", value)
    trailing_match = re.search(r"[ \t\r\n\f\v]*$", value)
    leading = leading_match.group(0) if leading_match else ""
    trailing = trailing_match.group(0) if trailing_match else ""
    end = len(value) - len(trailing) if trailing else len(value)
    return leading, trailing, value[len(leading) : end]


def _block_slots(
    root: etree._Element,
    block: etree._Element,
    *,
    resource_href: str,
    resource_sha256: str,
    parse_mode: str,
    anchor: str,
) -> list:
    from trans_novel.ingest.models import EpubTextSlot

    slots: list[EpubTextSlot] = []
    slot_index = 0

    def add(owner: etree._Element, field: str, value: str | None) -> None:
        nonlocal slot_index
        if not value or not value.strip():
            return
        leading, trailing, core = _slot_parts(value)
        if not core:
            return
        relative = tuple(index for index in _element_path(block, owner))
        slot_index += 1
        slots.append(
            EpubTextSlot(
                id=f"{anchor}:s{slot_index}",
                element_path=relative,
                field=field,
                source_value=value,
                leading_whitespace=leading,
                trailing_whitespace=trailing,
                source_core=core,
            )
        )

    def walk(owner: etree._Element) -> None:
        tag = owner.tag.rsplit("}", 1)[-1].lower() if isinstance(owner.tag, str) else ""
        if tag in _IMMUTABLE_TEXT_TAGS or tag in _ATOMIC_TEXT_TAGS or _is_footnote_marker(owner):
            return
        add(owner, "text", owner.text)
        for child in owner:
            # Comment/PI nodes have no QName-bearing path step.  Their tails
            # remain immutable rather than being guessed by source text later.
            if not isinstance(child.tag, str):
                continue
            walk(child)
            add(child, "tail", child.tail)

    walk(block)
    return slots


def _resource_fingerprint(block: etree._Element) -> str:
    return hashlib.sha256(etree.tostring(block, encoding="utf-8", with_tail=False)).hexdigest()


def _resource_parser(data: bytes) -> tuple[etree._ElementTree, str, list[dict[str, object]]]:
    if len(data) > 512 * 1024 * 1024:
        raise ValueError("EPUB XHTML resource exceeds 512 MiB limit")
    diagnostics: list[dict[str, object]] = []
    strict = etree.XMLParser(
        no_network=True,
        recover=False,
        resolve_entities=False,
        remove_comments=False,
        remove_pis=False,
        strip_cdata=False,
    )
    try:
        return etree.fromstring(data, strict).getroottree(), "xml", diagnostics
    except etree.XMLSyntaxError as strict_error:
        first = strict_error.error_log[0] if strict_error.error_log else None
        if first is not None:
            diagnostics = [
                {
                    "level": first.level_name,
                    "domain": first.domain_name,
                    "type": first.type_name,
                    "line": first.line,
                    "column": first.column,
                }
            ]
        recovered = etree.HTMLParser(
            recover=True,
            no_network=True,
            remove_comments=False,
            remove_pis=False,
        )
        tree = etree.parse(io.BytesIO(data), recovered)
        root = tree.getroot()
        if root is None or root.find(".//body") is None:
            raise ValueError(
                "EPUB malformed XHTML recovery did not produce a document body"
            ) from strict_error
        return tree, "recovered", diagnostics[:20]


def _lxml_targets(root: etree._Element, *, skip_navigation: bool) -> list[etree._Element]:
    candidates: list[etree._Element] = []
    nav_roots = _nav_toc_roots(root) if skip_navigation else []
    for element in root.iter():
        if (
            not isinstance(element.tag, str)
            or element.tag.rsplit("}", 1)[-1].lower() not in _BLOCK_CANDIDATE_TAGS
        ):
            continue
        local = element.tag.rsplit("}", 1)[-1].lower()
        if any(
            isinstance(parent.tag, str)
            and parent.tag.rsplit("}", 1)[-1].lower() in _IMMUTABLE_TEXT_TAGS | _ATOMIC_TEXT_TAGS
            for parent in element.iterancestors()
        ):
            continue
        if skip_navigation and any(
            element is nav_root or any(element is node for node in nav_root.iterdescendants())
            for nav_root in nav_roots
        ):
            continue
        descendants = [
            child
            for child in element.iterdescendants()
            if isinstance(child.tag, str)
            and child.tag.rsplit("}", 1)[-1].lower() in _BLOCK_CANDIDATE_TAGS
            and _visible_text(child)
        ]
        if local == "li":
            direct_link = next(
                (
                    child
                    for child in element
                    if isinstance(child.tag, str)
                    and child.tag.rsplit("}", 1)[-1].lower() == "a"
                    and _visible_text(child)
                ),
                None,
            )
            if direct_link is not None and not any(
                isinstance(child.tag, str)
                and child.tag.rsplit("}", 1)[-1].lower() in _BLOCK_CANDIDATE_TAGS
                and _visible_text(child)
                for child in direct_link.iterdescendants()
            ):
                candidates.append(direct_link)
                continue
        if descendants:
            continue
        candidates.append(element)
    return [candidate for candidate in candidates if _visible_text(candidate)]


def _annotate_lxml_resource(
    data: bytes,
    resource_index: int,
    href: str,
    *,
    book_title: str = "",
    skip_navigation: bool = False,
) -> tuple[str, list[Segment], dict[str, object]]:
    tree, parse_mode, diagnostics = _resource_parser(data)
    root = tree.getroot()
    digest = hashlib.sha256(data).hexdigest()
    segments: list[Segment] = []
    fragment_anchors: dict[str, str | None] = {}
    for index, block in enumerate(_lxml_targets(root, skip_navigation=skip_navigation)):
        anchor = f"tn{resource_index}_{index}"
        slots = _block_slots(
            root,
            block,
            resource_href=href,
            resource_sha256=digest,
            parse_mode=parse_mode,
            anchor=anchor,
        )
        if not slots:
            continue
        runs: list[list] = [[]]
        direct_children = _element_children(block)
        for slot in slots:
            if not slot.element_path:
                run_index = 0
            else:
                child_index = slot.element_path[0]
                run_index = sum(
                    1
                    for child in direct_children[: child_index + 1]
                    if child.tag.rsplit("}", 1)[-1].lower() == "br"
                )
            while len(runs) <= run_index:
                runs.append([])
            runs[run_index].append(slot)

        kind = KIND_HEADING if block.tag.rsplit("}", 1)[-1].lower() in _HEADING_TAGS else KIND_TEXT
        for run_index, run_slots in enumerate(runs):
            if not run_slots:
                continue
            run_anchor = anchor if run_index == 0 else f"{anchor}_br{run_index}"
            if run_anchor != anchor:
                run_slots = [
                    slot.model_copy(update={"id": f"{run_anchor}:s{slot_index}"})
                    for slot_index, slot in enumerate(run_slots, 1)
                ]
            source = _WS_RE.sub(
                " ",
                "".join(
                    slot.leading_whitespace + slot.source_core + slot.trailing_whitespace
                    for slot in run_slots
                ),
            ).strip()
            state = EpubSegmentState(
                resource_href=href,
                resource_sha256=digest,
                block_path=_element_path(root, block),
                block_fingerprint=_resource_fingerprint(block),
                parse_mode=parse_mode,
                slots=run_slots,
                slot_contract_sha256=_slot_contract_digest(run_slots),
            )
            segments.append(
                Segment(
                    index=len(segments),
                    source=source,
                    kind=kind,
                    anchor=run_anchor,
                    resource_href=href,
                    epub_state=state,
                )
            )

    def resolve(path: tuple[int, ...]) -> etree._Element:
        current = root
        for child_index in path:
            current = _element_children(current)[child_index]
        return current

    ordered_nodes = list(root.iter())
    node_positions = {id(node): index for index, node in enumerate(ordered_nodes)}
    segment_blocks = [
        (segment, resolve(segment.epub_state.block_path))
        for segment in segments
        if segment.epub_state is not None
    ]
    block_positions = [
        (segment, block, node_positions[id(block)]) for segment, block in segment_blocks
    ]

    def containing_segment(node: etree._Element):
        for segment, block, _block_index in block_positions:
            if node is block or any(node is child for child in block.iterdescendants()):
                top = node
                while top.getparent() is not block and top.getparent() is not None:
                    top = top.getparent()
                direct = _element_children(block)
                if top is block:
                    run_index = 0
                elif top in direct:
                    top_index = direct.index(top)
                    run_index = sum(
                        1
                        for child in direct[:top_index]
                        if child.tag.rsplit("}", 1)[-1].lower() == "br"
                    )
                else:
                    run_index = 0
                return segment.anchor if run_index == 0 else f"{segment.anchor}_br{run_index}"
        return None

    for node_index, node in enumerate(ordered_nodes):
        if not isinstance(node.tag, str):
            continue
        identifiers = [value for value in (node.get("id"), node.get("name")) if value]
        if not identifiers:
            continue
        anchor_for_node = containing_segment(node)
        if anchor_for_node is None:
            later = [
                segment
                for segment, _block, block_index in block_positions
                if block_index > node_index
            ]
            anchor_for_node = later[0].anchor if later else None
        for identifier in identifiers:
            fragment_anchors.setdefault(identifier, anchor_for_node)

    title = ""
    for heading in root.iter():
        if isinstance(heading.tag, str) and heading.tag.rsplit("}", 1)[-1].lower() in _HEADING_TAGS:
            title = _visible_text(heading)
            if title:
                break
    if not title:
        for title_node in root.iter():
            if (
                isinstance(title_node.tag, str)
                and title_node.tag.rsplit("}", 1)[-1].lower() == "title"
            ):
                candidate = _visible_text(title_node)
                if candidate and not _looks_like_internal_title(candidate, href, book_title):
                    title = candidate
                break
    return (
        title,
        segments,
        {
            "href": href,
            "index": resource_index,
            "resource_sha256": digest,
            "parse_mode": parse_mode,
            "parser_diagnostics": diagnostics,
            "fragment_anchors": fragment_anchors,
        },
    )


def annotate_epub_resource(
    html: str,
    resource_index: int,
    href: str,
    *,
    book_title: str = "",
    skip_navigation: bool = False,
) -> tuple[str, list[Segment], str]:
    """Legacy marker/template helper retained for bilingual compatibility."""
    soup = BeautifulSoup(html, "html.parser")
    segments: list[Segment] = []
    first_heading: Tag | None = None
    heading_title_parts: list[str] = []
    idx = 0
    toc_lists = (
        [ol for scope in nav_toc_scopes(soup) if (ol := nav_root_list(scope)) is not None]
        if skip_navigation
        else []
    )
    for el in _translation_targets(soup, skip_navigation=skip_navigation, toc_lists=toc_lists):
        footnote_roots = _footnote_marker_roots(el)
        for descendant in list(el.find_all(True)):
            if _inside_roots(descendant, footnote_roots) or not descendant.get_text(strip=True):
                continue
            anchor_attrs = {
                key: descendant.attrs.pop(key) for key in ("id", "name") if key in descendant.attrs
            }
            if anchor_attrs:
                marker = soup.new_tag("a")
                marker.attrs.update(anchor_attrs)
                descendant.insert_before(marker)
        anchor = f"tn{resource_index}_{idx}"
        text, meta = _segment_content(el, anchor)
        if not text:
            continue
        el["data-tn-id"] = anchor
        kind = (
            KIND_HEADING
            if el.name in _HEADING_TAGS or el.find_parent(_HEADING_TAGS) is not None
            else KIND_TEXT
        )
        if kind == KIND_HEADING:
            heading = el if el.name in _HEADING_TAGS else el.find_parent(_HEADING_TAGS)
            if isinstance(heading, Tag):
                if first_heading is None:
                    first_heading = heading
                if heading is first_heading:
                    heading_title_parts.append(text)
        segments.append(
            Segment(
                index=idx,
                source=text,
                kind=kind,
                anchor=anchor,
                resource_href=href,
                meta=meta,
            )
        )
        idx += 1
    title = " ".join(heading_title_parts)
    if not title and soup.title and soup.title.string:
        candidate = soup.title.string.strip()
        if not _looks_like_internal_title(candidate, href, book_title):
            title = candidate
    return title, segments, str(soup)


def _inside_navigation_list(element: Tag, toc_lists: list[Tag]) -> bool:
    """判断块元素是否位于目录列表（``ol``）或其目录项（``li``）内。

    不依赖祖先是否为 ``<nav>``：``toc_lists`` 由调用方通过
    ``epub_toc.nav_toc_scopes``/``nav_root_list`` 定位，与解析端规则一致，
    因此也兼容 body > ol > li > a 这类没有 ``<nav>`` 包装的非标准 NAV。只保护
    ``li`` 及其内部块，避免普通回填清空链接和嵌套 ``ol``。
    """
    if not toc_lists:
        return False
    inside_toc_list = False
    inside_list_item = element.name == "li"
    for parent in element.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name == "li":
            inside_list_item = True
        if any(parent is ol for ol in toc_lists):
            inside_toc_list = True
    return inside_toc_list and inside_list_item


def _fragment_anchor_map(template: str) -> dict[str, str | None]:
    """把 XHTML 中的 id/name 定位到 Segment 锚点。

    值为 ``None`` 表示该 ID 确实存在，但它位于该资源最后一个可翻译块
    之后；这与“fragment 根本不存在”（key 缺失）必须区分，否则两种
    情况在切章阶段都会被误判为同一种损坏。
    """
    soup = BeautifulSoup(template, "html.parser")
    mapping: dict[str, str | None] = {}
    for node in soup.find_all(True):
        identifiers = [node.get("id"), node.get("name")]
        if not any(isinstance(value, str) and value for value in identifiers):
            continue
        block = (
            node if node.has_attr("data-tn-id") else node.find_parent(attrs={"data-tn-id": True})
        )
        if not isinstance(block, Tag):
            block = node.find_next(attrs={"data-tn-id": True})
        raw_anchor = block.get("data-tn-id") if isinstance(block, Tag) else None
        anchor = raw_anchor if isinstance(raw_anchor, str) and raw_anchor else None
        for value in identifiers:
            if isinstance(value, str) and value:
                mapping.setdefault(value, anchor)
    return mapping


def _logical_chapters(
    resources: list[dict[str, object]],
    toc_entries: list[dict[str, object]],
) -> tuple[list[Chapter], str, str]:
    """按本地切章规则把物理资源的 Segment 流切成逻辑 Chapter。

    无可用目录边界时回退为每个非空 spine XHTML 一章（与历来行为一
    致）。首个目录边界前若仍有正文，独立成前置章，不丢内容。无论走
    哪条策略，Chapter.template 恒为 None：标注模板不随章持久化，统一
    由 read_epub 写进 ``Document.meta["epub_resource_templates"]``。
    """
    all_segments: list[Segment] = []
    anchor_positions: dict[str, int] = {}
    resource_starts: dict[str, int] = {}
    resource_by_href: dict[str, dict[str, object]] = {}
    for resource in resources:
        href = str(resource["href"])
        resource_by_href[href] = resource
        resource_starts[href] = len(all_segments)
        raw_segments = resource.get("segments")
        segments = raw_segments if isinstance(raw_segments, list) else []
        for segment in segments:
            if not isinstance(segment, Segment):
                continue
            if segment.anchor:
                anchor_positions[segment.anchor] = len(all_segments)
            all_segments.append(segment)

    for entry in toc_entries:
        href = entry.get("resource_href")
        if not isinstance(href, str) or href not in resource_starts:
            continue
        fragment = entry.get("fragment")
        has_fragment = isinstance(fragment, str) and bool(fragment)
        resource = resource_by_href[href]
        raw_fragment_map = resource.get("fragment_anchors")
        fragment_map = raw_fragment_map if isinstance(raw_fragment_map, dict) else {}
        if has_fragment and fragment not in fragment_map:
            # 规则 1：损坏的 fragment 不能悄悄退回资源开头，否则会在
            # 错误位置切章，并把首个 heading 的译文写给错误目录项。
            continue
        segment_anchor = fragment_map.get(fragment) if has_fragment else None
        if not has_fragment:
            raw_segments = resource.get("segments")
            resource_segments = raw_segments if isinstance(raw_segments, list) else []
            first = next(
                (segment for segment in resource_segments if isinstance(segment, Segment)), None
            )
            segment_anchor = first.anchor if first is not None else None
        if isinstance(segment_anchor, str) and segment_anchor in anchor_positions:
            entry["segment_anchor"] = segment_anchor
            entry["boundary_position"] = anchor_positions[segment_anchor]
        elif has_fragment:
            raw_segments = resource.get("segments")
            segment_count = (
                sum(isinstance(segment, Segment) for segment in raw_segments)
                if isinstance(raw_segments, list)
                else 0
            )
            # 规则 2：fragment 存在但位于最后一个可翻译块之后 → 将边界设为
            # 资源末尾，以区别于“fragment 不存在”。
            entry["boundary_position"] = resource_starts[href] + segment_count
        else:
            # 规则 3：无文字标题页也是有效目录边界，边界为资源起点，
            # 后续 spine 正文因此仍能归入该逻辑章。
            entry["boundary_position"] = resource_starts[href]

    # 规则 4：无 href 的分组节点（“部”）继承第一个可定位子节点的边界，
    # 但不继承 segment_anchor，避免把子章 heading 的译文误当分组标题。
    toc_paths = {
        str(entry.get("toc_path"))
        for entry in toc_entries
        if isinstance(entry.get("toc_path"), str) and entry.get("toc_path")
    }
    for toc_path in toc_paths:
        path_entries = [entry for entry in toc_entries if entry.get("toc_path") == toc_path]
        children: dict[int, list[dict[str, object]]] = {}
        for entry in path_entries:
            parent_index = entry.get("parent_index")
            if isinstance(parent_index, int):
                children.setdefault(parent_index, []).append(entry)
        for entry in reversed(path_entries):
            if isinstance(entry.get("boundary_position"), int):
                continue
            if entry.get("raw_href"):
                # 只有无链接的结构分组可以继承子节点；已显式给出但无法
                # 解析的链接属于损坏数据，不应被悄悄改成别的目标。
                continue
            node_index = entry.get("node_index")
            if not isinstance(node_index, int):
                continue
            descendant = next(
                (
                    child
                    for child in children.get(node_index, [])
                    if isinstance(child.get("boundary_position"), int)
                ),
                None,
            )
            if descendant is not None:
                entry["boundary_position"] = descendant["boundary_position"]
                entry["inherited_boundary_from"] = descendant.get("entry_id")

    # 规则 7：多份目录时取第一份能产出边界的（NAV 由 _parse_opf 排在前）。
    ordered_toc_paths = list(
        dict.fromkeys(
            str(entry.get("toc_path"))
            for entry in toc_entries
            if isinstance(entry.get("toc_path"), str) and entry.get("toc_path")
        )
    )
    segment_lengths = [len(segment.source) for segment in all_segments]
    canonical_toc_path = ""
    boundaries: list[dict[str, object]] = []
    selected_depth = 0
    for toc_path in ordered_toc_paths:
        candidates, depth = select_boundaries(
            [entry for entry in toc_entries if entry.get("toc_path") == toc_path],
            segment_lengths,
        )
        if candidates:
            canonical_toc_path = toc_path
            boundaries = candidates
            selected_depth = depth
            break
    boundaries.sort(key=lambda item: int(item["boundary_position"]))

    if not boundaries:
        # 规则 6：无任何可用目录边界 → spine-fallback，每个非空资源一章。
        chapters: list[Chapter] = []
        for resource in resources:
            raw_segments = resource.get("segments")
            segments = (
                [s for s in raw_segments if isinstance(s, Segment)]
                if isinstance(raw_segments, list)
                else []
            )
            if not segments:
                continue
            for index, segment in enumerate(segments):
                segment.index = index
            chapters.append(
                Chapter(
                    index=len(chapters),
                    title=str(resource.get("title") or ""),
                    segments=segments,
                    href=str(resource.get("href") or "") or None,
                    template=None,
                    meta={"epub_split_strategy": _STRATEGY_SPINE_FALLBACK},
                )
            )
        return chapters, _STRATEGY_SPINE_FALLBACK, canonical_toc_path

    slices: list[tuple[int, int, dict[str, object] | None]] = []
    first_position = int(boundaries[0]["boundary_position"])
    if first_position > 0:
        # 规则 5：首个边界前仍有正文 → 独立前置章。
        slices.append((0, first_position, None))
    for index, boundary in enumerate(boundaries):
        start = int(boundary["boundary_position"])
        end = (
            int(boundaries[index + 1]["boundary_position"])
            if index + 1 < len(boundaries)
            else len(all_segments)
        )
        if end > start:
            slices.append((start, end, boundary))

    strategy = f"toc-depth-{selected_depth}"
    chapters = []
    for start, end, boundary in slices:
        segments = all_segments[start:end]
        for index, segment in enumerate(segments):
            segment.index = index
        if boundary is not None:
            title = str(boundary.get("title") or "")
            toc_entry_id = boundary.get("entry_id")
            first_href = segments[0].resource_href or str(boundary.get("resource_href") or "")
        else:
            first_href = segments[0].resource_href or ""
            title = segments[0].source if segments[0].kind == KIND_HEADING else ""
            toc_entry_id = None
        meta: dict[str, object] = {"epub_split_strategy": strategy}
        if isinstance(toc_entry_id, str):
            meta["toc_entry_id"] = toc_entry_id
        chapters.append(
            Chapter(
                index=len(chapters),
                title=title,
                segments=segments,
                href=first_href or None,
                template=None,
                meta=meta,
            )
        )
    return chapters, strategy, canonical_toc_path


def read_epub(path: str, source_lang: str, target_lang: str) -> Document:
    """Read a source EPUB into schema-3 structural text-slot state."""
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        opf_path = _find_opf_path(zf)
        book_title, hrefs, toc_paths = _parse_opf(zf, opf_path)
        toc_entries = parse_toc_entries(zf, toc_paths)

        resources: list[dict[str, object]] = []
        archive_hash = hashlib.sha256()
        with open(path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                archive_hash.update(chunk)
        resource_hrefs = list(dict.fromkeys([*hrefs, *toc_paths]))
        for resource_index, href in enumerate(resource_hrefs):
            if href not in names:
                continue
            info = zf.getinfo(href)
            if info.file_size > 512 * 1024 * 1024:
                raise ValueError(f"EPUB resource exceeds 512 MiB limit: {href}")
            if not href.lower().endswith(_HTML_EXTS):
                continue
            data = zf.read(href)
            title, segments, resource = _annotate_lxml_resource(
                data,
                resource_index,
                href,
                book_title=book_title,
                skip_navigation=href in toc_paths,
            )
            resources.append(
                {
                    **resource,
                    "title": title,
                    "segments": segments,
                }
            )
        spine_resources = [resource for resource in resources if resource["href"] in hrefs]
        chapters, split_strategy, split_toc_path = _logical_chapters(spine_resources, toc_entries)

    return Document(
        title=book_title or os.path.splitext(os.path.basename(path))[0],
        source_lang=source_lang,
        target_lang=target_lang,
        fmt="epub",
        source_path=os.path.abspath(path),
        chapters=chapters,
        meta={
            "epub_schema": 3,
            "epub_sha256": archive_hash.hexdigest(),
            "opf_path": opf_path,
            "toc_paths": toc_paths,
            "toc_entries": toc_entries,
            "epub_resources": [
                {
                    "index": resource["index"],
                    "href": resource["href"],
                    "resource_sha256": resource["resource_sha256"],
                    "parse_mode": resource["parse_mode"],
                    "parser_diagnostics": resource["parser_diagnostics"],
                }
                for resource in resources
            ],
            "epub_split_strategy": split_strategy,
            "epub_split_toc_path": split_toc_path,
        },
    )
