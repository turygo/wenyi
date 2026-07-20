"""EPUB 读取器（纯标准库 + BeautifulSoup）。

EPUB 即一个 zip：
  META-INF/container.xml → 指向 OPF
  OPF → manifest（资源清单）+ spine（阅读顺序）

读取时先按 spine 逐个物理 XHTML 标注 Segment（锚点按物理资源序号生成，
与逻辑章号无关），再依据 NCX/NAV 的顶层目录边界把整书 Segment 流切成
逻辑 Chapter。因此 Chapter 与 XHTML 不再是一对一：切章之后，每个
Segment 的 ``resource_href`` 仍记录它所属的物理资源，写回时据此按
物理文件聚合。标注模板不再随 Chapter 持久化，而是统一放进
``Document.meta["epub_resource_templates"]``（键为物理资源 href），
由 RunStore 写入独立状态文件，与频繁重写的 manifest 解耦。
"""

from __future__ import annotations

import os
import posixpath
import xml.etree.ElementTree as ET
import zipfile

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, UnicodeDammit

from .epub_toc import (
    nav_root_list,
    nav_toc_scopes,
    parse_toc_entries,
    resolve_epub_href,
    select_top_level_boundaries,
)
from .models import KIND_HEADING, KIND_TEXT, Chapter, Document, Segment

_CONTAINER = "META-INF/container.xml"
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
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_INLINE_META_KEY = "epub_inline"
_INLINE_ID_ATTR = "data-tn-inline-id"
_ATOMIC_INLINE_TAGS = {
    "audio",
    "br",
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
_STRATEGY_TOP_LEVEL_TOC = "top-level-toc"
_STRATEGY_SPINE_FALLBACK = "spine-fallback"


def _preserved_inline_roots(block: Tag) -> list[Tag]:
    """返回需要原样回填的非文本节点，并尽量保留其无文字包装标签。"""
    roots: list[Tag] = []
    seen: set[int] = set()
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
        if id(root) not in seen:
            seen.add(id(root))
            roots.append(root)
    return roots


def _segment_content(block: Tag, anchor: str) -> tuple[str, dict[str, object]]:
    """提取可翻译文本，并给内联非文本节点写入稳定 ID 和位置元数据。"""
    roots = _preserved_inline_roots(block)
    root_ids = {id(node) for node in roots}
    text_parts: list[str] = []
    node_offsets: list[tuple[Tag, int]] = []
    raw_length = 0

    def walk(parent: Tag) -> None:
        nonlocal raw_length
        for child in parent.children:
            if isinstance(child, Tag):
                if id(child) in root_ids:
                    node_offsets.append((child, raw_length))
                else:
                    walk(child)
            elif isinstance(child, NavigableString) and not isinstance(child, Comment):
                value = str(child)
                text_parts.append(value)
                raw_length += len(value)

    walk(block)
    raw_text = "".join(text_parts)
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


def annotate_epub_resource(
    html: str,
    resource_index: int,
    href: str,
    *,
    book_title: str = "",
    skip_navigation: bool = False,
) -> tuple[str, list[Segment], str]:
    """标注单个物理 XHTML，返回 (标题, segments, 带标记的模板 HTML)。

    锚点使用物理资源序号而非最终 Chapter 序号，因此即使目录切章边界
    发生变化，同一物理文件重新标注仍能生成相同的 ``data-tn-id``。
    """
    soup = BeautifulSoup(html, "html.parser")
    segments: list[Segment] = []
    idx = 0
    # 目录项保护范围与解析端 epub_toc._parse_nav 使用相同的 nav_toc_scopes/
    # nav_root_list 定位规则，也能识别 body > ol > li > a 这类没有 <nav> 包装的非标准 NAV。
    toc_lists = (
        [ol for scope in nav_toc_scopes(soup) if (ol := nav_root_list(scope)) is not None]
        if skip_navigation
        else []
    )
    for el in soup.find_all(_BLOCK_TAGS):
        if skip_navigation and _inside_navigation_list(el, toc_lists):
            # NAV 可以同时是 spine 中的可见目录页。目录 li 中嵌套着
            # a/ol，若当普通段落回填会清空整棵目录结构；nav 内独立的
            # “Contents”等 heading/p 仍可安全作为普通正文翻译。
            continue
        # 跳过嵌套在另一个块级元素内的块（避免重复计数，如 blockquote 里的 p）
        if any(getattr(p, "name", None) in _BLOCK_TAGS for p in el.parents):
            continue
        # 带文字的内联 id/name 包装会在回填纯译文时被拍平。先把它改成
        # 同位置的空锚点，便可复用现有内联非文本节点恢复机制。
        for descendant in list(el.find_all(True)):
            if not descendant.get_text(strip=True):
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
        kind = KIND_HEADING if el.name in _HEADING_TAGS else KIND_TEXT
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

    # 物理资源的备用标题：首个 heading → 非内部文件名/书名的 <title> →
    # 无标题。逻辑章标题在切章阶段直接取完整 TOC 节点 title，不看这里。
    title = ""
    for s in segments:
        if s.kind == KIND_HEADING:
            title = s.source
            break
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
    canonical_toc_path = ""
    boundaries: list[dict[str, object]] = []
    for toc_path in ordered_toc_paths:
        candidates = select_top_level_boundaries(
            [entry for entry in toc_entries if entry.get("toc_path") == toc_path]
        )
        if candidates:
            canonical_toc_path = toc_path
            boundaries = candidates
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
        meta: dict[str, object] = {"epub_split_strategy": _STRATEGY_TOP_LEVEL_TOC}
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
    return chapters, _STRATEGY_TOP_LEVEL_TOC, canonical_toc_path


def read_epub(path: str, source_lang: str, target_lang: str) -> Document:
    """按 spine 读取物理资源，再按顶层目录边界生成逻辑章节（schema 2）。"""
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        opf_path = _find_opf_path(zf)
        book_title, hrefs, toc_paths = _parse_opf(zf, opf_path)
        toc_entries = parse_toc_entries(zf, toc_paths)

        resources: list[dict[str, object]] = []
        for resource_index, href in enumerate(hrefs):
            if href not in names:
                continue
            html = _decode_markup(zf.read(href))
            title, segments, template = annotate_epub_resource(
                html,
                resource_index,
                href,
                book_title=book_title,
                skip_navigation=href in toc_paths,
            )
            resources.append(
                {
                    "index": resource_index,
                    "href": href,
                    "title": title,
                    "segments": segments,
                    "template": template,
                    "fragment_anchors": _fragment_anchor_map(template),
                }
            )
        chapters, split_strategy, split_toc_path = _logical_chapters(resources, toc_entries)

    return Document(
        title=book_title or os.path.splitext(os.path.basename(path))[0],
        source_lang=source_lang,
        target_lang=target_lang,
        fmt="epub",
        source_path=os.path.abspath(path),
        chapters=chapters,
        meta={
            "epub_schema": 2,
            "opf_path": opf_path,
            "toc_paths": toc_paths,
            "toc_entries": toc_entries,
            "epub_resources": [
                {"index": resource["index"], "href": resource["href"]} for resource in resources
            ],
            "epub_split_strategy": split_strategy,
            "epub_split_toc_path": split_toc_path,
            "epub_resource_templates": {
                str(resource["href"]): str(resource["template"]) for resource in resources
            },
        },
    )
