"""回填：把译文写回原格式。

- 纯文本：按章重建，标题 + 段落（空行分隔）。
- EPUB：重开原始 zip，逐条目原样拷贝；schema 2 状态按物理资源 href 聚合
  全书 Segment，每个物理 XHTML 用已保存的模板渲染一次；schema 1 旧状态
  沿用逐章 chapter.template 渲染。两条路径都按 data-tn-id 锚点替换为
  译文后写回，非正文资源（图片/CSS/字体）不动。
缺失译文的段回退使用原文，保证不丢内容。
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from copy import deepcopy

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, UnicodeDammit
from lxml import etree

from trans_novel.assemble.bilingual_dom import (
    BILINGUAL_CSS,
    BILINGUAL_DIRECT_TARGET_ATTRS,
    BILINGUAL_SOURCE_CLASS,
    BILINGUAL_STYLE_ID,
    append_bilingual_style,
    dedupe_segment_mappings,
    direct_run_boundary,
    direct_run_is_active,
    direct_run_source_copy,
    has_reserved_source_collision,
    is_bilingual_container_tag,
    japanese_ruby_source_copy,
    sanitized_source_copy,
    segment_needs_source,
)
from trans_novel.ingest.epub_toc import nav_root_list, nav_toc_scopes
from trans_novel.ingest.fb2_reader import read_fb2_binaries
from trans_novel.ingest.models import (
    KIND_HEADING,
    Chapter,
    Segment,
    _normalized_slot_text,
    _slot_contract_digest,
)
from trans_novel.pipeline.runstore import RunStore
from trans_novel.postprocess.punct import normalize_heading_numbering

_ILLEGAL_FN = re.compile(r'[\\/:*?"<>|\r\n\t]+')
_HTML_EXTS = (".xhtml", ".html", ".htm")
_VERTICAL_MARKERS = (
    re.compile(rb"(?:-epub-|-webkit-)?writing-mode\s*:\s*(?:vertical-rl|vertical-lr|tb-rl)", re.I),
    re.compile(rb"page-progression-direction\s*=\s*['\"]rtl['\"]", re.I),
    re.compile(rb"\bclass\s*=\s*['\"][^'\"]*\bvrtl\b", re.I),
)
_HORIZONTAL_OVERRIDE_ID = "trans-novel-horizontal-override"
_HORIZONTAL_OVERRIDE_CSS = (
    "html, body { "
    "writing-mode: horizontal-tb !important; "
    "-epub-writing-mode: horizontal-tb !important; "
    "-webkit-writing-mode: horizontal-tb !important; "
    "direction: ltr !important; "
    "text-orientation: mixed !important; "
    "} "
    '.vrtl, .vertical, [class*="vrtl"] { '
    "writing-mode: horizontal-tb !important; "
    "-epub-writing-mode: horizontal-tb !important; "
    "-webkit-writing-mode: horizontal-tb !important; "
    "direction: ltr !important; "
    "}"
)
_IMAGE_EXTENSION_BY_TYPE = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
}
_BILINGUAL_STYLE_ID = BILINGUAL_STYLE_ID
_BILINGUAL_CSS = BILINGUAL_CSS


def _is_bilingual_container_tag(tag: str) -> bool:
    return is_bilingual_container_tag(tag)


_XML_ENCODING = re.compile(
    r"(<\?xml[^>]*\bencoding\s*=\s*)(['\"])[^'\"]+\2",
    re.IGNORECASE,
)


def _sanitize_filename(name: str, fallback: str = "translated") -> str:
    name = _ILLEGAL_FN.sub(" ", name or "").strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    return name[:120] or fallback


def _default_out(
    source_path: str,
    out_format: str,
    title: str | None = None,
    *,
    bilingual: bool = False,
) -> str:
    ext = ".epub" if out_format == "epub" else ".txt"
    if title and title.strip():
        # 保留给显式调用方使用；默认 assemble 不传书名译名。
        d = os.path.dirname(os.path.abspath(source_path))
        return os.path.join(d, _sanitize_filename(title) + ext)
    base, _ = os.path.splitext(source_path)
    suffix = ".zh-bi" if bilingual else ".zh"
    return f"{base}{suffix}{ext}"


def bilingual_out_path(out_path: str) -> str:
    """调用方显式指定了 out_path 时，派生双语版路径：stem 追加 -bi。"""
    base, ext = os.path.splitext(out_path)
    return f"{base}-bi{ext}"


def _ch_title(c: dict) -> str:
    """章节展示标题：优先译名，回退原标题；标题编号数字风格统一为汉字。"""
    title = (c.get("title_translated") or c.get("title") or "").strip()
    return normalize_heading_numbering(title)


def _seg_text(seg) -> str:
    return seg.target if (seg.target and seg.target.strip()) else seg.source


def _epub_lang(lang: str | None) -> str:
    """EPUB 元数据语言码；中文目标默认标成简体中文。"""
    normalized = (lang or "").strip().replace("_", "-").lower()
    if normalized in {"", "zh", "zh-cn", "zh-hans", "cn"}:
        return "zh-Hans"
    return lang or "zh-Hans"


def _merged_paragraphs(chapter: Chapter) -> list[tuple[str, str, str]]:
    """把章内 Segment 合并为段落，cont 续段并回上一段。返回 [(kind, target, source), ...]。"""
    paras: list[list[str]] = []  # 每段累积的译文片段
    srcs: list[list[str]] = []  # 每段累积的原文片段
    kinds: list[str] = []
    for s in chapter.segments:
        if not s.source.strip():
            continue
        if s.cont and paras:
            paras[-1].append(_seg_text(s))
            srcs[-1].append(s.source)
        else:
            paras.append([_seg_text(s)])
            srcs.append([s.source])
            kinds.append(s.kind)
    return [
        (
            k,
            normalize_heading_numbering("".join(p)) if k == KIND_HEADING else "".join(p),
            "".join(sr),
        )
        for k, p, sr in zip(kinds, paras, srcs, strict=False)
    ]


def _bilingual_source(source: str, target: str) -> str:
    """双语原文去重：原文为空白，或与译文相同（翻译回退到原文）时不输出原文。"""
    return source if (source.strip() and source != target) else ""


# ── 纯文本 ──────────────────────────────────────────────────────────────────
def _assemble_text(
    store: RunStore,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    m = store.load_manifest()
    chapter_blocks: list[str] = []
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        blocks: list[str] = []
        for kind, target, source in _merged_paragraphs(ch):
            src = _bilingual_source(source, target) if (bilingual and kind != KIND_HEADING) else ""
            if not src:
                blocks.append(target)
            elif order == "source_first":
                blocks.extend((src, target))
            else:
                blocks.extend((target, src))
        chapter_blocks.append("\n\n".join(blocks))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chapter_blocks) + "\n")
    return out_path


# ── EPUB ────────────────────────────────────────────────────────────────────
_INLINE_META_KEY = "epub_inline"
_INLINE_ID_ATTR = "data-tn-inline-id"
_LINE_WRAPPER_ATTR = "data-tn-line"


def _japanese_ruby_source(element: Tag, source_lang: str) -> str:
    """日语双语原文保留 ruby 注音，同时拍平其它文本内联标签。"""
    normalized_lang = source_lang.strip().replace("_", "-").lower()
    if not (normalized_lang == "ja" or normalized_lang.startswith("ja-")):
        return ""
    if element.find("ruby") is None:
        return ""

    fragment = BeautifulSoup(str(element), "html.parser")
    root = fragment.find(element.name)
    if not isinstance(root, Tag):
        return ""
    for comment in list(root.find_all(string=lambda node: isinstance(node, Comment))):
        comment.extract()
    for tag in list(
        root.find_all(
            [
                "audio",
                "canvas",
                "embed",
                "hr",
                "iframe",
                "img",
                "math",
                "object",
                "script",
                "source",
                "style",
                "svg",
                "video",
            ]
        )
    ):
        tag.decompose()
    ruby_tags = {"ruby", "rb", "rt", "rp", "rtc", "br"}
    for tag in list(root.find_all(True)):
        if tag.name not in ruby_tags:
            tag.unwrap()
            continue
        for attr in ("id", "name", "data-tn-id", _INLINE_ID_ATTR, _LINE_WRAPPER_ATTR):
            tag.attrs.pop(attr, None)
    return root.decode_contents()


def _append_source(element: Tag, source: str, markup: str) -> None:
    """向双语原文块写入纯文本，或写入已清理的日语 ruby 片段。"""
    if not markup:
        element.append(source)
        return
    fragment = BeautifulSoup(markup, "html.parser")
    for child in list(fragment.contents):
        element.append(child.extract())


def _append_text_with_breaks(soup: BeautifulSoup, element: Tag, text: str) -> None:
    """向元素追加文本，并把译文换行转换为 XHTML ``br``。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        if line:
            element.append(line)
        if index + 1 < len(lines):
            element.append(soup.new_tag("br"))


def _replace_block_content(
    soup: BeautifulSoup,
    el: Tag,
    text: str,
    meta: dict[str, object],
) -> None:
    """替换块内文字，按解析阶段记录的位置恢复图片，并按译文换行生成 ``br``。"""
    raw_inline = meta.get(_INLINE_META_KEY)
    inline = raw_inline if isinstance(raw_inline, dict) else {}
    raw_nodes = inline.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    source_length = inline.get("source_length")
    if not isinstance(source_length, int) or source_length < 0:
        source_length = 0
    source_text = re.sub(r"\s+", " ", el.get_text("", strip=False))
    if source_length <= 0:
        source_length = len(source_text)
    captured_links: list[tuple[int, Tag]] = []
    link_search_start = 0
    for link in list(el.find_all("a", href=True)):
        inline_parent = link.find_parent(attrs={_INLINE_ID_ATTR: True})
        if link.has_attr(_INLINE_ID_ATTR) or (
            inline_parent is not None and inline_parent is not el
        ):
            continue
        link_text = re.sub(r"\s+", " ", link.get_text("", strip=False))
        offset = source_text.find(link_text, link_search_start) if link_text else 0
        if offset < 0:
            offset = 0
        else:
            link_search_start = offset + len(link_text)
        clone_soup = BeautifulSoup(str(link), "html.parser")
        clone = clone_soup.find("a")
        if not isinstance(clone, Tag):
            continue
        adjacent_identity: Tag | None = None
        for sibling in list(link.previous_siblings) + list(link.next_siblings):
            if isinstance(sibling, NavigableString) and not str(sibling).strip():
                continue
            if (
                isinstance(sibling, Tag)
                and sibling.name == "a"
                and not sibling.get_text(strip=True)
                and any(sibling.get(key) for key in ("id", "name"))
            ):
                adjacent_identity = sibling
            break
        if adjacent_identity is not None:
            for key in ("id", "name"):
                value = adjacent_identity.get(key)
                if isinstance(value, str) and value:
                    clone[key] = value
            adjacent_identity.extract()
        captured_links.append((offset, clone))

    restored: list[tuple[int, int, Tag]] = []
    for order, record in enumerate(nodes):
        if not isinstance(record, dict):
            continue
        inline_id = record.get("id")
        offset = record.get("offset")
        if not isinstance(inline_id, str) or not isinstance(offset, int):
            continue
        node = el.find(True, attrs={_INLINE_ID_ATTR: inline_id})
        if not isinstance(node, Tag):
            continue
        node.extract()
        node.attrs.pop(_INLINE_ID_ATTR, None)
        if offset <= 0:
            target_offset = 0
        elif source_length <= 0 or offset >= source_length:
            target_offset = len(text)
        else:
            target_offset = round(offset * len(text) / source_length)
        restored.append((target_offset, order, node))
    for source_offset, node in captured_links:
        target_offset = (
            0
            if source_offset <= 0
            else len(text)
            if source_offset >= source_length
            else round(source_offset * len(text) / source_length)
        )
        restored.append((target_offset, len(restored), node))

    el.clear()
    cursor = 0
    for target_offset, _order, node in sorted(restored):
        target_offset = min(max(target_offset, cursor), len(text))
        if target_offset > cursor:
            _append_text_with_breaks(soup, el, text[cursor:target_offset])
        el.append(node)
        cursor = target_offset
    if cursor < len(text):
        _append_text_with_breaks(soup, el, text[cursor:])


def _render_segments_html(
    template: str,
    segments: list[Segment],
    *,
    bilingual: bool = False,
    order: str = "target_first",
    source_lang: str = "",
) -> str:
    """把同一物理 HTML 资源内的译文按锚点一次性回填。

    EPUB 的逻辑章节边界可以落在同一个 XHTML 中，也可以跨越多个 XHTML；
    真正的回填单位是物理资源而非 Chapter，调用方需先把属于同一
    ``resource_href`` 的 Segment（可能来自多个 Chapter）聚合后再调用本函数。
    """
    soup = BeautifulSoup(template or "", "html.parser")
    # 合并 cont 续段：续段文本并回其所属 anchor 元素
    by_anchor: dict[str, str] = {}
    src_by_anchor: dict[str, str] = {}
    kind_by_anchor: dict[str, str] = {}
    meta_by_anchor: dict[str, dict] = {}
    cur_anchor: str | None = None
    for s in segments:
        if s.cont and cur_anchor is not None:
            by_anchor[cur_anchor] += _seg_text(s)
            src_by_anchor[cur_anchor] += s.source
        elif s.anchor:
            cur_anchor = s.anchor
            by_anchor[cur_anchor] = _seg_text(s)
            src_by_anchor[cur_anchor] = s.source
            kind_by_anchor[cur_anchor] = s.kind
            meta_by_anchor[cur_anchor] = s.meta
    for anchor, text in by_anchor.items():
        el = soup.find(True, attrs={"data-tn-id": anchor})
        if el is None:
            continue
        line_wrapper = el.has_attr(_LINE_WRAPPER_ATTR)
        if kind_by_anchor.get(anchor) == KIND_HEADING:
            text = normalize_heading_numbering(text)
        src = (
            _bilingual_source(src_by_anchor.get(anchor, ""), text)
            if bilingual and kind_by_anchor.get(anchor) != KIND_HEADING
            else ""
        )
        source_markup = _japanese_ruby_source(el, source_lang) if src else ""
        _replace_block_content(soup, el, text, meta_by_anchor.get(anchor, {}))
        del el["data-tn-id"]
        if not src:
            continue
        # p 的原文可作为相邻段落插入；li/blockquote/td/th 则必须留在原容器内，
        # 避免生成 <ul><li>...</li><p>...</p></ul> 或
        # <table><tr>...</tr><p>...</p></table> 之类的非法结构，
        # 同时保留列表、引用块和表格单元格的语义和样式。
        nested_source = el.name in {"li", "blockquote", "td", "th"}
        src_el = soup.new_tag("span" if line_wrapper else "div" if nested_source else "p")
        src_el["class"] = ["tn-source", "ibooks-dark-theme-use-custom-text-color"]
        _append_source(src_el, src, source_markup)
        if line_wrapper and order == "source_first":
            el.insert_before(src_el)
            src_el.insert_after(soup.new_tag("br"))
        elif line_wrapper:
            el.insert_after(src_el)
            el.insert_after(soup.new_tag("br"))
        elif nested_source and order == "source_first":
            el.insert(0, src_el)
        elif nested_source:
            el.append(src_el)
        elif order == "source_first":
            el.insert_before(src_el)
        else:
            el.insert_after(src_el)
    # br 拆行包装只用于提供独立回填锚点；完成后去掉 span，恢复干净 DOM。
    for wrapper in list(soup.find_all(True, attrs={_LINE_WRAPPER_ATTR: True})):
        wrapper.unwrap()
    return str(soup)


def _render_chapter_html(
    chapter: Chapter,
    *,
    bilingual: bool = False,
    order: str = "target_first",
    source_lang: str = "",
) -> str:
    """回填旧版“每章一个模板”的 HTML/EPUB 章节（仅用于 schema 1 状态）。

    schema 2 状态的物理资源改由 :func:`_render_segments_html` 按聚合后的
    Segment 一次性回填，见 ``_assemble_epub``。
    """
    return _render_segments_html(
        chapter.template or "",
        chapter.segments,
        bilingual=bilingual,
        order=order,
        source_lang=source_lang,
    )


def _segments_by_resource(chapters: list[Chapter]) -> dict[str, list[Segment]]:
    """按源文顺序，将各逻辑章节中的 EPUB Segment 按物理资源分组。"""
    grouped: dict[str, list[Segment]] = {}
    for chapter in chapters:
        for segment in chapter.segments:
            href = segment.resource_href
            if href:
                grouped.setdefault(href, []).append(segment)
    return grouped


def _base_no_frag(href: str) -> str:
    """取 href 的文件名（去目录、去 #锚点），用于跨文件相对路径匹配。"""
    return os.path.basename((href or "").split("#", 1)[0])


def _attr_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _rewrite_opf_metadata(
    data: bytes,
    *,
    book_title: str,
    lang: str,
    force_horizontal: bool,
) -> bytes:
    """更新 OPF 元数据：书名可选改写，译后语言改为目标语言，竖排源书改横排方向。"""
    try:
        soup = BeautifulSoup(data, "xml")
        if book_title:
            title_el = soup.find("dc:title") or soup.find("title")
            if title_el is not None:
                title_el.clear()
                title_el.append(book_title)

        lang_el = soup.find("dc:language") or soup.find("language")
        if lang_el is None:
            metadata = soup.find("metadata")
            if metadata is not None:
                lang_el = soup.new_tag("dc:language")
                metadata.append(lang_el)
        if lang_el is not None:
            lang_el.clear()
            lang_el.append(lang)

        if force_horizontal:
            for spine in soup.find_all("spine"):
                spine["page-progression-direction"] = "ltr"
        return soup.encode()
    except Exception:
        return data


def _epub_looks_vertical(zf: zipfile.ZipFile) -> bool:
    """粗略检测 EPUB 是否声明了竖排排版。"""
    for info in zf.infolist():
        low = info.filename.lower()
        if not low.endswith((".opf", ".css", ".xhtml", ".html", ".htm")):
            continue
        try:
            data = zf.read(info.filename)
        except Exception:
            continue
        if any(marker.search(data) for marker in _VERTICAL_MARKERS):
            return True
    return False


def _rewrite_html_document(
    data: bytes | str,
    *,
    lang: str,
    force_horizontal: bool,
    bilingual: bool = False,
    rewrite_language: bool = True,
) -> bytes:
    """Rewrite HTML language/layout and optionally inject the bilingual style."""
    try:
        if isinstance(data, bytes):
            text = UnicodeDammit(data).unicode_markup
            if text is None:
                text = data.decode("utf-8", errors="replace")
        else:
            text = data
        soup = BeautifulSoup(text, "html.parser")
        html = soup.find("html")
        if html is None:
            return text.encode("utf-8")
        if rewrite_language:
            html["lang"] = lang
            html["xml:lang"] = lang
        classes = html.get("class")
        if force_horizontal and isinstance(classes, list) and "vrtl" in classes:
            html["class"] = [c for c in classes if c != "vrtl"]
        if force_horizontal and soup.find(id=_HORIZONTAL_OVERRIDE_ID) is None:
            head = soup.find("head")
            if head is None:
                head = soup.new_tag("head")
                html.insert(0, head)
            style = soup.new_tag("style", id=_HORIZONTAL_OVERRIDE_ID)
            style.string = _HORIZONTAL_OVERRIDE_CSS
            head.append(style)

        if bilingual and soup.find(id=_BILINGUAL_STYLE_ID) is None:
            head = soup.find("head")
            if head is None:
                head = soup.new_tag("head")
                html.insert(0, head)
            style = soup.new_tag("style", id=_BILINGUAL_STYLE_ID)
            style.string = _BILINGUAL_CSS
            head.append(style)
        output = _XML_ENCODING.sub(r'\1"utf-8"', str(soup))
        return output.encode("utf-8")
    except Exception:
        return data if isinstance(data, bytes) else data.encode("utf-8")


def _direct_child(parent: Tag | BeautifulSoup, name: str) -> Tag | None:
    """返回 ``parent`` 的首个指定直接子元素。"""
    child = parent.find(name, recursive=False)
    return child if isinstance(child, Tag) else None


def _nav_label_nodes(soup: BeautifulSoup) -> list[tuple[Tag, str]]:
    """按 preorder 列出 EPUB3 NAV 目录条目标签及原始 href。

    枚举顺序复用 ``epub_toc.nav_toc_scopes``/``nav_root_list`` 定位规则，
    并按 ``epub_toc._parse_nav`` 同样的 ``li`` 直接子 ``a``/``span`` 规则
    遍历，保证此处的 node_index 与解析阶段完全一致。
    """
    labels: list[tuple[Tag, str]] = []

    def walk_list(ordered_list: Tag) -> None:
        for li in ordered_list.find_all("li", recursive=False):
            if not isinstance(li, Tag):
                continue
            label = _direct_child(li, "a") or _direct_child(li, "span")
            if label is not None:
                labels.append((label, _attr_str(label.get("href"))))
            nested = _direct_child(li, "ol")
            if nested is not None:
                walk_list(nested)

    for scope in nav_toc_scopes(soup):
        root = nav_root_list(scope)
        if root is not None:
            walk_list(root)
    return labels


def _ncx_nav_points(soup: BeautifulSoup) -> list[Tag]:
    """按 preorder 列出 NCX ``navPoint``，遍历规则与 ``epub_toc._parse_ncx`` 一致。"""
    nav_map = soup.find("navMap")
    if not isinstance(nav_map, Tag):
        return []
    points: list[Tag] = []

    def walk(parent: Tag) -> None:
        for child in parent.children:
            if not isinstance(child, Tag) or child.name != "navPoint":
                continue
            points.append(child)
            walk(child)

    walk(nav_map)
    return points


def _translated_toc_title(entry: dict[str, object]) -> str:
    """返回目录条目的有效译名（标题编号统一为汉字），缺失时回退原标题。"""
    value = entry.get("title_translated") or entry.get("title")
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    return normalize_heading_numbering(stripped) if stripped else ""


def _indexed_toc_entries(
    entries: list[dict[str, object]], toc_path: str
) -> dict[int, dict[str, object]]:
    """按 ``toc_path + node_index`` 建立目录节点的精确索引。"""
    indexed: dict[int, dict[str, object]] = {}
    for entry in entries:
        if entry.get("toc_path") != toc_path:
            continue
        node_index = entry.get("node_index")
        if isinstance(node_index, int) and node_index >= 0:
            indexed[node_index] = entry
    return indexed


def _toc_kind_at(toc_entries: list[dict[str, object]], name: str) -> str | None:
    """返回目录节点中 ``toc_path == name`` 的 ``kind``（``"ncx"``/``"nav"``）。

    未匹配到该 zip 成员的精确条目时返回 ``None``，调用方据此改用后缀判断
    （兼容旧状态）。同一 ``toc_path`` 下所有条目的 ``kind`` 相同，取首条即可。
    """
    for entry in toc_entries:
        if entry.get("toc_path") == name:
            kind = entry.get("kind")
            return kind if isinstance(kind, str) else None
    lowered = name.lower()
    if lowered.endswith(".ncx"):
        return "ncx"
    if lowered.endswith(("/nav.xhtml", "/nav.html", "nav.xhtml", "nav.html")):
        return "nav"
    return None


def _rewrite_toc(
    data: bytes,
    entries_or_legacy_titles: list[dict[str, object]] | dict[str, str],
    *,
    is_ncx: bool,
    toc_path: str = "",
) -> bytes:
    """回填 NCX/NAV 的可见标题，``src``/``href`` 属性原样保留。

    新状态传入目录项列表：按 ``toc_path + node_index`` 精确定位节点，
    同一 XHTML 中的多个 fragment 分别使用对应译名；回填前核对 ``raw_href``
    是否与源文件一致，不一致（状态与源书不匹配）时跳过该节点，不误改。
    传入 ``{basename: title}`` 字典时使用旧版模式，沿用按 href 文件名
    匹配的逻辑，供 schema 1 旧状态导出使用。
    """
    try:
        exact_entries = (
            _indexed_toc_entries(entries_or_legacy_titles, toc_path)
            if isinstance(entries_or_legacy_titles, list)
            else {}
        )
        legacy_titles = (
            entries_or_legacy_titles if isinstance(entries_or_legacy_titles, dict) else {}
        )
        if is_ncx:
            soup = BeautifulSoup(data, "xml")
            for node_index, nav_point in enumerate(_ncx_nav_points(soup)):
                nav_label = _direct_child(nav_point, "navLabel")
                label = nav_label.find("text") if nav_label is not None else None
                if not isinstance(label, Tag):
                    continue
                content = _direct_child(nav_point, "content")
                entry = exact_entries.get(node_index)
                if entry is not None:
                    raw_src = _attr_str(content.get("src")) if content else ""
                    expected = entry.get("raw_href")
                    if isinstance(expected, str) and expected != raw_src:
                        continue  # 状态与源书不匹配，宁可保留原标题也不改错节点
                    title = _translated_toc_title(entry)
                else:
                    title = legacy_titles.get(
                        _base_no_frag(_attr_str(content.get("src")) if content else "")
                    )
                if title:
                    label.clear()
                    label.append(title)
            return soup.encode()

        # EPUB3 nav.xhtml：只改 epub:type="toc" 的导航，避免误改 landmarks / page-list
        soup = BeautifulSoup(data, "html.parser")
        if legacy_titles:
            toc_navs = [
                n
                for n in soup.find_all("nav")
                if "toc" in (_attr_str(n.get("epub:type")) or _attr_str(n.get("type"))).split()
            ]
            scopes = toc_navs or [soup]  # 找不到带类型的 toc nav 时退回全局
            for scope in scopes:
                for a in scope.find_all("a", href=True):
                    t = legacy_titles.get(_base_no_frag(_attr_str(a.get("href"))))
                    if t:
                        a.clear()
                        a.append(t)
            return str(soup).encode("utf-8")
        for node_index, (label, raw_href) in enumerate(_nav_label_nodes(soup)):
            entry = exact_entries.get(node_index)
            if entry is None:
                continue
            expected = entry.get("raw_href")
            if isinstance(expected, str) and expected != raw_href:
                continue  # 状态与源书不匹配，宁可保留原标题也不改错节点
            title = _translated_toc_title(entry)
            if title:
                label.clear()
                label.append(title)
        return str(soup).encode("utf-8")
    except Exception:
        return data


def _parse_source_markup(data: bytes, expected_mode: str | None = None):
    from trans_novel.ingest.epub_reader import _resource_parser

    tree, mode, _diagnostics = _resource_parser(data)
    if expected_mode and mode != expected_mode:
        raise ValueError(f"EPUB parse mode mismatch: expected {expected_mode}, got {mode}")
    return tree, mode


def _element_children_lxml(element: etree._Element) -> list[etree._Element]:
    return [child for child in element if isinstance(child.tag, str)]


def _resolve_element_path(root: etree._Element, path: tuple[int, ...]) -> etree._Element:
    current = root
    for index in path:
        children = _element_children_lxml(current)
        if index < 0 or index >= len(children):
            raise ValueError("EPUB block/slot locator mismatch")
        current = children[index]
    return current


def _serialize_source_tree(tree: etree._ElementTree, data: bytes, mode: str) -> bytes:
    probe = data
    for bom in (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff"):
        if probe.startswith(bom):
            probe = probe[len(bom) :]
            break
    declaration = bool(re.match(rb"\s*<\?xml\b", probe))
    if mode == "recovered":
        return etree.tostring(tree, encoding="UTF-8", xml_declaration=declaration, method="xml")
    encoding = tree.docinfo.encoding or "UTF-8"
    return etree.tostring(
        tree,
        encoding=encoding,
        xml_declaration=declaration,
        doctype=tree.docinfo.doctype or None,
        method="xml",
    )


def _rewrite_markup_languages(tree: etree._Element, target_lang: str) -> None:
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    root = tree.getroottree().getroot()
    for key in ("lang", xml_lang):
        if key in root.attrib:
            root.attrib[key] = target_lang


def _set_visible_label(node: etree._Element, title: str) -> None:
    node.text = title
    for descendant in node.iterdescendants():
        descendant.text = None
        descendant.tail = None


def _rewrite_toc_lxml(
    data: bytes,
    entries: list[dict[str, object]],
    *,
    is_ncx: bool,
    toc_path: str,
    target_lang: str,
    expected_mode: str | None = None,
) -> bytes:
    tree, mode = _parse_source_markup(data, expected_mode)
    root = tree.getroot()
    indexed = _indexed_toc_entries(entries, toc_path)
    if is_ncx:
        points = [
            node
            for node in root.iter()
            if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1] == "navPoint"
        ]
        for node_index, point in enumerate(points):
            entry = indexed.get(node_index)
            if entry is None:
                continue
            content = next(
                (
                    child
                    for child in point.iter()
                    if isinstance(child.tag, str) and child.tag.rsplit("}", 1)[-1] == "content"
                ),
                None,
            )
            expected = entry.get("raw_href")
            if isinstance(expected, str) and content is not None and content.get("src") != expected:
                raise ValueError(f"EPUB TOC href mismatch in {toc_path}")
            label = next(
                (
                    node
                    for node in point.iter()
                    if isinstance(node.tag, str) and node.tag.rsplit("}", 1)[-1] == "text"
                ),
                None,
            )
            title = _translated_toc_title(entry)
            if label is not None and title:
                _set_visible_label(label, title)

    else:
        labels: list[tuple[etree._Element, str]] = []
        from trans_novel.ingest.epub_reader import _nav_toc_roots

        def visit_li(li: etree._Element) -> None:
            label = next(
                (
                    child
                    for child in li
                    if isinstance(child.tag, str)
                    and child.tag.rsplit("}", 1)[-1].lower() in {"a", "span"}
                ),
                None,
            )
            if label is not None:
                labels.append((label, _attr_local_lxml(label, "href")))
            child_ol = next(
                (
                    child
                    for child in li
                    if isinstance(child.tag, str) and child.tag.rsplit("}", 1)[-1].lower() == "ol"
                ),
                None,
            )
            if child_ol is not None:
                for child in child_ol:
                    if isinstance(child.tag, str) and child.tag.rsplit("}", 1)[-1].lower() == "li":
                        visit_li(child)

        for root_list in _nav_toc_roots(root):
            for li in root_list:
                if isinstance(li.tag, str) and li.tag.rsplit("}", 1)[-1].lower() == "li":
                    visit_li(li)
        for node_index, (label, raw_href) in enumerate(labels):
            entry = indexed.get(node_index)
            if entry is None:
                continue
            expected = entry.get("raw_href")
            if isinstance(expected, str) and expected != raw_href:
                raise ValueError(f"EPUB TOC href mismatch in {toc_path}")
            title = _translated_toc_title(entry)
            if title:
                _set_visible_label(label, title)
    _rewrite_markup_languages(root, target_lang)
    return _serialize_source_tree(tree, data, mode)


def _resolve_slot_owner(
    block: etree._Element, path: tuple[int, ...], field: str, source_value: str
) -> etree._Element:
    owner = _resolve_element_path(block, path)
    if field != "tail" or owner.tail == source_value:
        return owner
    matches = [
        node for node in block.iter() if not isinstance(node.tag, str) and node.tail == source_value
    ]
    if len(matches) != 1:
        raise ValueError("EPUB slot locator mismatch")
    return matches[0]


def _attr_local_lxml(element: etree._Element, name: str) -> str:
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].split(":", 1)[-1] == name:
            return value
    return ""


def _sanitized_source_copy(original: etree._Element, block: etree._Element) -> etree._Element:
    """Copy only safe source text/inline markup from the original block."""
    return sanitized_source_copy(original)


def _japanese_ruby_source_copy(
    original: etree._Element,
    source_lang: str,
    source_tag: str,
) -> etree._Element | None:
    """Build the shared canonical Japanese ruby source subtree."""
    return japanese_ruby_source_copy(original, source_lang, source_tag)


def _bilingual_source_copy(
    original: etree._Element,
    block: etree._Element,
    *,
    source_lang: str,
    source_tag: str,
) -> etree._Element:
    copied = _japanese_ruby_source_copy(original, source_lang, source_tag)
    if copied is None:
        copied = sanitized_source_copy(original, source_tag)
    return copied


def _eligible_bilingual_segment(segment: Segment) -> bool:
    return segment_needs_source(segment)


def _add_bilingual_sources(
    root: etree._Element,
    segments: list[Segment],
    *,
    order: str = "target_first",
    source_lang: str = "",
    source_blocks: dict[tuple[int, ...], etree._Element] | None = None,
    block_refs: dict[tuple[int, ...], etree._Element] | None = None,
) -> int:
    """Add source copies while preserving ruby/inline structure and ordering."""
    grouped: dict[tuple[int, ...], list[Segment]] = {}
    for segment in segments:
        if not _eligible_bilingual_segment(segment):
            continue
        state = segment.epub_state
        assert state is not None
        grouped.setdefault(state.block_path, []).append(segment)
    added = 0

    for block_path, block_segments in grouped.items():
        block = block_refs.get(block_path) if block_refs else None
        if block is None:
            block = _resolve_element_path(root, block_path)
        original = source_blocks.get(block_path) if source_blocks else None
        tag = block.tag if isinstance(block.tag, str) else "p"
        container = _is_bilingual_container_tag(tag)
        direct_br = any(
            isinstance(child.tag, str) and child.tag.rsplit("}", 1)[-1].lower() == "br"
            for child in (original if original is not None else block)
        )

        # Container entries have one legal nested source block.  This is also
        # the direct-br representation: preserving the original ``br`` and
        # inline descendants in one sanitized copy avoids invalid siblings
        # under ``tr``/``ul``/``ol``/``dl``.
        if container:
            namespace = block.nsmap.get(None)
            source_tag = "div"
            source_name = f"{{{namespace}}}{source_tag}" if namespace else source_tag
            source = (
                _bilingual_source_copy(
                    original,
                    block,
                    source_lang=source_lang,
                    source_tag=source_name,
                )
                if original is not None
                else etree.Element(source_name)
            )
            if original is None:
                source.text = block_segments[0].source
            source.set("class", BILINGUAL_SOURCE_CLASS)
            source.tail = None
            if order == "source_first":
                source.tail = block.text
                block.text = None
                block.insert(0, source)
            else:
                block.append(source)
            added += 1
            continue

        # Direct-br runs preserve the original inline owners.  Target wrappers
        # stay at the slot owner (so an original link/ruby still renders the
        # translated text), while source wrappers are lifted outside the
        # outermost active owner.  All owner/boundary references are captured
        # before the first insertion; persisted slot paths remain authoritative.
        if direct_br:
            namespace = block.nsmap.get(None)
            span_name = f"{{{namespace}}}span" if namespace else "span"
            owner_map: dict[int, list[tuple[object, etree._Element, etree._Element]]] = {}
            for segment in block_segments:
                state = segment.epub_state
                assert state is not None
                entries: list[tuple[object, etree._Element, etree._Element]] = []
                for slot in state.slots:
                    owner = _resolve_element_path(block, slot.element_path)
                    boundary = direct_run_boundary(block, owner)
                    entries.append((slot, owner, boundary))
                owner_map[id(segment)] = entries

            pending_sources: list[
                tuple[
                    object,
                    etree._Element,
                    etree._Element,
                    etree._Element,
                    etree._Element,
                ]
            ] = []
            for segment in block_segments:
                state = segment.epub_state
                assert state is not None
                for slot, owner, boundary in owner_map[id(segment)]:
                    original_owner = (
                        _resolve_element_path(original, slot.element_path)
                        if original is not None
                        else owner
                    )
                    source = direct_run_source_copy(
                        original if original is not None else block,
                        original_owner,
                        source_lang=source_lang,
                        source_tag=span_name,
                        source_value=slot.source_value,
                        ruby_source=slot.field == "text",
                    )
                    target = etree.Element(span_name, **BILINGUAL_DIRECT_TARGET_ATTRS)
                    target_core = (
                        slot.target_core if slot.target_core is not None else slot.source_core
                    )
                    target.text = slot.leading_whitespace + target_core + slot.trailing_whitespace
                    if slot.field == "text":
                        owner.text = None
                        owner.insert(0, target)
                    else:
                        owner.tail = None
                        parent = owner.getparent()
                        if parent is None:
                            raise ValueError("EPUB direct-br tail slot has no parent")
                        parent.insert(parent.index(owner) + 1, target)
                    pending_sources.append((slot, owner, boundary, source, target))

            # Source wrappers around active boundaries are siblings of those
            # boundaries.  Safe text owners keep both wrappers local, while a
            # safe tail owner uses its ordinary sibling boundary.
            grouped_sources: dict[
                int, list[tuple[object, etree._Element, etree._Element, etree._Element]]
            ] = {}
            for slot, owner, boundary, source, target in pending_sources:
                if (
                    slot.field == "text"
                    and boundary is owner
                    and boundary is not block
                    and not direct_run_is_active(boundary)
                ):
                    if order == "source_first":
                        owner.insert(0, source)
                    else:
                        owner.insert(owner.index(target) + 1, source)
                    continue
                if (
                    slot.field == "tail"
                    and boundary is owner
                    and boundary is not block
                    and not direct_run_is_active(boundary)
                ):
                    parent = owner.getparent()
                    if parent is None:
                        raise ValueError("EPUB direct-br tail source has no parent")
                    index = parent.index(target)
                    parent.insert(index if order == "source_first" else index + 1, source)
                    continue
                grouped_sources.setdefault(id(boundary), []).append(
                    (slot, boundary, source, target)
                )
            for entries in grouped_sources.values():
                boundary = entries[0][1]
                parent = boundary if boundary is block else boundary.getparent()
                if parent is None:
                    raise ValueError("EPUB direct-br source boundary has no parent")
                ordered = list(reversed(entries)) if order == "target_first" else entries
                for _slot, _boundary, source, target in ordered:
                    if boundary is block:
                        target_index = parent.index(target)
                        parent.insert(
                            target_index if order == "source_first" else target_index + 1,
                            source,
                        )
                    elif order == "source_first":
                        boundary.addprevious(source)
                    elif target.getparent() is parent:
                        target.addnext(source)
                    else:
                        boundary.addnext(source)
            added += len(pending_sources)
            continue

        sources: list[etree._Element] = []
        block_name = tag.rsplit("}", 1)[-1].lower()
        source_tag = block_name if block_name in {"p", "div"} else "p"
        for segment in block_segments:
            namespace = block.nsmap.get(None)
            source_name = f"{{{namespace}}}{source_tag}" if namespace else source_tag
            if original is not None and len(block_segments) == 1:
                source = _bilingual_source_copy(
                    original,
                    block,
                    source_lang=source_lang,
                    source_tag=source_name,
                )
            else:
                source = etree.Element(source_name)
                source.text = segment.source
            source.set("class", BILINGUAL_SOURCE_CLASS)
            source.tail = None
            sources.append(source)

        added += len(sources)
        if order == "source_first":
            for source in reversed(sources):
                block.addprevious(source)
        else:
            old_tail = block.tail
            block.tail = None
            anchor = block
            for index, source in enumerate(sources):
                if index == len(sources) - 1:
                    source.tail = old_tail
                anchor.addnext(source)
                anchor = source
    return added


def _render_source_resource(
    data: bytes,
    href: str,
    segments: list[Segment],
    *,
    expected_digest: str,
    expected_mode: str,
    target_lang: str,
    bilingual: bool = False,
    order: str = "target_first",
    source_lang: str = "",
) -> bytes:
    if order not in {"target_first", "source_first"}:
        raise ValueError(f"invalid bilingual order: {order!r}")
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError(f"EPUB resource digest mismatch: {href}")
    tree, mode = _parse_source_markup(data, expected_mode)
    root = tree.getroot()
    if bilingual and has_reserved_source_collision(root):
        raise ValueError(f"EPUB reserved bilingual marker collision: {href}")
    writes: list[tuple[etree._Element, str, str]] = []
    source_blocks: dict[tuple[int, ...], etree._Element] = {}
    block_refs: dict[tuple[int, ...], etree._Element] = {}
    for segment in segments:
        state = segment.epub_state
        if state is None:
            raise ValueError(f"EPUB segment missing slot state: {href}")
        if state.resource_href != href or state.resource_sha256 != actual_digest:
            raise ValueError(f"EPUB segment resource contract mismatch: {href}")
        if state.slot_contract_sha256 != _slot_contract_digest(state.slots):
            raise ValueError(f"EPUB slot contract digest mismatch: {href}")
        block = _resolve_element_path(root, state.block_path)
        block_refs.setdefault(state.block_path, block)
        if bilingual and state.block_path not in source_blocks:
            source_blocks[state.block_path] = deepcopy(block)
        expected_fingerprint = hashlib.sha256(
            etree.tostring(block, encoding="utf-8", with_tail=False)
        ).hexdigest()
        if expected_fingerprint != state.block_fingerprint:
            raise ValueError(f"EPUB block fingerprint mismatch: {href}")
        if segment.source != _normalized_slot_text(state.slots, target=False):
            raise ValueError(f"EPUB segment source derivation mismatch: {href}")
        expected_target = _normalized_slot_text(state.slots, target=True)
        if segment.target is None:
            if any(slot.target_core is not None for slot in state.slots):
                raise ValueError(f"EPUB segment target derivation mismatch: {href}")
        elif segment.target != expected_target:
            raise ValueError(f"EPUB segment target derivation mismatch: {href}")
        for slot in state.slots:
            owner = _resolve_element_path(block, slot.element_path)
            value = owner.text if slot.field == "text" else owner.tail
            if value != slot.source_value:
                raise ValueError(f"EPUB slot source mismatch: {href}")
            core = slot.target_core if slot.target_core is not None else slot.source_core
            if segment.kind == KIND_HEADING:
                core = normalize_heading_numbering(core)
            if not slot.source_core.strip() and core.strip():
                raise ValueError(f"EPUB whitespace-only slot target: {href}")
            writes.append(
                (owner, slot.field, slot.leading_whitespace + core + slot.trailing_whitespace)
            )
    for owner, field, replacement in writes:
        if field == "text":
            owner.text = replacement
        else:
            owner.tail = replacement
    if bilingual:
        added = _add_bilingual_sources(
            root,
            segments,
            order=order,
            source_lang=source_lang,
            source_blocks=source_blocks,
            block_refs=block_refs,
        )
        if added:
            append_bilingual_style(root)
    _rewrite_markup_languages(root, target_lang)
    return _serialize_source_tree(tree, data, mode)


def _assemble_source_epub(
    store: RunStore,
    source_path: str,
    out_path: str,
    *,
    target_lang: str,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    manifest = store.load_manifest()
    raw_meta = manifest.get("meta")
    raw_source_lang = manifest.get("source_lang", "")
    source_lang = raw_source_lang if isinstance(raw_source_lang, str) else ""
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    schema = meta.get("epub_schema")
    if not isinstance(schema, int) or schema < 3:
        raise ValueError(
            f"Unsupported EPUB state schema {schema!r}; start a fresh translation for schema 3"
        )
    if order not in {"target_first", "source_first"}:
        raise ValueError(f"invalid bilingual order: {order!r}")
    expected_archive = meta.get("epub_sha256")
    with open(source_path, "rb") as source_file:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
        actual_archive = digest.hexdigest()
    if expected_archive != actual_archive:
        raise ValueError("EPUB source archive digest mismatch")
    resources_meta = {
        str(item.get("href")): item
        for item in meta.get("epub_resources", [])
        if isinstance(item, dict) and isinstance(item.get("href"), str)
    }
    chapters = [store.load_chapter(c["index"]) for c in manifest["chapters"]]
    all_segments = [segment for chapter in chapters for segment in chapter.segments]
    deduped_segments = dedupe_segment_mappings(all_segments)
    grouped: dict[str, list[Segment]] = {}
    for segment in deduped_segments:
        if segment.epub_state is None or not segment.resource_href:
            raise ValueError("EPUB state contains a segment without schema-3 slot metadata")
        state = segment.epub_state
        if state.slot_contract_sha256 != _slot_contract_digest(state.slots):
            raise ValueError(f"EPUB slot contract digest mismatch: {segment.resource_href}")
        grouped.setdefault(segment.resource_href, []).append(segment)
    toc_entries = [entry for entry in meta.get("toc_entries", []) if isinstance(entry, dict)]
    with zipfile.ZipFile(source_path, "r") as zin, zipfile.ZipFile(out_path, "w") as zout:
        zout.comment = zin.comment
        for info in zin.infolist():
            name = info.filename
            data = zin.read(name)
            low = name.lower()
            resource_info = resources_meta.get(name)
            if resource_info is not None and hashlib.sha256(data).hexdigest() != str(
                resource_info.get("resource_sha256", "")
            ):
                raise ValueError(f"EPUB resource digest mismatch: {name}")
            if name == "mimetype":
                zout.writestr(info, data, zipfile.ZIP_STORED)
            elif low.endswith(".opf"):
                tree, mode = _parse_source_markup(data)
                _rewrite_opf_language_lxml(tree, target_lang)
                zout.writestr(info, _serialize_source_tree(tree, data, mode))
            elif name in grouped:
                resource = resources_meta.get(name)
                if resource is None:
                    raise ValueError(f"EPUB resource missing persisted metadata: {name}")
                rendered = _render_source_resource(
                    data,
                    name,
                    grouped[name],
                    expected_digest=str(resource.get("resource_sha256", "")),
                    expected_mode=str(resource.get("parse_mode", "")),
                    target_lang=target_lang,
                    bilingual=bilingual,
                    order=order,
                    source_lang=source_lang,
                )
                toc_kind = _toc_kind_at(toc_entries, name)
                if toc_kind in {"nav", "ncx"}:
                    rendered = _rewrite_toc_lxml(
                        rendered,
                        toc_entries,
                        is_ncx=toc_kind == "ncx",
                        toc_path=name,
                        target_lang=target_lang,
                    )
                zout.writestr(info, rendered)
            elif _toc_kind_at(toc_entries, name) in {"nav", "ncx"}:
                resource = resources_meta.get(name)
                expected_mode = str(resource.get("parse_mode", "")) if resource else None
                kind = _toc_kind_at(toc_entries, name)
                zout.writestr(
                    info,
                    _rewrite_toc_lxml(
                        data,
                        toc_entries,
                        is_ncx=kind == "ncx",
                        toc_path=name,
                        target_lang=target_lang,
                        expected_mode=expected_mode,
                    ),
                )
            elif name in resources_meta and low.endswith(_HTML_EXTS):
                resource = resources_meta[name]
                tree, mode = _parse_source_markup(data, str(resource.get("parse_mode", "")))
                _rewrite_markup_languages(tree.getroot(), target_lang)
                zout.writestr(info, _serialize_source_tree(tree, data, mode))
            else:
                zout.writestr(info, data)
    return out_path


def _rewrite_opf_language_lxml(tree: etree._ElementTree, target_lang: str) -> None:
    dc_language = "{http://purl.org/dc/elements/1.1/}language"
    for node in tree.getroot().iter():
        if node.tag == dc_language:
            node.text = target_lang
            break


def _assemble_epub(
    store: RunStore,
    source_path: str,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """复制原 EPUB，并按物理资源替换正文、回填目录及目标语言元数据。

    schema 2 状态（``resource_templates.json`` 非空）按 ``Segment.resource_href``
    把全书 Segment 聚合到物理 href，每个物理 XHTML 只渲染一次——天然兼容
    “一个文件含多个逻辑章”和“一章跨多个文件”。schema 1 旧状态（模板仍
    随 Chapter 存储）继续按旧版逻辑逐章渲染。
    """
    m = store.load_manifest()
    target_lang = _epub_lang(m.get("target_lang", "zh"))
    meta = m.get("meta") if isinstance(m.get("meta"), dict) else {}
    schema = meta.get("epub_schema")
    if not isinstance(schema, int) or schema < 3:
        raise ValueError(
            f"Unsupported EPUB state schema {schema!r}; start a fresh translation for schema 3"
        )
    return _assemble_source_epub(
        store,
        source_path,
        out_path,
        target_lang=target_lang,
        bilingual=bilingual,
        order=order,
    )
    raw_source_lang = m.get("source_lang", "")
    source_lang = raw_source_lang if isinstance(raw_source_lang, str) else ""
    raw_meta = m.get("meta")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    raw_toc_entries = meta.get("toc_entries", [])
    toc_entries: list[dict[str, object]] = (
        [entry for entry in raw_toc_entries if isinstance(entry, dict)]
        if isinstance(raw_toc_entries, list)
        else []
    )

    chapters = [store.load_chapter(c["index"]) for c in m["chapters"]]
    resource_templates = store.load_resource_templates()

    # href -> 渲染后的 XHTML
    rendered: dict[str, str] = {}
    if meta.get("epub_schema") == 2:
        if not resource_templates:
            raise ValueError(
                "EPUB 翻译状态使用 schema 2，但缺少 resource_templates.json（状态不完整，无法导出）"
            )
        grouped = _segments_by_resource(chapters)
        undeclared = sorted(set(grouped) - set(resource_templates))
        if undeclared:
            raise ValueError("EPUB 翻译状态引用了未登记的正文资源：" + ", ".join(undeclared[:3]))
        for href, segments in grouped.items():
            rendered[href] = _render_segments_html(
                resource_templates[href],
                segments,
                bilingual=bilingual,
                order=order,
                source_lang=source_lang,
            )
    else:
        # schema 1 旧状态：模板仍随 Chapter 存储，逐章渲染。
        for chapter in chapters:
            if chapter.href and chapter.template:
                rendered[chapter.href] = _render_chapter_html(
                    chapter,
                    bilingual=bilingual,
                    order=order,
                    source_lang=source_lang,
                )

    # 目录标题：兼容旧状态的 basename 映射（用于旧状态导出，以及精确模式未命中时的回退）。
    legacy_titles: dict[str, str] = {}
    for c in m["chapters"]:
        base = _base_no_frag(c.get("href") or "")
        t = _ch_title(c)
        if base and t:
            legacy_titles[base] = t
    for entry in toc_entries:
        href = entry.get("resource_href") or entry.get("href")
        base = _base_no_frag(href if isinstance(href, str) else "")
        title = _translated_toc_title(entry)
        if base and title:
            legacy_titles[base] = title
    book_title = ""

    with zipfile.ZipFile(source_path, "r") as zin:
        force_horizontal = _epub_looks_vertical(zin)
        infos = zin.infolist()
        with zipfile.ZipFile(out_path, "w") as zout:
            for info in infos:
                name = info.filename
                low = name.lower()
                data = zin.read(name)
                toc_kind = _toc_kind_at(toc_entries, name)
                if name == "mimetype":
                    zout.writestr(info, data, zipfile.ZIP_STORED)
                elif low.endswith(".opf"):
                    zout.writestr(
                        info,
                        _rewrite_opf_metadata(
                            data,
                            book_title=book_title,
                            lang=target_lang,
                            force_horizontal=force_horizontal,
                        ),
                    )
                elif toc_kind == "ncx" or (toc_kind is None and low.endswith(".ncx")):
                    # 优先按 toc_entries 中的 toc_path + kind 路由（OPF 可把 NCX 命名为
                    # 任意扩展名，如 toc.xml）；没有精确匹配的目录项时，才改用 .ncx 后缀判断。
                    exact = _indexed_toc_entries(toc_entries, name)
                    if exact:
                        zout.writestr(
                            info, _rewrite_toc(data, toc_entries, is_ncx=True, toc_path=name)
                        )
                    else:
                        zout.writestr(info, data)
                elif toc_kind == "nav" or (toc_kind is None and low.endswith(_HTML_EXTS)):
                    html_data = rendered[name].encode("utf-8") if name in rendered else data
                    exact = _indexed_toc_entries(toc_entries, name)
                    if exact:
                        # 存在精确匹配的目录项时，无条件使用精确模式，不依赖 _is_nav 探测
                        # （解析端 nav_toc_scopes 也能识别缺少 epub:type 的 NAV）。
                        html_data = _rewrite_toc(
                            html_data, toc_entries, is_ncx=False, toc_path=name
                        )
                    elif _is_nav(html_data):
                        # 兼容旧状态的回退逻辑：没有 toc_entries 时，根据内容特征识别 NAV。
                        html_data = _rewrite_toc(
                            html_data, legacy_titles, is_ncx=False, toc_path=name
                        )
                    zout.writestr(
                        info,
                        _rewrite_html_document(
                            html_data,
                            lang=target_lang,
                            force_horizontal=force_horizontal,
                            bilingual=bilingual,
                        ),
                    )
                else:
                    zout.writestr(info, data)
    return out_path


def _is_nav(data: bytes) -> bool:
    return b"epub:type" in data and b"toc" in data


def _inject_bilingual_style(out_path: str, chapter_filenames: set[str], lang: str) -> None:
    """ebooklib 写盘时按模板重建每章 <head>，内联样式会被丢弃；这里对写好的 zip
    做一次后处理，把双语样式补回各章节 head（复用 _rewrite_html_document）。"""
    with zipfile.ZipFile(out_path, "r") as zin:
        infos = zin.infolist()
        entries = {info.filename: zin.read(info.filename) for info in infos}
    tmp_path = out_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for info in infos:
                data = entries[info.filename]
                if os.path.basename(info.filename) in chapter_filenames:
                    data = _rewrite_html_document(
                        data,
                        lang=lang,
                        force_horizontal=False,
                        bilingual=True,
                        rewrite_language=False,
                    )
                zout.writestr(info, data)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _build_epub_from_chapters(
    store: RunStore,
    source_path: str,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """从章节数据生成规范的 EPUB 3，并恢复 FB2 中内嵌的图片资源。"""
    from html import escape

    from ebooklib import epub

    m = store.load_manifest()
    title = m.get("title", "translated")
    lang = _epub_lang(m.get("target_lang", "zh"))

    book = epub.EpubBook()
    book.set_identifier(f"trans-novel-{title}")
    book.set_title(title)
    book.set_language(lang)

    spine: list = ["nav"]
    toc: list = []
    chapter_filenames: set[str] = set()
    image_hrefs: dict[str, str] = {}
    raw_meta = m.get("meta")
    manifest_meta = raw_meta if isinstance(raw_meta, dict) else {}
    if m.get("fmt") == "fb2":
        binaries = read_fb2_binaries(source_path)
        cover_id = manifest_meta.get("fb2_cover_image")
        used_hrefs: set[str] = set()
        for index, (resource_id, (content_type, payload)) in enumerate(binaries.items()):
            stem, extension = os.path.splitext(os.path.basename(resource_id))
            safe_stem = _sanitize_filename(stem, f"image-{index}")
            extension = extension.lower() or _IMAGE_EXTENSION_BY_TYPE.get(content_type, ".bin")
            href = f"images/{safe_stem}{extension}"
            suffix = 2
            while href in used_hrefs:
                href = f"images/{safe_stem}-{suffix}{extension}"
                suffix += 1
            used_hrefs.add(href)
            image_hrefs[resource_id] = href
            if resource_id == cover_id:
                book.set_cover(href, payload, create_page=True)
            else:
                book.add_item(
                    epub.EpubItem(
                        uid=f"fb2-image-{index}",
                        file_name=href,
                        media_type=content_type,
                        content=payload,
                    )
                )
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        ch_title = _ch_title(c) or ch.title
        body_parts = []
        images_by_position: dict[int, list[str]] = {}
        raw_images = ch.meta.get("fb2_images")
        if isinstance(raw_images, list):
            for image in raw_images:
                if not isinstance(image, dict):
                    continue
                position = image.get("position")
                resource_id = image.get("id")
                if not isinstance(position, int) or not isinstance(resource_id, str):
                    continue
                href = image_hrefs.get(resource_id)
                if href:
                    images_by_position.setdefault(position, []).append(href)

        paragraphs = _merged_paragraphs(ch)
        for position, (kind, target, source) in enumerate(paragraphs):
            body_parts.extend(
                f'<div class="fb2-image"><img src="{escape(href, quote=True)}" alt=""/></div>'
                for href in images_by_position.get(position, [])
            )
            tag = "h1" if kind == KIND_HEADING else "p"
            target_html = f"<{tag}>{escape(target)}</{tag}>"
            src = _bilingual_source(source, target) if (bilingual and kind != KIND_HEADING) else ""
            if not src:
                body_parts.append(target_html)
                continue
            src_html = (
                f'<p class="tn-source ibooks-dark-theme-use-custom-text-color">{escape(src)}</p>'
            )
            if order == "source_first":
                body_parts.extend((src_html, target_html))
            else:
                body_parts.extend((target_html, src_html))
        body_parts.extend(
            f'<div class="fb2-image"><img src="{escape(href, quote=True)}" alt=""/></div>'
            for href in images_by_position.get(len(paragraphs), [])
        )
        fname = f"ch{c['index']}.xhtml"
        chapter_filenames.add(fname)
        item = epub.EpubHtml(title=ch_title, file_name=fname, lang=lang)
        item.content = (
            f'<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="{lang}">'
            f"<head><title>{escape(ch_title)}</title></head>"
            f"<body>{''.join(body_parts)}</body></html>"
        )
        book.add_item(item)
        spine.append(item)
        toc.append(item)

    book.toc = toc
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(out_path, book)
    if bilingual:
        _inject_bilingual_style(out_path, chapter_filenames, lang)
    return out_path


def assemble(
    store: RunStore,
    source_path: str,
    out_path: str | None = None,
    out_format: str = "epub",
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """生成译文文件（默认 EPUB）。

    out_format="epub"（默认）：
      - 原文是 EPUB → 按原模板回填，保留排版/资源；
      - 原文是纯文本 → 生成一个规范的 EPUB（标题 h1 + 段落 p）。
    out_format="txt"：无论原文格式，按章重建为纯文本。
    bilingual=True 时额外输出原文（淡背景块），order 控制译文/原文先后。
    """
    m = store.load_manifest()
    # 唯一权威就绪门禁：身份核验 + 完整度检查。不完整状态直接拒绝，
    # 绝不静默回退成“部分原文 + 部分译文”的混合产物。
    from trans_novel.pipeline.readiness import ensure_assemble_ready

    ensure_assemble_ready(store, source_path)
    if order not in {"target_first", "source_first"}:
        raise ValueError(f"invalid bilingual order: {order!r}")
    if out_format == "txt":
        out_path = out_path or _default_out(source_path, "txt", "", bilingual=bilingual)
        return _assemble_text(store, out_path, bilingual=bilingual, order=order)
    # epub
    from trans_novel.assemble.epub_verifier import publish_epub

    out_path = out_path or _default_out(source_path, "epub", "", bilingual=bilingual)
    if m["fmt"] == "epub":
        mode = "bilingual" if bilingual else "monolingual"
        return publish_epub(
            store,
            source_path,
            out_path,
            mode=mode,
            bilingual=bilingual,
            bilingual_order=order,
            writer=lambda temp_path: _assemble_epub(
                store,
                source_path,
                temp_path,
                bilingual=bilingual,
                order=order,
            ),
        )
    return publish_epub(
        store,
        None,
        out_path,
        mode="generated",
        bilingual=bilingual,
        bilingual_order=order,
        writer=lambda temp_path: _build_epub_from_chapters(
            store,
            source_path,
            temp_path,
            bilingual=bilingual,
            order=order,
        ),
    )
