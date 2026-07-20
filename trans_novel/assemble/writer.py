"""回填：把译文写回原格式。

- 纯文本：按章重建，标题 + 段落（空行分隔）。
- EPUB：重开原始 zip，逐条目原样拷贝；schema 2 状态按物理资源 href 聚合
  全书 Segment，每个物理 XHTML 用已保存的模板渲染一次；schema 1 旧状态
  沿用逐章 chapter.template 渲染。两条路径都按 data-tn-id 锚点替换为
  译文后写回，非正文资源（图片/CSS/字体）不动。
缺失译文的段回退使用原文，保证不丢内容。
"""

from __future__ import annotations

import os
import re
import zipfile

from bs4 import BeautifulSoup, Tag, UnicodeDammit

from ..ingest.epub_toc import nav_root_list, nav_toc_scopes
from ..ingest.models import KIND_HEADING, Chapter, Segment
from ..pipeline.runstore import RunStore
from ..postprocess.punct import normalize_heading_numbering

_ILLEGAL_FN = re.compile(r'[\\/:*?"<>|\r\n\t]+')
_HTML_EXTS = (".xhtml", ".html", ".htm")
_VERTICAL_MARKERS = (
    re.compile(rb"(?:-epub-|-webkit-)?writing-mode\s*:\s*(?:vertical-rl|vertical-lr|tb-rl)", re.I),
    re.compile(rb"page-progression-direction\s*=\s*['\"]rtl['\"]", re.I),
    re.compile(rb"\bclass\s*=\s*['\"][^'\"]*\bvrtl\b", re.I),
)
_HORIZONTAL_OVERRIDE_ID = "trans-novel-horizontal-override"
_BILINGUAL_STYLE_ID = "tn-bilingual-style"
_XML_ENCODING = re.compile(
    r"(<\?xml[^>]*\bencoding\s*=\s*)(['\"])[^'\"]+\2",
    re.IGNORECASE,
)
_BILINGUAL_CSS = """\
.tn-source {
  font-size: 0.88em;
  line-height: 1.55;
  color: #6b6b6b;
  background-color: #f4f3f0;
  padding: 0.5em 0.8em;
  border-radius: 5px;
  margin: 0.2em 0 1em;
}
@media (prefers-color-scheme: dark) {
  .tn-source {
    color: #a8a8a8;
    background-color: #2a2a2a;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.14);
  }
}
"""


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
        for k, p, sr in zip(kinds, paras, srcs)
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


def _replace_block_content(el: Tag, text: str, meta: dict[str, object]) -> None:
    """替换块内文字，同时按解析阶段记录的位置恢复图片等非文本节点。"""
    raw_inline = meta.get(_INLINE_META_KEY)
    inline = raw_inline if isinstance(raw_inline, dict) else {}
    raw_nodes = inline.get("nodes")
    nodes = raw_nodes if isinstance(raw_nodes, list) else []
    source_length = inline.get("source_length")
    if not isinstance(source_length, int) or source_length < 0:
        source_length = 0

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

    el.clear()
    cursor = 0
    for target_offset, _order, node in sorted(restored):
        target_offset = min(max(target_offset, cursor), len(text))
        if target_offset > cursor:
            el.append(text[cursor:target_offset])
        el.append(node)
        cursor = target_offset
    if cursor < len(text):
        el.append(text[cursor:])


def _render_segments_html(
    template: str,
    segments: list[Segment],
    *,
    bilingual: bool = False,
    order: str = "target_first",
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
        if kind_by_anchor.get(anchor) == KIND_HEADING:
            text = normalize_heading_numbering(text)
        _replace_block_content(el, text, meta_by_anchor.get(anchor, {}))
        del el["data-tn-id"]
        if not bilingual or kind_by_anchor.get(anchor) == KIND_HEADING:
            continue
        src = _bilingual_source(src_by_anchor.get(anchor, ""), text)
        if not src:
            continue
        src_el = soup.new_tag("p")
        src_el["class"] = ["tn-source", "ibooks-dark-theme-use-custom-text-color"]
        src_el.append(src)
        if order == "source_first":
            el.insert_before(src_el)
        else:
            el.insert_after(src_el)
    return str(soup)


def _render_chapter_html(
    chapter: Chapter,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """回填旧版“每章一个模板”的 HTML/EPUB 章节（仅用于 schema 1 状态）。

    schema 2 状态的物理资源改由 :func:`_render_segments_html` 按聚合后的
    Segment 一次性回填，见 ``_assemble_epub``。
    """
    return _render_segments_html(
        chapter.template or "", chapter.segments, bilingual=bilingual, order=order
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
) -> bytes:
    """给 XHTML/HTML 写入译后语言；必要时注入横排覆盖样式/双语原文样式。"""
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
        html["lang"] = lang
        html["xml:lang"] = lang
        classes = html.get("class")
        if isinstance(classes, list) and "vrtl" in classes:
            html["class"] = [c for c in classes if c != "vrtl"]

        if force_horizontal and soup.find(id=_HORIZONTAL_OVERRIDE_ID) is None:
            head = soup.find("head")
            if head is None:
                head = soup.new_tag("head")
                html.insert(0, head)
            style = soup.new_tag("style", id=_HORIZONTAL_OVERRIDE_ID)
            style.string = (
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
                resource_templates[href], segments, bilingual=bilingual, order=order
            )
    else:
        # schema 1 旧状态：模板仍随 Chapter 存储，逐章渲染。
        for chapter in chapters:
            if chapter.href and chapter.template:
                rendered[chapter.href] = _render_chapter_html(
                    chapter, bilingual=bilingual, order=order
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
                    toc_source: list[dict[str, object]] | dict[str, str] = (
                        toc_entries if exact else legacy_titles
                    )
                    zout.writestr(info, _rewrite_toc(data, toc_source, is_ncx=True, toc_path=name))
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
                    )
                zout.writestr(info, data)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _build_epub_from_chapters(
    store: RunStore,
    out_path: str,
    *,
    bilingual: bool = False,
    order: str = "target_first",
) -> str:
    """从章节数据生成一个规范的 EPUB3（用于纯文本输入），使用 ebooklib。"""
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
    for c in m["chapters"]:
        ch = store.load_chapter(c["index"])
        ch_title = _ch_title(c) or ch.title
        body_parts = []
        for kind, target, source in _merged_paragraphs(ch):
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
    if out_format == "txt":
        out_path = out_path or _default_out(source_path, "txt", "", bilingual=bilingual)
        return _assemble_text(store, out_path, bilingual=bilingual, order=order)
    # epub
    out_path = out_path or _default_out(source_path, "epub", "", bilingual=bilingual)
    if m["fmt"] == "epub":
        return _assemble_epub(store, source_path, out_path, bilingual=bilingual, order=order)
    # fb2 / text → 从章节数据生成规范 EPUB
    return _build_epub_from_chapters(store, out_path, bilingual=bilingual, order=order)
