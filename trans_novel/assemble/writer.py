"""Write translated output files.

Source EPUBs are reopened from their original archive and rendered through
the schema-3 lxml slot contract. TXT/FB2 inputs use the generated EbookLib
path. Missing translations fall back to source text so output is complete.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from copy import copy, deepcopy

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
    ruby_base_count,
    sanitized_source_copy,
    segment_needs_source,
)
from trans_novel.assemble.zip_safety import (
    ZipSafetyError,
    preflight_zip,
    read_member,
)
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
_IMAGE_EXTENSION_BY_TYPE = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
}


class _MetadataZipFile(zipfile.ZipFile):
    """Zip writer that retains each source member's complete flag word."""

    def _open_to_write(self, zinfo: zipfile.ZipInfo, force_zip64: bool = False):
        if force_zip64 and not self._allowZip64:
            raise zipfile.LargeZipFile("force_zip64 is True, but ZIP64 is disabled")
        if self._writing:
            raise ValueError("Can't write to ZIP file while another member is open")
        source_flags = zinfo.flag_bits
        zinfo.compress_size = 0
        zinfo.CRC = 0
        zinfo.flag_bits = source_flags
        if zinfo.compress_type == zipfile.ZIP_LZMA:
            zinfo.flag_bits |= zipfile._MASK_COMPRESS_OPTION_1
        if not self._seekable:
            zinfo.flag_bits |= zipfile._MASK_USE_DATA_DESCRIPTOR
        zip64 = force_zip64 or zinfo.file_size * 1.05 > zipfile.ZIP64_LIMIT
        if not self._allowZip64 and zip64:
            raise zipfile.LargeZipFile("Filesize would require ZIP64 extensions")
        if self._seekable:
            self.fp.seek(self.start_dir)
        zinfo.header_offset = self.fp.tell()
        self._writecheck(zinfo)
        self.fp.write(zinfo.FileHeader(zip64))
        self._writing = True
        return zipfile._ZipWriteFile(self, zinfo, zip64)


def _write_source_member(
    zout: _MetadataZipFile,
    info: zipfile.ZipInfo,
    data: bytes,
    *,
    compress_type: int | None = None,
) -> None:
    preserved = copy(info)
    if compress_type is not None:
        preserved.compress_type = compress_type
    zout.writestr(preserved, data)


def _is_bilingual_container_tag(tag: str) -> bool:
    return is_bilingual_container_tag(tag)


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


def _reject_output_alias(source_path: str, out_path: str) -> None:
    """Reject output paths that resolve to the authoritative input bytes."""
    source = os.path.abspath(os.fspath(source_path))
    output = os.path.abspath(os.fspath(out_path))
    try:
        if os.path.realpath(source) == os.path.realpath(output):
            raise ValueError("input and output paths must differ")
        if os.path.exists(source) and os.path.exists(output) and os.path.samefile(source, output):
            raise ValueError("input and output paths must differ")
    except OSError:
        # A missing output cannot be a hardlink alias; retain the lexical and
        # resolved-path checks above without turning path probing into a leak.
        return


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
        if isinstance(descendant.tag, str):
            descendant.text = None
            descendant.tail = None
        else:
            # Comments and processing instructions are immutable; only their
            # visible tail belongs to the authorized label text.
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
    if not is_ncx:
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
            seen_rubies: set[int] = set()
            for segment in block_segments:
                state = segment.epub_state
                assert state is not None
                for slot, owner, boundary in owner_map[id(segment)]:
                    original_owner = (
                        _resolve_element_path(original, slot.element_path)
                        if original is not None
                        else owner
                    )
                    ruby = next(
                        (
                            node
                            for node in (original_owner, *original_owner.iterancestors())
                            if node is not block
                            and isinstance(node.tag, str)
                            and node.tag.rsplit("}", 1)[-1].lower() == "ruby"
                        ),
                        None,
                    )
                    if ruby is not None and slot.field == "tail" and original_owner is ruby:
                        ruby = None
                    if ruby is not None and ruby_base_count(ruby) <= 1:
                        ruby = None
                    ruby_id = id(ruby) if ruby is not None else None
                    source = None
                    if ruby_id is None or ruby_id not in seen_rubies:
                        source = direct_run_source_copy(
                            original if original is not None else block,
                            original_owner,
                            source_lang=source_lang,
                            source_tag=span_name,
                            source_value=slot.source_value,
                            ruby_source=ruby_id is not None,
                        )
                        if ruby_id is not None:
                            seen_rubies.add(ruby_id)
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
                    if source is not None:
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
    if schema != 3:
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
    with zipfile.ZipFile(source_path, "r") as zin:
        try:
            preflight_zip(zin)
        except ZipSafetyError as exc:
            raise ValueError(f"EPUB archive rejected: {exc.code}") from exc
        with _MetadataZipFile(out_path, "w") as zout:
            zout.comment = zin.comment
            opf_path = str(meta.get("opf_path") or "")
            for info in zin.infolist():
                name = info.filename
                data = read_member(zin, info)
                low = name.lower()
                resource_info = resources_meta.get(name)
                if resource_info is not None and hashlib.sha256(data).hexdigest() != str(
                    resource_info.get("resource_sha256", "")
                ):
                    raise ValueError(f"EPUB resource digest mismatch: {name}")
                if name == "mimetype":
                    _write_source_member(zout, info, data, compress_type=zipfile.ZIP_STORED)
                elif name == opf_path:
                    tree, mode = _parse_source_markup(data)
                    _rewrite_opf_language_lxml(tree, target_lang)
                    _write_source_member(zout, info, _serialize_source_tree(tree, data, mode))
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
                    _write_source_member(zout, info, rendered)
                elif _toc_kind_at(toc_entries, name) in {"nav", "ncx"}:
                    resource = resources_meta.get(name)
                    expected_mode = str(resource.get("parse_mode", "")) if resource else None
                    kind = _toc_kind_at(toc_entries, name)
                    _write_source_member(
                        zout,
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
                    _write_source_member(zout, info, _serialize_source_tree(tree, data, mode))
                else:
                    _write_source_member(zout, info, data)
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
    """Render a schema-3 source EPUB from its original archive and slot state."""
    manifest = store.load_manifest()
    meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    schema = meta.get("epub_schema")
    if schema != 3:
        raise ValueError(
            f"Unsupported EPUB state schema {schema!r}; start a fresh translation for schema 3"
        )
    return _assemble_source_epub(
        store,
        source_path,
        out_path,
        target_lang=_epub_lang(manifest.get("target_lang", "zh")),
        bilingual=bilingual,
        order=order,
    )


def _inject_generated_bilingual_style(out_path: str, chapter_filenames: set[str]) -> None:
    """Add the bilingual style to generated EbookLib chapter documents only."""
    with zipfile.ZipFile(out_path, "r") as zin:
        infos = zin.infolist()
        entries = {info.filename: zin.read(info.filename) for info in infos}
    tmp_path = out_path + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w") as zout:
            for info in infos:
                data = entries[info.filename]
                if os.path.basename(
                    info.filename
                ) in chapter_filenames and info.filename.lower().endswith(_HTML_EXTS):
                    try:
                        tree = etree.fromstring(
                            data,
                            etree.XMLParser(
                                no_network=True,
                                recover=False,
                                resolve_entities=False,
                                remove_comments=False,
                                remove_pis=False,
                            ),
                        ).getroottree()
                        root = tree.getroot()
                        namespace = root.nsmap.get(None)

                        def qualified(name: str, namespace: str | None = namespace) -> str:
                            return f"{{{namespace}}}{name}" if namespace else name

                        head = next(
                            (
                                child
                                for child in root
                                if isinstance(child.tag, str)
                                and child.tag.rsplit("}", 1)[-1].lower() == "head"
                            ),
                            None,
                        )
                        if head is None:
                            head = etree.Element(qualified("head"))
                            root.insert(0, head)
                        if not any(
                            isinstance(style.tag, str)
                            and style.tag.rsplit("}", 1)[-1].lower() == "style"
                            and style.get("id") == BILINGUAL_STYLE_ID
                            for style in head
                        ):
                            style = etree.Element(qualified("style"), id=BILINGUAL_STYLE_ID)
                            style.text = BILINGUAL_CSS
                            head.append(style)
                        data = etree.tostring(tree, encoding="UTF-8", xml_declaration=True)
                    except (etree.XMLSyntaxError, ValueError):
                        pass
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
        _inject_generated_bilingual_style(out_path, chapter_filenames)
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
    else:
        out_path = out_path or _default_out(source_path, "epub", "", bilingual=bilingual)
    _reject_output_alias(source_path, out_path)
    if order not in {"target_first", "source_first"}:
        raise ValueError(f"invalid bilingual order: {order!r}")
    # 唯一权威就绪门禁：身份核验 + 完整度检查。不完整状态直接拒绝，
    # 绝不静默回退成“部分原文 + 部分译文”的混合产物。
    from trans_novel.pipeline.readiness import ensure_assemble_ready

    ensure_assemble_ready(store, source_path)
    if out_format == "txt":
        return _assemble_text(store, out_path, bilingual=bilingual, order=order)
    # epub
    from trans_novel.assemble.epub_verifier import publish_epub

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
        source_identity_path=source_path,
    )
