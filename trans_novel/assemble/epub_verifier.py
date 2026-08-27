"""Deterministic, privacy-safe EPUB verification and atomic publication.

The verifier only reads local ZIP members. It never follows URLs and never
emits input filesystem paths or source/target prose in its evidence.
"""

from __future__ import annotations

import errno
import hashlib
import lzma
import os
import posixpath
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, Tag
from lxml import etree

from trans_novel.assemble.bilingual_dom import (
    BILINGUAL_CSS as _BILINGUAL_CSS,
)
from trans_novel.assemble.bilingual_dom import (
    BILINGUAL_DIRECT_TARGET_ATTRS,
    BILINGUAL_SOURCE_CLASS,
    BILINGUAL_STYLE_ID,
    dedupe_segment_mappings,
    direct_run_boundary,
    direct_run_has_active_ancestor,
    direct_run_is_active,
    direct_run_source_copy,
    is_bilingual_container_tag,
    japanese_ruby_source_copy,
    ruby_base_count,
    sanitized_source_copy,
    segment_needs_source,
    source_node_is_valid,
    style_shape_is_valid,
)
from trans_novel.assemble.zip_safety import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_MEMBERS,
    MAX_MEMBER_BYTES,
    ZipSafetyError,
    preflight_zip,
)
from trans_novel.assemble.zip_safety import (
    read_member as _safe_read_member,
)
from trans_novel.ingest.epub_toc import nav_toc_scopes

_SCHEMA_VERSION = 1
_CATEGORIES = (
    "zip",
    "resources",
    "spine",
    "nav",
    "internal_links",
    "anchors",
    "footnotes",
    "assets",
    "placeholders",
    "parse",
    "bilingual_source",
)
_MAX_MEMBER_BYTES = MAX_MEMBER_BYTES
_MAX_ARCHIVE_BYTES = MAX_ARCHIVE_BYTES
_MAX_ARCHIVE_MEMBERS = MAX_ARCHIVE_MEMBERS
_INTERNAL_ATTRIBUTES = {"data-tn-id", "data-tn-inline-id", "data-tn-line"}
_EXTERNAL_SCHEMES = {"http", "https", "mailto", "data"}
_HTML_MEDIA = {"application/xhtml+xml", "text/html"}
_NCX_MEDIA = "application/x-dtbncx+xml"
_BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "td", "th", "dt", "dd"}
_BLOCK_CANDIDATE_TAGS = _BLOCK_TAGS | {"div"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


class _MemberError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _archive_label(path: str) -> str:
    """Return a stable archive-relative label, never an OS path."""
    if not path or "\x00" in path or "\\" in path or path.startswith("/"):
        return "<opaque:" + hashlib.sha256(path.encode("utf-8", "replace")).hexdigest()[:16] + ">"
    normalized = posixpath.normpath(path)
    if normalized == ".." or normalized.startswith("../"):
        return "<opaque:" + hashlib.sha256(path.encode("utf-8", "replace")).hexdigest()[:16] + ">"
    return normalized


def _item(category: str, code: str, path: str, detail: str) -> dict[str, str]:
    return {"category": category, "code": code, "path": _archive_label(path), "detail": detail}


def _sort_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        items, key=lambda item: tuple(item[key] for key in ("category", "code", "path", "detail"))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, as_bytes: bool = True) -> bytes:
    """Read a member through the shared bounded CRC/decompression guard."""
    try:
        data = _safe_read_member(zf, info, max_member_bytes=_MAX_MEMBER_BYTES)
    except ZipSafetyError as exc:
        raise _MemberError(exc.code) from None
    return data if as_bytes else b""


def _member_hash(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    try:
        if info.file_size > _MAX_MEMBER_BYTES:
            raise _MemberError("member_too_large")
        with zf.open(info, "r") as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, _MAX_MEMBER_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_MEMBER_BYTES:
                    raise _MemberError("member_too_large")
                digest.update(chunk)
    except _MemberError:
        raise
    except (
        OSError,
        RuntimeError,
        NotImplementedError,
        EOFError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ValueError,
        zlib.error,
        lzma.LZMAError,
    ) as exc:
        raise _MemberError(
            "crc_error" if type(exc).__name__ in {"BadZipFile", "BadCRC"} else "member_read"
        ) from None
    if total != info.file_size:
        raise _MemberError("member_size_mismatch")
    return digest.hexdigest(), total


def _parse_xml(data: bytes) -> ET.Element | None:
    try:
        return ET.fromstring(data)
    except (ET.ParseError, ValueError, TypeError):
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_archive_name(name: str) -> bool:
    from trans_novel.assemble.zip_safety import safe_name

    return safe_name(name)


def _resolve(base: str, reference: str) -> tuple[str | None, str | None, str | None]:
    """Return (archive path, decoded fragment, scheme/error marker)."""
    raw = reference.strip()
    if "\x00" in raw or re.search(r"%(?![0-9A-Fa-f]{2})", raw):
        return None, None, "unsafe"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None, None, "unsafe"
    scheme = parsed.scheme.lower()
    if scheme in _EXTERNAL_SCHEMES:
        return None, unquote(parsed.fragment) if parsed.fragment else None, scheme
    if scheme or parsed.netloc:
        return None, None, scheme or "unsupported"
    decoded_path = unquote(parsed.path)
    joined = (
        base
        if not decoded_path
        else posixpath.normpath(posixpath.join(posixpath.dirname(base), decoded_path))
    )
    if not _safe_archive_name(joined):
        return None, None, "unsafe"
    fragment = unquote(parsed.fragment) if parsed.fragment else None
    return joined, fragment, None


def _manifest_href(opf_path: str, href: str) -> str | None:
    target, _, external = _resolve(opf_path, href)
    return target if external is None else None


def _manifest_items(root: ET.Element) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for element in root.iter():
        if _local_name(element.tag) != "item":
            continue
        attrs = dict(element.attrib)
        result.append(
            {
                "id": attrs.get("id", "").strip(),
                "href": attrs.get("href", "").strip(),
                "media": attrs.get("media-type", "").strip(),
                "properties": attrs.get("properties", "").strip(),
                "fallback": attrs.get("fallback", "").strip(),
                "media_overlay": attrs.get("media-overlay", "").strip(),
                "attrs": attrs,
            }
        )
    return result


def _content_model(root: ET.Element, opf_path: str, archive: set[str]) -> dict[str, Any]:
    items = _manifest_items(root)
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        if item["id"] and item["id"] not in by_id:
            by_id[item["id"]] = item
    spine_element = next((e for e in root.iter() if _local_name(e.tag) == "spine"), None)
    spine_ids = [
        e.attrib.get("idref", "").strip()
        for e in root.iter()
        if _local_name(e.tag) == "itemref" and e.attrib.get("idref", "").strip()
    ]
    resolved: list[dict[str, Any]] = []
    for item in items:
        target = _manifest_href(opf_path, item["href"])
        if target in archive:
            resolved.append({**item, "path": target})
    nav_items = [item for item in resolved if "nav" in item["properties"].split()]
    ncx_items = [item for item in resolved if item["media"] == _NCX_MEDIA]
    path_by_id = {item["id"]: item["path"] for item in resolved if item["id"]}
    spine_paths = [path_by_id[item_id] for item_id in spine_ids if item_id in path_by_id]
    return {
        "items": items,
        "resolved": resolved,
        "by_id": by_id,
        "spine_ids": spine_ids,
        "spine_paths": spine_paths,
        "spine_toc": spine_element.attrib.get("toc", "").strip()
        if spine_element is not None
        else "",
        "nav_items": nav_items,
        "ncx_items": ncx_items,
    }


def _document_root(soup: BeautifulSoup) -> Tag | None:
    for child in soup.contents:
        if isinstance(child, Tag):
            return child
    return None


def _scheme_detail(scheme: str) -> str:
    normalized = scheme.strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9+.-]{0,15}", normalized):
        return normalized
    return "scheme:" + hashlib.sha256(scheme.encode("utf-8", "replace")).hexdigest()[:16]


def _html_soup(data: bytes, media: str) -> tuple[BeautifulSoup, bool]:
    """Parse according to media type; only fatal HTML diagnostics invalidate it."""
    if media == "text/html":
        parser = etree.HTMLParser(recover=True)
        valid = True
        try:
            etree.fromstring(data, parser)
            valid = not any(entry.level_name == "FATAL" for entry in parser.error_log)
        except (UnicodeError, etree.XMLSyntaxError, ValueError, TypeError):
            valid = False
        if any(byte == 0 or (byte < 32 and byte not in {9, 10, 13}) for byte in data):
            valid = False
        return BeautifulSoup(data, "html.parser"), valid
    parser = etree.XMLParser(recover=False)
    try:
        etree.fromstring(data, parser)
        valid = True
    except (UnicodeError, etree.XMLSyntaxError, ValueError, TypeError):
        valid = False
    return BeautifulSoup(data, "xml"), valid


def _check_nav_semantics(
    soup: BeautifulSoup,
    path: str,
    failures: list[dict[str, str]],
    checked: dict[str, int],
    *,
    allow_typeless: bool = False,
) -> None:
    checked["nav"] += 1
    root = _document_root(soup)
    if root is None or root.name != "html":
        failures.append(_item("nav", "nav_root_invalid", path, "html"))
        return
    candidate_scopes = nav_toc_scopes(soup)
    scopes = [
        nav
        for nav in candidate_scopes
        if isinstance(nav, Tag)
        and nav.name == "nav"
        and "toc" in (str(nav.get("epub:type") or nav.get("type") or "")).split()
    ]
    if not scopes and allow_typeless:
        scopes = [
            nav
            for nav in candidate_scopes
            if isinstance(nav, Tag)
            and nav.name == "nav"
            and not nav.get("epub:type")
            and not nav.get("type")
        ][:1]
    if not scopes:
        failures.append(_item("nav", "nav_toc_missing", path, "toc"))
        return
    for nav in scopes:
        root_list = nav.find("ol")
        if not isinstance(root_list, Tag):
            failures.append(_item("nav", "nav_root_list_missing", path, "ol"))
            continue
        items = [
            child for child in root_list.find_all("li", recursive=False) if isinstance(child, Tag)
        ]
        if not items:
            failures.append(_item("nav", "nav_root_list_empty", path, "li"))
            continue
        has_target = False
        for item in items:
            label = item.find(["a", "span"], recursive=False)
            if not isinstance(label, Tag):
                failures.append(_item("nav", "nav_item_label_missing", path, "a_or_span"))
                continue
            if label.name == "a":
                href = str(label.get("href") or "").strip()
                if not href:
                    failures.append(_item("nav", "nav_target_missing", path, "href"))
                else:
                    has_target = True
        if not has_target and not root_list.find("a"):
            failures.append(_item("nav", "nav_target_missing", path, "href"))


def _check_ncx_semantics(
    soup: BeautifulSoup,
    path: str,
    failures: list[dict[str, str]],
    checked: dict[str, int],
) -> None:
    checked["nav"] += 1
    root = _document_root(soup)
    if root is None or root.name != "ncx":
        failures.append(_item("nav", "ncx_root_invalid", path, "ncx"))
        return
    nav_map = root.find("navMap", recursive=False)
    if not isinstance(nav_map, Tag):
        failures.append(_item("nav", "ncx_navmap_missing", path, "navMap"))
        return
    points = [point for point in nav_map.find_all("navPoint") if isinstance(point, Tag)]
    if not points:
        failures.append(_item("nav", "ncx_navpoint_missing", path, "navPoint"))
        return
    for point in points:
        content = point.find("content", recursive=False)
        if not isinstance(content, Tag) or not str(content.get("src") or "").strip():
            failures.append(_item("nav", "ncx_content_missing", path, "src"))


def _ids(soup: BeautifulSoup) -> set[str]:
    result: set[str] = set()
    for tag in soup.find_all(True):
        for attr in ("id", "name"):
            value = tag.get(attr)
            if isinstance(value, str) and value:
                result.add(value)
    return result


def _identifier_map(soup: BeautifulSoup) -> dict[str, Counter[str]]:
    result: dict[str, Counter[str]] = {"id": Counter(), "name": Counter()}
    for tag in soup.find_all(True):
        for attr in result:
            value = tag.get(attr)
            if isinstance(value, str) and value:
                result[attr][value] += 1
    return result


def _check_nesting(soup: BeautifulSoup, failures: list[dict[str, str]], path: str) -> None:
    allowed: dict[str, set[str]] = {
        "ul": {"li", "script", "template"},
        "ol": {"li", "script", "template"},
        "table": {"caption", "colgroup", "thead", "tbody", "tfoot", "tr", "script", "template"},
        "thead": {"tr", "script", "template"},
        "tbody": {"tr", "script", "template"},
        "tfoot": {"tr", "script", "template"},
        "tr": {"th", "td", "script", "template"},
    }
    for parent_name, names in allowed.items():
        for parent in soup.find_all(parent_name):
            for child in parent.find_all(recursive=False):
                if isinstance(child, Tag) and child.name not in names:
                    failures.append(
                        _item("parse", "illegal_nesting", path, f"{parent_name}>{child.name}")
                    )


def _check_document_features(
    soup: BeautifulSoup,
    path: str,
    failures: list[dict[str, str]],
    checked: dict[str, int],
    *,
    content: bool = True,
) -> dict[str, Counter[str]]:
    identifiers = _identifier_map(soup)
    for counts in identifiers.values():
        for value, count in sorted(counts.items()):
            checked["anchors"] += 1
            if count > 1:
                failures.append(_item("anchors", "duplicate_anchor", path, value))
    _check_nesting(soup, failures, path)
    for tag in soup.find_all(True):
        checked["placeholders"] += 1
        if any(attribute in tag.attrs for attribute in _INTERNAL_ATTRIBUTES):
            failures.append(_item("placeholders", "internal_attribute", path, tag.name))
    if content:
        body = soup.body
        meaningful = bool(body and body.get_text(" ", strip=True))
        if body is not None and not meaningful:
            for tag in body.find_all(["img", "svg", "audio", "video", "object", "embed"]):
                if tag.name == "svg" or any(
                    tag.get(attr) for attr in ("src", "data", "href", "xlink:href")
                ):
                    meaningful = True
                    break
        if body is None or not meaningful:
            failures.append(_item("parse", "empty_content", path, "empty_body"))
    return identifiers


def _external_warning(value: str, scheme: str, category: str = "internal_links") -> dict[str, str]:
    identifier = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]
    return _item(category, "external_skipped", "<reference>", f"{scheme}:{identifier}")


def _unsupported_scheme_detail(value: str, scheme: str) -> str:
    identifier = hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{_scheme_detail(scheme)}:{identifier}"[:64]


def _check_links(
    soup: BeautifulSoup,
    path: str,
    archive: set[str],
    ids_by_path: dict[str, set[str]],
    failures: list[dict[str, str]],
    warnings: list[dict[str, str]],
    checked: dict[str, int],
    *,
    category: str = "internal_links",
) -> Counter[tuple[str, str, str, str, str]]:
    graph: Counter[tuple[str, str, str, str, str]] = Counter()
    for tag in soup.find_all(True):
        for attr in ("href", "src", "xlink:href"):
            value = tag.get(attr)
            if not isinstance(value, str) or not value:
                continue
            checked[category] += 1
            target, fragment, external = _resolve(path, value)
            if external is not None:
                if external == "unsafe":
                    failures.append(_item(category, "unsafe_reference", path, "unsafe"))
                elif external in _EXTERNAL_SCHEMES:
                    warnings.append(_external_warning(value, external, category))
                else:
                    failures.append(
                        _item(
                            category,
                            "unsupported_scheme",
                            "<reference>",
                            _unsupported_scheme_detail(value, external),
                        )
                    )
                continue
            if target is None:
                failures.append(_item(category, "unsafe_reference", path, "unsafe"))
                continue
            graph[(path, tag.name or "", attr, target, fragment or "")] += 1
            if target not in archive:
                failures.append(_item(category, "missing_resource", path, target))
            elif fragment is not None and fragment not in ids_by_path.get(target, set()):
                failures.append(_item("anchors", "missing_fragment", path, "missing"))
    return graph


def _check_footnotes(
    soups: dict[str, BeautifulSoup],
    ids_by_path: dict[str, set[str]],
    failures: list[dict[str, str]],
    warnings: list[dict[str, str]],
    checked: dict[str, int],
) -> None:
    """Require a noteref/footnote target and a backlink to its own marker."""
    for path, soup in soups.items():
        for anchor in soup.find_all("a"):
            href = anchor.get("href")
            epub_type = str(anchor.get("epub:type", ""))
            if not isinstance(href, str) or (
                "noteref" not in epub_type and "footnote" not in href.lower()
            ):
                continue
            checked["footnotes"] += 1
            target, fragment, external = _resolve(path, href)
            if external is not None:
                if external in _EXTERNAL_SCHEMES:
                    warnings.append(_item("footnotes", "external_skipped", "<reference>", external))
                else:
                    failures.append(
                        _item("footnotes", "unsupported_scheme", path, _scheme_detail(external))
                    )
                continue
            if target is None or fragment is None:
                failures.append(_item("footnotes", "missing_target", path, "missing"))
                continue
            target_soup = soups.get(target)
            if target_soup is None or fragment not in ids_by_path.get(target, set()):
                failures.append(_item("footnotes", "missing_target", path, "missing"))
                continue
            target_tag = target_soup.find(id=fragment) or target_soup.find(attrs={"name": fragment})
            if not isinstance(target_tag, Tag):
                failures.append(_item("footnotes", "missing_target", path, "missing"))
                continue
            source_id = anchor.get("id") or anchor.get("name")
            if not source_id:
                parent = anchor.parent
                while isinstance(parent, Tag) and parent.name not in {"body", "html"}:
                    candidate = parent.get("id") or parent.get("name")
                    if candidate:
                        source_id = candidate
                        break
                    parent = parent.parent
            if not source_id:
                failures.append(_item("footnotes", "missing_backlink", path, "missing"))
                continue
            backlink = False
            for back in target_tag.find_all("a"):
                back_href = back.get("href")
                if not isinstance(back_href, str):
                    continue
                back_target, back_fragment, back_external = _resolve(target, back_href)
                if back_external is None and back_target == path and back_fragment == source_id:
                    backlink = True
                    break
            if not backlink:
                failures.append(_item("footnotes", "missing_backlink", path, "missing"))


def _resource_hashes(
    zf: zipfile.ZipFile,
    model: dict[str, Any],
    opf_path: str,
    failures: list[dict[str, str]],
    checked: dict[str, int],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    resolved = model.get("resolved")
    if not isinstance(resolved, list):
        nested = model.get("model")
        resolved = nested.get("resolved", []) if isinstance(nested, dict) else []
    for item in resolved:
        if (
            item["path"] == opf_path
            or item["media"] in _HTML_MEDIA
            or item["media"] == _NCX_MEDIA
            or "nav" in item["properties"].split()
        ):
            continue
        checked["assets"] += 1
        try:
            info = zf.getinfo(item["path"])
            digest, size = _member_hash(zf, info)
            result[item["path"]] = {"sha256": digest, "size": size, "media": item["media"]}
        except (KeyError, _MemberError) as exc:
            code = exc.code if isinstance(exc, _MemberError) else "member_read"
            failures.append(_item("zip", code, item["path"], "member"))
    return result


def _model_read(zf: zipfile.ZipFile, name: str, failures: list[dict[str, str]]) -> bytes | None:
    try:
        return _read_member(zf, zf.getinfo(name))
    except (KeyError, _MemberError) as exc:
        code = exc.code if isinstance(exc, _MemberError) else "member_read"
        failures.append(_item("zip", code, name, "member"))
        return None


def _archive_model(
    zf: zipfile.ZipFile, failures: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    failures = failures if failures is not None else []
    names = [info.filename for info in zf.infolist()]
    archive = set(names)
    container_data = (
        _model_read(zf, "META-INF/container.xml", failures)
        if "META-INF/container.xml" in archive
        else None
    )
    container = _parse_xml(container_data) if container_data is not None else None
    roots = (
        [
            e.attrib.get("full-path", "").strip()
            for e in container.iter()
            if _local_name(e.tag) == "rootfile"
        ]
        if container is not None
        else []
    )
    opf_path = roots[0] if roots else ""
    opf_data = _model_read(zf, opf_path, failures) if opf_path in archive else None
    opf = _parse_xml(opf_data) if opf_data is not None else None
    if opf is None:
        return {
            "archive": archive,
            "opf_path": opf_path,
            "model": {
                "items": [],
                "resolved": [],
                "spine_paths": [],
                "spine_ids": [],
                "spine_toc": "",
                "nav_items": [],
                "ncx_items": [],
            },
            "graphs": Counter(),
            "ids": set(),
        }
    model = _content_model(opf, opf_path, archive)
    return {"archive": archive, "opf_path": opf_path, "model": model}


def _graph_from_soups(
    soups: dict[str, BeautifulSoup],
    archive: set[str],
    ids_by_path: dict[str, set[str]],
    failures: list[dict[str, str]],
    warnings: list[dict[str, str]],
    checked: dict[str, int],
    category: str = "internal_links",
) -> Counter[tuple[str, str, str, str, str]]:
    graph: Counter[tuple[str, str, str, str, str]] = Counter()
    for resource, soup in soups.items():
        graph.update(
            _check_links(
                soup, resource, archive, ids_by_path, failures, warnings, checked, category=category
            )
        )
    return graph


def _load_doc_segments(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Use production EPUB parsing to obtain per-resource merged source blocks."""
    try:
        from trans_novel.ingest.segmenter import load_document

        doc = load_document(str(path), "en", "zh")
    except Exception:
        return {}
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for chapter in doc.chapters:
        for segment in chapter.segments:
            resource = segment.resource_href or ""
            if not resource:
                continue
            if segment.cont and grouped[resource]:
                old_kind, old_text = grouped[resource][-1]
                grouped[resource][-1] = (old_kind, old_text + segment.source)
            else:
                grouped[resource].append((segment.kind, segment.source))
    return dict(grouped)


def _leaf_line_texts(element: Tag) -> list[str]:
    if not element.find("br", recursive=False):
        return [_norm_text(element.get_text("", strip=False))]
    lines: list[str] = []
    current: list[str] = []
    for child in element.children:
        if isinstance(child, Tag) and child.name == "br":
            lines.append(_norm_text("".join(current)))
            current = []
        elif isinstance(child, Tag):
            current.append(child.get_text("", strip=False))
        else:
            current.append(str(child))
    lines.append(_norm_text("".join(current)))
    return [line for line in lines if line]


def _dom_segments(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Extract ordered leaf structural blocks directly from EPUB resources."""
    result: dict[str, list[tuple[str, str]]] = defaultdict(list)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            info = _archive_model(zf, [])
            for item in info["model"]["resolved"]:
                if item["media"] not in _HTML_MEDIA or "nav" in item["properties"].split():
                    continue
                data = _model_read(zf, item["path"], [])
                if data is None:
                    continue
                soup, _ = _html_soup(data, item["media"])
                candidates = _BLOCK_TAGS | {"div"}
                for element in soup.find_all(list(candidates)):
                    text = _norm_text(element.get_text("", strip=False))
                    if not text:
                        continue
                    has_descendant = any(
                        isinstance(descendant, Tag)
                        and descendant.name in candidates
                        and _norm_text(descendant.get_text("", strip=False))
                        for descendant in element.find_all(True)
                    )
                    if has_descendant:
                        continue
                    kind = "heading" if element.name in _HEADING_TAGS else "text"
                    for line in _leaf_line_texts(element):
                        result[item["path"]].append((kind, line))
    except (OSError, zipfile.BadZipFile):
        return {}
    return dict(result)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _source_span_target_candidates(node: Tag) -> list[str]:
    parent = node.parent
    if node.name != "span" or not isinstance(parent, Tag) or parent.name not in _BLOCK_TAGS:
        return []
    candidates: list[str] = []
    for sibling, direction in (
        (node.previous_sibling, "previous"),
        (node.next_sibling, "next"),
    ):
        crossed_separator = False
        parts: list[str] = []
        while sibling is not None:
            if isinstance(sibling, Tag):
                if sibling.name == "br":
                    if crossed_separator:
                        break
                    crossed_separator = True
                    sibling = getattr(sibling, f"{direction}_sibling", None)
                    continue
                if "tn-source" in sibling.get("class", []):
                    break
                text = _norm_text(sibling.get_text("", strip=False))
            else:
                text = _norm_text(str(sibling))
            if text:
                parts.append(text)
            sibling = getattr(sibling, f"{direction}_sibling", None)
        if parts:
            candidates.append(_norm_text(" ".join(parts)))
    return candidates


def _source_span_target_text(node: Tag) -> str | None:
    candidates = _source_span_target_candidates(node)
    return candidates[0] if candidates else None


def _source_node_attached(node: Tag) -> bool:
    parent = node.parent
    if not isinstance(parent, Tag) or parent.name in _HEADING_TAGS:
        return False
    if node.name == "span":
        return _source_span_target_text(node) is not None
    block_adjacent = any(
        isinstance(sibling, Tag) and sibling.name in _BLOCK_TAGS
        for sibling in list(node.previous_siblings) + list(node.next_siblings)
    )
    return parent.name in {"li", "blockquote", "td", "th"} or block_adjacent


def _source_subset(
    path: Path,
    source_path: Path,
    soups: dict[str, BeautifulSoup],
    failures: list[dict[str, str]],
    checked: dict[str, int],
) -> None:
    source_segments = _dom_segments(source_path)
    for resource, soup in soups.items():
        nodes = soup.select(".tn-source")
        if not nodes:
            continue
        blocks = source_segments.get(resource)
        if blocks is None:
            blocks = next(
                (value for key, value in source_segments.items() if key.endswith(resource)), []
            )
        allowed = Counter(
            hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()
            for kind, text in blocks
            if kind != "heading"
        )
        seen = Counter()
        for node in nodes:
            checked["bilingual_source"] += 1
            digest = hashlib.sha256(
                _norm_text(node.get_text("", strip=False)).encode("utf-8")
            ).hexdigest()
            seen[digest] += 1
            if seen[digest] > allowed[digest]:
                failures.append(
                    _item(
                        "bilingual_source", "source_node_unexpected", resource, "not_source_segment"
                    )
                )
        if len(nodes) > sum(allowed.values()):
            failures.append(
                _item("bilingual_source", "source_node_count", resource, "count_mismatch")
            )


def _exact_bilingual_proof(
    source_path: Path,
    mono_path: Path,
    bilingual_soups: dict[str, BeautifulSoup],
    failures: list[dict[str, str]],
    checked: dict[str, int],
) -> None:
    source_segments = _dom_segments(source_path)
    mono_segments = _dom_segments(mono_path)
    resources = sorted(set(source_segments) | set(mono_segments) | set(bilingual_soups))
    for resource in resources:
        source_blocks = source_segments.get(resource, [])
        mono_blocks = mono_segments.get(resource, [])
        if len(source_blocks) != len(mono_blocks):
            failures.append(
                _item("bilingual_source", "segment_structure_mismatch", resource, "mono")
            )
        expected: list[tuple[str, str]] = []
        for index, (kind, source_text) in enumerate(source_blocks):
            target_text = mono_blocks[index][1] if index < len(mono_blocks) else ""
            if (
                kind != "heading"
                and _norm_text(source_text)
                and _norm_text(source_text) != _norm_text(target_text)
            ):
                expected.append(
                    (
                        hashlib.sha256(_norm_text(source_text).encode("utf-8")).hexdigest(),
                        hashlib.sha256(_norm_text(target_text).encode("utf-8")).hexdigest(),
                    )
                )
        soup = bilingual_soups.get(resource)
        if soup is None:
            soup = next(
                (value for key, value in bilingual_soups.items() if key.endswith(resource)), None
            )
        actual_nodes = soup.select(".tn-source") if soup is not None else []
        observed: list[tuple[str, str]] = []
        for node in actual_nodes:
            source_norm = _norm_text(node.get_text("", strip=False))
            source_hash = hashlib.sha256(source_norm.encode("utf-8")).hexdigest()
            candidate_texts = _source_span_target_candidates(node)
            expected_target_hashes = {
                target_hash
                for expected_source_hash, target_hash in expected
                if expected_source_hash == source_hash
            }
            direct_target_hash = next(
                (
                    hashlib.sha256(candidate.encode("utf-8")).hexdigest()
                    for candidate in candidate_texts
                    if hashlib.sha256(candidate.encode("utf-8")).hexdigest()
                    in expected_target_hashes
                ),
                None,
            )
            if direct_target_hash is not None:
                observed.append((source_hash, direct_target_hash))
                continue
            if candidate_texts:
                observed.append(
                    (
                        source_hash,
                        hashlib.sha256(candidate_texts[0].encode("utf-8")).hexdigest(),
                    )
                )
                continue
            parent = node.parent
            target_tag = (
                parent
                if isinstance(parent, Tag) and parent.name in {"li", "blockquote", "td", "th"}
                else None
            )
            if (
                target_tag is None
                and isinstance(node.next_sibling, Tag)
                and (node.next_sibling.name in _BLOCK_TAGS or node.next_sibling.name == "span")
                and "tn-source" not in node.next_sibling.get("class", [])
            ):
                target_tag = node.next_sibling
            if (
                target_tag is None
                and isinstance(node.previous_sibling, Tag)
                and (
                    node.previous_sibling.name in _BLOCK_TAGS
                    or node.previous_sibling.name == "span"
                )
                and "tn-source" not in node.previous_sibling.get("class", [])
            ):
                target_tag = node.previous_sibling
            if isinstance(target_tag, Tag):
                target_norm = _norm_text(target_tag.get_text("", strip=False))
                if target_tag is parent:
                    target_norm = _norm_text(target_norm.replace(source_norm, "", 1))
                observed.append(
                    (source_hash, hashlib.sha256(target_norm.encode("utf-8")).hexdigest())
                )
        checked["bilingual_source"] += max(len(expected), len(observed), 1)
        if observed != expected:
            failures.append(
                _item("bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch")
            )
        for node in actual_nodes:
            if not _source_node_attached(node):
                failures.append(
                    _item("bilingual_source", "source_node_misplaced", resource, "unattached")
                )


def _validate_bilingual_nodes(
    soups: dict[str, BeautifulSoup],
    failures: list[dict[str, str]],
    checked: dict[str, int],
) -> None:
    """Require nonempty source nodes attached to adjacent translated blocks."""
    total = sum(len(soup.select(".tn-source")) for soup in soups.values())
    if total == 0:
        failures.append(_item("bilingual_source", "missing_source_nodes", "<output>", "missing"))
    for resource, soup in soups.items():
        for node in soup.select(".tn-source"):
            checked["bilingual_source"] += 1
            if not _norm_text(node.get_text("", strip=False)):
                failures.append(_item("bilingual_source", "source_node_empty", resource, "empty"))
                continue
            if not _source_node_attached(node):
                failures.append(
                    _item("bilingual_source", "source_node_misplaced", resource, "unattached")
                )


def _compare_source_models(
    source_model: dict[str, Any],
    output_model: dict[str, Any],
    source_assets: dict[str, dict[str, Any]],
    output_assets: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
) -> None:
    sm = source_model["model"]
    om = output_model["model"]
    source_by_path = {item["path"]: item for item in sm["resolved"]}
    output_by_path = {item["path"]: item for item in om["resolved"]}
    output_by_id = {item["id"]: item for item in om["resolved"] if item["id"]}
    for path, item in sorted(source_by_path.items()):
        other = output_by_path.get(path) or output_by_id.get(item["id"])
        if other is None:
            failures.append(_item("resources", "missing_manifest_resource", path, "source"))
            continue
        for field in ("media", "properties", "fallback", "media_overlay"):
            if item[field] != other[field]:
                failures.append(_item("resources", "manifest_metadata_mismatch", path, field))
        source_href = _manifest_href(source_model["opf_path"], item["href"])
        output_href = _manifest_href(output_model["opf_path"], other["href"])
        if source_href != output_href:
            failures.append(_item("resources", "manifest_metadata_mismatch", path, "href"))
    source_item_keys = {(item["id"], item["href"], item["media"]) for item in sm["items"]}
    for item in om["items"]:
        key = (item["id"], item["href"], item["media"])
        if key not in source_item_keys and not (
            item["id"] == "tn-bilingual-style" and item["media"] == "text/css"
        ):
            failures.append(
                _item("resources", "extra_manifest_resource", item["id"] or "manifest", "output")
            )
    for path in sorted(set(output_by_path) - set(source_by_path)):
        item = output_by_path[path]
        if not (item["id"] == "tn-bilingual-style" and item["media"] == "text/css"):
            failures.append(_item("resources", "extra_manifest_resource", path, "output"))
    if sm["spine_paths"] != om["spine_paths"]:
        failures.append(_item("spine", "sequence_mismatch", "<spine>", "source"))
    source_nav = sorted(item["path"] for item in sm["nav_items"])
    output_nav = sorted(item["path"] for item in om["nav_items"])
    source_ncx = sorted(item["path"] for item in sm["ncx_items"])
    output_ncx = sorted(item["path"] for item in om["ncx_items"])
    if source_nav != output_nav or source_ncx != output_ncx:
        failures.append(_item("nav", "declaration_mismatch", "<manifest>", "toc"))
    if source_nav and not output_nav:
        failures.append(_item("nav", "missing_source_nav", "<manifest>", "source"))
    if source_ncx and not output_ncx:
        failures.append(_item("nav", "missing_source_ncx", "<manifest>", "source"))
    source_members = {name for name in source_model["archive"] if not name.endswith("/")}
    output_members = {name for name in output_model["archive"] if not name.endswith("/")}
    for name in sorted(source_members - output_members):
        failures.append(_item("resources", "missing_resource", name, "source"))
    for name in sorted(output_members - source_members):
        failures.append(_item("resources", "unmanifested_resource", name, "output"))
    for asset in sorted(set(source_assets) | set(output_assets)):
        if asset not in output_assets:
            failures.append(_item("assets", "missing_asset", asset, "missing"))
        elif asset not in source_assets:
            failures.append(_item("assets", "extra_asset", asset, "extra"))
        elif source_assets[asset]["sha256"] != output_assets[asset]["sha256"]:
            failures.append(_item("assets", "changed_asset", asset, "changed"))


def _inline_hashes(soups: dict[str, BeautifulSoup]) -> dict[str, list[tuple[str, str, str, str]]]:
    result: dict[str, list[tuple[str, str, str, str]]] = {}
    for resource, soup in soups.items():
        entries: list[tuple[str, str, str, str]] = []
        for tag in soup.find_all(["style", "script"]):
            text = tag.get_text("", strip=False)
            attrs = repr(sorted((str(key), str(value)) for key, value in tag.attrs.items()))
            entries.append(
                (
                    tag.name,
                    str(tag.get("id") or ""),
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    hashlib.sha256(attrs.encode("utf-8")).hexdigest(),
                )
            )
        result[resource] = entries
    return result


def _validate_one(
    path: Path,
    *,
    source_path: Path | None,
    bilingual: bool | None,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checked = dict.fromkeys(_CATEGORIES, 0)
    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "path_sha256": _sha256(path),
        "structural_pass": False,
        "counts": {
            category: {"checked": 0, "failures": 0, "warnings": 0} for category in _CATEGORIES
        },
        "failures": [],
        "warnings": [],
        "generated_resources": [],
    }
    if not path.is_file():
        failures.append(_item("zip", "missing_file", "<input>", "EPUB path does not exist"))
        checked["zip"] += 1
        return _finish(result, failures, warnings, checked)
    try:
        zf = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile):
        failures.append(_item("zip", "invalid_zip", "<input>", "invalid_zip"))
        checked["zip"] += 1
        return _finish(result, failures, warnings, checked)
    with zf:
        try:
            infos = preflight_zip(
                zf,
                max_member_bytes=_MAX_MEMBER_BYTES,
                max_archive_bytes=_MAX_ARCHIVE_BYTES,
                max_archive_members=_MAX_ARCHIVE_MEMBERS,
            )
        except ZipSafetyError as exc:
            failures.append(_item("zip", exc.code, exc.name or "<archive>", "archive"))
            return _finish(result, failures, warnings, checked)
        names = [info.filename for info in infos]
        archive = set(names)
        checked["zip"] += len(infos) + 1
        if not infos or infos[0].filename != "mimetype":
            failures.append(
                _item("zip", "mimetype_not_first", "mimetype", "mimetype must be first")
            )
        counts = Counter(names)
        for name, count in sorted(counts.items()):
            if count > 1:
                failures.append(
                    _item(
                        "zip",
                        "mimetype_duplicate" if name == "mimetype" else "duplicate_entry",
                        name,
                        "duplicate entry",
                    )
                )
        for info in infos:
            if not _safe_archive_name(info.filename):
                failures.append(
                    _item("zip", "unsafe_entry", info.filename, "archive entry escapes root")
                )
            if info.flag_bits & 0x1:
                failures.append(_item("zip", "encrypted_entry", info.filename, "encrypted"))
        if sum(info.file_size for info in infos) > _MAX_ARCHIVE_BYTES:
            failures.append(
                _item(
                    "zip", "archive_too_large", "<archive>", "archive declared size exceeds limit"
                )
            )
            return _finish(result, failures, warnings, checked)
        if any(info.file_size > _MAX_MEMBER_BYTES for info in infos):
            failures.append(_item("zip", "member_too_large", "<archive>", "member exceeds limit"))
            return _finish(result, failures, warnings, checked)
        if counts.get("mimetype") != 1:
            failures.append(_item("zip", "mimetype_duplicate", "mimetype", "exactly one mimetype"))
        elif "mimetype" in archive:
            info = next(i for i in infos if i.filename == "mimetype")
            if info.compress_type != zipfile.ZIP_STORED:
                failures.append(
                    _item("zip", "mimetype_compressed", "mimetype", "mimetype must be stored")
                )
            try:
                data = _read_member(zf, info)
                if data != b"application/epub+zip":
                    failures.append(
                        _item("zip", "mimetype_invalid", "mimetype", "invalid mimetype bytes")
                    )
            except _MemberError as exc:
                failures.append(_item("zip", exc.code, "mimetype", "member"))
        for info in infos:
            try:
                _read_member(zf, info, as_bytes=False)
            except _MemberError as exc:
                failures.append(_item("zip", exc.code, info.filename, "member"))
        if any(not _safe_archive_name(name) for name in names) or any(
            info.flag_bits & 0x1 for info in infos
        ):
            return _finish(result, failures, warnings, checked)
        if "META-INF/container.xml" not in archive:
            failures.append(
                _item(
                    "resources",
                    "missing_container",
                    "META-INF/container.xml",
                    "container.xml is required",
                )
            )
            return _finish(result, failures, warnings, checked)
        model_info = _archive_model(zf, failures)
        opf_path = model_info["opf_path"]
        model = model_info["model"]
        container_data = _model_read(zf, "META-INF/container.xml", failures)
        container = _parse_xml(container_data) if container_data is not None else None
        checked["resources"] += 1
        if container is None:
            failures.append(
                _item("parse", "invalid_container", "META-INF/container.xml", "malformed XML")
            )
            checked["parse"] += 1
            return _finish(result, failures, warnings, checked)
        roots = [
            e.attrib.get("full-path", "").strip()
            for e in container.iter()
            if _local_name(e.tag) == "rootfile"
        ]
        if len(roots) != 1 or not _safe_archive_name(roots[0]) or roots[0] not in archive:
            failures.append(
                _item(
                    "resources", "invalid_rootfile", "META-INF/container.xml", "rootfile unresolved"
                )
            )
            return _finish(result, failures, warnings, checked)
        opf_data = _model_read(zf, opf_path, failures)
        opf = _parse_xml(opf_data) if opf_data is not None else None
        checked["parse"] += 1
        if opf is None:
            failures.append(_item("parse", "invalid_opf", opf_path, "malformed XML"))
            return _finish(result, failures, warnings, checked)
        model = _content_model(opf, opf_path, archive)
        items = model["items"]
        manifest_ids = [item["id"] for item in items]
        for item in items:
            if not item["id"]:
                failures.append(_item("resources", "manifest_id_missing", opf_path, "manifest"))
            if not item["href"]:
                failures.append(
                    _item("resources", "manifest_href_missing", opf_path, item["id"] or "manifest")
                )
            if not item["media"]:
                failures.append(
                    _item("resources", "manifest_media_missing", opf_path, item["id"] or "manifest")
                )
        for item_id, count in sorted(Counter(manifest_ids).items()):
            if item_id and count > 1:
                failures.append(_item("resources", "manifest_id_duplicate", opf_path, item_id))
        for item_index, item in enumerate(items):
            checked["resources"] += 1
            if not item["id"] or any(
                previous["id"] == item["id"] for previous in items[:item_index]
            ):
                continue
            target = _manifest_href(opf_path, item["href"])
            if target is None or target not in archive:
                failures.append(
                    _item("resources", "missing_manifest_resource", opf_path, item["id"])
                )
        resolved = model["resolved"]
        manifest_paths = {item["path"] for item in resolved}
        special = {"mimetype", "META-INF/container.xml", opf_path}
        for name in sorted(archive - manifest_paths - special):
            if name.endswith("/"):
                continue
            if not name.startswith("META-INF/"):
                failures.append(
                    _item(
                        "resources",
                        "unmanifested_resource",
                        name,
                        "output" if source_path else "archive",
                    )
                )
        for item in resolved:
            properties = item["properties"].split()
            if "nav" in properties and item["media"] != "application/xhtml+xml":
                failures.append(
                    _item("nav", "nav_manifest_media", opf_path, item["id"] or item["href"])
                )
            if (
                item["href"].split("#", 1)[0].lower().endswith(".ncx")
                and item["media"] != _NCX_MEDIA
            ):
                failures.append(
                    _item("nav", "ncx_manifest_media", opf_path, item["id"] or item["href"])
                )
        if model["spine_toc"] and not any(
            item["id"] == model["spine_toc"] and item["media"] == _NCX_MEDIA for item in resolved
        ):
            failures.append(_item("nav", "spine_toc_unresolved", opf_path, "toc"))
        spine_paths: list[str] = []
        for item_id in model["spine_ids"]:
            checked["spine"] += 1
            matching = [item for item in resolved if item["id"] == item_id]
            if len(matching) != 1:
                failures.append(_item("spine", "unresolved_idref", opf_path, item_id))
            elif matching[0]["media"] not in _HTML_MEDIA:
                failures.append(_item("spine", "non_content_item", opf_path, item_id))
            else:
                spine_paths.append(matching[0]["path"])
        if not spine_paths:
            failures.append(_item("spine", "empty_spine", opf_path, "spine has no content"))
        content_items = [
            item
            for item in resolved
            if item["media"] in _HTML_MEDIA and "nav" not in item["properties"].split()
        ]
        toc_items = model["nav_items"] + model["ncx_items"]
        soups: dict[str, BeautifulSoup] = {}
        ids_by_path: dict[str, set[str]] = {}
        for item in content_items + toc_items:
            content_path = item["path"]
            if content_path in soups:
                continue
            data = _model_read(zf, content_path, failures)
            if data is None:
                continue
            soup, valid = _html_soup(data, item["media"])
            checked["parse"] += 1
            if not valid:
                failures.append(
                    _item(
                        "parse",
                        "malformed_content" if item["media"] in _HTML_MEDIA else "invalid_toc",
                        content_path,
                        "malformed",
                    )
                )
            soups[content_path] = soup
            ids_by_path[content_path] = _ids(soup)
            _check_document_features(
                soup, content_path, failures, checked, content=item in content_items
            )
            styles = soup.find_all("style")
            for style in styles:
                style_id = str(style.get("id") or "")
                if style_id == BILINGUAL_STYLE_ID:
                    if bilingual is False or style.get_text() != _BILINGUAL_CSS:
                        failures.append(
                            _item(
                                "resources",
                                "generated_resource_mismatch",
                                content_path,
                                "tn-bilingual-style",
                            )
                        )
                    else:
                        result["generated_resources"].append(f"{content_path}#tn-bilingual-style")
        if not model["nav_items"] and not model["ncx_items"]:
            checked["nav"] += 1
            failures.append(_item("nav", "missing_toc", opf_path, "toc"))
        else:
            checked["nav"] += len(model["nav_items"]) + len(model["ncx_items"])
        nav_graph = Counter()
        for item in model["nav_items"] + model["ncx_items"]:
            soup = soups.get(item["path"])
            if soup is None:
                continue
            if item["media"] == _NCX_MEDIA:
                _check_ncx_semantics(soup, item["path"], failures, checked)
            elif item["media"] in _HTML_MEDIA:
                _check_nav_semantics(
                    soup,
                    item["path"],
                    failures,
                    checked,
                    allow_typeless="nav" in item["properties"].split(),
                )
            nav_graph.update(
                _check_links(
                    soup,
                    item["path"],
                    archive,
                    ids_by_path,
                    failures,
                    warnings,
                    checked,
                    category="nav",
                )
            )
        content_paths_set = {item["path"] for item in content_items}
        current_graph = _graph_from_soups(
            {resource: soups[resource] for resource in soups if resource in content_paths_set},
            archive,
            ids_by_path,
            failures,
            warnings,
            checked,
        )
        current_graph.update(nav_graph)
        _check_footnotes(soups, ids_by_path, failures, warnings, checked)
        current_assets = _resource_hashes(
            zf, model_info | {"model": model}, opf_path, failures, checked
        )
        if source_path is not None:
            if not source_path.is_file():
                failures.append(_item("assets", "source_missing", "<source>", "source_unreadable"))
            else:
                try:
                    with zipfile.ZipFile(source_path, "r") as source_zip:
                        source_info = _archive_model(source_zip, [])
                        source_model = source_info["model"]
                        source_asset_failures: list[dict[str, str]] = []
                        source_assets = _resource_hashes(
                            source_zip,
                            source_info,
                            source_info["opf_path"],
                            source_asset_failures,
                            checked,
                        )
                        failures.extend(source_asset_failures)
                        _compare_source_models(
                            source_info,
                            model_info | {"model": model},
                            source_assets,
                            current_assets,
                            failures,
                        )
                        source_soups: dict[str, BeautifulSoup] = {}
                        source_ids: dict[str, set[str]] = {}
                        source_failures: list[dict[str, str]] = []
                        for source_item in source_model["resolved"]:
                            if (
                                source_item["media"] not in _HTML_MEDIA
                                and source_item["media"] != _NCX_MEDIA
                                and "nav" not in source_item["properties"].split()
                            ):
                                continue
                            source_data = _model_read(
                                source_zip, source_item["path"], source_failures
                            )
                            if source_data is None:
                                continue
                            source_soup, _ = _html_soup(source_data, source_item["media"])
                            source_soups[source_item["path"]] = source_soup
                            source_ids[source_item["path"]] = _ids(source_soup)
                        temp_checked = dict.fromkeys(_CATEGORIES, 0)
                        source_graph = _graph_from_soups(
                            source_soups,
                            source_info["archive"],
                            source_ids,
                            [],
                            [],
                            temp_checked,
                        )
                        failures.extend(source_failures)
                        for key in set(source_graph) | set(current_graph):
                            if current_graph[key] != source_graph[key]:
                                failures.append(
                                    _item(
                                        "internal_links",
                                        "reference_graph_mismatch",
                                        key[0],
                                        "source",
                                    )
                                )
                        source_identifier_set = {
                            (resource, identifier)
                            for resource, soup in source_soups.items()
                            for identifier in _ids(soup)
                        }
                        current_identifier_set = {
                            (resource, identifier)
                            for resource, soup in soups.items()
                            for identifier in _ids(soup)
                        }
                        for resource, _identifier in sorted(
                            source_identifier_set - current_identifier_set
                        ):
                            failures.append(
                                _item("anchors", "anchor_graph_mismatch", resource, "source")
                            )
                        source_inline = _inline_hashes(source_soups)
                        output_inline = _inline_hashes(soups)
                        for resource in sorted(set(source_inline) | set(output_inline)):
                            expected_inline = source_inline.get(resource, [])
                            actual_inline = output_inline.get(resource, [])
                            if bilingual is not None:
                                actual_inline = [
                                    entry
                                    for entry in actual_inline
                                    if not (
                                        entry[0] == "style"
                                        and entry[1] == "tn-bilingual-style"
                                        and entry[2]
                                        == hashlib.sha256(
                                            _BILINGUAL_CSS.encode("utf-8")
                                        ).hexdigest()
                                        and entry[3]
                                        == hashlib.sha256(
                                            repr([("id", "tn-bilingual-style")]).encode("utf-8")
                                        ).hexdigest()
                                    )
                                ]
                            if actual_inline != expected_inline:
                                failures.append(
                                    _item(
                                        "resources",
                                        "inline_resource_mismatch",
                                        resource,
                                        "style_or_script",
                                    )
                                )
                except (OSError, zipfile.BadZipFile):
                    failures.append(
                        _item("assets", "source_unreadable", "<source>", "source_unreadable")
                    )
        if bilingual is not None:
            if bilingual:
                _validate_bilingual_nodes(soups, failures, checked)
                if source_path is not None and source_path.is_file():
                    _source_subset(path, source_path, soups, failures, checked)
            else:
                checked["bilingual_source"] += 1
                if any(soup.select(".tn-source") for soup in soups.values()):
                    failures.append(
                        _item(
                            "bilingual_source", "unexpected_source_nodes", "<output>", "unexpected"
                        )
                    )
            try:
                from trans_novel.ingest.segmenter import load_document

                reopened = load_document(str(path), "en", "zh")
                checked["bilingual_source"] += 1
                if not reopened.chapters or not any(ch.text_segments for ch in reopened.chapters):
                    failures.append(_item("bilingual_source", "reopen_empty", "<output>", "empty"))
            except Exception:
                failures.append(
                    _item("bilingual_source", "reopen_failed", "<output>", "unreadable")
                )
    return _finish(result, failures, warnings, checked)


def _finish(
    result: dict[str, Any],
    failures: list[dict[str, str]],
    warnings: list[dict[str, str]],
    checked: dict[str, int],
) -> dict[str, Any]:
    failures = _sort_items(failures)
    warnings = _sort_items(warnings)
    result["failures"] = failures
    result["warnings"] = warnings
    result["structural_pass"] = not failures
    result["counts"] = {
        category: {
            "checked": checked.get(category, 0),
            "failures": sum(1 for item in failures if item["category"] == category),
            "warnings": sum(1 for item in warnings if item["category"] == category),
        }
        for category in _CATEGORIES
    }
    result["generated_resources"] = sorted(set(result.get("generated_resources", [])))
    return result


def validate_epub(
    path: Path, *, source_path: Path | None = None, bilingual: bool | None = None
) -> dict[str, Any]:
    return _validate_one(
        Path(path),
        source_path=Path(source_path) if source_path is not None else None,
        bilingual=bilingual,
    )


def validate_epub_triplet(
    source_path: Path, mono_path: Path, bilingual_path: Path
) -> dict[str, Any]:
    source = validate_epub(Path(source_path))
    mono = validate_epub(Path(mono_path), source_path=Path(source_path), bilingual=False)
    bilingual = validate_epub(Path(bilingual_path), source_path=Path(source_path), bilingual=True)
    proof_failures: list[dict[str, str]] = []
    proof_checked = dict.fromkeys(_CATEGORIES, 0)
    if Path(source_path).is_file() and Path(mono_path).is_file() and Path(bilingual_path).is_file():
        try:
            with zipfile.ZipFile(bilingual_path, "r") as zf:
                info = _archive_model(zf, proof_failures)
                model = info["model"]
                soups: dict[str, BeautifulSoup] = {}
                for item in model["resolved"]:
                    if item["media"] not in _HTML_MEDIA or "nav" in item["properties"].split():
                        continue
                    data = _model_read(zf, item["path"], proof_failures)
                    if data is not None:
                        soups[item["path"]] = _html_soup(data, item["media"])[0]
            _exact_bilingual_proof(
                Path(source_path), Path(mono_path), soups, proof_failures, proof_checked
            )
        except (OSError, zipfile.BadZipFile):
            proof_failures.append(
                _item("bilingual_source", "proof_unreadable", "<output>", "unreadable")
            )
    bilingual["failures"].extend(proof_failures)
    bilingual = _finish(
        bilingual,
        bilingual["failures"],
        bilingual["warnings"],
        {
            category: bilingual["counts"][category]["checked"] + proof_checked.get(category, 0)
            for category in _CATEGORIES
        },
    )
    return {
        "schema_version": _SCHEMA_VERSION,
        "structural_pass": bool(
            source["structural_pass"] and mono["structural_pass"] and bilingual["structural_pass"]
        ),
        "source": source,
        "mono": mono,
        "bilingual": bilingual,
    }


class EpubVerificationError(RuntimeError):
    """A temporary EPUB failed independent post-write verification."""

    def __init__(self, report: dict[str, Any], *, cause: BaseException | None = None):
        self.report = report
        self.published = False
        super().__init__("EPUB verification failed")
        if cause is not None:
            self.__cause__ = cause


class EpubPublishError(RuntimeError):
    """Publication or post-publication durability failed."""

    def __init__(
        self,
        report: dict[str, Any],
        *,
        published: bool,
        cause: BaseException | None = None,
    ):
        self.report = report
        self.published = published
        super().__init__("EPUB publication failed")
        if cause is not None:
            self.__cause__ = cause


def _output_label(path: str | os.PathLike[str]) -> str:
    name = os.path.basename(os.fspath(path))
    name = re.sub(r"[\\/:*?\"<>|\r\n\t]+", " ", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name)[:120]
    return name or "translated.epub"


_REPORT_DETAILS = {
    "archive",
    "atomic",
    "changed",
    "count_mismatch",
    "duplicate entry",
    "extra",
    "fallback",
    "href",
    "id",
    "immutable",
    "invalid",
    "malformed",
    "manifest",
    "media",
    "media_overlay",
    "member",
    "missing",
    "opf",
    "output",
    "pair_mismatch",
    "properties",
    "recovered",
    "rejected",
    "schema3_required",
    "source",
    "source_mode",
    "state",
    "strict_required",
    "target",
    "toc",
    "unattached",
    "unreadable",
    "unsupported",
    "writer",
    "xhtml",
}


def _report_item(item: dict[str, str]) -> dict[str, str]:
    """Convert validator evidence to fixed, privacy-safe report vocabulary."""
    detail = str(item.get("detail", "invalid"))
    return {
        "category": str(item.get("category", "resources")),
        "code": str(item.get("code", "invalid")),
        "path": _archive_label(str(item.get("path", "<output>"))),
        "detail": detail if detail in _REPORT_DETAILS else "invalid",
    }


def _state_resources(store: Any) -> tuple[dict[str, Any], list[Any], list[dict[str, str]]]:
    """Reload manifest and chapters, never borrowing writer-owned objects."""
    failures: list[dict[str, str]] = []
    try:
        manifest = store.load_manifest()
        chapters = [store.load_chapter(int(entry["index"])) for entry in manifest["chapters"]]
    except Exception:
        return {}, [], [_item("state", "state_unreadable", "<state>", "unreadable")]
    meta = manifest.get("meta") if isinstance(manifest.get("meta"), dict) else {}
    schema = meta.get("epub_schema")
    if schema != 3:
        failures.append(_item("state", "unsupported_schema", "<state>", "schema3_required"))
    resources = {
        str(item.get("href")): item
        for item in meta.get("epub_resources", [])
        if isinstance(item, dict) and isinstance(item.get("href"), str)
    }
    return resources, chapters, failures


def _element_children_lxml(node: etree._Element) -> list[etree._Element]:
    return [child for child in node if isinstance(child.tag, str)]


def _resolve_path_lxml(root: etree._Element, path: tuple[int, ...]) -> etree._Element | None:
    current = root
    for index in path:
        children = _element_children_lxml(current)
        if index < 0 or index >= len(children):
            return None
        current = children[index]
    return current


def _element_path_lxml(root: etree._Element, target: etree._Element) -> tuple[int, ...] | None:
    """Return the element-only path used by the persisted schema3 locators."""
    if root is target:
        return ()
    parent = target.getparent()
    if parent is None:
        return None
    parent_path = _element_path_lxml(root, parent)
    if parent_path is None:
        return None
    children = _element_children_lxml(parent)
    try:
        return (*parent_path, children.index(target))
    except ValueError:
        return None


def _diagnostic_codes(values: object) -> list[tuple[str, str]]:
    """Extract only stable, privacy-safe parser domain/type codes."""
    if not isinstance(values, list):
        return []
    result: list[tuple[str, str]] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        domain = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value.get("domain", "")))[:64]
        kind = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value.get("type", "")))[:64]
        result.append((domain, kind))
    return sorted(result)


def _source_node_visible_text(node: etree._Element) -> str:
    """Match EPUB segment visibility while retaining ruby structure separately."""
    parts: list[str] = []

    def walk(current: etree._Element) -> None:
        name = current.tag.rsplit("}", 1)[-1].lower() if isinstance(current.tag, str) else ""
        if name in {"script", "style", "rt", "rp"}:
            return
        if current.text:
            parts.append(current.text)
        for child in current:
            if isinstance(child.tag, str):
                walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(node)
    return _norm_text("".join(parts))


def _ruby_signatures(node: etree._Element) -> list[tuple[Any, ...]]:
    """Canonicalize ruby markup using the writer's sanitized-attribute contract."""
    result: list[tuple[Any, ...]] = []
    for ruby in node.iter():
        if not isinstance(ruby.tag, str) or _local_name(ruby.tag).lower() != "ruby":
            continue

        def signature(current: etree._Element) -> tuple[Any, ...]:
            attrs = tuple(
                sorted(
                    (key, value)
                    for key, value in current.attrib.items()
                    if key.rsplit("}", 1)[-1]
                    not in {"id", "name", "data-tn-id", "data-tn-inline-id", "data-tn-line"}
                )
            )
            children = tuple(signature(child) for child in current if isinstance(child.tag, str))
            tails = tuple(child.tail for child in current if isinstance(child.tag, str))
            return (current.tag, attrs, current.text, children, tails)

        result.append(signature(ruby))
    return result


def _fragment_signature(markup: str) -> tuple[Any, ...]:
    soup = BeautifulSoup(f"<w>{markup}</w>", "xml")
    root = soup.find("w")
    if not isinstance(root, Tag):
        return ()

    def signature(node: Tag) -> tuple[Any, ...]:
        attrs = tuple(sorted((str(key), str(value)) for key, value in node.attrs.items()))
        children = tuple(
            signature(child) for child in node.find_all(recursive=False) if isinstance(child, Tag)
        )
        return (node.name, attrs, node.decode_contents(formatter="minimal"), children)

    return signature(root)


def _source_subtree_signature(node: etree._Element, *, root: bool = True) -> tuple[Any, ...]:
    attrs = tuple(
        sorted((key, value) for key, value in node.attrib.items() if not (root and key == "class"))
    )
    children = tuple(
        (
            _source_subtree_signature(child, root=False),
            child.tail,
        )
        for child in node
        if isinstance(child.tag, str)
    )
    return (_local_name(node.tag).lower(), attrs, node.text, children)


def _nav_label_locations(
    root: etree._Element, *, is_ncx: bool
) -> list[tuple[etree._Element, tuple[int, ...]]]:
    """Locate exactly the labels enumerated by the production TOC parser."""
    locations: list[tuple[etree._Element, tuple[int, ...]]] = []

    def direct(parent: etree._Element, name: str) -> etree._Element | None:
        return next(
            (
                child
                for child in _element_children_lxml(parent)
                if child.tag.rsplit("}", 1)[-1].lower() == name.lower()
            ),
            None,
        )

    if is_ncx:
        nav_map = next(
            (node for node in root.iter() if _local_name(node.tag).lower() == "navmap"), None
        )
        if nav_map is None:
            return locations

        def walk_ncx(parent: etree._Element) -> None:
            for point in _element_children_lxml(parent):
                if _local_name(point.tag).lower() != "navpoint":
                    continue
                label_parent = direct(point, "navLabel")
                label = (
                    next(
                        (
                            child
                            for child in label_parent.iter()
                            if isinstance(child.tag, str) and _local_name(child.tag) == "text"
                        ),
                        None,
                    )
                    if label_parent is not None
                    else None
                )
                if label is not None:
                    path = _element_path_lxml(root, label)
                    if path is not None:
                        locations.append((label, path))
                walk_ncx(point)

        walk_ncx(nav_map)
        return locations

    navs = [
        node
        for node in root.iter()
        if isinstance(node.tag, str)
        and _local_name(node.tag).lower() == "nav"
        and "toc"
        in (
            str(node.get("epub:type", node.get("type", "")))
            + " "
            + str(node.get("{http://www.idpf.org/2007/ops}type", ""))
        ).split()
    ]
    if not navs:
        all_navs = [
            node
            for node in root.iter()
            if isinstance(node.tag, str) and _local_name(node.tag).lower() == "nav"
        ]
        navs = all_navs[:1]

    def walk_nav(ordered_list: etree._Element) -> None:
        for li in _element_children_lxml(ordered_list):
            if _local_name(li.tag).lower() != "li":
                continue
            label = next(
                (
                    child
                    for child in _element_children_lxml(li)
                    if _local_name(child.tag).lower() in {"a", "span"}
                ),
                None,
            )
            if label is not None:
                path = _element_path_lxml(root, label)
                if path is not None:
                    locations.append((label, path))
            nested = direct(li, "ol")
            if nested is not None:
                walk_nav(nested)

    for nav in navs:
        ordered = direct(nav, "ol")
        if ordered is None:
            ordered = next(
                (node for node in nav.iter() if _local_name(node.tag).lower() == "ol"),
                None,
            )
        if ordered is not None:
            walk_nav(ordered)
    return locations


def _bilingual_proof(
    root_source: etree._Element,
    root_output: etree._Element,
    segments: list[Any],
    *,
    source_lang: str,
    order: str,
    resource: str,
    failures: list[dict[str, str]],
) -> int:
    """Validate and remove only the current schema3 bilingual insertions."""
    source_nodes = [
        node
        for node in root_output.iter()
        if isinstance(node.tag, str) and "tn-source" in str(node.get("class", "")).split()
    ]
    expected: list[tuple[str, tuple[int, ...]]] = []
    expected_total = 0
    container_paths: set[tuple[int, ...]] = set()
    direct_source_keys: set[tuple[tuple[int, ...], tuple[int, ...] | str]] = set()
    for segment in segments:
        if not segment_needs_source(segment):
            continue
        state = segment.epub_state
        assert state is not None
        block_path = tuple(state.block_path)
        source_block = _resolve_path_lxml(root_source, block_path)
        if source_block is not None and is_bilingual_container_tag(source_block.tag):
            if block_path not in container_paths:
                container_paths.add(block_path)
                expected_source = japanese_ruby_source_copy(source_block, source_lang, "div")
                if expected_source is None:
                    expected_source = sanitized_source_copy(source_block, "div")
                expected.append((_source_node_visible_text(expected_source), block_path))
                expected_total += 1
            continue
        direct_segment = source_block is not None and any(
            isinstance(child.tag, str) and _local_name(child.tag).lower() == "br"
            for child in source_block
        )
        if direct_segment:
            for slot_index, slot in enumerate(state.slots):
                owner = _resolve_path_lxml(source_block, tuple(slot.element_path))
                ruby = (
                    next(
                        (
                            node
                            for node in (owner, *owner.iterancestors())
                            if node is not source_block
                            and isinstance(node.tag, str)
                            and _local_name(node.tag) == "ruby"
                        ),
                        None,
                    )
                    if owner is not None
                    else None
                )
                if ruby is not None and slot.field == "tail" and owner is ruby:
                    ruby = None
                if ruby is not None and ruby_base_count(ruby) <= 1:
                    ruby = None
                ruby_path = _element_path_lxml(root_source, ruby) if ruby is not None else None
                key = (
                    block_path,
                    ruby_path or ("slot", getattr(slot, "id", f"{id(segment)}:{slot_index}")),
                )
                if key not in direct_source_keys:
                    direct_source_keys.add(key)
                    expected_total += 1
        else:
            expected.append((segment.source, block_path))
            expected_total += 1
    for node in source_nodes:
        attrs = dict(node.attrib)
        if not source_node_is_valid(node) or attrs != {"class": BILINGUAL_SOURCE_CLASS}:
            failures.append(
                _item("bilingual_source", "source_node_attributes", resource, "invalid")
            )
        if node.tag.rsplit("}", 1)[-1].lower() not in {"p", "div", "span"}:
            failures.append(_item("bilingual_source", "source_node_shape", resource, "invalid"))
        for child in node.iter():
            if not isinstance(child.tag, str):
                continue
            name = child.tag.rsplit("}", 1)[-1].lower()
            if name in {
                "audio",
                "canvas",
                "embed",
                "iframe",
                "img",
                "object",
                "script",
                "source",
                "style",
                "svg",
                "video",
            }:
                failures.append(
                    _item("bilingual_source", "source_node_active_media", resource, "media")
                )
            if any(
                key.rsplit("}", 1)[-1].lower() in {"id", "name", "href", "src"}
                or key.lower().startswith("on")
                for key in child.attrib
            ):
                failures.append(
                    _item("bilingual_source", "source_node_active_attribute", resource, "media")
                )

    node_parents = {id(node): node.getparent() for node in source_nodes}
    node_siblings = {
        id(parent): _element_children_lxml(parent)
        for parent in node_parents.values()
        if parent is not None
    }
    node_text_context = {
        id(node): ((node.getparent().text if node.getparent() is not None else None), node.tail)
        for node in source_nodes
    }
    node_mixed_context: dict[
        int, tuple[str | None, list[tuple[etree._Element, str | None, str | None]]]
    ] = {}
    for parent in node_parents.values():
        if parent is None or id(parent) in node_mixed_context:
            continue
        node_mixed_context[id(parent)] = (
            parent.text,
            [(child, child.text, child.tail) for child in _element_children_lxml(parent)],
        )
    style_nodes = [
        node
        for node in root_output.iter()
        if isinstance(node.tag, str)
        and _local_name(node.tag).lower() == "style"
        and node.get("id") == BILINGUAL_STYLE_ID
    ]
    if (expected_total and len(style_nodes) != 1) or (not expected_total and style_nodes):
        failures.append(
            _item("bilingual_source", "bilingual_style_count", resource, "count_mismatch")
        )

    direct_target_total = sum(
        1
        for node in root_output.iter()
        if isinstance(node.tag, str) and dict(node.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS
    )
    direct_target_used = 0

    def remove_preserving_tail(node: etree._Element) -> None:
        parent = node.getparent()
        if parent is None:
            return
        tail = node.tail
        previous = node.getprevious()
        if tail:
            if previous is not None:
                previous.tail = (previous.tail or "") + tail
            else:
                parent.text = (parent.text or "") + tail
        parent.remove(node)

    def unwrap_preserving_text(node: etree._Element) -> None:
        parent = node.getparent()
        if parent is None:
            return
        previous = node.getprevious()
        text = (node.text or "") + (node.tail or "")
        if previous is not None:
            previous.tail = (previous.tail or "") + text
        else:
            parent.text = (parent.text or "") + text
        parent.remove(node)

    # Direct-br runs keep target/source span pairs inside the original
    # paragraph.  Prove those pairs from the pre-removal sibling snapshot:
    # the block path alone resolves to the paragraph, not its generated
    # target spans, so the generic parent-order branch cannot validate them.

    direct_groups: dict[tuple[int, ...], list[Any]] = defaultdict(list)
    for segment in segments:
        if not segment_needs_source(segment):
            continue
        state = segment.epub_state
        assert state is not None
        source_block = _resolve_path_lxml(root_source, tuple(state.block_path))
        if (
            source_block is None
            or is_bilingual_container_tag(source_block.tag)
            or not any(
                isinstance(child.tag, str) and _local_name(child.tag).lower() == "br"
                for child in source_block
            )
        ):
            continue
        direct_groups[tuple(state.block_path)].append(segment)

    direct_source_paths: set[tuple[int, ...]] = set()
    direct_source_object_ids: set[int] = set()
    for block_path, direct_segments in direct_groups.items():
        target_block = _resolve_path_lxml(root_output, block_path)
        if target_block is None:
            failures.append(
                _item("bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch")
            )
            continue

        def structural_children(parent: etree._Element) -> list[etree._Element]:
            return [
                child
                for child in _element_children_lxml(parent)
                if (
                    "tn-source" not in str(child.get("class", "")).split()
                    and dict(child.attrib) != BILINGUAL_DIRECT_TARGET_ATTRS
                )
            ]

        def resolve_owner(
            path: tuple[int, ...], target_block: etree._Element = target_block
        ) -> etree._Element | None:
            current = target_block
            for index in path:
                children = structural_children(current)
                if index < 0 or index >= len(children):
                    return None
                current = children[index]
            return current

        def source_boundary(
            original_owner: etree._Element,
            source_block: etree._Element = source_block,
            block_path: tuple[int, ...] = block_path,
            target_block: etree._Element = target_block,
            resolve_owner: Callable[[tuple[int, ...]], etree._Element | None] = resolve_owner,
        ) -> etree._Element:
            original_boundary = direct_run_boundary(source_block, original_owner)
            boundary_path = _element_path_lxml(root_source, original_boundary)
            if boundary_path is None or boundary_path[: len(block_path)] != block_path:
                return target_block
            boundary = resolve_owner(tuple(boundary_path[len(block_path) :]))
            return boundary if boundary is not None else target_block

        block_direct_sources = [
            node
            for node in source_nodes
            if any(ancestor is target_block for ancestor in node.iterancestors())
        ]
        assigned: set[tuple[int, ...]] = set()
        last_source_index: dict[int, int] = {}
        processed_rubies: set[tuple[int, ...]] = set()
        for segment in direct_segments:
            state = segment.epub_state
            assert state is not None
            for slot in state.slots:
                owner = resolve_owner(tuple(slot.element_path))
                original_owner = _resolve_path_lxml(root_source, (*block_path, *slot.element_path))
                if owner is None or original_owner is None:
                    failures.append(
                        _item(
                            "bilingual_source",
                            "source_target_pair_mismatch",
                            resource,
                            "pair_mismatch",
                        )
                    )
                    continue
                ruby = next(
                    (
                        node
                        for node in (original_owner, *original_owner.iterancestors())
                        if node is not source_block
                        and isinstance(node.tag, str)
                        and _local_name(node.tag) == "ruby"
                    ),
                    None,
                )
                if ruby is not None and slot.field == "tail" and original_owner is ruby:
                    ruby = None
                if ruby is not None and ruby_base_count(ruby) <= 1:
                    ruby = None
                ruby_path = _element_path_lxml(root_source, ruby) if ruby is not None else None
                grouped_ruby = ruby_path is not None
                boundary = source_boundary(original_owner)
                leading = getattr(slot, "leading_whitespace", "")
                trailing = getattr(slot, "trailing_whitespace", "")
                target_core = getattr(slot, "target_core", None)
                source_core = getattr(slot, "source_core", slot.source_value)
                expected_target = (
                    leading + (target_core if target_core is not None else source_core) + trailing
                )

                if slot.field == "text":
                    target_candidates = [
                        child
                        for child in _element_children_lxml(owner)
                        if dict(child.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS
                    ]
                    target_node = next(
                        (
                            child
                            for child in target_candidates
                            if _source_node_visible_text(child) == _norm_text(expected_target)
                        ),
                        None,
                    )
                    source_parent = (
                        owner
                        if boundary is owner and not direct_run_is_active(owner)
                        else boundary.getparent()
                    )
                else:
                    source_parent = boundary if boundary is target_block else boundary.getparent()
                    target_parent = owner.getparent()
                    target_node = None
                    if target_parent is not None:
                        siblings = _element_children_lxml(target_parent)
                        try:
                            owner_index = siblings.index(owner)
                        except ValueError:
                            owner_index = -1
                        if owner_index >= 0:
                            target_node = next(
                                (
                                    child
                                    for child in siblings[owner_index + 1 :]
                                    if dict(child.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS
                                    and _source_node_visible_text(child)
                                    == _norm_text(expected_target)
                                ),
                                None,
                            )
                if target_node is None or dict(target_node.attrib) != BILINGUAL_DIRECT_TARGET_ATTRS:
                    failures.append(
                        _item(
                            "bilingual_source",
                            "source_target_pair_mismatch",
                            resource,
                            "pair_mismatch",
                        )
                    )
                else:
                    direct_target_used += 1
                if grouped_ruby and ruby_path in processed_rubies:
                    continue
                if grouped_ruby and ruby_path is not None:
                    processed_rubies.add(ruby_path)

                if source_parent is None:
                    failures.append(
                        _item(
                            "bilingual_source",
                            "source_target_pair_mismatch",
                            resource,
                            "pair_mismatch",
                        )
                    )
                    continue
                expected_source_node = (
                    direct_run_source_copy(
                        source_block,
                        original_owner,
                        source_lang=source_lang,
                        source_tag="span",
                        source_value=slot.source_value,
                        ruby_source=True,
                    )
                    if grouped_ruby
                    else None
                )
                source_match_value = (
                    _source_node_visible_text(expected_source_node)
                    if expected_source_node is not None
                    else _norm_text(slot.source_value)
                )
                candidates = [
                    node
                    for node in _element_children_lxml(source_parent)
                    if "tn-source" in str(node.get("class", "")).split()
                    and _element_path_lxml(root_output, node) not in assigned
                    and _source_node_visible_text(node) == source_match_value
                ]
                source_node = None
                if (
                    boundary is owner and slot.field == "text" and not direct_run_is_active(owner)
                ) or (
                    slot.field == "tail"
                    and boundary is owner
                    and not direct_run_is_active(owner)
                    and target_node is not None
                ):
                    siblings = _element_children_lxml(source_parent)
                    target_index = siblings.index(target_node)
                    source_node = next(
                        (
                            node
                            for node in candidates
                            if (siblings.index(node) < target_index) == (order == "source_first")
                        ),
                        None,
                    )
                elif (
                    slot.field == "tail"
                    and boundary is owner
                    and target_node is not None
                    and target_node.getparent() is source_parent
                ):
                    siblings = _element_children_lxml(source_parent)
                    target_index = siblings.index(target_node)
                    boundary_index = siblings.index(boundary) if boundary in siblings else -1
                    source_node = next(
                        (
                            node
                            for node in candidates
                            if boundary_index >= 0
                            and ((siblings.index(node) < target_index) == (order == "source_first"))
                            and (
                                (siblings.index(node) < boundary_index) == (order == "source_first")
                            )
                        ),
                        None,
                    )
                elif boundary is not target_block:
                    siblings = _element_children_lxml(source_parent)
                    boundary_index = siblings.index(boundary) if boundary in siblings else -1
                    source_node = next(
                        (
                            node
                            for node in candidates
                            if boundary_index >= 0
                            and (
                                (siblings.index(node) < boundary_index) == (order == "source_first")
                            )
                        ),
                        None,
                    )
                else:
                    source_node = candidates[0] if candidates else None
                if source_node is None:
                    failures.append(
                        _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
                    )
                    continue
                source_path = _element_path_lxml(root_output, source_node)
                if source_path is None:
                    failures.append(
                        _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
                    )
                    continue
                siblings = _element_children_lxml(source_parent)
                source_index = siblings.index(source_node)
                parent_key = id(source_parent)
                prior_index = last_source_index.get(parent_key)
                if prior_index is not None and source_index <= prior_index:
                    failures.append(
                        _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
                    )
                last_source_index[parent_key] = source_index
                direct_source_object_ids.add(id(source_node))
                assigned.add(source_path)
                direct_source_paths.add(source_path)
                if direct_run_has_active_ancestor(target_block, source_node):
                    failures.append(
                        _item("bilingual_source", "source_node_active_ancestor", resource, "active")
                    )
                expected_source = direct_run_source_copy(
                    source_block,
                    original_owner,
                    source_lang=source_lang,
                    source_tag=source_node.tag,
                    source_value=slot.source_value,
                    ruby_source=slot.field == "text",
                )
                if _source_subtree_signature(expected_source) != _source_subtree_signature(
                    source_node
                ):
                    failures.append(
                        _item(
                            "bilingual_source", "source_node_subtree_mismatch", resource, "invalid"
                        )
                    )
        if len(block_direct_sources) != len(assigned):
            failures.append(
                _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
            )

    if direct_target_used != direct_target_total:
        failures.append(_item("bilingual_source", "source_node_order", resource, "pair_mismatch"))
    for style in style_nodes:
        if not style_shape_is_valid(style):
            failures.append(
                _item("bilingual_source", "bilingual_style_mismatch", resource, "invalid")
            )
        remove_preserving_tail(style)
    for node in list(source_nodes):
        remove_preserving_tail(node)
    for node in list(root_output.iter()):
        if dict(node.attrib) == BILINGUAL_DIRECT_TARGET_ATTRS and not len(node):
            unwrap_preserving_text(node)

    if len(source_nodes) != expected_total:
        failures.append(_item("bilingual_source", "source_node_count", resource, "count_mismatch"))

    matched: set[int] = set()
    for node in source_nodes:
        if id(node) in direct_source_object_ids:
            continue
        text = _source_node_visible_text(node)
        candidates = [
            (index, source_text, block_path)
            for index, (source_text, block_path) in enumerate(expected)
            if index not in matched and _norm_text(source_text) == text
        ]
        chosen: tuple[int, str, tuple[int, ...]] | None = None
        for candidate in candidates:
            target = _resolve_path_lxml(root_output, candidate[2])
            parent = node_parents.get(id(node))
            if target is None or parent is None:
                continue
            if parent is target or parent is target.getparent():
                chosen = candidate
                break
        if chosen is None:
            failures.append(
                _item("bilingual_source", "source_target_pair_mismatch", resource, "pair_mismatch")
            )
            continue
        source_block = _resolve_path_lxml(root_source, chosen[2])
        if source_block is not None and not (
            _local_name(node.tag).lower() == "span"
            and any(
                _local_name(descendant.tag).lower() == "br" for descendant in source_block.iter()
            )
        ):
            expected_source = japanese_ruby_source_copy(
                source_block,
                source_lang,
                node.tag,
            )
            if expected_source is None:
                expected_source = sanitized_source_copy(source_block, node.tag)
            if _source_subtree_signature(expected_source) != _source_subtree_signature(node):
                failures.append(
                    _item(
                        "bilingual_source",
                        "source_node_subtree_mismatch",
                        resource,
                        "invalid",
                    )
                )
        target = _resolve_path_lxml(root_output, chosen[2])
        parent = node_parents.get(id(node))
        original_children = node_siblings.get(id(parent), []) if parent is not None else []
        if (
            target is None
            or parent is None
            or node not in original_children
            or (parent is not target and target not in original_children)
        ):
            continue
        node_index = original_children.index(node)
        _original_parent_text, _original_node_tail = node_text_context.get(id(node), (None, None))
        if parent is target and is_bilingual_container_tag(target.tag):
            mixed = node_mixed_context.get(id(parent))
            before_parts: list[str] = []
            after_parts: list[str] = []
            seen_source = False
            if mixed is not None:
                parent_text, entries = mixed
                before_parts.append(parent_text or "")
                for child, _child_text, child_tail in entries:
                    if child is node:
                        seen_source = True
                        after_parts.append(child_tail or "")
                        continue
                    visible = _source_node_visible_text(child)
                    if seen_source:
                        after_parts.extend((visible, child_tail or ""))
                    else:
                        before_parts.extend((visible, child_tail or ""))
            if (order == "target_first" and _norm_text("".join(after_parts))) or (
                order == "source_first" and _norm_text("".join(before_parts))
            ):
                failures.append(
                    _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
                )
        elif parent is target.getparent():
            target_index = original_children.index(target)
            between = original_children[
                min(node_index, target_index) + 1 : max(node_index, target_index)
            ]
            between = [
                child for child in between if child not in source_nodes and child not in style_nodes
            ]
            if any(_local_name(child.tag).lower() != "br" for child in between):
                failures.append(
                    _item("bilingual_source", "source_node_misplaced", resource, "unattached")
                )
            if (node_index < target_index) != (order == "source_first"):
                failures.append(
                    _item("bilingual_source", "source_node_order", resource, "pair_mismatch")
                )
    return len(source_nodes)


def _lang_attr(key: str) -> bool:
    return key in {"lang", "{http://www.w3.org/XML/1998/namespace}lang"}


def _compare_dom(
    source: etree._Element,
    output: etree._Element,
    slots: dict[tuple[tuple[int, ...], str], Any],
    path: tuple[int, ...] = (),
    *,
    toc_label_paths: set[tuple[int, ...]] | None = None,
    allow_root_language: bool = True,
) -> bool:
    if source.tag != output.tag:
        return False
    source_attrs = {
        key: value
        for key, value in source.attrib.items()
        if not (allow_root_language and _lang_attr(key) and path == ())
    }
    output_attrs = {
        key: value
        for key, value in output.attrib.items()
        if not (allow_root_language and _lang_attr(key) and path == ())
    }
    if source_attrs != output_attrs:
        return False
    if (path, "text") not in slots and source.text != output.text:
        return False
    source_children = list(source)
    output_children = list(output)
    if len(source_children) != len(output_children):
        return False
    allow_toc_tail = bool(
        toc_label_paths
        and any(path[: len(label_path)] == label_path for label_path in toc_label_paths)
    )
    element_index = 0
    for source_child, output_child in zip(source_children, output_children, strict=True):
        if not isinstance(source_child.tag, str) or not isinstance(output_child.tag, str):
            if source_child.tag != output_child.tag or source_child.text != output_child.text:
                return False
            if (not allow_toc_tail and source_child.tail != output_child.tail) or (
                allow_toc_tail and output_child.tail is not None
            ):
                return False
            continue
        child_path = (*path, element_index)
        element_index += 1
        if not _compare_dom(
            source_child,
            output_child,
            slots,
            child_path,
            toc_label_paths=toc_label_paths,
            allow_root_language=allow_root_language,
        ):
            return False
        if (child_path, "tail") not in slots and source_child.tail != output_child.tail:
            return False
    return True


def _slot_proof(
    source_path: Path,
    output_path: Path,
    store: Any,
    resources: dict[str, Any],
    chapters: list[Any],
    *,
    bilingual: bool,
    target_lang: str | None = None,
    bilingual_order: str = "target_first",
    failures: list[dict[str, str]],
    warnings: list[dict[str, str]],
    checked: dict[str, int],
) -> dict[str, int]:
    from trans_novel.ingest.epub_reader import _resource_parser

    all_segments = [
        segment
        for chapter in chapters
        for segment in chapter.segments
        if segment.epub_state is not None and segment.epub_state.resource_href
    ]
    try:
        deduped_segments = dedupe_segment_mappings(all_segments)
    except ValueError:
        failures.append(_item("state", "slot_mapping_ambiguous", "<state>", "schema3"))
        deduped_segments = all_segments
    by_resource: dict[str, list[Any]] = defaultdict(list)
    for segment in deduped_segments:
        state = segment.epub_state
        assert state is not None
        by_resource[state.resource_href].append(segment)
    differences = {"text_slots": 0, "toc_labels": 0, "language_fields": 0, "bilingual_nodes": 0}
    try:
        source_zip = zipfile.ZipFile(source_path, "r")
        output_zip = zipfile.ZipFile(output_path, "r")
    except (OSError, zipfile.BadZipFile):
        failures.append(_item("state", "source_reopen_failed", "<source>", "unreadable"))
        return differences
    try:
        archive_model = _archive_model(source_zip, [])
        xml_resources = {
            item["path"]
            for item in archive_model.get("model", {}).get("resolved", [])
            if item.get("media") in _HTML_MEDIA
            or item.get("media") == _NCX_MEDIA
            or "nav" in item.get("properties", "").split()
        }
        for resource in sorted(path for path in xml_resources if path):
            by_resource.setdefault(resource, [])
    except Exception:
        pass
    try:
        manifest = store.load_manifest()
    except Exception:
        manifest = {}
    raw_meta = manifest.get("meta") if isinstance(manifest, dict) else {}
    raw_toc = raw_meta.get("toc_entries") if isinstance(raw_meta, dict) else []
    toc_entries = (
        [entry for entry in raw_toc if isinstance(entry, dict)] if isinstance(raw_toc, list) else []
    )

    with source_zip, output_zip:
        source_names = set(source_zip.namelist())
        output_names = set(output_zip.namelist())
        for resource, segments in sorted(by_resource.items()):
            if resource not in source_names or resource not in output_names:
                failures.append(_item("state", "resource_missing", resource, "schema3_locator"))
                continue
            try:
                source_data = _read_member(source_zip, source_zip.getinfo(resource))
                output_data = _read_member(output_zip, output_zip.getinfo(resource))
            except (KeyError, _MemberError):
                failures.append(_item("zip", "member_read", resource, "member"))
                continue
            expected = resources.get(resource, {})
            source_digest = hashlib.sha256(source_data).hexdigest()
            if expected.get("resource_sha256") and source_digest != str(
                expected["resource_sha256"]
            ):
                failures.append(_item("assets", "source_digest_mismatch", resource, "state"))
                continue
            try:
                source_tree, source_mode, source_diag = _resource_parser(source_data)
                output_tree, output_mode, output_diag = _resource_parser(output_data)
            except Exception:
                failures.append(_item("parse", "resource_unreadable", resource, "xhtml"))
                continue
            expected_mode = str(expected.get("parse_mode", ""))
            checked["parse"] = checked.get("parse", 0) + 2
            if expected_mode and source_mode != expected_mode:
                failures.append(_item("parse", "parse_mode_mismatch", resource, "state"))
            if source_mode == "recovered":
                warnings.append(_item("parse", "recovered_resource", resource, "recovered"))
                persisted_codes = _diagnostic_codes(expected.get("parser_diagnostics"))
                actual_codes = _diagnostic_codes(source_diag)
                if not isinstance(expected.get("parser_diagnostics"), list):
                    failures.append(
                        _item("parse", "recovered_diagnostic_missing", resource, "state")
                    )
                elif persisted_codes != actual_codes:
                    failures.append(
                        _item("parse", "recovered_diagnostic_mismatch", resource, "state")
                    )
                else:
                    for domain, kind in actual_codes:
                        code = f"recovered_diagnostic_{domain or 'unknown'}_{kind or 'unknown'}"
                        warnings.append(_item("parse", code, resource, "recovered"))
                if output_mode == "recovered" and _diagnostic_codes(output_diag) != persisted_codes:
                    failures.append(
                        _item("parse", "recovered_output_diagnostic_mismatch", resource, "state")
                    )
            if output_mode == "recovered" and expected_mode == "xml":
                failures.append(_item("parse", "unexpected_recovery", resource, "strict_required"))

            root_source, root_output = source_tree.getroot(), output_tree.getroot()
            slot_map: dict[tuple[tuple[int, ...], str], Any] = {}
            toc_label_paths: set[tuple[int, ...]] = set()
            direct_cleared: set[tuple[tuple[int, ...], str]] = set()
            is_ncx_resource = any(
                isinstance(node.tag, str) and _local_name(node.tag).lower() == "navmap"
                for node in root_source.iter()
            )
            if target_lang and not is_ncx_resource:
                source_lang_attrs = {
                    key: value for key, value in root_source.attrib.items() if _lang_attr(key)
                }
                output_lang_attrs = {
                    key: value for key, value in root_output.attrib.items() if _lang_attr(key)
                }
                if set(source_lang_attrs) != set(output_lang_attrs) or any(
                    value != target_lang for value in output_lang_attrs.values()
                ):
                    failures.append(_item("dom", "language_mismatch", resource, "target"))
                else:
                    for key, value in source_lang_attrs.items():
                        if output_lang_attrs[key] != value:
                            differences["language_fields"] += 1

            for segment in segments:
                state = segment.epub_state
                assert state is not None
                block_source = _resolve_path_lxml(root_source, tuple(state.block_path))
                if block_source is None:
                    failures.append(_item("dom", "block_locator_missing", resource, "state"))
                    continue
                if (
                    bilingual
                    and segment_needs_source(segment)
                    and any(
                        isinstance(child.tag, str) and _local_name(child.tag).lower() == "br"
                        for child in block_source
                    )
                ):
                    direct_cleared.update(
                        (tuple(state.block_path) + tuple(slot.element_path), slot.field)
                        for slot in state.slots
                    )
                fingerprint = hashlib.sha256(
                    etree.tostring(block_source, encoding="utf-8", with_tail=False)
                ).hexdigest()
                if fingerprint != state.block_fingerprint:
                    failures.append(_item("dom", "block_fingerprint_mismatch", resource, "state"))
                from trans_novel.ingest.models import _slot_contract_digest

                if state.slot_contract_sha256 != _slot_contract_digest(state.slots):
                    failures.append(_item("state", "slot_contract_mismatch", resource, "state"))
                seen_slot_locations: set[tuple[tuple[int, ...], str]] = set()
                for slot in state.slots:
                    location = (tuple(state.block_path) + tuple(slot.element_path), slot.field)
                    if location in seen_slot_locations:
                        failures.append(_item("state", "slot_overlap", resource, "state"))
                    seen_slot_locations.add(location)
                    slot_map[location] = slot
                    owner = _resolve_path_lxml(root_source, location[0])
                    if owner is None:
                        failures.append(_item("dom", "slot_locator_missing", resource, "state"))
                    elif getattr(owner, slot.field) != slot.source_value:
                        failures.append(_item("dom", "source_slot_mismatch", resource, "state"))
            if bilingual and any(
                isinstance(node.tag, str) and _local_name(node.tag).lower() in {"html", "body"}
                for node in root_source.iter()
            ):
                try:
                    source_lang = str(manifest.get("source_lang", ""))
                except AttributeError:
                    source_lang = ""
                differences["bilingual_nodes"] += _bilingual_proof(
                    root_source,
                    root_output,
                    segments,
                    source_lang=source_lang,
                    order=bilingual_order,
                    resource=resource,
                    failures=failures,
                )

            is_navigation = any(
                isinstance(node.tag, str) and _local_name(node.tag).lower() in {"nav", "navmap"}
                for node in root_source.iter()
            )
            if is_navigation:
                is_ncx = any(
                    _local_name(node.tag).lower() == "navmap" for node in root_source.iter()
                )
                locations = _nav_label_locations(root_source, is_ncx=is_ncx)
                entries = sorted(
                    (
                        entry
                        for entry in toc_entries
                        if entry.get("toc_path") == resource
                        and isinstance(entry.get("node_index"), int)
                    ),
                    key=lambda entry: int(entry["node_index"]),
                )
                if entries and len(locations) != len(entries):
                    failures.append(_item("nav", "label_count_mismatch", resource, "toc"))
                for entry in entries:
                    index = int(entry["node_index"])
                    if index < 0 or index >= len(locations):
                        failures.append(_item("nav", "label_locator_missing", resource, "toc"))
                        continue
                    label, path = locations[index]
                    toc_label_paths.add(path)
                    from trans_novel.assemble.writer import _translated_toc_title

                    expected_title = _translated_toc_title(entry)
                    slot_map[(path, "text")] = {
                        "kind": "toc",
                        "expected": expected_title,
                        "source": label.text,
                        "count": True,
                    }

                    def allow_cleared_descendants(
                        parent: etree._Element,
                        parent_path: tuple[int, ...],
                        *,
                        allowed_slots: dict[tuple[tuple[int, ...], str], Any] = slot_map,
                    ) -> None:
                        element_index = 0
                        for child in parent:
                            if not isinstance(child.tag, str):
                                continue
                            child_path = (*parent_path, element_index)
                            element_index += 1
                            allowed_slots[(child_path, "text")] = {
                                "kind": "toc",
                                "expected": None,
                                "source": child.text,
                                "count": False,
                            }
                            allowed_slots[(child_path, "tail")] = {
                                "kind": "toc",
                                "expected": None,
                                "source": child.tail,
                                "count": False,
                            }
                            allow_cleared_descendants(child, child_path)

                    allow_cleared_descendants(label, path)

            if not _compare_dom(
                root_source,
                root_output,
                slot_map,
                toc_label_paths=toc_label_paths,
                allow_root_language=not is_ncx_resource,
            ):
                failures.append(_item("dom", "unauthorized_dom_change", resource, "immutable"))

            for (location, field), allowed in slot_map.items():
                owner = _resolve_path_lxml(root_output, location)
                if owner is None:
                    failures.append(_item("dom", "slot_locator_missing", resource, "output"))
                    continue
                if isinstance(allowed, dict) and allowed.get("kind") == "toc":
                    expected_value = allowed.get("expected")
                    actual_value = getattr(owner, field)
                    if actual_value != expected_value:
                        failures.append(_item("nav", "label_value_mismatch", resource, "target"))
                    if allowed.get("count") and actual_value != allowed.get("source"):
                        differences["toc_labels"] += 1
                    continue
                slot = allowed
                expected_value = (
                    slot.leading_whitespace
                    + (slot.target_core if slot.target_core is not None else slot.source_core)
                    + slot.trailing_whitespace
                )
                if owner.tag.rsplit("}", 1)[-1].lower() in _HEADING_TAGS:
                    from trans_novel.postprocess.punct import normalize_heading_numbering

                    expected_value = normalize_heading_numbering(expected_value)
                actual_value = getattr(owner, field)
                cleared = actual_value is None and (location, field) in direct_cleared
                if actual_value != expected_value and not cleared:
                    failures.append(_item("dom", "slot_value_mismatch", resource, "target"))
                if cleared or actual_value != slot.source_value:
                    differences["text_slots"] += 1
    return differences


def verify_epub(
    output_path: str | os.PathLike[str],
    *,
    source_path: str | os.PathLike[str] | None = None,
    store: Any | None = None,
    mode: str = "generated",
    bilingual: bool = False,
    target_lang: str | None = None,
    bilingual_order: str = "target_first",
) -> dict[str, Any]:
    """Reopen an on-disk EPUB and return deterministic report v1 evidence."""
    output = Path(output_path)
    source = Path(source_path) if source_path is not None else None
    source_sha = _sha256(source) if source is not None else None
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    checked = dict.fromkeys(_CATEGORIES, 0)
    differences = {"text_slots": 0, "toc_labels": 0, "language_fields": 0, "bilingual_nodes": 0}
    resources: dict[str, Any] = {}
    chapters: list[Any] = []
    if mode in {"monolingual", "bilingual"}:
        if store is None or source is None:
            failures.append(_item("state", "state_required", "<state>", "source_mode"))
        else:
            resources, chapters, state_failures = _state_resources(store)
            failures.extend(state_failures)
            try:
                persisted_sha = store.load_manifest().get("meta", {}).get("epub_sha256")
                if persisted_sha != source_sha:
                    failures.append(_item("state", "source_digest_mismatch", "<source>", "schema3"))
            except Exception:
                failures.append(_item("state", "state_unreadable", "<state>", "digest"))
    if target_lang is None and store is not None:
        try:
            from trans_novel.assemble.writer import _epub_lang

            target_lang = _epub_lang(store.load_manifest().get("target_lang"))
        except Exception:
            target_lang = None
    structural = _validate_one(
        output,
        source_path=source if mode in {"monolingual", "bilingual"} else None,
        bilingual=bilingual,
    )
    if mode in {"monolingual", "bilingual"} and source is not None:
        # A source EPUB may legitimately omit both navigation formats.  Keep
        # that source contract, while still rejecting removal from a source
        # which actually has a TOC.
        source_probe = _validate_one(source, source_path=None, bilingual=None)
        source_missing_toc = any(
            item.get("code") == "missing_toc" for item in source_probe.get("failures", [])
        )
        source_unmanifested = {
            item.get("path")
            for item in source_probe.get("failures", [])
            if item.get("code") == "unmanifested_resource"
        }
    else:
        source_missing_toc = False
        source_unmanifested = set()
    if mode in {"monolingual", "bilingual"} and source is not None:
        try:
            with (
                zipfile.ZipFile(source, "r") as source_zip,
                zipfile.ZipFile(output, "r") as output_zip,
            ):
                source_infos = source_zip.infolist()
                output_infos = output_zip.infolist()
                source_names = [info.filename for info in source_infos]
                output_names = [info.filename for info in output_infos]
                if source_names != output_names:
                    failures.append(_item("zip", "member_set_mismatch", "<archive>", "source"))
                elif len(source_names) != len(set(source_names)):
                    failures.append(_item("zip", "duplicate_source_member", "<archive>", "source"))
                else:
                    for source_info, output_info in zip(source_infos, output_infos, strict=True):
                        metadata = (
                            "compress_type",
                            "date_time",
                            "external_attr",
                            "internal_attr",
                            "extra",
                            "comment",
                        )
                        flags_match = source_info.flag_bits == output_info.flag_bits
                        if (
                            any(
                                getattr(source_info, key) != getattr(output_info, key)
                                for key in metadata
                            )
                            or not flags_match
                        ):
                            failures.append(
                                _item(
                                    "zip",
                                    "member_metadata_mismatch",
                                    source_info.filename,
                                    "source",
                                )
                            )
                            break
        except (OSError, zipfile.BadZipFile):
            failures.append(_item("zip", "source_reopen_failed", "<source>", "archive"))
        try:
            with (
                zipfile.ZipFile(source, "r") as source_zip,
                zipfile.ZipFile(output, "r") as output_zip,
            ):
                source_archive_info = _archive_model(source_zip, [])
                output_info = {info.filename: info for info in output_zip.infolist()}
                toc_paths = {
                    item["path"]
                    for item in source_archive_info.get("model", {}).get("nav_items", [])
                    + source_archive_info.get("model", {}).get("ncx_items", [])
                }
                if store is not None:
                    try:
                        persisted_toc = store.load_manifest().get("meta", {}).get("toc_entries", [])
                    except Exception:
                        persisted_toc = []
                    toc_paths.update(
                        entry.get("toc_path")
                        for entry in persisted_toc
                        if isinstance(entry, dict) and isinstance(entry.get("toc_path"), str)
                    )
                authorized_names = set(toc_paths)
                source_opf = source_archive_info.get("opf_path")
                if isinstance(source_opf, str):
                    authorized_names.add(source_opf)
                authorized_names.update(
                    item["path"]
                    for item in source_archive_info.get("model", {}).get("resolved", [])
                    if item.get("media") in _HTML_MEDIA
                    or item.get("media") == _NCX_MEDIA
                    or "nav" in item.get("properties", "").split()
                )
                if source_zip.comment != output_zip.comment:
                    failures.append(_item("zip", "archive_comment_mismatch", "<archive>", "source"))
                for info in source_zip.infolist():
                    name = info.filename
                    if name not in output_info or name == "mimetype" or name in authorized_names:
                        continue
                    if _read_member(source_zip, info) != _read_member(
                        output_zip, output_info[name]
                    ):
                        failures.append(_item("assets", "changed_asset", name, "changed"))
        except (OSError, zipfile.BadZipFile):
            failures.append(_item("assets", "source_unreadable", "<source>", "archive"))
        try:
            with (
                zipfile.ZipFile(source, "r") as source_zip,
                zipfile.ZipFile(output, "r") as output_zip,
            ):
                source_info = _archive_model(source_zip, [])
                output_info = _archive_model(output_zip, [])
                source_opf = source_info.get("opf_path")
                output_opf = output_info.get("opf_path")
                if source_opf and output_opf and source_opf == output_opf:
                    source_root = etree.fromstring(
                        _read_member(source_zip, source_zip.getinfo(source_opf))
                    )
                    output_root = etree.fromstring(
                        _read_member(output_zip, output_zip.getinfo(output_opf))
                    )
                    language_seen = False
                    opf_language_changed = 0

                    def compare_package(left: etree._Element, right: etree._Element) -> bool:
                        nonlocal language_seen, opf_language_changed
                        if left.tag != right.tag or dict(left.attrib) != dict(right.attrib):
                            return False
                        is_language = left.tag == "{http://purl.org/dc/elements/1.1/}language"
                        first_language = is_language and not language_seen
                        if is_language:
                            language_seen = True
                        if first_language:
                            if target_lang and right.text != target_lang:
                                return False
                            if left.text != right.text:
                                opf_language_changed += 1
                        elif left.text != right.text:
                            return False
                        if left.tail != right.tail:
                            return False
                        if len(left) != len(right):
                            return False
                        return all(compare_package(a, b) for a, b in zip(left, right, strict=True))

                    if not compare_package(source_root, output_root):
                        failures.append(
                            _item("resources", "package_metadata_mismatch", source_opf, "opf")
                        )
                    differences["language_fields"] += opf_language_changed
        except (OSError, zipfile.BadZipFile, etree.XMLSyntaxError):
            failures.append(_item("resources", "package_unreadable", "<package>", "opf"))
    failures.extend(
        item
        for item in structural.get("failures", [])
        if not (
            item.get("code") in {"reopen_failed", "reopen_empty"}
            or (item.get("code") == "missing_toc" and source_missing_toc)
            or (
                item.get("code") == "unmanifested_resource"
                and item.get("path") in source_unmanifested
            )
        )
    )
    warnings.extend(structural.get("warnings", []))
    checked.update(
        {
            category: int(structural.get("counts", {}).get(category, {}).get("checked", 0))
            for category in _CATEGORIES
        }
    )
    if mode in {"monolingual", "bilingual"} and source is not None and store is not None:
        slot_differences = _slot_proof(
            source,
            output,
            store,
            resources,
            chapters,
            bilingual=bilingual,
            target_lang=target_lang,
            bilingual_order=bilingual_order,
            failures=failures,
            warnings=warnings,
            checked=checked,
        )
        for key, value in slot_differences.items():
            differences[key] += value
    failures = _sort_items([_report_item(item) for item in failures])
    warnings = _sort_items([_report_item(item) for item in warnings])
    assurance = "verified"
    if any(str(item.get("parse_mode")) == "recovered" for item in resources.values()):
        assurance = "recovered"
    return {
        "schema_version": 1,
        "mode": mode,
        "assurance": assurance,
        "passed": not failures,
        "published": False,
        "source_sha256": source_sha,
        "output_sha256": _sha256(output),
        "output_label": _output_label(output),
        "failures": failures,
        "warnings": warnings,
        "checked": {category: int(checked.get(category, 0)) for category in _CATEGORIES},
        "authorized_differences": differences,
    }


def _fsync_file(path: str) -> None:
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def _is_unsupported_dir_fsync(error: OSError) -> bool:
    return error.errno in {
        value for value in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", -1)) if value
    }


def _persist_failure(
    store: Any, report: dict[str, Any], cause: BaseException | None = None
) -> None:
    try:
        store.save_epub_verification(report)
        store.log_event_required(
            "epub_verification_failed",
            output=report["output_label"],
            assurance=report["assurance"],
            failure_count=len(report["failures"]),
            warning_count=len(report["warnings"]),
            published=bool(report["published"]),
        )
    except Exception as error:
        raise EpubPublishError(report, published=bool(report["published"]), cause=error) from error


def _raise_preflight(
    store: Any,
    final: str,
    source_path: str | os.PathLike[str] | None,
    mode: str,
    code: str,
    detail: str,
    cause: BaseException | None = None,
) -> None:
    report = {
        "schema_version": 1,
        "mode": mode,
        "assurance": "verified",
        "passed": False,
        "published": False,
        "source_sha256": _sha256(Path(source_path)) if source_path else None,
        "output_sha256": "",
        "output_label": _output_label(final),
        "failures": [_item("publish", code, "<output>", detail)],
        "warnings": [],
        "checked": {},
        "authorized_differences": {
            "text_slots": 0,
            "toc_labels": 0,
            "language_fields": 0,
            "bilingual_nodes": 0,
        },
    }
    _persist_failure(store, report, cause)
    raise EpubPublishError(report, published=False, cause=cause)


def publish_epub(
    store: Any,
    source_path: str | os.PathLike[str] | None,
    final_path: str | os.PathLike[str],
    *,
    mode: str,
    bilingual: bool = False,
    bilingual_order: str = "target_first",
    writer: Callable[[str], object],
    source_identity_path: str | os.PathLike[str] | None = None,
) -> str:
    """Run the only EPUB publication path: owned temp, reopen, verify, replace."""
    final = os.fspath(final_path)
    identity = source_identity_path if source_identity_path is not None else source_path
    if identity is not None:
        source = os.fspath(identity)
        try:
            aliases = os.path.realpath(source) == os.path.realpath(final)
            if os.path.exists(source) and os.path.exists(final):
                aliases = aliases or os.path.samefile(source, final)
        except OSError:
            aliases = False
        if aliases:
            _raise_preflight(store, final, source_path, mode, "input_output_alias", "rejected")
    parent = os.path.dirname(os.path.abspath(final)) or "."
    if not os.path.isdir(parent) or not os.access(parent, os.W_OK):
        _raise_preflight(store, final, source_path, mode, "parent_unwritable", "parent")
    target_lang = None
    if store is not None:
        try:
            from trans_novel.assemble.writer import _epub_lang

            target_lang = _epub_lang(store.load_manifest().get("target_lang"))
        except Exception:
            target_lang = None
    if os.path.lexists(final):
        try:
            if os.path.islink(final) or os.path.isdir(final):
                _raise_preflight(store, final, source_path, mode, "final_not_regular", "rejected")
        except EpubPublishError:
            raise
        except OSError as error:
            _raise_preflight(
                store,
                final,
                source_path,
                mode,
                "final_unreadable",
                "rejected",
                error,
            )
    fd, temp = tempfile.mkstemp(
        prefix=f".{_output_label(final)}.epub-verify-",
        suffix=".tmp",
        dir=parent,
    )
    os.close(fd)
    report: dict[str, Any]
    try:
        try:
            writer(temp)
            report = verify_epub(
                temp,
                source_path=source_path,
                store=store,
                mode=mode,
                bilingual=bilingual,
                target_lang=target_lang,
                bilingual_order=bilingual_order,
            )
            report["output_label"] = _output_label(final)
        except Exception as cause:
            report = verify_epub(
                temp,
                source_path=source_path,
                store=store if mode in {"monolingual", "bilingual"} else None,
                mode=mode,
                bilingual=bilingual,
                target_lang=target_lang,
                bilingual_order=bilingual_order,
            )
            report["output_label"] = _output_label(final)
            report["passed"] = False
            report["failures"] = _sort_items(
                report["failures"] + [_item("publish", "writer_failed", "<output>", "writer")]
            )
            _persist_failure(store, report, cause)
            raise EpubVerificationError(report, cause=cause) from cause
        if not report["passed"]:
            report["published"] = False
            _persist_failure(store, report)
            raise EpubVerificationError(report)
        report["published"] = False
        try:
            store.save_epub_verification(report)
        except Exception as cause:
            raise EpubPublishError(report, published=False, cause=cause) from cause
        replaced = False
        try:
            os.replace(temp, final)
            replaced = True
            _fsync_file(final)
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError as error:
                if _is_unsupported_dir_fsync(error):
                    report["warnings"] = _sort_items(
                        report["warnings"]
                        + [
                            _item(
                                "publish", "directory_fsync_unsupported", "<output>", "unsupported"
                            )
                        ]
                    )
                else:
                    raise
        except OSError as cause:
            if replaced:
                report["published"] = True
                report["passed"] = False
                report["failures"] = _sort_items(
                    report["failures"]
                    + [_item("publish", "durability_failed", "<output>", "fsync")]
                )
                _persist_failure(store, report, cause)
                raise EpubPublishError(report, published=True, cause=cause) from cause
            report["passed"] = False
            report["published"] = False
            report["failures"] = _sort_items(
                report["failures"] + [_item("publish", "replace_failed", "<output>", "atomic")]
            )
            _persist_failure(store, report, cause)
            raise EpubPublishError(report, published=False, cause=cause) from cause
        report["published"] = True
        try:
            store.save_epub_verification(report)
            store.log_event_required(
                "epub_verification_passed",
                output=report["output_label"],
                assurance=report["assurance"],
                failure_count=0,
                warning_count=len(report["warnings"]),
                published=True,
            )
        except Exception as cause:
            raise EpubPublishError(report, published=True, cause=cause) from cause
        return final
    finally:
        if os.path.exists(temp):
            with suppress(OSError):
                os.unlink(temp)


__all__ = [
    "EpubPublishError",
    "EpubVerificationError",
    "publish_epub",
    "validate_epub",
    "validate_epub_triplet",
    "verify_epub",
]
