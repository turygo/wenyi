"""Deterministic, privacy-safe structural checks for benchmark EPUB artifacts.

The validator only reads local ZIP members.  It never follows URLs and never emits
input filesystem paths or source text in its JSON evidence.
"""

from __future__ import annotations

import hashlib
import lzma
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup, Tag
from lxml import etree

from trans_novel.assemble.writer import (
    _BILINGUAL_CSS,
    _HORIZONTAL_OVERRIDE_CSS,
    _HORIZONTAL_OVERRIDE_ID,
    _epub_looks_vertical,
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
_MAX_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_EXACT_MARKERS = {"spine-fallback", "journal.json"}
_EXTERNAL_SCHEMES = {"http", "https", "mailto", "data"}
_INTERNAL_ATTRIBUTES = {"data-tn-id", "data-tn-inline-id", "data-tn-line"}
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
    """Read one member with declared, streamed, and CRC/decompression guards."""
    if info.file_size > _MAX_MEMBER_BYTES:
        raise _MemberError("member_too_large")
    chunks: list[bytes] = []
    total = 0
    try:
        with zf.open(info, "r") as stream:
            while True:
                chunk = stream.read(min(1024 * 1024, _MAX_MEMBER_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_MEMBER_BYTES:
                    raise _MemberError("member_too_large")
                if as_bytes:
                    chunks.append(chunk)
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
        name = type(exc).__name__
        raise _MemberError(
            "crc_error" if name in {"BadZipFile", "BadCRC"} else "member_read"
        ) from None
    return b"".join(chunks) if as_bytes else b""


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
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        return False
    normalized = posixpath.normpath(name)
    return normalized not in {"", "."} and normalized != ".." and not normalized.startswith("../")


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
    found_markers: set[str] = set()
    for tag in soup.find_all(True):
        checked["placeholders"] += 1
        if any(attribute in tag.attrs for attribute in _INTERNAL_ATTRIBUTES):
            failures.append(_item("placeholders", "internal_attribute", path, tag.name))
        for value in tag.attrs.values():
            if isinstance(value, str):
                found_markers.update(marker for marker in _EXACT_MARKERS if marker in value)
    for text_node in soup.find_all(string=True):
        found_markers.update(marker for marker in _EXACT_MARKERS if marker in str(text_node))
    for marker in sorted(found_markers):
        failures.append(_item("placeholders", "marker", path, marker))
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
    path: Path, *, source_path: Path | None, bilingual: bool | None
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
        infos = zf.infolist()
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
        source_vertical = False
        if source_path is not None and source_path.is_file():
            try:
                with zipfile.ZipFile(source_path, "r") as source_zip:
                    source_vertical = _epub_looks_vertical(source_zip)
            except (OSError, zipfile.BadZipFile):
                source_vertical = False
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
                if style_id == "tn-bilingual-style":
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
                elif style_id == _HORIZONTAL_OVERRIDE_ID:
                    if style.get_text() != _HORIZONTAL_OVERRIDE_CSS or (
                        source_path is not None and not source_vertical
                    ):
                        failures.append(
                            _item(
                                "resources",
                                "generated_resource_mismatch",
                                content_path,
                                _HORIZONTAL_OVERRIDE_ID,
                            )
                        )
                    else:
                        result["generated_resources"].append(
                            f"{content_path}#{_HORIZONTAL_OVERRIDE_ID}"
                        )
            if (
                bilingual
                and item in content_items
                and not soup.find("style", id="tn-bilingual-style")
            ):
                failures.append(
                    _item(
                        "resources",
                        "generated_resource_missing",
                        content_path,
                        "tn-bilingual-style",
                    )
                )
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
                                        (
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
                                        or (
                                            source_vertical
                                            and entry[0] == "style"
                                            and entry[1] == _HORIZONTAL_OVERRIDE_ID
                                            and entry[2]
                                            == hashlib.sha256(
                                                _HORIZONTAL_OVERRIDE_CSS.encode("utf-8")
                                            ).hexdigest()
                                            and entry[3]
                                            == hashlib.sha256(
                                                repr([("id", _HORIZONTAL_OVERRIDE_ID)]).encode(
                                                    "utf-8"
                                                )
                                            ).hexdigest()
                                        )
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


__all__ = ["validate_epub", "validate_epub_triplet"]
