"""EPUB container, OPF manifest, and resource path helpers."""

from __future__ import annotations

import posixpath
import xml.etree.ElementTree as ET
import zipfile

from trans_novel.epub.archive import ZipSafetyError, read_member
from trans_novel.epub.navigation import resolve_epub_href

_CONTAINER = "META-INF/container.xml"
_HTML_EXTS = (".xhtml", ".html", ".htm")


def find_opf_path(zf: zipfile.ZipFile) -> str:
    try:
        data = read_member(zf, zf.getinfo(_CONTAINER))
    except (KeyError, ZipSafetyError) as exc:
        raise ValueError("EPUB 损坏：container.xml 不可读取") from exc
    root = ET.fromstring(data)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "rootfile":
            path = element.attrib.get("full-path", "").strip()
            if path:
                return path
    raise ValueError("EPUB 损坏：container.xml 未找到有效的 rootfile full-path")


def zip_href(base_path: str, href: str) -> str:
    """Resolve an EPUB-relative href to a normalized zip member path."""
    return resolve_epub_href(base_path, href).resource_href


def manifest_xhtml_paths(zf: zipfile.ZipFile, opf_path: str) -> list[str]:
    root = ET.fromstring(read_member(zf, zf.getinfo(opf_path)))
    paths: list[str] = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "item":
            continue
        href = element.attrib.get("href", "")
        media = element.attrib.get("media-type", "")
        if "html" not in media and not href.lower().endswith(_HTML_EXTS):
            continue
        resolved = zip_href(opf_path, href)
        if resolved and resolved not in paths:
            paths.append(resolved)
    return paths


def parse_opf(zf: zipfile.ZipFile, opf_path: str) -> tuple[str, list[str], list[str]]:
    """Return the title, spine XHTML paths, and NAV/NCX paths."""
    root = ET.fromstring(read_member(zf, zf.getinfo(opf_path)))

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    title = ""
    manifest: dict[str, tuple[str, str, str]] = {}
    spine_ids: list[str] = []
    toc_ids: list[str] = []
    for element in root.iter():
        name = local(element.tag)
        if name == "title" and not title and element.text:
            title = element.text.strip()
        elif name == "item":
            item_id = element.attrib.get("id", "").strip()
            if item_id:
                manifest[item_id] = (
                    element.attrib.get("href", ""),
                    element.attrib.get("media-type", ""),
                    element.attrib.get("properties", ""),
                )
        elif name == "itemref":
            idref = element.attrib.get("idref", "").strip()
            if idref:
                spine_ids.append(idref)
        elif name == "spine":
            toc = element.attrib.get("toc")
            if toc:
                toc_ids.append(toc)

    hrefs: list[str] = []
    for item_id in spine_ids:
        if item_id not in manifest:
            continue
        href, media, _properties = manifest[item_id]
        if "html" not in media and not href.lower().endswith(_HTML_EXTS):
            continue
        resolved = zip_href(opf_path, href)
        if resolved and resolved not in hrefs:
            hrefs.append(resolved)

    nav_ids = [
        item_id
        for item_id, (_href, _media, properties) in manifest.items()
        if "nav" in properties.split()
    ]
    ncx_ids = [
        item_id
        for item_id, (_href, media, _properties) in manifest.items()
        if media == "application/x-dtbncx+xml"
    ]
    toc_paths: list[str] = []
    for item_id in nav_ids + toc_ids + ncx_ids:
        if item_id not in manifest:
            continue
        resolved = zip_href(opf_path, manifest[item_id][0])
        if resolved and resolved not in toc_paths:
            toc_paths.append(resolved)
    return title, hrefs, toc_paths


def looks_like_internal_title(title: str, href: str, book_title: str = "") -> bool:
    base = posixpath.basename(href).rsplit(".", 1)[0]
    stripped = title.strip()
    return (bool(base) and stripped == base) or (
        bool(book_title) and stripped == book_title.strip()
    )
