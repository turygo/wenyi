from __future__ import annotations

import posixpath
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from tests.fixtures.migration_artifacts import sha256_bytes
from tests.fixtures.migration_errors import ObservationError

_BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "caption"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xml(data: bytes, label: str) -> ElementTree.Element:
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ObservationError(f"unexpected DTD or entity in synthetic XML: {label}")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ObservationError(f"invalid XML resource: {label}") from error


def _resolve_member(base: str, reference: str) -> tuple[str, str]:
    parsed = urlsplit(reference)
    path = unquote(parsed.path)
    if parsed.scheme or parsed.netloc or path.startswith("/") or "\\" in path:
        raise ObservationError(f"non-local EPUB reference: {reference}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base), path)) if path else base
    if resolved == ".." or resolved.startswith("../"):
        raise ObservationError(f"EPUB reference escapes archive: {reference}")
    return resolved, unquote(parsed.fragment)


def _is_source(node: ElementTree.Element) -> bool:
    return "tn-source" in node.get("class", "").split()


def _text(node: ElementTree.Element, *, exclude_sources: bool = False) -> str:
    parts = [node.text or ""]
    for child in node:
        if not exclude_sources or not _is_source(child):
            parts.append(_text(child, exclude_sources=exclude_sources))
        parts.append(child.tail or "")
    return "".join(parts)


def _shape(node: ElementTree.Element) -> dict[str, Any]:
    return {
        "tag": node.tag,
        "attributes": dict(sorted(node.attrib.items())),
        "text": node.text,
        "tail": node.tail,
        "children": [_shape(child) for child in node],
    }


def _nav_data(data: bytes, kind: str) -> list[dict[str, str]]:
    root = _parse_xml(data, kind)
    records = []
    for node in root.iter():
        if kind == "nav" and _local(node.tag) == "a" and node.get("href"):
            records.append({"label": _text(node).strip(), "href": node.attrib["href"]})
        elif kind == "ncx" and _local(node.tag) == "navPoint":
            label = next((child for child in node if _local(child.tag) == "navLabel"), None)
            content = next((child for child in node if _local(child.tag) == "content"), None)
            if label is None or content is None or not content.get("src"):
                raise ObservationError("NCX navigation point is incomplete")
            records.append({"label": _text(label).strip(), "href": content.attrib["src"]})
    return records


def _observe_html_resource(
    data: bytes, resource: str
) -> tuple[dict[str, Any], list[dict], set[str]]:
    root = _parse_xml(data, resource)
    elements, target_elements, blocks, copies, links = [], [], [], [], []
    identifiers: set[str] = set()

    def visit(node: ElementTree.Element, path: tuple[int, ...], source: bool = False) -> None:
        source = source or _is_source(node)
        tag = _local(node.tag)
        record = {
            "path": list(path),
            "tag": tag,
            "attributes": dict(sorted(node.attrib.items())),
            "text": _text(node, exclude_sources=not source),
            "raw_text": node.text,
            "tail": node.tail,
        }
        elements.append(record)
        if not source:
            target_elements.append(record)
        identifier = node.get("id")
        if identifier:
            if identifier in identifiers:
                raise ObservationError(f"duplicate XHTML identity: {resource}#{identifier}")
            identifiers.add(identifier)
        for attribute in ("href", "src"):
            if node.get(attribute):
                links.append(
                    {"from": resource, "attribute": attribute, "to": node.attrib[attribute]}
                )
        if _is_source(node):
            copies.append({"text": _text(node), "path": list(path), "tree": _shape(node)})
        if (
            not source
            and tag in _BLOCKS
            and not any(
                _local(child.tag) in _BLOCKS and not _is_source(child)
                for child in node.iter()
                if child is not node
            )
        ):
            text = _text(node, exclude_sources=True).strip()
            if text:
                blocks.append(text)
        for index, child in enumerate(node):
            visit(child, (*path, index), source)

    visit(root, ())
    return (
        {
            "elements": elements,
            "target_elements": target_elements,
            "blocks": blocks,
            "source_copies": copies,
            "tree": _shape(root),
        },
        links,
        identifiers,
    )


def _check_png(data: bytes) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ObservationError("fixture image is not a PNG")
    offset, payload = 8, bytearray()
    dimensions = None
    ended = False
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + chunk) != crc:
            raise ObservationError("fixture PNG chunk CRC is invalid")
        if kind == b"IHDR":
            dimensions = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            payload.extend(chunk)
        elif kind == b"IEND":
            ended = True
        offset += length + 12
    if dimensions != (1, 1, 8, 6, 0, 0, 0) or not ended or len(zlib.decompress(payload)) != 5:
        raise ObservationError("fixture PNG is not a decoded 1x1 RGBA image")


def _package_manifest(archive: zipfile.ZipFile, opf_path: str) -> dict[str, Any]:
    opf = _parse_xml(archive.read(opf_path), opf_path)
    items: dict[str, dict[str, Any]] = {}
    metadata = []
    for node in opf.iter():
        tag = _local(node.tag)
        if tag in {"title", "language", "identifier"}:
            metadata.append(
                {
                    "tag": node.tag,
                    "attributes": dict(sorted(node.attrib.items())),
                    "text": _text(node),
                }
            )
        if tag != "item":
            continue
        identifier, href = node.get("id"), node.get("href")
        if not identifier or not href or identifier in items:
            raise ObservationError("invalid or duplicate package manifest identity")
        resolved, fragment = _resolve_member(opf_path, href)
        if fragment or resolved not in archive.namelist():
            raise ObservationError(f"missing package resource: {resolved}")
        data = archive.read(resolved)
        media = node.get("media-type", "")
        if media == "image/png":
            _check_png(data)
        items[identifier] = {
            "id": identifier,
            "path": resolved,
            "media_type": media,
            "properties": node.get("properties", ""),
            "size_bytes": len(data),
            "sha256": sha256_bytes(data),
            "attributes": dict(sorted(node.attrib.items())),
        }
    spine = []
    for node in opf.iter():
        if _local(node.tag) == "itemref":
            if node.get("idref") not in items:
                raise ObservationError("spine references an unknown package identity")
            spine.append(items[node.attrib["idref"]]["path"])
    if not spine:
        raise ObservationError("package spine is empty")
    return {"items": items, "spine": spine, "metadata": metadata, "opf_tree": _shape(opf)}


def _read_package(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (
            not names
            or names[0] != "mimetype"
            or archive.read("mimetype") != b"application/epub+zip"
        ):
            raise ObservationError("invalid EPUB mimetype")
        if archive.getinfo("mimetype").compress_type != zipfile.ZIP_STORED or len(names) != len(
            set(names)
        ):
            raise ObservationError("invalid mimetype compression or duplicate ZIP member")
        for name in names:
            if (
                name.startswith("/")
                or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
            ):
                raise ObservationError("unsafe synthetic archive member")
        container = _parse_xml(archive.read("META-INF/container.xml"), "container")
        roots = [
            node.get("full-path") for node in container.iter() if _local(node.tag) == "rootfile"
        ]
        if len(roots) != 1 or roots[0] not in names:
            raise ObservationError("container does not declare one existing package document")
        opf_path = roots[0]
        manifest = _package_manifest(archive, opf_path)
        html, graph, anchors = {}, [], {}
        nav, ncx = [], []
        for item in manifest["items"].values():
            resource, media = item["path"], item["media_type"]
            if media == "application/xhtml+xml":
                observed, links, identities = _observe_html_resource(
                    archive.read(resource), resource
                )
                html[resource], anchors[resource] = observed, identities
                graph.extend(links)
            if "nav" in item["properties"].split():
                nav = _nav_data(archive.read(resource), "nav")
            if media == "application/x-dtbncx+xml":
                ncx = _nav_data(archive.read(resource), "ncx")
                graph.extend(
                    {"from": resource, "attribute": "src", "to": record["href"]} for record in ncx
                )
        for link in graph:
            target, fragment = _resolve_member(link["from"], link["to"])
            if target not in names or (fragment and fragment not in anchors.get(target, set())):
                raise ObservationError(f"unresolved EPUB reference: {link}")
        return {
            "members": names,
            "member_sha256": {name: sha256_bytes(archive.read(name)) for name in names},
            "resources": sorted(manifest["items"].values(), key=lambda item: item["id"]),
            "opf_path": opf_path,
            "metadata": manifest["metadata"],
            "spine": manifest["spine"],
            "nav": nav,
            "ncx": ncx,
            "html": html,
            "link_graph": sorted(
                graph, key=lambda item: (item["from"], item["attribute"], item["to"])
            ),
            "target_blocks": {name: html[name]["blocks"] for name in manifest["spine"]},
            "source_copies": {name: html[name]["source_copies"] for name in manifest["spine"]},
        }


def observe_package(path: Path) -> dict[str, Any]:
    try:
        return _read_package(path)
    except ObservationError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        zlib.error,
        struct.error,
    ) as error:
        raise ObservationError(f"invalid or incomplete EPUB archive: {path}") from error
