"""Read EPUB sources into schema-4 structural text-slot state."""

from __future__ import annotations

import hashlib
import os
import zipfile

from trans_novel.epub.archive import ZipSafetyError, preflight_zip, read_member
from trans_novel.epub.navigation import parse_toc_entries
from trans_novel.ingest.epub.chapters import logical_chapters
from trans_novel.ingest.epub.markup import annotate_resource
from trans_novel.ingest.epub.package import (
    find_opf_path,
    manifest_xhtml_paths,
    parse_opf,
)
from trans_novel.ingest.models import Document

_HTML_EXTS = (".xhtml", ".html", ".htm")


def read_epub(path: str, source_lang: str, target_lang: str) -> Document:
    """Read a source EPUB into schema-4 structural text-slot state."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            preflight_zip(zf)
            names = {info.filename for info in zf.infolist()}
            opf_path = find_opf_path(zf)
            book_title, hrefs, toc_paths = parse_opf(zf, opf_path)
            manifest_hrefs = manifest_xhtml_paths(zf, opf_path)
            toc_entries = parse_toc_entries(zf, toc_paths)

            resources: list[dict[str, object]] = []
            archive_hash = hashlib.sha256()
            with open(path, "rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    archive_hash.update(chunk)
            resource_hrefs = list(dict.fromkeys([*manifest_hrefs, *toc_paths]))
            for resource_index, href in enumerate(resource_hrefs):
                if href not in names or not href.lower().endswith(_HTML_EXTS):
                    continue
                data = read_member(zf, zf.getinfo(href))
                title, segments, resource = annotate_resource(
                    data,
                    resource_index,
                    href,
                    book_title=book_title,
                    skip_navigation=href in toc_paths,
                )
                resources.append({**resource, "title": title, "segments": segments})
            spine_resources = [resource for resource in resources if resource["href"] in hrefs]
            chapters, split_strategy, split_toc_path = logical_chapters(
                spine_resources, toc_entries
            )
    except ZipSafetyError as exc:
        raise ValueError(f"EPUB archive rejected: {exc.code}") from exc

    return Document(
        title=book_title or os.path.splitext(os.path.basename(path))[0],
        source_lang=source_lang,
        target_lang=target_lang,
        fmt="epub",
        source_path=os.path.abspath(path),
        chapters=chapters,
        meta={
            "epub_schema": 4,
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
