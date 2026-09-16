"""原文保留范围的语言校验证据。"""

from __future__ import annotations

import os
import zipfile
from collections import defaultdict

from lxml import etree

from trans_novel.assemble.epub.metadata import epub_language, translated_toc_title
from trans_novel.assemble.epub.rendering.source_dom import effective_language
from trans_novel.assemble.epub.verification import archive_model, dom
from trans_novel.assemble.epub.verification import bilingual as bilingual_module
from trans_novel.ingest import canonical_title_id, segment_preserves_source


def check_root_language(
    root_source, root_output, resource, target_lang, is_ncx, segments, failures, differences
):
    if not target_lang:
        return
    source_attrs = {
        key: value for key, value in root_source.attrib.items() if bilingual_module.lang_attr(key)
    }
    output_attrs = {
        key: value for key, value in root_output.attrib.items() if bilingual_module.lang_attr(key)
    }
    if segments and all(segment_preserves_source(segment) for segment in segments):
        if source_attrs != output_attrs:
            failures.append(archive_model.item("dom", "language_mismatch", resource, "source"))
        return
    if set(source_attrs) != set(output_attrs) or any(
        value != target_lang for value in output_attrs.values()
    ):
        failures.append(archive_model.item("dom", "language_mismatch", resource, "target"))
        return
    for key, value in source_attrs.items():
        if output_attrs[key] != value:
            differences["language_fields"] += 1


def preserved_language_paths(root_source, root_output, resource, segments, source_lang, failures):
    paths: set[tuple[int, ...]] = set()
    if segments and all(segment_preserves_source(segment) for segment in segments):
        return paths
    by_block: dict[tuple[int, ...], set[bool]] = defaultdict(set)
    for segment in segments:
        state = segment.epub_state
        assert state is not None
        by_block[state.block_path].add(segment_preserves_source(segment))
    for path, decisions in by_block.items():
        if len(decisions) > 1:
            failures.append(
                archive_model.item("state", "preserve_range_ambiguous", resource, "block")
            )
            continue
        if decisions != {True}:
            continue
        source_block = dom.resolve_path_lxml(root_source, path)
        output_block = dom.resolve_path_lxml(root_output, path)
        if source_block is None or output_block is None:
            failures.append(archive_model.item("dom", "block_locator_missing", resource, "output"))
            continue
        source_attrs = {
            key: value
            for key, value in source_block.attrib.items()
            if bilingual_module.lang_attr(key)
        }
        output_attrs = {
            key: value
            for key, value in output_block.attrib.items()
            if bilingual_module.lang_attr(key)
        }
        expected = effective_language(source_block, source_lang or None)
        actual = effective_language(output_block)
        extras = {key: value for key, value in output_attrs.items() if key not in source_attrs}
        existing_changed = any(
            output_attrs.get(key) != value for key, value in source_attrs.items()
        )
        extra_invalid = bool(
            extras
            and (
                set(extras) != {"{http://www.w3.org/XML/1998/namespace}lang"}
                or expected is None
                or next(iter(extras.values())) != expected
            )
        )
        if expected != actual or existing_changed or extra_invalid:
            failures.append(
                archive_model.item("dom", "language_mismatch", resource, "preserved_range")
            )
        if source_attrs != output_attrs:
            paths.add(path)
    return paths


def _generated_navigation_proof(archive, names, chapters, chapter_titles, target_lang, failures):
    nav_roots = [
        etree.fromstring(
            archive.read(name),
            etree.XMLParser(no_network=True, resolve_entities=False),
        )
        for name in names
        if name.lower().endswith(("/nav.xhtml", "/nav.html"))
    ]
    for chapter in chapters:
        filename = f"ch{chapter.index}.xhtml"
        labels = [
            node
            for root in nav_roots
            for node in root.iter()
            if isinstance(node.tag, str)
            and node.tag.rsplit("}", 1)[-1].lower() == "a"
            and os.path.basename((node.get("href") or "").split("#", 1)[0]) == filename
        ]
        label = labels[0] if len(labels) == 1 else None
        root = label.getroottree().getroot() if label is not None else None
        lang = next(
            (
                value
                for node in (label, root)
                if node is not None
                for key, value in node.attrib.items()
                if bilingual_module.lang_attr(key)
            ),
            "",
        )
        expected_title = chapter_titles.get(chapter.index) or chapter.title
        if (
            label is None
            or "".join(label.itertext()).strip() != expected_title.strip()
            or (target_lang and lang != target_lang)
        ):
            failures.append(
                archive_model.item("nav", "preserved_generated_label_mismatch", filename, "target")
            )


def _generated_content_proof(
    archive,
    names,
    chapters,
    source_lang,
    target_lang,
    failures,
    checked,
):
    from trans_novel.assemble.epub.rendering.bilingual import BILINGUAL_SOURCE_CLASSES
    from trans_novel.assemble.text import merged_paragraphs
    from trans_novel.ingest import KIND_HEADING

    for chapter in chapters:
        suffix = f"/ch{chapter.index}.xhtml"
        matches = [name for name in names if name.endswith(suffix)]
        if len(matches) != 1:
            failures.append(
                archive_model.item("state", "preserved_chapter_missing", suffix, "source")
            )
            continue
        root = etree.fromstring(
            archive.read(matches[0]),
            etree.XMLParser(no_network=True, resolve_entities=False),
        )
        body = next(
            (
                node
                for node in root.iter()
                if isinstance(node.tag, str)
                and archive_model.local_name(node.tag).lower() == "body"
            ),
            None,
        )
        source_copies = [
            node
            for node in root.iter()
            if isinstance(node.tag, str)
            and BILINGUAL_SOURCE_CLASSES.intersection(node.get("class", "").split())
        ]
        if source_copies:
            failures.append(
                archive_model.item(
                    "bilingual_source",
                    "preserved_generated_duplicate",
                    matches[0],
                    "source",
                )
            )
        content_nodes = [
            node
            for node in (body if body is not None else ())
            if isinstance(node.tag, str)
            and archive_model.local_name(node.tag).lower() in {"h1", "p"}
            and not BILINGUAL_SOURCE_CLASSES.intersection(node.get("class", "").split())
        ]
        actual = [
            (archive_model.local_name(node.tag).lower(), "".join(node.itertext()))
            for node in content_nodes
        ]
        paragraphs = merged_paragraphs(chapter)
        expected = [
            ("h1" if kind == KIND_HEADING else "p", target)
            for kind, target, _source, _preserve in paragraphs
        ]
        expected_languages = [
            source_lang if preserve and source_lang else target_lang
            for _kind, _target, _source, preserve in paragraphs
        ]
        checked["bilingual_source"] += max(len(expected), 1)
        root_languages = {
            key: value for key, value in root.attrib.items() if bilingual_module.lang_attr(key)
        }
        if actual != expected:
            failures.append(
                archive_model.item("dom", "preserved_generated_text_mismatch", matches[0], "source")
            )
        if target_lang and (
            not root_languages or any(value != target_lang for value in root_languages.values())
        ):
            failures.append(archive_model.item("dom", "language_mismatch", matches[0], "target"))
        if [
            effective_language(node, target_lang or None) for node in content_nodes
        ] != expected_languages:
            failures.append(
                archive_model.item("dom", "language_mismatch", matches[0], "preserved_range")
            )


def generated_chapter_proof(output_path, store, failures, checked):
    try:
        manifest = store.load_manifest()
    except Exception:
        return
    declared = any(
        (
            isinstance(processing := meta.get("processing"), dict)
            and processing.get("action") == "preserve"
        )
        or getattr(processing, "action", None) == "preserve"
        for meta in manifest.get("chapters", [])
    )
    if not declared:
        return
    try:
        chapter_titles: dict[int, str] = {}
        chapters = []
        for meta in manifest["chapters"]:
            index = meta["index"]
            if not isinstance(index, int) or index in chapter_titles:
                raise ValueError("chapter indexes must be unique integers")
            chapter_titles[index] = translated_toc_title(meta)
            chapter = store.load_chapter(index)
            if chapter.preserve_source:
                chapters.append(chapter)
        canonical_targets = {
            f"chapter:{index}": title for index, title in chapter_titles.items() if title
        }
        raw_meta = manifest.get("meta")
        raw_toc = raw_meta.get("toc_entries", []) if isinstance(raw_meta, dict) else []
        if isinstance(raw_toc, list):
            canonical_targets.update(
                {
                    entry["entry_id"]: entry["title_translated"]
                    for entry in raw_toc
                    if isinstance(entry, dict)
                    and isinstance(entry.get("entry_id"), str)
                    and isinstance(entry.get("title_translated"), str)
                }
            )
        for chapter in chapters:
            for segment in chapter.segments:
                title_id = canonical_title_id(segment)
                if title_id is not None and canonical_targets.get(title_id) != segment.target:
                    failures.append(
                        archive_model.item(
                            "state",
                            "canonical_title_mismatch",
                            f"chapter:{chapter.index}",
                            "target",
                        )
                    )
    except Exception:
        failures.append(archive_model.item("state", "state_unreadable", "<state>", "source"))
        return
    source_lang = str(manifest.get("source_lang") or "")
    target_lang = epub_language(manifest.get("target_lang"))
    try:
        with zipfile.ZipFile(output_path) as archive:
            names = archive.namelist()
            _generated_navigation_proof(
                archive, names, chapters, chapter_titles, target_lang, failures
            )
            _generated_content_proof(
                archive,
                names,
                chapters,
                source_lang,
                target_lang,
                failures,
                checked,
            )
    except (OSError, zipfile.BadZipFile, etree.XMLSyntaxError):
        failures.append(
            archive_model.item("state", "preserved_chapter_unreadable", "<output>", "source")
        )
