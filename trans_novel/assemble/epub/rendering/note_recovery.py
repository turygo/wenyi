"""独立重证可恢复的注释引用，不改变源槽位或既有译文。"""

from __future__ import annotations

import zipfile

from trans_novel.epub.archive import preflight_zip, read_member
from trans_novel.epub.markup import resource_parser
from trans_novel.epub.navigation import parse_nav_landmarks
from trans_novel.epub.notes import (
    NoteRelations,
    detect_recoverable_note_relations,
    note_resources_by_semantics,
)
from trans_novel.epub.package import read_package
from trans_novel.epub.richtext import InlineMark, NoteReference, RichSource
from trans_novel.ingest.models import Chapter


def _inside(path: tuple[int, ...], parent: tuple[int, ...]) -> bool:
    return len(path) > len(parent) and path[: len(parent)] == parent


def _content_chain(source: RichSource, link: InlineMark) -> tuple[str, ...] | None:
    """仅允许可完整复制的单链引用内容，拒绝分支及内部原子对象。"""
    if link.kind != "link" or link.tag.rsplit("}", 1)[-1].lower() != "a":
        return None
    descendants = sorted(
        (mark for mark in source.marks if _inside(mark.path, link.path)),
        key=lambda mark: mark.path,
    )
    previous = link.path
    for mark in descendants:
        if mark.tag.rsplit("}", 1)[-1].lower() not in {"span", "sup"} or not _inside(
            mark.path, previous
        ):
            return None
        previous = mark.path
    runs = [run for run in source.runs if link.id in run.marks]
    if (
        any(run.atom is not None for run in runs)
        or any(_inside(atom.path, link.path) for atom in source.atoms)
        or "".join(run.text for run in runs) != link.source_text
    ):
        return None
    return (link.id, *(mark.id for mark in descendants))


def note_references_for_source(
    source: RichSource,
    resource_href: str,
    block_path: tuple[int, ...],
    relations: NoteRelations,
) -> list[NoteReference]:
    """将已验证关系匹配到原有普通链接，不改变其文字或几何。"""
    markers = {
        tuple(marker["path"]): marker
        for marker in relations["markers"]
        if marker["resource_href"] == resource_href
    }
    references = []
    for mark in source.marks:
        marker = markers.get((*block_path, *mark.path))
        if marker is None or mark.source_text != marker["label"]:
            continue
        content = _content_chain(source, mark)
        if content is None:
            continue
        references.append(
            NoteReference(
                mark_id=mark.id,
                label=marker["label"],
                kind=marker["kind"],
                target_resource=marker["target_resource"],
                target_path=tuple(marker["target_path"]),
                content_marks=content,
            )
        )
    return references


def _read_relations(source_path: str) -> NoteRelations:
    """只读取安全包中声明的 HTML 资源，并保留目录排除及注释上下文。"""
    with zipfile.ZipFile(source_path) as archive:
        preflight_zip(archive)
        failures = []
        package = read_package(archive, failures)
        if failures:
            first = failures[0]
            raise ValueError(f"note recovery package rejected: {first['code']} ({first['path']})")
        model = package["model"]
        resources = {}
        for href in model["content_paths"]:
            tree, _, _ = resource_parser(read_member(archive, archive.getinfo(href)))
            resources[href] = tree.getroot()
        landmarks = parse_nav_landmarks(archive, [str(item["path"]) for item in model["nav_items"]])
        return detect_recoverable_note_relations(
            resources,
            excluded_resources=model["toc_paths"],
            note_resources=note_resources_by_semantics([*model["guide_entries"], *landmarks]),
        )


def _reference_inventory(
    source_path: str, chapters: list[Chapter]
) -> list[tuple[RichSource, list[NoteReference]]]:
    sources = [
        segment.epub_state
        for chapter in chapters
        for segment in chapter.segments
        if segment.epub_state is not None and segment.epub_state.rich_source is not None
    ]
    if not sources:
        return []
    relations = _read_relations(source_path)
    return [
        (
            state.rich_source,
            note_references_for_source(
                state.rich_source, state.resource_href, state.block_path, relations
            ),
        )
        for state in sources
    ]


def bind_note_references(source_path: str, chapters: list[Chapter]) -> None:
    """仅绑定附加证明；不写入旧脚注元数据、源字段、源 runs 或译文。"""
    for source, references in _reference_inventory(source_path, chapters):
        source.note_references = references


def validate_note_references(source_path: str, chapters: list[Chapter]) -> None:
    """重证已附加证明的完整集合；历史空清单仍由目标契约限制对象使用。"""
    if not any(
        segment.epub_state is not None
        and segment.epub_state.rich_source is not None
        and segment.epub_state.rich_source.note_references
        for chapter in chapters
        for segment in chapter.segments
    ):
        return
    for source, references in _reference_inventory(source_path, chapters):
        if source.note_references and source.note_references != references:
            raise ValueError("rich source note reference proof mismatch")


__all__ = [
    "bind_note_references",
    "note_references_for_source",
    "validate_note_references",
]
