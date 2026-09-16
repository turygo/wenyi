"""Deterministic whole-book title request catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TitleItem:
    id: str
    source: str
    kind: str
    chapter: int | None
    depth: int
    parent_id: str | None

    def request_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind,
            "chapter": self.chapter,
            "depth": self.depth,
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True, slots=True)
class TitleCatalog:
    items: tuple[TitleItem, ...]
    entry_aliases: dict[str, str]
    chapter_title_ids: dict[int, str]


def _flat(value: object) -> str:
    return " ".join(str(value or "").split())


def _entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    meta = manifest.get("meta")
    raw = meta.get("toc_entries", []) if isinstance(meta, dict) else []
    return [entry for entry in raw if isinstance(entry, dict)] if isinstance(raw, list) else []


def _canonical_path(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> str:
    meta = manifest.get("meta")
    selected = meta.get("epub_split_toc_path") if isinstance(meta, dict) else None
    if isinstance(selected, str) and selected:
        return selected
    return next(
        (path for entry in entries if isinstance((path := entry.get("toc_path")), str) and path),
        "",
    )


def build_title_catalog(manifest: dict[str, Any]) -> TitleCatalog:
    """Build the one-request title topology and all deterministic reuse links."""
    items = [TitleItem("book", _flat(manifest.get("title")), "book", None, 0, None)]
    if not items[0].source:
        raise ValueError("运行清单缺少书名")
    entries = _entries(manifest)
    raw_chapters = manifest.get("chapters", [])
    chapters = raw_chapters if isinstance(raw_chapters, list) else []
    entry_chapters = {
        chapter["toc_entry_id"]: chapter["index"]
        for chapter in chapters
        if isinstance(chapter, dict)
        and type(chapter.get("index")) is int
        and isinstance(chapter.get("toc_entry_id"), str)
    }
    canonical_path = _canonical_path(manifest, entries)
    eligible_entry_ids = [
        entry["entry_id"]
        for entry in entries
        if isinstance(entry.get("entry_id"), str)
        and _flat(entry.get("title"))
        and not entry.get("external")
    ]
    if "book" in eligible_entry_ids or len(eligible_entry_ids) != len(set(eligible_entry_ids)):
        raise ValueError("标题稳定 ID 冲突")
    reserved_entry_ids = set(eligible_entry_ids)
    canonical = [entry for entry in entries if entry.get("toc_path") == canonical_path]
    remaining = [entry for entry in entries if entry.get("toc_path") != canonical_path]
    aliases: dict[str, str] = {}
    requested_ids = {"book"}
    target_ids: dict[str, str] = {}
    entries_by_path_index = {
        (entry.get("toc_path"), entry.get("node_index")): entry
        for entry in entries
        if type(entry.get("node_index")) is int
    }

    def add(entry: dict[str, Any]) -> None:
        entry_id = entry.get("entry_id")
        source = _flat(entry.get("title"))
        if not isinstance(entry_id, str) or not entry_id or not source or entry.get("external"):
            return
        if entry_id in requested_ids:
            raise ValueError(f"标题稳定 ID 冲突: {entry_id}")
        target_key = entry.get("target_key")
        if isinstance(target_key, str) and target_key in target_ids:
            aliases[entry_id] = target_ids[target_key]
            return
        parent = entries_by_path_index.get((entry.get("toc_path"), entry.get("parent_index")))
        parent_entry_id = parent.get("entry_id") if isinstance(parent, dict) else None
        parent_id = (
            aliases.get(parent_entry_id, parent_entry_id)
            if isinstance(parent_entry_id, str)
            else None
        )
        if parent_id not in requested_ids:
            parent_id = None
        depth = entry.get("depth")
        items.append(
            TitleItem(
                entry_id,
                source,
                "toc",
                entry_chapters.get(entry_id),
                depth if type(depth) is int and depth >= 0 else 0,
                parent_id,
            )
        )
        aliases[entry_id] = entry_id
        requested_ids.add(entry_id)
        if isinstance(target_key, str) and target_key:
            target_ids[target_key] = entry_id

    for entry in canonical:
        add(entry)
    for entry in remaining:
        add(entry)

    chapter_ids: dict[int, str] = {}
    for chapter in chapters:
        if not isinstance(chapter, dict) or type(chapter.get("index")) is not int:
            continue
        chapter_index = chapter["index"]
        entry_id = chapter.get("toc_entry_id")
        aliased = aliases.get(entry_id) if isinstance(entry_id, str) else None
        if aliased is not None:
            chapter_ids[chapter_index] = aliased
            continue
        source = _flat(chapter.get("title"))
        if not source:
            continue
        title_id = f"chapter:{chapter_index}"
        if title_id in requested_ids or title_id in reserved_entry_ids:
            raise ValueError(f"标题稳定 ID 冲突: {title_id}")
        items.append(TitleItem(title_id, source, "chapter", chapter_index, 0, None))
        chapter_ids[chapter_index] = title_id
        requested_ids.add(title_id)
    return TitleCatalog(tuple(items), aliases, chapter_ids)


__all__ = ["TitleCatalog", "TitleItem", "build_title_catalog"]
