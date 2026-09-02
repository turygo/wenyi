"""Public metadata helpers for assembled EPUB output."""

from __future__ import annotations

from trans_novel.postprocess.punct import normalize_heading_numbering


def epub_language(lang: str | None) -> str:
    """EPUB 元数据语言码；中文目标默认标成简体中文。"""
    normalized = (lang or "").strip().replace("_", "-").lower()
    if normalized in {"", "zh", "zh-cn", "zh-hans", "cn"}:
        return "zh-Hans"
    return lang or "zh-Hans"


def translated_toc_title(entry: dict[str, object]) -> str:
    """返回目录条目的有效译名（标题编号统一为汉字），缺失时回退原标题。"""
    value = entry.get("title_translated") or entry.get("title")
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    return normalize_heading_numbering(stripped) if stripped else ""


__all__ = ["epub_language", "translated_toc_title"]
