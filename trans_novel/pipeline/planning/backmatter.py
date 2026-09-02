"""Classification policy for front and back matter chapters."""

from __future__ import annotations

import re

# 英文关键词：词边界、大小写不敏感匹配
_EN_KEYWORDS = (
    "notes",
    "endnotes",
    "footnotes",
    "index",
    "bibliography",
    "references",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "copyright",
    "works cited",
    "contents",
    "epigraph",
    "dedication",
    "title page",
)

# 中文关键词：子串匹配
_ZH_KEYWORDS = (
    "注释",
    "尾注",
    "脚注",
    "索引",
    "参考文献",
    "参考书目",
    "引用文献",
    "致谢",
    "鸣谢",
    "版权",
    "关于作者",
    "作者简介",
    "目录",
    "扉页",
    "献词",
    "题词",
)

# 位置门控：章序落在全书前 15%（版权页/献词/致谢）或后 35%（注释/索引/
# 参考文献/致谢）之外时，关键词命中视为正文撞词，不判为附属章。
_FRONT_ZONE = 0.15
_BACK_ZONE = 0.65

_BM_RANK = {"skip": 0, "light": 1, "full": 2}


def back_matter_mode(policy, title: str, index: int, total: int) -> str | None:
    """Return the configured skip/light mode when the chapter is back matter."""
    mode = getattr(policy, "back_matter", policy)
    if mode in ("skip", "light") and is_back_matter(title, index=index, total=total):
        return mode
    return None


def back_matter_rank(mode: str) -> int | None:
    return _BM_RANK.get(mode)


def is_back_matter_upgrade(previous: str | None, current: str) -> bool:
    previous_rank = back_matter_rank(previous) if previous is not None else None
    current_rank = back_matter_rank(current)
    return previous_rank is not None and current_rank is not None and current_rank > previous_rank


def is_back_matter(title: str, *, index: int | None = None, total: int | None = None) -> bool:
    """标题是否像附属章节（注释/索引/致谢等）。空串返回 False。

    给出 index/total（全书章序、总章数）时启用位置门控：仅当该章位于
    全书首部或尾部才可能判真；不给则仅按标题匹配（向后兼容）。
    """
    title = (title or "").strip()
    if not title:
        return False
    if index is not None and total is not None and total > 1:
        frac = index / (total - 1)
        if _FRONT_ZONE < frac < _BACK_ZONE:
            return False
    for kw in _EN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", title, re.I):
            return True
    # 短语：about the author / about the authors
    if re.search(r"\babout the authors?\b", title, re.I):
        return True
    return any(kw in title for kw in _ZH_KEYWORDS)


__all__ = ["back_matter_mode", "back_matter_rank", "is_back_matter", "is_back_matter_upgrade"]
