"""附属章节（back matter）标题识别。

用途：按章节标题判断是否为注释/索引/致谢/版权等附属内容，供流水线
在 skip/light 档旁路完整翻译（省成本、避免污染全书概览）。

YAGNI：只做标题关键词匹配，不做链接密度、数字密度等正文特征判断。
"""

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
)


def is_back_matter(title: str) -> bool:
    """标题是否像附属章节（注释/索引/致谢等）。空串返回 False。"""
    title = (title or "").strip()
    if not title:
        return False
    for kw in _EN_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", title, re.I):
            return True
    # 短语：about the author / about the authors
    if re.search(r"\babout the authors?\b", title, re.I):
        return True
    for kw in _ZH_KEYWORDS:
        if kw in title:
            return True
    return False
