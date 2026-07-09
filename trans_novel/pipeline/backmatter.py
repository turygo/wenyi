"""附属章节（back matter）标题识别。

用途：按章节标题判断是否为注释/索引/致谢/版权等附属内容，供流水线
在 skip/light 档旁路完整翻译（省成本、避免污染全书概览）。

识别 = 标题关键词 + 位置门控：附属内容只出现在全书首部（版权/致谢）或
尾部（注释/索引/参考文献）。正文区标题撞词（如 "The Index Case"、
"Notes from Underground"）不得旁路——误报的代价是整章静默降质。
调用方给不出章序时（如单元测试、目录项）退化为纯标题匹配。

YAGNI：不做链接密度、数字密度等正文特征判断。
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


# 位置门控：章序落在全书前 15%（版权页/献词/致谢）或后 35%（注释/索引/
# 参考文献/致谢）之外时，关键词命中视为正文撞词，不判为附属章。
_FRONT_ZONE = 0.15
_BACK_ZONE = 0.65


def is_back_matter(title: str, *, index: int | None = None,
                   total: int | None = None) -> bool:
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
    for kw in _ZH_KEYWORDS:
        if kw in title:
            return True
    return False
