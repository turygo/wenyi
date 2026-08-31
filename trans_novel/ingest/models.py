"""核心数据结构：Document → Chapter → Segment。

Segment 是最小可对齐 / 可回填的翻译单元（通常一个段落或一个标题）。
EPUB Segment 额外保留原 XHTML 文本槽位，译文由代码按原文槽位长度比例分配，
据此回填内联结构，不要求模型处理槽位标记。

用 pydantic v2 BaseModel 做校验与序列化；to_dict()/from_dict() 包装保留，
供 runstore 断点续跑与既有调用方使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from trans_novel.epub.slots import EpubSegmentState

KIND_TEXT = "text"
KIND_HEADING = "heading"


class Segment(BaseModel):
    """一个可翻译单元。"""

    index: int  # 章内序号（从 0 起）
    source: str  # 原文
    kind: str = KIND_TEXT  # text | heading
    target: str | None = None  # 译文（翻译/润色后填入）
    anchor: str | None = None  # EPUB 资源内稳定的 Segment 定位键
    resource_href: str | None = None  # EPUB：Segment 所属的物理 XHTML 路径
    cont: bool = False  # 超长段被拆分后的续段：回填时并回上一段，不另起段落
    epub_state: EpubSegmentState | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Segment:
        return cls.model_validate(d)


class Chapter(BaseModel):
    """一章：有序的 Segment 列表 + 回填所需的结构信息。"""

    index: int  # 全书章序号（从 0 起）
    title: str = ""
    segments: list[Segment] = Field(default_factory=list)
    href: str | None = None  # EPUB spine item 内部路径
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def text_segments(self) -> list[Segment]:
        """需要送翻译的非空 Segment。"""
        return [s for s in self.segments if s.source.strip()]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chapter:
        return cls.model_validate(d)


class Document(BaseModel):
    """整本书。"""

    title: str = ""
    source_lang: str
    target_lang: str
    fmt: str  # epub | text
    source_path: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls.model_validate(d)
