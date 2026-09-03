"""核心数据结构：Document → Chapter → Segment。

Segment 是最小可对齐 / 可回填的翻译单元（通常一个段落或一个标题）。
EPUB Segment 额外保留原 XHTML 文本槽位，译文由代码按原文槽位长度比例分配，
据此回填内联结构，不要求模型处理槽位标记。

用 pydantic v2 BaseModel 做校验与序列化；to_dict()/from_dict() 包装保留，
供 runstore 断点续跑与既有调用方使用。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from trans_novel.epub.slots import (
    EpubSegmentState,
    normalized_target_text,
    validate_slot_transport,
)

_XML10_FORBIDDEN = frozenset(
    (*range(0x09), 0x0B, 0x0C, *range(0x0E, 0x20), *range(0xD800, 0xE000), 0xFFFE, 0xFFFF)
)
_XML10_DELETE = dict.fromkeys(_XML10_FORBIDDEN)


def sanitize_generated_text(value: str) -> str:
    """Remove only characters forbidden by XML 1.0."""
    return value.translate(_XML10_DELETE)


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

    @model_validator(mode="after")
    def _sanitize_targets(self) -> Segment:
        self.target = sanitize_generated_text(self.target) if self.target is not None else None
        state = self.epub_state
        if state is None:
            return self
        state.slots = [
            slot.model_copy(
                update={
                    "target_value": (
                        sanitize_generated_text(slot.target_value)
                        if slot.target_value is not None
                        else None
                    )
                }
            )
            for slot in state.slots
        ]
        if self.target is not None and all(slot.target_value is not None for slot in state.slots):
            self.target = normalized_target_text(state.slots)
        return self

    def assign_translation(
        self, translation: str | list[dict[str, str]] | list[tuple[str, str]]
    ) -> None:
        """Validate then atomically assign the segment translation."""
        state = self.epub_state
        if state is None:
            if not isinstance(translation, str):
                raise ValueError("non-EPUB segment translation must be a string")
            self.target = sanitize_generated_text(translation)
            return
        parsed = validate_slot_transport(state, translation)
        new_slots = [
            slot.model_copy(update={"target_value": sanitize_generated_text(value)})
            for slot, (_slot_id, value) in zip(state.slots, parsed, strict=True)
        ]
        state.slots = new_slots
        self.target = normalized_target_text(new_slots)

    def reset_translation(self) -> None:
        state = self.epub_state
        if state is None:
            self.target = None
            return
        state.slots = [slot.model_copy(update={"target_value": None}) for slot in state.slots]
        self.target = None

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
