"""核心数据结构：Document → Chapter → Segment。

Segment 是最小可对齐 / 可回填的翻译单元（通常一个段落或一个标题）。
翻译时多个 Segment 组成一个 batch 一起发给模型，模型必须返回等长的译文数组，
据此做句段对齐校验、防止整段漏译。

用 pydantic v2 BaseModel 做校验与序列化；to_dict()/from_dict() 包装保留，
供 runstore 断点续跑与既有调用方使用。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

KIND_TEXT = "text"
KIND_HEADING = "heading"


def _slot_contract_digest(slots: list[EpubTextSlot]) -> str:
    payload = [
        {
            "id": slot.id,
            "element_path": list(slot.element_path),
            "field": slot.field,
            "source_value": slot.source_value,
            "leading_whitespace": slot.leading_whitespace,
            "trailing_whitespace": slot.trailing_whitespace,
            "source_core": slot.source_core,
        }
        for slot in slots
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class EpubTextSlot(BaseModel):
    """One authoritative lxml ``text``/``tail`` value in a source XHTML block."""

    id: str
    element_path: tuple[int, ...] = ()
    field: Literal["text", "tail"]
    source_value: str
    leading_whitespace: str = ""
    trailing_whitespace: str = ""
    source_core: str
    target_core: str | None = None

    @model_validator(mode="after")
    def _validate_source_parts(self) -> EpubTextSlot:
        if self.source_value != (
            self.leading_whitespace + self.source_core + self.trailing_whitespace
        ):
            raise ValueError("EPUB slot whitespace/core does not reconstruct source value")
        if not self.source_core.strip() and self.target_core not in (None, ""):
            raise ValueError("EPUB whitespace-only slot cannot have a translated core")
        return self


class EpubSegmentState(BaseModel):
    """Persisted source coordinates and ordered slot contract for an EPUB segment."""

    resource_href: str
    resource_sha256: str
    block_path: tuple[int, ...] = ()
    block_fingerprint: str
    parse_mode: Literal["xml", "recovered"]
    slots: list[EpubTextSlot] = Field(default_factory=list)
    slot_contract_sha256: str


def _normalized_slot_text(slots: list[EpubTextSlot], *, target: bool) -> str:
    values = []
    for slot in slots:
        core = slot.target_core if target else slot.source_core
        if core is None:
            core = slot.source_core
        values.append(slot.leading_whitespace + core + slot.trailing_whitespace)
    return re.sub(r"[ \t\r\n\f\v]+", " ", "".join(values)).strip()


def slot_transport(segment: Segment, *, target: bool = True) -> list[dict[str, str]]:
    """Return the exact ordered opaque slot records used by EPUB agents."""
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    return [
        {
            "id": slot.id,
            "core": (
                slot.target_core if target and slot.target_core is not None else slot.source_core
            ),
        }
        for slot in state.slots
    ]


def validate_slot_transport(segment: Segment, translation: object) -> list[tuple[str, str]]:
    """Validate an agent slot response without mutating the segment."""
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    if not isinstance(translation, list) or len(translation) != len(state.slots):
        raise ValueError(f"EPUB segment slot count mismatch for {state.resource_href}")
    parsed: list[tuple[str, str]] = []
    for item in translation:
        if isinstance(item, dict):
            slot_id, core = item.get("id"), item.get("core")
        elif isinstance(item, tuple | list) and len(item) == 2:
            slot_id, core = item
        else:
            raise ValueError(f"invalid EPUB slot record for {state.resource_href}")
        if not isinstance(slot_id, str) or not isinstance(core, str):
            raise ValueError(f"invalid EPUB slot record for {state.resource_href}")
        parsed.append((slot_id, core))
    expected = [slot.id for slot in state.slots]
    if [slot_id for slot_id, _ in parsed] != expected:
        raise ValueError(f"EPUB slot IDs/order mismatch for {state.resource_href}")
    for slot, (_slot_id, core) in zip(state.slots, parsed, strict=True):
        if not slot.source_core.strip() and core.strip():
            raise ValueError(f"whitespace-only EPUB slot translated for {state.resource_href}")
        if slot.source_core.strip() and not core.strip():
            raise ValueError(f"empty EPUB slot translation for {state.resource_href}")
    return parsed


def translation_text(segment: Segment, translation: object) -> str:
    """Validate transport and derive the public normalized target text."""
    state = segment.epub_state
    if state is None:
        if not isinstance(translation, str):
            raise ValueError("non-EPUB segment translation must be a string")
        return translation
    parsed = validate_slot_transport(segment, translation)
    slots = [
        slot.model_copy(update={"target_core": core})
        for slot, (_slot_id, core) in zip(state.slots, parsed, strict=True)
    ]
    return _normalized_slot_text(slots, target=True)


def assign_segment_translation(
    segment: Segment, translation: str | list[dict[str, str]] | list[tuple[str, str]]
) -> None:
    """Assign a translation through the authoritative EPUB slot contract."""
    state = segment.epub_state
    if state is None:
        if not isinstance(translation, str):
            raise ValueError("non-EPUB segment translation must be a string")
        segment.target = translation
        return
    parsed = validate_slot_transport(segment, translation)
    for slot, (_slot_id, core) in zip(state.slots, parsed, strict=True):
        slot.target_core = core
    segment.target = _normalized_slot_text(state.slots, target=True)


def normalize_slot_transport(segment: Segment, translation: object) -> list[dict[str, str]]:
    """Normalize ordered slot cores while retaining the single slot contract."""
    parsed = validate_slot_transport(segment, translation)
    from trans_novel.postprocess.punct import normalize_zh_parts

    cores = normalize_zh_parts([core for _slot_id, core in parsed])
    return [
        {"id": slot_id, "core": core}
        for (slot_id, _raw_core), core in zip(parsed, cores, strict=True)
    ]


def reset_segment_translation(segment: Segment) -> None:
    """Clear a segment translation while retaining its authoritative source slots."""

    state = segment.epub_state
    if state is None:
        segment.target = None
        return
    for slot in state.slots:
        slot.target_core = None
    segment.target = None


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
