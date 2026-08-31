"""Authoritative EPUB text-slot value contract."""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from trans_novel.ingest.models import Segment

_WS_RE = re.compile(r"[ \t\r\n\f\v]+")


class EpubTextSlot(BaseModel):
    """One owned lxml ``text`` or ``tail`` location."""

    model_config = ConfigDict(extra="forbid")

    id: str
    element_path: tuple[int, ...] = ()
    field: Literal["text", "tail"]
    source_value: str
    target_value: str | None = None


class EpubSegmentState(BaseModel):
    """Persisted source coordinates and ordered slot contract for a segment."""

    model_config = ConfigDict(extra="forbid")

    resource_href: str
    resource_sha256: str
    block_path: tuple[int, ...] = ()
    block_fingerprint: str
    parse_mode: Literal["xml", "recovered"]
    slots: list[EpubTextSlot] = Field(default_factory=list)
    slot_contract_sha256: str

    @model_validator(mode="after")
    def _validate_slots(self) -> EpubSegmentState:
        ids = [slot.id for slot in self.slots]
        if len(ids) != len(set(ids)):
            raise ValueError("EPUB slot IDs must be unique")
        locations = [(slot.element_path, slot.field) for slot in self.slots]
        if len(locations) != len(set(locations)):
            raise ValueError("EPUB slot locations must be unique")
        for slot in self.slots:
            if not slot.source_value.strip() and slot.target_value not in (
                None,
                "",
                slot.source_value,
            ):
                raise ValueError("EPUB whitespace-only slot must be empty or exact source")
        return self


def slot_contract_digest(slots: list[EpubTextSlot]) -> str:
    payload = [
        {
            "id": slot.id,
            "element_path": list(slot.element_path),
            "field": slot.field,
            "source_value": slot.source_value,
        }
        for slot in slots
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def source_slot_text(slots: list[EpubTextSlot]) -> str:
    """Return the exact source run represented by ordered slots."""
    return "".join(slot.source_value for slot in slots)


def target_slot_text(slots: list[EpubTextSlot], *, require_assigned: bool = True) -> str:
    """Return exact target values, optionally rejecting unassigned slots."""
    values: list[str] = []
    for slot in slots:
        if slot.target_value is None:
            if require_assigned:
                raise ValueError("EPUB slot target is not assigned")
            values.append("")
        else:
            values.append(slot.target_value)
    return "".join(values)


def normalized_source_text(slots: list[EpubTextSlot]) -> str:
    return _WS_RE.sub(" ", source_slot_text(slots)).strip()


def normalized_target_text(slots: list[EpubTextSlot]) -> str:
    return _WS_RE.sub(" ", target_slot_text(slots)).strip()


def source_slot_transport(segment: Segment) -> list[dict[str, str]]:
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    return [
        {"id": slot.id, "value": slot.source_value if slot.source_value.strip() else ""}
        for slot in state.slots
    ]


def source_passthrough_transport(segment: Segment) -> list[dict[str, str]]:
    """Return an atomic transport preserving each source XML value exactly."""
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    return [{"id": slot.id, "value": slot.source_value} for slot in state.slots]


def target_slot_transport(segment: Segment) -> list[dict[str, str]]:
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    if any(slot.target_value is None for slot in state.slots):
        raise ValueError("EPUB slot target is not assigned")
    return [{"id": slot.id, "value": slot.target_value} for slot in state.slots]


def distribute_slot_translation(segment: Segment, translation: str) -> list[dict[str, str]]:
    """Distribute complete translation losslessly by source-content weight."""
    state = segment.epub_state
    if state is None:
        raise ValueError("slot distribution requires an EPUB segment")
    if not isinstance(translation, str):
        raise ValueError("slot distribution requires a string translation")

    values = [""] * len(state.slots)
    active = [index for index, slot in enumerate(state.slots) if slot.source_value.strip()]
    if not active:
        if translation:
            raise ValueError(f"translation has no writable EPUB slot for {state.resource_href}")
    else:
        weights = [len(state.slots[index].source_value.strip()) for index in active]
        total = sum(weights)
        previous = 0
        enough_for_nonempty = len(translation) >= len(active)
        for position, index in enumerate(active[:-1], 1):
            cut = round(len(translation) * sum(weights[:position]) / total)
            if enough_for_nonempty:
                cut = max(previous + 1, min(cut, len(translation) - (len(active) - position)))
            else:
                cut = max(previous, min(cut, len(translation)))
            values[index] = translation[previous:cut]
            previous = cut
        values[active[-1]] = translation[previous:]
    return [
        {"id": slot.id, "value": value} for slot, value in zip(state.slots, values, strict=True)
    ]


def validate_slot_transport(segment: Segment, translation: object) -> list[tuple[str, str]]:
    state = segment.epub_state
    if state is None:
        raise ValueError("slot transport requires an EPUB segment")
    if not isinstance(translation, list) or len(translation) != len(state.slots):
        raise ValueError(f"EPUB segment slot count mismatch for {state.resource_href}")
    parsed: list[tuple[str, str]] = []
    for item in translation:
        if isinstance(item, dict):
            slot_id, value = item.get("id"), item.get("value")
        elif isinstance(item, tuple | list) and len(item) == 2:
            slot_id, value = item
        else:
            raise ValueError(f"invalid EPUB slot record for {state.resource_href}")
        if not isinstance(slot_id, str) or not isinstance(value, str):
            raise ValueError(f"invalid EPUB slot record for {state.resource_href}")
        parsed.append((slot_id, value))
    if [slot_id for slot_id, _value in parsed] != [slot.id for slot in state.slots]:
        raise ValueError(f"EPUB slot IDs/order mismatch for {state.resource_href}")
    source_passthrough = [value for _slot_id, value in parsed] == [
        slot.source_value for slot in state.slots
    ]
    for slot, (_slot_id, value) in zip(state.slots, parsed, strict=True):
        if not slot.source_value.strip() and value and not source_passthrough:
            raise ValueError(f"whitespace-only EPUB slot translated for {state.resource_href}")
    return parsed


def translation_text(segment: Segment, translation: object) -> str:
    if segment.epub_state is None:
        if not isinstance(translation, str):
            raise ValueError("non-EPUB segment translation must be a string")
        return translation
    parsed = validate_slot_transport(segment, translation)
    slots = [
        slot.model_copy(update={"target_value": value})
        for slot, (_slot_id, value) in zip(segment.epub_state.slots, parsed, strict=True)
    ]
    return _WS_RE.sub(" ", target_slot_text(slots)).strip()


def assign_segment_translation(
    segment: Segment, translation: str | list[dict[str, str]] | list[tuple[str, str]]
) -> None:
    """Validate then atomically assign every EPUB slot and its public target."""
    state = segment.epub_state
    if state is None:
        if not isinstance(translation, str):
            raise ValueError("non-EPUB segment translation must be a string")
        segment.target = translation
        return
    parsed = validate_slot_transport(segment, translation)
    new_slots = [
        slot.model_copy(update={"target_value": value})
        for slot, (_slot_id, value) in zip(state.slots, parsed, strict=True)
    ]
    state.slots = new_slots
    segment.target = _WS_RE.sub(" ", target_slot_text(new_slots)).strip()


def normalize_slot_transport(segment: Segment, translation: object) -> list[dict[str, str]]:
    """Normalize the complete target before deterministic slot distribution."""
    parsed = validate_slot_transport(segment, translation)
    if [value for _slot_id, value in parsed] == [
        slot.source_value for slot in segment.epub_state.slots
    ]:
        return [{"id": slot_id, "value": value} for slot_id, value in parsed]
    from trans_novel.postprocess.punct import normalize_zh

    complete = normalize_zh("".join(value for _slot_id, value in parsed))
    return distribute_slot_translation(segment, complete)


def reset_segment_translation(segment: Segment) -> None:
    state = segment.epub_state
    if state is None:
        segment.target = None
        return
    state.slots = [slot.model_copy(update={"target_value": None}) for slot in state.slots]
    segment.target = None


__all__ = [
    "EpubSegmentState",
    "EpubTextSlot",
    "assign_segment_translation",
    "distribute_slot_translation",
    "normalize_slot_transport",
    "normalized_source_text",
    "normalized_target_text",
    "reset_segment_translation",
    "slot_contract_digest",
    "source_passthrough_transport",
    "source_slot_text",
    "source_slot_transport",
    "target_slot_text",
    "target_slot_transport",
    "translation_text",
    "validate_slot_transport",
]
