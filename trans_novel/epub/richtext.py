"""源文证据与独立译文内联结构的数据契约。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RICHTEXT_VERSION = 1


class InlineMark(BaseModel):
    """源内联节点的身份和语义证据，不约束译文顺序。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    path: tuple[int, ...]
    tag: str
    attributes: dict[str, str] = Field(default_factory=dict)
    source_text: str
    kind: Literal["bold", "italic", "link", "style", "decoration"] = "style"
    style_evidence: list[str] = Field(default_factory=list)


class InlineAtom(BaseModel):
    """原书中不可翻译的对象，内容和链接身份由原书提供。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    path: tuple[int, ...]
    kind: Literal["note", "anchor", "linebreak", "media", "opaque"]
    label: str = ""


class InlineRun(BaseModel):
    """译文文字及其格式，或者一个不可拆分的源对象引用。"""

    model_config = ConfigDict(extra="forbid")
    text: str = ""
    marks: tuple[str, ...] = ()
    atom: str | None = None
    slot_id: str | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> InlineRun:
        if self.atom is not None and (self.text or self.slot_id is not None):
            raise ValueError("rich atom run cannot contain prose or a source slot")
        if len(set(self.marks)) != len(self.marks):
            raise ValueError("rich run has duplicate marks")
        return self


class RichSource(BaseModel):
    """已绑定原书的内联来源清单。"""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    marks: list[InlineMark] = Field(default_factory=list)
    atoms: list[InlineAtom] = Field(default_factory=list)
    runs: list[InlineRun] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    @model_validator(mode="after")
    def _check_inventory(self) -> RichSource:
        ids = [item.id for item in (*self.marks, *self.atoms)]
        if len(ids) != len(set(ids)):
            raise ValueError("rich source IDs must be unique")
        known_marks = {mark.id for mark in self.marks}
        known_atoms = {atom.id for atom in self.atoms}
        if any(set(run.marks) - known_marks for run in self.runs):
            raise ValueError("rich source run has an unknown mark")
        actual = Counter(run.atom for run in self.runs if run.atom is not None)
        if actual != Counter(dict.fromkeys(known_atoms, 1)):
            raise ValueError("rich source must own each atom exactly once")
        return self


class RichTarget(BaseModel):
    """独立于源槽位数量及顺序的已接受译文结构。"""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    runs: list[InlineRun] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


def validate_rich_target(
    source: RichSource, target: RichTarget, *, expected_text: str | None = None
) -> None:
    """校验来源引用和文字守恒；不冒充语义正确性证明。"""
    if expected_text is not None and target.text != expected_text:
        raise ValueError("rich annotation changed the target text")
    marks = {mark.id: mark for mark in source.marks}
    atoms = {atom.id for atom in source.atoms}
    used: set[str] = set()
    for run in target.runs:
        if run.slot_id is not None:
            raise ValueError("rich target cannot contain source slot coordinates")
        if set(run.marks) - marks.keys():
            raise ValueError("rich target contains unknown marks")
        if run.text.strip():
            used.update(run.marks)
        elif (
            not run.text
            and run.atom is None
            and run.marks
            and not any(
                marks[mark_id].attributes.get("id") or marks[mark_id].attributes.get("name")
                for mark_id in run.marks
            )
        ):
            raise ValueError("empty marked run requires an original anchor identity")
    actual = Counter(run.atom for run in target.runs if run.atom is not None)
    if actual != Counter(dict.fromkeys(atoms, 1)):
        raise ValueError("rich target must preserve each source atom exactly once")
    required = {
        mark.id
        for mark in source.marks
        if mark.source_text.strip()
        and (
            mark.kind in {"bold", "italic", "link"}
            or (
                mark.kind == "style"
                and any(
                    re.search(
                        r"font-(?:weight\s*:\s*(?:bold|[7-9]00)|style\s*:\s*(?:italic|oblique))",
                        evidence,
                        re.I,
                    )
                    for evidence in mark.style_evidence
                )
            )
        )
    }
    if required - used:
        raise ValueError("rich target lost a required semantic mark")
    identity_marks = {
        mark.id for mark in source.marks if mark.attributes.get("id") or mark.attributes.get("name")
    }
    represented = {mark_id for run in target.runs for mark_id in run.marks}
    if identity_marks - represented:
        raise ValueError("rich target lost an inline anchor identity")


def rich_source_digest(source: RichSource) -> str:
    """冻结格式证据供恢复与输出指纹使用。"""
    import hashlib

    return hashlib.sha256(source.model_dump_json().encode("utf-8")).hexdigest()


__all__ = [
    "RICHTEXT_VERSION",
    "InlineAtom",
    "InlineMark",
    "InlineRun",
    "RichSource",
    "RichTarget",
    "rich_source_digest",
    "validate_rich_target",
]
