"""源文证据与独立译文内联结构的数据契约。"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

RICHTEXT_VERSION = 2
SemanticRole = Literal["bold", "italic", "link", "superscript", "subscript", "literal"]


def tag_semantics(tag: str) -> tuple[SemanticRole, ...]:
    """从不可改写的节点类型提取语义，不把源坐标当作中文边界。"""
    return {
        "b": ("bold",),
        "strong": ("bold",),
        "i": ("italic",),
        "em": ("italic",),
        "a": ("link",),
        "sup": ("superscript",),
        "sub": ("subscript",),
        "code": ("literal",),
        "kbd": ("literal",),
        "samp": ("literal",),
        "pre": ("literal",),
    }.get(tag.rsplit("}", 1)[-1].lower(), ())


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
    semantics: tuple[SemanticRole, ...] = ()
    required: bool = False


class InlineAtom(BaseModel):
    """原书中不可翻译的对象，内容和链接身份由原书提供。"""

    model_config = ConfigDict(extra="forbid")
    id: str
    path: tuple[int, ...]
    kind: Literal["note", "anchor", "linebreak", "media", "opaque"]
    label: str = ""


class NoteReference(BaseModel):
    """在全书原始资源上已证明的双向注释关系，不改变源文字账本。"""

    model_config = ConfigDict(extra="forbid")
    mark_id: str
    label: str
    kind: Literal["noteref", "backlink"]
    target_resource: str
    target_path: tuple[int, ...]
    content_marks: tuple[str, ...]


class InlineRun(BaseModel):
    """译文文字及其格式，或者一个不可拆分的源对象引用。"""

    model_config = ConfigDict(extra="forbid")
    text: str = ""
    marks: tuple[str, ...] = ()
    atom: str | None = None
    slot_id: str | None = None
    omission: Literal["implicit", "merged"] | None = None
    explanation: str = ""
    note: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        value = handler(self)
        if self.omission is None:
            # 普通旧记录不新增字段，保持旧 JSON 字节和源摘要稳定。
            value.pop("omission", None)
            value.pop("explanation", None)
        if self.note is None:
            value.pop("note", None)
        return value

    @model_validator(mode="after")
    def _check_shape(self) -> InlineRun:
        if self.atom is not None and (self.text or self.slot_id is not None):
            raise ValueError("rich atom run cannot contain prose or a source slot")
        if self.note is not None and (
            self.text
            or self.slot_id is not None
            or self.atom is not None
            or self.omission is not None
        ):
            raise ValueError("rich note object cannot contain prose, a slot or another object")
        if len(set(self.marks)) != len(self.marks):
            raise ValueError("rich run has duplicate marks")
        if self.omission is not None:
            if self.text or self.atom is not None or self.slot_id is not None or not self.marks:
                raise ValueError("rich omission requires an empty marked target range")
            if not self.explanation.strip():
                raise ValueError("rich omission requires an explanation")
        elif self.explanation:
            raise ValueError("rich explanation requires an omission")
        return self


class RichSource(BaseModel):
    """已绑定原书的内联来源清单。"""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1, 2] = 2
    marks: list[InlineMark] = Field(default_factory=list)
    atoms: list[InlineAtom] = Field(default_factory=list)
    runs: list[InlineRun] = Field(default_factory=list)
    note_references: list[NoteReference] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict:
        value = handler(self)
        if not self.note_references:
            # 不增加空证明字段，旧待标注来源摘要保持逐字节相同。
            value.pop("note_references", None)
        return value

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    @model_validator(mode="after")
    def _check_inventory(self) -> RichSource:
        if any(
            run.omission is not None or run.explanation or run.note is not None for run in self.runs
        ):
            raise ValueError("rich source cannot contain target omission declarations")
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
        _note_reference_inventory(self)
        return self


class RichTarget(BaseModel):
    """独立于源槽位数量及顺序的已接受译文结构。"""

    model_config = ConfigDict(extra="forbid")
    version: Literal[1, 2, 3] = 2
    runs: list[InlineRun] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)

    @model_validator(mode="after")
    def _check_version(self) -> RichTarget:
        if self.version != 3 and any(
            run.omission is not None or run.note is not None for run in self.runs
        ):
            raise ValueError("rich omission or note object requires target contract version 3")
        return self


def note_label_key(label: str) -> str:
    """仅对短注释编号做同号比较；真实译文字符从不规范化或删改。"""
    normalized = unicodedata.normalize("NFKC", label).strip()
    if re.fullmatch(r"(?:[0-9]{1,4}|\[[0-9]{1,4}\]|\([0-9]{1,4}\))\.?", normalized):
        return normalized.removesuffix(".").strip("[]()")
    if re.fullmatch(r"(?:\*{1,3}|†{1,3}|‡{1,3}|§{1,3}|¶{1,3})", normalized):
        return normalized
    raise ValueError("rich note reference requires a short original label")


def _note_reference_inventory(source: RichSource) -> dict[str, NoteReference]:
    marks = {mark.id: mark for mark in source.marks}
    inventory: dict[str, NoteReference] = {}
    for reference in source.note_references:
        if source.version != 2 or reference.mark_id in inventory:
            raise ValueError("rich note reference requires unique source version 2 evidence")
        mark = marks.get(reference.mark_id)
        if (
            mark is None
            or mark.tag.rsplit("}", 1)[-1].lower() != "a"
            or mark.source_text != reference.label
            or not reference.content_marks
            or reference.content_marks[0] != reference.mark_id
        ):
            raise ValueError("rich note reference source identity mismatch")
        note_label_key(reference.label)
        chain = [mark]
        for key in reference.content_marks[1:]:
            child = marks.get(key)
            if (
                child is None
                or child.tag.rsplit("}", 1)[-1].lower() not in {"span", "sup"}
                or len(child.path) <= len(chain[-1].path)
                or child.path[: len(chain[-1].path)] != chain[-1].path
            ):
                raise ValueError("rich note reference has an invalid content chain")
            chain.append(child)
        descendants = {
            item.id
            for item in source.marks
            if len(item.path) > len(mark.path) and item.path[: len(mark.path)] == mark.path
        }
        if set(reference.content_marks[1:]) != descendants:
            raise ValueError("rich note reference content chain is incomplete")
        inventory[reference.mark_id] = reference
    return inventory


def _validate_note_targets(source: RichSource, target: RichTarget) -> None:
    references = _note_reference_inventory(source)
    restored: set[str] = set()
    marks = {mark.id: mark for mark in source.marks}
    for run in target.runs:
        if run.note is None:
            continue
        reference = references.get(run.note)
        if reference is None or source.version != 2 or target.version != 3:
            raise ValueError(
                "rich note object requires proven source evidence and target version 3"
            )
        if run.note in restored:
            raise ValueError("rich target has a duplicate note object")
        if run.marks[-len(reference.content_marks) :] != reference.content_marks:
            raise ValueError("rich note object lost its source content chain")
        parent_path = marks[run.note].path
        if any(
            len(marks[key].path) >= len(parent_path)
            or parent_path[: len(marks[key].path)] != marks[key].path
            for key in run.marks[: -len(reference.content_marks)]
        ):
            raise ValueError("rich note object has an unrelated parent scope")
        restored.add(run.note)
    for key, reference in references.items():
        label = "".join(run.text for run in target.runs if key in run.marks)
        if key in restored:
            if label:
                raise ValueError("rich note reference has both restored and existing text")
        elif note_label_key(label) != note_label_key(reference.label):
            raise ValueError("rich note reference lost its existing target label")


def _omitted_marks(
    source: RichSource, target: RichTarget, marks: Mapping[str, InlineMark]
) -> set[str]:
    omitted: set[str] = set()
    meaningful = {
        key
        for run in target.runs
        if run.text or run.atom is not None or run.note is not None
        for key in run.marks
    }
    for run in target.runs:
        if run.omission is None:
            continue
        if source.version != 2 or target.version != 3:
            raise ValueError("rich omission requires source version 2 and target version 3")
        for key in run.marks:
            mark = marks[key]
            roles = semantic_roles(mark)
            if (
                not mark.source_text.strip()
                or mark.kind == "decoration"
                or bool(roles - {"bold", "italic"})
                or (not roles and not (mark.kind == "style" and mark.required))
            ):
                raise ValueError("rich omission contains an ineligible source mark")
            if key in omitted:
                raise ValueError("rich source mark has duplicate omissions")
            if key in meaningful:
                raise ValueError("rich source mark has both omission and meaningful target range")
            omitted.add(key)
    return omitted


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
        run._check_shape()
        if run.slot_id is not None:
            raise ValueError("rich target cannot contain source slot coordinates")
        if set(run.marks) - marks.keys():
            raise ValueError("rich target contains unknown marks")
        if run.text.strip() or run.note is not None:
            used.update(run.marks)
        elif (
            not run.text
            and run.atom is None
            and run.marks
            and run.omission is None
            and not any(
                marks[mark_id].attributes.get("id") or marks[mark_id].attributes.get("name")
                for mark_id in run.marks
            )
        ):
            raise ValueError("empty marked run requires an original anchor identity")
    used.update(_omitted_marks(source, target, marks))
    _validate_note_targets(source, target)
    actual = Counter(run.atom for run in target.runs if run.atom is not None)
    if actual != Counter(dict.fromkeys(atoms, 1)):
        raise ValueError("rich target must preserve each source atom exactly once")
    if source.version == 2:
        if target.version not in {2, 3}:
            raise ValueError("rich target requires explicit contract migration")
        validate_rich_nesting(marks, target.runs)
        note_ids = {atom.id for atom in source.atoms if atom.kind == "note"}
        if any(
            run.atom in note_ids and any("link" in semantic_roles(marks[key]) for key in run.marks)
            for run in target.runs
        ):
            raise ValueError("rich target contains nested note links")
    required = {
        mark.id
        for mark in source.marks
        if mark.source_text.strip()
        and (
            (source.version == 2 and (bool(semantic_roles(mark)) or mark.required))
            or (source.version == 1 and mark.kind in {"bold", "italic", "link"})
            or (
                source.version == 1
                and mark.kind == "style"
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


def semantic_roles(mark: InlineMark) -> frozenset[SemanticRole]:
    """标签语义不可被可选样式覆盖，纯装饰没有必须翻译的语义。"""
    roles = set(tag_semantics(mark.tag))
    if mark.kind != "decoration":
        roles.update(mark.semantics)
        if mark.kind in {"bold", "italic", "link"}:
            roles.add(mark.kind)
    return frozenset(roles)


def validate_rich_nesting(marks: Mapping[str, InlineMark], runs: Sequence[InlineRun]) -> None:
    """在块级拒绝非法链接嵌套及重开的源链接，不限制强调换序。"""
    meaningful = {
        key
        for run in runs
        if run.text or run.atom is not None or run.note is not None
        for key in run.marks
    }
    active: tuple[str, ...] = ()
    opened_links: set[str] = set()
    for run in runs:
        if not run.text and run.atom is None and run.note is None and set(run.marks) <= meaningful:
            continue
        links = [key for key in run.marks if "link" in semantic_roles(marks[key])]
        if len(links) > 1:
            raise ValueError("rich target contains nested links")
        common = 0
        while common < min(len(active), len(run.marks)) and active[common] == run.marks[common]:
            common += 1
        for key in run.marks[common:]:
            if "link" in semantic_roles(marks[key]):
                if key in opened_links:
                    raise ValueError("rich target splits a source link entity")
                opened_links.add(key)
        active = run.marks


def rich_source_digest(source: RichSource) -> str:
    """冻结格式证据供恢复与输出指纹使用。"""
    import hashlib

    return hashlib.sha256(source.model_dump_json().encode("utf-8")).hexdigest()


__all__ = [
    "RICHTEXT_VERSION",
    "InlineAtom",
    "InlineMark",
    "InlineRun",
    "NoteReference",
    "RichSource",
    "RichTarget",
    "SemanticRole",
    "note_label_key",
    "rich_source_digest",
    "semantic_roles",
    "tag_semantics",
    "validate_rich_nesting",
    "validate_rich_target",
]
