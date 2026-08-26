"""Strict schemas for the local benchmark corpus workflow."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

_NONEMPTY = Annotated[str, Field(min_length=1)]
Split = Literal["screen", "formal", "hidden"]
Subset = Literal["screen", "continuous", "stratified", "context", "hidden"]
Stratum = Literal[
    "narrative",
    "dialogue",
    "literary",
    "long_sentence",
    "idiom_metaphor_wordplay",
    "terminology",
    "numbers_entities",
    "special_format",
]
ChallengeType = Literal[
    "pronoun_reference",
    "polysemy",
    "nickname_title",
    "abbreviation",
    "omitted_subject",
    "callback_joke_metaphor",
    "cross_segment_sentence",
    "chapter_transition",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_blank_strings(cls, value: object, info: ValidationInfo) -> object:
        if (
            isinstance(value, str)
            and not value.strip()
            and info.field_name not in {"style", "book_synopsis", "gender", "reading", "note"}
        ):
            raise ValueError("string must not be empty")
        return value


class BookEntry(StrictModel):
    book_id: _NONEMPTY
    path: str
    split: Split


class BookSpec(StrictModel):
    schema_version: Literal[1]
    source_language: Literal["en"]
    target_language: Literal["zh"]
    books: list[BookEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_books(self) -> BookSpec:
        ids = [book.book_id for book in self.books]
        if len(set(ids)) != len(ids):
            raise ValueError("book_id values must be unique")
        by_path: dict[str, list[BookEntry]] = {}
        for book in self.books:
            by_path.setdefault(book.path, []).append(book)
        if any(len(entries) > 1 for entries in by_path.values()):
            raise ValueError("a physical source may occur only once")
        counts = {
            split: sum(book.split == split for book in self.books)
            for split in ("screen", "formal", "hidden")
        }
        if counts["screen"] < 3:
            raise ValueError("at least 3 screen books are required")
        if counts["formal"] < 6:
            raise ValueError("at least 6 formal books are required")
        if counts["hidden"] < 1:
            raise ValueError("at least 1 hidden book is required")
        return self


class SegmentCoordinate(StrictModel):
    chapter_index: int = Field(ge=0)
    segment_index: int = Field(ge=0)


class FrozenTargetBefore(StrictModel):
    chapter_index: int = Field(ge=0)
    segment_index: int = Field(ge=0)
    target: _NONEMPTY


class ContextChallenge(StrictModel):
    challenge_type: ChallengeType
    source_before: list[SegmentCoordinate] = Field(min_length=1)
    source_after: list[SegmentCoordinate] = Field(default_factory=list)
    frozen_target_before: list[FrozenTargetBefore] = Field(min_length=1)
    answer_key: _NONEMPTY
    rationale: _NONEMPTY

    @model_validator(mode="after")
    def matching_frozen_targets(self) -> ContextChallenge:
        if len(self.source_before) != len(self.frozen_target_before):
            raise ValueError("frozen_target_before must match source_before length")
        return self


class PassageSelection(StrictModel):
    subset: Subset
    book_id: _NONEMPTY
    chapter_index: int = Field(ge=0)
    start_segment_index: int = Field(ge=0)
    end_segment_index: int = Field(ge=0)
    strata: list[Stratum] = Field(default_factory=list)
    context: ContextChallenge | None = None

    @model_validator(mode="after")
    def validate_range_and_context(self) -> PassageSelection:
        if self.end_segment_index < self.start_segment_index:
            raise ValueError("end_segment_index must be >= start_segment_index")
        if self.subset == "context" and self.context is None:
            raise ValueError("context selection requires context challenge")
        if self.subset != "context" and self.context is not None:
            raise ValueError("only context selection may declare context challenge")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("strata must be unique")
        return self


class Selection(StrictModel):
    schema_version: Literal[1]
    benchmark_name: _NONEMPTY
    quota_tolerance: float = Field(default=0.05, ge=0, le=0.20)
    passages: list[PassageSelection] = Field(min_length=1)


class ArtifactModel(BaseModel):
    """Strict models for the JSON artifacts emitted by corpus build."""

    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*", mode="before")
    @classmethod
    def reject_blank_strings(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            raise ValueError("string must not be blank")
        return value


class EmittedSegment(ArtifactModel):
    segment_id: _NONEMPTY
    index: int = Field(ge=0)
    source: _NONEMPTY
    kind: _NONEMPTY
    cont: bool
    anchor: str | None
    resource_href: str | None
    meta: object


class EmittedContextReference(ArtifactModel):
    segment_id: _NONEMPTY
    source: _NONEMPTY


class EmittedFrozenTarget(ArtifactModel):
    segment_id: _NONEMPTY
    target: _NONEMPTY


class EmittedContext(ArtifactModel):
    challenge_type: ChallengeType
    source_before: list[EmittedContextReference] = Field(min_length=1)
    source_after: list[EmittedContextReference] = Field(default_factory=list)
    frozen_target_before: list[EmittedFrozenTarget] = Field(min_length=1)


class EmittedRunnerRecord(ArtifactModel):
    passage_id: _NONEMPTY
    subset: Subset
    book_id: _NONEMPTY
    chapter_index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    word_count: int = Field(ge=0)
    strata: list[Stratum] = Field(default_factory=list)
    segments: list[EmittedSegment] = Field(min_length=1)
    context: EmittedContext | None

    @model_validator(mode="after")
    def validate_range_and_context(self) -> EmittedRunnerRecord:
        if self.end < self.start:
            raise ValueError("end must be >= start")
        if self.subset == "context" and self.context is None:
            raise ValueError("context subset requires context")
        if self.subset != "context" and self.context is not None:
            raise ValueError("only context subset may declare context")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("strata must be unique")
        return self


class EmittedChallengeKey(ArtifactModel):
    passage_id: _NONEMPTY
    challenge_type: ChallengeType
    answer_key: _NONEMPTY
    rationale: _NONEMPTY


class EmittedManifestBook(ArtifactModel):
    book_id: _NONEMPTY
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    basename: _NONEMPTY
    split: Split
    format: _NONEMPTY
    title: _NONEMPTY
    chapter_count: int = Field(ge=0)
    parser_schema: int


class EmittedManifest(ArtifactModel):
    schema_version: Literal[1]
    run_input_schema_version: int
    books: list[EmittedManifestBook] = Field(min_length=1)


class Candidate(StrictModel):
    candidate_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    )
    primary_model: _NONEMPTY
    editor_model: _NONEMPTY

    @field_validator("candidate_id")
    @classmethod
    def reject_parent_path(cls, value: str) -> str:
        if ".." in value:
            raise ValueError("candidate_id must not contain '..'")
        return value


class CandidateSpec(StrictModel):
    schema_version: Literal[1]
    benchmark_id: _NONEMPTY
    provider: Literal[
        "deepseek",
        "opencode-go",
        "bailian",
        "openai",
        "openrouter",
        "openai-compatible",
        "ollama",
        "vllm",
        "fake",
    ]
    fast_model: _NONEMPTY
    temperature: float
    seed: Literal[None]
    replicates: int = Field(ge=1, le=3)
    candidates: list[Candidate] = Field(min_length=1)

    @field_validator("temperature")
    @classmethod
    def exact_temperature(cls, value: float) -> float:
        if value != 0.1:
            raise ValueError("temperature must be exactly 0.1")
        return value

    @field_validator("fast_model", "seed", mode="before")
    @classmethod
    def explicit_values(cls, value):
        if value is None:
            return value
        return value

    @model_validator(mode="after")
    def validate_candidates(self) -> CandidateSpec:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_id values must be unique")
        if not self.fast_model.endswith(":off"):
            raise ValueError("fast_model must explicitly disable thinking with ':off'")
        thinking_suffixes = tuple(f":{level}" for level in ("off", "low", "medium", "high", "max"))
        seen_pairs: set[tuple[str, str]] = set()
        for candidate in self.candidates:
            for field_name in ("primary_model", "editor_model"):
                value = getattr(candidate, field_name)
                if not value.endswith(thinking_suffixes):
                    raise ValueError(f"{field_name} must explicitly select a thinking level")
            pair = (candidate.primary_model, candidate.editor_model)
            if pair in seen_pairs:
                raise ValueError("duplicate primary/editor pair")
            seen_pairs.add(pair)
        return self


__all__ = [
    "ArtifactModel",
    "BookEntry",
    "BookSpec",
    "Candidate",
    "CandidateSpec",
    "ChallengeType",
    "ContextChallenge",
    "EmittedChallengeKey",
    "EmittedContext",
    "EmittedContextReference",
    "EmittedFrozenTarget",
    "EmittedManifest",
    "EmittedManifestBook",
    "EmittedRunnerRecord",
    "EmittedSegment",
    "FrozenTargetBefore",
    "PassageSelection",
    "SegmentCoordinate",
    "Selection",
    "Split",
    "Stratum",
    "Subset",
]
