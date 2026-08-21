"""Strict schemas for the local benchmark corpus workflow."""

from __future__ import annotations

from typing import Annotated, Any, Literal

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
            and info.field_name not in {"style", "book_synopsis", "gender"}
        ):
            raise ValueError("string must not be empty")
        return value


class BookEntry(StrictModel):
    book_id: _NONEMPTY
    path: str
    split: Split
    license_note: _NONEMPTY


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
    license_note: _NONEMPTY
    split: Split
    format: _NONEMPTY
    title: _NONEMPTY
    chapter_count: int = Field(ge=0)
    parser_schema: int


class EmittedManifest(ArtifactModel):
    schema_version: Literal[1]
    run_input_schema_version: int
    books: list[EmittedManifestBook] = Field(min_length=1)


class GlossaryPreparation(StrictModel):
    """A complete, production-compatible frozen glossary row."""

    source: _NONEMPTY
    target: _NONEMPTY
    reading: str = ""
    type: _NONEMPTY = "术语"
    gender: str = ""
    aliases: list[str] = Field(default_factory=list)
    first_chapter: int | None = Field(default=None, ge=0)
    note: str = ""
    confidence: _NONEMPTY = "medium"
    locked: bool = False
    status: _NONEMPTY = "ok"


class ChapterSourceDigest(StrictModel):
    """Ordered proof tying a frozen chapter to its original source bytes."""

    chapter_index: int = Field(ge=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BookPreparation(StrictModel):
    book_id: _NONEMPTY
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis: dict[str, Any]
    style: str = ""
    style_brief: str = ""
    book_synopsis: str = ""
    chapter_digests: dict[str, str] = Field(default_factory=dict)
    source_digests: list[ChapterSourceDigest] = Field(default_factory=list)
    glossary: list[GlossaryPreparation] = Field(default_factory=list)
    node_fingerprints: dict[str, str] = Field(default_factory=dict)
    # Physical preparation evidence lives beside the immutable semantic fields.
    # It is intentionally excluded from preparation_sha256 (see runner).
    usage: dict[str, Any] = Field(default_factory=dict)
    telemetry_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    telemetry_path: str | None = None

    @model_validator(mode="after")
    def validate_chapter_keys(self) -> BookPreparation:
        if any(not key.isdecimal() for key in self.chapter_digests):
            raise ValueError("chapter_digests keys must be decimal chapter indexes")
        indices = [row.chapter_index for row in self.source_digests]
        if indices != sorted(set(indices)):
            raise ValueError("source_digests must be ordered by unique chapter index")
        if self.telemetry_path is not None:
            path = self.telemetry_path.replace("\\", "/")
            if path.startswith("/") or ":" in path or any(part == ".." for part in path.split("/")):
                raise ValueError("telemetry_path must be relative")
        return self


class PreparationSpec(StrictModel):
    schema_version: Literal[1]
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
    primary_model: _NONEMPTY
    editor_model: _NONEMPTY
    fast_model: _NONEMPTY
    temperature: float
    seed: Literal[None]
    require_catalogued_model: Literal[True] = True
    require_thinking_disabled: Literal[True] = True

    @field_validator("temperature")
    @classmethod
    def exact_temperature(cls, value: float) -> float:
        if value != 0.1:
            raise ValueError("temperature must be exactly 0.1")
        return value

    @model_validator(mode="after")
    def explicit_thinking_disabled(self) -> PreparationSpec:
        for field_name in ("primary_model", "editor_model", "fast_model"):
            if not getattr(self, field_name).endswith(":off"):
                raise ValueError(f"{field_name} must explicitly disable thinking with ':off'")
        return self


class PreparationBundle(StrictModel):
    schema_version: Literal[1]
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_spec: PreparationSpec
    preparation_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    books: dict[_NONEMPTY, BookPreparation] = Field(min_length=1)

    @model_validator(mode="after")
    def matching_book_ids(self) -> PreparationBundle:
        mismatched = [key for key, book in self.books.items() if book.book_id != key]
        if mismatched:
            raise ValueError(f"book_id does not match books key: {mismatched}")
        return self


class Candidate(StrictModel):
    candidate_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
    )
    primary_model: _NONEMPTY
    editor_model: str | None

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
    default_context_strategy: Literal["c2"]
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
        seen_pairs: set[tuple[str, str | None]] = set()
        primaries: set[str] = set()
        controls: set[str] = set()
        for candidate in self.candidates:
            if not candidate.primary_model.endswith(":off"):
                raise ValueError("primary_model must explicitly disable thinking with ':off'")
            if candidate.editor_model is not None and not candidate.editor_model.endswith(":off"):
                raise ValueError("editor_model must explicitly disable thinking with ':off'")
            pair = (candidate.primary_model, candidate.editor_model)
            if pair in seen_pairs:
                raise ValueError("duplicate primary/editor pair")
            seen_pairs.add(pair)
            primaries.add(candidate.primary_model)
            if candidate.editor_model is None:
                controls.add(candidate.primary_model)
        if primaries != controls:
            missing = sorted(primaries - controls)
            raise ValueError(f"each primary_model needs an unpolished control: {missing}")
        return self


__all__ = [
    "ArtifactModel",
    "BookEntry",
    "BookPreparation",
    "BookSpec",
    "Candidate",
    "CandidateSpec",
    "ChallengeType",
    "ChapterSourceDigest",
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
    "GlossaryPreparation",
    "PassageSelection",
    "PreparationBundle",
    "PreparationSpec",
    "SegmentCoordinate",
    "Selection",
    "Split",
    "Stratum",
    "Subset",
]


class StudyProtocol(StrictModel):
    eligibility_text: _NONEMPTY
    consent_text: _NONEMPTY
    compensation_text: _NONEMPTY
    retention_text: _NONEMPTY


class EvaluationSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    ratings_per_output: int = Field(default=3, ge=1)


class PairwiseSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    ratings_per_comparison: int = Field(default=2, ge=1)


class PolishSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    ratings_per_pair: int = Field(default=3, ge=1)


class MQMSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    annotators_per_output: int = Field(default=2, ge=1)


class PosteditSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    editors_per_output: int = Field(default=1, ge=1)


class ContextSurfaceConfig(StrictModel):
    target_source_words: int = Field(gt=0)
    ratings_per_output: int = Field(default=3, ge=1)


class EvaluationSpec(StrictModel):
    schema_version: Literal[1]
    benchmark_id: _NONEMPTY
    run_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    raters: list[str] = Field(min_length=3)
    candidate_ids: list[str] = Field(min_length=2, max_length=6)
    calibration_units: int = Field(default=30, ge=30, le=30)
    hidden_duplicate_fraction: float = Field(default=0.10, ge=0.0, le=0.20)
    absolute: EvaluationSurfaceConfig = EvaluationSurfaceConfig(
        target_source_words=12000, ratings_per_output=3
    )
    pairwise: PairwiseSurfaceConfig = PairwiseSurfaceConfig(
        target_source_words=20000, ratings_per_comparison=2
    )
    mqm: MQMSurfaceConfig = MQMSurfaceConfig(target_source_words=10000, annotators_per_output=2)
    polish: PolishSurfaceConfig = PolishSurfaceConfig(target_source_words=10000, ratings_per_pair=3)
    postedit: PosteditSurfaceConfig = PosteditSurfaceConfig(
        target_source_words=5000, editors_per_output=1
    )
    context: ContextSurfaceConfig = ContextSurfaceConfig(
        target_source_words=5000, ratings_per_output=3
    )
    enabled_surfaces: list[Literal["attribution_final", "full_final"]] = Field(
        default_factory=lambda: ["attribution_final", "full_final"], min_length=1
    )
    study_protocol: StudyProtocol

    @field_validator("enabled_surfaces")
    @classmethod
    def unique_surfaces(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("enabled_surfaces must be unique")
        return values

    @field_validator("raters", "candidate_ids")
    @classmethod
    def safe_identifiers(cls, values: list[str]) -> list[str]:
        import re

        pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
        if len(set(values)) != len(values):
            raise ValueError("identifiers must be unique")
        for value in values:
            if not pattern.fullmatch(value) or ".." in value:
                raise ValueError("identifier is unsafe")
        return values

    @field_validator("seed")
    @classmethod
    def reject_bool_seed(cls, value: int) -> int:
        if isinstance(value, bool):
            raise ValueError("seed must be an integer, not bool")
        return value

    @model_validator(mode="after")
    def exact_rater_cardinality(self) -> EvaluationSpec:
        if self.absolute.ratings_per_output != 3:
            raise ValueError("absolute ratings_per_output must be 3")
        if self.pairwise.ratings_per_comparison != 2:
            raise ValueError("pairwise ratings_per_comparison must be 2")
        if self.polish.ratings_per_pair != 3:
            raise ValueError("polish ratings_per_pair must be 3")
        if self.mqm.annotators_per_output != 2:
            raise ValueError("mqm annotators_per_output must be 2")
        if self.postedit.editors_per_output != 1:
            raise ValueError("postedit editors_per_output must be 1")
        if self.context.ratings_per_output != 3:
            raise ValueError("context ratings_per_output must be 3")
        return self


class EvaluationResponseBase(StrictModel):
    schema_version: Literal[1] = 1
    assignment_id: _NONEMPTY
    rater_id: _NONEMPTY
    pack_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    started_at: str
    submitted_at: str
    active_ms: int = Field(ge=0)

    @field_validator("started_at", "submitted_at")
    @classmethod
    def utc_timestamp(cls, value: str) -> str:
        from datetime import datetime

        if not value.endswith("Z"):
            raise ValueError("timestamp must use UTC Z")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError as error:
            raise ValueError("invalid UTC timestamp") from error
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp must use UTC")
        return value


class AbsoluteResponse(EvaluationResponseBase):
    kind: Literal["absolute"]
    fidelity: int = Field(ge=1, le=5)
    naturalness: int = Field(ge=1, le=5)
    style_voice: int = Field(ge=1, le=5)
    consistency: int = Field(ge=1, le=5)
    context_handling: int = Field(ge=1, le=5)
    readability: int = Field(ge=1, le=5)
    format_integrity: int = Field(ge=1, le=5)
    note: str | None = None


class PairwiseResponse(EvaluationResponseBase):
    kind: Literal["pairwise"]
    preference: Literal[
        "a_much_better",
        "a_slightly_better",
        "tie",
        "b_slightly_better",
        "b_much_better",
    ]
    note: str | None = None


class PolishResponse(EvaluationResponseBase):
    kind: Literal["polish"]
    outcome: Literal[
        "clearly_improved",
        "slightly_improved",
        "no_material_change",
        "fluent_but_semantic_damage",
        "quality_declined",
    ]
    note: str | None = None


class MQMError(StrictModel):
    segment_id: _NONEMPTY
    severity: Literal["critical", "major", "minor"]
    type: Literal[
        "mistranslation",
        "omission",
        "addition",
        "hallucination",
        "terminology",
        "named_entity",
        "pronoun_reference",
        "style_register",
        "fluency",
        "formatting",
    ]
    source_quote: str | None = None
    target_quote: str | None = None
    note: _NONEMPTY

    @field_validator("source_quote", "target_quote")
    @classmethod
    def nonblank_quotes(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("quotes must be nonblank when present")
        return value.strip() if value is not None else None


class MQMResponse(EvaluationResponseBase):
    kind: Literal["mqm"]
    errors: list[MQMError]
    note: str | None = None


class ContextResponse(EvaluationResponseBase):
    kind: Literal["context"]
    judgment: Literal["correct", "incorrect", "uncertain"]
    note: str | None = None

    @model_validator(mode="after")
    def explain_uncertain(self) -> ContextResponse:
        if self.judgment != "correct" and not self.note:
            raise ValueError("context note is required for incorrect or uncertain")
        return self


class PosteditResponse(EvaluationResponseBase):
    kind: Literal["postedit"]
    edited_target: _NONEMPTY
    note: str | None = None


__all__ += [
    "AbsoluteResponse",
    "ContextResponse",
    "ContextSurfaceConfig",
    "EvaluationResponseBase",
    "EvaluationSpec",
    "EvaluationSurfaceConfig",
    "MQMError",
    "MQMResponse",
    "MQMSurfaceConfig",
    "PairwiseResponse",
    "PairwiseSurfaceConfig",
    "PolishResponse",
    "PolishSurfaceConfig",
    "PosteditResponse",
    "PosteditSurfaceConfig",
    "StudyProtocol",
]
