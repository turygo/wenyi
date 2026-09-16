"""全源文语义章节分类。"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from trans_novel.agents.base import Agent, WorkflowProtocolError, retry_protocol


class ChapterObservation(BaseModel):
    """一次完整章节或连续源文分块的模型观察。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    chapter_id: int
    kind: Literal["reference_only", "translatable", "uncertain"]
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be empty")
        return value


_SYSTEM_PROMPT = """You classify every supplied character of book SOURCE content for translation.
Return exactly one JSON object with chapter_id, kind, and a concise nonempty reason.
kind must be one of reference_only, translatable, uncertain. Translation is the default;
reference_only is a strict exception for content that is exclusively citations, bibliography,
endnote references, or a bare reference list with no reader-facing prose.

Always classify a table of contents, dedication, epigraph, acknowledgements, chapter narrative,
or reader-facing explanatory appendix as translatable. Short length, list formatting, front/back
matter position, or publishing/legal vocabulary alone never makes content reference_only. A pure
bibliography or pure endnote reference list is reference_only. Publishing/legal material mixed
with credits, explanation, acknowledgements, or other reader-facing text is uncertain. Any mixture
of reference-only material and substantive or reader-facing content is uncertain.

Examples:
- "For M., who made this possible." => translatable (dedication).
- "Acknowledgements ... I thank my editor." => translatable.
- "Contents / Chapter 1 / Chapter 2" => translatable.
- "[1] Smith, A. Title. 2020. [2] Doe, B. Title. 2021." => reference_only.
- "Copyright 2024 ... Thanks to the production team for their support." => uncertain.
Title, structural context, and semantic hints are advisory only: never preserve from them alone."""


class ChapterClassifier(Agent):
    def classify(
        self,
        *,
        chapter_id: int,
        title: str,
        context: str,
        source: str,
        hints: list[str],
    ) -> ChapterObservation:
        user = json.dumps(
            {
                "chapter_id": chapter_id,
                "title_context": title,
                "structural_context": context,
                "semantic_hints": hints,
                "source": source,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        def call() -> ChapterObservation:
            data = self._ask_json(
                _SYSTEM_PROMPT,
                user,
                agent="analyst",
                operation="chapter.classify",
                strict=True,
                max_tokens=500,
            )
            try:
                result = ChapterObservation.model_validate(data)
            except ValidationError as error:
                raise WorkflowProtocolError("chapter_classification_invalid") from error
            if result.chapter_id != chapter_id:
                raise WorkflowProtocolError("chapter_classification_id_mismatch")
            return result

        return retry_protocol(call, retries=self.config.pipeline.protocol_retry_limit)


__all__ = ["ChapterClassifier", "ChapterObservation"]
