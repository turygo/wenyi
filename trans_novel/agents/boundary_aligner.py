"""Align complete translations to generic source text boundaries."""

from __future__ import annotations

import json

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.ingest.models import SLOT_BOUNDARY_MARKER


class TextBoundaryAligner(Agent):
    """Insert validated markers without exposing EPUB coordinates to the model."""

    def align_batch(
        self,
        source_parts: list[list[str]],
        translations: list[str],
    ) -> list[str]:
        if len(source_parts) != len(translations):
            raise ValueError("boundary alignment input count mismatch")
        if not source_parts:
            return []
        records = [
            {"source_parts": parts, "translation": translation}
            for parts, translation in zip(source_parts, translations, strict=True)
        ]
        system = prompts.render("boundary_aligner_system", marker=SLOT_BOUNDARY_MARKER)
        user = prompts.render(
            "boundary_aligner_user",
            marker=SLOT_BOUNDARY_MARKER,
            records=json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        )
        for _ in range(self.config.pipeline.align_retry_limit + 1):
            try:
                aligned = self._ask_json(
                    system,
                    user,
                    key="aligned",
                    agent="analyst",
                    operation="align.boundaries",
                    strict=True,
                )
            except WorkflowProtocolError:
                continue
            if (
                isinstance(aligned, list)
                and len(aligned) == len(records)
                and all(
                    isinstance(text, str)
                    and text.count(SLOT_BOUNDARY_MARKER) == len(parts) - 1
                    and text.replace(SLOT_BOUNDARY_MARKER, "") == translation
                    and all(
                        not source.strip() or target.strip()
                        for source, target in zip(
                            parts, text.split(SLOT_BOUNDARY_MARKER), strict=True
                        )
                    )
                    for parts, translation, text in zip(
                        source_parts, translations, aligned, strict=True
                    )
                )
            ):
                return aligned
        raise WorkflowProtocolError(
            "boundary_alignment_mismatch",
            "译文边界对齐失败：模型未仅插入规定数量的边界标记",
        )
