"""翻译 Agent。

核心保证：句段对齐——输入 N 段，输出必须是 N 段，一一对应。
策略：
1. 整批翻译并要求等长 JSON 数组；
2. 段数不符则重试（最多 align_retry_limit 次）；
3. 仍不符则逐段单独翻译兜底，从结构上保证 1:1，杜绝整段漏译。

模型路由按功能 Agent 选择：正文走 translator（operation=translate.batch）；附属章
旁路走 light-translator（operation=translate.back_matter，由调用方显式传 agent）。
operation 只作用量/调试归因，不参与路由。
"""

from __future__ import annotations

import json

from trans_novel.agents import langprofile, prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.glossary.store import GlossaryTerm
from trans_novel.ingest.models import (
    Segment,
    slot_transport,
    validate_slot_transport,
)
from trans_novel.llm.errors import JSONParseError


class AlignmentError(WorkflowProtocolError):
    """句段对齐失败：协议错误的子类，reason 是稳定标识，message 是可读的中文说明。"""


def _slot_prompt(segments: list[Segment] | None) -> str:
    if not segments or not any(segment.epub_state is not None for segment in segments):
        return ""
    if any(segment.epub_state is None for segment in segments):
        raise ValueError("EPUB slot transport cannot mix EPUB and flat segments")
    expected = [
        {"segment": index, "slots": slot_transport(segment, target=False)}
        for index, segment in enumerate(segments)
    ]
    return (
        "\n【EPUB 槽位协议】\n"
        + json.dumps(expected, ensure_ascii=False, separators=(",", ":"))
        + '\n每项必须输出 {"slots":[{"id":"原始槽位 ID","core":"译文核心"}]}；'
        "槽位数量、顺序和 ID 必须完全一致；不得输出扁平字符串。\n"
    )


class Translator(Agent):
    def _call_batch(
        self,
        sources: list[str],
        glossary_terms: list[GlossaryTerm],
        style: str,
        context: str,
        book_synopsis: str = "",
        chapter_digest: str = "",
        *,
        agent: str,
        operation: str = "translate.batch",
        segments: list[Segment] | None = None,
    ) -> list:
        n = len(sources)
        system = prompts.render(
            "translator_system",
            src=self.src,
            tgt=self.tgt,
            n=n,
            lang_guidance=langprofile.translate_guidance(self.src, self.config.honorific_strategy),
        )
        user = prompts.render(
            "translator_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "（无）",
            book_synopsis=book_synopsis or "（无）",
            glossary=prompts.render_glossary(glossary_terms),
            chapter_digest=chapter_digest or "（无）",
            context=context or "（无）",
            n=n,
            n_minus_1=n - 1,
            numbered_source=prompts.numbered(sources),
        ) + _slot_prompt(segments)
        items = self._ask_json(system, user, key="translations", agent=agent, operation=operation)
        if not isinstance(items, list) or len(items) != n:
            raise AlignmentError(
                "translation_count_mismatch",
                f"译文数量不匹配：期望 {n} 段，实际 {len(items) if isinstance(items, list) else '非数组'}",
            )
        if segments and any(segment.epub_state is not None for segment in segments):
            if any(segment.epub_state is None for segment in segments):
                raise ValueError("EPUB slot transport cannot mix EPUB and flat segments")
            try:
                return [
                    validate_slot_transport(segment, item["slots"])
                    if isinstance(item, dict) and isinstance(item.get("slots"), list)
                    else (_ for _ in ()).throw(ValueError("EPUB translation item is not an object"))
                    for segment, item in zip(segments, items, strict=True)
                ]
            except (ValueError, TypeError) as error:
                raise AlignmentError(
                    "translation_slot_mismatch", "EPUB 槽位返回结果不匹配"
                ) from error
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise AlignmentError("translation_item_invalid", "模型返回了空译文或非字符串译文")
        return items

    def _translate_one(
        self,
        source,
        glossary_terms,
        style,
        context,
        book_synopsis,
        chapter_digest,
        *,
        agent: str,
        operation: str = "translate.batch",
        segment: Segment | None = None,
    ) -> object:
        out = self._call_batch(
            [source],
            glossary_terms,
            style,
            context,
            book_synopsis,
            chapter_digest,
            agent=agent,
            operation=operation,
            segments=[segment] if segment else None,
        )
        return out[0]

    def retranslate_with_feedback(
        self,
        source: str,
        *,
        feedback: str,
        operation: str,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        context_before: str = "",
        context_after: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        segment: Segment | None = None,
    ) -> object:
        system = prompts.render(
            "translator_system",
            src=self.src,
            tgt=self.tgt,
            n=1,
            lang_guidance=langprofile.translate_guidance(self.src, self.config.honorific_strategy),
        )
        user = prompts.render(
            "translator_fix_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "（无）",
            book_synopsis=book_synopsis or "（无）",
            glossary=prompts.render_glossary(glossary_terms or []),
            chapter_digest=chapter_digest or "（无）",
            context_before=context_before or "（无）",
            context_after=context_after or "（无）",
            feedback=feedback or "（无）",
            source=source,
        ) + _slot_prompt([segment] if segment else None)
        items = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if not isinstance(items, list) or not items:
            return ""
        if segment is not None and segment.epub_state is not None:
            if len(items) != 1 or not isinstance(items[0], dict):
                raise AlignmentError("translation_slot_mismatch", "EPUB 槽位返回结果不匹配")
            return validate_slot_transport(segment, items[0].get("slots"))
        return items[0].strip() if isinstance(items[0], str) and items[0].strip() else ""

    def retranslate_batch_with_feedback(
        self,
        items: list[tuple[int, str, str]],
        batch_targets: list[str],
        *,
        operation: str,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        segments: list[Segment] | None = None,
    ) -> list:
        if not items:
            return []
        n = len(items)
        system = prompts.render(
            "translator_system",
            src=self.src,
            tgt=self.tgt,
            n=n,
            lang_guidance=langprofile.translate_guidance(self.src, self.config.honorific_strategy),
        )
        user = prompts.render(
            "translator_fix_multi_user",
            src=self.src,
            tgt=self.tgt,
            style=style or "（无）",
            book_synopsis=book_synopsis or "（无）",
            glossary=prompts.render_glossary(glossary_terms or []),
            chapter_digest=chapter_digest or "（无）",
            batch_targets=prompts.numbered(batch_targets),
            n=n,
            items=prompts.numbered_feedback(items),
        ) + _slot_prompt([segments[idx] for idx, _source, _feedback in items] if segments else None)
        out = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if not isinstance(out, list) or len(out) != n:
            return []
        if segments and any(segment.epub_state is not None for segment in segments):
            try:
                return [
                    validate_slot_transport(segments[idx], item["slots"])
                    if isinstance(item, dict) and isinstance(item.get("slots"), list)
                    else (_ for _ in ()).throw(ValueError("EPUB translation item is not an object"))
                    for item, (idx, _source, _feedback) in zip(out, items, strict=True)
                ]
            except (ValueError, TypeError) as error:
                raise AlignmentError(
                    "translation_slot_mismatch", "EPUB 槽位返回结果不匹配"
                ) from error
        return (
            [x.strip() for x in out if isinstance(x, str) and x.strip()]
            if all(isinstance(x, str) and x.strip() for x in out)
            else []
        )

    def translate_batch(
        self,
        sources: list[str],
        *,
        agent: str,
        operation: str = "translate.batch",
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        context: str = "",
        book_synopsis: str = "",
        chapter_digest: str = "",
        segments: list[Segment] | None = None,
    ) -> list:
        glossary_terms = glossary_terms or []
        if not sources:
            return []
        attempts = self.config.pipeline.align_retry_limit + 1
        for _ in range(attempts):
            try:
                return self._call_batch(
                    sources,
                    glossary_terms,
                    style,
                    context,
                    book_synopsis,
                    chapter_digest,
                    agent=agent,
                    operation=operation,
                    segments=segments,
                )
            except (AlignmentError, JSONParseError):
                pass
        targets: list = []
        for index, source in enumerate(sources):
            try:
                targets.append(
                    self._translate_one(
                        source,
                        glossary_terms,
                        style,
                        context,
                        book_synopsis,
                        chapter_digest,
                        agent=agent,
                        operation=operation,
                        segment=segments[index] if segments else None,
                    )
                )
            except (AlignmentError, JSONParseError) as error:
                raise AlignmentError(
                    "translation_segment_fallback_failed", f"索引为 {index} 的段落在兜底翻译时失败"
                ) from error
        return targets
