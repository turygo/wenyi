"""翻译 Agent。

balanced/quality 每次调用只发送一个待译段，并严格校验单值 JSON，从调用边界保证
源段与译文一一对应。economy 保留批量翻译；批量协议失败时仍逐段兜底。

模型路由按功能 Agent 选择：正文走 translator（operation=translate.batch）；附属章
旁路走 light-translator（operation=translate.back_matter，由调用方显式传 agent）。
operation 只作用量/调试归因，不参与路由。
"""

from __future__ import annotations

import json

from trans_novel.agents import langprofile, prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.glossary.store import GlossaryTerm
from trans_novel.llm.errors import JSONParseError
from trans_novel.pipeline import checks


class AlignmentError(WorkflowProtocolError):
    """句段对齐失败：协议错误的子类，reason 是稳定标识，message 是可读的中文说明。"""


class Translator(Agent):
    def _call_batch(
        self,
        sources: list[str],
        glossary_terms: list[GlossaryTerm],
        style: str,
        context: str,
        *,
        agent: str,
        operation: str = "translate.batch",
    ) -> list[str]:
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
            glossary=prompts.render_glossary(glossary_terms),
            context=context or "（无）",
            n=n,
            n_minus_1=n - 1,
            numbered_source=prompts.numbered(sources),
        )
        items = self._ask_json(system, user, key="translations", agent=agent, operation=operation)
        if not isinstance(items, list) or len(items) != n:
            raise AlignmentError(
                "translation_count_mismatch",
                f"译文数量不匹配：期望 {n} 段，实际 {len(items) if isinstance(items, list) else '非数组'}",
            )
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise AlignmentError("translation_item_invalid", "模型返回了空译文或非字符串译文")
        return items

    def _translate_one(
        self,
        source,
        glossary_terms,
        style,
        context,
        *,
        agent: str,
        operation: str = "translate.batch",
    ) -> str:
        system = prompts.render("translator_single_system", src=self.src)
        user = prompts.render(
            "translator_single_user",
            src=self.src,
            source=source,
            glossary=prompts.render_glossary(glossary_terms),
            style=style or "（无）",
            context=context or "（无）",
        )
        target = self._ask_text(system, user, agent=agent, operation=operation, strict=True)
        if not target or len(target) > max(256, len(source) * 4):
            raise AlignmentError("translation_item_invalid", "模型返回了空译文或异常长译文")
        return target

    def _translate_one_strict(
        self,
        source: str,
        glossary_terms: list[GlossaryTerm],
        style: str,
        context: str,
        *,
        agent: str,
        operation: str,
    ) -> str:
        system = prompts.render("translator_strict_system", src=self.src)
        user = prompts.render(
            "translator_strict_user",
            src=self.src,
            source=source,
            glossary=prompts.render_glossary(glossary_terms),
            style=style or "（无）",
            context=context or "（无）",
        )
        raw = self.client.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            json_mode=True,
            stage=type(self).__name__,
            agent=agent,
            operation=operation,
        )
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise JSONParseError("单段译文不是严格 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"translation"}:
            raise AlignmentError(
                "translation_schema_invalid",
                "单段译文必须是仅含 translation 键的 JSON 对象",
            )
        target = payload["translation"]
        if (
            not isinstance(target, str)
            or not target.strip()
            or len(target.strip()) > max(256, len(source) * 4)
        ):
            raise AlignmentError(
                "translation_item_invalid",
                "模型返回了空译文、非字符串译文或异常长译文",
            )
        return target.strip()

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
    ) -> str:
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
            glossary=prompts.render_glossary(glossary_terms or []),
            context_before=context_before or "（无）",
            context_after=context_after or "（无）",
            feedback=feedback or "（无）",
            source=source,
        )
        items = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if not isinstance(items, list) or len(items) != 1:
            return ""
        return items[0].strip() if isinstance(items[0], str) and items[0].strip() else ""

    def retranslate_batch_with_feedback(
        self,
        items: list[tuple[int, str, str]],
        batch_targets: list[str],
        *,
        operation: str,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
    ) -> list[str]:
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
            glossary=prompts.render_glossary(glossary_terms or []),
            batch_targets=prompts.numbered(batch_targets),
            n=n,
            items=prompts.numbered_feedback(items),
        )
        out = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if not isinstance(out, list) or len(out) != n:
            return []
        return (
            [x.strip() for x in out] if all(isinstance(x, str) and x.strip() for x in out) else []
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
    ) -> list[str]:
        glossary_terms = glossary_terms or []
        if not sources:
            return []
        translated_indices = [
            index
            for index, source in enumerate(sources)
            if any(character.isalpha() for character in source)
            and not checks.is_machine_literal(source)
        ]
        targets = list(sources)
        if not translated_indices:
            return targets

        attempts = self.config.pipeline.align_retry_limit + 1
        if self.config.pipeline.single_segment_translation:
            for index in translated_indices:
                for _ in range(attempts):
                    try:
                        targets[index] = self._translate_one_strict(
                            sources[index],
                            glossary_terms,
                            style,
                            context,
                            agent=agent,
                            operation=operation,
                        )
                        break
                    except (AlignmentError, JSONParseError):
                        pass
                else:
                    raise AlignmentError(
                        "translation_segment_contract_failed",
                        f"索引为 {index} 的段落未能返回合法单段译文",
                    )
            return targets

        translated_sources = [sources[index] for index in translated_indices]
        for _ in range(attempts):
            try:
                translated = self._call_batch(
                    translated_sources,
                    glossary_terms,
                    style,
                    context,
                    agent=agent,
                    operation=operation,
                )
                for index, target in zip(translated_indices, translated, strict=True):
                    targets[index] = target
                return targets
            except (AlignmentError, JSONParseError):
                pass
        for index in translated_indices:
            for attempt in range(attempts):
                try:
                    targets[index] = self._translate_one(
                        sources[index],
                        glossary_terms if attempt == 0 else [],
                        style if attempt == 0 else "",
                        context if attempt == 0 else "",
                        agent=agent,
                        operation=operation,
                    )
                    break
                except (AlignmentError, JSONParseError):
                    pass
            else:
                raise AlignmentError(
                    "translation_segment_fallback_failed",
                    f"索引为 {index} 的段落在兜底翻译时失败",
                )
        return targets
