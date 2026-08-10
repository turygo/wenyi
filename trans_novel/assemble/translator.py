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

from ..agents import langprofile, prompts
from ..agents.base import Agent
from ..glossary.store import GlossaryTerm


class AlignmentError(Exception):
    pass


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
            book_synopsis=book_synopsis or "（无）",
            glossary=prompts.render_glossary(glossary_terms),
            chapter_digest=chapter_digest or "（无）",
            context=context or "（无）",
            n=n,
            n_minus_1=n - 1,
            numbered_source=prompts.numbered(sources),
        )
        # 不传 default：调用失败照常抛出，由 translate_batch 的重试/兜底逻辑处理
        items = self._ask_json(system, user, key="translations", agent=agent, operation=operation)
        if not isinstance(items, list):
            raise AlignmentError("模型未返回译文数组")
        if len(items) != n:
            raise AlignmentError(f"译文数量不匹配：期望 {n} 段，实际 {len(items)} 段")
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise AlignmentError("模型返回了空译文或非字符串译文")
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
    ) -> str:
        out = self._call_batch(
            [source],
            glossary_terms,
            style,
            context,
            book_synopsis,
            chapter_digest,
            agent=agent,
            operation=operation,
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
    ) -> str:
        """带审校意见定向重译单段（lint 层单段回退/章末 autofix 用）。失败返回空串，由调用方决定弃用。

        复用 translator_system（与主翻译共享稳定前缀，命中缓存）；
        user 用 translator_fix_user：前缀块与主翻译一致，上下文换成前文+后文译文，附审校意见。
        operation 由调用方按来源区分 translate.lint_fix / translate.review_fix，
        采纳/拒绝在各自调用点记录，本方法不产生 outcome。
        """
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
        )
        items = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if isinstance(items, list) and items:
            return str(items[0]).strip()
        return ""

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
    ) -> list[str]:
        """根据审校意见批量重译同一批次中的多个待修复段落（合并为一次调用，可减少 N-1 次
        请求）。调用失败、返回值不是数组或数组长度不等于 N 时返回 []，由调用方回退到逐段
        调用；各段是否采纳仍由调用方独立判断，本方法不记录 outcome。

        items：（批内段号，源文，审校意见）三元组，长度为 N（N > 1，由调用方保证）。批内段号
        必须与 batch_targets 的下标口径一致，模型才能据此在整批译文里定位原译及前后文。
        batch_targets：按批内段号排列的整批当前译文，包含待修复段落的旧译文；按原顺序编号后注入。
        """
        n = len(items)
        if n == 0:
            return []
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
        )
        out = self._ask_json(
            system, user, key="translations", default=None, agent="translator", operation=operation
        )
        if (
            isinstance(out, list)
            and len(out) == n
            and all(isinstance(x, str) and x.strip() for x in out)
        ):
            return [x.strip() for x in out]
        return []

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
    ) -> list[str]:
        """翻译一批源段，返回与之等长的译文列表。

        agent 选择功能 Agent 路由：正文 translator；附属章旁路 light-translator。
        operation 只作用量/调试归因（正文 translate.batch；附属章 translate.back_matter），
        不参与路由。
        """
        glossary_terms = glossary_terms or []
        n = len(sources)
        if n == 0:
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
                )
            except Exception:
                pass

        # 兜底：逐段翻译。任一段仍失败时显式中断，保留已落盘
        # 批次供续跑；不能用空字符串占位，否则章节会被错误标记为已完成。
        targets: list[str] = []
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
                    )
                )
            except Exception as error:
                raise AlignmentError(f"索引为 {index} 的段落在兜底翻译时失败") from error
        return targets
