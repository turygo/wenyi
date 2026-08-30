"""润色 Agent。

在直译稿上做中文文学性二次加工：不增删信息、保持段数不变。
注入源文逐段对照，让润色在提升表达的同时自查忠实度，避免因精简而丢失修饰语/限定语。
对齐失败（段数不符）时保守地返回原译文，绝不因润色而引入漏译。
"""

from __future__ import annotations

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.glossary.store import GlossaryTerm


class Polisher(Agent):
    def polish(
        self,
        targets: list[str],
        sources: list[str],
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        strict: bool = False,
    ) -> list[str]:
        if not targets:
            return []
        n = len(targets)
        system = prompts.render("polisher_system", src=self.src, tgt=self.tgt, n=n)
        user = prompts.render(
            "polisher_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(glossary_terms or []),
            style=style or "（无）",
            n=n,
            numbered_source=prompts.numbered(sources),
            numbered_target=prompts.numbered(targets),
        )

        def decode(values: object) -> list[str]:
            if (
                not isinstance(values, list)
                or len(values) != n
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                return []
            return [value.strip() for value in values]

        decoded = decode(
            self._ask_json(
                system,
                user,
                key="polished",
                default=None,
                agent="editor",
                operation="polish.batch",
                strict=strict,
            )
        )
        if decoded:
            return decoded
        if not strict:
            return list(targets)
        retry_user = (
            f"{user}\n\nYour previous response violated the output contract: `polished` must "
            f"contain exactly {n} strings, all non-empty. Return the complete JSON object again."
        )
        decoded = decode(
            self._ask_json(
                system,
                retry_user,
                key="polished",
                default=None,
                agent="editor",
                operation="polish.batch",
                strict=True,
            )
        )
        if decoded:
            return decoded
        recovered = []
        for source, target in zip(sources, targets, strict=True):
            single_user = prompts.render(
                "polisher_user",
                src=self.src,
                tgt=self.tgt,
                glossary=prompts.render_glossary(glossary_terms or []),
                style=style or "（无）",
                n=1,
                numbered_source=prompts.numbered([source]),
                numbered_target=prompts.numbered([target]),
            )
            single_items = self._ask_json(
                prompts.render("polisher_system", src=self.src, tgt=self.tgt, n=1),
                single_user,
                key="polished",
                default=None,
                agent="editor",
                operation="polish.segment",
                strict=True,
            )
            one = (
                decode(single_items)
                if n == 1
                else (
                    [single_items[0].strip()]
                    if isinstance(single_items, list)
                    and len(single_items) == 1
                    and isinstance(single_items[0], str)
                    and single_items[0].strip()
                    else []
                )
            )
            if not one:
                raise WorkflowProtocolError("polish_count_mismatch")
            recovered.append(one[0])
        return recovered
