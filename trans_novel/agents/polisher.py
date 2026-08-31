"""润色 Agent。

逐段对照源文与直译稿做中文文学性二次加工，避免批量响应在段落之间串位。
非严格模式下，单段响应无效时只保留该段原译文。
"""

from __future__ import annotations

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.glossary.store import GlossaryTerm
from trans_novel.pipeline import checks


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
        if len(targets) != len(sources):
            raise ValueError("polish source/target count mismatch")
        polished = []
        for source, target in zip(sources, targets, strict=True):
            if checks.is_machine_literal(source):
                polished.append(target)
                continue
            system = prompts.render("polisher_system", src=self.src, tgt=self.tgt, n=1)
            user = prompts.render(
                "polisher_user",
                src=self.src,
                tgt=self.tgt,
                glossary=prompts.render_glossary(glossary_terms or []),
                style=style or "（无）",
                n=1,
                numbered_source=prompts.numbered([source]),
                numbered_target=prompts.numbered([target]),
            )
            values = self._ask_json(
                system,
                user,
                key="polished",
                default=None,
                agent="editor",
                operation="polish.segment",
                strict=strict,
            )
            one = (
                values[0].strip()
                if isinstance(values, list)
                and len(values) == 1
                and isinstance(values[0], str)
                and values[0].strip()
                else ""
            )
            if one:
                polished.append(one)
            elif strict:
                raise WorkflowProtocolError("polish_item_invalid")
            else:
                polished.append(target)
        return polished
