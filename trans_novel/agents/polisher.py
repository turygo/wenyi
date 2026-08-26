"""润色 Agent。

在直译稿上做中文文学性二次加工：不增删信息、保持段数不变。
注入源文逐段对照，让润色在提升表达的同时自查忠实度，避免因精简而丢失修饰语/限定语。
对齐失败（段数不符）时保守地返回原译文，绝不因润色而引入漏译。
"""

from __future__ import annotations

import json

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent, WorkflowProtocolError
from trans_novel.glossary.store import GlossaryTerm
from trans_novel.ingest.models import Segment, validate_slot_transport


class Polisher(Agent):
    def polish(
        self,
        targets: list,
        sources: list[str],
        *,
        glossary_terms: list[GlossaryTerm] | None = None,
        style: str = "",
        strict: bool = False,
        segments: list[Segment] | None = None,
    ) -> list:
        if not targets:
            return []
        n = len(targets)
        epub = bool(segments and any(segment.epub_state is not None for segment in segments))
        if epub and (
            not segments or len(segments) != n or any(s.epub_state is None for s in segments)
        ):
            raise ValueError("EPUB polish transport requires one slot contract per segment")
        displayed = [
            json.dumps(target, ensure_ascii=False, separators=(",", ":")) if epub else target
            for target in targets
        ]
        system = prompts.render("polisher_system", src=self.src, tgt=self.tgt, n=n)
        user = prompts.render(
            "polisher_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(glossary_terms or []),
            style=style or "（无）",
            n=n,
            numbered_source=prompts.numbered(sources),
            numbered_target=prompts.numbered(displayed),
        )
        if epub:
            user += (
                '\n【EPUB 槽位协议】每项必须输出 {"slots":[{"id":"原始槽位 ID",'
                '"core":"润色核心"}]}；槽位数量、顺序和 ID 必须完全一致；不得输出扁平字符串。\n'
            )
        items = self._ask_json(
            system,
            user,
            key="polished",
            default=None,
            agent="editor",
            operation="polish.batch",
            strict=strict,
        )

        def decode(values: object) -> list:
            if not isinstance(values, list) or len(values) != n:
                return []
            if not epub:
                return [str(x) for x in values]
            try:
                return [
                    validate_slot_transport(segment, item["slots"])
                    if isinstance(item, dict) and isinstance(item.get("slots"), list)
                    else (_ for _ in ()).throw(ValueError("EPUB polish item is not an object"))
                    for segment, item in zip(segments, values, strict=True)
                ]
            except (ValueError, TypeError):
                return []

        decoded = decode(items)
        if decoded:
            return decoded
        if not strict:
            return list(targets)
        retry_user = f"{user}\n\n" + (
            f"Your previous response violated the output contract: `polished` must contain "
            f"exactly {n} strings. Return the complete JSON object again."
            if not epub
            else f"Your previous response violated the output contract: return exactly {n} "
            "ordered records with every original slot ID."
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
        recovered: list = []
        for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
            single_system = prompts.render("polisher_system", src=self.src, tgt=self.tgt, n=1)
            single_user = prompts.render(
                "polisher_user",
                src=self.src,
                tgt=self.tgt,
                glossary=prompts.render_glossary(glossary_terms or []),
                style=style or "（无）",
                n=1,
                numbered_source=prompts.numbered([source]),
                numbered_target=prompts.numbered(
                    [
                        json.dumps(target, ensure_ascii=False, separators=(",", ":"))
                        if epub
                        else target
                    ]
                ),
            )
            if epub:
                single_user += "\n【EPUB 槽位协议】仅输出带原始槽位 ID 的 slots 记录。\n"
            single_items = self._ask_json(
                single_system,
                single_user,
                key="polished",
                default=None,
                agent="editor",
                operation="polish.segment",
                strict=True,
            )
            if not isinstance(single_items, list) or len(single_items) != 1:
                raise WorkflowProtocolError("polish_count_mismatch")
            try:
                one = (
                    [validate_slot_transport(segments[index], single_items[0]["slots"])]
                    if epub
                    else [str(single_items[0])]
                )
            except (KeyError, TypeError, ValueError) as error:
                raise WorkflowProtocolError("polish_slot_mismatch") from error
            recovered.append(one[0])
        return recovered
