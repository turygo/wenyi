"""术语抽取 Agent。"""

from __future__ import annotations

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent
from trans_novel.glossary.store import GlossaryTerm


def _text(value: object, default: str = "") -> str:
    """把模型返回的标量字段规整为字符串。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return default


class GlossaryExtractor(Agent):
    def extract(
        self, source_text: str, target_text: str, existing: list[GlossaryTerm]
    ) -> list[GlossaryTerm]:
        system = prompts.render("glossary_extractor_system", src=self.src, tgt=self.tgt)
        user = prompts.render(
            "glossary_extractor_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(existing),
            source=source_text,
            target=target_text,
        )
        raw = self._ask_json(
            system, user, key="terms", default=[], agent="preparer", operation="glossary.extract"
        )
        terms: list[GlossaryTerm] = []
        for d in self.dict_items(raw):
            source = _text(d.get("source"))
            target = _text(d.get("target"))
            if not source or not target:
                continue
            raw_aliases = d.get("aliases")
            aliases = raw_aliases if isinstance(raw_aliases, list) else []
            gender = _text(d.get("gender"))
            terms.append(
                GlossaryTerm(
                    source=source,
                    target=target,
                    reading=_text(d.get("reading")),
                    type=_text(d.get("type"), "术语"),
                    gender="" if gender == "未知" else gender,
                    aliases=[alias for a in aliases if (alias := _text(a))],
                    note=_text(d.get("note")),
                    confidence="medium",
                )
            )
        return terms
