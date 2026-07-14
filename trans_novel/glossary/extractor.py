"""术语抽取 Agent（廉价档）+ 入库（含冲突裁决）。

每翻完一章，从"原文 + 译文"里抽取应进表的专有名词，
依据实际译法入库；冲突裁决由 GlossaryStore.upsert_term 完成。

注入模型的 existing 术语表按本次原文命中裁剪（未命中的历史词条不随全表增长白烧
token），遗漏的重复提案交由 upsert_term 的幂等裁决兜底。
"""

from __future__ import annotations

from ..agents import prompts
from ..agents.base import Agent
from .store import TYPE_PERSON, GlossaryStore, GlossaryTerm


def _text(value: object, default: str = "") -> str:
    """把模型返回的标量字段规整为字符串。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
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
            system, user, tier="fast", key="terms", default=[], operation="glossary.extract"
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

    def store_terms(
        self, store: GlossaryStore, terms: list[GlossaryTerm], chapter: int
    ) -> tuple[dict[str, int], list[GlossaryTerm]]:
        summary = {"inserted": 0, "updated": 0, "conflict": 0, "unchanged": 0}
        changed: list[GlossaryTerm] = []
        for t in terms:
            t.first_chapter = chapter
            result = store.upsert_term(t, chapter=chapter)
            summary[result] = summary.get(result, 0) + 1
            if result in ("inserted", "updated"):
                changed.append(t)
        return summary, changed

    def extract_and_store(
        self,
        store: GlossaryStore,
        source_text: str,
        target_text: str,
        chapter: int,
    ) -> tuple[dict[str, int], list[GlossaryTerm]]:
        """抽取并入库，返回 (入库汇总, 实际 inserted/updated 的词条)。

        changed 供调用方做章级快照的条件刷新（命中剩余源文才重建，保前缀缓存）。
        """
        existing = store.all_terms()
        hit = {t.source for t in GlossaryStore.terms_in(existing, source_text)}
        existing = [t for t in existing if t.source in hit or (t.type == TYPE_PERSON and t.locked)]
        terms = self.extract(source_text, target_text, existing)
        return self.store_terms(store, terms, chapter)
