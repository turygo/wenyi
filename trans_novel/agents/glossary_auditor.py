"""LLM decision agent for glossary audit candidates."""

from __future__ import annotations

from typing import Any

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent


class GlossaryAuditor(Agent):
    def decide(self, candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        lines = []
        for candidate in candidates.values():
            all_variants = [candidate["current"]] + [
                variant for variant in candidate["variants"] if variant != candidate["current"]
            ]
            lines.append(
                f"- {candidate['source']}（{candidate['type'] or '?'}）: 现有译法/变体 = "
                f"{', '.join(all_variants)}"
            )
        user = (
            "下列原文词在术语表或正文里出现了多种译法/形近变体，请为每个裁定唯一规范译法：\n"
            + "\n".join(lines)
            + '\n\n输出 JSON：{"unifications":[{"source":"...","canonical":"...","variants":["..."],"reason":"..."}]}'
        )
        system = prompts.render("glossary_audit_system", src=self.src, tgt=self.tgt)
        unifications = self._ask_json(
            system,
            user,
            key="unifications",
            default=[],
            agent="analyst",
            operation="glossary.audit",
        )
        result: list[dict[str, Any]] = []
        for unification in self.dict_items(unifications):
            if not unification.get("source") or not unification.get("canonical"):
                continue
            allowed = set(candidates.get(str(unification["source"]), {}).get("variants", []))
            unification = dict(unification)
            unification["variants"] = [
                variant
                for variant in unification.get("variants", [])
                if str(variant).strip() in allowed
            ]
            result.append(unification)
        return result
