"""跨章一致性 QA。

汇总术语表 + 各章译文摘要，让模型扫描术语译法漂移、代词性别不一致、语气文体漂移。
摘要只取每章首尾若干段并截断，控制 token。
"""

from __future__ import annotations

from typing import Any

from trans_novel.agents import prompts
from trans_novel.agents.base import Agent
from trans_novel.glossary.store import GlossaryStore
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore


class ConsistencyChecker(Agent):
    def _chapter_digests(self, store: RunStore, max_chars_each: int = 600) -> str:
        m = store.load_manifest()
        parts: list[str] = []
        for c in m["chapters"]:
            if store.load_progress(c["index"]).status != STATUS_DONE:
                continue
            ch = store.load_chapter(c["index"])
            targets = [s.target or "" for s in ch.text_segments]
            head = targets[:3]
            tail = targets[-2:] if len(targets) > 3 else []
            snippet = "……".join([t for t in head + tail if t])[:max_chars_each]
            parts.append(f"[第{c['index']}章 {c['title']}]\n{snippet}")
        return "\n\n".join(parts)

    def check(
        self, store: RunStore, glossary: GlossaryStore, *, strict: bool = False
    ) -> list[dict[str, Any]]:
        """跨章一致性扫描：汇总术语表与各章译文摘要，让模型报出译法漂移等问题。

        strict=True（workflow 必需节点用）：provider 失败原样抛出；空摘要等
        确定性的“无事可扫”仍返回空列表（成功零发现，绝非失败）。
        """
        digests = self._chapter_digests(store)
        if not digests.strip():
            return []
        system = prompts.render("consistency_system", src=self.src, tgt=self.tgt)
        user = (
            "【专有名词对照表】\n"
            + prompts.render_glossary(glossary.all_terms())
            + "\n\n【各章译文摘要】\n"
            + digests
            + '\n\n请输出 JSON：{"issues":[...]}。'
        )
        return self.dict_items(
            self._ask_json(
                system,
                user,
                key="issues",
                default=[],
                agent="reviewer",
                operation="consistency.check",
                strict=strict,
                items_are_dicts=strict,
            )
        )
