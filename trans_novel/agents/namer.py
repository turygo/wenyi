"""全书一次性定名 Agent（强档）。

源文侧挖出候选（见 glossary/miner.py）后，一次性裁定全书统一译名——取代译后
逐批抽取，避免把首译直译固化成铁律、避免普通名词/一次性修辞污染术语库。
有权丢弃不值得入表的候选（普通名词短语、亲属称谓、引文/文献标题、一次性习语）。
产物写入术语库后，翻译期只读（见 orchestrator._build_understanding）。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from ..glossary.miner import Candidate
from ..glossary.store import TYPE_PERSON, GlossaryTerm
from . import prompts
from .base import Agent

# 候选分组的字符预算：组内一次强档调用，量级参考 agents/synopsis.py 的 _REDUCE_BUDGET。
_GROUP_CHAR_BUDGET = 6000


def _render_candidates(candidates: list[Candidate]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        line = f"[{i}] {c.surface}（出现{c.count}次）"
        if c.contexts:
            line += " 例：" + " / ".join(c.contexts)
        lines.append(line)
    return "\n".join(lines)


class CastNamer(Agent):
    def name_terms(
        self,
        candidates: list[Candidate],
        analysis_brief: str,
        digests: list[str],
        existing: list[GlossaryTerm] | None = None,
        *,
        concurrency: int = 1,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[GlossaryTerm]:
        """给候选定唯一中文译名 + type/gender/note；候选按字符预算分组，每组一次强档
        调用，组间独立无 reduce（候选已按 surface 去重，组间不会互相冲突）。

        existing：已入库条目（analyzer.seed_glossary 的样章种入等），渲染进 prompt
        要求模型沿用已有译法、不重复输出。

        各组相互独立（同一 existing 快照、无跨组 reduce）→ 按 concurrency 并行，
        输出按输入组序合并，与串行完全一致。on_progress(done, total) 按完成组数回调
        （主线程触发，供进度条使用）。任一组异常整体冒泡，交 orchestrator 捕获后
        放弃本次落 term_mining_done（下次续跑重试），绝不静默吞成空定名。
        """
        if not candidates:
            return []
        digest_text = "\n".join(d for d in digests if d and d.strip())
        if len(digest_text) > _GROUP_CHAR_BUDGET:
            digest_text = digest_text[:_GROUP_CHAR_BUDGET]
        glossary_text = prompts.render_glossary(existing or [])
        groups = self._group(candidates, _GROUP_CHAR_BUDGET)

        def name_one(group: list[Candidate]) -> list[GlossaryTerm]:
            return self._name_group(group, analysis_brief, digest_text, glossary_text)

        if on_progress:
            on_progress(0, len(groups))
        results: dict[int, list[GlossaryTerm]] = {}
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futs = {ex.submit(name_one, g): i for i, g in enumerate(groups)}
            for done, fut in enumerate(as_completed(futs), 1):
                results[futs[fut]] = fut.result()  # 异常冒泡；池随 with 收尾
                if on_progress:
                    on_progress(done, len(groups))
        return [t for i in range(len(groups)) for t in results[i]]

    # ── 内部 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _group(candidates: list[Candidate], budget: int) -> list[list[Candidate]]:
        """按字符预算贪心打包候选（含样例上下文长度），避免单组 prompt 超长。"""
        groups: list[list[Candidate]] = []
        cur: list[Candidate] = []
        size = 0
        for c in candidates:
            item_len = len(c.surface) + sum(len(x) for x in c.contexts) + 20
            if cur and size + item_len > budget:
                groups.append(cur)
                cur, size = [], 0
            cur.append(c)
            size += item_len
        if cur:
            groups.append(cur)
        return groups

    def _name_group(
        self, group: list[Candidate], analysis_brief: str, digest_text: str, glossary_text: str
    ) -> list[GlossaryTerm]:
        system = prompts.render("cast_naming_system", src=self.src, tgt=self.tgt)
        user = prompts.render(
            "cast_naming_user",
            src=self.src,
            tgt=self.tgt,
            glossary=glossary_text,
            brief=analysis_brief or "（无）",
            digests=digest_text or "（无）",
            candidates=_render_candidates(group),
        )
        # 不设 default：一次强档失败若被兜成空列表，term_mining_done 会静默永久落盘，
        # 续跑再也不重试——异常整体冒泡，交由 orchestrator 捕获并放弃本次落标记。
        raw = self._ask_json(
            system, user, tier="strong", key="terms", operation="prescan.name_terms"
        )
        out: list[GlossaryTerm] = []
        for d in self.dict_items(raw):
            # 裁定标准（见 prompts.CAST_NAMING_SYSTEM）：宁缺勿滥。实现选择省略式——
            # 模型对不值得入表的候选直接不输出（比显式 drop 标记更简单）；同时容忍
            # 模型误加 drop:true 兜底，两种表达都跳过。
            if d.get("drop"):
                continue
            source = str(d.get("source", "")).strip()
            target = str(d.get("target", "")).strip()
            if not source or not target:
                continue
            term_type = d.get("type", "术语")
            out.append(
                GlossaryTerm(
                    source=source,
                    target=target,
                    reading=str(d.get("reading", "")).strip(),
                    type=term_type,
                    gender=d.get("gender", "") if d.get("gender") not in ("未知", None) else "",
                    note=d.get("note", ""),
                    confidence="high",
                    locked=term_type == TYPE_PERSON,
                    first_chapter=0,
                )
            )
        return out
