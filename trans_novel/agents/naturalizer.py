"""去翻译腔闭环：单语审读 → 单语改写 → 三道关卡 → 带事件日志写回。

经验事实（真书实验验证）：LLM 绝对检测轻度翻译腔召回低但精度高（只标有把握的，近似抽检）；
成对判断（正反两序）可靠。因此检测只决定预算花在哪，**写回安全完全由三道关卡保证**：
关卡①确定性 lint（不得引入原译没有的 issue 类型，零成本先跑）→
关卡③双语忠实度判断（对照源文，防止改写偷改信息，cheap 档）→
关卡②成对判断（正反两序皆胜才采纳，两次调用最贵放最后）。

作为主流水线章级环节接入（见 orchestrator._translate_chapter，config: pipeline.naturalize），
也保留为独立的 `tools naturalize` 命令供单独跑批/补跑。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..glossary.store import GlossaryStore, GlossaryTerm
from ..ingest.models import KIND_TEXT, Chapter, Segment
from ..pipeline import lint
from ..pipeline.backmatter import is_back_matter
from ..pipeline.runstore import RunStore
from ..postprocess.punct import normalize_zh
from . import prompts
from .base import Agent

_SCREEN_BATCH_SIZE = 20
_HAN_RATIO_MIN = 0.2  # 汉字占比阈值：≤ 视为非中文段，跳过
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _han_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_HAN_RE.findall(text)) / len(text)


def candidate_segments(chapter: Chapter) -> list[Segment]:
    """挑出可安全改写的候选段（确定性、安全优先）。

    规则：kind==text 且非 cont；target 非空；汉字占比 > 0.2；
    若紧跟的下一段是 cont 续段（本段落被切分跨多 segment），整段跳过——
    写回拆分复杂，YAGNI。
    """
    segs = chapter.segments
    out: list[Segment] = []
    for i, seg in enumerate(segs):
        if seg.kind != KIND_TEXT or seg.cont:
            continue
        if not seg.target or not seg.target.strip():
            continue
        if i + 1 < len(segs) and segs[i + 1].cont:
            continue
        if _han_ratio(seg.target) <= _HAN_RATIO_MIN:
            continue
        out.append(seg)
    return out


class Naturalizer(Agent):
    """审读/改写/成对判断三合一 agent。"""

    def screen(self, texts: list[str]) -> list[dict]:
        """批量审读中文段，返回 [{index,quote,reason,rewrite}]（单语，宁缺勿滥）。"""
        if not texts:
            return []
        system = prompts.render("naturalize_screen_system")
        user = prompts.render(
            "naturalize_screen_user", n=len(texts), numbered=prompts.numbered(texts)
        )
        return self.dict_items(
            self._ask_json(
                system, user, tier="cheap", key="issues", default=[], operation="naturalize.screen"
            )
        )

    def rewrite(self, text: str, quote: str, reason: str) -> str:
        """单语改写整段（strong 档）；失败或空结果回退原文。"""
        system = prompts.render("naturalize_rewrite_system")
        user = prompts.render("naturalize_rewrite_user", text=text, quote=quote, reason=reason)
        data = self._ask_json(
            system, user, tier="strong", default={}, operation="naturalize.rewrite"
        )
        rewritten = data.get("rewritten") if isinstance(data, dict) else None
        return rewritten.strip() if isinstance(rewritten, str) and rewritten.strip() else text

    def judge_pair(self, a: str, b: str) -> str:
        """单次成对判断，返回 "A"/"B"/"tie"（异常或不合法输出保守视为 tie）。"""
        system = prompts.render("naturalize_pair_system")
        user = prompts.render("naturalize_pair_user", a=a, b=b)
        data = self._ask_json(system, user, tier="cheap", default={}, operation="naturalize.pair")
        winner = data.get("winner") if isinstance(data, dict) else None
        return winner if winner in ("A", "B", "tie") else "tie"

    def pairwise_accept(
        self, orig: str, rewritten: str, executor: "ThreadPoolExecutor | None" = None
    ) -> bool:
        """正反两序各判一次；改写版两序皆胜才采纳，tie/负任一次即拒。

        executor 给出时正反两次 judge_pair 并发提交（章级复用的 2-worker 池）；
        不给时保持原顺序串行调用（独立调用/测试场景）。判定条件与结果语义不变。
        """
        if executor is not None:
            fut1 = executor.submit(self.judge_pair, orig, rewritten)  # A=原译 B=改写
            fut2 = executor.submit(self.judge_pair, rewritten, orig)  # A=改写 B=原译
            order1, order2 = fut1.result(), fut2.result()
        else:
            order1 = self.judge_pair(orig, rewritten)
            order2 = self.judge_pair(rewritten, orig)
        return order1 == "B" and order2 == "A"

    def fidelity_check(self, source: str, orig: str, rewritten: str) -> bool:
        """关卡③：双语忠实度判断，解析失败/字段缺失按不通过处理（保守）。"""
        system = prompts.render("naturalize_fidelity_system")
        user = prompts.render(
            "naturalize_fidelity_user", source=source, orig=orig, rewritten=rewritten
        )
        return bool(
            self._ask_json(
                system,
                user,
                tier="cheap",
                key="faithful",
                default=False,
                operation="naturalize.fidelity",
            )
        )


def _lint_introduces_new_issue(
    source: str,
    orig: str,
    rewritten: str,
    locked_terms: list[GlossaryTerm],
    src_lang: str,
) -> bool:
    """关卡①：与 orchestrator L1013-1020 的 polish 回退用的是相同逻辑——按 issue 类型集合比较。"""
    orig_types = {
        it.type
        for it in lint.lint_targets([source], [orig], locked_terms=locked_terms, src_lang=src_lang)
    }
    new_types = {
        it.type
        for it in lint.lint_targets(
            [source], [rewritten], locked_terms=locked_terms, src_lang=src_lang
        )
    }
    return bool(new_types - orig_types)


def naturalize_chapter(
    agent: Naturalizer,
    chapter: Chapter,
    ci: int,
    total: int,
    locked_terms: list[GlossaryTerm],
    config,
    store: RunStore,
    *,
    dry_run: bool,
    remaining: int | None,
) -> dict[str, Any]:
    """处理单章，返回统计并按需写回（save_chapter + log_event）。

    remaining：本次运行剩余可采纳配额（None=无限）；用尽即停止本章后续审读，控制预算。
    非 dry_run 时，无论本章是否有改写被采纳，函数末尾都会置
    chapter.meta["naturalized"] = True 并调用一次 store.save_chapter(chapter)——
    标记与（可能存在的）改写在同一次保存中一并落盘，避免 caller 二次保存造成的
    崩溃窗口（改写已落盘但标记未写，续跑重复审读）。dry_run 不置标记、不落盘。
    """
    stats = {
        "screened": 0,
        "suspects": 0,
        "rewritten": 0,
        "lint_rejected": 0,
        "fidelity_rejected": 0,
        "pairwise_rejected": 0,
        "applied": 0,
        "applied_entries": [],
    }
    cands: list[Segment] = []
    if not is_back_matter(chapter.title, index=ci, total=total):
        cands = candidate_segments(chapter)

    with ThreadPoolExecutor(max_workers=2) as pair_executor:
        for start in range(0, len(cands), _SCREEN_BATCH_SIZE):
            if remaining is not None and remaining <= 0:
                break
            batch = cands[start : start + _SCREEN_BATCH_SIZE]
            texts = [s.target or "" for s in batch]
            stats["screened"] += len(texts)
            issues = agent.screen(texts)
            for issue in issues:
                if remaining is not None and remaining <= 0:
                    break
                idx = issue.get("index")
                if not isinstance(idx, int) or not (0 <= idx < len(batch)):
                    continue
                seg = batch[idx]
                stats["suspects"] += 1
                quote = str(issue.get("quote", ""))
                reason = str(issue.get("reason", ""))
                before = seg.target or ""
                rewritten = agent.rewrite(before, quote, reason)
                if rewritten.strip() == before.strip():
                    agent.client.usage.record_outcome("naturalize.rewrite", accepted=False)
                    continue
                stats["rewritten"] += 1

                if _lint_introduces_new_issue(
                    seg.source, before, rewritten, locked_terms, config.source_lang
                ):
                    stats["lint_rejected"] += 1
                    agent.client.usage.record_outcome("naturalize.rewrite", accepted=False)
                    if not dry_run:
                        store.log_event(
                            "naturalize_rejected",
                            chapter=ci,
                            index=seg.index,
                            gate="lint",
                            detail={"quote": quote, "reason": reason, "rewritten": rewritten},
                        )
                    continue

                # 关卡③忠实度：is True 才放行成对判断——异常/默认 False 一律短路，
                # 绝不在忠实度未确认前发出两次成对判断请求。
                if agent.fidelity_check(seg.source, before, rewritten) is not True:
                    stats["fidelity_rejected"] += 1
                    agent.client.usage.record_outcome("naturalize.rewrite", accepted=False)
                    if not dry_run:
                        store.log_event(
                            "naturalize_rejected",
                            chapter=ci,
                            index=seg.index,
                            gate="fidelity",
                            detail={"quote": quote, "reason": reason, "rewritten": rewritten},
                        )
                    continue

                if not agent.pairwise_accept(before, rewritten, pair_executor):
                    stats["pairwise_rejected"] += 1
                    agent.client.usage.record_outcome("naturalize.rewrite", accepted=False)
                    if not dry_run:
                        store.log_event(
                            "naturalize_rejected",
                            chapter=ci,
                            index=seg.index,
                            gate="pairwise",
                            detail={"quote": quote, "reason": reason, "rewritten": rewritten},
                        )
                    continue

                final = rewritten
                if config.punctuation_normalize:
                    final = normalize_zh(final)
                stats["applied"] += 1
                agent.client.usage.record_outcome("naturalize.rewrite", accepted=True)
                stats["applied_entries"].append(
                    {"chapter": ci, "index": seg.index, "before": before, "after": final}
                )
                if remaining is not None:
                    remaining -= 1
                if not dry_run:
                    seg.target = final
                    store.log_event(
                        "naturalize_applied",
                        chapter=ci,
                        index=seg.index,
                        before=before,
                        after=final,
                        quote=quote,
                        reason=reason,
                    )

    if not dry_run:
        chapter.meta["naturalized"] = True
        store.save_chapter(chapter)
    return stats


_STAT_KEYS = (
    "screened",
    "suspects",
    "rewritten",
    "lint_rejected",
    "fidelity_rejected",
    "pairwise_rejected",
    "applied",
)


def run_naturalize(
    agent: Naturalizer,
    store: RunStore,
    glossary: GlossaryStore,
    config,
    *,
    chapters: list[int] | None = None,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """跑指定（或全部正文）章节的去翻译腔闭环，返回汇总统计。"""
    manifest = store.load_manifest()
    all_indices = [c["index"] for c in manifest["chapters"]]
    total = len(all_indices)
    target_indices = chapters if chapters is not None else all_indices
    locked_terms = [t for t in glossary.all_terms() if t.locked]

    totals: dict[str, Any] = {k: 0 for k in _STAT_KEYS}
    totals["applied_entries"] = []
    remaining = limit
    for ci in target_indices:
        if remaining is not None and remaining <= 0:
            break
        chapter = store.load_chapter(ci)
        stats = naturalize_chapter(
            agent,
            chapter,
            ci,
            total,
            locked_terms,
            config,
            store,
            dry_run=dry_run,
            remaining=remaining,
        )
        for k in _STAT_KEYS:
            totals[k] += stats[k]
        totals["applied_entries"].extend(stats["applied_entries"])
        if remaining is not None:
            remaining -= stats["applied"]
    return totals
