"""Deterministic planner for the minimal translation workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import ExecutionGoal, NodeAction
from trans_novel.pipeline.definition import WorkflowDefinition
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import (
    NODE_ANALYZE,
    NODE_ASSEMBLE,
    NODE_DETERMINISTIC_QA,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REPAIR,
    NODE_REPORT,
    NODE_SUCCEEDED,
    NODE_TITLES,
    NODE_TRANSLATE,
    SCOPE_BOOK,
    SCOPE_CHAPTER,
    RunState,
    chapter_node_key,
)

_BM_RANK = {"skip": 0, "light": 1, "full": 2}
_CHAIN = (NODE_TRANSLATE, NODE_POLISH)


@dataclass(frozen=True)
class WorkflowPolicy:
    polish: bool = False
    back_matter: str = "light"
    prescan_concurrency: int = 4

    @classmethod
    def from_config(cls, config) -> WorkflowPolicy:
        p = config.pipeline
        return cls(
            polish=p.polish, back_matter=p.back_matter, prescan_concurrency=p.prescan_concurrency
        )


@dataclass
class PrescanInputs:
    mine_fingerprint: Callable[[], str] | None = None
    name_terms_fingerprint: Callable[[], str] | None = None
    prepare_fingerprint: Callable[[], str] | None = None
    analyze_fingerprint: Callable[[], str] | None = None
    translate_fingerprint: Callable[[int], str] | None = None
    polish_fingerprint: Callable[[int], str] | None = None
    titles_fingerprint: Callable[[], str] | None = None
    deterministic_qa_fingerprint: Callable[[], str] | None = None
    report_fingerprint: Callable[[], str] | None = None
    assemble_fingerprint: Callable[[], str] | None = None


@dataclass(frozen=True)
class PlanEntry:
    node_id: str
    key: str
    ci: int | None
    scope: str
    action: NodeAction = "run"
    finalize_chapter: bool = False


@dataclass
class PlannedStage:
    entries: list[PlanEntry]
    parallel: bool = False
    max_workers: int = 1


@dataclass
class WorkflowPlan:
    stages: list[PlannedStage] = field(default_factory=list)
    goal: ExecutionGoal | None = None
    targets: list[int] = field(default_factory=list)

    def entry_keys(self) -> set[str]:
        return {e.key for stage in self.stages for e in stage.entries}


class Planner:
    def __init__(self, definition: WorkflowDefinition):
        self.definition = definition

    def build_plan(
        self,
        *,
        goal: ExecutionGoal,
        store: RunStore,
        policy: WorkflowPolicy,
        prescan: PrescanInputs,
    ) -> WorkflowPlan:
        self.definition.validate_goal(goal.phases)
        state = store.load_state() if store.exists() else RunState()
        chapters = list(state.chapters)
        if goal.only_chapter is not None and goal.only_chapter not in {c.index for c in chapters}:
            raise ValueError(f"章节编号 {goal.only_chapter} 不存在")
        if "translate" in goal.phases:
            self._reopen_upgraded(store, policy, chapters)
            state = store.load_state()
            chapters = list(state.chapters)
        if store.exists():
            computed = {}
            for key in state.nodes:
                base, sep, suffix = key.partition(":")
                if base not in self.definition.node_ids:
                    continue
                ci = int(suffix) if sep and suffix.isdigit() else None
                fingerprint = self._fingerprint(base, ci, prescan)
                if fingerprint is not None:
                    computed[key] = fingerprint
            store.reconcile_fingerprints(computed)
            state = store.load_state()
            changed = False
            for key, fingerprint in computed.items():
                node = state.nodes.get(key)
                if (
                    node is not None
                    and node.status == NODE_SUCCEEDED
                    and not node.input_fingerprint
                ):
                    node.input_fingerprint = fingerprint
                    changed = True
            if changed:
                store.save_state(state)
        plan = WorkflowPlan(goal=goal)
        needed: dict[str, PlanEntry] = {}

        def add(node: str, ci: int | None = None, action: NodeAction = "run", final=False):
            key = chapter_node_key(node, ci) if ci is not None else node
            needed.setdefault(
                key,
                PlanEntry(
                    node, key, ci, SCOPE_CHAPTER if ci is not None else SCOPE_BOOK, action, final
                ),
            )

        def need(node: str, ci: int | None = None, *, force: bool = False):
            key = chapter_node_key(node, ci) if ci is not None else node
            if key in needed:
                return
            if (
                ci is not None
                and self._back_matter(chapters, ci, policy)
                and node != NODE_TRANSLATE
            ):
                return
            disabled = node == NODE_POLISH and not policy.polish
            if disabled:
                add(node, ci, "skip")
                return
            fn = self._fingerprint(node, ci, prescan)
            current = state.nodes.get(key)
            satisfied = (
                current is not None
                and current.status == NODE_SUCCEEDED
                and (fn is None or not current.input_fingerprint or current.input_fingerprint == fn)
            )
            if force or not satisfied:
                add(node, ci)
            if node == NODE_TRANSLATE:
                return
            if node == NODE_POLISH:
                need(NODE_TRANSLATE, ci)
                return
            if node == NODE_NAME_TERMS:
                need(NODE_ANALYZE)
                need(NODE_MINE_TERMS, force=force or not satisfied)
            elif node == NODE_TITLES:
                for c in chapters:
                    need(NODE_TRANSLATE, c.index)
                    if policy.polish and not self._back_matter(chapters, c.index, policy):
                        need(NODE_POLISH, c.index)
            elif node == NODE_DETERMINISTIC_QA:
                need(NODE_TITLES)
            elif node == NODE_REPORT:
                need(NODE_REPAIR, force=force)
            elif node == NODE_REPAIR:
                need(NODE_DETERMINISTIC_QA)
            elif node == NODE_ASSEMBLE:
                need(NODE_REPORT)

        if "prepare" in goal.phases:
            need(NODE_PREPARE)
            need(NODE_ANALYZE)
        if "prescan" in goal.phases:
            need(NODE_MINE_TERMS)
            need(NODE_NAME_TERMS)
        if "translate" in goal.phases:
            targets = (
                [goal.only_chapter]
                if goal.only_chapter is not None
                else sorted(store.pending_chapters())
            )
            plan.targets = [ci for ci in targets if ci is not None]
            for ci in plan.targets:
                need(NODE_TRANSLATE, ci)
                need(NODE_POLISH, ci)
        if "titles" in goal.phases:
            need(NODE_TITLES)
        if "qa" in goal.phases:
            need(NODE_DETERMINISTIC_QA, force=True)
        if "repair" in goal.phases:
            need(NODE_REPAIR, force=True)
        if "report" in goal.phases:
            need(NODE_REPORT, force=True)
        if "assemble" in goal.phases:
            need(NODE_ASSEMBLE, force=True)
        self._schedule(plan, needed, chapters, policy)
        return plan

    @staticmethod
    def _fingerprint(node: str, ci: int | None, prescan: PrescanInputs) -> str | None:
        """Select the injected fingerprint formula for a current workflow node."""
        callbacks = {
            NODE_PREPARE: prescan.prepare_fingerprint,
            NODE_ANALYZE: prescan.analyze_fingerprint,
            NODE_MINE_TERMS: prescan.mine_fingerprint,
            NODE_NAME_TERMS: prescan.name_terms_fingerprint,
            NODE_TRANSLATE: prescan.translate_fingerprint,
            NODE_POLISH: prescan.polish_fingerprint,
            NODE_TITLES: prescan.titles_fingerprint,
            NODE_DETERMINISTIC_QA: prescan.deterministic_qa_fingerprint,
            NODE_REPORT: prescan.report_fingerprint,
            NODE_ASSEMBLE: prescan.assemble_fingerprint,
        }
        callback = callbacks.get(node)
        if callback is None:
            return None
        try:
            return callback(ci) if ci is not None else callback()
        except TypeError:
            return None

    def _schedule(self, plan, needed, chapters, policy):
        def take(node, ci=None):
            return needed.get(chapter_node_key(node, ci) if ci is not None else node)

        prep = [take(NODE_PREPARE), take(NODE_ANALYZE)]
        if any(prep):
            plan.stages.append(PlannedStage([x for x in prep if x]))
        mine = take(NODE_MINE_TERMS)
        name = take(NODE_NAME_TERMS)
        if mine is not None:
            plan.stages.append(PlannedStage([mine]))
        if name is not None:
            plan.stages.append(PlannedStage([name]))
        by = {c.index: c for c in chapters}
        entries = []
        for ci in sorted(by):
            chapter_entries = [item for node in _CHAIN if (item := take(node, ci)) is not None]
            if chapter_entries:
                terminal = chapter_entries[-1].key
                entries.extend(
                    PlanEntry(
                        item.node_id,
                        item.key,
                        item.ci,
                        item.scope,
                        item.action,
                        item.key == terminal,
                    )
                    for item in chapter_entries
                )
        if entries:
            plan.stages.append(PlannedStage(entries))
        for node in (NODE_TITLES, NODE_DETERMINISTIC_QA, NODE_REPAIR, NODE_REPORT, NODE_ASSEMBLE):
            item = take(node)
            if item is not None:
                plan.stages.append(PlannedStage([item]))

    def _back_matter(self, chapters, ci, policy):
        chapter = next((c for c in chapters if c.index == ci), None)
        return (
            chapter is not None
            and policy.back_matter in {"skip", "light"}
            and is_back_matter(chapter.title, index=ci, total=len(chapters))
        )

    def _reopen_upgraded(self, store, policy, chapters):
        for chapter in chapters:
            progress = store.load_progress(chapter.index)
            prev = progress.back_matter_mode or "full"
            current = (
                policy.back_matter if self._back_matter(chapters, chapter.index, policy) else "full"
            )
            if (
                progress.status == STATUS_DONE
                and prev in _BM_RANK
                and _BM_RANK[current] > _BM_RANK[prev]
            ):
                store.reopen_back_matter_chapter(
                    chapter.index, prev_mode=prev, mode=current, title=chapter.title
                )


__all__ = [
    "PlanEntry",
    "PlannedStage",
    "Planner",
    "PrescanInputs",
    "WorkflowPlan",
    "WorkflowPolicy",
]
