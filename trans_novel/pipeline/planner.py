"""Planner：从执行目标 + 质量策略 + 持久化 NodeState/指纹 计算确定性计划。

产出按阶段有序、可并行的 PlanEntry 列表：
  prepare → analyze → [digest ∥ mine_terms]（并行层）→ name_terms → book_synopsis
  → 逐章 translate/polish/naturalize/review/backtranslate（串行，章内保序）
  → titles → consistency_qa → report → assemble

- 目标阶段决定“根节点”（prepare 阶段隐含 analyze）；planner 对根节点求依赖
  闭包（WorkflowDefinition.depends_on / aggregates 展开到章节），只把“未满足”
  的依赖加入计划：
    已满足 = 持久化 succeeded（有指纹契约的节点还需指纹一致，空指纹的 legacy
            成功态视为已满足——V1 迁移合成状态）或策略性 skipped（本轮禁用）；
    已跳过 = 本轮策略禁用的可选节点，以显式 skipped 计划项持久化。
  服务目标（qa/report/assemble/titles）同样不能绕过硬编码阶段表之外的前置节点；
  附属章旁路链的“不适用”节点视为已满足（旁路章无质量环节、非正文章无 digest）。
- 指纹对账在 build_plan 开头执行（生产路径由 runner 在运行锁内调用本函数）：
  只失效“变了”的节点与后代，清除对应产物，再据此选目标；translate 译文与
  titles 译名的清除由 RunStore 层完成。
- 附属章升档重开扫描在目标筛选前全局执行（only_chapter 场景也不漏）。
- 计划本身不再修改持久化状态：失效清除与升档重开是 build_plan 的副作用
  （对账语义要求“先清后选”），幂等且必须与执行同临界区。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import ExecutionGoal, NodeAction
from trans_novel.pipeline.definition import WorkflowDefinition
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import (
    _NODE_DESCENDANTS,
    NODE_ANALYZE,
    NODE_ASSEMBLE,
    NODE_BACKTRANSLATE,
    NODE_BOOK_SYNOPSIS,
    NODE_CONSISTENCY_QA,
    NODE_DIGEST,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_NATURALIZE,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REPORT,
    NODE_REVIEW,
    NODE_SUCCEEDED,
    NODE_TITLES,
    NODE_TRANSLATE,
    SCOPE_BOOK,
    SCOPE_CHAPTER,
    RunState,
    chapter_node_key,
)

_BM_RANK = {"skip": 0, "light": 1, "full": 2}

# 章链顺序（含策略禁用的占位）；backtranslate 常驻收尾，永远是正文链末节点。
_CHAIN = (NODE_TRANSLATE, NODE_POLISH, NODE_NATURALIZE, NODE_REVIEW, NODE_BACKTRANSLATE)


@dataclass(frozen=True)
class WorkflowPolicy:
    """质量/成本档位展开后的运行策略（planner 的禁用依据）。"""

    book_understanding: bool = True
    review: bool = True
    autofix_severe: bool = True
    polish: bool = False
    naturalize: bool = True
    backtranslate_sample: float = 0.0
    consistency_qa: bool = False
    back_matter: str = "light"
    prescan_concurrency: int = 4

    @classmethod
    def from_config(cls, config) -> WorkflowPolicy:
        p = config.pipeline
        return cls(
            book_understanding=p.book_understanding,
            review=p.review,
            autofix_severe=p.autofix_severe,
            polish=p.polish,
            naturalize=p.naturalize,
            backtranslate_sample=p.backtranslate_sample,
            consistency_qa=p.consistency_qa,
            back_matter=p.back_matter,
            prescan_concurrency=p.prescan_concurrency,
        )


@dataclass
class PrescanInputs:
    """预扫/全节点输入指纹计算器（由组合根注入，内部读取 store/config/agents）。

    每个可复算节点一个公式；None 表示该节点无指纹契约（按状态满足判定）。
    """

    digest_fingerprint: Callable[[int], str] | None = None
    mine_fingerprint: Callable[[], str] | None = None
    name_terms_fingerprint: Callable[[], str] | None = None
    synopsis_fingerprint: Callable[[list[str]], str] | None = None
    prepare_fingerprint: Callable[[], str] | None = None
    analyze_fingerprint: Callable[[], str] | None = None
    translate_fingerprint: Callable[[int], str] | None = None
    polish_fingerprint: Callable[[int], str] | None = None
    naturalize_fingerprint: Callable[[int], str] | None = None
    review_fingerprint: Callable[[int], str] | None = None
    backtranslate_fingerprint: Callable[[int], str] | None = None
    titles_fingerprint: Callable[[], str] | None = None
    consistency_fingerprint: Callable[[], str] | None = None
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
        return {e.key for s in self.stages for e in s.entries}


class Planner:
    def __init__(self, definition: WorkflowDefinition):
        self.definition = definition

    # ── 入口 ──────────────────────────────────────────────────────────────
    def build_plan(
        self,
        *,
        goal: ExecutionGoal,
        store: RunStore,
        policy: WorkflowPolicy,
        prescan: PrescanInputs,
    ) -> WorkflowPlan:
        self.definition.validate_goal(goal.phases)
        disabled = self._disabled_nodes(policy)
        self.definition.validate_disablement(disabled)
        if goal.do_qa is True:
            # 显式 QA 目标：无论质量档位如何都强制运行（tools qa / run_all --qa）。
            disabled = disabled - {NODE_CONSISTENCY_QA}
        elif goal.do_qa is False:
            # 显式关闭 QA：质量档位开着也跳过（translate --no-qa 契约）。
            disabled = disabled | {NODE_CONSISTENCY_QA}

        if store.exists():
            # 指纹对账（生产路径由 runner 在运行锁内调用）：失效“变了”的节点与
            # 后代、清除对应产物；再读最新状态选目标。
            computed = self._computed_fingerprints(store, policy, prescan)
            store.reconcile_fingerprints(computed)
            # 迁移合成/legacy 的空指纹成功态：首次 V2 规划时原子回填基线指纹，
            # 此后参与对账——配置/模型变化能正确失效迁移产物及其后代。
            self._backfill_fingerprints(store, computed)
            state = store.load_state()
        else:
            # 全新运行：prepare 阶段暂存文档前没有任何章信息（章相关规划在
            # prepare 完成后由组合根二次构建）。
            state = RunState()
        chapters = list(state.chapters)
        indices = [c.index for c in chapters]
        n = len(chapters)
        if goal.only_chapter is not None and goal.only_chapter not in indices:
            available = sorted(i for i in indices if isinstance(i, int))
            if available:
                raise ValueError(
                    f"章节编号 {goal.only_chapter} 不存在；可用范围：0–{available[-1]}"
                )
            raise ValueError(f"章节编号 {goal.only_chapter} 不存在；当前没有可翻译的章节")
        if "translate" in goal.phases:
            # 附属章升档重开先于目标筛选与依赖闭包（only_chapter 场景也不漏）：
            # 重开会把章改回 pending、清空译文，闭包据此把 translate 重新纳入计划。
            self._reopen_upgraded(store, policy, chapters, n)
            state = store.load_state()
            chapters = list(state.chapters)
            indices = [c.index for c in chapters]
            n = len(chapters)

        plan = WorkflowPlan(goal=goal)

        # 目标阶段 → 根节点集合（闭包会补上未满足的前置依赖）。
        roots: list[tuple[str, int | None]] = []
        if "prepare" in goal.phases:
            roots.append((NODE_PREPARE, None))
            roots.append((NODE_ANALYZE, None))
        if "prescan" in goal.phases:
            roots.append((NODE_MINE_TERMS, None))
            for c in chapters:
                if self._body_chapter(c, n, policy):
                    roots.append((NODE_DIGEST, c.index))
            roots.append((NODE_NAME_TERMS, None))
            roots.append((NODE_BOOK_SYNOPSIS, None))
        if "translate" in goal.phases:
            chain_roots = [goal.only_chapter] if goal.only_chapter is not None else indices
            for ci in chain_roots:
                for node_id in _CHAIN:
                    roots.append((node_id, ci))
        if "titles" in goal.phases:
            roots.append((NODE_TITLES, None))
        if "qa" in goal.phases:
            roots.append((NODE_CONSISTENCY_QA, None))
        if "report" in goal.phases:
            roots.append((NODE_REPORT, None))
        if "assemble" in goal.phases:
            roots.append((NODE_ASSEMBLE, None))

        needed: dict[str, PlanEntry] = {}
        if "prepare" in goal.phases:
            # 身份核验门：prepare 每次运行都执行（幂等：核验身份 + 续跑事件）。
            # 只靠 succeeded 状态跳过会让“源文件被替换”的续跑静默通过身份校验。
            # 服务目标（无 prepare 阶段）不强制：存量书直读状态，正式回填的身份
            # 核验由 writer 就绪门禁负责（与旧 tools 行为一致）。
            needed[NODE_PREPARE] = self._run_entry(NODE_PREPARE, None)
        # 显式请求的动作根（qa/report/assemble/titles）总是执行：命令的可观察输出
        # 就是该节点本轮的内存产物/副作用，不能因上次 succeeded 被剪枝
        # （第二次 run_all / tools report|qa|assemble 必须重新产出）。
        forced_actions = {
            NODE_CONSISTENCY_QA: "qa" in goal.phases,
            NODE_REPORT: "report" in goal.phases,
            NODE_ASSEMBLE: "assemble" in goal.phases,
            NODE_TITLES: "titles" in goal.phases,
        }
        for node_id, ci in roots:
            # 先走依赖闭包（拉前置依赖 + 判定满足），再覆盖动作根为强制执行。
            self._require(
                node_id,
                ci,
                store=store,
                policy=policy,
                disabled=disabled,
                prescan=prescan,
                state=state,
                n=n,
                needed=needed,
            )
            if forced_actions.get(node_id):
                key = node_id
                if key not in needed:
                    # 动作根自身 succeeded 导致 _require 提前返回：仍必须递归要求其
                    # 依赖——pending/失败的前置（崩溃恢复）不能因根已成功而被跳过。
                    self._require_dependencies(
                        node_id,
                        ci,
                        store=store,
                        policy=policy,
                        disabled=disabled,
                        prescan=prescan,
                        state=state,
                        n=n,
                        needed=needed,
                    )
                needed[key] = (
                    self._skip_entry(node_id, None)
                    if node_id in disabled
                    else self._run_entry(node_id, None)
                )
        # 同轮生产者→消费者级联：本轮计划内的节点将产出新产物，其消费者（依赖该
        # 产物参与指纹/后代表）必须一并重跑——例如 book_understanding 从禁用切到
        # 启用时，digest/name/synopsis 在本轮产出，translate/titles 的指纹基于旧
        # （空）概览，必须在本轮重跑而非等第二次 run。
        self._force_planned_consumers(
            goal,
            roots,
            disabled,
            needed,
            store=store,
            policy=policy,
            state=state,
            prescan=prescan,
        )

        # 按固定阶段结构把 needed 落成有序计划。
        self._schedule(
            plan,
            goal=goal,
            store=store,
            policy=policy,
            disabled=disabled,
            state=state,
            chapters=chapters,
            n=n,
            needed=needed,
        )
        return plan

    def _require_dependencies(
        self,
        node_id: str,
        ci: int | None,
        *,
        store: RunStore,
        policy: WorkflowPolicy,
        disabled: set[str],
        prescan: PrescanInputs,
        state: RunState,
        n: int,
        needed: dict[str, PlanEntry],
    ) -> None:
        """递归要求节点依赖（跳过“自身已满足”的提前返回）：动作根即使 succeeded
        也必须补齐其前置链（pending 崩溃恢复）。"""
        spec = self.definition.spec(node_id)
        for dep in spec.depends_on:
            dep_spec = self.definition.spec(dep)
            if dep_spec.scope == SCOPE_CHAPTER:
                if spec.scope == SCOPE_BOOK:
                    for c in state.chapters:
                        self._require(
                            dep,
                            c.index,
                            store=store,
                            policy=policy,
                            disabled=disabled,
                            prescan=prescan,
                            state=state,
                            n=n,
                            needed=needed,
                        )
                else:
                    self._require(
                        dep,
                        ci,
                        store=store,
                        policy=policy,
                        disabled=disabled,
                        prescan=prescan,
                        state=state,
                        n=n,
                        needed=needed,
                    )
            else:
                self._require(
                    dep,
                    None,
                    store=store,
                    policy=policy,
                    disabled=disabled,
                    prescan=prescan,
                    state=state,
                    n=n,
                    needed=needed,
                )

    # ── 迁移指纹基线回填 ──────────────────────────────────────────────────
    def _backfill_fingerprints(self, store: RunStore, computed: dict[str, str]) -> None:
        """为“成功但空指纹”的节点（V1 迁移合成/legacy）原子回填当前输入指纹。

        回填后这些节点参与后续 reconcile 对账：provider/model 或其它指纹输入
        变化能正确失效迁移产物及其后代（不再全局豁免空指纹成功态）。
        """
        state = store.load_state()
        updates: dict[str, str] = {}
        for key, fp in computed.items():
            node = state.nodes.get(key)
            if node is not None and node.status == NODE_SUCCEEDED and not node.input_fingerprint:
                updates[key] = fp
        if updates:
            state = store.load_state()
            for key, fp in updates.items():
                state.nodes[key].input_fingerprint = fp
            store.save_state(state)

    # ── 同轮生产者→消费者级联 ─────────────────────────────────────────────
    def _force_planned_consumers(
        self,
        goal: ExecutionGoal,
        roots: list[tuple[str, int | None]],
        disabled: set[str],
        needed: dict[str, PlanEntry],
        *,
        store: RunStore,
        policy: WorkflowPolicy,
        state: RunState,
        prescan: PrescanInputs,
    ) -> None:
        """本轮计划内的生产者将产出新产物，其消费者（依赖该产物参与指纹/后代表）
        必须一并重跑——例如 book_understanding 从禁用切到启用时，digest/name/
        synopsis 在本轮产出，translate/titles 的指纹基于旧（空）概览，必须在本轮
        重跑而非等第二次 run。级联只落在目标阶段允许的节点与活跃章内
        （only_chapter 只强制目标章，不会把全书拉进计划）。
        """
        allowed: set[str] = set()
        if "prepare" in goal.phases:
            allowed |= {NODE_PREPARE, NODE_ANALYZE}
        if "prescan" in goal.phases:
            allowed |= {NODE_DIGEST, NODE_MINE_TERMS, NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS}
        if "translate" in goal.phases:
            allowed |= set(_CHAIN)
        if "titles" in goal.phases:
            allowed.add(NODE_TITLES)
        if "qa" in goal.phases:
            allowed.add(NODE_CONSISTENCY_QA)
        if "report" in goal.phases:
            allowed.add(NODE_REPORT)
        if "assemble" in goal.phases:
            allowed.add(NODE_ASSEMBLE)
        if not allowed:
            return
        # 活跃章 = 翻译阶段根节点触及的章（only_chapter 只含目标章，全量含全部章）。
        # 注意：预扫 digest 的根节点覆盖全部正文章（ci 含非目标章），不能计入——
        # book_synopsis→translate:all 级联会据此把 only_chapter 之外的书拉进计划。
        active_ci: set[int] = {ci for node_id, ci in roots if ci is not None and node_id in _CHAIN}
        forced: set[str] = set(needed)
        queue = list(forced)
        while queue:
            key = queue.pop()
            base, sep, suffix = key.partition(":")
            for child, mode in _NODE_DESCENDANTS.get(base, ()):
                if mode == "all":
                    if child not in allowed:
                        continue
                    children = [chapter_node_key(child, ci) for ci in sorted(active_ci)]
                elif mode is True:
                    if child not in allowed:
                        continue
                    children = [f"{child}:{suffix}"] if sep else []
                else:
                    children = [child] if child in allowed else []
                for child_key in children:
                    if child_key in forced:
                        continue
                    child_node, _, child_suffix = child_key.partition(":")
                    if self._full_bm_satisfied(child_node, child_suffix, state, policy):
                        # full 档产出的附属章满足更低旁路档：级联不得强制它重跑旁路
                        # （会覆盖高质量译文）。
                        continue
                    forced.add(child_key)
                    queue.append(child_key)
        for key in sorted(forced - set(needed)):
            node_id, sep, suffix = key.partition(":")
            ci = int(suffix) if sep and suffix.isdigit() else None
            if node_id in disabled:
                needed[key] = self._skip_entry(node_id, ci)
            else:
                needed[key] = self._run_entry(node_id, ci)

    @staticmethod
    def _full_bm_satisfied(node_id: str, suffix: str, state: RunState, policy) -> bool:
        """full 档产出的已完附属章（mode None）满足更低的旁路档策略。"""
        if node_id != NODE_TRANSLATE or not suffix.isdigit():
            return False
        ci = int(suffix)
        progress = state.progress.get(ci)
        if progress is None or progress.status != "done" or progress.back_matter_mode is not None:
            return False
        chapter = next((c for c in state.chapters if c.index == ci), None)
        if chapter is None:
            return False
        mode = policy.back_matter
        return bool(
            mode in ("skip", "light")
            and is_back_matter(chapter.title, index=chapter.index, total=len(state.chapters))
        )

    # ── 依赖闭包 ──────────────────────────────────────────────────────────
    def _require(
        self,
        node_id: str,
        ci: int | None,
        *,
        store: RunStore,
        policy: WorkflowPolicy,
        disabled: set[str],
        prescan: PrescanInputs,
        state: RunState,
        n: int,
        needed: dict[str, PlanEntry],
    ) -> None:
        key = chapter_node_key(node_id, ci) if ci is not None else node_id
        if key in needed:
            return
        if ci is not None and not self._applies_to_chapter(node_id, ci, state, policy, n):
            return  # 该章不适用（旁路章无质量环节 / 非正文章无 digest）→ 无工作
        if node_id in disabled:
            needed[key] = self._skip_entry(node_id, ci)
            return
        if self._node_satisfied(node_id, ci, store, policy, prescan, state):
            return
        needed[key] = self._run_entry(node_id, ci)
        spec = self.definition.spec(node_id)
        for dep in spec.depends_on:
            dep_spec = self.definition.spec(dep)
            if dep_spec.scope == SCOPE_CHAPTER:
                if spec.scope == SCOPE_BOOK:
                    # 聚合边（fan-in）：book 节点对 chapter 依赖展开到全部章节
                    # （titles 的终端分支、book_synopsis 的 digest 全集）。
                    for c in state.chapters:
                        self._require(
                            dep,
                            c.index,
                            store=store,
                            policy=policy,
                            disabled=disabled,
                            prescan=prescan,
                            state=state,
                            n=n,
                            needed=needed,
                        )
                else:
                    # 章内链：chapter 节点的 chapter 依赖沿用同一章 ci。
                    self._require(
                        dep,
                        ci,
                        store=store,
                        policy=policy,
                        disabled=disabled,
                        prescan=prescan,
                        state=state,
                        n=n,
                        needed=needed,
                    )
            else:
                self._require(
                    dep,
                    None,
                    store=store,
                    policy=policy,
                    disabled=disabled,
                    prescan=prescan,
                    state=state,
                    n=n,
                    needed=needed,
                )

    def _applies_to_chapter(
        self, node_id: str, ci: int, state: RunState, policy: WorkflowPolicy, n: int
    ) -> bool:
        chapter = next((c for c in state.chapters if c.index == ci), None)
        if chapter is None:
            return False
        if node_id == NODE_DIGEST:
            return self._body_chapter(chapter, n, policy)
        mode = self._back_matter_mode(chapter, n, policy)
        if mode:
            # 旁路章：只有 translate（自理收尾），无质量环节。
            return node_id == NODE_TRANSLATE
        return node_id in _CHAIN

    def _node_satisfied(
        self,
        node_id: str,
        ci: int | None,
        store: RunStore,
        policy: WorkflowPolicy,
        prescan: PrescanInputs,
        state: RunState,
    ) -> bool:
        key = chapter_node_key(node_id, ci) if ci is not None else node_id
        node = state.nodes.get(key)
        if node is None or node.status != NODE_SUCCEEDED:
            return False
        if node_id == NODE_MINE_TERMS and not state.analysis_flags.term_mining_done:
            # 与 name_terms 共享检查点：定名失败时 term_mining_done 未落盘，
            # 续跑必须重跑挖掘（否则 name_terms 拿不到候选、静默永久跳过）。
            return False
        # 恢复标记是权威续跑信号：pending_polish / review_pending 存在时节点必须补跑
        # （即使节点状态为 succeeded——崩溃发生在清标记/清待办之前）。
        if ci is not None:
            progress = state.progress.get(ci)
            if progress is not None:
                if node_id == NODE_TRANSLATE and progress.status != "done":
                    return False  # 章未完成 → translate 必须重跑（批级续跑/重检）
                if node_id == NODE_POLISH and progress.pending_polish:
                    return False
                if node_id == NODE_REVIEW and progress.review_pending:
                    return False
                if node_id == NODE_TRANSLATE and progress.back_matter_mode is None:
                    # full 档产出的已完附属章（mode None）：高质量译文满足更低的
                    # 旁路档策略，降档不回退——不重跑旁路（不覆盖译文）。
                    chapter = next((c for c in state.chapters if c.index == ci), None)
                    if chapter is not None and self._back_matter_mode(
                        chapter, len(state.chapters), policy
                    ):
                        return True
        fp = self._fingerprint_for(node_id, ci, store, policy, prescan, state)
        if fp is None:
            return True
        # 空指纹 = legacy 成功态（V1 迁移合成/空章无输入），视为已满足。
        return (not node.input_fingerprint) or node.input_fingerprint == fp

    def _fingerprint_for(
        self,
        node_id: str,
        ci: int | None,
        store: RunStore,
        policy: WorkflowPolicy,
        prescan: PrescanInputs,
        state: RunState,
    ) -> str | None:
        fn = {
            NODE_PREPARE: prescan.prepare_fingerprint,
            NODE_ANALYZE: prescan.analyze_fingerprint,
            NODE_MINE_TERMS: prescan.mine_fingerprint,
            NODE_NAME_TERMS: prescan.name_terms_fingerprint,
            NODE_BOOK_SYNOPSIS: (
                (lambda: prescan.synopsis_fingerprint(self._digests(state, store)))
                if prescan.synopsis_fingerprint is not None
                else None
            ),
            NODE_TRANSLATE: prescan.translate_fingerprint,
            NODE_POLISH: prescan.polish_fingerprint,
            NODE_NATURALIZE: prescan.naturalize_fingerprint,
            NODE_REVIEW: prescan.review_fingerprint,
            NODE_BACKTRANSLATE: prescan.backtranslate_fingerprint,
            NODE_TITLES: prescan.titles_fingerprint,
            NODE_CONSISTENCY_QA: prescan.consistency_fingerprint,
            NODE_REPORT: prescan.report_fingerprint,
            NODE_ASSEMBLE: prescan.assemble_fingerprint,
        }.get(node_id)
        if fn is None:
            return None
        try:
            return fn(ci) if ci is not None else fn()
        except TypeError:
            # 组合根注入的章节级公式要求 ci；退化时按无指纹契约处理。
            return None

    @staticmethod
    def _digests(state: RunState, store: RunStore) -> list[str]:
        return [store.load_progress(c.index).source_digest for c in state.chapters]

    # ── 落阶段 ────────────────────────────────────────────────────────────
    def _schedule(
        self,
        plan: WorkflowPlan,
        *,
        goal: ExecutionGoal,
        store: RunStore,
        policy: WorkflowPolicy,
        disabled: set[str],
        state: RunState,
        chapters,
        n: int,
        needed: dict[str, PlanEntry],
    ) -> None:
        def take(node_id: str, ci: int | None) -> PlanEntry | None:
            key = chapter_node_key(node_id, ci) if ci is not None else node_id
            return needed.get(key)

        # prepare 阶段：prepare + analyze（同一串行层，analyze 在后）。
        prep_entries: list[PlanEntry] = []
        for node_id in (NODE_PREPARE, NODE_ANALYZE):
            entry = take(node_id, None)
            if entry is not None:
                prep_entries.append(entry)
        if prep_entries:
            plan.stages.append(PlannedStage(prep_entries))

        # 预扫：book_understanding 关闭时整体显式 skipped；否则并行层 + name + synopsis。
        if NODE_DIGEST in disabled:
            body = [c for c in chapters if self._body_chapter(c, n, policy)]
            skip_entries = [
                PlanEntry(
                    node_id=NODE_DIGEST,
                    key=chapter_node_key(NODE_DIGEST, c.index),
                    ci=c.index,
                    scope=SCOPE_CHAPTER,
                    action="skip",
                )
                for c in body
            ]
            for node_id in (NODE_MINE_TERMS, NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS):
                skip_entries.append(
                    PlanEntry(
                        node_id=node_id, key=node_id, ci=None, scope=SCOPE_BOOK, action="skip"
                    )
                )
            plan.stages.append(PlannedStage(skip_entries))
        else:
            parallel: list[PlanEntry] = []
            mine = take(NODE_MINE_TERMS, None)
            if mine is not None:
                parallel.append(mine)
            for c in chapters:
                entry = take(NODE_DIGEST, c.index)
                if entry is not None:
                    parallel.append(entry)
            if parallel:
                plan.stages.append(
                    PlannedStage(
                        parallel,
                        parallel=True,
                        max_workers=max(1, policy.prescan_concurrency),
                    )
                )
            for node_id in (NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS):
                entry = take(node_id, None)
                if entry is not None:
                    plan.stages.append(PlannedStage([entry]))

        # 翻译阶段：章链（章序升序、章内按链序）。异步审校续跑优先。
        # 服务目标经依赖闭包也可能需要章链（如不完整书上的 assemble/qa 目标），
        # 不能只按“目标含 translate 阶段”决定是否落计划。
        if "translate" in goal.phases or any(
            e.action == "run" and e.node_id in _CHAIN for e in needed.values()
        ):
            self._schedule_translate(
                plan,
                goal=goal,
                store=store,
                policy=policy,
                disabled=disabled,
                state=state,
                chapters=chapters,
                n=n,
                needed=needed,
            )

        for node_id, ci in (
            (NODE_TITLES, None),
            (NODE_CONSISTENCY_QA, None),
            (NODE_REPORT, None),
            (NODE_ASSEMBLE, None),
        ):
            entry = take(node_id, ci)
            if entry is not None:
                plan.stages.append(PlannedStage([entry]))

    def _schedule_translate(
        self,
        plan: WorkflowPlan,
        *,
        goal: ExecutionGoal,
        store: RunStore,
        policy: WorkflowPolicy,
        disabled: set[str],
        state: RunState,
        chapters,
        n: int,
        needed: dict[str, PlanEntry],
    ) -> None:
        if not chapters:
            return  # 全新运行 prepare 完成前无章信息；章相关规划在暂存后二次构建
        targets = self._translate_targets(
            store, policy, chapters, n, only_chapter=goal.only_chapter
        )
        plan.targets = targets
        by_index = {c.index: c for c in chapters}
        # 会写回正文译文的节点：其重跑迫使所有下游质量节点重跑。
        # review 仅 autofix 模式写正文（异步审校只落 issues）。
        writers = {NODE_TRANSLATE, NODE_POLISH, NODE_NATURALIZE}
        if policy.autofix_severe:
            writers.add(NODE_REVIEW)
        entries: list[PlanEntry] = []

        def chapter_chain(ci: int) -> list[PlanEntry]:
            chapter = by_index.get(ci)
            mode = self._back_matter_mode(chapter, n, policy) if chapter is not None else None
            chain: list[PlanEntry] = []

            def run_entry(node_id: str) -> PlanEntry:
                # backtranslate 是正文链末节点：负责收尾 done 与 chapter_done 事件。
                if node_id == NODE_BACKTRANSLATE:
                    return PlanEntry(
                        node_id=node_id,
                        key=chapter_node_key(node_id, ci),
                        ci=ci,
                        scope=SCOPE_CHAPTER,
                        finalize_chapter=True,
                    )
                return self._run_entry(node_id, ci)

            if mode:
                # 旁路章：仅 translate（自理收尾），无质量环节。
                entry = needed.get(chapter_node_key(NODE_TRANSLATE, ci))
                if entry is not None:
                    chain.append(entry)
                return chain
            force = False
            for node_id in _CHAIN:
                key = chapter_node_key(node_id, ci)
                if node_id in disabled:
                    chain.append(self._skip_entry(node_id, ci))
                    continue
                if force:
                    # 上游译文已重写 → 本节点必须重跑（即使原状态已满足）。
                    chain.append(run_entry(node_id))
                    continue
                entry = needed.get(key)
                if entry is None:
                    continue  # 已满足，无条目
                if entry.action == "run" and node_id == NODE_BACKTRANSLATE:
                    entry = run_entry(node_id)
                chain.append(entry)
                if entry.action == "run" and node_id in writers:
                    force = True
            return chain

        # 异步审校断点续跑：已 done 但 review_pending 的章先补跑（与后续翻译重叠）。
        processed: set[int] = set()
        for ci in store.review_pending_chapters():
            if ci in targets or ci in processed:
                continue
            entries.extend(chapter_chain(ci))
            processed.add(ci)

        for ci in sorted(by_index):
            if ci in processed:
                continue
            entries.extend(chapter_chain(ci))

        if entries:
            plan.stages.append(PlannedStage(entries))

    # ── 翻译目标 ──────────────────────────────────────────────────────────
    def _reopen_upgraded(self, store: RunStore, policy: WorkflowPolicy, chapters, n: int) -> None:
        """附属章档位升档（skip→light/full、light→full）：全局扫描已完成章并重开
        重译（降档不回退）。必须在依赖闭包之前执行。"""
        for c in chapters:
            progress = store.load_progress(c.index)
            # full 档产出的附属章记 mode=None；视为 effective full（最高档），
            # 不存在可升档空间（降档由 _node_satisfied 的满足判定承接，不回退）。
            prev = progress.back_matter_mode or "full"
            if progress.status != STATUS_DONE or prev not in _BM_RANK:
                continue
            cur = self._back_matter_mode(c, n, policy) or "full"
            if _BM_RANK[cur] > _BM_RANK[prev]:
                store.reopen_back_matter_chapter(c.index, prev_mode=prev, mode=cur, title=c.title)

    def _translate_targets(
        self,
        store: RunStore,
        policy: WorkflowPolicy,
        chapters,
        n: int,
        *,
        only_chapter: int | None,
    ) -> list[int]:
        if only_chapter is not None:
            return [only_chapter]
        return sorted(store.pending_chapters())

    # ── 指纹对账 ──────────────────────────────────────────────────────────
    def _computed_fingerprints(
        self, store: RunStore, policy: WorkflowPolicy, prescan: PrescanInputs
    ) -> dict[str, str]:
        state = store.load_state()
        computed: dict[str, str] = {}
        n = len(state.chapters)

        def add(key: str, fn: Callable[[], str] | None) -> None:
            if fn is not None:
                computed[key] = fn()

        def chapter_fn(fn: Callable[[int], str] | None, ci: int) -> Callable[[], str] | None:
            return (lambda: fn(ci)) if fn is not None else None

        add(NODE_PREPARE, prescan.prepare_fingerprint)
        add(NODE_ANALYZE, prescan.analyze_fingerprint)
        for c in state.chapters:
            if not self._body_chapter(c, n, policy):
                continue
            ci = c.index
            add(
                chapter_node_key(NODE_DIGEST, ci),
                chapter_fn(prescan.digest_fingerprint, ci),
            )
            add(
                chapter_node_key(NODE_TRANSLATE, ci),
                chapter_fn(prescan.translate_fingerprint, ci),
            )
            add(
                chapter_node_key(NODE_POLISH, ci),
                chapter_fn(prescan.polish_fingerprint, ci),
            )
            add(
                chapter_node_key(NODE_NATURALIZE, ci),
                chapter_fn(prescan.naturalize_fingerprint, ci),
            )
            add(
                chapter_node_key(NODE_REVIEW, ci),
                chapter_fn(prescan.review_fingerprint, ci),
            )
            add(
                chapter_node_key(NODE_BACKTRANSLATE, ci),
                chapter_fn(prescan.backtranslate_fingerprint, ci),
            )
        add(NODE_MINE_TERMS, prescan.mine_fingerprint)
        digests = self._digests(state, store)
        add(
            NODE_BOOK_SYNOPSIS,
            (lambda: prescan.synopsis_fingerprint(digests))
            if prescan.synopsis_fingerprint is not None
            else None,
        )
        add(NODE_TITLES, prescan.titles_fingerprint)
        add(NODE_CONSISTENCY_QA, prescan.consistency_fingerprint)
        add(NODE_REPORT, prescan.report_fingerprint)
        add(NODE_ASSEMBLE, prescan.assemble_fingerprint)
        return computed

    # ── 策略禁用 ──────────────────────────────────────────────────────────
    @staticmethod
    def _disabled_nodes(policy: WorkflowPolicy) -> set[str]:
        disabled: set[str] = set()
        if not policy.book_understanding:
            disabled |= {
                NODE_DIGEST,
                NODE_MINE_TERMS,
                NODE_NAME_TERMS,
                NODE_BOOK_SYNOPSIS,
            }
        if not policy.review:
            disabled.add(NODE_REVIEW)
        if not policy.naturalize:
            disabled.add(NODE_NATURALIZE)
        if not policy.polish:
            disabled.add(NODE_POLISH)
        if not policy.consistency_qa:
            disabled.add(NODE_CONSISTENCY_QA)
        return disabled

    @staticmethod
    def _run_entry(node_id: str, ci: int | None) -> PlanEntry:
        key = chapter_node_key(node_id, ci) if ci is not None else node_id
        scope = SCOPE_CHAPTER if ci is not None else SCOPE_BOOK
        return PlanEntry(node_id=node_id, key=key, ci=ci, scope=scope)

    @staticmethod
    def _skip_entry(node_id: str, ci: int | None) -> PlanEntry:
        key = chapter_node_key(node_id, ci) if ci is not None else node_id
        scope = SCOPE_CHAPTER if ci is not None else SCOPE_BOOK
        return PlanEntry(node_id=node_id, key=key, ci=ci, scope=scope, action="skip")

    # ── 附属章分类 ────────────────────────────────────────────────────────
    @staticmethod
    def _back_matter_mode(chapter, total: int, policy: WorkflowPolicy) -> str | None:
        """skip/light 且标题+位置命中时返回该档；full 或未命中返回 None（完整流水线）。"""
        mode = policy.back_matter
        if mode in ("skip", "light") and is_back_matter(
            chapter.title, index=chapter.index, total=total
        ):
            return mode
        return None

    @staticmethod
    def _body_chapter(chapter, total: int, policy: WorkflowPolicy) -> bool:
        """正文章（进 digest/mine 输入集合）：不受 back_matter=full 影响，只用
        is_back_matter 排除（与迁移前 _build_understanding 口径一致）。"""
        return not is_back_matter(chapter.title, index=chapter.index, total=total)


__all__ = [
    "PlanEntry",
    "PlannedStage",
    "Planner",
    "PrescanInputs",
    "WorkflowPlan",
    "WorkflowPolicy",
]
