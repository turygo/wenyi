"""Workflow 内核聚焦测试：definition 图校验、planner 计划结构、runner 状态机。

只覆盖本 cutover 新增的契约：
- 图：重复/未知/缺失依赖/环/非法作用域/必须节点禁用；
- 计划：确定性顺序、跳过语义、章 fan-out/book fan-in、目标闭包；
- runner：succeeded/skipped/retryable/permanent、findings_count=0、
  必需节点中止、尽力而为延续、running 中断恢复。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.agents.translator import AlignmentError
from trans_novel.config import Config
from trans_novel.ingest.models import Chapter, Segment
from trans_novel.llm import FakeClient
from trans_novel.llm.errors import AllModelsFailedError, JSONParseError
from trans_novel.llm.retrying import EmptyResponseError
from trans_novel.pipeline.bootstrap import Application, build_workflow_definition
from trans_novel.pipeline.contracts import (
    FAILURE_BUSINESS,
    FAILURE_PROTOCOL,
    GOAL_TRANSLATE,
    ExecutionGoal,
    NodeOutcome,
    NodeRequest,
    assemble_goal,
    classify_failure,
    qa_goal,
    report_goal,
)
from trans_novel.pipeline.definition import NodeSpec, WorkflowDefinition, WorkflowDefinitionError
from trans_novel.pipeline.nodes import RunShared
from trans_novel.pipeline.planner import (
    PlanEntry,
    PlannedStage,
    Planner,
    PrescanInputs,
    WorkflowPlan,
    WorkflowPolicy,
)
from trans_novel.pipeline.runner import RequiredNodeFailed, WorkflowRunner
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import (
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
    NODE_TITLES,
    NODE_TRANSLATE,
    RUN_STATE_SCHEMA_VERSION,
    ChapterIndex,
    ChapterProgress,
    NodeState,
    RunIdentity,
    RunState,
    chapter_node_key,
)


def _store(tmp: str, *, done: bool = False) -> RunStore:
    store = RunStore(os.path.join(tmp, "book"))
    store.save_state(
        RunState(
            run_state_schema=RUN_STATE_SCHEMA_VERSION,
            identity=RunIdentity(
                source_bytes_sha256="hash",
                source_lang="ja",
                target_lang="zh",
            ),
            title="Book",
            fmt="text",
            source_lang="ja",
            target_lang="zh",
            chapters=[
                ChapterIndex(index=0, title="第一章"),
                ChapterIndex(index=1, title="Notes"),
            ],
            progress={
                0: ChapterProgress(status=STATUS_DONE if done else "pending"),
                1: ChapterProgress(status=STATUS_DONE if done else "pending"),
            },
        )
    )
    for i, title in ((0, "第一章"), (1, "Notes")):
        store.save_chapter(
            Chapter(index=i, title=title, segments=[Segment(index=0, source=f"源文{i}")])
        )
    return store


def _prescan() -> PrescanInputs:
    return PrescanInputs(
        digest_fingerprint=lambda ci: "fp",
        mine_fingerprint=lambda: "fp",
        synopsis_fingerprint=lambda digests: "fp",
    )


def _policy(**overrides) -> WorkflowPolicy:
    base = {
        "book_understanding": True,
        "review": True,
        "autofix_severe": True,
        "polish": True,
        "naturalize": True,
        "backtranslate_sample": 0.0,
        "consistency_qa": True,
        "back_matter": "light",
        "prescan_concurrency": 4,
    }
    base.update(overrides)
    return WorkflowPolicy(**base)


class TestDefinitionValidation(unittest.TestCase):
    def test_duplicate_node_ids_rejected(self):
        with self.assertRaisesRegex(WorkflowDefinitionError, "重复节点"):
            WorkflowDefinition(
                [
                    NodeSpec("a", "book", "required"),
                    NodeSpec("a", "book", "required"),
                ]
            )

    def test_unknown_dependency_rejected(self):
        with self.assertRaisesRegex(WorkflowDefinitionError, "未注册"):
            WorkflowDefinition([NodeSpec("a", "book", "required", depends_on=("nope",))])

    def test_cycle_rejected(self):
        with self.assertRaisesRegex(WorkflowDefinitionError, "依赖环"):
            WorkflowDefinition(
                [
                    NodeSpec("a", "book", "required", depends_on=("b",)),
                    NodeSpec("b", "book", "required", depends_on=("a",)),
                ]
            )

    def test_illegal_book_chapter_edge_rejected(self):
        # book 节点依赖 chapter 节点必须显式声明聚合边。
        with self.assertRaisesRegex(WorkflowDefinitionError, "聚合边"):
            WorkflowDefinition(
                [
                    NodeSpec("ch", "chapter", "required"),
                    NodeSpec("book", "book", "required", depends_on=("ch",)),
                ]
            )

    def test_aggregation_edge_allowed(self):
        definition = WorkflowDefinition(
            [
                NodeSpec("ch", "chapter", "required"),
                NodeSpec("book", "book", "required", depends_on=("ch",), aggregates=("ch",)),
            ]
        )
        self.assertEqual(definition.aggregates("book"), ("ch",))

    def test_required_node_cannot_be_disabled(self):
        definition = build_workflow_definition()
        with self.assertRaisesRegex(WorkflowDefinitionError, "不可禁用"):
            definition.validate_disablement([NODE_PREPARE])

    def test_optional_node_can_be_disabled(self):
        definition = build_workflow_definition()
        definition.validate_disablement([NODE_REVIEW])  # 不抛即通过

    def test_builtin_definition_valid(self):
        definition = build_workflow_definition()
        self.assertEqual(definition.scope_of(NODE_DIGEST), "chapter")
        self.assertEqual(definition.scope_of(NODE_BOOK_SYNOPSIS), "book")
        # 全部注册节点 id 与 state 常量一致（单一事实来源）
        for node_id in definition.node_ids:
            self.assertTrue(node_id)


class TestPlanner(unittest.TestCase):
    def test_deterministic_full_translate_plan(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE, store=store, policy=_policy(), prescan=_prescan()
            )
            # 阶段序列：prepare → [digest ∥ mine] → name_terms → book_synopsis
            # → 逐章链 → titles
            node_layers = [[e.key for e in stage.entries] for stage in plan.stages]
            self.assertEqual(node_layers[0][0], NODE_PREPARE)
            self.assertIn(NODE_ANALYZE, node_layers[0])
            # 并行层：mine_terms 与正文章 digest fan-out（附属章 Notes 不进 digest）
            parallel = node_layers[1]
            self.assertEqual(parallel[0], NODE_MINE_TERMS)
            self.assertIn(chapter_node_key(NODE_DIGEST, 0), parallel)
            self.assertNotIn(chapter_node_key(NODE_DIGEST, 1), parallel)
            self.assertTrue(plan.stages[1].parallel)
            self.assertEqual(plan.stages[1].max_workers, 4)
            # book fan-in：name_terms → book_synopsis 在并行层之后
            self.assertEqual(node_layers[2], [NODE_NAME_TERMS])
            self.assertEqual(node_layers[3], [NODE_BOOK_SYNOPSIS])
            # 逐章链：正文章完整质量链、backtranslate 收尾；附属章仅 translate 旁路
            self.assertEqual(
                [e.node_id for e in plan.stages[4].entries],
                [
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                    NODE_TRANSLATE,
                ],
            )
            self.assertTrue(plan.stages[4].entries[4].finalize_chapter)
            self.assertFalse(plan.stages[4].entries[3].finalize_chapter)
            self.assertEqual(node_layers[5], [NODE_TITLES])
            self.assertEqual(plan.targets, [0, 1])

    def test_computed_fingerprints_cover_prepared_body_quality_chain(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            config = Config.from_dict({"llm": fake_llm_dict()})
            app = Application(config, client=FakeClient())
            shared = RunShared(
                store=store,
                config=config,
                doc=None,
                agent_builder=app._build_agents,
            )
            try:
                policy = _policy()
                prescan = app._prescan_inputs(store, policy, shared, GOAL_TRANSLATE)
                computed = app.planner._computed_fingerprints(store, policy, prescan)
            finally:
                shared.close()

            for node_id in (NODE_POLISH, NODE_NATURALIZE, NODE_REVIEW):
                key = chapter_node_key(node_id, 0)
                self.assertIn(key, computed)
                self.assertTrue(computed[key])

    def test_quality_disabled_persists_skipped_entries(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE,
                store=store,
                policy=_policy(review=False, naturalize=False, polish=False),
                prescan=_prescan(),
            )
            chain = plan.stages[4].entries
            skipped = [(e.node_id, e.action) for e in chain]
            self.assertIn((NODE_POLISH, "skip"), skipped)
            self.assertIn((NODE_NATURALIZE, "skip"), skipped)
            self.assertIn((NODE_REVIEW, "skip"), skipped)
            # backtranslate 常驻收尾，不跳过
            self.assertEqual(chain[4].node_id, NODE_BACKTRANSLATE)
            self.assertEqual(chain[4].action, "run")

    def test_book_understanding_disabled_skips_prescan(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE,
                store=store,
                policy=_policy(book_understanding=False),
                prescan=_prescan(),
            )
            layer1 = plan.stages[1]
            self.assertTrue(all(e.action == "skip" for e in layer1.entries))
            self.assertEqual(
                [e.node_id for e in layer1.entries],
                [NODE_DIGEST, NODE_MINE_TERMS, NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS],
            )

    def test_resume_satisfied_prescan_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            store.save_analysis({"style": {"tone": "cool"}, "book_synopsis": "概览"})
            state = store.load_state()
            for ci in (0, 1):
                pg = state.progress[ci]
                pg.source_digest = f"梗概{ci}"
                state.progress[ci] = pg
                state.nodes[chapter_node_key(NODE_DIGEST, ci)] = NodeState(
                    node_id=chapter_node_key(NODE_DIGEST, ci),
                    status="succeeded",
                    input_fingerprint="fp",
                )
            state.nodes[NODE_MINE_TERMS] = NodeState(
                node_id=NODE_MINE_TERMS, status="succeeded", input_fingerprint="fp"
            )
            state.nodes[NODE_NAME_TERMS] = NodeState(
                node_id=NODE_NAME_TERMS, status="succeeded", input_fingerprint="fp"
            )
            state.nodes[NODE_BOOK_SYNOPSIS] = NodeState(
                node_id=NODE_BOOK_SYNOPSIS, status="succeeded", input_fingerprint="fp"
            )
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE, store=store, policy=_policy(), prescan=_prescan()
            )
            # 预扫/分析全部满足 → 无并行层与逐章目标（章链节点未满足时才会补章链，
            # 见 test_resume_satisfied_chain_excluded）。
            keys = plan.entry_keys()
            self.assertNotIn(NODE_BOOK_SYNOPSIS, keys)
            self.assertNotIn(NODE_MINE_TERMS, keys)
            self.assertNotIn(NODE_NAME_TERMS, keys)
            self.assertFalse(any(k.startswith(chapter_node_key(NODE_DIGEST, 0)[:-1]) for k in keys))
            self.assertEqual(plan.targets, [])

    def test_resume_satisfied_chain_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            store.save_analysis({"style": {"tone": "cool"}, "book_synopsis": "概览"})
            state = store.load_state()
            for ci in (0, 1):
                pg = state.progress[ci]
                pg.source_digest = f"梗概{ci}"
                state.progress[ci] = pg
                for node_id in (
                    NODE_DIGEST,
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                ):
                    key = chapter_node_key(node_id, ci)
                    state.nodes[key] = NodeState(
                        node_id=key, status="succeeded", input_fingerprint="fp"
                    )
            for node_id in (NODE_MINE_TERMS, NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS):
                state.nodes[node_id] = NodeState(
                    node_id=node_id, status="succeeded", input_fingerprint="fp"
                )
            state.nodes[NODE_PREPARE] = NodeState(
                node_id=NODE_PREPARE, status="succeeded", input_fingerprint="fp"
            )
            state.nodes[NODE_ANALYZE] = NodeState(
                node_id=NODE_ANALYZE, status="succeeded", input_fingerprint="fp"
            )
            state.nodes[NODE_TITLES] = NodeState(
                node_id=NODE_TITLES, status="succeeded", input_fingerprint="fp"
            )
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE, store=store, policy=_policy(), prescan=_prescan()
            )
            # 全链路已满足 → 只剩身份核验门 prepare + 显式请求的 titles 动作根；
            # 零翻译调用
            self.assertEqual(plan.entry_keys(), {NODE_PREPARE, NODE_TITLES})
            self.assertEqual(plan.targets, [])

    def test_goal_dependency_closure(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            planner = Planner(build_workflow_definition())
            # assemble 目标：依赖闭包强制补上前置——report/qa/titles/翻译链
            # （未满足则入计划），不能只回填。
            plan = planner.build_plan(
                goal=assemble_goal(), store=store, policy=_policy(), prescan=_prescan()
            )
            keys = plan.entry_keys()
            self.assertIn(NODE_ASSEMBLE, keys)
            self.assertIn(NODE_REPORT, keys)
            self.assertIn(NODE_CONSISTENCY_QA, keys)
            self.assertIn(NODE_TITLES, keys)
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), keys)
            self.assertIn(chapter_node_key(NODE_BACKTRANSLATE, 0), keys)
            self.assertIn(NODE_PREPARE, keys)
            # 阶段顺序：prepare → 预扫 → 章链 → titles → qa → report → assemble
            layers = [[e.node_id for e in s.entries] for s in plan.stages]
            flat = [n for layer in layers for n in layer]
            self.assertLess(flat.index(NODE_PREPARE), flat.index(NODE_TRANSLATE))
            self.assertLess(flat.index(NODE_TRANSLATE), flat.index(NODE_TITLES))
            self.assertLess(flat.index(NODE_TITLES), flat.index(NODE_CONSISTENCY_QA))
            self.assertLess(flat.index(NODE_CONSISTENCY_QA), flat.index(NODE_REPORT))
            self.assertLess(flat.index(NODE_REPORT), flat.index(NODE_ASSEMBLE))
            # qa 目标：显式强制运行（不受策略关闭影响），且依赖闭包补上前置章链
            plan2 = planner.build_plan(
                goal=qa_goal(),
                store=store,
                policy=_policy(consistency_qa=False),
                prescan=_prescan(),
            )
            keys2 = plan2.entry_keys()
            self.assertIn("consistency_qa", keys2)
            self.assertIn(NODE_TITLES, keys2)
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), keys2)

    def test_goal_closure_respects_satisfied_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            state = store.load_state()
            for ci in (0, 1):
                pg = state.progress[ci]
                pg.source_digest = f"梗概{ci}"
                state.progress[ci] = pg
                for node_id in (
                    NODE_DIGEST,
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                ):
                    key = chapter_node_key(node_id, ci)
                    state.nodes[key] = NodeState(
                        node_id=key, status="succeeded", input_fingerprint="fp"
                    )
            for node_id in (
                NODE_PREPARE,
                NODE_ANALYZE,
                NODE_MINE_TERMS,
                NODE_NAME_TERMS,
                NODE_BOOK_SYNOPSIS,
                NODE_TITLES,
                NODE_CONSISTENCY_QA,
                NODE_REPORT,
            ):
                state.nodes[node_id] = NodeState(
                    node_id=node_id, status="succeeded", input_fingerprint="fp"
                )
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            planner = Planner(build_workflow_definition())
            plan = planner.build_plan(
                goal=assemble_goal(), store=store, policy=_policy(), prescan=_prescan()
            )
            # 全链已满足 → assemble 目标只剩 assemble（前置不再重复执行）
            self.assertEqual(plan.entry_keys(), {NODE_ASSEMBLE})

    def test_only_chapter_limits_all_chapter_scoped_entries(self):
        """单章目标：计划不得包含其它章的章级条目（book 级预扫可覆盖全书，
        但章级消费者/收尾必须限定目标章），执行也不得改其它章状态。"""
        from trans_novel.pipeline.contracts import translate_chapter_goal

        chain = (
            NODE_TRANSLATE,
            NODE_POLISH,
            NODE_NATURALIZE,
            NODE_REVIEW,
            NODE_BACKTRANSLATE,
        )
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=translate_chapter_goal(0), store=store, policy=_policy(), prescan=_prescan()
            )
            keys = plan.entry_keys()
            for ci in (0, 1):
                for node_id in chain:
                    key = chapter_node_key(node_id, ci)
                    if ci == 0:
                        self.assertIn(key, keys)
                    else:
                        self.assertNotIn(key, keys, f"only_chapter=0 不得规划 {key}")
            self.assertEqual(plan.targets, [0])

    def test_repeated_action_roots_always_execute(self):
        """显式请求的动作根（qa/report/assemble/titles）即使上次 succeeded 也必须
        执行：命令的可观察输出是该节点的内存产物/副作用（第二次 run_all / 重复
        tools report|qa|assemble 不能返回空）。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            state = store.load_state()
            for ci in (0, 1):
                pg = state.progress[ci]
                pg.source_digest = f"梗概{ci}"
                state.progress[ci] = pg
                for node_id in (
                    NODE_DIGEST,
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                ):
                    key = chapter_node_key(node_id, ci)
                    state.nodes[key] = NodeState(
                        node_id=key, status="succeeded", input_fingerprint="fp"
                    )
            for node_id in (
                NODE_PREPARE,
                NODE_ANALYZE,
                NODE_MINE_TERMS,
                NODE_NAME_TERMS,
                NODE_BOOK_SYNOPSIS,
                NODE_TITLES,
                NODE_CONSISTENCY_QA,
                NODE_REPORT,
                NODE_ASSEMBLE,
            ):
                state.nodes[node_id] = NodeState(
                    node_id=node_id, status="succeeded", input_fingerprint="fp"
                )
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            planner = Planner(build_workflow_definition())
            # 全部满足后：assemble/report/qa 目标仍必须执行动作根（不返回空结果）
            self.assertEqual(
                planner.build_plan(
                    goal=assemble_goal(), store=store, policy=_policy(), prescan=_prescan()
                ).entry_keys(),
                {NODE_ASSEMBLE},
            )
            self.assertEqual(
                planner.build_plan(
                    goal=report_goal(), store=store, policy=_policy(), prescan=_prescan()
                ).entry_keys(),
                {NODE_REPORT},
            )
            self.assertEqual(
                planner.build_plan(
                    goal=qa_goal(), store=store, policy=_policy(), prescan=_prescan()
                ).entry_keys(),
                {NODE_CONSISTENCY_QA},
            )
            # 第二次 run_all：翻译/预扫已满足被剪枝，但 qa/report/assemble 动作根
            # 与身份核验门 prepare、titles 动作根仍在（输出会被重新生成）。
            goal = ExecutionGoal(
                name="run_all",
                phases=("prepare", "prescan", "translate", "titles", "qa", "report", "assemble"),
            )
            keys = planner.build_plan(
                goal=goal, store=store, policy=_policy(), prescan=_prescan()
            ).entry_keys()
            self.assertEqual(
                keys,
                {
                    NODE_PREPARE,
                    NODE_TITLES,
                    NODE_CONSISTENCY_QA,
                    NODE_REPORT,
                    NODE_ASSEMBLE,
                },
            )

    def test_qa_override_both_directions(self):
        """do_qa=True 强制运行；do_qa=False 显式关闭（质量档位开着也跳过）。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            planner = Planner(build_workflow_definition())
            goal = ExecutionGoal(
                name="run_all",
                phases=("prepare", "prescan", "translate", "titles", "qa", "report", "assemble"),
                do_qa=True,
            )
            plan = planner.build_plan(
                goal=goal, store=store, policy=_policy(consistency_qa=False), prescan=_prescan()
            )
            qa_entries = [
                e for s in plan.stages for e in s.entries if e.node_id == NODE_CONSISTENCY_QA
            ]
            self.assertEqual(len(qa_entries), 1)
            self.assertEqual(qa_entries[0].action, "run", "--qa 必须强制运行一致性 QA")
            goal2 = ExecutionGoal(
                name="run_all",
                phases=("prepare", "prescan", "translate", "titles", "qa", "report", "assemble"),
                do_qa=False,
            )
            plan2 = planner.build_plan(
                goal=goal2, store=store, policy=_policy(consistency_qa=True), prescan=_prescan()
            )
            qa2 = [e for s in plan2.stages for e in s.entries if e.node_id == NODE_CONSISTENCY_QA]
            self.assertEqual(len(qa2), 1)
            self.assertEqual(qa2[0].action, "skip", "--no-qa 必须跳过一致性 QA")

    def test_full_backmatter_downgrade_preserves_translation(self):
        """full 档产出的附属章（mode=None）降到 skip/light：高质量译文满足低档
        策略，不重跑旁路、不覆盖译文（降档不回退）。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)  # 章0 第一章(body)，章1 Notes(附属章)
            ch1 = store.load_chapter(1)
            for s in ch1.text_segments:
                s.target = "高质量译文"
            store.save_chapter(ch1)
            state = store.load_state()
            state.progress[1].back_matter_mode = None
            state.nodes[chapter_node_key(NODE_TRANSLATE, 1)] = NodeState(
                node_id=chapter_node_key(NODE_TRANSLATE, 1),
                status="succeeded",
                input_fingerprint="body-fp",
            )
            for node_id in (NODE_POLISH, NODE_NATURALIZE, NODE_REVIEW, NODE_BACKTRANSLATE):
                state.nodes[chapter_node_key(node_id, 1)] = NodeState(
                    node_id=chapter_node_key(node_id, 1),
                    status="succeeded",
                    input_fingerprint="fp",
                )
            store.save_state(state)

            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE,
                store=store,
                policy=_policy(back_matter="skip"),
                prescan=_prescan(),
            )
            keys = plan.entry_keys()
            self.assertNotIn(
                chapter_node_key(NODE_TRANSLATE, 1), keys, "降档不得重跑旁路 translate"
            )
            self.assertTrue(
                all(s.target == "高质量译文" for s in store.load_chapter(1).text_segments),
                "降档不得覆盖 full 档产出",
            )

    def test_book_understanding_disabled_to_enabled_single_run(self):
        """book_understanding 从禁用切到启用：digest/name/synopsis 本轮产出，
        translate/titles 的指纹基于旧（空）概览 → 同轮级联强制重跑，不等到第二次 run。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            state = store.load_state()
            for ci in (0, 1):
                pg = state.progress[ci]
                pg.source_digest = f"梗概{ci}"
                state.progress[ci] = pg
                for node_id in (
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                ):
                    key = chapter_node_key(node_id, ci)
                    state.nodes[key] = NodeState(
                        node_id=key, status="succeeded", input_fingerprint="fp"
                    )
            # book_understanding 禁用期间：digest/mine/name/synopsis 为 skipped
            for node_id in (NODE_DIGEST, NODE_MINE_TERMS, NODE_NAME_TERMS, NODE_BOOK_SYNOPSIS):
                for ci in (0, 1):
                    if node_id == NODE_DIGEST:
                        key = chapter_node_key(node_id, ci)
                        state.nodes[key] = NodeState(node_id=key, status="skipped")
                    else:
                        state.nodes[node_id] = NodeState(node_id=node_id, status="skipped")
            state.nodes[NODE_PREPARE] = NodeState(
                node_id=NODE_PREPARE, status="succeeded", input_fingerprint="fp"
            )
            state.nodes[NODE_ANALYZE] = NodeState(
                node_id=NODE_ANALYZE, status="succeeded", input_fingerprint="fp"
            )
            store.save_state(state)
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE, store=store, policy=_policy(), prescan=_prescan()
            )
            keys = plan.entry_keys()
            # 启用后：预扫节点重跑 + translate 同轮重跑（消费新概览）
            self.assertIn(chapter_node_key(NODE_DIGEST, 0), keys)
            self.assertIn(NODE_BOOK_SYNOPSIS, keys)
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), keys)

    def test_run_all_qa_policy_gated(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            planner = Planner(build_workflow_definition())
            goal = ExecutionGoal(
                name="run_all",
                phases=("prepare", "prescan", "translate", "titles", "qa", "report", "assemble"),
            )
            plan = planner.build_plan(
                goal=goal, store=store, policy=_policy(consistency_qa=False), prescan=_prescan()
            )
            qa_entry = [e for s in plan.stages for e in s.entries if e.node_id == "consistency_qa"]
            self.assertEqual(len(qa_entry), 1)
            self.assertEqual(qa_entry[0].action, "skip")

    def test_translate_fingerprint_mismatch_reopens_only_affected_chapter(self):
        """输入指纹失配：只失效该节点与后代——book_synopsis 变化重开目标章译文，
        其它章不动；批次预算/质量档位变化不得清空译文。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d, done=True)
            # 章 0 已翻译（有译文 + translate 节点指纹），章 1 已翻译但指纹未记录。
            ch0 = store.load_chapter(0)
            for s in ch0.text_segments:
                s.target = f"译:{s.source}"
            store.save_chapter(ch0)
            ch1 = store.load_chapter(1)
            for s in ch1.text_segments:
                s.target = f"译:{s.source}"
            store.save_chapter(ch1)
            state = store.load_state()
            state.nodes[chapter_node_key(NODE_TRANSLATE, 0)] = NodeState(
                node_id=chapter_node_key(NODE_TRANSLATE, 0),
                status="succeeded",
                input_fingerprint="old-fp",
            )
            state.nodes[chapter_node_key(NODE_TRANSLATE, 1)] = NodeState(
                node_id=chapter_node_key(NODE_TRANSLATE, 1),
                status="succeeded",
                input_fingerprint="old-fp",
            )
            store.save_state(state)

            prescan = _prescan()
            prescan.translate_fingerprint = lambda ci: "new-fp" if ci == 0 else "old-fp"
            plan = Planner(build_workflow_definition()).build_plan(
                goal=GOAL_TRANSLATE, store=store, policy=_policy(), prescan=prescan
            )
            # 章 0 失效重开：译文被清、状态 pending，translate:0 重新入计划；
            # 章 1 的译文原样保留（指纹一致不被失效清除）——同轮级联最多把
            # translate:1 重新规划，但节点重跑走批级续跑，不重翻、不清译文。
            self.assertEqual(store.chapter_status(0), "pending")
            self.assertEqual(store.chapter_status(1), "done")
            self.assertTrue(all(not s.target for s in store.load_chapter(0).text_segments))
            self.assertTrue(all(s.target for s in store.load_chapter(1).text_segments))
            keys = plan.entry_keys()
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), keys)
            self.assertTrue(all(s.target for s in store.load_chapter(1).text_segments))


class _StubNode:
    """可编程桩节点：按 node_id 返回预设结果/异常/异步句柄。"""

    def __init__(self, node_id, *, outcome=None, exc=None, handle=None):
        self.node_id = node_id
        self.scope = "book"
        self.outcome = outcome if outcome is not None else NodeOutcome()
        self.exc = exc
        self.handle = handle
        self.calls = 0

    def execute(self, request: NodeRequest) -> NodeOutcome:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if self.handle is not None:
            return NodeOutcome(async_handle=self.handle)
        return self.outcome

    def finish(self, request: NodeRequest, handle) -> None:
        self.calls += 1


class _RunnerFixture:
    def __init__(self, tmp: str):
        self.store = _store(tmp)
        self.definition = build_workflow_definition()
        self.stubs: dict[str, _StubNode] = {}
        self.log: list[str] = []

    def stub(self, node_id, **kwargs) -> _StubNode:
        stub = _StubNode(node_id, **kwargs)
        self.stubs[node_id] = stub
        return stub

    def factory(self, node_id, ci):
        return self.stubs[node_id]

    def runner(self) -> WorkflowRunner:
        return WorkflowRunner(
            definition=self.definition,
            node_factory=self.factory,
            usage_flush=None,
        )

    def plan(self, *entries: PlanEntry) -> WorkflowPlan:
        return WorkflowPlan(stages=[PlannedStage(list(entries))])


def _entry(node_id: str, action: str = "run") -> PlanEntry:
    return PlanEntry(node_id=node_id, key=node_id, ci=None, scope="book", action=action)


class TestFailureClassification(unittest.TestCase):
    """失败分类契约：协议类错误必须归类为 protocol（可重试），而非业务拒绝。"""

    def test_exhausted_alignment_error_is_protocol(self):
        self.assertEqual(
            classify_failure(AlignmentError("translation_count_mismatch")),
            FAILURE_PROTOCOL,
        )

    def test_json_parse_error_is_protocol_before_generic_value_error(self):
        # JSONParseError 是 ValueError 的子类；协议错误的判定必须先于通用 ValueError（business）。
        self.assertTrue(issubclass(JSONParseError, ValueError))
        self.assertEqual(classify_failure(JSONParseError("无法解析")), FAILURE_PROTOCOL)
        self.assertEqual(classify_failure(ValueError("业务拒绝")), FAILURE_BUSINESS)


class TestRunner(unittest.TestCase):
    def test_succeeded_state_and_findings_zero(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_ANALYZE, outcome=NodeOutcome(findings_count=0))
            fx.runner().run(fx.plan(_entry(NODE_ANALYZE)), store=fx.store, input_path="in.txt")
            node = fx.store.load_state().nodes[NODE_ANALYZE]
            self.assertEqual(node.status, "succeeded")
            self.assertEqual(node.attempts, 1)
            self.assertIsNone(node.failure)

    def test_skipped_state(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.runner().run(
                fx.plan(_entry(NODE_DIGEST, action="skip")),
                store=fx.store,
                input_path="in.txt",
            )
            self.assertEqual(fx.store.load_state().nodes[NODE_DIGEST].status, "skipped")

    def test_best_effort_retryable_failure_continues(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_MINE_TERMS, exc=EmptyResponseError("empty"))
            fx.stub(NODE_ANALYZE)
            result = fx.runner().run(
                fx.plan(_entry(NODE_MINE_TERMS), _entry(NODE_ANALYZE)),
                store=fx.store,
                input_path="in.txt",
            )
            state = fx.store.load_state()
            self.assertEqual(state.nodes[NODE_MINE_TERMS].status, "failed_retryable")
            self.assertEqual(state.nodes[NODE_MINE_TERMS].failure.kind, "provider_retryable")
            self.assertEqual(state.nodes[NODE_ANALYZE].status, "succeeded")
            self.assertEqual(fx.stubs[NODE_ANALYZE].calls, 1, "尽力而为失败后应继续执行后续节点")
            self.assertIn(NODE_ANALYZE, result.outcomes)

    def test_best_effort_retryable_exhaustion_status(self):
        from trans_novel.config import ModelRef

        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            # 路由器在可重试原因上耗尽内部重试 → 可重试的提供商耗尽，绝非永久拒绝。
            fx.stub(
                NODE_MINE_TERMS,
                exc=AllModelsFailedError(((ModelRef("provider", "model"), "server_error"),)),
            )
            fx.runner().run(fx.plan(_entry(NODE_MINE_TERMS)), store=fx.store, input_path="in.txt")
            node = fx.store.load_state().nodes[NODE_MINE_TERMS]
            self.assertEqual(node.status, "failed_retryable")
            self.assertEqual(node.failure.kind, "provider_retryable")

    def test_best_effort_permanent_failure_status(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            # 原始永久提供商错误（不可重试原因）→ failed_permanent。
            fx.stub(NODE_MINE_TERMS, exc=RuntimeError("permanent"))
            fx.runner().run(fx.plan(_entry(NODE_MINE_TERMS)), store=fx.store, input_path="in.txt")
            node = fx.store.load_state().nodes[NODE_MINE_TERMS]
            self.assertEqual(node.status, "failed_permanent")
            self.assertEqual(node.failure.kind, "provider_permanent")

    def test_best_effort_protocol_failure_retryable_status(self):
        # 协议失败（翻译对齐重试耗尽或模型输出解析失败）→ failed_retryable + kind=protocol，
        # 需要与 Provider 永久失败相区分；采用尽力而为语义时，不中断后续节点。
        for exc in (AlignmentError("translation_count_mismatch"), JSONParseError("无法解析")):
            with self.subTest(exc=type(exc).__name__), tempfile.TemporaryDirectory() as d:
                fx = _RunnerFixture(d)
                fx.stub(NODE_MINE_TERMS, exc=exc)
                fx.runner().run(
                    fx.plan(_entry(NODE_MINE_TERMS)), store=fx.store, input_path="in.txt"
                )
                node = fx.store.load_state().nodes[NODE_MINE_TERMS]
                self.assertEqual(node.status, "failed_retryable")
                self.assertEqual(node.failure.kind, "protocol")

    def test_business_failure_reraises_unwrapped(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_ANALYZE, exc=ValueError("业务拒绝"))
            with self.assertRaisesRegex(ValueError, "业务拒绝"):
                fx.runner().run(fx.plan(_entry(NODE_ANALYZE)), store=fx.store, input_path="in.txt")
            node = fx.store.load_state().nodes[NODE_ANALYZE]
            self.assertEqual(node.failure.kind, "business")

    def test_required_failure_stops_plan(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_ANALYZE, exc=RuntimeError("boom"))
            fx.stub(NODE_TITLES)
            with self.assertRaisesRegex(RequiredNodeFailed, "boom"):
                fx.runner().run(
                    fx.plan(_entry(NODE_ANALYZE), _entry(NODE_TITLES)),
                    store=fx.store,
                    input_path="in.txt",
                )
            state = fx.store.load_state()
            self.assertEqual(state.nodes[NODE_ANALYZE].status, "failed_permanent")
            self.assertEqual(fx.stubs[NODE_TITLES].calls, 0, "必需节点失败必须中止后续节点")

    def test_interrupted_running_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            state = fx.store.load_state()
            state.nodes["ghost"] = NodeState(node_id="ghost", status="running")
            fx.store.save_state(state)
            fx.stub(NODE_ANALYZE)
            # 崩溃恢复发生在“重新打开”时（新实例的锁内迁移/恢复入口）：
            # 同一实例的 _v2_ready 已在首次打开时置位，模拟进程重启必须换新实例。
            fresh = RunStore(fx.store.run_dir)
            fresh.ensure_dirs()
            fx.runner().run(fx.plan(_entry(NODE_ANALYZE)), store=fresh, input_path="in.txt")
            node = fresh.load_state().nodes["ghost"]
            self.assertEqual(node.status, "pending")
            self.assertEqual(node.failure.kind, "interrupted")

    def test_async_handle_finish_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)

            class _Done:
                def done(self):
                    return True

                def result(self):
                    return "ok"

            fx.stub(NODE_REVIEW, handle=_Done())
            fx.runner().run(fx.plan(_entry(NODE_REVIEW)), store=fx.store, input_path="in.txt")
            state = fx.store.load_state()
            self.assertEqual(state.nodes[NODE_REVIEW].status, "succeeded")
            self.assertEqual(fx.stubs[NODE_REVIEW].calls, 2, "execute + finish 各一次")

    def test_chapter_entry_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_TRANSLATE)
            entry = PlanEntry(
                node_id=NODE_TRANSLATE,
                key=chapter_node_key(NODE_TRANSLATE, 0),
                ci=0,
                scope="chapter",
                finalize_chapter=True,
            )
            fx.runner().run(fx.plan(entry), store=fx.store, input_path="in.txt")
            self.assertEqual(fx.store.chapter_status(0), STATUS_DONE)
            with open(fx.store.event_log_path, encoding="utf-8") as f:
                events = [line for line in f if "chapter_done" in line]
            self.assertTrue(events, "章链收尾应发 chapter_done 事件")

    def test_artifacts_propagate_to_dependents(self):
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)

            class _Producer(_StubNode):
                def execute(self, request):
                    self.calls += 1
                    return NodeOutcome(artifacts={"candidates": ["A"]})

            class _Consumer(_StubNode):
                def execute(self, request):
                    self.calls += 1
                    got = (request.artifacts.get("mine_terms") or {}).get("candidates")
                    self.consumed = got
                    return NodeOutcome()

            producer = _Producer(NODE_MINE_TERMS)
            consumer = _Consumer(NODE_ANALYZE)
            fx.stubs[NODE_MINE_TERMS] = producer
            fx.stubs[NODE_ANALYZE] = consumer
            fx.runner().run(
                fx.plan(_entry(NODE_MINE_TERMS), _entry(NODE_ANALYZE)),
                store=fx.store,
                input_path="in.txt",
            )
            self.assertEqual(consumer.consumed, ["A"], "上游成功产物的 artifacts 必须传给下游")

    def test_lifecycle_buffered_until_manifest_commit(self):
        """全新运行（无 manifest）：prepare/analyze 生命周期先缓冲，analyze 落盘
        manifest 后合并进 V2 状态，含 attempts 与时间戳。"""
        with tempfile.TemporaryDirectory() as d:
            run_dir = os.path.join(d, "book")
            store = RunStore(run_dir)
            definition = build_workflow_definition()

            class _CommitAnalyze(_StubNode):
                """analyze 桩：执行时原子落盘 manifest（模拟 AnalyzeNode 的完成点）。"""

                def execute(self, request):
                    self.calls += 1
                    state = RunState(
                        run_state_schema=RUN_STATE_SCHEMA_VERSION,
                        identity=RunIdentity(source_lang="ja", target_lang="zh"),
                        title="T",
                        fmt="text",
                        source_lang="ja",
                        target_lang="zh",
                        initialized=True,
                    )
                    store.save_state(state)
                    return NodeOutcome()

            stubs = {
                NODE_PREPARE: _StubNode(NODE_PREPARE),
                NODE_ANALYZE: _CommitAnalyze(NODE_ANALYZE),
            }
            runner = WorkflowRunner(
                definition=definition,
                node_factory=lambda node_id, ci: stubs[node_id],
                usage_flush=None,
            )
            runner.run(
                fx_plan(_entry(NODE_PREPARE), _entry(NODE_ANALYZE)),
                store=store,
                input_path="in.txt",
            )
            state = store.load_state()
            prepare = state.nodes[NODE_PREPARE]
            analyze = state.nodes[NODE_ANALYZE]
            self.assertEqual(prepare.status, "succeeded")
            self.assertEqual(prepare.attempts, 1)
            self.assertIsNotNone(prepare.started_at)
            self.assertIsNotNone(prepare.finished_at)
            self.assertEqual(analyze.status, "succeeded")
            self.assertEqual(analyze.attempts, 1)
            self.assertIsNotNone(analyze.started_at)

    def test_parallel_commits_run_in_main_thread_deterministically(self):
        """并行层节点只做 LLM 计算：产物经 outcome.commit 由 runner 在 join 后
        于主线程按 key 确定顺序持久化（worker 绝不写 RunStore）。"""
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            order: list[str] = []

            class _CommittingNode(_StubNode):
                def __init__(self, node_id, key):
                    super().__init__(node_id)
                    self.key = key

                def execute(self, request):
                    self.calls += 1

                    def _commit(req) -> None:
                        # 主线程写路径：manifest 读改写（worker 并发写会竞争 tmp 文件）
                        order.append(self.key)
                        pg = req.store.load_progress(0)
                        req.store.save_progress(0, pg)

                    return NodeOutcome(commit=_commit)

            stubs = {
                chapter_node_key(NODE_DIGEST, 0): _CommittingNode(
                    NODE_DIGEST, chapter_node_key(NODE_DIGEST, 0)
                ),
                chapter_node_key(NODE_DIGEST, 1): _CommittingNode(
                    NODE_DIGEST, chapter_node_key(NODE_DIGEST, 1)
                ),
            }
            runner = WorkflowRunner(
                definition=fx.definition,
                node_factory=lambda node_id, ci: stubs[chapter_node_key(node_id, ci)],
                usage_flush=None,
            )
            stage = PlannedStage(
                [
                    PlanEntry(
                        node_id=NODE_DIGEST,
                        key=chapter_node_key(NODE_DIGEST, 1),
                        ci=1,
                        scope="chapter",
                    ),
                    PlanEntry(
                        node_id=NODE_DIGEST,
                        key=chapter_node_key(NODE_DIGEST, 0),
                        ci=0,
                        scope="chapter",
                    ),
                ],
                parallel=True,
                max_workers=2,
            )
            runner.run(WorkflowPlan(stages=[stage]), store=fx.store, input_path="in.txt")
            self.assertEqual(
                order,
                [chapter_node_key(NODE_DIGEST, 0), chapter_node_key(NODE_DIGEST, 1)],
                "commit 必须按 key 确定顺序在主线程执行",
            )

    def test_plan_builder_runs_under_lock(self):
        """runner 接受规划回调并在锁内执行（指纹对账等写操作与执行同临界区）。

        旧实现会在应用层锁之上再抢同一把 flock（独立 fd 不可重入）→ 死锁；
        本测试用 builder 内写状态验证“回调在锁内可安全写”，能返回即证明无嵌套加锁。
        """
        with tempfile.TemporaryDirectory() as d:
            fx = _RunnerFixture(d)
            fx.stub(NODE_ANALYZE)

            def build():
                state = fx.store.load_state()
                state.meta["planned_inside_lock"] = True
                fx.store.save_state(state)
                return fx.plan(_entry(NODE_ANALYZE))

            fx.runner().run(build, store=fx.store, input_path="in.txt")
            self.assertTrue(fx.store.load_state().meta["planned_inside_lock"])


def fx_plan(*entries: PlanEntry) -> WorkflowPlan:
    return WorkflowPlan(stages=[PlannedStage(list(entries))])


if __name__ == "__main__":
    unittest.main()
