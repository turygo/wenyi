"""组合根：唯一构造具体 Agent/节点/工作流定义/runner 的生产位置。

- 依赖注入：每个具体节点构造时收到精确依赖（AgentBundle / config / 目标参数）；
- 语言惰性解析：auto 检测后的源语言由 prepare 写入 RunShared，AgentBundle 在其后
  按不可变语言对构造（不恢复旧版全局可变 _apply_language 行为）；
- 应用门面（Application）暴露 CLI 需要的全部目标与服务：
  prepare / prepare_for_translation / run / run_all / run_goal_result /
  translate_titles / qa / report / assemble / naturalize / glossary_audit /
  flush_usage。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from trans_novel.agents.analyzer import Analyzer
from trans_novel.agents.glossary_auditor import GlossaryAuditor
from trans_novel.agents.naturalizer import Naturalizer, run_naturalize
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import Chapter, Document
from trans_novel.ingest.segmenter import load_document
from trans_novel.llm.base import LLMClient
from trans_novel.llm.factory import build_client
from trans_novel.llm.usage import merge_usage_summaries, usage_delta
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import (
    GOAL_PREPARE,
    GOAL_RUN_ALL,
    GOAL_TRANSLATE,
    BatchCommitHook,
    ExecutionGoal,
    assemble_goal,
    qa_goal,
    report_goal,
    titles_goal,
    translate_chapter_goal,
)
from trans_novel.pipeline.definition import NodeSpec, WorkflowDefinition
from trans_novel.pipeline.fingerprints import (
    analyze_input_fingerprint,
    assemble_input_fingerprint,
    back_matter_translate_input_fingerprint,
    backtranslate_input_fingerprint,
    consistency_input_fingerprint,
    editor_fast_model_profile,
    editor_model_profile,
    fast_model_profile,
    frozen_input_fingerprint,
    glossary_semantic_fingerprint_part,
    naturalize_input_fingerprint,
    polish_input_fingerprint,
    prepare_input_fingerprint,
    primary_fast_model_profile,
    primary_model_profile,
    report_input_fingerprint,
    review_input_fingerprint,
    titles_input_fingerprint,
    translate_input_fingerprint,
)
from trans_novel.pipeline.nodes import AgentBundle, RunShared
from trans_novel.pipeline.nodes.common import count_segments, sample_text
from trans_novel.pipeline.nodes.finish import (
    AssembleNode,
    ConsistencyQANode,
    ReportNode,
    TitlesNode,
)
from trans_novel.pipeline.nodes.prepare import AnalyzeNode, PrepareNode
from trans_novel.pipeline.nodes.prescan import (
    BookSynopsisNode,
    DigestNode,
    MineTermsNode,
    NameTermsNode,
    book_synopsis_input_fingerprint,
    digest_input_fingerprint,
    mine_terms_input_fingerprint,
)
from trans_novel.pipeline.nodes.quality import BacktranslateNode, NaturalizeNode, ReviewNode
from trans_novel.pipeline.nodes.translate import PolishNode, TranslateNode
from trans_novel.pipeline.planner import Planner, PrescanInputs, WorkflowPlan, WorkflowPolicy
from trans_novel.pipeline.runner import RunResult, WorkflowRunner
from trans_novel.pipeline.runstore import RunStore, slugify
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
    SCOPE_BOOK,
    SCOPE_CHAPTER,
    normalize_lang_code,
)

# 注册的全部内置节点（id 必须与 state.NODE_* 一致；planner/runner 均以这些
# 稳定 id 为契约，配置只能引用已注册 id）。
_NODE_SPECS = (
    NodeSpec(NODE_PREPARE, SCOPE_BOOK, "required"),
    NodeSpec(NODE_ANALYZE, SCOPE_BOOK, "required", depends_on=(NODE_PREPARE,)),
    NodeSpec(
        NODE_DIGEST,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_PREPARE,),
        optional=True,
    ),
    NodeSpec(
        NODE_MINE_TERMS,
        SCOPE_BOOK,
        "best_effort",
        depends_on=(NODE_PREPARE,),
        optional=True,
    ),
    NodeSpec(
        NODE_NAME_TERMS,
        SCOPE_BOOK,
        "best_effort",
        depends_on=(NODE_ANALYZE, NODE_DIGEST, NODE_MINE_TERMS),
        optional=True,
        aggregates=(NODE_DIGEST,),
    ),
    NodeSpec(
        NODE_BOOK_SYNOPSIS,
        SCOPE_BOOK,
        "required",
        depends_on=(NODE_ANALYZE, NODE_DIGEST, NODE_NAME_TERMS),
        optional=True,
        aggregates=(NODE_DIGEST,),
    ),
    NodeSpec(
        NODE_TRANSLATE,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_BOOK_SYNOPSIS,),
    ),
    NodeSpec(
        NODE_POLISH,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_TRANSLATE,),
        optional=True,
    ),
    NodeSpec(
        NODE_NATURALIZE,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_POLISH,),
        optional=True,
    ),
    NodeSpec(
        NODE_REVIEW,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_NATURALIZE,),
        optional=True,
    ),
    NodeSpec(
        NODE_BACKTRANSLATE,
        SCOPE_CHAPTER,
        "required",
        depends_on=(NODE_NATURALIZE, NODE_REVIEW),
    ),
    NodeSpec(
        NODE_TITLES,
        SCOPE_BOOK,
        "required",
        # 章级终端依赖：正文链以 backtranslate 收尾，附属章旁路以 translate 自理
        # 收尾——两条显式的策略相关终端分支，planner 按章选实际终端做 fan-in。
        depends_on=(NODE_BACKTRANSLATE, NODE_TRANSLATE),
        aggregates=(NODE_BACKTRANSLATE, NODE_TRANSLATE),
    ),
    NodeSpec(
        NODE_CONSISTENCY_QA,
        SCOPE_BOOK,
        "required",
        depends_on=(NODE_TITLES,),
        optional=True,
    ),
    NodeSpec(NODE_REPORT, SCOPE_BOOK, "required", depends_on=(NODE_CONSISTENCY_QA,)),
    NodeSpec(NODE_ASSEMBLE, SCOPE_BOOK, "required", depends_on=(NODE_REPORT,)),
)


def build_workflow_definition() -> WorkflowDefinition:
    return WorkflowDefinition(_NODE_SPECS)


class Application:
    """工作流应用门面：CLI 的唯一生产入口（组合根）。"""

    def __init__(
        self,
        config: Config,
        client: LLMClient | None = None,
        *,
        frozen_preparation=None,
        batch_commit_hook: BatchCommitHook | None = None,
        backtranslation_sample_scope: str = "",
    ):
        self.config = config
        self.client = client or build_client(config)
        # client 的统计是进程内累计；checkpoint 用于每次落盘时只提取新增部分。
        self._usage_checkpoint = self.client.usage_summary()
        self.frozen_preparation = frozen_preparation
        self.batch_commit_hook = batch_commit_hook
        self.backtranslation_sample_scope = backtranslation_sample_scope
        self.definition = build_workflow_definition()
        self.planner = Planner(self.definition)

    # ── 组合根：Agent/节点构造（唯一生产位置）──────────────────────────────
    def _build_agents(self, src: str, tgt: str) -> AgentBundle:
        return AgentBundle(client=self.client, config=self.config, src=src, tgt=tgt)

    def _node_factory(self, shared: RunShared, goal: ExecutionGoal):
        def _style() -> str:
            return shared.style_brief()

        builders: dict[str, Callable[[RunShared, int | None], Any]] = {
            NODE_PREPARE: lambda shared, ci: PrepareNode(
                client=self.client, config=self.config, doc=shared.doc
            ),
            NODE_ANALYZE: lambda shared, ci: AnalyzeNode(
                analyzer=shared.agents.analyzer,
                config=self.config,
                doc=shared.doc,
                glossary=shared.glossary(),
                frozen_book=shared.frozen_book(),
            ),
            NODE_DIGEST: lambda shared, ci: DigestNode(
                synopsizer=shared.agents.synopsizer,
                config=self.config,
                frozen_book=shared.frozen_book(),
            ),
            NODE_MINE_TERMS: lambda shared, ci: MineTermsNode(
                namer=shared.agents.namer,
                config=self.config,
                frozen_book=shared.frozen_book(),
            ),
            NODE_NAME_TERMS: lambda shared, ci: NameTermsNode(
                namer=shared.agents.namer,
                analyzer=shared.agents.analyzer,
                glossary=shared.glossary(),
                config=self.config,
                frozen_book=shared.frozen_book(),
            ),
            NODE_BOOK_SYNOPSIS: lambda shared, ci: BookSynopsisNode(
                synopsizer=shared.agents.synopsizer,
                analyzer=shared.agents.analyzer,
                glossary=shared.glossary(),
                config=self.config,
                frozen_book=shared.frozen_book(),
            ),
            NODE_TRANSLATE: lambda shared, ci: TranslateNode(
                translator=shared.agents.translator,
                extractor=shared.agents.extractor,
                polisher=shared.agents.polisher,
                glossary=shared.glossary(),
                config=self.config,
                style_brief=_style(),
                rolling_context=shared.rolling_context(),
                frozen_book=shared.frozen_book(),
                frozen_preparation=shared.frozen_preparation,
                batch_commit_hook=self.batch_commit_hook,
            ),
            NODE_POLISH: lambda shared, ci: PolishNode(
                polisher=shared.agents.polisher,
                extractor=shared.agents.extractor,
                glossary=shared.glossary(),
                config=self.config,
                style_brief=_style(),
                frozen_book=shared.frozen_book(),
                frozen_preparation=shared.frozen_preparation,
            ),
            NODE_NATURALIZE: lambda shared, ci: NaturalizeNode(
                naturalizer=shared.agents.naturalizer,
                glossary=shared.glossary(),
                config=self.config,
                frozen_book=shared.frozen_book(),
                frozen_preparation=shared.frozen_preparation,
            ),
            NODE_REVIEW: lambda shared, ci: ReviewNode(
                reviewer=shared.agents.reviewer,
                translator=shared.agents.translator,
                glossary=shared.glossary(),
                config=self.config,
                style_brief=_style(),
                frozen_book=shared.frozen_book(),
                frozen_preparation=shared.frozen_preparation,
            ),
            NODE_BACKTRANSLATE: lambda shared, ci: BacktranslateNode(
                backtrans=shared.agents.backtrans,
                config=self.config,
                frozen_book=shared.frozen_book(),
                frozen_preparation=shared.frozen_preparation,
                backtranslation_sample_scope=self.backtranslation_sample_scope,
            ),
            NODE_TITLES: lambda shared, ci: TitlesNode(
                client=self.client,
                config=self.config,
                src=shared.agents.src,
                tgt=shared.agents.tgt,
                glossary=shared.glossary(),
            ),
            NODE_CONSISTENCY_QA: lambda shared, ci: ConsistencyQANode(
                checker=shared.agents.consistency,
                glossary=shared.glossary(),
                config=self.config,
            ),
            NODE_REPORT: lambda shared, ci: ReportNode(glossary=shared.glossary()),
            NODE_ASSEMBLE: lambda shared, ci: AssembleNode(
                config=self.config,
                out_format=goal.out_format,
                out_path=goal.out_path,
            ),
        }

        def factory(node_id: str, ci: int | None):
            builder = builders.get(node_id)
            if builder is None:
                raise KeyError(f"未注册节点构造器: {node_id}")
            return builder(shared, ci)

        return factory

    def _prescan_inputs(
        self,
        store: RunStore,
        policy: WorkflowPolicy,
        shared: RunShared,
        goal: ExecutionGoal,
    ) -> PrescanInputs:
        """全部节点输入指纹计算器：planner 需要-run 判定与节点记录共用同一公式。

        assemble 指纹必须与 AssembleNode 实际执行的输出契约一致（目标 out_format、
        mono/bilingual 开关与双语顺序），否则换格式/换序请求会被误判已满足。
        """
        cfg = self.config

        def _src_lang() -> str:
            state = store.load_state()
            return state.identity.source_lang or normalize_lang_code(cfg.source_lang)

        def _tgt_lang() -> str:
            state = store.load_state()
            return state.identity.target_lang or normalize_lang_code(cfg.target_lang)

        def _style_brief() -> str:
            # style_brief 是纯函数（不触发 LLM）；构造临时 Analyzer 仅为复用格式逻辑。
            return Analyzer(self.client, cfg).style_brief(store.load_analysis() or {})

        def _book_synopsis() -> str:
            return (store.load_analysis() or {}).get("book_synopsis") or ""

        def _source_text(ci: int) -> str:
            return "\n".join(s.source for s in store.load_chapter(ci).text_segments)

        def _targets_text(ci: int) -> str:
            return "\n".join(s.target or "" for s in store.load_chapter(ci).text_segments)

        def _mode(ci: int) -> str | None:
            state = store.load_state()
            n = len(state.chapters)
            chapter = next((c for c in state.chapters if c.index == ci), None)
            if chapter is None:
                return None
            mode = policy.back_matter
            if mode in ("skip", "light") and is_back_matter(
                chapter.title, index=chapter.index, total=n
            ):
                return mode
            return None

        def _glossary_sources() -> list[str]:
            if frozen_book is not None:
                return [t.source for t in frozen_book.glossary]
            return [t.source for t in shared.glossary().all_terms()]

        def _titles() -> list[str]:
            state = store.load_state()
            titles = [c.title for c in state.chapters if c.title]
            meta = state.meta if isinstance(state.meta, dict) else {}
            raw_toc = meta.get("toc_entries")
            if isinstance(raw_toc, list):
                titles.extend(
                    str(e.get("title", ""))
                    for e in raw_toc
                    if isinstance(e, dict) and e.get("title")
                )
            return titles

        def _done_targets_text() -> str:
            state = store.load_state()
            parts: list[str] = []
            for c in state.chapters:
                if store.load_progress(c.index).status != "done":
                    continue
                parts.append(_targets_text(c.index))
            return "\n".join(parts)

        frozen_book = shared.frozen_book() if shared.frozen_preparation is not None else None
        style_brief = frozen_book.style_brief if frozen_book is not None else _style_brief()

        def digest_fp(ci: int) -> str:
            if frozen_book is not None:
                original_index = shared.frozen_chapter_index(ci)
                digest = frozen_book.chapter_digest(ci, original_index=original_index)
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_DIGEST,
                    (frozen_book.book_id, original_index),
                    digest,
                )
            return digest_input_fingerprint(_source_text(ci), _src_lang(), fast_model_profile(cfg))

        def name_terms_fp() -> str | None:
            if frozen_book is None:
                return None
            return frozen_input_fingerprint(
                shared.frozen_preparation.preparation_sha256,
                NODE_NAME_TERMS,
                frozen_book.book_id,
                [t.source for t in frozen_book.glossary],
            )

        def mine_fp() -> str:
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_MINE_TERMS,
                    frozen_book.book_id,
                    [t.source for t in frozen_book.glossary],
                )
            state = store.load_state()
            n = len(state.chapters)
            mine = [
                c.index
                for c in state.chapters
                if not is_back_matter(c.title, index=c.index, total=n)
            ]
            texts = [_source_text(ci) for ci in mine]
            return mine_terms_input_fingerprint(
                texts, _src_lang(), policy.prescan_concurrency, fast_model_profile(cfg)
            )

        def synopsis_fp(digests: list[str]) -> str:
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_BOOK_SYNOPSIS,
                    frozen_book.book_id,
                    (frozen_book.book_synopsis, tuple(digests)),
                )
            return book_synopsis_input_fingerprint(
                digests, style_brief, _src_lang(), _tgt_lang(), fast_model_profile(cfg)
            )

        def translate_fp(ci: int) -> str:
            src = _source_text(ci)
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_TRANSLATE,
                    (frozen_book.book_id, shared.frozen_chapter_index(ci)),
                    src,
                )
            if _mode(ci):
                return back_matter_translate_input_fingerprint(
                    src,
                    _src_lang(),
                    _tgt_lang(),
                    punctuation_normalize=cfg.punctuation_normalize,
                    model=fast_model_profile(cfg),
                )
            return translate_input_fingerprint(
                src,
                _src_lang(),
                _tgt_lang(),
                book_synopsis=_book_synopsis(),
                style_brief=style_brief,
                punctuation_normalize=cfg.punctuation_normalize,
                honorific_strategy=cfg.honorific_strategy,
                glossary_scope=cfg.pipeline.glossary_scope,
                model=primary_model_profile(cfg),
            )

        def polish_fp(ci: int) -> str:
            src = _source_text(ci)
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_POLISH,
                    (frozen_book.book_id, shared.frozen_chapter_index(ci)),
                    src,
                )
            return polish_input_fingerprint(
                src,
                _src_lang(),
                style_brief,
                punctuation_normalize=cfg.punctuation_normalize,
                model=editor_model_profile(cfg),
            )

        def naturalize_fp(ci: int) -> str:
            src = _source_text(ci)
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_NATURALIZE,
                    (frozen_book.book_id, shared.frozen_chapter_index(ci)),
                    src,
                )
            return naturalize_input_fingerprint(
                src,
                punctuation_normalize=cfg.punctuation_normalize,
                model=editor_fast_model_profile(cfg),
            )

        def review_fp(ci: int) -> str:
            src = _source_text(ci)
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_REVIEW,
                    (frozen_book.book_id, shared.frozen_chapter_index(ci)),
                    (src, policy.autofix_severe),
                )
            return review_input_fingerprint(
                src,
                autofix_severe=policy.autofix_severe,
                review_output_retries=cfg.pipeline.review_output_retries,
                model=primary_fast_model_profile(cfg),
            )

        def backtranslate_fp(ci: int) -> str:
            src = _source_text(ci)
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_BACKTRANSLATE,
                    (frozen_book.book_id, shared.frozen_chapter_index(ci)),
                    (src, policy.backtranslate_sample, self.backtranslation_sample_scope),
                )
            return backtranslate_input_fingerprint(
                src,
                backtranslate_sample=policy.backtranslate_sample,
                model=(
                    f"{fast_model_profile(cfg)}"
                    f"|backtranslation_sample_scope={self.backtranslation_sample_scope}"
                ),
            )

        def titles_fp() -> str:
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_TITLES,
                    frozen_book.book_id,
                    _titles(),
                )
            return titles_input_fingerprint(
                _titles(), _src_lang(), _tgt_lang(), _book_synopsis(), primary_model_profile(cfg)
            )

        def consistency_fp() -> str:
            return consistency_input_fingerprint(
                _done_targets_text(),
                glossary_semantic_fingerprint_part(shared.glossary().all_terms()),
                fast_model_profile(cfg),
            )

        def report_fp() -> str:
            state = store.load_state()
            review_issues: list[dict] = []
            bt_issues: list[dict] = []
            titles: list[str] = []
            for c in state.chapters:
                pg = store.load_progress(c.index)
                review_issues.extend(pg.review_issue_dicts())
                bt_issues.extend(pg.backtranslation_issue_dicts())
                if c.title:
                    titles.append(c.title)
            qa_node = state.nodes.get(NODE_CONSISTENCY_QA)
            consistency_issues = (qa_node.output or {}).get("issues") if qa_node else None
            return report_input_fingerprint(
                review_issues,
                bt_issues,
                consistency_issues or [],
                _glossary_sources(),
                titles,
            )

        def analyze_fp() -> str:
            if frozen_book is not None:
                return frozen_input_fingerprint(
                    shared.frozen_preparation.preparation_sha256,
                    NODE_ANALYZE,
                    frozen_book.book_id,
                    frozen_book.analysis,
                )
            # 风格/角色分析使用 primary（analyst）模型；样本从持久化章节重建
            # （与 AnalyzeNode 对同一解析文档的 sample_text 计算一致）。
            state = store.load_state()
            chapters = []
            for c in state.chapters:
                ch = store.load_chapter(c.index)
                chapters.append(
                    Chapter(
                        index=c.index,
                        title=c.title,
                        segments=ch.text_segments,
                    )
                )
            doc = Document(
                title=state.title,
                fmt=state.fmt,
                source_lang=state.source_lang,
                target_lang=state.target_lang,
                source_path=state.source_path,
                chapters=chapters,
            )
            return analyze_input_fingerprint(sample_text(doc), primary_model_profile(cfg))

        def assemble_fp() -> str:
            return assemble_input_fingerprint(
                _done_targets_text(),
                mono=cfg.output.mono,
                bilingual=cfg.output.bilingual,
                out_format=goal.out_format,
                bilingual_order=cfg.output.bilingual_order,
            )

        return PrescanInputs(
            digest_fingerprint=digest_fp,
            mine_fingerprint=mine_fp,
            name_terms_fingerprint=name_terms_fp,
            synopsis_fingerprint=synopsis_fp,
            prepare_fingerprint=lambda: prepare_input_fingerprint(
                store.load_state().identity.source_bytes_sha256, _src_lang(), _tgt_lang()
            ),
            analyze_fingerprint=analyze_fp,
            translate_fingerprint=translate_fp,
            polish_fingerprint=polish_fp,
            naturalize_fingerprint=naturalize_fp,
            review_fingerprint=review_fp,
            backtranslate_fingerprint=backtranslate_fp,
            titles_fingerprint=titles_fp,
            consistency_fingerprint=consistency_fp,
            report_fingerprint=report_fp,
            assemble_fingerprint=assemble_fp,
        )

    # ── 目标执行 ──────────────────────────────────────────────────────────
    # 阶段分段：prepare（不依赖暂存章节）→ 翻译闭包（prescan/translate/titles）→
    # 收尾（qa/report/assemble）。章相关规划必须等 prepare 暂存文档后才能进行；
    # “翻译完成”标签夹在翻译闭包与收尾之间（与旧 run()/finish 顺序一致）。
    _PREPARE_PHASES = ("prepare",)
    _TRANSLATE_PHASES = ("prescan", "translate", "titles")
    _FINISH_PHASES = ("qa", "report", "assemble")

    def run_goal(
        self,
        input_path: str,
        goal: ExecutionGoal,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[RunResult, RunStore]:
        doc = load_document(
            input_path,
            self.config.source_lang,
            self.config.target_lang,
            split_segments=self.config.segment.max_chars_per_segment,
        )
        return self._run_document_goal(doc, input_path, goal, progress=progress)

    def run_document_goal(
        self,
        doc,
        identity_path: str,
        goal: ExecutionGoal,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[RunResult, RunStore]:
        """Run the built-in workflow for an already constructed Document."""
        return self._run_document_goal(doc, identity_path, goal, progress=progress)

    def _run_document_goal(
        self,
        doc,
        identity_path: str,
        goal: ExecutionGoal,
        *,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[RunResult, RunStore]:
        run_dir = os.path.join(self.config.state_dir, slugify(doc.title))
        store = RunStore(run_dir)
        if store.exists() and doc.fmt == "epub":
            store.load_manifest()
        shared = RunShared(
            store=store,
            config=self.config,
            doc=doc,
            agent_builder=self._build_agents,
            frozen_preparation=self.frozen_preparation,
        )
        policy = WorkflowPolicy.from_config(self.config)
        result: RunResult | None = None
        try:
            prep_phases = [p for p in goal.phases if p in self._PREPARE_PHASES]
            if prep_phases:
                prep_goal = ExecutionGoal(name="prepare", phases=tuple(prep_phases))
                result = self._run_plan(
                    store, shared, policy, prep_goal, identity_path, progress, "prepare"
                )

            translate_phases = [p for p in goal.phases if p in self._TRANSLATE_PHASES]
            if translate_phases:
                translate_goal = ExecutionGoal(
                    name=goal.name,
                    phases=tuple(translate_phases),
                    only_chapter=goal.only_chapter,
                    out_format=goal.out_format,
                    out_path=goal.out_path,
                )

                def build_translate_plan():
                    prescan = self._prescan_inputs(store, policy, shared, translate_goal)
                    plan = self.planner.build_plan(
                        goal=translate_goal, store=store, policy=policy, prescan=prescan
                    )
                    shared.segments_done = 0
                    shared.segments_total = count_segments(store, plan.targets)
                    store.log_event(
                        "translate_run_started",
                        only_chapter=translate_goal.only_chapter,
                        chapters=plan.targets,
                        total_segments=shared.segments_total,
                    )
                    return plan

                result = self._run_plan(
                    store,
                    shared,
                    policy,
                    translate_goal,
                    identity_path,
                    progress,
                    "translate" if "translate" in translate_phases else "prepare",
                    plan_builder=build_translate_plan,
                )
                if "translate" in translate_phases:
                    if progress and shared.segments_total:
                        progress(shared.segments_total, shared.segments_total, "翻译完成")
                    store.log_event("translate_run_finished", total_segments=shared.segments_total)

            finish_phases = [p for p in goal.phases if p in self._FINISH_PHASES]
            if finish_phases:
                finish_goal = ExecutionGoal(
                    name=goal.name,
                    phases=tuple(finish_phases),
                    out_format=goal.out_format,
                    out_path=goal.out_path,
                    do_qa=goal.do_qa,
                )
                result = self._run_plan(
                    store, shared, policy, finish_goal, identity_path, progress, "pipeline"
                )
            assert result is not None
            return result, store
        finally:
            shared.close()

    def _run_plan(
        self,
        store: RunStore,
        shared: RunShared,
        policy: WorkflowPolicy,
        goal: ExecutionGoal,
        input_path: str,
        progress,
        usage_scope: str | None,
        plan_builder: Callable[[], WorkflowPlan] | None = None,
    ) -> RunResult:
        """构建并执行一个阶段的计划；运行锁由 runner 持有（唯一锁边界）。

        默认 builder 在锁内调用 build_plan（指纹对账/附属章升档重开必须与执行
        同临界区）；调用方可注入自定义 builder 以夹带锁内书签（进度计数/事件）。
        """

        def build() -> WorkflowPlan:
            prescan = self._prescan_inputs(store, policy, shared, goal)
            return self.planner.build_plan(goal=goal, store=store, policy=policy, prescan=prescan)

        runner = WorkflowRunner(
            definition=self.definition,
            node_factory=self._node_factory(shared, goal),
            usage_flush=lambda s, scope: self.flush_usage(s, scope=scope),
        )
        return runner.run(
            plan_builder or build,
            store=store,
            input_path=input_path,
            progress=progress,
            shared=shared,
            usage_scope=usage_scope,
        )

    @staticmethod
    def _usage_scope(goal: ExecutionGoal) -> str | None:
        if "translate" in goal.phases:
            return "translate"
        if any(p in goal.phases for p in ("qa", "report", "assemble")):
            return "pipeline"
        if "prepare" in goal.phases or "prescan" in goal.phases:
            return "prepare"
        return None

    # ── CLI 目标 ──────────────────────────────────────────────────────────
    def prepare(self, input_path: str, *, progress=None) -> RunStore:
        """解析 + 初始化 + 风格分析（不预扫、不翻译）。"""
        _, store = self.run_goal(
            input_path, ExecutionGoal(name="prepare", phases=("prepare",)), progress=progress
        )
        return store

    def prepare_for_translation(self, input_path: str, *, progress=None) -> RunStore:
        """完成文档解析、全书预扫和术语定名，但不翻译正文。"""
        _, store = self.run_goal(input_path, GOAL_PREPARE, progress=progress)
        return store

    def run(
        self,
        input_path: str,
        *,
        only_chapter: int | None = None,
        progress=None,
    ) -> RunStore:
        """翻译（only_chapter 时只译一章，不做收尾）。"""
        goal = translate_chapter_goal(only_chapter) if only_chapter is not None else GOAL_TRANSLATE
        _, store = self.run_goal(input_path, goal, progress=progress)
        return store

    def run_all(
        self,
        input_path: str,
        *,
        progress=None,
        out_format: str = "epub",
        out_path: str | None = None,
        do_qa: bool | None = None,
    ) -> dict[str, Any]:
        """翻译 → 一致性 QA → 报告 → 回填，返回结果汇总。"""
        goal = ExecutionGoal(
            name="run_all",
            phases=GOAL_RUN_ALL.phases,
            out_format=out_format,
            out_path=out_path,
            do_qa=do_qa,
        )
        return self._steps_result(input_path, goal, progress=progress)

    def run_goal_result(
        self,
        input_path: str,
        goal: ExecutionGoal,
        *,
        progress=None,
    ) -> dict[str, Any]:
        """执行目标并整理 CLI/测试需要的汇总（store/output/outputs/report/qa_issues）。"""
        return self._steps_result(input_path, goal, progress=progress)

    def _steps_result(self, input_path: str, goal: ExecutionGoal, *, progress=None) -> dict:
        result, store = self.run_goal(input_path, goal, progress=progress)
        outputs = result.artifact("assemble", "outputs", [])
        report = result.artifact("report", "report")
        qa_issues = result.artifact("consistency_qa", "issues", [])
        return {
            "store": store,
            "output": outputs[0] if outputs else None,
            "outputs": outputs,
            "report": report,
            "qa_issues": qa_issues,
        }

    def translate_titles(self, store: RunStore) -> None:
        """仅翻译章标题/目录项（独立工具/测试复用 titles 节点）。"""
        self._service_goal(store, titles_goal())

    def qa(self, store: RunStore) -> list[dict]:
        result = self._service_goal(store, qa_goal())
        return result.artifact("consistency_qa", "issues", [])

    def report(self, store: RunStore) -> dict:
        result = self._service_goal(store, report_goal())
        return result.artifact("report", "report")

    def assemble(
        self,
        store: RunStore,
        input_path: str,
        *,
        out_format: str = "epub",
        out_path: str | None = None,
        mono: bool | None = None,
        bilingual: bool | None = None,
    ) -> list[str]:
        """对已有状态回填（tools assemble；输出开关按 CLI flag 覆盖）。"""
        old_mono, old_bi = self.config.output.mono, self.config.output.bilingual
        if mono is not None:
            self.config.output.mono = mono
        if bilingual is not None:
            self.config.output.bilingual = bilingual
        try:
            goal = assemble_goal(out_format=out_format, out_path=out_path)
            result = self._service_goal(store, goal, input_path=input_path)
            return result.artifact("assemble", "outputs", [])
        finally:
            self.config.output.mono, self.config.output.bilingual = old_mono, old_bi

    def naturalize(
        self,
        store: RunStore,
        *,
        chapters: list[int] | None = None,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """存量书手动补跑去翻译腔（tools naturalize）。"""
        agent = Naturalizer(self.client, self.config)
        # 独立工具按运行状态里解析后的语言对绑定（auto 检测书的 lint 语言检查
        # 必须与显式 --source-language 一致，不能用未解析的 "auto"）。
        state = store.load_state()
        agent.src = state.identity.source_lang or state.source_lang or self.config.source_lang
        agent.tgt = state.identity.target_lang or state.target_lang or self.config.target_lang
        glossary = GlossaryStore(store.glossary_path)
        try:
            return run_naturalize(
                agent,
                store,
                glossary,
                self.config,
                chapters=chapters,
                dry_run=dry_run,
                limit=limit,
            )
        finally:
            glossary.close()

    def glossary_audit(self, store: RunStore) -> list[dict]:
        """术语 AI 审计统一（tools glossary audit）。"""
        glossary = GlossaryStore(store.glossary_path)
        try:
            return GlossaryAuditor(self.client, self.config).audit(store, glossary)
        finally:
            glossary.close()

    # ── 服务目标（复用 runner 跑单阶段计划）────────────────────────────────
    def _service_goal(
        self,
        store: RunStore,
        goal: ExecutionGoal,
        *,
        input_path: str | None = None,
    ) -> RunResult:
        shared = RunShared(
            store=store,
            config=self.config,
            doc=None,
            agent_builder=self._build_agents,
        )
        policy = WorkflowPolicy.from_config(self.config)
        try:
            runner = WorkflowRunner(
                definition=self.definition,
                node_factory=self._node_factory(shared, goal),
                usage_flush=lambda s, scope: self.flush_usage(s, scope=scope),
            )

            def build() -> WorkflowPlan:
                prescan = self._prescan_inputs(store, policy, shared, goal)
                return self.planner.build_plan(
                    goal=goal, store=store, policy=policy, prescan=prescan
                )

            source_path = input_path or store.load_state().source_path or ""
            return runner.run(
                build,
                store=store,
                input_path=source_path,
                progress=None,
                shared=shared,
                usage_scope=self._usage_scope(goal),
            )
        finally:
            shared.close()

    # ── 用量落盘（把 client 尚未落盘的增量合并到本书 usage.json）──────────
    def flush_usage(self, store: RunStore, *, scope: str) -> dict[str, Any]:
        """把当前 client 尚未落盘的用量增量合并到本书 usage.json。

        持久化门控不能只看 totals.calls：一次完全失败的逻辑调用（Agent 捕获异常
        回退 default）不会走 usage.record()，但 attempts/failed_attempts/
        logical_calls 仍会增长——这类仅含归因计数的增量同样必须落盘。
        """
        current = self.client.usage_summary()
        increment = usage_delta(current, self._usage_checkpoint)
        self._usage_checkpoint = current
        accumulated = store.load_usage() or {}
        has_activity = (
            bool(increment["totals"]["calls"])
            or bool(increment.get("by_agent"))
            or bool(increment.get("by_operation"))
        )
        if not has_activity:
            return merge_usage_summaries(accumulated, increment)
        cumulative = merge_usage_summaries(accumulated, increment)
        store.save_usage(cumulative)
        store.log_event(
            "usage_summary",
            scope=scope,
            increment=increment,
        )
        return cumulative


__all__ = ["Application", "build_workflow_definition"]
