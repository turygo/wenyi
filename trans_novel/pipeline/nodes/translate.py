"""章翻译节点：translate（批翻译 + lint/修复 + 检查点）与 polish（章末排干润色）。

从迁移前的 _translate_chapter 平移，边界按节点契约拆分：
- translate 保留批翻译、滚动上下文重建、确定性 lint 与定向重译修复、翻译检查点、
  可选 in-flight 术语抽取、附属章旁路（skip/light/full 是翻译内部策略）、
  升档重开、标题无关的章完成（done 由章链收尾节点落）；
- polish 保留 pending_polish 排干（本轮新提交 + 续跑遗留），逐批 lint 回退保护；
- 审校/自然化/回译已拆到 quality 节点；runner 只提供共享线程池与 store。
"""

from __future__ import annotations

from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.segmenter import batch_segments
from trans_novel.pipeline import checkpoint, lint
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    back_matter_translate_input_fingerprint,
    fast_model_profile,
    polish_input_fingerprint,
    primary_model_profile,
    translate_input_fingerprint,
)
from trans_novel.pipeline.nodes.common import chapter_term_snapshot, resume_batches
from trans_novel.pipeline.runstore import STATUS_DONE, STATUS_PENDING
from trans_novel.pipeline.state import (
    NODE_POLISH,
    NODE_TRANSLATE,
    SCOPE_CHAPTER,
    PolishBatch,
    chapter_node_key,
)
from trans_novel.postprocess.punct import normalize_zh

_BM_RANK = {"skip": 0, "light": 1, "full": 2}


class TranslateNode:
    """逐章批翻译：滚动上下文、lint 修复、检查点落盘、附属章旁路。"""

    node_id = NODE_TRANSLATE
    scope = SCOPE_CHAPTER

    def __init__(
        self,
        *,
        translator,
        extractor,
        polisher,
        glossary: GlossaryStore,
        config: Config,
        style_brief: str,
        rolling_context,
    ):
        self.translator = translator
        self.extractor = extractor
        self.polisher = polisher
        self.glossary = glossary
        self.config = config
        self.style_brief = style_brief
        self.rolling_context = rolling_context

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        config = self.config
        chapter = store.load_chapter(ci)
        text_segs = chapter.text_segments
        n_ch = request.total_chapters
        bm_mode = self._back_matter_mode(chapter.title, ci, n_ch)
        if not text_segs:
            store.set_chapter_status(ci, STATUS_DONE)
            store.log_event("chapter_skipped", chapter=ci, reason="empty")
            fp = self._fingerprint("", store, bm_mode)
            return NodeOutcome(chapter_finalized=True, fingerprint=fp)

        # 附属章档位升档（skip→light/full、light→full）：先重开已完成的章，
        # 再按当前档位走旁路或完整流水线。降档不回退。
        if store.chapter_status(ci) == STATUS_DONE:
            self._reopen_if_upgraded(ci, chapter, store, n_ch)
        chapter_progress = store.load_progress(ci)

        if bm_mode:
            return self._translate_back_matter(bm_mode, ci, chapter, text_segs, store, request)
        # full/正文路径：附属章照常翻译，但不抽术语（skip/light 已在上方旁路返回）。
        bm = is_back_matter(chapter.title, index=ci, total=n_ch)
        chapter_progress.back_matter_mode = None
        chapter_digest = chapter_progress.source_digest
        glossary = self.glossary
        context = self.rolling_context
        style = self.style_brief
        book_synopsis = (store.load_analysis() or {}).get("book_synopsis", "")

        batches = resume_batches(text_segs, config.segment.max_chars_per_batch)
        label = f"第{ci}章 {chapter.title}"
        if request.progress:
            request.progress(request.shared.segments_done, request.shared.segments_total, label)
        term_snapshot = chapter_term_snapshot(glossary, text_segs, config)

        # 保留历史审校项（不含 lint/length 层）；lint 项单独累积，章末并入。
        review_issues: list[dict] = [
            i
            for i in chapter_progress.review_issue_dicts()
            if i.get("stage") not in ("length", "lint")
        ]
        lint_review_issues: list[dict] = []
        polish_on = config.pipeline.polish
        seg_base = 0  # 当前批首段的章内段号
        for b in batches:
            existing_targets = [s.target for s in b if s.target and s.target.strip()]
            if len(existing_targets) == len(b):
                # 断点续跑：整批复用，只重建滚动上下文；确定性 lint 零成本复检一遍。
                context.add_targets(existing_targets)
                summary = None
                if not bm and config.pipeline.inflight_glossary:
                    summary, changed = self._extract_batch_glossary(
                        glossary, store, ci, seg_base, b
                    )
                    remaining_src = "\n".join(s.source for s in text_segs[seg_base + len(b) :])
                    if changed and GlossaryStore.terms_in(changed, remaining_src):
                        term_snapshot = chapter_term_snapshot(glossary, text_segs, config)
                locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
                for it in lint.lint_targets(
                    [s.source for s in b],
                    existing_targets,
                    locked_terms=locked,
                    src_lang=self.translator.src,
                ):
                    lint_review_issues.append(
                        {
                            "chapter": ci,
                            "index": seg_base + it.index,
                            "type": it.type,
                            "detail": it.detail,
                            "stage": "lint",
                            "fixed": False,
                        }
                    )
                store.log_event(
                    "batch_skipped",
                    chapter=ci,
                    start_index=seg_base,
                    count=len(b),
                    reason="already_translated",
                    glossary_extraction=summary,
                    segments=[
                        {"index": seg_base + i, "source": s.source, "target": s.target}
                        for i, s in enumerate(b)
                    ],
                )
                request.shared.segments_done += len(b)
                seg_base += len(b)
                if request.progress:
                    request.progress(
                        request.shared.segments_done, request.shared.segments_total, label
                    )
                continue

            ctx_text = context.render(config.pipeline.rolling_context_segments)
            raw_targets = self._process_batch(
                b, term_snapshot, ctx_text, style, book_synopsis, chapter_digest
            )

            # 确定性 lint（零 LLM）：flag 段带审校意见定向重译，每段最多一轮。
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            lint_issues = lint.lint_targets(
                [s.source for s in b],
                raw_targets,
                locked_terms=locked,
                src_lang=self.translator.src,
            )
            if lint_issues:
                store.log_event(
                    "batch_linted",
                    chapter=ci,
                    start_index=seg_base,
                    issues=[
                        {"index": seg_base + it.index, "type": it.type, "detail": it.detail}
                        for it in lint_issues
                    ],
                )
                by_idx: dict[int, list] = {}
                for it in lint_issues:
                    by_idx.setdefault(it.index, []).append(it)
                actionable_idx: list[int] = []
                for idx, seg_issues in sorted(by_idx.items()):
                    if not any(it.type in lint.ACTIONABLE_TYPES for it in seg_issues):
                        for it in seg_issues:
                            lint_review_issues.append(
                                {
                                    "chapter": ci,
                                    "index": seg_base + idx,
                                    "type": it.type,
                                    "detail": it.detail,
                                    "stage": "lint",
                                    "fixed": False,
                                }
                            )
                        continue
                    actionable_idx.append(idx)

                def _apply_fix_result(
                    idx: int,
                    new_t: str,
                    *,
                    by_idx: dict[int, list] = by_idx,
                    b: list = b,
                    locked: list = locked,
                    seg_base: int = seg_base,
                    raw_targets: list = raw_targets,
                ) -> None:
                    seg_issues = by_idx[idx]
                    seg = b[idx]
                    new_issues = (
                        lint.lint_targets(
                            [seg.source],
                            [new_t],
                            locked_terms=locked,
                            src_lang=self.translator.src,
                        )
                        if new_t
                        else []
                    )
                    if new_t and len(new_issues) < len(seg_issues):
                        self.translator.client.usage.record_outcome(
                            "translator", "translate.lint_fix", accepted=True
                        )
                        store.log_event(
                            "lint_refixed",
                            chapter=ci,
                            index=seg_base + idx,
                            before=raw_targets[idx],
                            after=new_t,
                            issues=[{"type": it.type, "detail": it.detail} for it in seg_issues],
                        )
                        raw_targets[idx] = new_t
                        remaining = new_issues
                    else:
                        self.translator.client.usage.record_outcome(
                            "translator", "translate.lint_fix", accepted=False
                        )
                        remaining = seg_issues
                    for it in remaining:
                        lint_review_issues.append(
                            {
                                "chapter": ci,
                                "index": seg_base + idx,
                                "type": it.type,
                                "detail": it.detail,
                                "stage": "lint",
                                "fixed": False,
                            }
                        )

                merged_targets: dict[int, str] = {}
                if len(actionable_idx) > 1:
                    merged = self.translator.retranslate_batch_with_feedback(
                        [
                            (idx, b[idx].source, "；".join(it.detail for it in by_idx[idx]))
                            for idx in actionable_idx
                        ],
                        raw_targets,
                        operation="translate.lint_fix",
                        glossary_terms=term_snapshot,
                        style=style,
                        book_synopsis=book_synopsis,
                        chapter_digest=chapter_digest,
                    )
                    if len(merged) == len(actionable_idx):
                        merged_targets = dict(zip(actionable_idx, merged, strict=False))

                if merged_targets:
                    for idx in actionable_idx:
                        _apply_fix_result(idx, merged_targets.get(idx, ""))
                else:
                    for idx in actionable_idx:
                        seg = b[idx]
                        feedback = "；".join(it.detail for it in by_idx[idx])
                        before = "\n".join(raw_targets[j] for j in range(max(0, idx - 2), idx))
                        after = "\n".join(
                            raw_targets[j] for j in range(idx + 1, min(len(b), idx + 3))
                        )
                        new_t = self.translator.retranslate_with_feedback(
                            seg.source,
                            feedback=feedback,
                            operation="translate.lint_fix",
                            glossary_terms=term_snapshot,
                            style=style,
                            context_before=before,
                            context_after=after,
                            book_synopsis=book_synopsis,
                            chapter_digest=chapter_digest,
                        )
                        _apply_fix_result(idx, new_t)
            for s, t in zip(b, raw_targets, strict=False):
                s.target = t
            batch_start = seg_base
            request.shared.segments_done += len(b)
            seg_base += len(b)

            if polish_on:
                context.add_targets(raw_targets)
                chapter_progress.pending_polish.append(PolishBatch(start=batch_start, count=len(b)))
                event_targets = raw_targets
                punctuation_normalized = False
            else:
                final_targets = raw_targets
                if config.punctuation_normalize:
                    final_targets = [normalize_zh(t) if t else t for t in final_targets]
                    for s, t in zip(b, final_targets, strict=False):
                        s.target = t
                context.add_targets(final_targets)
                event_targets = final_targets
                punctuation_normalized = config.punctuation_normalize

            store.log_event(
                "batch_translated",
                chapter=ci,
                start_index=batch_start,
                count=len(b),
                polished=False,
                punctuation_normalized=punctuation_normalized,
                segments=[
                    {"index": batch_start + i, "source": s.source, "target": t}
                    for i, (s, t) in enumerate(zip(b, event_targets, strict=False))
                ],
            )
            # 增量持久化：本批译文（+ pending_polish / review_issues 标记）立即落盘。
            # 崩溃一致性（polish_on 时）：译文在章节文件、标记在 manifest，两次独立
            # 原子写；先写检查点日志再落盘，崩溃后由锁内恢复补齐/清除标记。
            chapter_progress.set_review_issue_dicts(review_issues)
            if polish_on:
                checkpoint.begin_translate(store, ci, batch_start, len(b))
            store.save_chapter(chapter)
            store.save_progress(ci, chapter_progress)
            checkpoint.clear(store)

            if polish_on:
                # 润色批间无依赖：提交共享线程池，章末由 polish 节点统一排干。
                request.shared.polish_futures[(ci, batch_start)] = request.executor.submit(
                    self.polisher.polish,
                    list(raw_targets),
                    [s.source for s in b],
                    glossary_terms=list(term_snapshot),
                    style=style,
                    strict=True,
                )

            if not bm and config.pipeline.inflight_glossary:
                batch_src = "\n".join(s.source for s in b)
                terms = self.extractor.extract(
                    batch_src,
                    "\n".join(raw_targets),
                    GlossaryStore.terms_in(term_snapshot, batch_src),
                )
                summary, changed = self.extractor.store_terms(glossary, terms, ci)
                remaining_src = "\n".join(s.source for s in text_segs[seg_base:])
                if changed and GlossaryStore.terms_in(changed, remaining_src):
                    term_snapshot = chapter_term_snapshot(glossary, text_segs, config)
                store.log_event(
                    "batch_glossary_extracted",
                    chapter=ci,
                    start_index=batch_start,
                    count=len(b),
                    summary=summary,
                )
            if request.progress:
                request.progress(request.shared.segments_done, request.shared.segments_total, label)

        # 章末：全章术语兜底抽取。polish 开启时改由 PolishNode 在润色落盘后抽取
        # （润色会改动术语所在译文，须在最终文本上抽）；关闭时保持译后立即抽取。
        if not bm and config.pipeline.inflight_glossary and not polish_on:
            src_text = "\n".join(s.source for s in text_segs)
            tgt_text = "\n".join(s.target or "" for s in text_segs)
            self.extractor.extract_and_store(glossary, src_text, tgt_text, ci)
            store.log_event("chapter_glossary_extracted", chapter=ci)

        # 审校项：review 开启时由 review 节点替换（只保留 lint 项）；关闭时保留历史非 lint 项。
        if config.pipeline.review:
            chapter_progress.set_review_issue_dicts(lint_review_issues)
        else:
            chapter_progress.set_review_issue_dicts(review_issues + lint_review_issues)
        store.save_chapter(chapter)
        store.save_progress(ci, chapter_progress)
        store.save_context(context.to_dict())
        fp = self._fingerprint("\n".join(s.source for s in text_segs), store, None)
        return NodeOutcome(fingerprint=fp)

    def _fingerprint(self, source_text: str, store, bm_mode: str | None) -> str:
        """translate 输入指纹：正文含概览/风格/提示配置；旁路只含源文/语言/标点。"""
        if bm_mode:
            return back_matter_translate_input_fingerprint(
                source_text,
                self.translator.src,
                self.translator.tgt,
                punctuation_normalize=self.config.punctuation_normalize,
                model=fast_model_profile(self.config),
            )
        book_synopsis = (store.load_analysis() or {}).get("book_synopsis", "") or ""
        return translate_input_fingerprint(
            source_text,
            self.translator.src,
            self.translator.tgt,
            book_synopsis=book_synopsis,
            style_brief=self.style_brief,
            punctuation_normalize=self.config.punctuation_normalize,
            honorific_strategy=self.config.honorific_strategy,
            glossary_scope=self.config.pipeline.glossary_scope,
            model=primary_model_profile(self.config),
        )

    # ── 附属章旁路 ────────────────────────────────────────────────────────
    def _back_matter_mode(self, title: str, index: int, total: int) -> str | None:
        mode = self.config.pipeline.back_matter
        if mode in ("skip", "light") and is_back_matter(title, index=index, total=total):
            return mode
        return None

    def _reopen_if_upgraded(self, ci: int, chapter, store, n_ch: int) -> None:
        """附属章档位升档时重开已完成的附属章（旁路档 target 非空，会被批级续跑
        当成已译整批复用——不清掉就升档形同虚设）。降档不回退。"""
        progress = store.load_progress(ci)
        prev = progress.back_matter_mode
        if prev not in _BM_RANK:
            return
        cur = self._back_matter_mode(chapter.title, ci, n_ch) or "full"
        if _BM_RANK[cur] <= _BM_RANK[prev]:
            return
        for s in chapter.segments:
            s.target = None
        progress.back_matter_mode = None
        progress.pending_polish = []
        progress.set_review_issue_dicts([])
        progress.set_backtranslation_issue_dicts([])
        store.save_progress(ci, progress)
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_PENDING)
        store.log_event(
            "back_matter_reopened",
            chapter=ci,
            title=chapter.title,
            prev_mode=prev,
            mode=cur,
        )

    def _translate_back_matter(self, mode, ci, chapter, text_segs, store, request) -> NodeOutcome:
        """附属章旁路：skip=原文直通；light=快速粗翻（走 translate.back_matter）。
        不碰 glossary/context/style/executor；由本节点收尾本章。"""
        label = f"第{ci}章 {chapter.title}"
        store.log_event("chapter_back_matter", chapter=ci, title=chapter.title, mode=mode)
        config = self.config
        if mode == "skip":
            for s in text_segs:
                s.target = s.source
            store.save_chapter(chapter)
            request.shared.segments_done += len(text_segs)
            if request.progress:
                request.progress(request.shared.segments_done, request.shared.segments_total, label)
        elif mode == "light":
            batches = batch_segments(text_segs, config.segment.max_chars_per_batch)
            seg_base = 0
            for b in batches:
                existing = [s.target for s in b if s.target and s.target.strip()]
                if len(existing) == len(b):
                    request.shared.segments_done += len(b)
                    seg_base += len(b)
                    if request.progress:
                        request.progress(
                            request.shared.segments_done, request.shared.segments_total, label
                        )
                    continue
                raw = self.translator.translate_batch(
                    [s.source for s in b],
                    glossary_terms=[],
                    style="",
                    context="",
                    book_synopsis="",
                    chapter_digest="",
                    agent="light-translator",
                    operation="translate.back_matter",
                )
                if config.punctuation_normalize:
                    raw = [normalize_zh(t) if t else t for t in raw]
                for s, t in zip(b, raw, strict=False):
                    s.target = t
                store.save_chapter(chapter)
                store.log_event(
                    "batch_translated",
                    chapter=ci,
                    start_index=seg_base,
                    count=len(b),
                    polished=False,
                    punctuation_normalized=config.punctuation_normalize,
                    back_matter=True,
                    operation="translate.back_matter",
                    segments=[
                        {"index": seg_base + i, "source": s.source, "target": t}
                        for i, (s, t) in enumerate(zip(b, raw, strict=False))
                    ],
                )
                request.shared.segments_done += len(b)
                seg_base += len(b)
                if request.progress:
                    request.progress(
                        request.shared.segments_done, request.shared.segments_total, label
                    )
        # 记录旁路档位 + 清陈旧润色/审校标记；旁路章由本节点收尾。
        bm_progress = store.load_progress(ci)
        bm_progress.back_matter_mode = mode
        bm_progress.pending_polish = []
        bm_progress.set_review_issue_dicts([])
        bm_progress.set_backtranslation_issue_dicts([])
        store.save_progress(ci, bm_progress)
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_DONE)
        store.log_event(
            "chapter_done",
            chapter=ci,
            title=chapter.title,
            segment_count=len(text_segs),
            review_issue_count=0,
            backtranslation_issue_count=0,
            back_matter=True,
            mode=mode,
        )
        fp = self._fingerprint("\n".join(s.source for s in text_segs), store, mode)
        return NodeOutcome(chapter_finalized=True, fingerprint=fp)

    # ── 批处理 ────────────────────────────────────────────────────────────
    def _process_batch(
        self,
        batch,
        terms,
        ctx_text: str,
        style: str,
        book_synopsis: str = "",
        chapter_digest: str = "",
    ) -> list[str]:
        sources = [s.source for s in batch]
        return self.translator.translate_batch(
            sources,
            agent="translator",
            glossary_terms=terms,
            style=style,
            context=ctx_text,
            book_synopsis=book_synopsis,
            chapter_digest=chapter_digest,
        )

    def _extract_batch_glossary(
        self,
        glossary: GlossaryStore,
        store,
        chapter: int,
        start_index: int,
        batch,
    ) -> tuple[dict, list]:
        """续跑批跳过时同步抽取术语入库（新译批次的抽取已挪到批循环内联流程）。"""
        src_text = "\n".join(s.source for s in batch)
        tgt_text = "\n".join(s.target or "" for s in batch)
        summary, changed = self.extractor.extract_and_store(glossary, src_text, tgt_text, chapter)
        store.log_event(
            "batch_glossary_extracted",
            chapter=chapter,
            start_index=start_index,
            count=len(batch),
            summary=summary,
        )
        return summary, changed


class PolishNode:
    """章末排干本章全部润色 future（本轮新提交的 + 续跑遗留的 pending_polish）。"""

    node_id = NODE_POLISH
    scope = SCOPE_CHAPTER

    def __init__(
        self,
        *,
        polisher,
        extractor,
        glossary: GlossaryStore,
        config: Config,
        style_brief: str,
    ):
        self.polisher = polisher
        self.extractor = extractor
        self.glossary = glossary
        self.config = config
        self.style_brief = style_brief

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        chapter = store.load_chapter(ci)
        chapter_progress = store.load_progress(ci)
        text_segs = chapter.text_segments
        glossary = self.glossary
        style = self.style_brief
        term_snapshot = chapter_term_snapshot(glossary, text_segs, self.config)
        pending = list(chapter_progress.pending_polish)
        if not pending:
            # 策略从禁用切到启用：此前 polish 关闭时翻译的章没有 pending 标记，
            # 也没有记录过润色指纹（skipped/从未润色）。从已译段推导批次一次性
            # 补润色——不清除译文、不重译。用“指纹为空”而非节点状态判断：节点
            # 执行期间状态已是 running，status == skipped 永远不成立。
            node = store.load_state().nodes.get(chapter_node_key(NODE_POLISH, ci))
            if node is None or not node.input_fingerprint:
                pending = []
                idx = 0
                for b in batch_segments(text_segs, self.config.segment.max_chars_per_batch):
                    pending.append(PolishBatch(start=idx, count=len(b)))
                    idx += len(b)
                chapter_progress.pending_polish = list(pending)
        self._drain_chapter_polish(
            chapter,
            chapter_progress,
            text_segs,
            request.shared.polish_futures,
            request.executor,
            style,
            term_snapshot,
            store,
            ci,
        )
        # 章级术语兜底抽取：润色落盘后（润色前的译文可能含被润色修正的术语变体）。
        # 与 TranslateNode 同口径：附属章（full 档下命中 is_back_matter）不抽术语。
        if self.config.pipeline.inflight_glossary and not is_back_matter(
            chapter.title, index=ci, total=request.total_chapters
        ):
            src_text = "\n".join(s.source for s in text_segs)
            tgt_text = "\n".join(s.target or "" for s in text_segs)
            self.extractor.extract_and_store(glossary, src_text, tgt_text, ci)
            store.log_event("chapter_glossary_extracted", chapter=ci)
        fp = polish_input_fingerprint(
            "\n".join(s.source for s in text_segs),
            self.polisher.src,
            style,
            punctuation_normalize=self.config.punctuation_normalize,
            model=primary_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp)

    def _drain_chapter_polish(
        self,
        chapter,
        chapter_progress,
        text_segs,
        futures_by_key: dict,
        executor,
        style: str,
        term_snapshot,
        store,
        ci: int,
    ) -> None:
        """写回按批序进行；每批清完立即落盘，保证续跑不丢润色（不变量 b）。
        异常时结果回退 raw，不阻断整章。"""
        pending = list(chapter_progress.pending_polish)
        if not pending:
            return
        for entry in pending:
            start = entry.start
            key = (ci, start)
            if key not in futures_by_key:
                count = entry.count
                raw = [text_segs[start + i].target or "" for i in range(count)]
                futures_by_key[key] = executor.submit(
                    self.polisher.polish,
                    raw,
                    [text_segs[start + i].source for i in range(count)],
                    glossary_terms=list(term_snapshot),
                    style=style,
                    strict=True,
                )
        for entry in sorted(pending, key=lambda e: e.start):
            start, count = entry.start, entry.count
            fut = futures_by_key.pop((ci, start), None)
            raw = [text_segs[start + i].target or "" for i in range(count)]
            # workflow 必需路径：provider/协议失败必须冒泡（runner 落失败态并重试），
            # 不得把失败伪装成“润色成功”；lint 引入问题才回退原文（本地质量门）。
            final = fut.result() if fut is not None else raw
            if self.config.punctuation_normalize:
                final = [normalize_zh(t) if t else t for t in final]
            srcs = [text_segs[start + i].source for i in range(count)]
            raw_normalized = (
                [normalize_zh(t) if t else t for t in raw]
                if self.config.punctuation_normalize
                else raw
            )
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            raw_types: dict[int, set[str]] = {}
            for it in lint.lint_targets(
                srcs, raw_normalized, locked_terms=locked, src_lang=self.polisher.src
            ):
                raw_types.setdefault(it.index, set()).add(it.type)
            final_types: dict[int, set[str]] = {}
            for it in lint.lint_targets(
                srcs, final, locked_terms=locked, src_lang=self.polisher.src
            ):
                final_types.setdefault(it.index, set()).add(it.type)
            for i in range(count):
                introduced = final_types.get(i, set()) - raw_types.get(i, set())
                if not introduced:
                    self.polisher.client.usage.record_outcome(
                        "editor", "polish.batch", accepted=True
                    )
                    continue
                self.polisher.client.usage.record_outcome("editor", "polish.batch", accepted=False)
                rejected_text = final[i]
                final[i] = raw_normalized[i]
                store.log_event(
                    "polish_rejected",
                    chapter=ci,
                    index=start + i,
                    reason=sorted(introduced),
                    polished=rejected_text,
                )
            for i, t in enumerate(final):
                text_segs[start + i].target = t
            chapter_progress.pending_polish = [
                e for e in chapter_progress.pending_polish if e.start != start
            ]
            store.log_event(
                "batch_polished",
                chapter=ci,
                start_index=start,
                count=count,
                segments=[
                    {"index": start + i, "source": text_segs[start + i].source, "target": t}
                    for i, t in enumerate(final)
                ],
            )
            # 崩溃一致性：润色结果在章节文件、清标记在 manifest；先写检查点日志再
            # 落盘，崩溃后由锁内恢复清除已提交标记（避免同一段被润色两次）。
            checkpoint.begin_polish(store, ci, start, count, final)
            store.save_chapter(chapter)
            store.save_progress(ci, chapter_progress)
            checkpoint.clear(store)


__all__ = ["PolishNode", "TranslateNode"]
