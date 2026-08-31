"""章翻译节点：translate（批翻译 + 检查点）与 polish（章末排干润色）。

translate 保留批翻译、滚动上下文重建、确定性 lint 清单、翻译检查点、
可选 in-flight 术语抽取、附属章旁路和升档重开；Repair 节点统一负责质量修复。
polish 保留 pending_polish 排干（本轮新提交 + 续跑遗留），逐批 lint 回退保护。
审校/自然化/回译已拆到 quality 节点；runner 只提供共享线程池与 store。
"""

from __future__ import annotations

from trans_novel.config import Config
from trans_novel.epub.slots import (
    assign_segment_translation,
    distribute_slot_translation,
    normalize_slot_transport,
    reset_segment_translation,
    source_passthrough_transport,
    target_slot_transport,
)
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import KIND_HEADING
from trans_novel.ingest.segmenter import batch_segments
from trans_novel.llm.errors import LLM_FALLBACK_ERRORS
from trans_novel.pipeline import checkpoint, checks, lint
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import BatchCommitHook, NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    back_matter_translate_input_fingerprint,
    fast_translation_model_profile,
    frozen_input_fingerprint,
    polish_input_fingerprint,
    polish_model_profile,
    translate_input_fingerprint,
    translation_model_profile,
    translation_structure_fingerprint_part,
)
from trans_novel.pipeline.nodes.common import chapter_term_snapshot, resume_batches
from trans_novel.pipeline.runstore import STATUS_DONE, STATUS_PENDING, stable_digest
from trans_novel.pipeline.state import (
    NODE_POLISH,
    NODE_TRANSLATE,
    SCOPE_CHAPTER,
    PolishBatch,
    chapter_node_key,
)
from trans_novel.postprocess.punct import normalize_heading_numbering, normalize_zh

_BM_RANK = {"skip": 0, "light": 1, "full": 2}


def _align_epub_translations(segments, translations: list[str]) -> list[object]:
    """Distribute complete translations across EPUB slots deterministically."""
    result: list[object] = list(translations)
    for index, (segment, translation) in enumerate(zip(segments, translations, strict=True)):
        if segment.epub_state is not None:
            complete = (
                normalize_heading_numbering(translation)
                if segment.kind == "heading"
                else translation
            )
            if complete == segment.source:
                result[index] = source_passthrough_transport(segment)
            else:
                result[index] = distribute_slot_translation(segment, complete)
    return result


class TranslateNode:
    """逐章批翻译：滚动上下文、lint 清单、检查点落盘、附属章旁路。"""

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
        frozen_book=None,
        frozen_preparation=None,
        batch_commit_hook: BatchCommitHook | None = None,
    ):
        self.translator = translator
        self.extractor = extractor
        self.polisher = polisher
        self.glossary = glossary
        self.config = config
        self.style_brief = style_brief
        self.rolling_context = rolling_context
        self.frozen_book = frozen_book
        self.frozen_preparation = frozen_preparation
        self.batch_commit_hook = batch_commit_hook

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
            fp = self._fingerprint("", store, bm_mode, ci, request.shared)
            return NodeOutcome(chapter_finalized=True, fingerprint=fp)

        # 附属章档位升档（skip→light/full、light→full）：先重开已完成的章，
        # 再按当前档位走旁路或完整流水线。降档不回退。
        if store.chapter_status(ci) == STATUS_DONE:
            self._reopen_if_upgraded(ci, chapter, store, n_ch)
        chapter_progress = store.load_progress(ci)

        if bm_mode:
            return self._translate_back_matter(bm_mode, ci, chapter, text_segs, store, request)
        bm = is_back_matter(chapter.title, index=ci, total=n_ch)
        # full/正文路径：附属章照常翻译，但不抽术语（skip/light 已在上方旁路返回）。
        chapter_progress.back_matter_mode = None
        glossary = self.glossary
        context = self.rolling_context
        style = self.style_brief

        batches = resume_batches(text_segs, config.segment.max_chars_per_batch)
        label = f"第{ci}章 {chapter.title}"
        if request.progress:
            request.progress(request.shared.segments_done, request.shared.segments_total, label)
        term_snapshot = chapter_term_snapshot(glossary, text_segs, config)
        lint_issues: list[dict] = []
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
                    lint_issues.append(
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
                    target_sha256=stable_digest(
                        [{"index": s.index, "target": s.target} for s in b]
                    ),
                )
                request.shared.segments_done += len(b)
                seg_base += len(b)
                if request.progress:
                    request.progress(
                        request.shared.segments_done, request.shared.segments_total, label
                    )
                continue

            ctx_text = context.render(config.pipeline.rolling_context_segments)
            raw_transports, translate_call_count = self._process_batch(
                b, term_snapshot, ctx_text, style
            )
            raw_targets: list[str] = []
            for segment, transport in zip(b, raw_transports, strict=True):
                assign_segment_translation(segment, transport)
                raw_targets.append(segment.target or "")

            # Deterministic lint only inventories issues; Repair owns all model fixes.
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            batch_issues = lint.lint_targets(
                [s.source for s in b],
                raw_targets,
                locked_terms=locked,
                src_lang=self.translator.src,
            )
            if batch_issues:
                lint_issues_payload = [
                    {"index": seg_base + it.index, "type": it.type, "detail": it.detail}
                    for it in batch_issues
                ]
                type_counts: dict[str, int] = {}
                for it in batch_issues:
                    type_counts[it.type] = type_counts.get(it.type, 0) + 1
                    lint_issues.append(
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
                    "batch_linted",
                    chapter=ci,
                    start_index=seg_base,
                    issue_count=len(batch_issues),
                    by_type={t: type_counts[t] for t in sorted(type_counts)},
                    issues_sha256=stable_digest(lint_issues_payload),
                )
            batch_start = seg_base
            request.shared.segments_done += len(b)
            seg_base += len(b)

            if polish_on:
                context.add_targets(raw_targets)
                chapter_progress.pending_polish.append(PolishBatch(start=batch_start, count=len(b)))
                event_targets = raw_targets
                punctuation_normalized = False
            else:
                if config.punctuation_normalize:
                    for s in b:
                        if s.epub_state is None:
                            target = s.target or ""
                            if target != s.source:
                                assign_segment_translation(s, normalize_zh(target))
                        else:
                            assign_segment_translation(
                                s,
                                normalize_slot_transport(
                                    s,
                                    target_slot_transport(s),
                                ),
                            )
                    final_targets = [s.target or "" for s in b]
                else:
                    final_targets = raw_targets
                context.add_targets(final_targets)
                event_targets = final_targets
                punctuation_normalized = config.punctuation_normalize
            chapter_progress.lint_issues = lint_issues
            if polish_on:
                checkpoint.begin_translate(store, ci, batch_start, len(b))
            store.save_chapter(chapter)
            store.save_progress(ci, chapter_progress)
            checkpoint.clear(store)

            # 增量持久化：本批译文（+ pending_polish / lint_issues 标记）立即落盘。
            event_payload = {
                "chapter": ci,
                "start_index": batch_start,
                "count": len(b),
                "translate_call_count": translate_call_count,
                "polished": False,
                "punctuation_normalized": punctuation_normalized,
                "target_sha256": stable_digest(
                    [
                        {"index": s.index, "target": t}
                        for s, t in zip(b, event_targets, strict=False)
                    ]
                ),
            }
            if self.batch_commit_hook is not None:
                store.log_event_required("batch_translated", **event_payload)
            else:
                store.log_event("batch_translated", **event_payload)
            if self.batch_commit_hook is not None:
                self.batch_commit_hook.after_batch_committed(ci, batch_start, len(b))

            if polish_on:
                # 润色批间无依赖：提交共享线程池，章末由 polish 节点统一排干。
                request.shared.polish_futures[(ci, batch_start)] = request.executor.submit(
                    self.polisher.polish,
                    [s.target or "" for s in b],
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

        # 正文已按翻译批次增量保存完整（续跑时直接复用译文的批次不会修改正文），
        # 这里只保存进度和上下文，不再冗余写入整章。
        chapter_progress.lint_issues = lint_issues
        store.save_progress(ci, chapter_progress)
        store.save_context(context.to_dict())
        fp = self._fingerprint(
            "\n".join(s.source for s in text_segs), store, None, ci, request.shared
        )
        return NodeOutcome(fingerprint=fp)

    def _fingerprint(
        self,
        source_text: str,
        store,
        bm_mode: str | None,
        chapter_index: int | None = None,
        shared=None,
    ) -> str:
        """Fingerprint text, EPUB slot geometry, and every model consumed by translation."""
        if chapter_index is not None:
            source_text += "\n" + translation_structure_fingerprint_part(
                store.load_chapter(chapter_index).text_segments
            )
        if self.frozen_book is not None and self.frozen_preparation is not None:
            return frozen_input_fingerprint(
                self.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, shared.frozen_chapter_index(chapter_index)),
                source_text,
            )
        if bm_mode:
            return back_matter_translate_input_fingerprint(
                source_text,
                self.translator.src,
                self.translator.tgt,
                punctuation_normalize=self.config.punctuation_normalize,
                model=fast_translation_model_profile(self.config),
            )
        return translate_input_fingerprint(
            source_text,
            self.translator.src,
            self.translator.tgt,
            style_brief=self.style_brief,
            punctuation_normalize=self.config.punctuation_normalize,
            honorific_strategy=self.config.honorific_strategy,
            glossary_scope=self.config.pipeline.glossary_scope,
            single_segment_translation=self.config.pipeline.single_segment_translation,
            model=translation_model_profile(self.config),
        )

    # ── 附属章旁路 ────────────────────────────────────────────────────────
    def _back_matter_mode(self, title: str, index: int, total: int) -> str | None:
        mode = self.config.pipeline.back_matter
        if mode in ("skip", "light") and is_back_matter(title, index=index, total=total):
            return mode
        return None

    def _reopen_if_upgraded(self, ci: int, chapter, store, n_ch: int) -> None:
        progress = store.load_progress(ci)
        prev = progress.back_matter_mode
        if prev not in _BM_RANK:
            return
        cur = self._back_matter_mode(chapter.title, ci, n_ch) or "full"
        if _BM_RANK[cur] <= _BM_RANK[prev]:
            return
        for segment in chapter.segments:
            reset_segment_translation(segment)
        progress.back_matter_mode = None
        progress.pending_polish = []
        progress.lint_issues = []
        store.save_progress(ci, progress)
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_PENDING)

    def _translate_back_matter(self, mode, ci, chapter, text_segs, store, request) -> NodeOutcome:
        """附属章旁路：skip=原文直通；light=快速粗翻（走 translate.back_matter）。
        不碰 glossary/context/style/executor；由本节点收尾本章。"""
        label = f"第{ci}章 {chapter.title}"
        store.log_event("chapter_back_matter", chapter=ci, title=chapter.title, mode=mode)
        config = self.config
        if mode == "skip":
            for s in text_segs:
                assign_segment_translation(
                    s,
                    s.source if s.epub_state is None else source_passthrough_transport(s),
                )
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
                try:
                    result = self.translator.translate_batch(
                        [s.source for s in b],
                        agent="light-translator",
                        operation="translate.back_matter",
                        glossary_terms=[],
                        style="",
                        context="",
                    )
                    raw = _align_epub_translations(b, list(result.translations))
                    translate_call_count = result.request_count
                except LLM_FALLBACK_ERRORS:
                    raw, translate_call_count = self._safe_batch_fallback(b)
                for s, t in zip(b, raw, strict=True):
                    if config.punctuation_normalize:
                        if s.epub_state is None:
                            if t != s.source:
                                t = normalize_zh(t)
                        else:
                            t = normalize_slot_transport(s, t)
                    assign_segment_translation(s, t)
                raw = [s.target or "" for s in b]
                store.save_chapter(chapter)
                store.log_event(
                    "batch_translated",
                    chapter=ci,
                    start_index=seg_base,
                    count=len(b),
                    polished=False,
                    translate_call_count=translate_call_count,
                    back_matter=True,
                    operation="translate.back_matter",
                    target_sha256=stable_digest(
                        [{"index": s.index, "target": t} for s, t in zip(b, raw, strict=False)]
                    ),
                )
                request.shared.segments_done += len(b)
                seg_base += len(b)
                if request.progress:
                    request.progress(
                        request.shared.segments_done, request.shared.segments_total, label
                    )
        bm_progress = store.load_progress(ci)
        bm_progress.back_matter_mode = mode
        bm_progress.pending_polish = []
        bm_progress.lint_issues = []
        store.save_progress(ci, bm_progress)
        store.set_chapter_status(ci, STATUS_DONE)
        store.log_event(
            "chapter_done",
            chapter=ci,
            title=chapter.title,
            segment_count=len(text_segs),
            back_matter=True,
            mode=mode,
        )
        fp = self._fingerprint(
            "\n".join(s.source for s in text_segs), store, mode, ci, request.shared
        )
        return NodeOutcome(chapter_finalized=True, fingerprint=fp)

    def _process_batch(self, batch, terms, ctx_text: str, style: str) -> tuple[list[object], int]:
        for segment in batch:
            if segment.epub_state is not None:
                from trans_novel.epub.slots import normalized_source_text

                if segment.source != normalized_source_text(segment.epub_state.slots):
                    raise ValueError(f"EPUB source slot coverage mismatch: {segment.resource_href}")
        try:
            request_count = 0
            if self.config.pipeline.single_segment_translation:
                translated = []
                for segment in batch:
                    result = self.translator.translate_batch(
                        [segment.source],
                        agent="analyst" if segment.kind == KIND_HEADING else "translator",
                        operation=(
                            "translate.heading"
                            if segment.kind == KIND_HEADING
                            else "translate.single"
                        ),
                        fallback_agent=None if segment.kind == KIND_HEADING else "analyst",
                        glossary_terms=terms,
                        style=style if segment.kind != KIND_HEADING else "",
                        context=ctx_text if segment.kind != KIND_HEADING else "",
                        kind=KIND_HEADING if segment.kind == KIND_HEADING else None,
                    )
                    translated.extend(result.translations)
                    request_count += result.request_count
            else:
                result = self.translator.translate_batch(
                    [s.source for s in batch],
                    agent="translator",
                    glossary_terms=terms,
                    style=style,
                    context=ctx_text,
                )
                translated = list(result.translations)
                request_count = result.request_count
            return _align_epub_translations(batch, translated), request_count
        except LLM_FALLBACK_ERRORS:
            return self._safe_batch_fallback(batch)

    @staticmethod
    def _safe_batch_fallback(batch) -> tuple[list[object], int]:
        return [
            source_passthrough_transport(segment)
            if segment.epub_state is not None
            else segment.source
            for segment in batch
        ], 0

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
        frozen_book=None,
        frozen_preparation=None,
    ):
        self.polisher = polisher
        self.extractor = extractor
        self.glossary = glossary
        self.config = config
        self.style_brief = style_brief
        self.frozen_book = frozen_book
        self.frozen_preparation = frozen_preparation

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
            # 也没有记录过润色指纹（skipped/从未润色）。从已译段推导批次一次性补润色。
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
        # 章级术语兜底抽取：润色落盘后再抽取，附属章不抽术语。
        if self.config.pipeline.inflight_glossary and not is_back_matter(
            chapter.title, index=ci, total=request.total_chapters
        ):
            src_text = "\n".join(s.source for s in text_segs)
            tgt_text = "\n".join(s.target or "" for s in text_segs)
            self.extractor.extract_and_store(glossary, src_text, tgt_text, ci)
            store.log_event("chapter_glossary_extracted", chapter=ci)
        source_text = "\n".join(s.source for s in text_segs)
        if self.frozen_book is not None and self.frozen_preparation is not None:
            fp = frozen_input_fingerprint(
                self.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, request.shared.frozen_chapter_index(ci)),
                source_text,
            )
        else:
            fp = polish_input_fingerprint(
                source_text,
                self.polisher.src,
                style,
                punctuation_normalize=self.config.punctuation_normalize,
                model=polish_model_profile(self.config),
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
                batch = text_segs[start : start + count]
                raw_plain = [segment.target or "" for segment in batch]
                futures_by_key[key] = executor.submit(
                    self.polisher.polish,
                    raw_plain,
                    [segment.source for segment in batch],
                    glossary_terms=list(term_snapshot),
                    style=style,
                    strict=True,
                )
        for entry in sorted(pending, key=lambda e: e.start):
            start, count = entry.start, entry.count
            fut = futures_by_key.pop((ci, start), None)
            batch = text_segs[start : start + count]
            srcs = [segment.source for segment in batch]
            raw_plain = [segment.target or "" for segment in batch]
            final_plain = fut.result() if fut is not None else raw_plain
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            results = [
                lint.polish_gate(
                    srcs[i],
                    raw_plain[i],
                    final_plain[i],
                    locked_terms=locked,
                    src_lang=self.polisher.src,
                    normalize_punctuation=self.config.punctuation_normalize,
                )
                for i in range(count)
            ]
            selected_plain = [
                raw_plain[i] if checks.is_machine_literal(srcs[i]) else result.selected
                for i, result in enumerate(results)
            ]
            selected_transport = _align_epub_translations(batch, selected_plain)
            for i, result in enumerate(results):
                if batch[i].epub_state is not None and self.config.punctuation_normalize:
                    selected_transport[i] = normalize_slot_transport(
                        batch[i], selected_transport[i]
                    )
                if result.accepted:
                    self.polisher.client.usage.record_outcome(
                        "editor", "polish.segment", accepted=True
                    )
                    continue
                self.polisher.client.usage.record_outcome(
                    "editor", "polish.segment", accepted=False
                )
                store.log_event(
                    "polish_rejected",
                    chapter=ci,
                    index=start + i,
                    reason=list(result.rejection_reasons),
                    proposal_sha256=stable_digest(result.proposal),
                )
            final = []
            for segment, value in zip(batch, selected_transport, strict=True):
                assign_segment_translation(segment, value)
                final.append(segment.target or "")
            chapter_progress.pending_polish = [
                e for e in chapter_progress.pending_polish if e.start != start
            ]

            changes = [
                {
                    "index": text_segs[start + i].index,
                    "before": raw_plain[i],
                    "after": final[i],
                }
                for i in range(count)
                if raw_plain[i] != final[i]
            ]
            # 崩溃一致性：润色结果在章节文件、清标记在 manifest；先写检查点日志再
            # 落盘，崩溃后由锁内恢复清除已提交标记（避免同一段被润色两次）。
            checkpoint.begin_polish(store, ci, start, count, final)
            store.save_chapter(chapter)
            store.save_progress(ci, chapter_progress)
            checkpoint.clear(store)

            # 提交后再审计：仅记录相对持久化前译文（raw）发生的改动，并用稳定段号
            # 标识改动。例行文本只记录最终 target 的摘要；raw_normalized 仅供 lint
            # 回退使用，并非持久化前的原始值，标点规范化导致的差异也必须如实上报。
            changes = [
                {
                    "index": text_segs[start + i].index,
                    "before": raw_plain[i],
                    "after": final[i],
                }
                for i in range(count)
                if raw_plain[i] != final[i]
            ]
            store.log_event(
                "batch_polished",
                chapter=ci,
                start_index=start,
                count=count,
                changed_count=len(changes),
                changes=changes,
                target_sha256=stable_digest(
                    [
                        {"index": text_segs[start + i].index, "target": final[i]}
                        for i in range(count)
                    ]
                ),
            )
        return None


__all__ = ["PolishNode", "TranslateNode"]
