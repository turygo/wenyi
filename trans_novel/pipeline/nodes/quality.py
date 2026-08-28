"""章级质量节点：naturalize / review / backtranslate。

- naturalize：去翻译腔闭环（三道关卡），幂等靠进度 naturalized 标记；
- review：整章分块审校；autofix_severe 时同步并含严重项定向重译（写回正文），
  否则异步提交（独立 review_executor 防嵌套死锁），结果由 runner 排干后
  finish() 写回；协议错误只做块内恢复，provider 异常原样冒泡；
- backtranslate：按采样率回译抽检；rate=0 时清残留回译问题并收尾本章
  （章链末节点，finalize_chapter）。
"""

from __future__ import annotations

import hashlib
from fractions import Fraction

from trans_novel.agents.naturalizer import naturalize_chapter
from trans_novel.agents.reviewer import ReviewOutputError
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import (
    assign_segment_translation,
    normalize_slot_transport,
    translation_text,
)
from trans_novel.pipeline import checks, lint
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    backtranslate_input_fingerprint,
    editor_fast_model_profile,
    fast_model_profile,
    frozen_input_fingerprint,
    naturalize_input_fingerprint,
    primary_fast_model_profile,
    review_input_fingerprint,
)
from trans_novel.pipeline.nodes.common import chapter_term_snapshot
from trans_novel.pipeline.runstore import stable_digest
from trans_novel.pipeline.state import (
    NODE_BACKTRANSLATE,
    NODE_NATURALIZE,
    NODE_REVIEW,
    SCOPE_CHAPTER,
)
from trans_novel.postprocess.punct import normalize_zh


class NaturalizeNode:
    """去翻译腔闭环：审读 → 改写 → 三道关卡 → 写回（幂等：已处理过则跳过）。"""

    node_id = NODE_NATURALIZE
    scope = SCOPE_CHAPTER

    def __init__(
        self,
        *,
        naturalizer,
        glossary: GlossaryStore,
        config: Config,
        frozen_book=None,
        frozen_preparation=None,
    ):
        self.naturalizer = naturalizer
        self.glossary = glossary
        self.config = config
        self.frozen_book = frozen_book
        self.frozen_preparation = frozen_preparation

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        source_text = "\n".join(s.source for s in store.load_chapter(ci).text_segments)
        if self.frozen_book is not None and self.frozen_preparation is not None:
            fp = frozen_input_fingerprint(
                self.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, request.shared.frozen_chapter_index(ci)),
                source_text,
            )
        else:
            fp = naturalize_input_fingerprint(
                source_text,
                punctuation_normalize=self.config.punctuation_normalize,
                model=editor_fast_model_profile(self.config),
            )
        progress = store.load_progress(ci)
        if progress.naturalized:
            return NodeOutcome(fingerprint=fp)  # 幂等：续跑不重复
        chapter = store.load_chapter(ci)
        glossary = self.glossary
        term_snapshot = chapter_term_snapshot(glossary, chapter.text_segments, self.config)
        locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
        naturalize_chapter(
            self.naturalizer,
            chapter,
            ci,
            request.total_chapters,
            locked,
            self.config,
            store,
            dry_run=False,
            remaining=None,
            strict_screen=True,  # 筛查失败必须冒泡，不得把整章标记为已自然化
        )
        return NodeOutcome(fingerprint=fp)


class ReviewNode:
    """整章审校：同步（autofix_severe）或异步（review_pending 持久标记 + finish 排干）。"""

    node_id = NODE_REVIEW
    scope = SCOPE_CHAPTER

    def __init__(
        self,
        *,
        reviewer,
        translator,
        glossary: GlossaryStore,
        config: Config,
        style_brief: str,
        frozen_book=None,
        frozen_preparation=None,
    ):
        self.reviewer = reviewer
        self.translator = translator
        self.glossary = glossary
        self.config = config
        self.style_brief = style_brief
        self.frozen_book = frozen_book
        self.frozen_preparation = frozen_preparation

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        source_text = "\n".join(s.source for s in store.load_chapter(ci).text_segments)
        if self.frozen_book is not None and self.frozen_preparation is not None:
            fp = frozen_input_fingerprint(
                self.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, request.shared.frozen_chapter_index(ci)),
                (source_text, self.config.pipeline.autofix_severe),
            )
        else:
            fp = review_input_fingerprint(
                source_text,
                autofix_severe=self.config.pipeline.autofix_severe,
                review_output_retries=self.config.pipeline.review_output_retries,
                model=primary_fast_model_profile(self.config),
            )
        if not self.config.pipeline.review:
            return NodeOutcome(fingerprint=fp)  # 不应到达（planner 已标记 skipped）
        chapter = store.load_chapter(ci)
        pairs = [(s.source, s.target or "") for s in chapter.text_segments]
        glossary = self.glossary
        term_snapshot = chapter_term_snapshot(glossary, chapter.text_segments, self.config)
        # 同步/autofix 路径按“当前策略 + 计划失效”决定，而不是旧章完成位：
        # autofix 从关闭切到启用时，已完成的章被重新规划 → 必须走同步严重项重译；
        # review_pending 标记（真正在途的异步审校）保持异步续跑路径。
        review_pending = store.load_progress(ci).review_pending
        if self.config.pipeline.autofix_severe and not review_pending:
            # 严重项定向重译要写回正文，必须留在关键路径上，完全同步（现状不变）。
            new_issues = self.review_chapter(pairs, list(term_snapshot), request.review_executor)
            style = self.style_brief
            book_synopsis = (store.load_analysis() or {}).get("book_synopsis", "")
            chapter_digest = store.load_progress(ci).source_digest
            accepted = self._autofix_severe(
                chapter.text_segments,
                new_issues,
                term_snapshot,
                style,
                book_synopsis,
                chapter_digest,
                store=store,
                chapter_index=ci,
            )
            # 先落盘正文改动，再标 fixed 并保存审校项——崩溃窗口不出现“报告称已修复
            # 但正文仍是修复前译文”的状态。无采纳改动时，译文在翻译或润色后未再
            # 变化，因此跳过整章写入。
            if accepted:
                store.save_chapter(chapter)
            for it in new_issues:
                it["chapter"] = ci
                it.setdefault("fixed", False)
                it["stage"] = "review"
            progress = store.load_progress(ci)
            lint_kept = [i for i in progress.review_issue_dicts() if i.get("stage") == "lint"]
            progress.set_review_issue_dicts(lint_kept + new_issues)
            store.save_progress(ci, progress)
            # 采纳事件在正文与审校项都提交后才发出。
            for entry in accepted:
                store.log_event("autofix_applied", **entry)
            # 审校项补齐 chapter/stage/fixed 并随进度落盘后，才发出审校事件。
            # 摘要必须以当前持久化的最终审校项为准；ReviewIssue 模型会将
            # fixed: 0 等原始值归一化为 False，因此应根据归一化后的持久化数据
            # 计算事件摘要。
            persisted_review_issues = [
                i for i in progress.review_issue_dicts() if i.get("stage") == "review"
            ]
            store.log_event(
                "chapter_reviewed",
                chapter=ci,
                issue_count=len(persisted_review_issues),
                issues_sha256=stable_digest(persisted_review_issues),
            )
            return NodeOutcome(findings_count=len(new_issues), fingerprint=fp)
        # 异步：提交共享线程池，保持 review_pending 持久标记；崩溃续跑据此补跑。
        fut = request.executor.submit(
            self.review_chapter, pairs, list(term_snapshot), request.review_executor
        )
        store.set_review_pending(ci, True)
        return NodeOutcome(async_handle=fut, fingerprint=fp)

    def finish(self, request: NodeRequest, handle) -> None:
        """异步审校排干：合并 lint 项、清 review_pending、发 chapter_reviewed。"""
        ci = request.ci
        store = request.store
        new_issues = handle.result()
        for it in new_issues:
            it["chapter"] = ci
            it.setdefault("fixed", False)
            it["stage"] = "review"
        progress = store.load_progress(ci)
        lint_kept = [i for i in progress.review_issue_dicts() if i.get("stage") == "lint"]
        progress.set_review_issue_dicts(lint_kept + new_issues)
        store.save_progress(ci, progress)
        store.set_review_pending(ci, False)
        # 审校项补齐 chapter/stage/fixed 并随进度落盘后，才发出审校事件。
        # 摘要必须以当前持久化的最终审校项为准；ReviewIssue 模型会将
        # fixed: 0 等原始值归一化为 False，因此应根据归一化后的持久化数据
        # 计算事件摘要。
        persisted_review_issues = [
            i for i in progress.review_issue_dicts() if i.get("stage") == "review"
        ]
        store.log_event(
            "chapter_reviewed",
            chapter=ci,
            issue_count=len(persisted_review_issues),
            issues_sha256=stable_digest(persisted_review_issues),
        )

    # ── 整章分块审校 ──────────────────────────────────────────────────────
    def review_chapter(self, pairs: list[tuple[str, str]], terms, review_executor) -> list[dict]:
        """整章分块审校，并在协议错误时递归拆分或重试单段。

        顶层块按源文字符预算连续打包；chunk 0 先完成以预热 provider 前缀缓存，
        其余顶层块再并发提交，并按源文顺序合并结果。
        """
        budget = self.config.segment.max_chars_per_batch * 3
        chunks = self._pack_contiguous(pairs, budget)
        warm = len(chunks) > 1
        results: list[list[dict]] = [[] for _ in chunks]
        if warm:
            first = chunks[0]
            results[0] = review_executor.submit(self._review_chunk, first, terms).result()
        pending = chunks[1:] if warm else chunks
        offset = 1 if warm else 0
        futures = [review_executor.submit(self._review_chunk, chunk, terms) for chunk in pending]
        for i, fut in enumerate(futures):
            results[offset + i] = fut.result()

        issues: list[dict] = []
        base = 0
        for chunk, result in zip(chunks, results, strict=False):
            for it in result:
                idx = it.get("index")
                if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(chunk):
                    raise ReviewOutputError("invalid_issue_index")
                it["index"] = base + idx
                issues.append(it)
            base += len(chunk)
        return issues

    def _review_chunk(self, chunk: list[tuple[str, str]], terms) -> list[dict]:
        """审校一个顶层块；仅协议错误进入递归恢复。"""
        try:
            return self.reviewer.review([s for s, _ in chunk], [t for _, t in chunk], terms)
        except ReviewOutputError as exc:
            return self._recover_review_chunk(chunk, terms, exc)

    def _recover_review_chunk(
        self,
        chunk: list[tuple[str, str]],
        terms,
        first_error: ReviewOutputError,
    ) -> list[dict]:
        """恢复一个已失败的块，返回相对该块起点的本地索引。"""
        if len(chunk) > 1:
            middle = len(chunk) // 2
            left = self._review_chunk(chunk[:middle], terms)
            right = self._review_chunk(chunk[middle:], terms)
            for issue in right:
                issue["index"] += middle
            return left + right

        last_error = first_error
        for _ in range(self.config.pipeline.review_output_retries):
            try:
                return self.reviewer.review([s for s, _ in chunk], [t for _, t in chunk], terms)
            except ReviewOutputError as exc:
                last_error = exc
        raise last_error

    @staticmethod
    def _pack_contiguous(pairs: list[tuple[str, str]], budget: int) -> list[list]:
        """按源文字符预算把 (source, target) 对保序打包成若干连续块。"""
        chunks: list[list] = []
        cur: list = []
        size = 0
        for p in pairs:
            src = p[0]
            if cur and size + len(src) > budget:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += len(src)
        if cur:
            chunks.append(cur)
        return chunks

    _SEVERE_TYPES = ("missing", "mistranslation")

    def _autofix_severe(
        self,
        text_segs,
        issues,
        terms,
        style,
        book_synopsis: str = "",
        chapter_digest: str = "",
        *,
        store=None,
        chapter_index: int | None = None,
    ) -> list[dict]:
        """对审校严重项（漏译/误译）带审校意见定向重译，每段最多一次。

        采纳条件 = 重译非空且过长度校验：采纳则标点规范化后更新 seg.target 并标
        fixed=True；不采纳保持 fixed=False 留人工。
        返回本段被采纳的审计条目（chapter/index/before/after/issues 快照）；
        autofix_applied 事件由调用方在正文与审校项都落盘后统一发出——保证
        “正文已提交”先于“采纳事件”。拒绝事件仍用于记录处理过程，因此在原处发出，
        只带摘要与提案指纹，不带 source/before/proposed 明文。
        """
        accepted_entries: list[dict] = []
        locked_terms = [term for term in terms if getattr(term, "locked", False)]
        by_seg: dict[int, list[dict]] = {}
        for it in issues:
            if it.get("type") in self._SEVERE_TYPES:
                by_seg.setdefault(it["index"], []).append(it)
        for idx, seg_issues in sorted(by_seg.items()):
            seg = text_segs[idx]
            before = "\n".join(text_segs[j].target or "" for j in range(max(0, idx - 2), idx))
            after = "\n".join(
                text_segs[j].target or "" for j in range(idx + 1, min(len(text_segs), idx + 3))
            )
            feedback = "；".join(
                f"{it.get('detail', '')}（建议：{it.get('suggestion', '')}）" for it in seg_issues
            )
            new_transport = self.translator.retranslate_with_feedback(
                seg.source,
                feedback=feedback,
                operation="translate.review_fix",
                glossary_terms=terms,
                style=style,
                context_before=before,
                context_after=after,
                book_synopsis=book_synopsis,
                chapter_digest=chapter_digest,
                segment=seg,
            )
            try:
                new_t = (
                    translation_text(seg, new_transport)
                    if seg.epub_state is not None
                    else new_transport
                )
            except (TypeError, ValueError):
                new_t = ""
            lint_gate = (
                lint.polish_gate(
                    seg.source,
                    seg.target or "",
                    new_t,
                    locked_terms=locked_terms,
                    src_lang=self.config.source_lang,
                    normalize_punctuation=False,
                )
                if isinstance(new_t, str)
                else None
            )
            accepted = (
                lint_gate is not None
                and bool(new_t)
                and lint_gate.accepted
                and not lint.drops_dialogue_quotes(seg.source, seg.target or "", new_t)
                and not checks.length_flags([seg.source], [new_t])
            )
            usage = getattr(getattr(self.translator, "client", None), "usage", None)
            if usage is not None:
                usage.record_outcome("translator", "translate.review_fix", accepted=accepted)
            if accepted:
                final_transport = new_transport
                if self.config.punctuation_normalize:
                    final_transport = (
                        normalize_zh(str(new_t))
                        if seg.epub_state is None
                        else normalize_slot_transport(seg, new_transport)
                    )
                old_t = seg.target
                assign_segment_translation(seg, final_transport)
                for it in seg_issues:
                    it["fixed"] = True
                accepted_entries.append(
                    {
                        "chapter": chapter_index,
                        "index": idx,
                        "before": old_t,
                        "after": seg.target,
                        "issues": [dict(it) for it in seg_issues],
                    }
                )
            elif store is not None:
                store.log_event(
                    "autofix_rejected",
                    chapter=chapter_index,
                    index=idx,
                    reason=sorted({str(it.get("type", "")) for it in seg_issues}),
                    issues_sha256=stable_digest(seg_issues),
                    proposal_sha256=stable_digest(new_t),
                )
        return accepted_entries


class BacktranslateNode:
    """回译抽检（rate=0 时仅清残留回译问题）；章链末节点，负责收尾 done。"""

    node_id = NODE_BACKTRANSLATE
    scope = SCOPE_CHAPTER

    def __init__(
        self,
        *,
        backtrans,
        config: Config,
        frozen_book=None,
        frozen_preparation=None,
        backtranslation_sample_scope: str = "",
    ):
        self.backtrans = backtrans
        self.config = config
        self.frozen_book = frozen_book
        self.frozen_preparation = frozen_preparation
        self.backtranslation_sample_scope = backtranslation_sample_scope

    def _sample_key(self, chapter_index: int, rate: float, text_segs) -> str:
        segments = [
            {
                "index": seg.index,
                "source_sha256": hashlib.sha256(seg.source.encode("utf-8")).hexdigest(),
            }
            for seg in text_segs
        ]
        return stable_digest(
            {
                "scope": self.backtranslation_sample_scope,
                "chapter": chapter_index,
                "rate": rate,
                "segments": segments,
            }
        )

    @staticmethod
    def _valid_persisted_indices(progress, text_segs, sample_key: str) -> list[int] | None:
        if progress.backtranslation_sample_key != sample_key:
            return None
        positions = {seg.index: position for position, seg in enumerate(text_segs)}
        indices = progress.backtranslation_sample_indices
        if (
            any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or len(set(indices)) != len(indices)
            or any(index not in positions for index in indices)
        ):
            return None
        ordered_positions = [positions[index] for index in indices]
        if ordered_positions != sorted(ordered_positions):
            return None
        return list(indices)

    def _select_indices(self, sample_key: str, rate: float, text_segs) -> list[int]:
        if rate <= 0:
            return []
        if rate >= 1:
            return [seg.index for seg in text_segs]
        fraction = Fraction(str(rate))
        scale = 1 << 256
        return [
            seg.index
            for seg in text_segs
            if int.from_bytes(
                hashlib.sha256(f"{sample_key}:{seg.index}".encode()).digest(),
                "big",
            )
            * fraction.denominator
            < fraction.numerator * scale
        ]

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        chapter = store.load_chapter(ci)
        text_segs = chapter.text_segments
        rate = self.config.pipeline.backtranslate_sample
        progress = store.load_progress(ci)
        sample_key = self._sample_key(ci, rate, text_segs)
        sample_indices = self._valid_persisted_indices(progress, text_segs, sample_key)
        if sample_indices is None:
            sample_indices = self._select_indices(sample_key, rate, text_segs)
            progress.backtranslation_sample_key = sample_key
            progress.backtranslation_sample_indices = sample_indices
        # Commit the selection before any provider call, including an empty selection.
        store.save_progress(ci, progress)
        segments_by_index = {seg.index: seg for seg in text_segs}
        bt_samples = [
            (segments_by_index[index].source, segments_by_index[index].target or "")
            for index in sample_indices
        ]
        bt_issues: list[dict] = []
        if bt_samples:
            srcs = [a for a, _ in bt_samples]
            tgts = [b for _, b in bt_samples]
            for it in self.backtrans.check(srcs, tgts, strict=True):
                it["chapter"] = ci
                bt_issues.append(it)
            store.log_event(
                "chapter_backtranslation_checked",
                chapter=ci,
                sample_count=len(bt_samples),
                issue_count=len(bt_issues),
                issues_sha256=stable_digest(bt_issues),
            )
        progress.set_backtranslation_issue_dicts(bt_issues)
        store.save_progress(ci, progress)
        source_text = "\n".join(s.source for s in text_segs)
        if self.frozen_book is not None and self.frozen_preparation is not None:
            fp = frozen_input_fingerprint(
                self.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, request.shared.frozen_chapter_index(ci)),
                (source_text, rate, self.backtranslation_sample_scope),
            )
        else:
            fp = backtranslate_input_fingerprint(
                source_text,
                backtranslate_sample=rate,
                model=(
                    f"{fast_model_profile(self.config)}"
                    f"|backtranslation_sample_scope={self.backtranslation_sample_scope}"
                ),
            )
        return NodeOutcome(findings_count=len(bt_issues), fingerprint=fp)


__all__ = ["BacktranslateNode", "NaturalizeNode", "ReviewNode"]
