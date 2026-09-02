"""Execution bypass for configured front and back matter chapters."""

from __future__ import annotations

from trans_novel.epub.slots import normalize_slot_transport, source_passthrough_transport
from trans_novel.ingest.segmenter import batch_segments
from trans_novel.pipeline.contracts import NodeOutcome
from trans_novel.pipeline.nodes.translation_batch import translate_back_matter_batch
from trans_novel.pipeline.state import STATUS_DONE, stable_digest
from trans_novel.postprocess.punct import normalize_zh


def translate_back_matter(
    mode,
    ci,
    chapter,
    text_segs,
    store,
    request,
    *,
    translator,
    config,
    fingerprint,
) -> NodeOutcome:
    """附属章旁路：skip=原文直通；light=快速粗翻。"""
    label = f"第{ci}章 {chapter.title}"
    store.log_event("chapter_back_matter", chapter=ci, title=chapter.title, mode=mode)
    if mode == "skip":
        for segment in text_segs:
            segment.assign_translation(
                segment.source
                if segment.epub_state is None
                else source_passthrough_transport(segment.epub_state)
            )
        store.save_chapter(chapter)
        request.shared.segments_done += len(text_segs)
        if request.progress:
            request.progress(request.shared.segments_done, request.shared.segments_total, label)
    elif mode == "light":
        seg_base = 0
        for batch in batch_segments(text_segs, config.segment.max_chars_per_batch):
            existing = [s.target for s in batch if s.target and s.target.strip()]
            if len(existing) == len(batch):
                request.shared.segments_done += len(batch)
                seg_base += len(batch)
                if request.progress:
                    request.progress(
                        request.shared.segments_done, request.shared.segments_total, label
                    )
                continue
            raw, call_count = translate_back_matter_batch(translator, batch)
            for segment, target in zip(batch, raw, strict=True):
                if config.punctuation_normalize:
                    if segment.epub_state is None:
                        if target != segment.source:
                            target = normalize_zh(target)
                    else:
                        target = normalize_slot_transport(segment.epub_state, target)
                segment.assign_translation(target)
            targets = [s.target or "" for s in batch]
            store.save_chapter(chapter)
            store.log_event(
                "batch_translated",
                chapter=ci,
                start_index=seg_base,
                count=len(batch),
                polished=False,
                translate_call_count=call_count,
                back_matter=True,
                operation="translate.back_matter",
                target_sha256=stable_digest(
                    [{"index": s.index, "target": t} for s, t in zip(batch, targets, strict=False)]
                ),
            )
            request.shared.segments_done += len(batch)
            seg_base += len(batch)
            if request.progress:
                request.progress(request.shared.segments_done, request.shared.segments_total, label)
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
    fp = fingerprint("\n".join(s.source for s in text_segs), store, mode, ci, request.shared)
    return NodeOutcome(chapter_finalized=True, fingerprint=fp)


__all__ = ["translate_back_matter"]
