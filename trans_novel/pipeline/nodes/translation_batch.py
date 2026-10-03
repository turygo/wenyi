"""Call-local translation batching, alignment, fallback, and glossary stages."""

from __future__ import annotations

from dataclasses import replace

from trans_novel.epub.richtext_edits import normalize_rich_target
from trans_novel.epub.slots import normalized_source_text, source_passthrough_transport
from trans_novel.ingest.models import (
    KIND_HEADING,
    InlineRun,
    RichTarget,
    Segment,
    validate_rich_target,
)
from trans_novel.llm.errors import LLM_FALLBACK_ERRORS
from trans_novel.pipeline.nodes.common import source_context_before
from trans_novel.pipeline.nodes.glossary import extract_and_store, store_extracted_terms
from trans_novel.pipeline.state import RollingContext
from trans_novel.postprocess.punct import normalize_heading_numbering, normalize_zh


def align_epub_translations(segments, translations: list[str], *, annotator=None) -> list[object]:
    """按语义标注中文范围，保留未修改的既有译文契约。"""
    return _align_epub_translations(segments, translations, annotator=annotator)[0]


def _align_epub_translations(segments, translations: list[str], *, annotator=None):
    result: list[object] = list(translations)
    pending = []
    sources = []
    targets = []
    for index, (segment, translation) in enumerate(zip(segments, translations, strict=True)):
        if segment.epub_state is None:
            continue
        complete = (
            normalize_heading_numbering(translation) if segment.kind == "heading" else translation
        )
        if segment.rich_target is not None and complete == segment.rich_target.text:
            result[index] = segment.rich_target.model_copy(deep=True)
            continue
        if complete == segment.source:
            result[index] = source_passthrough_transport(segment.epub_state)
            continue
        source = segment.epub_state.rich_source
        if source is None:
            raise ValueError("EPUB translation requires a rich source inventory")
        if not source.marks and not source.atoms:
            result[index] = RichTarget(runs=[InlineRun(text=complete)])
            continue
        if annotator is None:
            raise ValueError("formatted EPUB translation requires a rich text annotator")
        pending.append(index)
        sources.append(source)
        targets.append(complete)
    count = 0
    if pending:
        annotated = annotator.annotate_batch(sources, targets)
        if len(annotated.targets) != len(pending):
            raise ValueError("rich annotation batch count mismatch")
        for index, source, expected, target in zip(
            pending, sources, targets, annotated.targets, strict=True
        ):
            validate_rich_target(source, target, expected_text=expected)
            result[index] = target
        count = annotated.request_count
    return result, count


def safe_batch_fallback(batch) -> tuple[list[object], int]:
    return [
        source_passthrough_transport(segment.epub_state)
        if segment.epub_state is not None
        else segment.source
        for segment in batch
    ], 0


def translate_batch(
    translator,
    batch,
    terms,
    context: RollingContext,
    style: str,
    *,
    chapter_segments: list[Segment],
    start_index: int,
    chapter_title: str,
    n_recent: int,
    single_segment_translation: bool = False,
    annotator=None,
    punctuation_normalize: bool = False,
) -> tuple[list[object], int]:
    """Translate one ordinary batch, preserving per-heading prompt semantics."""
    for segment in batch:
        if segment.epub_state is not None and segment.source != normalized_source_text(
            segment.epub_state.slots
        ):
            raise ValueError(f"EPUB source slot coverage mismatch: {segment.resource_href}")
    local_context = replace(context, recent_targets=list(context.recent_targets))
    try:
        translated: list[str] = []
        request_count = 0
        offset = 0
        while offset < len(batch):
            heading = batch[offset].kind == KIND_HEADING
            end = offset + 1
            if not heading and not single_segment_translation:
                while end < len(batch) and batch[end].kind != KIND_HEADING:
                    end += 1
            result = translator.translate_batch(
                [segment.translation_source for segment in batch[offset:end]],
                agent="analyst" if heading else "translator",
                operation=(
                    "translate.heading"
                    if heading
                    else "translate.single"
                    if single_segment_translation
                    else "translate.batch"
                ),
                fallback_agent="analyst" if single_segment_translation and not heading else None,
                glossary_terms=terms,
                style="" if heading else style,
                context="" if heading else local_context.render(n_recent),
                source_context=(
                    "" if heading else source_context_before(chapter_segments, start_index + offset)
                ),
                chapter_title="" if heading else chapter_title,
                kind=KIND_HEADING if heading else None,
            )
            translated.extend(result.translations)
            local_context.add_targets(list(result.translations))
            request_count += result.request_count
            offset = end
    except LLM_FALLBACK_ERRORS:
        return safe_batch_fallback(batch)
    if punctuation_normalize:
        translated = [
            normalize_zh(target)
            if segment.epub_state is None and target != segment.source
            else target
            for segment, target in zip(batch, translated, strict=True)
        ]
    aligned, annotation_count = _align_epub_translations(batch, translated, annotator=annotator)
    if punctuation_normalize:
        aligned = [
            normalize_rich_target(target, source=segment.epub_state.rich_source)
            if isinstance(target, RichTarget)
            else target
            for segment, target in zip(batch, aligned, strict=True)
        ]
    return aligned, request_count + annotation_count


def extract_batch_glossary(
    extractor, glossary, store, chapter: int, start_index: int, batch, existing_terms=None
):
    """Extract and persist one batch's terms, including its event."""
    src_text = "\n".join(s.source for s in batch)
    tgt_text = "\n".join(s.target or "" for s in batch)
    if existing_terms is None:
        summary, changed = extract_and_store(extractor, glossary, src_text, tgt_text, chapter)
    else:
        terms = extractor.extract(src_text, tgt_text, existing_terms)
        summary, changed = store_extracted_terms(glossary, terms, chapter)
    store.log_event(
        "batch_glossary_extracted",
        chapter=chapter,
        start_index=start_index,
        count=len(batch),
        summary=summary,
    )
    return summary, changed


__all__ = [
    "align_epub_translations",
    "extract_batch_glossary",
    "safe_batch_fallback",
    "translate_batch",
]
