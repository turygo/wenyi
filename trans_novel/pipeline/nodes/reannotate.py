"""对已支付的完整译文建立语义格式标注，按批次保存以支持恢复。"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor

from trans_novel.assemble.epub.richtext_pages import settle_page_anchors
from trans_novel.assemble.epub.richtext_sources import hydrate_rich_sources
from trans_novel.epub.richtext import (
    RICHTEXT_VERSION,
    InlineRun,
    RichTarget,
    rich_source_digest,
    validate_rich_target,
)
from trans_novel.epub.richtext_edits import omit_decorations
from trans_novel.ingest.models import segment_preserves_source
from trans_novel.pipeline.nodes.reannotation_inputs import raw_target_digest
from trans_novel.pipeline.state import RunStore
from trans_novel.pipeline.state.models import (
    STATUS_DONE,
    TRANSLATION_POLICY_VERSION,
    source_bytes_hash,
    stable_digest,
)


def _target_digest(chapters) -> str:
    """使用固定章节及段落顺序证明译文没有发生任何字符变化。"""
    return stable_digest(
        [
            [chapter.index, segment.index, segment.target]
            for chapter in chapters
            for segment in chapter.segments
        ]
    )


def _migration_input(store, input_path):
    state = store.load_state()
    digest = source_bytes_hash(input_path)
    old_policy = state.identity.translation_policy_version
    if old_policy not in {3, 4, TRANSLATION_POLICY_VERSION}:
        raise ValueError(
            "rich migration requires legacy policy 3, 4 or the current translation policy"
        )
    if state.fmt != "epub" or state.meta.get("epub_schema") != 4:
        raise ValueError("rich migration requires a completed schema-4 EPUB run")
    if (
        not digest
        or digest != state.identity.source_bytes_sha256
        or digest != state.meta.get("epub_sha256")
    ):
        raise ValueError("rich migration source hash mismatch")
    if any(
        state.progress.get(chapter.index) is None
        or state.progress[chapter.index].status != STATUS_DONE
        for chapter in state.chapters
    ):
        raise ValueError("rich migration requires all chapters to be complete")
    if any(progress.pending_polish for progress in state.progress.values()):
        raise ValueError("rich migration requires completed polish checkpoints")
    original = raw_target_digest(store, [chapter.index for chapter in state.chapters])
    chapters = [
        store.load_chapter(chapter.index)
        for chapter in sorted(state.chapters, key=lambda c: c.index)
    ]
    if any(segment.target is None for chapter in chapters for segment in chapter.segments):
        raise ValueError("rich migration requires complete saved targets")
    if _target_digest(chapters) != original:
        raise ValueError("rich migration model loading changed saved target text")
    record = state.meta.get("richtext_migration")
    expected = {"source_sha256": digest, "target_sha256": original}
    if record is not None and (
        not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items())
    ):
        raise ValueError("rich migration source or saved target changed during resume")
    old_inventories = {
        (chapter.index, segment.index): segment.epub_state.rich_source.model_copy(deep=True)
        for chapter in chapters
        for segment in chapter.segments
        if segment.epub_state is not None and segment.epub_state.rich_source is not None
    }
    protected = {}
    for marker in state.meta.get("epub_notes", {}).get("markers", []):
        protected.setdefault(marker["resource_href"], set()).add(tuple(marker["path"]))
    restored = hydrate_rich_sources(
        input_path,
        chapters,
        protected_paths=protected,
        restore_legacy_whitespace=old_policy == 3,
    )
    old_sources = _reuse_decoration_annotations(chapters, old_inventories)
    state.meta["richtext_migration"] = {**(record or {}), **expected}
    added = []
    if restored:
        audit = state.meta["richtext_migration"].setdefault("legacy_whitespace_restored", [])
        known = {stable_digest(item) for item in audit}
        added = [item for item in restored if stable_digest(item) not in known]
        audit.extend(added)
    store.save_state(state)
    if restored and added:
        store.log_event("richtext_legacy_whitespace_restored", records=added)
    return state, chapters, original, old_sources


def _legacy_geometry_matches(previous, source) -> bool:
    """原始几何必须相同；旧 CSS 证据不是身份，新语义仍需重新严格校验。"""
    if previous.runs != source.runs or previous.atoms != source.atoms:
        return False
    fields = ("id", "path", "tag", "attributes", "source_text", "kind")
    return len(previous.marks) == len(source.marks) and all(
        all(getattr(old, field) == getattr(new, field) for field in fields)
        for old, new in zip(previous.marks, source.marks, strict=True)
    )


def _reuse_decoration_annotations(chapters, old_inventories):
    """只接受已重新核对原书的装饰分类变化，不绕过其他来源变更。"""
    old_sources = {}
    for chapter in chapters:
        for segment in chapter.segments:
            if segment.rich_target is None:
                if "richtext_pending" in segment.meta:
                    segment.meta["richtext_pending"] = {
                        "version": 2,
                        "source_sha256": rich_source_digest(segment.epub_state.rich_source),
                        "target_sha256": stable_digest(segment.target),
                    }
                continue
            key = chapter.index, segment.index
            previous = old_inventories.get(key)
            source = segment.epub_state.rich_source
            if previous is not None:
                current_marks = {mark.id: mark for mark in source.marks}
                for mark in previous.marks:
                    current = current_marks.get(mark.id)
                    if (
                        current is not None
                        and mark.kind == "style"
                        and current.kind == "decoration"
                    ):
                        mark.kind = "decoration"
            legacy = segment.rich_target.version == 1
            matches = previous is not None and (
                _legacy_geometry_matches(previous, source)
                if legacy
                else previous.model_copy(update={"note_references": source.note_references})
                == source
            )
            candidate = omit_decorations(segment.rich_target, source)
            if legacy:
                candidate = candidate.model_copy(update={"version": RICHTEXT_VERSION})
            if matches:
                try:
                    validate_rich_target(source, candidate, expected_text=segment.target)
                except ValueError:
                    matches = False
            if matches:
                segment.assign_translation(candidate)
                segment.meta.pop("richtext_pending", None)
                old_sources[key] = rich_source_digest(source)
            else:
                segment.rich_target = None
                for slot in segment.epub_state.slots:
                    slot.target_value = None
                segment.meta["richtext_pending"] = {
                    "version": 2,
                    "source_sha256": rich_source_digest(source),
                    "target_sha256": stable_digest(segment.target),
                }
    return old_sources


def _annotation_jobs(chapters, old_sources, batch_size):
    jobs = []
    reused = 0
    deterministic = 0
    for chapter in chapters:
        pending = []
        for segment in chapter.segments:
            source = segment.epub_state.rich_source
            if segment_preserves_source(segment) or segment.target == segment.source:
                reused += 1
                continue
            old = old_sources.get((chapter.index, segment.index))
            if segment.rich_target is not None and old == rich_source_digest(source):
                validate_rich_target(source, segment.rich_target, expected_text=segment.target)
                reused += 1
                continue
            if not source.marks and not source.atoms:
                target = RichTarget(runs=[InlineRun(text=segment.target)])
                validate_rich_target(source, target, expected_text=segment.target)
                segment.assign_translation(target)
                segment.meta.pop("richtext_pending", None)
                deterministic += 1
                continue
            pending.append(segment)
        jobs.extend(
            (chapter, pending[start : start + batch_size])
            for start in range(0, len(pending), batch_size)
        )
    return jobs, reused, deterministic


def _annotate_batch(annotator, segments):
    sources = [segment.epub_state.rich_source.model_copy(deep=True) for segment in segments]
    texts = [segment.target for segment in segments]
    result = annotator.annotate_batch(sources, texts)
    if len(result.targets) != len(segments):
        raise ValueError("rich migration annotation batch count mismatch")
    for source, target, text in zip(sources, result.targets, texts, strict=True):
        validate_rich_target(source, target, expected_text=text)
    return result


def _commit_batches(store, jobs, annotator, progress, total, reused):
    annotated = 0
    requests = 0
    pending = deque()
    iterator = iter(jobs)
    first_error = None
    first_failure = None
    with ThreadPoolExecutor(max_workers=4) as executor:

        def schedule():
            job = next(iterator, None)
            if job is not None:
                chapter, segments = job
                pending.append(
                    (chapter, segments, executor.submit(_annotate_batch, annotator, segments))
                )

        for _ in range(4):
            schedule()
        while pending:
            chapter, segments, future = pending.popleft()
            try:
                result = future.result()
            except BaseException as error:
                if first_error is None:
                    first_error = error
                    first_failure = {
                        "chapter_index": chapter.index,
                        "segment_indices": [segment.index for segment in segments],
                        "error_type": type(error).__name__,
                        "error_detail": str(error),
                    }
                continue
            for segment, target in zip(segments, result.targets, strict=True):
                expected = segment.target
                target = omit_decorations(target, segment.epub_state.rich_source)
                segment.assign_translation(target)
                if segment.target != expected:
                    raise ValueError("rich migration assignment changed saved target")
                segment.meta.pop("richtext_pending", None)
            store.save_chapter(chapter)
            annotated += len(segments)
            requests += result.request_count
            if progress:
                progress(reused + annotated, total, "恢复译文语义格式…")
            if first_error is None:
                schedule()
    if first_error is not None:
        store.log_event(
            "richtext_migration_interrupted",
            annotated=annotated,
            model_requests=requests,
            failed_batch=first_failure,
        )
        raise first_error
    return annotated, requests


def reannotate_store(
    store: RunStore, input_path: str, *, annotator, progress=None, batch_size: int = 8
) -> dict:
    """调用方持运行锁；只新增格式证据和批次检查点，禁止改写已有文字。"""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("rich migration batch size must be positive")
    state, chapters, original, old_sources = _migration_input(store, input_path)
    jobs, reused, deterministic = _annotation_jobs(chapters, old_sources, batch_size)
    total = sum(len(chapter.segments) for chapter in chapters)
    for chapter in chapters:
        store.save_chapter(chapter)
    if progress:
        progress(reused + deterministic, total, "恢复译文语义格式…")
    annotated, requests = _commit_batches(
        store, jobs, annotator, progress, total, reused + deterministic
    )
    annotated += deterministic
    page_anchors_moved = settle_page_anchors(input_path, chapters)
    for chapter in chapters:
        store.save_chapter(chapter)
    persisted_raw = raw_target_digest(store, [chapter.index for chapter in chapters])
    persisted = [store.load_chapter(chapter.index) for chapter in chapters]
    if persisted_raw != original or _target_digest(persisted) != original:
        raise ValueError("rich migration changed saved target text")
    state.identity.translation_policy_version = TRANSLATION_POLICY_VERSION
    state.meta["richtext_version"] = RICHTEXT_VERSION
    state.meta["richtext_target_version"] = max(
        (
            segment.rich_target.version
            for chapter in chapters
            for segment in chapter.segments
            if segment.rich_target is not None
        ),
        default=2,
    )
    state.meta["richtext_migration"]["complete"] = True
    store.save_state(state)
    report = {
        "segments": total,
        "annotated": annotated,
        "reused": reused,
        "model_requests": requests,
        "deterministic": deterministic,
        "page_anchors_moved": page_anchors_moved,
        "target_sha256": original,
        "target_contract_version": state.meta["richtext_target_version"],
    }
    store.log_event("richtext_migration_completed", **report)
    return report


__all__ = ["reannotate_store"]
