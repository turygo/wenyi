"""翻译节点共用的纯函数辅助。

所有节点共用同一套段落匹配与裁剪口径，避免维护第二套逻辑。
"""

from __future__ import annotations

from trans_novel.glossary.store import GlossaryStore, terms_matching_text
from trans_novel.ingest.segmenter import batch_segments


def chapter_term_snapshot(glossary: GlossaryStore, text_segs, config) -> list:
    """返回当前章节要注入的术语快照；实时入库后可重新调用刷新。"""
    terms = glossary.all_terms()
    if config.pipeline.glossary_scope != "chapter":
        return terms
    src_text = "\n".join(s.source for s in text_segs)
    return terms_matching_text(terms, src_text)


def resume_batches(segments, max_chars: int) -> list[list]:
    """按字符预算分批后，再按“已完成/待翻译”的状态边界拆分（断点续跑不重翻）。"""
    batches: list[list] = []
    for raw_batch in batch_segments(segments, max_chars):
        current: list = []
        current_done: bool | None = None
        for segment in raw_batch:
            done = bool(segment.target and segment.target.strip())
            if current and done != current_done:
                batches.append(current)
                current = []
            current.append(segment)
            current_done = done
        if current:
            batches.append(current)
    return batches


def count_segments(store, chapter_indices: list[int]) -> int:
    total = 0
    for ci in chapter_indices:
        total += len(store.load_chapter(ci).text_segments)
    return total
