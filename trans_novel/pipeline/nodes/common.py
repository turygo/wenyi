"""具体节点共用的确定性工具：术语快照/裁剪、批续跑拆分、风格采样。

从迁移前的单块编排器平移，语义不变；多节点（translate/polish/review/titles/
naturalize）共用同一套匹配口径，避免维护第二套裁剪逻辑。
"""

from __future__ import annotations

import re

from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore
from trans_novel.ingest.segmenter import batch_segments

# 任意文字系统的「词」（字母序列，不含数字/下划线）；CJK 无空格文本会成为整段长 run，
# 由 _person_mentioned 的汉字分支单独处理。
_WORD_RE = re.compile(r"[^\W\d_]+")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _person_mentioned(term, text: str, words: set[str]) -> bool:
    """锁定人物是否以「部分形式」出现在文本里（全名/别名的整体匹配由 terms_in 负责）。"""
    for name in (term.source, *(term.aliases or [])):
        parts = [p for p in _WORD_RE.findall(name) if len(p) >= 2]
        if len(parts) >= 2:
            for part in parts:
                if _HAN_RE.search(part):
                    if part in text:
                        return True
                elif part[0].isupper() and part in words:
                    return True
        elif _HAN_RE.search(name):
            for plen in (2, 3):
                if plen < len(name) and name[:plen] in text:
                    return True
    return False


def terms_matching_text(terms: list, text: str) -> list:
    """按纯文本裁剪术语表：source/alias 命中 + 锁定人物以「部分形式」出现。"""
    hit = {t.source for t in GlossaryStore.terms_in(terms, text)}
    words = set(_WORD_RE.findall(text))
    return [
        t
        for t in terms
        if t.source in hit
        or (t.type == TYPE_PERSON and t.locked and _person_mentioned(t, text, words))
    ]


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


def sample_text(doc, *, labeled: bool = True) -> str:
    """取风格分析样章。labeled=True 多点采样带中文标注；False 返回单段纯源文（语言检测用）。"""
    texts = ["\n".join(s.source for s in ch.text_segments) for ch in doc.chapters]
    texts = [t for t in texts if len(t) > 200]
    if not texts:  # 兜底：全书都是短章
        joined = "\n".join(s.source for ch in doc.chapters[:2] for s in ch.text_segments)
        return joined[:6000]
    if not labeled:
        return texts[0][:6000]
    picks = [(0, "开头样章"), (len(texts) // 2, "中部样章"), (len(texts) - 1, "结尾样章")]
    parts: list[str] = []
    seen: set[int] = set()
    for idx, tag in picks:
        if idx in seen:  # 短书（1-2 章）去重
            continue
        seen.add(idx)
        t = texts[idx]
        chunk = t[-2800:] if tag == "结尾样章" else t[:2800]
        parts.append(f"【{tag}】\n{chunk}")
    return "\n\n".join(parts)


def count_segments(store, chapter_indices: list[int]) -> int:
    total = 0
    for ci in chapter_indices:
        total += len(store.load_chapter(ci).text_segments)
    return total
