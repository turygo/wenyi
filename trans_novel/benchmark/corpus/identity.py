"""Pure, deterministic corpus identities and text measurements."""

from __future__ import annotations

import re
from collections.abc import Iterable

from trans_novel.benchmark.artifacts import canonical_json, sha256_bytes

WORD_COUNTER = "en-v1"
WORD_RE = re.compile(r"[A-Za-z]+(?:[’'-][A-Za-z]+)*|\d+(?:[.,]\d+)*")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def source_digest(source: str) -> str:
    return sha256_bytes(source.encode("utf-8"))


def segment_id(book_sha256: str, chapter_index: int, segment_index: int, source: str) -> str:
    return f"{book_sha256}:c{chapter_index:04d}:s{segment_index:04d}:{source_digest(source)[:8]}"


def passage_id(
    book_id: str,
    chapter_index: int,
    start_segment_index: int,
    end_segment_index: int,
    sources: Iterable[str],
) -> str:
    joined = "\n".join(sources)
    return (
        f"{book_id}:c{chapter_index:04d}:s{start_segment_index:04d}-"
        f"{end_segment_index:04d}:{source_digest(joined)[:12]}"
    )


__all__ = [
    "WORD_COUNTER",
    "canonical_json",
    "count_words",
    "passage_id",
    "segment_id",
    "sha256_bytes",
    "source_digest",
]
