"""Document ingestion public API."""

from trans_novel.ingest.models import (
    CANONICAL_TITLE_ID_META,
    CHAPTER_SEMANTICS_VERSION,
    KIND_HEADING,
    Chapter,
    ChapterProcessing,
    Document,
    Segment,
    canonical_title_id,
    chapter_source_digest,
    segment_preserves_source,
)
from trans_novel.ingest.segmenter import load_document

__all__ = [
    "CANONICAL_TITLE_ID_META",
    "CHAPTER_SEMANTICS_VERSION",
    "KIND_HEADING",
    "Chapter",
    "ChapterProcessing",
    "Document",
    "Segment",
    "canonical_title_id",
    "chapter_source_digest",
    "load_document",
    "segment_preserves_source",
]
