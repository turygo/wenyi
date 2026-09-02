"""EPUB assembly public API."""

from trans_novel.assemble.epub.metadata import epub_language, translated_toc_title
from trans_novel.assemble.epub.publication import (
    EpubPublishError,
    EpubVerificationError,
    publish_epub,
)
from trans_novel.assemble.epub.verification import (
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_MEMBERS,
    MAX_MEMBER_BYTES,
    validate_epub,
    validate_epub_triplet,
    verify_epub,
)

__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "MAX_MEMBER_BYTES",
    "EpubPublishError",
    "EpubVerificationError",
    "epub_language",
    "publish_epub",
    "translated_toc_title",
    "validate_epub",
    "validate_epub_triplet",
    "verify_epub",
]
