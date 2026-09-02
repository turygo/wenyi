"""Facade for canonical production EPUB verification."""

from __future__ import annotations

from trans_novel.assemble.epub.verification import (
    MAX_ARCHIVE_BYTES,
    MAX_MEMBER_BYTES,
)
from trans_novel.assemble.epub.verification import (
    validate_epub_triplet_with_limits as _validate_epub_triplet,
)
from trans_novel.assemble.epub.verification import (
    validate_epub_with_limits as _validate_epub,
)

_MAX_MEMBER_BYTES = MAX_MEMBER_BYTES
_MAX_ARCHIVE_BYTES = MAX_ARCHIVE_BYTES


def validate_epub(path, *, source_path=None, bilingual=None):
    return _validate_epub(
        path,
        source_path=source_path,
        bilingual=bilingual,
        max_member_bytes=_MAX_MEMBER_BYTES,
        max_archive_bytes=_MAX_ARCHIVE_BYTES,
    )


def validate_epub_triplet(source_path, mono_path, bilingual_path):
    return _validate_epub_triplet(
        source_path,
        mono_path,
        bilingual_path,
        max_member_bytes=_MAX_MEMBER_BYTES,
        max_archive_bytes=_MAX_ARCHIVE_BYTES,
    )


__all__ = ["validate_epub", "validate_epub_triplet"]
