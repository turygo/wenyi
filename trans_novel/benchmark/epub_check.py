"""Compatibility facade for the production EPUB verifier.

Benchmark callers retain the historical ``validate_epub`` and
``validate_epub_triplet`` APIs.  Validation rules live in
:mod:`trans_novel.assemble.epub_verifier` so benchmark and production exports
cannot diverge.
"""

from __future__ import annotations

from trans_novel.assemble import epub_verifier

_MAX_MEMBER_BYTES = epub_verifier._MAX_MEMBER_BYTES
_MAX_ARCHIVE_BYTES = epub_verifier._MAX_ARCHIVE_BYTES


def validate_epub(path, *, source_path=None, bilingual=None):
    previous_member_limit = epub_verifier._MAX_MEMBER_BYTES
    previous_archive_limit = epub_verifier._MAX_ARCHIVE_BYTES
    epub_verifier._MAX_MEMBER_BYTES = _MAX_MEMBER_BYTES
    epub_verifier._MAX_ARCHIVE_BYTES = _MAX_ARCHIVE_BYTES
    try:
        return epub_verifier.validate_epub(
            path,
            source_path=source_path,
            bilingual=bilingual,
        )
    finally:
        epub_verifier._MAX_MEMBER_BYTES = previous_member_limit
        epub_verifier._MAX_ARCHIVE_BYTES = previous_archive_limit


def validate_epub_triplet(source_path, mono_path, bilingual_path):
    previous_member_limit = epub_verifier._MAX_MEMBER_BYTES
    previous_archive_limit = epub_verifier._MAX_ARCHIVE_BYTES
    epub_verifier._MAX_MEMBER_BYTES = _MAX_MEMBER_BYTES
    epub_verifier._MAX_ARCHIVE_BYTES = _MAX_ARCHIVE_BYTES
    try:
        return epub_verifier.validate_epub_triplet(source_path, mono_path, bilingual_path)
    finally:
        epub_verifier._MAX_MEMBER_BYTES = previous_member_limit
        epub_verifier._MAX_ARCHIVE_BYTES = previous_archive_limit


__all__ = ["validate_epub", "validate_epub_triplet"]
