"""Internal immutable source for benchmark-frozen preparation data.

This module intentionally has no dependency on :mod:`trans_novel.benchmark`.
The benchmark importer constructs these values at its boundary and production
nodes consume only this protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from trans_novel.glossary.store import GlossaryTerm
from trans_novel.pipeline.state.models import input_fingerprint


def frozen_input_fingerprint(
    preparation_sha256: str,
    node_id: str,
    source_mapping=None,
    content=None,
) -> str:
    """Fingerprint a node against immutable frozen preparation semantics.

    Candidate model roles are deliberately absent: a candidate may change
    without regenerating preparation, while a changed bundle invalidates all
    frozen consumers.
    """
    return input_fingerprint(
        "frozen-preparation-v1",
        preparation_sha256,
        node_id,
        source_mapping,
        content,
    )


@dataclass(frozen=True, slots=True)
class FrozenBookPreparation:
    book_id: str
    source_sha256: str
    analysis: Mapping[str, Any]
    style: str
    style_brief: str
    glossary: tuple[GlossaryTerm, ...]
    node_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "analysis", MappingProxyType(dict(self.analysis)))
        object.__setattr__(
            self, "node_fingerprints", MappingProxyType(dict(self.node_fingerprints))
        )
        object.__setattr__(self, "glossary", tuple(self.glossary))


@runtime_checkable
class FrozenPreparationSource(Protocol):
    """Lookup protocol used by production nodes for a completed frozen book."""

    preparation_sha256: str

    def book_for(self, *, book_id: str, source_sha256: str) -> FrozenBookPreparation:
        """Return the exact book or raise a permanent integrity error."""

    def node_fingerprint(
        self,
        *,
        book: FrozenBookPreparation,
        node_id: str,
        source_mapping: Any = None,
        content: Any = None,
    ) -> str:
        """Return a stable fingerprint for a frozen node input."""


class FrozenPreparationMap:
    """Small production-side implementation for imported completed bundles."""

    def __init__(self, preparation_sha256: str, books: Mapping[str, FrozenBookPreparation]):
        self.preparation_sha256 = preparation_sha256
        self._books = MappingProxyType(dict(books))

    def book_for(self, *, book_id: str, source_sha256: str) -> FrozenBookPreparation:
        book = self._books.get(book_id)
        if book is None or book.source_sha256 != source_sha256:
            raise ValueError(f"frozen preparation identity mismatch: {book_id}")
        return book

    def node_fingerprint(
        self,
        *,
        book: FrozenBookPreparation,
        node_id: str,
        source_mapping: Any = None,
        content: Any = None,
    ) -> str:
        return frozen_input_fingerprint(
            self.preparation_sha256,
            node_id,
            source_mapping,
            content,
        )


__all__ = [
    "FrozenBookPreparation",
    "FrozenPreparationMap",
    "FrozenPreparationSource",
    "frozen_input_fingerprint",
]
