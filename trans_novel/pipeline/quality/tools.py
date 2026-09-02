"""Small public glossary operations used by CLI tools."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from trans_novel.glossary import resolver
from trans_novel.glossary.store import GlossaryStore


@contextmanager
def open_glossary(path: str | Path) -> Iterator[GlossaryStore]:
    glossary = GlossaryStore(path)
    try:
        yield glossary
    finally:
        glossary.close()


def lock(glossary: GlossaryStore, source: str) -> None:
    resolver.lock(glossary, source)


def resolve(glossary: GlossaryStore, source: str, target: str) -> None:
    resolver.resolve(glossary, source, target)


__all__ = ["lock", "open_glossary", "resolve"]
