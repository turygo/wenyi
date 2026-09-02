"""Public CLI access to an existing translation run."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from trans_novel.config import Config
from trans_novel.ingest import load_document

if TYPE_CHECKING:
    from trans_novel.pipeline.state.store import RunStore


def runstore_for(config: Config, input_path: str) -> RunStore:
    """Resolve a source document to its non-creating persisted run store."""
    from trans_novel.pipeline.state.store import RunStore, slugify

    document = load_document(input_path, config.source_lang, config.target_lang)
    run_dir = os.path.join(config.state_dir, slugify(document.title))
    return RunStore(run_dir, create=False)


__all__ = ["runstore_for"]
