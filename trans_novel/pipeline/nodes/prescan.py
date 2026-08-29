"""Best-effort terminology mining and naming nodes."""

from __future__ import annotations

from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.config import Config
from trans_novel.glossary.miner import mine_candidates
from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    fast_model_profile,
    frozen_input_fingerprint,
    name_terms_input_fingerprint,
    primary_model_profile,
)
from trans_novel.pipeline.state import (
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    SCOPE_BOOK,
    input_fingerprint,
)


def mine_terms_input_fingerprint(
    chapter_texts: list[str], src_lang: str, concurrency: int, model: str = ""
) -> str:
    return input_fingerprint(chapter_texts, src_lang, concurrency, model)


def _import_frozen_glossary(glossary: GlossaryStore, frozen_book) -> None:
    for term in frozen_book.glossary:
        existing = glossary.get_term(term.source)
        if existing is not None and existing != term:
            raise ValueError(f"frozen glossary conflict: {term.source}")
        if existing is None:
            glossary.upsert_term(term)


class MineTermsNode:
    node_id = NODE_MINE_TERMS
    scope = SCOPE_BOOK

    def __init__(self, *, namer, config: Config, frozen_book=None):
        self.namer, self.config, self.frozen_book = namer, config, frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if self.frozen_book is not None:
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                self.frozen_book.book_id,
                [t.source for t in self.frozen_book.glossary],
            )
            return NodeOutcome(fingerprint=fp, artifacts={"candidates": []})
        state = store.load_state()
        total = len(state.chapters)
        src_chapters = [
            (c.index, "\n".join(s.source for s in store.load_chapter(c.index).text_segments))
            for c in state.chapters
            if not is_back_matter(c.title, index=c.index, total=total)
        ]
        on_progress = (
            (lambda i, n: request.progress(i, n, "查找专有名词…")) if request.progress else None
        )
        candidates = mine_candidates(
            self.namer.src,
            src_chapters,
            self.namer,
            concurrency=max(1, self.config.pipeline.prescan_concurrency),
            on_progress=on_progress,
        )
        fp = mine_terms_input_fingerprint(
            [text for _, text in src_chapters],
            self.namer.src,
            self.config.pipeline.prescan_concurrency,
            fast_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp, artifacts={"candidates": candidates})


class NameTermsNode:
    node_id = NODE_NAME_TERMS
    scope = SCOPE_BOOK

    def __init__(
        self, *, namer, analyzer, glossary: GlossaryStore, config: Config, frozen_book=None
    ):
        self.namer, self.analyzer, self.glossary = namer, analyzer, glossary
        self.config, self.frozen_book = config, frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if self.frozen_book is not None:
            _import_frozen_glossary(self.glossary, self.frozen_book)
            state = store.load_state()
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                self.frozen_book.book_id,
                [t.source for t in self.frozen_book.glossary],
            )
            return NodeOutcome(fingerprint=fp, artifacts={"named_terms": []})
        candidates = (request.artifacts.get("mine_terms") or {}).get("candidates")
        if candidates is None:
            raise WorkflowProtocolError("missing_mine_candidates")
        state = store.load_state()
        mine_node = state.nodes.get(NODE_MINE_TERMS)
        mine_fingerprint = mine_node.input_fingerprint if mine_node else ""
        if not mine_fingerprint:
            raise WorkflowProtocolError("missing_mine_fingerprint")
        style = self.analyzer.style_brief(store.load_analysis() or {})
        named = self.namer.name_terms(
            candidates,
            style,
            existing=self.glossary.all_terms(),
            concurrency=max(1, self.config.pipeline.prescan_concurrency),
            on_progress=(lambda i, n: request.progress(i, n, "统一译名…"))
            if request.progress
            else None,
        )
        for term in named:
            self.glossary.upsert_term(term, chapter=0)
            if term.type == TYPE_PERSON:
                self.glossary.confirm_locked(term.source, term.target)
        state.analysis_flags.term_mining_done = True
        store.save_state(state)
        fp = name_terms_input_fingerprint(
            mine_fingerprint,
            style,
            self.config.pipeline.prescan_concurrency,
            primary_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp, artifacts={"named_terms": named})


__all__ = [
    "MineTermsNode",
    "NameTermsNode",
    "mine_terms_input_fingerprint",
    "name_terms_input_fingerprint",
]
