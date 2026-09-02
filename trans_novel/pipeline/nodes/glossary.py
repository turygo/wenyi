"""Pipeline orchestration for chapter glossary extraction."""

from __future__ import annotations

from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore, GlossaryTerm


def store_extracted_terms(
    store: GlossaryStore, terms: list[GlossaryTerm], chapter: int
) -> tuple[dict[str, int], list[GlossaryTerm]]:
    summary = {"inserted": 0, "updated": 0, "conflict": 0, "unchanged": 0}
    changed: list[GlossaryTerm] = []
    for term in terms:
        term.first_chapter = chapter
        result = store.upsert_term(term, chapter=chapter)
        summary[result] = summary.get(result, 0) + 1
        if result in ("inserted", "updated"):
            changed.append(term)
    return summary, changed


def extract_and_store(
    extractor,
    store: GlossaryStore,
    source_text: str,
    target_text: str,
    chapter: int,
) -> tuple[dict[str, int], list[GlossaryTerm]]:
    """Extract terms using the agent and persist them through the glossary store."""
    existing = store.all_terms()
    hit = {t.source for t in GlossaryStore.terms_in(existing, source_text)}
    existing = [t for t in existing if t.source in hit or (t.type == TYPE_PERSON and t.locked)]
    terms = extractor.extract(source_text, target_text, existing)
    return store_extracted_terms(store, terms, chapter)
