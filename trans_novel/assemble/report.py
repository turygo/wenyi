"""Deterministic QA report for a translated book."""

from __future__ import annotations

from typing import Any

from trans_novel.glossary.store import GlossaryStore
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import (
    NODE_DETERMINISTIC_QA,
    NODE_FAILED_PERMANENT,
    NODE_FAILED_RETRYABLE,
)


def build_report(store: RunStore, glossary: GlossaryStore) -> dict[str, Any]:
    manifest = store.load_manifest()
    chapters_total = len(manifest["chapters"])
    lint_issues: list[dict] = []
    empty_targets: list[dict] = []
    back_matter: list[dict] = []
    chapters_done = 0
    for chapter_meta in manifest["chapters"]:
        ci = chapter_meta["index"]
        progress = store.load_progress(ci)
        if progress.status != STATUS_DONE:
            continue
        chapters_done += 1
        chapter = store.load_chapter(ci)
        lint_issues.extend(progress.lint_issues)
        if progress.back_matter_mode:
            back_matter.append(
                {
                    "chapter": ci,
                    "title": chapter_meta.get("title", ""),
                    "mode": progress.back_matter_mode,
                }
            )
        for segment in chapter.text_segments:
            if not (segment.target and segment.target.strip()):
                empty_targets.append(
                    {"chapter": ci, "index": segment.index, "source": segment.source[:60]}
                )
    state = store.load_state()
    qa = state.nodes.get(NODE_DETERMINISTIC_QA)
    deterministic_issues = ((qa.output or {}).get("issues") or []) if qa else []
    failed_nodes = [
        {
            "node": key,
            "status": node.status,
            "kind": node.failure.kind if node.failure else None,
            "message": (node.failure.message if node.failure else "")[:200],
            "attempts": node.attempts,
        }
        for key, node in sorted(state.nodes.items())
        if node.status in (NODE_FAILED_RETRYABLE, NODE_FAILED_PERMANENT)
    ]
    conflicts = glossary.open_conflicts()
    low_conf = [
        {
            "source": t.source,
            "target": t.target,
            "type": t.type,
            "confidence": t.confidence,
            "status": t.status,
        }
        for t in glossary.low_confidence_terms()
    ]
    gstats = glossary.stats()
    return {
        "summary": {
            "chapters_total": chapters_total,
            "chapters_done": chapters_done,
            "terms": gstats["terms"],
            "open_conflicts": len(conflicts),
            "lint_issues": len(lint_issues),
            "deterministic_issues": len(deterministic_issues),
            "empty_targets": len(empty_targets),
            "back_matter_chapters": len(back_matter),
            "failed_nodes": len(failed_nodes),
        },
        "open_conflicts": conflicts,
        "low_confidence_terms": low_conf,
        "lint_issues": lint_issues,
        "deterministic_issues": deterministic_issues,
        "empty_targets": empty_targets,
        "back_matter_chapters": back_matter,
        "failed_nodes": failed_nodes,
    }
