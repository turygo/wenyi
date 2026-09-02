"""Evidence extraction from the public RunStore contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trans_novel.benchmark.corpus import canonical_json, segment_id, sha256_bytes
from trans_novel.llm.usage import usage_delta


def usage_of(client: Any) -> dict[str, Any]:
    usage = getattr(client, "usage", None)
    return usage.summary() if usage is not None and hasattr(usage, "summary") else {}


def candidate_store(state_dir: Path, *, error_type: type[Exception] = ValueError) -> Any:
    stores = (
        [
            path
            for path in state_dir.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        ]
        if state_dir.exists()
        else []
    )
    if len(stores) != 1:
        raise error_type("candidate Application state root is missing or ambiguous")
    from trans_novel.pipeline.state import RunStore

    return RunStore(str(stores[0]))


def target_hash(rows: list[dict[str, Any]]) -> str:
    ordered = [
        {"chapter": row["chapter_index"], "index": row["segment_index"], "target": row["target"]}
        for row in rows
    ]
    return sha256_bytes(canonical_json(ordered).encode("utf-8"))


def segment_rows(
    store: Any, source_sha256: str, *, error_type: type[Exception] = ValueError
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state = store.load_state()
    qa_node = state.nodes.get("deterministic_qa")
    qa_findings = (qa_node.output or {}).get("issues", []) if qa_node else []
    for chapter in state.chapters:
        progress = store.load_progress(chapter.index)
        lint_findings = progress.lint_issues
        for segment in store.load_chapter(chapter.index).text_segments:
            target = segment.target or ""
            if not target.strip():
                raise error_type("completed candidate contains an empty translation")
            rows.append(
                {
                    "segment_id": segment_id(
                        source_sha256, chapter.index, segment.index, segment.source
                    ),
                    "chapter_index": chapter.index,
                    "chapter_title": chapter.title,
                    "segment_index": segment.index,
                    "kind": segment.kind,
                    "source": segment.source,
                    "target": target,
                    "lint_findings": [
                        item for item in lint_findings if item.get("index") in {None, segment.index}
                    ],
                    "deterministic_findings": [
                        item
                        for item in qa_findings
                        if item.get("chapter") == chapter.index
                        and item.get("index") == segment.index
                    ],
                    "source_sha256": sha256_bytes(segment.source.encode("utf-8")),
                    "target_sha256": sha256_bytes(target.encode("utf-8")),
                }
            )
    if not rows:
        raise error_type("completed candidate contains no translated segments")
    return rows


def clone_usage_delta(
    usage: dict[str, Any], baseline: dict[str, Any] | None = None
) -> dict[str, Any]:
    return usage_delta(usage, baseline or {})


__all__ = [
    "candidate_store",
    "clone_usage_delta",
    "segment_rows",
    "target_hash",
    "usage_of",
]
