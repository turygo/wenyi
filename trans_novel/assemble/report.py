"""QA 报告：把所有需要人工关注的点集中汇总。

人工只需看这一处，即可裁决术语冲突、补查疑似漏译/误译。
"""

from __future__ import annotations

from typing import Any

from trans_novel.glossary.store import GlossaryStore
from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import NODE_FAILED_PERMANENT, NODE_FAILED_RETRYABLE


def build_report(store: RunStore, glossary: GlossaryStore) -> dict[str, Any]:
    m = store.load_manifest()
    chapters_total = len(m["chapters"])

    review_issues: list[dict] = []
    bt_issues: list[dict] = []
    empty_targets: list[dict] = []
    back_matter: list[dict] = []
    chapters_done = 0

    for c in m["chapters"]:
        progress = store.load_progress(c["index"])
        if progress.status != STATUS_DONE:
            continue
        chapters_done += 1
        ch = store.load_chapter(c["index"])
        review_issues.extend(progress.review_issue_dicts())
        bt_issues.extend(progress.backtranslation_issue_dicts())
        bm_mode = progress.back_matter_mode
        if bm_mode:
            # 旁路章（skip=原文直通 / light=fast 粗翻）列给人工复核；
            # 若误伤正文，可用 --back-matter full 重跑。
            back_matter.append(
                {"chapter": c["index"], "title": c.get("title", ""), "mode": bm_mode}
            )
        for s in ch.text_segments:
            if not (s.target and s.target.strip()):
                empty_targets.append(
                    {"chapter": c["index"], "index": s.index, "source": s.source[:60]}
                )

    # 失败节点（尽力而为失败允许继续并回填）：报告必须可见，不能呈现“一切正常”。
    failed_nodes: list[dict] = []
    state = store.load_state()
    for key, node in sorted(state.nodes.items()):
        if node.status not in (NODE_FAILED_RETRYABLE, NODE_FAILED_PERMANENT):
            continue
        failed_nodes.append(
            {
                "node": key,
                "status": node.status,
                "kind": node.failure.kind if node.failure else None,
                "message": (node.failure.message if node.failure else "")[:200],
                "attempts": node.attempts,
            }
        )

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
            "review_issues": len(review_issues),
            "backtranslation_issues": len(bt_issues),
            "empty_targets": len(empty_targets),
            "back_matter_chapters": len(back_matter),
            "failed_nodes": len(failed_nodes),
        },
        "open_conflicts": conflicts,
        "low_confidence_terms": low_conf,
        "review_issues": review_issues,
        "backtranslation_issues": bt_issues,
        "empty_targets": empty_targets,
        "back_matter_chapters": back_matter,
        "failed_nodes": failed_nodes,
    }
