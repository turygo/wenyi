"""计算注释槽位迁移前后的规范指纹。"""

from __future__ import annotations

from trans_novel.ingest import Chapter
from trans_novel.pipeline.contracts import GOAL_RUN_ALL
from trans_novel.pipeline.planning.planner import WorkflowPolicy
from trans_novel.pipeline.planning.prescan import build_prescan_inputs
from trans_novel.pipeline.state import NODE_SUCCEEDED


class _ChapterView:
    """只读章节覆盖层；其余读取仍由持锁的 RunStore 完成。"""

    def __init__(self, store, chapters: list[Chapter]):
        self._store = store
        self._chapters = {chapter.index: chapter for chapter in chapters}

    def exists(self) -> bool:
        return self._store.exists()

    def load_state(self):
        state = self._store.load_state()
        return state.model_copy(
            update={
                "chapters": [
                    item.model_copy(update={"processing": self._chapters[item.index].processing})
                    for item in state.chapters
                ]
            }
        )

    def load_chapter(self, index: int) -> Chapter:
        return self._chapters[index]

    def load_progress(self, index: int):
        return self._store.load_progress(index)


_FIELDS = {
    "prepare": "prepare_fingerprint",
    "analyze": "analyze_fingerprint",
    "layout": "layout_fingerprint",
    "mine_terms": "mine_fingerprint",
    "name_terms": "name_terms_fingerprint",
    "translate": "translate_fingerprint",
    "polish": "polish_fingerprint",
    "titles": "titles_fingerprint",
    "deterministic_qa": "deterministic_qa_fingerprint",
    "report": "report_fingerprint",
    "assemble": "assemble_fingerprint",
}


def _value(inputs, key: str) -> str | None:
    base, separator, suffix = key.partition(":")
    callback = getattr(inputs, _FIELDS.get(base, ""), None)
    if callback is None:
        return None
    try:
        return callback(int(suffix)) if separator and suffix.isdigit() else callback()
    except TypeError:
        return None


def _canonical_note_fingerprints(
    config, store, context, chapters: list[Chapter], *, include_layout: bool = True
) -> dict[str, str]:
    policy = WorkflowPolicy.from_config(config)
    inputs = build_prescan_inputs(
        config, _ChapterView(store, chapters), policy, context, GOAL_RUN_ALL
    )
    state = store.load_state()
    values: dict[str, str] = {}
    excluded = {"report", "assemble"} if include_layout else {"report", "assemble", "layout"}
    for key, node in state.nodes.items():
        if node.status != NODE_SUCCEEDED or key.partition(":")[0] in excluded:
            continue
        value = _value(inputs, key)
        if value is not None:
            values[key] = value
    return values


def content_fingerprints(config, store, context) -> dict[str, str]:
    """返回已成功内容节点在当前配置和源证据下的规范指纹。"""
    chapters = [store.load_chapter(chapter.index) for chapter in store.load_state().chapters]
    return _canonical_note_fingerprints(config, store, context, chapters, include_layout=False)


def note_fingerprint_updates(
    config, store, context, before: list[Chapter], after: list[Chapter]
) -> dict[str, tuple[str, str]]:
    """仅返回已保存指纹匹配迁移前规范值的变更。"""
    old_values = _canonical_note_fingerprints(config, store, context, before)
    new_values = _canonical_note_fingerprints(config, store, context, after)
    state = store.load_state()
    updates: dict[str, tuple[str, str]] = {}
    for key, old in old_values.items():
        node = state.nodes.get(key)
        new = new_values.get(key)
        if (
            node is not None
            and node.status == NODE_SUCCEEDED
            and node.input_fingerprint == old
            and new is not None
            and old != new
        ):
            updates[key] = (old, new)
    return updates


__all__ = ["content_fingerprints", "note_fingerprint_updates"]
