"""显式接受富文本迁移后的内容依赖，保留旧付费指纹的审计记录。"""

from __future__ import annotations

from trans_novel.pipeline.planning import content_fingerprints
from trans_novel.pipeline.state import NODE_SUCCEEDED
from trans_novel.pipeline.state.models import TRANSLATION_POLICY_VERSION


def rebase_migrated_content(config, store, context) -> None:
    """调用方持锁；仅重基准已成功节点，不改变内容或生命周期状态。"""
    state = store.load_state()
    migration = state.meta.get("richtext_migration")
    if (
        state.identity.translation_policy_version != TRANSLATION_POLICY_VERSION
        or not isinstance(migration, dict)
        or migration.get("complete") is not True
    ):
        raise ValueError("content fingerprint rebase requires completed rich migration")
    values = content_fingerprints(config, store, context)
    audit = migration.setdefault("fingerprint_rebase", {})
    changed: dict[str, dict[str, str]] = {}
    for key, value in values.items():
        node = state.nodes.get(key)
        if node is None or node.status != NODE_SUCCEEDED or node.input_fingerprint == value:
            continue
        previous = node.input_fingerprint
        change = {"old": previous, "new": value}
        record = audit.setdefault(key, {"old": previous, "new": value, "audit": []})
        record["new"] = value
        record["audit"].append(change)
        node.input_fingerprint = value
        changed[key] = change
    if changed:
        store.save_state(state)
        store.log_event("richtext_content_fingerprints_rebased", changes=changed)


__all__ = ["rebase_migrated_content"]
