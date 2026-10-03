"""在模型加载和清理之前绑定磁盘原始译文，避免旧槽位掩盖改字。"""

from __future__ import annotations

from trans_novel.epub.richtext import RichSource, rich_source_digest
from trans_novel.epub.slots import EpubTextSlot, normalized_target_text
from trans_novel.pipeline.state.models import stable_digest


def raw_target_digest(store, chapter_indices) -> str:
    """逐章直接读 JSON；两种目标布局都必须与完整 target 字符串一致。"""
    rows = []
    has_pending = False
    for index in sorted(chapter_indices):
        raw = store.read_json(store.chapter_path_v2(index))
        if not isinstance(raw, dict) or raw.get("index") != index:
            raise ValueError("rich migration raw chapter identity mismatch")
        segments = raw.get("segments")
        if not isinstance(segments, list):
            raise ValueError("rich migration requires complete raw segments")
        for segment in segments:
            if not isinstance(segment, dict) or type(segment.get("index")) is not int:
                raise ValueError("rich migration raw segment identity mismatch")
            target = segment.get("target")
            if not isinstance(target, str):
                raise ValueError("rich migration requires complete saved targets")
            rich = segment.get("rich_target")
            if rich is not None:
                if not isinstance(rich, dict) or not isinstance(rich.get("runs"), list):
                    raise ValueError("rich migration requires complete raw rich target")
                texts = []
                for run in rich["runs"]:
                    if not isinstance(run, dict) or not isinstance(run.get("text", ""), str):
                        raise ValueError("rich migration raw target text must be a string")
                    texts.append(run.get("text", ""))
                rendered = "".join(texts)
            else:
                source = segment.get("epub_state")
                if not isinstance(source, dict) or not isinstance(source.get("slots"), list):
                    raise ValueError("rich migration requires complete raw source slots")
                slots = [EpubTextSlot.model_validate(slot) for slot in source["slots"]]
                if all(slot.target_value is not None for slot in slots):
                    rendered = normalized_target_text(slots)
                else:
                    metadata = segment.get("meta")
                    pending = (
                        metadata.get("richtext_pending") if isinstance(metadata, dict) else None
                    )
                    if not isinstance(pending, dict):
                        raise ValueError("rich migration requires a proven pending rich target")
                    inventory = RichSource.model_validate(source.get("rich_source"))
                    expected = {
                        "version": 2,
                        "source_sha256": rich_source_digest(inventory),
                        "target_sha256": stable_digest(target),
                    }
                    if inventory.version != 2 or pending != expected:
                        raise ValueError("rich migration requires a proven pending rich target")
                    has_pending = True
                    rendered = target
            if rendered != target:
                raise ValueError("rich migration raw target disagrees with its saved layout")
            rows.append([index, segment["index"], target])
    digest = stable_digest(rows)
    if has_pending:
        manifest = store.read_json(store.manifest_path)
        frozen = (manifest.get("meta") or {}).get("richtext_migration")
        if not isinstance(frozen, dict) or frozen.get("target_sha256") != digest:
            raise ValueError("rich migration pending targets require a frozen full digest")
    return digest


__all__ = ["raw_target_digest"]
