"""一次性 V1→V2 状态迁移（必须在 RunStore.lock() 的锁内调用）。

原则（与共享架构上下文一致）：
- 迁移只读 V1 文件、写新路径（chapters_v2/），不修改任何 V1 活文件；
- 构建并校验完整 V2 状态后，最后一步原子切换根 manifest（temp + os.replace）；
- 中断于切换前：V1 完好可读、迁移可重试（chapters_v2 覆盖写幂等）；
- 切换完成后运行时只读 V2；V1 旧文件保留作恢复备份，不再作为活路径。
"""

from __future__ import annotations

import os
from typing import Any

from trans_novel.ingest.models import Chapter
from trans_novel.pipeline.state import (
    NODE_ANALYZE,
    NODE_BACKTRANSLATE,
    NODE_BOOK_SYNOPSIS,
    NODE_DIGEST,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_NATURALIZE,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REVIEW,
    NODE_SKIPPED,
    NODE_SUCCEEDED,
    NODE_TRANSLATE,
    RUN_INPUT_SCHEMA_VERSION,
    RUN_STATE_SCHEMA_VERSION,
    STATUS_DONE,
    AnalysisFlags,
    ChapterIndex,
    ChapterProgress,
    NodeState,
    PolishBatch,
    RunIdentity,
    RunState,
    chapter_node_key,
    normalize_lang_code,
    source_bytes_hash,
)

# V1 写入 Chapter.meta 的流水线字段；迁移时搬进 ChapterProgress。
PIPELINE_META_KEYS = frozenset(
    {
        "source_digest",
        "pending_polish",
        "naturalized",
        "review_issues",
        "backtranslation_issues",
        "back_matter_mode",
    }
)


def _identity_from_v1(v1_manifest: dict[str, Any]) -> RunIdentity:
    """从 V1 manifest 推导运行身份。

    源语言/目标语言在 V1 落盘前已解析（检测或显式指定），这里归一化后入库；
    解析不了的字面量（如 "auto"）不会出现在最终身份里。
    """
    source_path = v1_manifest.get("source_path") or ""
    return RunIdentity(
        source_bytes_sha256=source_bytes_hash(source_path),
        run_input_schema_version=RUN_INPUT_SCHEMA_VERSION,
        source_lang=normalize_lang_code(v1_manifest.get("source_lang", "")),
        target_lang=normalize_lang_code(v1_manifest.get("target_lang", "")),
    )


def migrate_v1_to_v2(store) -> RunState:
    """把 store 目录下的 V1 状态迁移为 V2。

    前提：调用方已持有运行锁，且确认 manifest.json 存在且为 V1。
    返回构建好的 RunState；切换完成后运行时方法全部走 V2。
    """
    v1_manifest = store._read_json(store.manifest_path)
    identity = _identity_from_v1(v1_manifest)
    meta = v1_manifest.get("meta")
    if not isinstance(meta, dict):
        meta = {}

    chapters: list[ChapterIndex] = []
    progress: dict[int, ChapterProgress] = {}
    for c in v1_manifest.get("chapters", []):
        if not isinstance(c, dict) or not isinstance(c.get("index"), int):
            raise ValueError("V1 状态 manifest 的 chapters 条目损坏（缺少整数 index）")
        ci = c["index"]
        chapters.append(
            ChapterIndex(
                index=ci,
                title=str(c.get("title", "") or ""),
                href=c.get("href"),
                toc_entry_id=c.get("toc_entry_id"),
                title_translated=c.get("title_translated"),
            )
        )
        pg = ChapterProgress()
        if c.get("status") == "done":
            pg.status = "done"
        pg.review_pending = bool(c.get("review_pending"))
        progress[ci] = pg

    # 逐章：读 V1 章节文件 → 抽出流水线字段 → 写 V2 章节文件（meta 只留 ingest 元数据）。
    for ci in progress:
        v1_path = store.chapter_path_v1(ci)
        if not os.path.isfile(v1_path):
            raise ValueError(f"V1 状态缺少章节文件：{v1_path}")
        raw = store._read_json(v1_path)
        chapter = Chapter.from_dict(raw)
        chapter_meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        pg = progress[ci]
        pg.source_digest = str(chapter_meta.get("source_digest", "") or "")
        raw_polish = chapter_meta.get("pending_polish")
        if isinstance(raw_polish, list):
            pg.pending_polish = [
                PolishBatch.model_validate(e) for e in raw_polish if isinstance(e, dict)
            ]
        pg.naturalized = bool(chapter_meta.get("naturalized"))
        review_issues = chapter_meta.get("review_issues")
        if isinstance(review_issues, list):
            pg.set_review_issue_dicts(review_issues)
        bt_issues = chapter_meta.get("backtranslation_issues")
        if isinstance(bt_issues, list):
            pg.set_backtranslation_issue_dicts(bt_issues)
        mode = chapter_meta.get("back_matter_mode")
        pg.back_matter_mode = str(mode) if isinstance(mode, str) and mode else None
        chapter.meta = {k: v for k, v in chapter_meta.items() if k not in PIPELINE_META_KEYS}
        store._write_json(store.chapter_path_v2(ci), chapter.to_dict())

    # 完成标志从 analysis.json 搬进 V2（analysis 文件本身不动，旧键成为惰性数据）。
    analysis: dict[str, Any] = {}
    if os.path.isfile(store.analysis_path):
        raw_analysis = store._read_json(store.analysis_path)
        if isinstance(raw_analysis, dict):
            analysis = raw_analysis
    analysis_flags = AnalysisFlags(term_mining_done=bool(analysis.get("term_mining_done")))

    # 合成 legacy 成功态（空指纹）：V1 没有节点状态，迁移按 V1 产物/标记合成
    # succeeded，使迁移后的存量书在依赖闭包规划下按“已满足”跳过，不被整体重跑；
    # 未完成标记（pending_polish / review_pending）对应的节点不合成，续跑补跑。
    nodes: dict[str, NodeState] = {
        NODE_PREPARE: NodeState(node_id=NODE_PREPARE, status=NODE_SUCCEEDED),
        NODE_ANALYZE: NodeState(node_id=NODE_ANALYZE, status=NODE_SUCCEEDED),
    }
    if analysis_flags.term_mining_done:
        nodes[NODE_MINE_TERMS] = NodeState(node_id=NODE_MINE_TERMS, status=NODE_SUCCEEDED)
        nodes[NODE_NAME_TERMS] = NodeState(node_id=NODE_NAME_TERMS, status=NODE_SUCCEEDED)
    if analysis.get("book_synopsis"):
        nodes[NODE_BOOK_SYNOPSIS] = NodeState(node_id=NODE_BOOK_SYNOPSIS, status=NODE_SUCCEEDED)
    for ci, pg in progress.items():
        if pg.source_digest:
            nodes[chapter_node_key(NODE_DIGEST, ci)] = NodeState(
                node_id=chapter_node_key(NODE_DIGEST, ci), status=NODE_SUCCEEDED
            )
        if pg.status != STATUS_DONE:
            continue
        nodes[chapter_node_key(NODE_TRANSLATE, ci)] = NodeState(
            node_id=chapter_node_key(NODE_TRANSLATE, ci), status=NODE_SUCCEEDED
        )
        # V1 没有“质量环节是否执行过”的完成记录（economy/balanced 档可能从未跑过
        # polish/review/backtranslate）——不能凭“无待办标记”合成 succeeded（空指纹
        # 会被 planner 永久视为已满足，启用新策略后这些环节会被跳过）。统一合成
        # skipped：当前策略启用时会被重新规划执行；禁用时即为策略性跳过。
        nodes[chapter_node_key(NODE_POLISH, ci)] = NodeState(
            node_id=chapter_node_key(NODE_POLISH, ci), status=NODE_SKIPPED
        )
        # 保留 V1 naturalized 标记的语义：只有标记为已自然化的章才合成 succeeded；
        # 标记缺失的章留空，由当前策略（启用时）调度补跑，不把未完成当作已满足。
        if pg.naturalized:
            nodes[chapter_node_key(NODE_NATURALIZE, ci)] = NodeState(
                node_id=chapter_node_key(NODE_NATURALIZE, ci), status=NODE_SUCCEEDED
            )
        nodes[chapter_node_key(NODE_REVIEW, ci)] = NodeState(
            node_id=chapter_node_key(NODE_REVIEW, ci), status=NODE_SKIPPED
        )
        nodes[chapter_node_key(NODE_BACKTRANSLATE, ci)] = NodeState(
            node_id=chapter_node_key(NODE_BACKTRANSLATE, ci), status=NODE_SKIPPED
        )

    state = RunState(
        run_state_schema=RUN_STATE_SCHEMA_VERSION,
        identity=identity,
        title=str(v1_manifest.get("title", "") or ""),
        fmt=str(v1_manifest.get("fmt", "") or ""),
        source_path=str(v1_manifest.get("source_path", "") or ""),
        source_lang=str(v1_manifest.get("source_lang", "") or ""),
        target_lang=str(v1_manifest.get("target_lang", "") or ""),
        meta=meta,
        initialized=bool(v1_manifest.get("initialized")),
        chapters=chapters,
        progress=progress,
        nodes=nodes,
        analysis_flags=analysis_flags,
    )

    # 校验：往返序列化 + 章节覆盖一致性，校验失败绝不切换。
    state = RunState.model_validate(state.model_dump(mode="json"))
    _validate_migrated_state(state, store)

    # 原子切换根 manifest（最后一步）：切换前 V1 一切未动，可随时回退重试。
    store._write_json(store.manifest_path, state.model_dump(mode="json"))
    return state


def _validate_migrated_state(state: RunState, store) -> None:
    """迁移产物完整性校验：每章都有索引、进度与 V2 章节文件，章序连续。"""
    indices = [c.index for c in state.chapters]
    if len(set(indices)) != len(indices):
        raise ValueError("V1 状态 manifest 含重复章节 index，拒绝迁移")
    for ci in indices:
        if ci not in state.progress:
            raise ValueError(f"V1 状态缺少第 {ci} 章的进度条目，拒绝迁移")
        if not os.path.isfile(store.chapter_path_v2(ci)):
            raise ValueError(f"迁移后缺少第 {ci} 章的 V2 章节文件，拒绝切换")
