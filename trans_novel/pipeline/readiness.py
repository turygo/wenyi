"""正式回填的单一权威就绪门禁：运行身份核验 + 完整度检查。

run_all 与 tools assemble 的每条正式产出路径都必须先过这里。不完整状态直接
拒绝，绝不静默回退成“部分原文 + 部分译文”的混合产物；也不存在显式的
“不完整预览”CLI 契约，因此本轮不提供预览旁路。

除章节/目标/标记检查外，本门禁还要求“适用的必需上游节点全部 succeeded”：
- required 节点缺失/pending/running/failed 一律拒绝（不是只看 failed）；
- 可选节点（polish/naturalize/review/consistency_qa/book_synopsis）允许
  策略性 skipped；尽力而为节点（mine_terms/name_terms）失败/缺失不阻塞；
- 正在执行的 assemble 节点自身不计入。
"""

from __future__ import annotations

from trans_novel.pipeline.runstore import STATUS_DONE, RunStore
from trans_novel.pipeline.state import (
    BEST_EFFORT_NODES,
    NODE_ANALYZE,
    NODE_BACKTRANSLATE,
    NODE_DIGEST,
    NODE_FAILED_PERMANENT,
    NODE_FAILED_RETRYABLE,
    NODE_PREPARE,
    NODE_REPORT,
    NODE_SKIPPED,
    NODE_SUCCEEDED,
    NODE_TITLES,
    NODE_TRANSLATE,
    chapter_node_key,
    source_bytes_hash,
)


class ReadinessError(RuntimeError):
    """正式回填被拒：运行身份失配或状态不完整。"""


def _chapter_node_base(key: str) -> str:
    return key.split(":", 1)[0]


def assemble_readiness_problems(store: RunStore) -> list[str]:
    """返回阻止正式回填的全部问题（空列表 = 可以回填）。

    覆盖：未完成章节、必需 target 为空、待润色批次、异步审校待办、
    适用的必需上游节点未成功（含缺失/pending/running/failed），以及失败态
    必需节点。尽力而为节点（如 mine_terms 定名失败）允许继续，不阻塞产出；
    可选节点（polish/naturalize/review/consistency/book_synopsis）的策略性
    skipped 视为已决策，不阻塞。
    """
    state = store.load_state()
    problems: list[str] = []

    for idx in state.chapters:
        pg = state.progress.get(idx.index)
        if pg is None or pg.status != STATUS_DONE:
            problems.append(f"第{idx.index}章未完成翻译")
            continue
        if pg.pending_polish:
            problems.append(f"第{idx.index}章润色待完成")
        if pg.review_pending:
            problems.append(f"第{idx.index}章异步审校待完成")

    # 必需 target 非空：done 章的正文段都必须有译文（旁路章 target=source 同样非空）。
    for idx in state.chapters:
        pg = state.progress.get(idx.index)
        if pg is None or pg.status != STATUS_DONE:
            continue
        chapter = store.load_chapter(idx.index)
        for seg in chapter.text_segments:
            if not (seg.target and seg.target.strip()):
                problems.append(f"第{idx.index}章存在未翻译段落")
                break

    # 适用的必需上游节点完成门（不只看失败态）。核心必需链（prepare/analyze/
    # translate/digest）只接受 succeeded——skipped 说明策略/状态异常，必须拒绝；
    # backtranslate 允许 skipped（V1 迁移合成/legacy 允许，不阻断底层 writer 单测）。
    # titles/report 仅在正式链路已参与（状态存在）时要求 succeeded：direct writer
    # 或未请求的动作根不适用，不重新阻断底层 writer 单测；失败仍经失败节点检查拒绝。
    def check_book(node_id: str, *, allow_skipped: bool = False) -> None:
        node = state.nodes.get(node_id)
        if node is None:
            problems.append(f"节点 {node_id} 未执行（必需上游未完成）")
        elif node.status == NODE_SUCCEEDED or (allow_skipped and node.status == NODE_SKIPPED):
            return
        else:
            problems.append(f"节点 {node_id} 状态为 {node.status}（必需上游未完成）")

    check_book(NODE_PREPARE)
    check_book(NODE_ANALYZE)

    for node_id in (NODE_TITLES, NODE_REPORT):
        node = state.nodes.get(node_id)
        if node is None:
            problems.append(f"节点 {node_id} 未执行（必需上游未完成）")
        elif node.status != NODE_SUCCEEDED:
            problems.append(f"节点 {node_id} 状态为 {node.status}（必需上游未完成）")

    def check_chapter_required(node_id: str, ci: int, *, allow_skipped: bool = False) -> None:
        key = chapter_node_key(node_id, ci)
        node = state.nodes.get(key)
        if node is None:
            problems.append(f"节点 {key} 未执行（必需上游未完成）")
        elif node.status == NODE_SUCCEEDED or (allow_skipped and node.status == NODE_SKIPPED):
            return
        else:
            problems.append(f"节点 {key} 状态为 {node.status}（必需上游未完成）")

    for idx in state.chapters:
        pg = state.progress.get(idx.index)
        if pg is None or pg.status != STATUS_DONE:
            continue
        ci = idx.index
        bypass = pg.back_matter_mode in ("skip", "light")
        check_chapter_required(NODE_TRANSLATE, ci)
        if pg.source_digest:
            check_chapter_required(NODE_DIGEST, ci)
        if not bypass:
            check_chapter_required(NODE_BACKTRANSLATE, ci, allow_skipped=True)

    for key, node in sorted(state.nodes.items()):
        if node.status in (NODE_FAILED_RETRYABLE, NODE_FAILED_PERMANENT):
            if _chapter_node_base(key) in BEST_EFFORT_NODES:
                continue
            problems.append(f"节点 {node.node_id} 处于失败状态")
    return problems


def ensure_assemble_ready(store: RunStore, source_path: str) -> None:
    """正式回填前的统一门禁：先核验运行身份，再检查完整度。

    源文件失配抛 IdentityMismatchError（翻译/回填复用前都必须拒绝）；其余
    完整度问题抛 ReadinessError。
    """
    state = store.load_state()
    # 回填以状态里的语言为权威（产物是状态的一部分），此处只核验源文件字节。
    store.verify_identity(
        source_bytes_sha256=source_bytes_hash(source_path),
        source_lang=state.identity.source_lang,
        target_lang=state.identity.target_lang,
    )
    problems = assemble_readiness_problems(store)
    if problems:
        raise ReadinessError("；".join(problems))
