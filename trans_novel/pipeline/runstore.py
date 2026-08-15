"""运行态持久化：支持断点续跑（V2 类型化状态）。

目录结构（state_dir/<book-slug>/）：
  manifest.json         V2 根状态（run_state_schema 标记；迁移时最后原子切换）
  chapters_v2/ch{n}.json  各章（含 source/target 的 Segment；meta 只含 ingest 元数据）
  chapters/             迁移前的 V1 章节文件（切换后仅作恢复备份，不再读取）
  context.json          滚动上下文（梗概 + 前文尾段）
  analysis.json         全局分析产物（book_synopsis 等；完成标志在 V2 状态里）
  glossary.db           术语库
  report.json           QA 报告
  usage.json            本书跨 translate/resume 累计的 LLM token 用量
  events.jsonl          追加式行为 / 改写 / 翻译结果日志

V1 状态在首次打开（持运行锁）时一次性迁移为 V2：先写新路径、校验，最后原子
切换根 manifest。切换后运行时只读 V2，不存在双 schema 分支。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from trans_novel.ingest.models import Chapter, Document
from trans_novel.pipeline.migration import migrate_v1_to_v2
from trans_novel.pipeline.state import (
    NODE_FAILED_PERMANENT,
    NODE_FAILED_RETRYABLE,
    NODE_POLISH,
    NODE_REVIEW,
    NODE_RUNNING,
    NODE_SKIPPED,
    NODE_SUCCEEDED,
    RUN_INPUT_SCHEMA_VERSION,
    RUN_STATE_SCHEMA_VERSION,
    STATUS_DONE,
    STATUS_PENDING,
    ChapterIndex,
    ChapterProgress,
    IdentityMismatchError,
    NodeFailure,
    NodeState,
    RunIdentity,
    RunState,
    normalize_lang_code,
    now_iso,
)

__all__ = [
    "STATUS_DONE",
    "STATUS_PENDING",
    "RunStore",
    "slugify",
    "stable_digest",
]


def stable_digest(payload) -> str:
    """将任意可序列化为 JSON 的载荷规范化为 UTF-8 字节，并计算稳定的 SHA-256 摘要。

    规范化参数固定为 ensure_ascii=False、sort_keys=True、紧凑分隔符与
    default=str，保证同一逻辑载荷在任何进程/版本下得到相同摘要；该摘要可作为
    例行翻译、跳过批次、issue 集和重写候选在事件日志中的紧凑指纹。
    """
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    s = re.sub(r"[^\w一-鿿぀-ヿ-]+", "_", name).strip("_")
    return s or "book"


class RunStore:
    def __init__(self, run_dir: str, *, create: bool = True):
        self.run_dir = run_dir
        self.chapters_dir = os.path.join(run_dir, "chapters")  # V1 遗留（备份）
        self.chapters_v2_dir = os.path.join(run_dir, "chapters_v2")  # V2 活路径
        self._v2_ready = False
        if create:
            self.ensure_dirs()

    def ensure_dirs(self) -> None:
        os.makedirs(self.chapters_v2_dir, exist_ok=True)

    # ── 锁 + 一次性迁移 ────────────────────────────────────────────────────
    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize mutations for one book across independent processes.

        持锁后先做一次性 V1→V2 迁移与中断恢复，再进入临界区。
        """
        self.ensure_dirs()
        lock_path = os.path.join(self.run_dir, ".run.lock")
        with open(lock_path, "a+b") as lock_file:
            if os.name == "nt":  # pragma: no cover - Windows-specific
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    self._migrate_if_needed()
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    self._migrate_if_needed()
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _ensure_migrated(self) -> None:
        """读路径入口：需要时在锁内完成一次性迁移。

        lock() 持锁期间调用不会重入（_v2_ready 已在迁移后置位）。
        """
        if self._v2_ready:
            return
        with self.lock():
            pass

    def _migrate_if_needed(self) -> None:
        """已持锁时调用：V1 迁移 → V2 中断恢复 → 检查点日志恢复。幂等。"""
        if self._v2_ready:
            return
        if not os.path.isfile(self.manifest_path):
            self._v2_ready = True
            return
        data = self._read_json(self.manifest_path)
        if isinstance(data, dict) and data.get("run_state_schema") == RUN_STATE_SCHEMA_VERSION:
            self._v2_ready = True
        else:
            migrate_v1_to_v2(self)
            self._v2_ready = True
        # 进程崩溃遗留的 running 节点：恢复为 pending 并记录中断。
        state = self.load_state()
        if state.recover_interrupted():
            self.save_state(state)
        # 崩溃一致性检查点：补齐/清除在途翻译·润色批次的标记（幂等）。
        from trans_novel.pipeline import checkpoint

        checkpoint.recover(self)

    # ── 路径 ──────────────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> str:
        return os.path.join(self.run_dir, "manifest.json")

    @property
    def context_path(self) -> str:
        return os.path.join(self.run_dir, "context.json")

    @property
    def analysis_path(self) -> str:
        return os.path.join(self.run_dir, "analysis.json")

    @property
    def glossary_path(self) -> str:
        return os.path.join(self.run_dir, "glossary.db")

    @property
    def report_path(self) -> str:
        return os.path.join(self.run_dir, "report.json")

    @property
    def usage_path(self) -> str:
        return os.path.join(self.run_dir, "usage.json")

    @property
    def resource_templates_path(self) -> str:
        return os.path.join(self.run_dir, "resource_templates.json")

    @property
    def event_log_path(self) -> str:
        return os.path.join(self.run_dir, "events.jsonl")

    def chapter_path(self, ci: int) -> str:
        return self.chapter_path_v2(ci)

    def chapter_path_v1(self, ci: int) -> str:
        return os.path.join(self.chapters_dir, f"ch{ci}.json")

    def chapter_path_v2(self, ci: int) -> str:
        return os.path.join(self.chapters_v2_dir, f"ch{ci}.json")

    def path_for(self, name: str) -> str:
        """目录内任意文件名 → 绝对路径（journal 等共享文件的统一定位）。"""
        return os.path.join(self.run_dir, name)

    # ── 崩溃一致性检查点日志 ───────────────────────────────────────────────
    @property
    def journal_path(self) -> str:
        return os.path.join(self.run_dir, "journal.json")

    def save_journal(self, record: dict | None) -> None:
        """原子写入在途批次记录（None 即删除，等价 clear_journal）。"""
        if record is None:
            self.clear_journal()
            return
        self._write_json(self.journal_path, record)

    def load_journal(self) -> dict | None:
        if not os.path.isfile(self.journal_path):
            return None
        data = self._read_json(self.journal_path)
        return data if isinstance(data, dict) else None

    def clear_journal(self) -> None:
        if os.path.isfile(self.journal_path):
            os.remove(self.journal_path)

    # ── 通用 JSON ─────────────────────────────────────────────────────────
    @staticmethod
    def _write_json(path: str, data) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子替换，防写一半中断

    @staticmethod
    def _read_json(path: str):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def exists(self) -> bool:
        return os.path.isfile(self.manifest_path)

    # ── V2 根状态 ─────────────────────────────────────────────────────────
    def load_state(self) -> RunState:
        self._ensure_migrated()
        return RunState.model_validate(self._read_json(self.manifest_path))

    def save_state(self, state: RunState) -> None:
        self._ensure_migrated()
        self._write_json(self.manifest_path, state.model_dump(mode="json"))

    def load_manifest(self) -> dict:
        return self.load_state().model_dump(mode="json")

    def save_manifest(self, manifest: dict) -> None:
        self.save_state(RunState.model_validate(manifest))

    def stage_document(self, doc: Document, identity: RunIdentity) -> dict:
        """写入初始 V2 章节文件并返回根状态 dict，但暂不写入 manifest。

        manifest 标志着本次运行已完成初始化；调用方应在分析结果、
        术语库和上下文全部落盘后再保存。

        EPUB schema 2 文档把标注模板存在 doc.meta["epub_resource_templates"]
        （键为物理资源 href）；这里先把它从 doc.meta 中移除，再以原子方式
        单独写入 resource_templates.json，不写入 manifest。manifest 会频繁
        重写，而模板体积较大；断点续跑时也不必在每次保存时重复写入。
        """
        templates = doc.meta.pop("epub_resource_templates", None)
        if isinstance(templates, dict) and templates:
            self._write_json(self.resource_templates_path, templates)
        state = RunState(
            run_state_schema=RUN_STATE_SCHEMA_VERSION,
            identity=identity,
            title=doc.title,
            fmt=doc.fmt,
            source_path=doc.source_path,
            source_lang=doc.source_lang,
            target_lang=doc.target_lang,
            meta=doc.meta,
            chapters=[
                ChapterIndex(
                    index=c.index,
                    title=c.title,
                    href=c.href,
                    toc_entry_id=c.meta.get("toc_entry_id"),
                )
                for c in doc.chapters
            ],
            progress={c.index: ChapterProgress() for c in doc.chapters},
        )
        for c in doc.chapters:
            self.save_chapter(c)
        return state.model_dump(mode="json")

    def load_resource_templates(self) -> dict[str, str]:
        """读取 EPUB 物理资源模板；文件缺失（非 EPUB 或旧 schema 1 run）返回空字典。"""
        if not os.path.isfile(self.resource_templates_path):
            return {}
        return self._read_json(self.resource_templates_path)

    # ── 章进度（V2 唯一权威来源）──────────────────────────────────────────
    def load_progress(self, ci: int) -> ChapterProgress:
        state = self.load_state()
        return state.progress.get(ci, ChapterProgress())

    def save_progress(self, ci: int, progress: ChapterProgress) -> None:
        state = self.load_state()
        state.progress[ci] = progress
        self.save_state(state)

    def chapter_status(self, ci: int) -> str:
        return self.load_progress(ci).status

    def set_chapter_status(self, ci: int, status: str) -> None:
        state = self.load_state()
        progress = state.progress.setdefault(ci, ChapterProgress())
        progress.status = status  # 仅接受 STATUS_PENDING/STATUS_DONE 常量
        self.save_state(state)

    def pending_chapters(self) -> list[int]:
        state = self.load_state()
        return [ci for ci, pg in state.progress.items() if pg.status != STATUS_DONE]

    def set_review_pending(self, ci: int, pending: bool) -> None:
        """标记/清除某章的异步审校待办（随章进度持久化）。

        异步审校（review 且非 autofix_severe）在章标 done 前打标；结果写回后清标。
        崩溃在两者之间时标记残留在磁盘，续跑据此补跑，异步审校结果不静默丢失。
        """
        state = self.load_state()
        progress = state.progress.setdefault(ci, ChapterProgress())
        progress.review_pending = pending
        self.save_state(state)

    def review_pending_chapters(self) -> list[int]:
        state = self.load_state()
        return [ci for ci, pg in state.progress.items() if pg.review_pending]

    def set_naturalized(self, ci: int, value: bool = True) -> None:
        """记录本章去翻译腔已完成（续跑幂等标记，随进度持久化）。"""
        state = self.load_state()
        progress = state.progress.setdefault(ci, ChapterProgress())
        progress.naturalized = value
        self.save_state(state)

    # ── 节点状态 / 指纹 ────────────────────────────────────────────────────
    def mark_node_running(self, key: str) -> None:
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        node.status = NODE_RUNNING
        node.attempts += 1
        node.failure = None
        node.started_at = now_iso()
        node.finished_at = None
        self.save_state(state)

    def record_node_fingerprint(self, key: str, fingerprint: str) -> None:
        """节点成功完成：记录输入指纹与完成时间。"""
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        node.status = NODE_SUCCEEDED
        node.input_fingerprint = fingerprint
        node.finished_at = now_iso()
        self.save_state(state)

    def record_node_output(self, key: str, output: dict) -> None:
        """把节点成功产物持久化到其 NodeState.output（跨调用可见，如 QA issues）。"""
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        node.output = dict(output)
        self.save_state(state)

    def mark_node_succeeded(self, key: str, fingerprint: str | None = None) -> None:
        """节点成功完成（无指纹契约的节点用；有指纹的走 record_node_fingerprint）。"""
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        node.status = NODE_SUCCEEDED
        if fingerprint:
            node.input_fingerprint = fingerprint
        node.finished_at = now_iso()
        self.save_state(state)

    def mark_node_skipped(self, key: str) -> None:
        """策略性禁用的可选节点：持久化为 skipped（覆盖上一次运行的 succeeded/skipped）。

        保留既有 input_fingerprint —— 它是“上次成功产出时的输入描述”，策略改回
        启用后节点重跑会重新计算；不清空它即可在状态里区分“本轮按策略跳过”与
        从未执行过。
        同时原子清除该节点对应的恢复标记（skip polish → 清 pending_polish；
        skip review → 清 review_pending）：策略显式禁用后，残留标记不得让章永远
        卡在“未完成”或让就绪门禁永久拒绝。
        """
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        node.status = NODE_SKIPPED
        node.finished_at = now_iso()
        base, sep, suffix = key.partition(":")
        if sep and suffix.isdigit():
            ci = int(suffix)
            progress = state.progress.setdefault(ci, ChapterProgress())
            if base == NODE_POLISH:
                progress.pending_polish = []
            elif base == NODE_REVIEW:
                progress.review_pending = False
        self.save_state(state)

    def fail_node(self, key: str, kind: str, message: str = "") -> None:
        """节点失败落盘：provider_permanent → failed_permanent，其余 → failed_retryable。

        尽力而为节点失败（可重试）；必须节点失败由 runner 冒泡前调用，保证
        失败状态对报告/就绪门禁可见。
        """
        state = self.load_state()
        node = state.nodes.get(key)
        if node is None:
            node = NodeState(node_id=key)
            state.nodes[key] = node
        if kind == "provider_permanent":
            node.status = NODE_FAILED_PERMANENT
        else:
            node.status = NODE_FAILED_RETRYABLE
        node.failure = NodeFailure(kind=kind, message=message, at=now_iso())
        self.save_state(state)

    def reconcile_fingerprints(self, computed: dict[str, str]) -> set[str]:
        """按最新计算的输入指纹失效失配节点及其后代（见 RunState.reconcile_fingerprints）。

        - 只对“已记录过指纹”的节点做对账：空指纹（V1 迁移合成的 legacy 成功态）
          不参与失效，避免迁移后的存量书被整体重跑；
        - 失效 translate 节点时清除该章译文并重开（pending），
          失效 titles 时清除已译标题 —— 这两类产物在章节文件/manifest，
          由本层在状态级失效之外一并清理（同一 state 对象上完成，避免互相覆盖）；
        - book_synopsis 产物在 analysis.json，一并清除。返回失效节点键集合。
        """
        state = self.load_state()
        invalidated = state.reconcile_fingerprints(computed)
        for key in list(invalidated):
            base, sep, suffix = key.partition(":")
            ci = int(suffix) if sep and suffix.isdigit() else None
            if base == "translate" and ci is not None:
                self._clear_translation_targets(ci, state)
            elif base == "titles":
                self._clear_translated_titles(state)
        if "book_synopsis" in invalidated:
            analysis = self.load_analysis() or {}
            if "book_synopsis" in analysis:
                analysis.pop("book_synopsis")
                self.save_analysis(analysis)
        if "analyze" in invalidated and os.path.isfile(self.analysis_path):
            # 风格分析输入（模型路由等）变化：清空 analysis.json 让 AnalyzeNode
            # 真正重新分析（幂等分支以“文件不存在”为准，不会跳过）。
            os.remove(self.analysis_path)
        self.save_state(state)
        return invalidated

    def _clear_translation_targets(self, ci: int, state) -> None:
        """translate 输入失配：清除该章译文与全部下游标记，章重开为 pending。

        只动目标章（“只失效该节点及其后代”）；其它章译文原样保留。
        在调用方持有的 state 对象上改进度字段（与节点失效同一次保存）。
        """
        chapter = self.load_chapter(ci)
        for seg in chapter.text_segments:
            seg.target = None
        self.save_chapter(chapter)
        progress = state.progress.setdefault(ci, ChapterProgress())
        progress.status = STATUS_PENDING
        progress.pending_polish = []
        progress.naturalized = False
        progress.review_issues = []
        progress.backtranslation_issues = []
        state.progress[ci] = progress
        self.log_event(
            "translate_invalidated",
            chapter=ci,
            reason="input_fingerprint_mismatch",
        )

    def _clear_translated_titles(self, state) -> None:
        """titles 输入失配：清除已译标题（manifest 章条目与 TOC entry）。"""
        for c in state.chapters:
            c.title_translated = None
        meta = state.meta
        raw_toc = meta.get("toc_entries") if isinstance(meta, dict) else None
        if isinstance(raw_toc, list):
            for e in raw_toc:
                if isinstance(e, dict):
                    e.pop("title_translated", None)
        self.log_event("titles_invalidated", reason="input_fingerprint_mismatch")

    def reopen_back_matter_chapter(self, ci: int, *, prev_mode: str, mode: str, title: str) -> None:
        """附属章档位升档（skip→light/full、light→full）：重开已完成章重译。

        清除原文副本/快速粗翻译文与全部下游标记，章状态回 pending；降档不回退。
        由 planner 在目标筛选前对全部已完成附属章执行（only_chapter 场景也不漏）。
        """
        chapter = self.load_chapter(ci)
        for seg in chapter.segments:
            seg.target = None
        state = self.load_state()
        progress = state.progress.setdefault(ci, ChapterProgress())
        progress.back_matter_mode = None
        progress.pending_polish = []
        progress.set_review_issue_dicts([])
        progress.set_backtranslation_issue_dicts([])
        progress.status = STATUS_PENDING
        state.progress[ci] = progress
        self.save_state(state)
        self.save_chapter(chapter)
        self.log_event(
            "back_matter_reopened",
            chapter=ci,
            title=title,
            prev_mode=prev_mode,
            mode=mode,
        )

    # ── 运行身份 ───────────────────────────────────────────────────────────
    def verify_identity(
        self,
        *,
        source_bytes_sha256: str,
        source_lang: str = "",
        target_lang: str = "",
    ) -> None:
        """核验当前输入与存储的运行身份一致；不一致抛 IdentityMismatchError。

        源语言/目标语言传空（auto 未解析）时跳过对应维度；语言比较用归一化值。
        """
        state = self.load_state()
        ident = state.identity
        problems: list[str] = []
        if ident.source_bytes_sha256:
            if not source_bytes_sha256:
                problems.append("无法读取当前源文件，不能核验运行身份")
            elif source_bytes_sha256 != ident.source_bytes_sha256:
                problems.append("源文件内容与运行状态不一致（文件被修改或不是同一本书）")
        else:
            problems.append(
                "运行状态缺少源文件校验信息（迁移时源文件不可读），无法确认与当前输入一致"
            )
        if ident.run_input_schema_version != RUN_INPUT_SCHEMA_VERSION:
            problems.append(
                f"运行输入 schema 版本不一致"
                f"（状态 {ident.run_input_schema_version}，当前 {RUN_INPUT_SCHEMA_VERSION}）"
            )
        if (
            source_lang
            and ident.source_lang
            and normalize_lang_code(source_lang) != ident.source_lang
        ):
            problems.append(f"源语言不一致（状态 {ident.source_lang}，当前 {source_lang}）")
        if (
            target_lang
            and ident.target_lang
            and normalize_lang_code(target_lang) != ident.target_lang
        ):
            problems.append(f"目标语言不一致（状态 {ident.target_lang}，当前 {target_lang}）")
        if problems:
            raise IdentityMismatchError("；".join(problems))

    # ── 章 ────────────────────────────────────────────────────────────────
    def save_chapter(self, chapter: Chapter) -> None:
        self._ensure_migrated()
        self._write_json(self.chapter_path(chapter.index), chapter.to_dict())

    def load_chapter(self, ci: int) -> Chapter:
        self._ensure_migrated()
        return Chapter.from_dict(self._read_json(self.chapter_path(ci)))

    # ── 上下文 / 分析 / 报告 ──────────────────────────────────────────────
    def save_context(self, data: dict) -> None:
        self._write_json(self.context_path, data)

    def load_context(self) -> dict | None:
        return self._read_json(self.context_path) if os.path.isfile(self.context_path) else None

    def save_analysis(self, data: dict) -> None:
        self._ensure_migrated()
        self._write_json(self.analysis_path, data)

    def load_analysis(self) -> dict | None:
        self._ensure_migrated()
        return self._read_json(self.analysis_path) if os.path.isfile(self.analysis_path) else None

    def save_report(self, data: dict) -> None:
        self._write_json(self.report_path, data)

    def save_usage(self, data: dict) -> None:
        self._write_json(self.usage_path, data)

    def load_usage(self) -> dict | None:
        return self._read_json(self.usage_path) if os.path.isfile(self.usage_path) else None

    # ── 追加式事件日志 ────────────────────────────────────────────────────
    def log_event(self, event: str, **data: Any) -> None:
        """追加一条 JSONL 事件，用于翻译行为、改写前后和产物对账。

        新行一律带 event_schema: 2；无该字段的历史行视为 V1，不回写。
        事件日志采用尽力写入策略，仅用于审计，不是恢复状态的依据。若目录创建
        或事件追加抛出 OSError，则发出 RuntimeWarning 后返回；这不会影响已
        持久化的状态，也不会触发重跑。序列化错误和编程错误仍照常抛出。
        """
        row = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "event": event,
            "event_schema": 2,
            **data,
        }
        try:
            self.ensure_dirs()
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError as exc:
            warnings.warn(
                f"event log append failed for {event!r}: {exc}", RuntimeWarning, stacklevel=2
            )
