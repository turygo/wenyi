"""V2 运行状态：类型化 RunState / ChapterProgress / NodeState 与运行身份。

V1 把流水线进度散落在 manifest（chapter status / review_pending）、Chapter.meta
（pending_polish / naturalized / source_digest / review_issues /
backtranslation_issues / back_matter_mode）与 analysis.json（term_mining_done）。
V2 把这些收敛为唯一权威的类型化进度：ChapterProgress 每章一份；NodeState 记录
节点级指纹/尝试/失败/时间戳。Chapter.meta 只保留 ingest/EPUB/FB2 元数据；
analysis.json 只保留产物（风格分析、book_synopsis），完成标志归 V2 的
AnalysisFlags 所有。

磁盘布局（run_dir/）：
  manifest.json          V2 根状态（run_state_schema 标记；迁移时最后原子切换）
  chapters_v2/ch{n}.json V2 章节文件（meta 不含流水线字段）
  chapters/              迁移前的 V1 章节文件（切换后仅作恢复备份，不再读取）
  analysis.json / context.json / glossary.db / report.json / usage.json /
  events.jsonl            共享文件，路径不变

``run_state_schema`` 是运行状态文件的 schema 版本，与 EPUB 文档格式 schema
（meta.epub_schema）无关：EPUB schema 描述书稿的物理组织方式，本常量描述
trans-novel 自己持久化运行进度所用的数据布局。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── schema 版本 ──────────────────────────────────────────────────────────────
# 运行状态文件的布局版本（V2）。迁移后 manifest.json 以此为标记。
RUN_STATE_SCHEMA_VERSION = 2
# 解析器/运行输入契约版本：参与运行身份，源文件解析规则升级时递增。
# 旧版本状态与当前解析器不兼容 → 拒绝复用，而非静默续跑。
RUN_INPUT_SCHEMA_VERSION = 1

# ── 章状态 ──────────────────────────────────────────────────────────────────
STATUS_PENDING = "pending"
STATUS_DONE = "done"
ChapterStatus = Literal["pending", "done"]

# ── 节点状态 ────────────────────────────────────────────────────────────────
NODE_PENDING = "pending"
NODE_RUNNING = "running"
NODE_SUCCEEDED = "succeeded"
NODE_SKIPPED = "skipped"
NODE_FAILED_RETRYABLE = "failed_retryable"
NODE_FAILED_PERMANENT = "failed_permanent"
NodeStatus = Literal[
    "pending", "running", "succeeded", "skipped", "failed_retryable", "failed_permanent"
]

# 内置节点 id（workflow 节点沿用同一套 id）。本章节范围的节点用
# chapter_node_key() 生成 "digest:3" 形式的复合键。
# 依赖图（见 definition.py）：prepare → analyze, digest*, mine_terms；
# analyze + digest* + mine_terms → name_terms；digest* + name_terms → book_synopsis；
# book_synopsis → 逐章 translate → polish → naturalize → review / backtranslate；
# 全部章末节点 → titles / consistency_qa → report → assemble。
NODE_PREPARE = "prepare"
NODE_ANALYZE = "analyze"
NODE_DIGEST = "digest"
NODE_MINE_TERMS = "mine_terms"
NODE_NAME_TERMS = "name_terms"
NODE_BOOK_SYNOPSIS = "book_synopsis"
NODE_TRANSLATE = "translate"
NODE_POLISH = "polish"
NODE_NATURALIZE = "naturalize"
NODE_REVIEW = "review"
NODE_BACKTRANSLATE = "backtranslate"
NODE_TITLES = "titles"
NODE_CONSISTENCY_QA = "consistency_qa"
NODE_REPORT = "report"
NODE_ASSEMBLE = "assemble"

# 节点作用域：book = 全书一次；chapter = 每章一个复合键（chapter_node_key）。
SCOPE_BOOK = "book"
SCOPE_CHAPTER = "chapter"

# 节点产物被其下游消费的固定依赖表：某节点指纹失配时，连同其后代一起失效
# （reconcile 做传递闭包展开）。键为节点基名；值为 (后代基名, 展开方式)：
#   True  = 保持父节点的章节后缀（章内链：translate:0 → polish:0）；
#   False = 后代是书级节点（digest:0 → book_synopsis）；
#   "all" = 书级父 → 章级后代，展开到全部章节（book_synopsis → translate:0..N）。
# 章链：translate 产物（译文）被 polish/naturalize/review/backtranslate 依次消费；
# 章末节点（backtranslate）→ titles → consistency_qa → report → assemble；
# mine_terms → name_terms（候选）→ book_synopsis（人物定名 cast）。
_NODE_DESCENDANTS: dict[str, tuple[tuple[str, bool | str], ...]] = {
    NODE_DIGEST: ((NODE_BOOK_SYNOPSIS, False),),
    NODE_MINE_TERMS: ((NODE_NAME_TERMS, False), (NODE_BOOK_SYNOPSIS, False)),
    NODE_NAME_TERMS: ((NODE_BOOK_SYNOPSIS, False),),
    NODE_BOOK_SYNOPSIS: ((NODE_TRANSLATE, "all"),),
    NODE_TRANSLATE: ((NODE_POLISH, True),),
    NODE_POLISH: ((NODE_NATURALIZE, True),),
    NODE_NATURALIZE: ((NODE_REVIEW, True), (NODE_BACKTRANSLATE, True)),
    NODE_REVIEW: ((NODE_BACKTRANSLATE, True),),
    NODE_BACKTRANSLATE: ((NODE_TITLES, False),),
    NODE_TITLES: ((NODE_CONSISTENCY_QA, False),),
    NODE_CONSISTENCY_QA: ((NODE_REPORT, False),),
    NODE_REPORT: ((NODE_ASSEMBLE, False),),
    NODE_ASSEMBLE: (),
}

# 尽力而为节点：失败后允许继续（事件/报告可见），不阻塞正式产出。
# mine_terms 与 name_terms 共享同一个 term_mining_done 检查点：任一失败都不得
# 落完成标记，续跑重试；也不得阻塞翻译（未定名术语不影响正文翻译）。
BEST_EFFORT_NODES = frozenset({NODE_MINE_TERMS, NODE_NAME_TERMS})


class IdentityMismatchError(RuntimeError):
    """运行身份与当前输入不一致（源文件内容/解析 schema/语言变化）。"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_lang_code(code: str | None) -> str:
    """语言名/代码 → ISO 639-1 两字母代码（模型检测结果归一化）。

    无法解析（auto/unknown/空）返回空串——运行身份里存的是解析后的值，
    绝不用 "auto" 之类未解析字面量充当最终身份。
    """
    aliases = {
        "japanese": "ja",
        "日语": "ja",
        "日文": "ja",
        "jp": "ja",
        "jpn": "ja",
        "english": "en",
        "英语": "en",
        "英文": "en",
        "eng": "en",
        "russian": "ru",
        "俄语": "ru",
        "俄文": "ru",
        "rus": "ru",
        "chinese": "zh",
        "中文": "zh",
        "汉语": "zh",
        "zh-cn": "zh",
        "zho": "zh",
        "korean": "ko",
        "韩语": "ko",
        "韩文": "ko",
        "kor": "ko",
        "french": "fr",
        "法语": "fr",
        "法文": "fr",
        "german": "de",
        "德语": "de",
        "德文": "de",
        "spanish": "es",
        "西班牙语": "es",
        "西班牙文": "es",
        "italian": "it",
        "意大利语": "it",
        "意大利文": "it",
        "portuguese": "pt",
        "葡萄牙语": "pt",
        "葡萄牙文": "pt",
    }
    c = (code or "").strip().lower()
    if not c or c in {"auto", "unknown", "und", "uncertain", "mixed", "多语言", "未知"}:
        return ""
    if c in aliases:
        return aliases[c]
    return c[:2] if c[:2].isalpha() else ""


def source_bytes_hash(path: str) -> str:
    """源文件字节的 sha256；文件不可读时返回空串（迁移期无法核验的标记）。"""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return ""


def input_fingerprint(*parts: object) -> str:
    """对节点输入做规范化 JSON 后取 sha256。

    只覆盖影响该节点的 config/artifact 输入；稳定、可复算，供续跑时判断
    “节点产物是否仍有效”。BaseModel 以 model_dump 参与序列化。
    """
    payload: list[Any] = []
    for part in parts:
        if isinstance(part, BaseModel):
            payload.append(part.model_dump(mode="json"))
        elif isinstance(part, list | tuple):
            payload.append(list(part))
        else:
            payload.append(part)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chapter_node_key(node_id: str, ci: int) -> str:
    """章节范围节点的复合键，如 digest:3。"""
    return f"{node_id}:{ci}"


class RunIdentity(BaseModel):
    """一次运行的唯一身份：源文件字节哈希 + 解析 schema + 解析后的语言。"""

    model_config = ConfigDict(extra="ignore")

    source_bytes_sha256: str = ""
    run_input_schema_version: int = RUN_INPUT_SCHEMA_VERSION
    source_lang: str = ""  # 已归一化（ISO 639-1 两字母，空=无法解析）
    target_lang: str = ""


class NodeFailure(BaseModel):
    """节点失败记录。kind 与提供商/协议/业务语义一一对应，保持可区分。"""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["provider_retryable", "provider_permanent", "protocol", "business", "interrupted"]
    message: str = ""
    at: str = ""


class NodeState(BaseModel):
    """单个节点的执行状态：状态机 + 输入指纹 + 尝试/失败/时间戳。

    ``output`` 是节点成功后需要跨调用持久化的产物（如一致性 QA 的 issues），
    供同书后续服务目标/续跑消费（runner 内 artifacts 只活一轮）。
    """

    model_config = ConfigDict(extra="ignore")

    node_id: str
    status: NodeStatus = NODE_PENDING
    attempts: int = 0
    input_fingerprint: str = ""
    failure: NodeFailure | None = None
    started_at: str | None = None
    finished_at: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)


class PolishBatch(BaseModel):
    """一个待润色批次的定位（章内段起点 + 段数）。"""

    model_config = ConfigDict(extra="ignore")

    start: int
    count: int


class ReviewIssue(BaseModel):
    """审校/确定性 lint 问题项；模型可能带回未知字段，原样保留。"""

    model_config = ConfigDict(extra="allow")

    index: int
    type: str
    detail: str = ""
    fixed: bool = False
    chapter: int | None = None
    stage: str | None = None  # "review" | "lint"
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ReviewIssue:
        return cls.model_validate(d)


class BacktranslationIssue(BaseModel):
    """回译抽检问题项；模型返回字段不固定，正文仅要求 chapter 归因。"""

    model_config = ConfigDict(extra="allow")

    chapter: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktranslationIssue:
        return cls.model_validate(d)


class ChapterProgress(BaseModel):
    """一章的唯一权威进度：状态 + 恢复标记 + 章级流水线产物。

    Chapter.meta 只保留 ingest/EPUB/FB2 元数据；本章所有流水线字段都在这。
    """

    model_config = ConfigDict(extra="ignore")

    status: ChapterStatus = STATUS_PENDING
    review_pending: bool = False
    source_digest: str = ""  # 预扫逐章梗概（产物）
    pending_polish: list[PolishBatch] = Field(default_factory=list)
    naturalized: bool = False
    review_issues: list[ReviewIssue] = Field(default_factory=list)
    backtranslation_issues: list[BacktranslationIssue] = Field(default_factory=list)
    backtranslation_sample_key: str = ""
    backtranslation_sample_indices: list[int] = Field(default_factory=list)
    back_matter_mode: str | None = None  # skip | light | full

    def review_issue_dicts(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.review_issues]

    def set_review_issue_dicts(self, items: list[dict[str, Any]]) -> None:
        self.review_issues = [ReviewIssue.from_dict(i) for i in items]

    def backtranslation_issue_dicts(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.backtranslation_issues]

    def set_backtranslation_issue_dicts(self, items: list[dict[str, Any]]) -> None:
        self.backtranslation_issues = [BacktranslationIssue.from_dict(i) for i in items]


class ChapterIndex(BaseModel):
    """章的不可变索引元数据（manifest 章序/标题/定位信息）。

    不含 status/review_pending 等可变进度——那是 ChapterProgress 的职责，
    manifest 不维护第二份可独立变化的副本。
    """

    model_config = ConfigDict(extra="ignore")

    index: int
    title: str = ""
    href: str | None = None
    toc_entry_id: str | None = None
    title_translated: str | None = None


class AnalysisFlags(BaseModel):
    """V2 拥有的全局完成标志（analysis.json 只留产物，不留标志）。"""

    model_config = ConfigDict(extra="ignore")

    term_mining_done: bool = False


class RunState(BaseModel):
    """V2 根状态：schema 标记 + 运行身份 + 章索引 + 章进度 + 节点状态。"""

    model_config = ConfigDict(extra="ignore")

    run_state_schema: int = RUN_STATE_SCHEMA_VERSION
    identity: RunIdentity = Field(default_factory=RunIdentity)
    title: str = ""
    fmt: str = ""
    source_path: str = ""
    source_lang: str = ""
    target_lang: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)
    initialized: bool = False
    chapters: list[ChapterIndex] = Field(default_factory=list)
    progress: dict[int, ChapterProgress] = Field(default_factory=dict)
    nodes: dict[str, NodeState] = Field(default_factory=dict)
    analysis_flags: AnalysisFlags = Field(default_factory=AnalysisFlags)

    # ── 节点指纹 ──────────────────────────────────────────────────────────
    def reconcile_fingerprints(self, computed: dict[str, str]) -> set[str]:
        """比较存储的节点指纹与最新计算的指纹，失配则失效该节点及其后代。

        只清节点自己的产物与后代的产物（ChapterProgress 字段 / AnalysisFlags /
        节点状态）；book_synopsis 产物在 analysis.json，由 RunStore 调用方负责
        清除；translate 译文与 titles 译名由 RunStore 层一并清除。返回失效的
        完整节点键集合（含后代，如 {"digest:3", "book_synopsis"}）。
        只对“已记录指纹”的节点对账：空指纹（V1 迁移合成的 legacy 成功态）不
        参与失效，避免迁移后的存量书被整体重跑；未出现在 computed 里的已存节点
        不受影响——不失效无关节点。
        """
        invalidated: set[str] = set()
        for key, fingerprint in computed.items():
            node = self.nodes.get(key)
            if node is None or not node.input_fingerprint:
                continue
            if node.input_fingerprint != fingerprint:
                invalidated.add(key)
        # 传递闭包：任一失效节点的全部后代（含书级 → 全部章节的 fan-in）一起失效。
        expanded = set(invalidated)
        queue = list(invalidated)
        while queue:
            key = queue.pop()
            base, sep, suffix = key.partition(":")
            for child, mode in _NODE_DESCENDANTS.get(base, ()):
                if mode == "all":
                    # 书级父 → 章级后代：展开到全部章节（fan-in）。
                    children = [f"{child}:{c.index}" for c in self.chapters]
                elif mode is True:
                    children = [f"{child}:{suffix}"] if sep else []
                else:
                    children = [child]
                for child_key in children:
                    if child_key not in expanded:
                        expanded.add(child_key)
                        queue.append(child_key)
        for key in expanded:
            self._reset_node(key)
            self._clear_node_artifact(key)
        return expanded

    def _reset_node(self, key: str) -> None:
        self.nodes[key] = NodeState(node_id=key)

    def _clear_node_artifact(self, key: str) -> None:
        base, sep, suffix = key.partition(":")
        ci = int(suffix) if sep and suffix.isdigit() else None
        progress = self.progress.get(ci) if ci is not None else None
        if base == NODE_DIGEST:
            if progress is not None:
                progress.source_digest = ""
        elif base in (NODE_MINE_TERMS, NODE_NAME_TERMS):
            self.analysis_flags.term_mining_done = False
        elif base == NODE_BOOK_SYNOPSIS:
            pass  # 产物在 analysis.json，由 RunStore 包装层清除
        elif base == NODE_TRANSLATE:
            # 译文在章节文件，由 RunStore 层清除并重开本章；这里清进度标记。
            if progress is not None:
                progress.pending_polish = []
                progress.naturalized = False
                progress.review_issues = []
                progress.backtranslation_issues = []
        elif base == NODE_POLISH:
            if progress is not None:
                progress.pending_polish = []
        elif base == NODE_NATURALIZE:
            if progress is not None:
                progress.naturalized = False
                progress.review_issues = []
                progress.backtranslation_issues = []
        elif base == NODE_REVIEW:
            if progress is not None:
                progress.review_issues = []
        elif base == NODE_BACKTRANSLATE:
            if progress is not None:
                progress.backtranslation_issues = []
        elif base == NODE_TITLES:
            pass  # 译名在 manifest，由 RunStore 包装层清除

    def recover_interrupted(self) -> bool:
        """进程崩溃遗留的 running 节点恢复为 pending 并记录中断。返回是否有变化。"""
        changed = False
        for node in self.nodes.values():
            if node.status == NODE_RUNNING:
                node.status = NODE_PENDING
                node.failure = NodeFailure(kind="interrupted", message="进程中断", at=now_iso())
                changed = True
        return changed
