"""Workflow 契约：节点协议、执行目标与失败分类。

这一层只定义形状，不包含任何具体 Agent/节点实现。runner 依赖本模块 +
definition/planner/state，绝不 import 具体节点或 Agent。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.agents.reviewer import ReviewOutputError
from trans_novel.llm.errors import AllModelsFailedError
from trans_novel.llm.retrying import classify_retry
from trans_novel.pipeline.readiness import ReadinessError
from trans_novel.pipeline.state import IdentityMismatchError

Scope = Literal["book", "chapter"]
FailurePolicy = Literal["required", "best_effort"]
NodeAction = Literal["run", "skip"]

# 失败 kind 与状态契约一一对应（见 state.NodeFailure）。
FAILURE_PROTOCOL = "protocol"
FAILURE_BUSINESS = "business"
FAILURE_PROVIDER_RETRYABLE = "provider_retryable"
FAILURE_PROVIDER_PERMANENT = "provider_permanent"


def classify_failure(exc: BaseException) -> str:
    """把节点执行期间冒出的异常归类为稳定的失败 kind。

    - ReviewOutputError / WorkflowProtocolError → protocol（输出协议错误，可重试）；
    - AllModelsFailedError → provider_retryable（路由器在可重试原因上耗尽内部重试，
      属于可重试的提供商耗尽，不是永久拒绝；永久原始提供商错误走非重试路径）；
    - IdentityMismatchError / ValueError / ReadinessError → business；
    - 其它 provider 可重试/可降级原因 → provider_retryable；
    - 其余 → provider_permanent。
    """
    if isinstance(exc, ReviewOutputError | WorkflowProtocolError):
        return FAILURE_PROTOCOL
    if isinstance(exc, AllModelsFailedError):
        return FAILURE_PROVIDER_RETRYABLE
    if isinstance(exc, IdentityMismatchError | ReadinessError):
        return FAILURE_BUSINESS
    if isinstance(exc, ValueError):
        return FAILURE_BUSINESS
    if classify_retry(exc) is not None:
        return FAILURE_PROVIDER_RETRYABLE
    return FAILURE_PROVIDER_PERMANENT


def failure_reason(exc: BaseException) -> str:
    """异步排干失败时的稳定 reason：协议错误用其稳定标识，其余归一为 provider_error。"""
    if isinstance(exc, ReviewOutputError | WorkflowProtocolError):
        return exc.reason
    return "provider_error"


ProgressFn = Callable[[int, int, str], None]


class RunRepository(Protocol):
    """runner 依赖的仓库面（组合根注入 RunStore 实现）。

    workflow kernel 只依赖本协议，不 import 具体 RunStore 的文件系统/迁移实现；
    节点仍通过 request.store 使用同一仓库（其完整 API 由节点契约另行承担）。
    """

    def exists(self) -> bool: ...

    def lock(self): ...  # 上下文管理器：进入即触发迁移/中断恢复/检查点恢复

    def load_state(self): ...

    def save_state(self, state) -> None: ...

    def mark_node_running(self, key: str) -> None: ...

    def mark_node_skipped(self, key: str) -> None: ...

    def mark_node_succeeded(self, key: str, fingerprint: str | None = None) -> None: ...

    def record_node_fingerprint(self, key: str, fingerprint: str) -> None: ...

    def fail_node(self, key: str, kind: str, message: str = "") -> None: ...

    def load_progress(self, ci: int): ...

    def save_progress(self, ci: int, progress) -> None: ...

    def load_chapter(self, ci: int): ...

    def log_event(self, event: str, **data: Any) -> None: ...

    def set_chapter_status(self, ci: int, status: str) -> None: ...


@dataclass
class NodeOutcome:
    """单个节点一次执行的结果。成功是默认路径；失败一律用异常表达。

    - ``fingerprint``：节点自己计算的输入指纹（无指纹契约的节点留空）；
    - ``async_handle``：异步节点（如异步审校）提交后台任务后立即返回的句柄，
      runner 排干时回调该节点的 ``finish()``；
    - ``artifacts``：供同轮后续节点消费的内存产物（如挖掘出的候选）；
    - ``commit``：并行层节点在 worker 线程只做 LLM 计算，产物先放进 artifacts，
      commit 回调由 runner 在 join 后于主线程按确定顺序执行（RunStore 写回
      主线程的仓库约定，见 DigestNode/MineTermsNode）；
    - ``chapter_finalized``：节点已自行完成本章收尾（旁路/空章），runner
      不再代做 done 标记与 chapter_done 事件。
    """

    fingerprint: str | None = None
    async_handle: Any = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    commit: Callable[[NodeRequest], None] | None = None
    chapter_finalized: bool = False
    findings_count: int = 0


@dataclass
class NodeRequest:
    """runner 交给节点的一次执行上下文。business 逻辑全部由节点自理。"""

    store: Any  # RunStore（避免本模块 import runstore，保持契约纯净）
    node_id: str
    key: str  # 复合键：book 节点=node_id，chapter 节点=node_id:ci
    ci: int | None
    scope: Scope
    input_path: str
    progress: ProgressFn | None
    executor: Any = None  # 共享 4-worker 线程池（翻译/润色/审校提交）
    review_executor: Any = None  # 独立审校 chunk 池（防嵌套死锁）
    artifacts: dict[str, Any] = field(default_factory=dict)
    shared: Any = None  # 本轮运行共享状态（滚动上下文等），runner 不透传语义
    finalize_chapter: bool = False
    total_chapters: int = 0


class WorkflowNode(Protocol):
    """具体节点实现协议。节点构造时收到全部精确依赖。"""

    node_id: str
    scope: Scope

    def execute(self, request: NodeRequest) -> NodeOutcome: ...

    def finish(self, request: NodeRequest, handle: Any) -> None:
        """异步节点排干：把后台任务结果写回（默认无操作）。"""
        return None


NodeFactory = Callable[[str, int | None], WorkflowNode]


@dataclass(frozen=True)
class ExecutionGoal:
    """一次运行的执行目标：有序阶段列表 + 章范围/输出参数。

    phases 取值（顺序即执行顺序）：
      "prepare"    准备：解析、语言、身份、暂存文档、风格分析；
      "prescan"    全书预扫：digest / mine_terms / name_terms / book_synopsis；
      "translate"  逐章翻译（含质量环节）与异步审校补跑；
      "titles"     章标题/目录项翻译；
      "qa"         跨章一致性扫描；
      "report"     QA 报告；
      "assemble"   正式回填（就绪门禁）。
    """

    name: str
    phases: tuple[str, ...]
    only_chapter: int | None = None
    out_format: str = "epub"
    out_path: str | None = None
    do_qa: bool | None = None  # run_all：覆盖质量档位中的 consistency_qa


GOAL_PREPARE = ExecutionGoal(name="prepare", phases=("prepare", "prescan"))
GOAL_TRANSLATE = ExecutionGoal(
    name="translate", phases=("prepare", "prescan", "translate", "titles")
)
GOAL_RUN_ALL = ExecutionGoal(
    name="run_all",
    phases=("prepare", "prescan", "translate", "titles", "qa", "report", "assemble"),
)


def translate_chapter_goal(ci: int) -> ExecutionGoal:
    """只翻指定章（调试用，不做收尾）：预扫照常，但只译一章、不译标题。"""
    return ExecutionGoal(
        name=f"translate_chapter:{ci}",
        phases=("prepare", "prescan", "translate"),
        only_chapter=ci,
    )


def qa_goal() -> ExecutionGoal:
    """显式 QA 目标：无论质量档位如何都强制运行（tools qa / 测试）。

    服务目标不含 prepare 阶段（存量书直接读状态，与旧 tools 命令一致；
    正式回填的身份核验由 writer 就绪门禁负责）。
    """
    return ExecutionGoal(name="qa", phases=("qa",), do_qa=True)


def report_goal() -> ExecutionGoal:
    return ExecutionGoal(name="report", phases=("report",))


def assemble_goal(*, out_format: str = "epub", out_path: str | None = None) -> ExecutionGoal:
    return ExecutionGoal(
        name="assemble",
        phases=("assemble",),
        out_format=out_format,
        out_path=out_path,
    )


def titles_goal() -> ExecutionGoal:
    """仅标题翻译（供独立工具/测试复用 titles 节点）。"""
    return ExecutionGoal(name="titles", phases=("titles",))
