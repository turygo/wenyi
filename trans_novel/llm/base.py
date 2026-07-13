"""LLM 抽象接口与具体实现。

设计要点：
- 三档 tier："strong"（deepseek-v4-pro + thinking，翻译/润色/分析/审计）、
  "cheap"（deepseek-v4-flash + thinking，审校/一致性等判断类）、
  "fast"（deepseek-v4-flash 免思考，梗概/术语抽取/回译等机械任务——
  thinking 推理 token 按输出计费，机械任务关掉可大幅省钱提速）。
  缺档时按回退链向"更便宜优先"回退（fast→cheap→strong），老双档配置行为不变。
- complete() 返回纯文本；complete_json() 强制 JSON 输出并 loose 解析。
- DeepSeekClient 经由 OpenAI SDK 调 https://api.deepseek.com，openai 惰性导入；
  未装 openai 时仍可用 FakeClient 跑通离线流程（切分/对齐/术语库/状态机）。
"""

from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Config, LLMConfig, TierConfig

Messages = list[dict[str, str]]

# 缺档回退链：向"更便宜优先"回退，绝不因缺档反而升到更贵的档
_TIER_FALLBACK = {"fast": ("cheap", "strong"), "cheap": ("strong",), "strong": ()}


def resolve_tier(tiers: dict[str, TierConfig], tier: str) -> TierConfig:
    """按回退链解析 tier 配置。缺 strong 时 KeyError（与旧行为一致）。"""
    if tier in tiers:
        return tiers[tier]
    for fb in _TIER_FALLBACK.get(tier, ("strong",)):
        if fb in tiers:
            return tiers[fb]
    return tiers["strong"]


# ── JSON 宽松解析 ────────────────────────────────────────────────────────
def _repair_unescaped_quotes(text: str) -> str:
    """转义 JSON 字符串值内部未转义的 ASCII 双引号。

    部分模型（尤其无原生 JSON 模式的 provider）会在译文里原样输出英文引号。
    启发式：字符串内的 `"` 后面（跳过空白）若不是 `,:]}`，视为内容引号转义之。
    中文译文以全角标点为主，误判面极小；仅作为常规解析失败后的兜底。
    """
    out: list[str] = []
    in_str = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
        elif c == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        elif c == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ",:]}":
                in_str = False
                out.append(c)
            else:
                out.append('\\"')
        else:
            out.append(c)
        i += 1
    return "".join(out)


def parse_json_loose(text: str) -> Any:
    """从模型输出里尽力解析 JSON。

    优先直接 json.loads；失败则剥离 ```json 围栏并截取首个 {…}/[…] 块再试。
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 去掉 markdown 代码围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        inner = fenced.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            text = inner
    # 截取首个 JSON 数组或对象
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = text.find(open_ch)
        end = text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                continue
    # 从首个 {/[ 起解析第一个完整 JSON 值，忽略尾部多余字符（如重复的 }）
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[min(starts) :])
            return value
        except Exception:
            pass
    # 最后兜底：修复字符串内未转义的引号，再从完整文本解析首个 JSON 值。
    # 必须先做这一步：若同时有未转义引号和尾部多余字符，直接截取内部数组
    # 会丢掉外层对象（如 {"translations": [...]}）。
    repaired = _repair_unescaped_quotes(text)
    starts = [i for i in (repaired.find("{"), repaired.find("[")) if i != -1]
    if starts:
        try:
            value, _ = json.JSONDecoder().raw_decode(repaired[min(starts) :])
            return value
        except Exception:
            pass

    # 修复后仍无法解析时，才依次尝试完整文本和对象/数组片段。
    for candidate in (
        text,
        *(
            text[s : e + 1]
            for o, c in (("[", "]"), ("{", "}"))
            for s, e in [(text.find(o), text.rfind(c))]
            if s != -1 and e > s
        ),
    ):
        try:
            return json.loads(_repair_unescaped_quotes(candidate))
        except Exception:
            continue
    raise ValueError(f"无法解析为 JSON：{text[:200]!r}")


# ── operation 名合法性（决策 44/59：production callsite 必须显式标注）───────
def _has_field(usage: Any, name: str) -> bool:
    if isinstance(usage, dict):
        return name in usage
    return hasattr(usage, name)


def _reasoning_tokens(usage: Any) -> int:
    """读取推理 token：优先直接字段，否则 completion_tokens_details.reasoning_tokens。

    不再叠加进 total_tokens（decision 46）。
    """
    if usage is None:
        return 0
    if _has_field(usage, "reasoning_tokens"):
        return _usage_int(usage, "reasoning_tokens")
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    if details is not None:
        return _usage_int(details, "reasoning_tokens")
    return 0


# ── Token 用量统计 ────────────────────────────────────────────────────────
_USAGE_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
)

# by_operation 槽位在 _USAGE_FIELDS 之上再加底层 attempt/latency/reasoning/采纳统计。
_OPERATION_EXTRA_FIELDS = (
    "logical_calls",
    "attempts",
    "failed_attempts",
    "elapsed_ms",
    "reasoning_tokens",
    "accepted",
    "rejected",
)
_OPERATION_FIELDS = _USAGE_FIELDS + _OPERATION_EXTRA_FIELDS


def _usage_int(usage: Any, name: str) -> int:
    """从响应 usage 对象/字典读取整数字段，缺失或非数返回 0。"""
    val = getattr(usage, name, None)
    if val is None and isinstance(usage, dict):
        val = usage.get(name)
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0


def _hit_rate(hit: int, miss: int) -> float:
    total = hit + miss
    return round(hit / total, 4) if total else 0.0


def _normalize_slot(values: dict, fields: tuple[str, ...]) -> dict[str, int]:
    """规范化单个槽位：canonical 字段集合，缺失按 0 补齐（向后兼容旧快照）。"""
    return {f: _usage_int(values, f) for f in fields}


def _usage_summary_from_parts(
    by_tier: dict[str, dict[str, int]],
    by_stage: dict[str, dict[str, int]],
    by_operation: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """由各档/各阶段/各 operation 计数生成规范汇总；总计由 by_tier 求和，各槽位补 cache_hit_rate。"""
    tiers = {t: _normalize_slot(v, _USAGE_FIELDS) for t, v in by_tier.items()}
    stages = {s: _normalize_slot(v, _USAGE_FIELDS) for s, v in by_stage.items()}
    operations = {o: _normalize_slot(v, _OPERATION_FIELDS) for o, v in (by_operation or {}).items()}
    totals: dict[str, Any] = dict.fromkeys(_USAGE_FIELDS, 0)
    for v in tiers.values():
        for f in _USAGE_FIELDS:
            totals[f] += v[f]
    for slot in (*tiers.values(), *stages.values(), *operations.values(), totals):
        slot["cache_hit_rate"] = _hit_rate(slot["cache_hit_tokens"], slot["cache_miss_tokens"])
    return {"totals": totals, "by_tier": tiers, "by_stage": stages, "by_operation": operations}


def _nonneg_delta(
    current: dict[str, dict[str, int]],
    previous: dict[str, dict[str, int]],
    fields: tuple[str, ...] = _USAGE_FIELDS,
) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    for key, values in current.items():
        old = previous.get(key) or {}
        slot = {f: max(0, _usage_int(values, f) - _usage_int(old, f)) for f in fields}
        if any(slot.values()):
            delta[key] = slot
    return delta


def usage_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """计算两个累计快照间的非负增量（by_tier / by_stage / by_operation 各自），用于避免重复落盘。"""
    return _usage_summary_from_parts(
        _nonneg_delta(current.get("by_tier") or {}, previous.get("by_tier") or {}),
        _nonneg_delta(current.get("by_stage") or {}, previous.get("by_stage") or {}),
        _nonneg_delta(
            current.get("by_operation") or {}, previous.get("by_operation") or {}, _OPERATION_FIELDS
        ),
    )


def merge_usage_summaries(accumulated: dict[str, Any], increment: dict[str, Any]) -> dict[str, Any]:
    """把一次运行增量合并进某本书的历史累计用量（by_tier / by_stage / by_operation 同时合并）。"""

    def _merge(field_name: str, fields: tuple[str, ...]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for summary in (accumulated, increment):
            for key, values in (summary.get(field_name) or {}).items():
                slot = merged.setdefault(key, dict.fromkeys(fields, 0))
                for f in fields:
                    slot[f] += _usage_int(values, f)
        return merged

    return _usage_summary_from_parts(
        _merge("by_tier", _USAGE_FIELDS),
        _merge("by_stage", _USAGE_FIELDS),
        _merge("by_operation", _OPERATION_FIELDS),
    )


class UsageTracker:
    """线程安全的 token 用量累加器，按 tier 分档、按 stage 分阶段、按 operation 归因统计
    （worker 线程并发共享一个 client）。

    DeepSeek 的 usage 里 prompt_cache_hit_tokens + prompt_cache_miss_tokens == prompt_tokens；
    缓存命中率 = cache_hit /(cache_hit + cache_miss)。fake provider 不产生 usage，保持空。
    stage 由调用方标注（agent 基类默认传类名，如 Translator/Reviewer），用于成本归因。
    operation 是更细的稳定业务标签（如 translate.batch/naturalize.pair），额外记录
    logical_calls/attempts/failed_attempts/elapsed_ms/reasoning_tokens/accepted/rejected。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_tier: dict[str, dict[str, int]] = {}
        self._by_stage: dict[str, dict[str, int]] = {}
        self._by_operation: dict[str, dict[str, int]] = {}

    def _op_slot_locked(self, operation: str) -> dict[str, int]:
        return self._by_operation.setdefault(operation, dict.fromkeys(_OPERATION_FIELDS, 0))

    def record(
        self, tier: str, usage: Any, stage: str | None = None, operation: str | None = None
    ) -> None:
        """累加一次响应的 usage；usage 缺失时静默跳过（不影响正常返回）。"""
        if usage is None:
            return
        pt = _usage_int(usage, "prompt_tokens")
        ct = _usage_int(usage, "completion_tokens")
        tt = _usage_int(usage, "total_tokens") or (pt + ct)
        hit = _usage_int(usage, "prompt_cache_hit_tokens")
        miss = _usage_int(usage, "prompt_cache_miss_tokens")
        reasoning = _reasoning_tokens(usage)
        with self._lock:
            slots = [self._by_tier.setdefault(tier, dict.fromkeys(_USAGE_FIELDS, 0))]
            if stage:
                slots.append(self._by_stage.setdefault(stage, dict.fromkeys(_USAGE_FIELDS, 0)))
            for slot in slots:
                slot["calls"] += 1
                slot["prompt_tokens"] += pt
                slot["completion_tokens"] += ct
                slot["total_tokens"] += tt
                slot["cache_hit_tokens"] += hit
                slot["cache_miss_tokens"] += miss
            if operation:
                op_slot = self._op_slot_locked(operation)
                op_slot["calls"] += 1
                op_slot["prompt_tokens"] += pt
                op_slot["completion_tokens"] += ct
                op_slot["total_tokens"] += tt
                op_slot["cache_hit_tokens"] += hit
                op_slot["cache_miss_tokens"] += miss
                op_slot["reasoning_tokens"] += reasoning

    def record_attempt(self, operation: str | None) -> None:
        """每次底层 provider create 尝试计一次（成功/失败均计）。"""
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["attempts"] += 1

    def record_attempt_failed(self, operation: str | None) -> None:
        """抛异常的底层尝试计一次。"""
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["failed_attempts"] += 1

    def record_logical_call(self, operation: str | None, elapsed_ms: float) -> None:
        """一次逻辑 complete() 调用计一次（含其全部重试等待），无论成败。"""
        if not operation:
            return
        with self._lock:
            slot = self._op_slot_locked(operation)
            slot["logical_calls"] += 1
            slot["elapsed_ms"] += int(round(elapsed_ms))

    def record_outcome(self, operation: str | None, *, accepted: bool) -> None:
        """记录一次业务采纳结果（润色/去腔改写/定向重译等，与 LLM 调用本身解耦）。"""
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["accepted" if accepted else "rejected"] += 1

    def summary(self) -> dict[str, Any]:
        """返回 {"totals", "by_tier", "by_stage", "by_operation"}，各槽位含 cache_hit_rate。"""
        with self._lock:
            by_tier = {t: dict(v) for t, v in self._by_tier.items()}
            by_stage = {s: dict(v) for s, v in self._by_stage.items()}
            by_operation = {o: dict(v) for o, v in self._by_operation.items()}
        return _usage_summary_from_parts(by_tier, by_stage, by_operation)


# ── 抽象接口 ──────────────────────────────────────────────────────────────
class LLMClient(ABC):
    """所有 provider 实现此接口。"""

    def __init__(self) -> None:
        self.usage = UsageTracker()

    def usage_summary(self) -> dict[str, Any]:
        """返回累计 token 用量快照（totals + by_tier + by_stage + by_operation + cache_hit_rate）。"""
        return self.usage.summary()

    @abstractmethod
    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> str:
        """返回模型回复的纯文本。stage 仅用于用量归因，不影响请求。

        operation 是稳定、可读的业务标签（如 translate.batch），驱动 by_operation 归因；
        production callsite 必须显式传入。
        """
        raise NotImplementedError

    def complete_json(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> Any:
        """要求 JSON 输出并解析。"""
        text = self.complete(
            messages,
            tier=tier,
            json_mode=True,
            max_tokens=max_tokens,
            stage=stage,
            operation=operation,
        )
        return parse_json_loose(text)


# ── DeepSeek（OpenAI SDK 兼容）────────────────────────────────────────────
class DeepSeekClient(LLMClient):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        if not cfg.tiers:
            raise ValueError("配置缺少 llm.tiers")
        self._client = None  # 惰性创建
        self._client_lock = threading.Lock()  # 预扫并行时防惰性初始化竞态

    def _ensure_client(self):
        with self._client_lock:
            return self._ensure_client_locked()

    def _ensure_client_locked(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "需要 openai SDK：pip install openai（或把 llm.provider 设为 fake 做离线测试）"
                ) from e
            api_key = self.cfg.api_key
            if not api_key:
                raise RuntimeError(f"未设置环境变量 {self.cfg.api_key_env}（DeepSeek API key）")
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.cfg.base_url,
                timeout=self.cfg.timeout,
            )
        return self._client

    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> str:
        # 一次逻辑 complete() 调用只计一次 logical_calls/elapsed_ms，无论成败，且
        # 覆盖从入口开始的全部工作——tier 解析失败（缺 strong 档 KeyError）、
        # _ensure_client 初始化失败（缺 API key/未装 openai SDK）都算作这次逻辑调用
        # 的一部分，必须先记账再原样重抛，而不仅是重试包裹的 create() 那一段。
        start = time.monotonic()
        try:
            tcfg = resolve_tier(self.cfg.tiers, tier)
            client = self._ensure_client()

            kwargs: dict[str, Any] = {
                "model": tcfg.model,
                "messages": messages,
                "stream": False,
            }
            if tcfg.thinking:
                kwargs["reasoning_effort"] = tcfg.reasoning_effort
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                # API 缺省是开思考：必须显式 disabled，否则"关思考省钱"的档位配置形同虚设
                # （实测同一请求：缺省 88 tok 带 reasoning，显式 disabled 10 tok）。
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if max_tokens:
                # DeepSeek thinking 模式下 max_tokens 含推理 token（总输出上限）。
                # 带紧上限的调用若经回退链落到 thinking 档，抬到安全下限防推理被截断。
                kwargs["max_tokens"] = max(max_tokens, 4096) if tcfg.thinking else max_tokens

            # 网络/限流/超时 → tenacity 指数退避重试（最多 max_retries 次重试）；
            # 每次真实 create() 尝试计 attempts，抛异常的尝试再计 failed_attempts。
            @retry(
                stop=stop_after_attempt(self.cfg.max_retries + 1),
                wait=wait_exponential(multiplier=1, max=30),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )
            def _call() -> str:
                self.usage.record_attempt(operation)
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception:
                    self.usage.record_attempt_failed(operation)
                    raise
                self.usage.record(tier, getattr(resp, "usage", None), stage, operation=operation)
                return resp.choices[0].message.content or ""

            return _call()
        finally:
            self.usage.record_logical_call(operation, (time.monotonic() - start) * 1000)


# ── 离线 Fake（测试 / 不发网络请求）───────────────────────────────────────
class FakeClient(LLMClient):
    """可编程的离线 client。

    handler(messages, tier, json_mode) -> str。默认对 json_mode 返回 "[]"，
    否则返回空串。测试通过注入 handler 模拟翻译/抽取等行为。
    """

    def __init__(self, handler: Optional[Callable[[Messages, str, bool], str]] = None):
        super().__init__()
        self.handler = handler
        self.calls: list[dict[str, Any]] = []  # 记录调用，便于断言
        self._calls_lock = threading.Lock()  # 只保护 calls 列表本身，绝不在持锁时调 handler

    def complete(
        self,
        messages: Messages,
        *,
        tier: str = "strong",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> str:
        record = {
            "messages": messages,
            "tier": tier,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "stage": stage,
            "operation": operation,
        }
        with self._calls_lock:
            self.calls.append(record)
        # FakeClient 不产生真实 provider usage，只记录 operation 的 attempt/logical_call
        # 计数，便于测试断言标注与并发行为；绝不伪造 token。
        self.usage.record_attempt(operation)
        start = time.monotonic()
        try:
            if self.handler is not None:
                return self.handler(messages, tier, json_mode)
            return "[]" if json_mode else ""
        except Exception:
            self.usage.record_attempt_failed(operation)
            raise
        finally:
            self.usage.record_logical_call(operation, (time.monotonic() - start) * 1000)


def build_client(config: Config) -> LLMClient:
    provider = config.llm.provider.lower()
    if provider == "deepseek":
        return DeepSeekClient(config.llm)
    if provider == "fake":
        return FakeClient()
    raise ValueError(f"未知 provider：{provider}（支持 deepseek / fake）")
