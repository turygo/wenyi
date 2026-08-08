"""线程安全的 Token 用量统计、增量计算与持久化合并。"""

from __future__ import annotations

import threading
from typing import Any

_USAGE_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
)
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


def _has_field(usage: Any, name: str) -> bool:
    if isinstance(usage, dict):
        return name in usage
    return hasattr(usage, name)


def _usage_int(usage: Any, name: str) -> int:
    """从响应 usage 对象或字典读取整数字段，缺失或非数返回 0。"""
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(name)
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _reasoning_tokens(usage: Any) -> int:
    """读取推理 token；不叠加进 total_tokens。"""
    if usage is None:
        return 0
    if _has_field(usage, "reasoning_tokens"):
        return _usage_int(usage, "reasoning_tokens")
    details = (
        usage.get("completion_tokens_details")
        if isinstance(usage, dict)
        else getattr(usage, "completion_tokens_details", None)
    )
    return _usage_int(details, "reasoning_tokens") if details is not None else 0


def _hit_rate(hit: int, miss: int) -> float:
    total = hit + miss
    return round(hit / total, 4) if total else 0.0


def _normalize_slot(values: dict, fields: tuple[str, ...]) -> dict[str, int]:
    return {field: _usage_int(values, field) for field in fields}


def _usage_summary_from_parts(
    by_tier: dict[str, dict[str, int]],
    by_stage: dict[str, dict[str, int]],
    by_operation: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """由各归因维度生成规范汇总，总计仅由 tier 求和。"""
    tiers = {name: _normalize_slot(values, _USAGE_FIELDS) for name, values in by_tier.items()}
    stages = {name: _normalize_slot(values, _USAGE_FIELDS) for name, values in by_stage.items()}
    operations = {
        name: _normalize_slot(values, _OPERATION_FIELDS)
        for name, values in (by_operation or {}).items()
    }
    totals: dict[str, Any] = dict.fromkeys(_USAGE_FIELDS, 0)
    for values in tiers.values():
        for field in _USAGE_FIELDS:
            totals[field] += values[field]
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
        slot = {
            field: max(0, _usage_int(values, field) - _usage_int(old, field)) for field in fields
        }
        if any(slot.values()):
            delta[key] = slot
    return delta


def usage_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """计算累计快照的非负增量，避免重复落盘。"""
    return _usage_summary_from_parts(
        _nonneg_delta(current.get("by_tier") or {}, previous.get("by_tier") or {}),
        _nonneg_delta(current.get("by_stage") or {}, previous.get("by_stage") or {}),
        _nonneg_delta(
            current.get("by_operation") or {},
            previous.get("by_operation") or {},
            _OPERATION_FIELDS,
        ),
    )


def merge_usage_summaries(accumulated: dict[str, Any], increment: dict[str, Any]) -> dict[str, Any]:
    """将一次运行增量合并进历史累计用量。"""

    def merge(field_name: str, fields: tuple[str, ...]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for summary in (accumulated, increment):
            for key, values in (summary.get(field_name) or {}).items():
                slot = merged.setdefault(key, dict.fromkeys(fields, 0))
                for field in fields:
                    slot[field] += _usage_int(values, field)
        return merged

    return _usage_summary_from_parts(
        merge("by_tier", _USAGE_FIELDS),
        merge("by_stage", _USAGE_FIELDS),
        merge("by_operation", _OPERATION_FIELDS),
    )


class UsageTracker:
    """线程安全的 token 用量累加器，按 tier、stage 与 operation 归因统计。"""

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
        """累加一次响应的 usage；usage 缺失时不影响正常返回。"""
        if usage is None:
            return
        prompt_tokens = _usage_int(usage, "prompt_tokens")
        completion_tokens = _usage_int(usage, "completion_tokens")
        total_tokens = _usage_int(usage, "total_tokens") or (prompt_tokens + completion_tokens)
        cache_hit_tokens = _usage_int(usage, "prompt_cache_hit_tokens")
        cache_miss_tokens = _usage_int(usage, "prompt_cache_miss_tokens")
        reasoning_tokens = _reasoning_tokens(usage)
        with self._lock:
            slots = [self._by_tier.setdefault(tier, dict.fromkeys(_USAGE_FIELDS, 0))]
            if stage:
                slots.append(self._by_stage.setdefault(stage, dict.fromkeys(_USAGE_FIELDS, 0)))
            for slot in slots:
                slot["calls"] += 1
                slot["prompt_tokens"] += prompt_tokens
                slot["completion_tokens"] += completion_tokens
                slot["total_tokens"] += total_tokens
                slot["cache_hit_tokens"] += cache_hit_tokens
                slot["cache_miss_tokens"] += cache_miss_tokens
            if operation:
                operation_slot = self._op_slot_locked(operation)
                operation_slot["calls"] += 1
                operation_slot["prompt_tokens"] += prompt_tokens
                operation_slot["completion_tokens"] += completion_tokens
                operation_slot["total_tokens"] += total_tokens
                operation_slot["cache_hit_tokens"] += cache_hit_tokens
                operation_slot["cache_miss_tokens"] += cache_miss_tokens
                operation_slot["reasoning_tokens"] += reasoning_tokens

    def record_attempt(self, operation: str | None) -> None:
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["attempts"] += 1

    def record_attempt_failed(self, operation: str | None) -> None:
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["failed_attempts"] += 1

    def record_logical_call(self, operation: str | None, elapsed_ms: float) -> None:
        if not operation:
            return
        with self._lock:
            slot = self._op_slot_locked(operation)
            slot["logical_calls"] += 1
            slot["elapsed_ms"] += int(round(elapsed_ms))

    def record_outcome(self, operation: str | None, *, accepted: bool) -> None:
        if not operation:
            return
        with self._lock:
            self._op_slot_locked(operation)["accepted" if accepted else "rejected"] += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            by_tier = {name: dict(values) for name, values in self._by_tier.items()}
            by_stage = {name: dict(values) for name, values in self._by_stage.items()}
            by_operation = {name: dict(values) for name, values in self._by_operation.items()}
        return _usage_summary_from_parts(by_tier, by_stage, by_operation)
