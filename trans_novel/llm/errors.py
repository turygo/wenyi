"""LLM 路由与降级的公共异常类型。"""

from __future__ import annotations

from ..config import ModelRef


class UnknownAgentError(ValueError):
    """agent 缺失或未在 llm.agents 中声明路由（调用方/配置错误）。"""


class AllModelsFailedError(RuntimeError):
    """全部可降级候选重试耗尽后抛出；records 为不可变的脱敏 (ModelRef, reason) 记录。

    只包含 provider:model 与固定归一化 reason，绝不包含 prompt、API key 或响应正文。
    """

    def __init__(self, records: tuple[tuple[ModelRef, str], ...]) -> None:
        self.records: tuple[tuple[ModelRef, str], ...] = tuple(records)
        summary = "; ".join(f"{ref.full_name}: {reason}" for ref, reason in self.records)
        super().__init__(summary)

    @property
    def records_data(self) -> tuple[tuple[str, str], ...]:
        """(provider:model, reason) 纯字符串视图，便于序列化/断言。"""
        return tuple((ref.full_name, reason) for ref, reason in self.records)
