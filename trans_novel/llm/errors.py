"""LLM 路由的公共异常类型。"""

from __future__ import annotations

from trans_novel.config import ModelRef


class JSONParseError(ValueError):
    """模型输出经宽松解析后仍无法得到 JSON。"""


class UnknownAgentError(ValueError):
    """agent 缺失或不是内置生产 Agent。"""


class AllModelsFailedError(RuntimeError):
    """当前模型耗尽内部重试后抛出。

    只包含 provider:model 与固定归一化原因，绝不包含 prompt、API key 或响应正文。
    """

    def __init__(self, records: tuple[tuple[ModelRef, str], ...]) -> None:
        self.records: tuple[tuple[ModelRef, str], ...] = tuple(records)
        summary = "; ".join(f"{ref.full_name}: {reason}" for ref, reason in self.records)
        super().__init__(summary)

    @property
    def records_data(self) -> tuple[tuple[str, str], ...]:
        """(provider:model, reason) 纯字符串视图，便于序列化/断言。"""
        return tuple((ref.full_name, reason) for ref, reason in self.records)
