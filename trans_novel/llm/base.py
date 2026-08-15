"""LLM provider 的稳定抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trans_novel.llm.json_parser import parse_json_loose
from trans_novel.llm.usage import UsageTracker

Messages = list[dict[str, str]]


class LLMClient(ABC):
    """所有 provider 实现此接口。"""

    def __init__(self) -> None:
        self.usage = UsageTracker()

    def usage_summary(self) -> dict[str, Any]:
        """返回累计 token 用量快照（schema v2）。"""
        return self.usage.summary()

    @abstractmethod
    def complete(
        self,
        messages: Messages,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        stage: str | None = None,
        agent: str,
        operation: str,
    ) -> str:
        """返回模型回复的纯文本。

        agent 是内置功能标识：AgentRouter 将其映射到 primary 或 fast 模型；
        缺失或未知时在请求前失败。operation 是内部业务标签，仅用于调试与用量归因；
        stage 是额外的诊断维度。
        """
        raise NotImplementedError

    def complete_json(
        self,
        messages: Messages,
        *,
        max_tokens: int | None = None,
        stage: str | None = None,
        agent: str,
        operation: str,
    ) -> Any:
        """要求 JSON 输出并解析；解析失败不触发模型降级。"""
        text = self.complete(
            messages,
            json_mode=True,
            max_tokens=max_tokens,
            stage=stage,
            agent=agent,
            operation=operation,
        )
        return parse_json_loose(text)
