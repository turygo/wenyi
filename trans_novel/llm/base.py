"""LLM provider 的稳定抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from .json_parser import parse_json_loose
from .usage import UsageTracker

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
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        agent: str,
        operation: str,
    ) -> str:
        """返回模型回复的纯文本。

        agent 是生产路由键：AgentRouter 按 llm.agents 中声明的 Agent 选择
        primary/fallback 模型；缺失/未知时在请求前失败。operation 只是内部
        业务标签（telemetry/调试归因），不参与路由，但生产调用方必须显式传入。
        stage 仅用于用量归因（诊断维度）。
        """
        raise NotImplementedError

    def complete_json(
        self,
        messages: Messages,
        *,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
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
