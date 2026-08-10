"""AgentRouter：按 Agent 解析 primary/fallback 路由并执行。

- agent 必填且必须已在 llm.agents 声明；缺失/未知在逻辑记账与任何请求之前失败。
- operation 是内部业务标签（telemetry/调试归因），不参与路由，但调用方必须显式传入。
- 每个候选由对应 provider 传输执行自己的 max_retries+1 次物理尝试；
  只有可重试错误在当前候选上的重试次数用尽后，才尝试下一个候选。
- 不可重试错误（配置/凭据/4xx 非 429/未知异常）立即原样抛出，不改用其他候选。
- 全部候选耗尽时抛 AllModelsFailedError，只含脱敏 (provider:model, reason) 记录。
- 每次逻辑调用都会在 by_agent / by_operation 中将 logical_calls 加 1，并累计 elapsed_ms。
- complete_json 只路由一次逻辑调用并恰好解析一次；解析失败不触发候选切换。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ..config import Config, ModelRef
from .base import LLMClient, Messages
from .errors import AllModelsFailedError, UnknownAgentError
from .json_parser import parse_json_loose
from .registry import ProviderRegistry
from .retrying import classify_retry


class AgentRouter(LLMClient):
    def __init__(
        self,
        config: Config,
        *,
        registry: Optional[ProviderRegistry] = None,
        transports: Optional[dict] = None,
    ) -> None:
        """registry/transports 是内部/测试注入边界：确定性 stub 传输无需生产插件或 YAML hook。"""
        super().__init__()
        self.config = config
        if registry is not None:
            self._registry = registry
            self.usage = registry.usage
        else:
            self._registry = ProviderRegistry(config.llm, self.usage, transports=transports)

    def _route(self, agent: Optional[str]):
        if not agent or agent not in self.config.llm.agents:
            raise UnknownAgentError(
                f"未知或缺失 Agent：{agent!r}（生产 LLM 调用必须提供配置中已声明的 Agent 路由）"
            )
        return self.config.llm.agents[agent]

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
        route = self._route(agent)  # 未知 Agent：请求与逻辑记账之前即失败
        start = time.monotonic()
        try:
            failures: list[tuple[ModelRef, str]] = []
            candidates = (route.model, *route.fallback)
            last_error: Optional[BaseException] = None
            for index, ref in enumerate(candidates):
                transport = self._registry.transport(ref.provider)
                try:
                    return transport.complete(
                        messages,
                        ref,
                        json_mode=json_mode,
                        max_tokens=max_tokens,
                        stage=stage,
                        agent=agent,
                        operation=operation,
                    )
                except Exception as error:
                    reason = classify_retry(error)
                    if reason is None:
                        raise  # 永久错误：原类型立即传播，不降级
                    failures.append((ref, reason))
                    last_error = error
                    if index + 1 < len(candidates):
                        self.usage.record_fallback(agent, operation)
            raise AllModelsFailedError(tuple(failures)) from last_error
        finally:
            self.usage.record_logical_call(agent, operation, (time.monotonic() - start) * 1000)

    def complete_json(
        self,
        messages: Messages,
        *,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        agent: str,
        operation: str,
    ) -> Any:
        text = self.complete(
            messages,
            json_mode=True,
            max_tokens=max_tokens,
            stage=stage,
            agent=agent,
            operation=operation,
        )
        return parse_json_loose(text)
