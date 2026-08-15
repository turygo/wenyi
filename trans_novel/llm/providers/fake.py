"""测试与离线流程使用的可编程 provider。

- FakeClient：直接注入 handler 的标准测试路径，handler 签名
  (messages, agent, operation, json_mode)。调用记录同时包含 agent 与
  operation 两个键，供离线测试确定性断言。缺省输出（纯文本 ""、JSON "[]"）
  是刻意的 fake 语义，绕过生产空响应校验，空输出计成功，不计 failed_attempts。
- FakeProviderTransport：provider 类型 fake 在 ProviderRegistry 中的共享传输，
  无凭据无网络，输出与 FakeClient 缺省一致；模型 ID 只用于目录一致性，不发送。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from trans_novel.config import LLMConfig, ModelRef
from trans_novel.llm.base import LLMClient, Messages
from trans_novel.llm.usage import UsageTracker

Handler = Callable[[Messages, str, str, bool], str]


class FakeClient(LLMClient):
    """可编程的离线 client。"""

    def __init__(self, handler: Handler | None = None) -> None:
        super().__init__()
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()

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
        record = {
            "messages": messages,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "stage": stage,
            "agent": agent,
            "operation": operation,
        }
        with self._calls_lock:
            self.calls.append(record)
        self.usage.record_attempt(agent=agent, operation=operation)
        start = time.monotonic()
        try:
            if self.handler is not None:
                return self.handler(messages, agent, operation, json_mode)
            return "[]" if json_mode else ""
        except Exception:
            self.usage.record_attempt_failed(agent=agent, operation=operation)
            raise
        finally:
            self.usage.record_logical_call(agent, operation, (time.monotonic() - start) * 1000)


class FakeProviderTransport:
    """provider 类型 fake 的共享传输：无凭据、无网络、无 handler。

    缺省输出与直接 FakeClient 一致（"" / "[]"），是生产空响应规则之外的
    显式测试例外：成功返回，不记 failed_attempts，不产生 token usage。
    """

    def __init__(self, cfg: LLMConfig, usage: UsageTracker) -> None:
        self.provider = cfg.provider
        self.cfg = cfg
        self.usage = usage

    def complete(
        self,
        messages: Messages,
        model_ref: ModelRef,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        stage: str | None = None,
        agent: str,
        operation: str,
    ) -> str:
        self.usage.record_attempt(
            agent=agent, operation=operation, provider=self.provider, model_ref=model_ref
        )
        return "[]" if json_mode else ""
