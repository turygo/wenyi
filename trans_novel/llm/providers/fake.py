"""测试和离线流程使用的可编程 provider。"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from ..base import LLMClient, Messages


class FakeClient(LLMClient):
    """可编程的离线 client。"""

    def __init__(
        self,
        handler: Optional[Callable[[Messages, str, bool], str]] = None,
    ) -> None:
        super().__init__()
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self._calls_lock = threading.Lock()

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
