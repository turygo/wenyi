"""单 Provider 模型传输。

模型选择和思考策略由 AgentRouter 解析成 ModelRef；传输层只负责构造请求、
执行固定重试策略并记录用量。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Protocol, runtime_checkable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from trans_novel.config import LLMConfig, ModelRef
from trans_novel.llm.base import Messages
from trans_novel.llm.retrying import EmptyResponseError, classify_retry
from trans_novel.llm.usage import UsageTracker
from trans_novel.model_profiles import (
    DIALECT_BAILIAN,
    DIALECT_DEEPSEEK,
    DIALECT_OPENAI,
    DIALECT_OPENROUTER,
    ModelCapabilities,
)
from trans_novel.model_profiles import capabilities_for as _capabilities_for

_REASONING_FLOOR_TOKENS = 4096
_REQUEST_TIMEOUT_SECONDS = 600
_MAX_RETRIES = 4


@runtime_checkable
class ProviderTransport(Protocol):
    provider: str
    usage: UsageTracker

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
    ) -> str: ...


def build_request_kwargs(
    capabilities: ModelCapabilities,
    model_ref: ModelRef,
    messages: Messages,
    *,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """按逐模型能力构造请求；路由意图只在模型明确支持时下发。"""
    kwargs: dict[str, Any] = {
        "model": model_ref.model,
        "messages": messages,
        "stream": False,
    }
    reasoning_enabled = (
        model_ref.reasoning_enabled and model_ref.reasoning_effort in capabilities.reasoning_efforts
    )
    dialect = capabilities.request_dialect
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if dialect == DIALECT_DEEPSEEK:
        if reasoning_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = model_ref.reasoning_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif dialect == DIALECT_BAILIAN:
        kwargs["extra_body"] = {"enable_thinking": reasoning_enabled}
        if reasoning_enabled:
            kwargs["reasoning_effort"] = model_ref.reasoning_effort
    elif dialect == DIALECT_OPENAI:
        if reasoning_enabled:
            kwargs["reasoning_effort"] = model_ref.reasoning_effort
    elif dialect == DIALECT_OPENROUTER:
        kwargs["extra_body"] = {
            "reasoning": (
                {"effort": model_ref.reasoning_effort} if reasoning_enabled else {"enabled": False}
            )
        }
    if max_tokens is not None:
        kwargs["max_tokens"] = (
            max(max_tokens, _REASONING_FLOOR_TOKENS) if reasoning_enabled else max_tokens
        )
    return kwargs


class OpenAICompatibleTransport:
    """延迟创建客户端，并使用固定、经过测试的超时与重试策略。"""

    def __init__(
        self,
        cfg: LLMConfig,
        usage: UsageTracker,
        *,
        provider_name: str,
        default_base_url: str | None,
        default_api_key_env: str | None,
        requires_api_key: bool,
    ) -> None:
        self.provider = cfg.provider
        self.cfg = cfg
        self.usage = usage
        self.provider_name = provider_name
        self.base_url = cfg.base_url or default_base_url
        self.api_key_env = cfg.api_key_env or default_api_key_env
        self.requires_api_key = requires_api_key
        if not self.base_url:
            raise ValueError(f"llm.base_url：{provider_name} provider 必须配置服务地址")
        self._client: Any = None
        self._client_lock = threading.Lock()

    def capabilities_for(self, model: str) -> ModelCapabilities:
        return _capabilities_for(self.provider, model)

    def _ensure_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as error:  # pragma: no cover
                    raise RuntimeError("需要 openai SDK：pip install openai") from error
                api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
                if (self.requires_api_key or self.api_key_env) and not api_key:
                    raise RuntimeError(
                        f"未设置环境变量 {self.api_key_env}（{self.provider_name} API key）"
                    )
                self._client = OpenAI(
                    api_key=api_key or "no-key",
                    base_url=self.base_url,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
        return self._client

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
        kwargs = build_request_kwargs(
            self.capabilities_for(model_ref.model),
            model_ref,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        client = self._ensure_client()

        @retry(
            stop=stop_after_attempt(_MAX_RETRIES + 1),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception(lambda error: classify_retry(error) is not None),
            reraise=True,
        )
        def call() -> str:
            self.usage.record_attempt(
                agent=agent,
                operation=operation,
                provider=self.provider,
                model_ref=model_ref,
            )
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                self.usage.record_attempt_failed(
                    agent=agent,
                    operation=operation,
                    provider=self.provider,
                    model_ref=model_ref,
                )
                raise
            self.usage.record(
                agent=agent,
                operation=operation,
                provider=self.provider,
                model_ref=model_ref,
                usage=getattr(response, "usage", None),
                stage=stage,
            )
            content = getattr(response.choices[0].message, "content", None)
            if not isinstance(content, str) or not content.strip():
                self.usage.record_attempt_failed(
                    agent=agent,
                    operation=operation,
                    provider=self.provider,
                    model_ref=model_ref,
                )
                raise EmptyResponseError("provider returned empty response content")
            return content

        return call()
