"""OpenAI Chat Completions 兼容 provider 的共享物理传输与请求方言。

provider 传输只做一件事：对已解析的 ProviderModelConfig 构造请求、执行
`max_retries + 1` 次物理尝试并记账。路由/降级（primary/fallback）在
AgentRouter 层，物理尝试之外不做任何模型选择。
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, Protocol, runtime_checkable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ...config import ModelRef, ProviderConfig, ProviderModelConfig
from ..base import Messages
from ..retrying import EmptyResponseError, classify_retry
from ..usage import UsageTracker

# 请求方言：deepseek/openai/openrouter 有专属 reasoning 激活字段；
# generic 覆盖 openai-compatible/custom/ollama/vllm，只有 4096 token 安全下限。
DIALECT_DEEPSEEK = "deepseek"
DIALECT_OPENAI = "openai"
DIALECT_OPENROUTER = "openrouter"
DIALECT_GENERIC = "generic"

_REASONING_FLOOR_TOKENS = 4096


@runtime_checkable
class ProviderTransport(Protocol):
    """AgentRouter 消费的物理传输接口（测试可用确定性 stub 注入）。"""

    alias: str
    usage: UsageTracker

    def complete(
        self,
        messages: Messages,
        model_ref: ModelRef,
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        agent: str,
        operation: str,
    ) -> str:
        """执行一次物理 complete（含重试）。不路由、不降级。"""
        ...


def deep_merge_override(generated: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """将 request_overrides 递归合并到生成的请求参数中。

    字典与字典递归合并；其他值（包括列表与 None）整体替换原值；不修改入参。
    """
    result = dict(generated)
    for key, value in overrides.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = deep_merge_override(current, value)
        else:
            result[key] = value
    return result


def build_request_kwargs(
    dialect: str,
    model_cfg: ProviderModelConfig,
    messages: Messages,
    *,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """构造 provider 专属请求 kwargs；request_overrides 最后递归覆盖。

    model/messages/stream 由系统生成且不可覆盖（配置加载时已拒绝保留键）。
    """
    reasoning = model_cfg.reasoning
    kwargs: dict[str, Any] = {
        "model": model_cfg.id,
        "messages": messages,
        "stream": False,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if dialect == DIALECT_DEEPSEEK:
        if reasoning.enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = reasoning.effort
        else:
            # DeepSeek 默认开思考，关闭必须显式下发。
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif dialect == DIALECT_OPENAI:
        if reasoning.enabled:
            kwargs["reasoning_effort"] = reasoning.effort
    elif dialect == DIALECT_OPENROUTER:
        kwargs["extra_body"] = {
            "reasoning": ({"effort": reasoning.effort} if reasoning.enabled else {"enabled": False})
        }
    # generic：标准 reasoning 只控制 max_tokens 安全下限；端点专属激活字段走 request_overrides。
    if max_tokens is not None:
        if reasoning.enabled:
            kwargs["max_tokens"] = max(max_tokens, _REASONING_FLOOR_TOKENS)
        else:
            kwargs["max_tokens"] = max_tokens
    if model_cfg.request_overrides:
        kwargs = deep_merge_override(kwargs, model_cfg.request_overrides)
    return kwargs


class OpenAICompatibleTransport:
    """单个 Provider 别名对应的底层传输：延迟初始化 OpenAI SDK 客户端，并负责重试和用量记账。"""

    def __init__(
        self,
        alias: str,
        cfg: ProviderConfig,
        usage: UsageTracker,
        *,
        dialect: str,
        provider_name: str,
        default_base_url: Optional[str],
        default_api_key_env: Optional[str],
        requires_api_key: bool,
    ) -> None:
        self.alias = alias
        self.cfg = cfg
        self.usage = usage
        self.dialect = dialect
        self.provider_name = provider_name
        self.base_url = cfg.base_url or default_base_url
        self.api_key_env = cfg.api_key_env or default_api_key_env
        self.requires_api_key = requires_api_key
        if not self.base_url:
            raise ValueError(
                f"llm.providers.{alias}.base_url：{provider_name} provider 必须配置 base_url"
            )
        self._client: Any = None
        self._client_lock = threading.Lock()

    def _ensure_client(self) -> Any:
        """凭据/配置校验（永久错误，发生在任何物理尝试之前）。"""
        with self._client_lock:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as error:  # pragma: no cover
                    raise RuntimeError(
                        "需要 openai SDK：pip install openai"
                        "（或把 provider 类型设为 fake 做离线测试）"
                    ) from error
                api_key = os.environ.get(self.api_key_env) if self.api_key_env else None
                if (self.requires_api_key or self.api_key_env) and not api_key:
                    raise RuntimeError(
                        f"未设置环境变量 {self.api_key_env}（{self.provider_name} API key）"
                    )
                self._client = OpenAI(
                    api_key=api_key or "no-key",
                    base_url=self.base_url,
                    timeout=self.cfg.timeout,
                )
        return self._client

    def complete(
        self,
        messages: Messages,
        model_ref: ModelRef,
        *,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        stage: Optional[str] = None,
        agent: str,
        operation: str,
    ) -> str:
        model_cfg = self.cfg.models[model_ref.model]
        kwargs = build_request_kwargs(
            self.dialect,
            model_cfg,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        client = self._ensure_client()

        @retry(
            stop=stop_after_attempt(self.cfg.max_retries + 1),
            wait=wait_exponential(multiplier=1, max=30),
            retry=retry_if_exception(lambda error: classify_retry(error) is not None),
            reraise=True,
        )
        def call() -> str:
            self.usage.record_attempt(
                agent=agent, operation=operation, provider=self.alias, model_ref=model_ref
            )
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                self.usage.record_attempt_failed(
                    agent=agent, operation=operation, provider=self.alias, model_ref=model_ref
                )
                raise
            self.usage.record(
                agent=agent,
                operation=operation,
                provider=self.alias,
                model_ref=model_ref,
                usage=getattr(response, "usage", None),
                stage=stage,
            )
            content = getattr(response.choices[0].message, "content", None)
            if not isinstance(content, str) or not content.strip():
                self.usage.record_attempt_failed(
                    agent=agent, operation=operation, provider=self.alias, model_ref=model_ref
                )
                raise EmptyResponseError("provider returned empty response content")
            return content

        return call()
