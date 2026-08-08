"""通过 DeepSeek 原生 OpenAI 兼容接口调用模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...config import LLMConfig
from ..base import Messages
from ._openai_compatible import (
    OpenAICompatibleBaseClient,
    ResolvedTier,
    base_request_kwargs,
    deep_merge,
    resolve_provider_tiers,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekTierOptions(BaseModel):
    """DeepSeek 档位的专属请求选项。"""

    model_config = ConfigDict(extra="forbid")

    thinking: bool = True
    reasoning_effort: str = "high"
    extra_body: dict[str, Any] = Field(default_factory=dict)


def _default_tiers() -> dict[str, ResolvedTier[DeepSeekTierOptions]]:
    return {
        "strong": ResolvedTier("deepseek-v4-pro", DeepSeekTierOptions()),
        "cheap": ResolvedTier("deepseek-v4-flash", DeepSeekTierOptions()),
        "fast": ResolvedTier("deepseek-v4-flash", DeepSeekTierOptions(thinking=False)),
    }


def build_request_kwargs(
    tier_config: ResolvedTier[DeepSeekTierOptions],
    messages: Messages,
    *,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    kwargs = base_request_kwargs(tier_config.model, messages, json_mode=json_mode)
    if tier_config.options.thinking:
        extra_body: dict[str, Any] = {"thinking": {"type": "enabled"}}
        kwargs["reasoning_effort"] = tier_config.options.reasoning_effort
    else:
        # DeepSeek defaults to enabled thinking, so disabling must be explicit.
        extra_body = {"thinking": {"type": "disabled"}}
    if tier_config.options.extra_body:
        extra_body = deep_merge(extra_body, tier_config.options.extra_body)
    kwargs["extra_body"] = extra_body
    if max_tokens:
        kwargs["max_tokens"] = max(max_tokens, 4096) if tier_config.options.thinking else max_tokens
    return kwargs


class DeepSeekClient(OpenAICompatibleBaseClient[DeepSeekTierOptions]):
    def __init__(self, cfg: LLMConfig):
        tiers = resolve_provider_tiers(
            cfg.tiers,
            options_type=DeepSeekTierOptions,
            defaults=_default_tiers() if not cfg.tiers else None,
        )
        super().__init__(
            cfg,
            provider_name="DeepSeek",
            default_base_url=DEFAULT_BASE_URL,
            default_api_key_env=DEFAULT_API_KEY_ENV,
            tiers=tiers,
            requires_api_key=True,
        )

    def _build_request_kwargs(
        self,
        tier_config: ResolvedTier[DeepSeekTierOptions],
        messages: Messages,
        *,
        json_mode: bool,
        max_tokens: Optional[int],
    ) -> dict[str, Any]:
        return build_request_kwargs(
            tier_config,
            messages,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
