"""ProviderRegistry：按 Provider 别名延迟创建底层传输，并持有模型目录。"""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..config import LLMConfig
from .providers.fake import FakeProviderTransport
from .providers.transport import OpenAICompatibleTransport, ProviderTransport
from .usage import UsageTracker

# provider 类型 → OpenAI-compatible 传输规格（方言 / 默认端点 / 默认密钥环境变量 / 是否强制鉴权）。
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "dialect": "deepseek",
        "provider_name": "DeepSeek",
        "default_base_url": "https://api.deepseek.com",
        "default_api_key_env": "DEEPSEEK_API_KEY",
        "requires_api_key": True,
    },
    "openai": {
        "dialect": "openai",
        "provider_name": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "requires_api_key": True,
    },
    "openrouter": {
        "dialect": "openrouter",
        "provider_name": "OpenRouter",
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_api_key_env": "OPENROUTER_API_KEY",
        "requires_api_key": True,
    },
    "openai-compatible": {
        "dialect": "generic",
        "provider_name": "OpenAI-compatible",
        "default_base_url": None,
        "default_api_key_env": None,
        "requires_api_key": False,
    },
    "custom": {
        "dialect": "generic",
        "provider_name": "custom",
        "default_base_url": None,
        "default_api_key_env": None,
        "requires_api_key": False,
    },
    "ollama": {
        "dialect": "generic",
        "provider_name": "Ollama",
        "default_base_url": "http://localhost:11434/v1",
        "default_api_key_env": None,
        "requires_api_key": False,
    },
    "vllm": {
        "dialect": "generic",
        "provider_name": "vLLM",
        "default_base_url": "http://localhost:8000/v1",
        "default_api_key_env": None,
        "requires_api_key": False,
    },
}


class ProviderRegistry:
    """每个已配置的 Provider 别名对应一个延迟创建的传输实例；注入的传输（测试 stub）优先。"""

    def __init__(
        self,
        cfg: LLMConfig,
        usage: UsageTracker,
        *,
        transports: Optional[dict[str, ProviderTransport]] = None,
    ) -> None:
        self.cfg = cfg
        self.usage = usage
        self._injected: dict[str, ProviderTransport] = {}
        for alias, transport in (transports or {}).items():
            transport.usage = usage  # 测试 stub 与共享记账器对齐
            self._injected[alias] = transport
        self._built: dict[str, ProviderTransport] = {}
        self._lock = threading.Lock()

    def transport(self, alias: str) -> ProviderTransport:
        with self._lock:
            if alias in self._injected:
                return self._injected[alias]
            transport = self._built.get(alias)
            if transport is None:
                transport = self._build(alias)
                self._built[alias] = transport
            return transport

    def _build(self, alias: str) -> ProviderTransport:
        try:
            provider = self.cfg.providers[alias]
        except KeyError:
            raise ValueError(f"未知 provider 别名：{alias!r}（配置中不存在）") from None
        if provider.type == "fake":
            return FakeProviderTransport(alias, provider, self.usage)
        spec = _PROVIDER_SPECS[provider.type]
        return OpenAICompatibleTransport(
            alias,
            provider,
            self.usage,
            dialect=spec["dialect"],
            provider_name=spec["provider_name"],
            default_base_url=spec["default_base_url"],
            default_api_key_env=spec["default_api_key_env"],
            requires_api_key=spec["requires_api_key"],
        )
