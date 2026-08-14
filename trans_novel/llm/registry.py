"""单 Provider 传输注册表。"""

from __future__ import annotations

import threading
from typing import Any, Optional

from ..config import LLMConfig
from .providers.fake import FakeProviderTransport
from .providers.transport import OpenAICompatibleTransport, ProviderTransport
from .usage import UsageTracker

_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "provider_name": "DeepSeek",
        "default_base_url": "https://api.deepseek.com",
        "default_api_key_env": "DEEPSEEK_API_KEY",
        "requires_api_key": True,
    },
    "opencode-go": {
        "provider_name": "OpenCode Go",
        "default_base_url": "https://opencode.ai/zen/go/v1",
        "default_api_key_env": "OPENCODE_API_KEY",
        "requires_api_key": True,
    },
    "bailian": {
        "provider_name": "Alibaba Cloud Model Studio",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_api_key_env": "BAILIAN_API_KEY",
        "requires_api_key": True,
    },
    "openai": {
        "provider_name": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "requires_api_key": True,
    },
    "openrouter": {
        "provider_name": "OpenRouter",
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_api_key_env": "OPENROUTER_API_KEY",
        "requires_api_key": True,
    },
    "openai-compatible": {
        "provider_name": "OpenAI-compatible",
        "default_base_url": None,
        "default_api_key_env": None,
        "requires_api_key": False,
    },
    "ollama": {
        "provider_name": "Ollama",
        "default_base_url": "http://localhost:11434/v1",
        "default_api_key_env": None,
        "requires_api_key": False,
    },
    "vllm": {
        "provider_name": "vLLM",
        "default_base_url": "http://localhost:8000/v1",
        "default_api_key_env": None,
        "requires_api_key": False,
    },
}


class ProviderRegistry:
    """为当前唯一 Provider 延迟创建传输；测试可按 Provider 名注入 stub。"""

    def __init__(
        self,
        cfg: LLMConfig,
        usage: UsageTracker,
        *,
        transports: Optional[dict[str, ProviderTransport]] = None,
    ) -> None:
        self.cfg = cfg
        self.usage = usage
        self._injected = dict(transports or {})
        for transport in self._injected.values():
            transport.usage = usage
        self._built: ProviderTransport | None = None
        self._lock = threading.Lock()

    def transport(self) -> ProviderTransport:
        with self._lock:
            injected = self._injected.get(self.cfg.provider)
            if injected is not None:
                return injected
            if self._built is None:
                self._built = self._build()
            return self._built

    def _build(self) -> ProviderTransport:
        if self.cfg.provider == "fake":
            return FakeProviderTransport(self.cfg, self.usage)
        spec = _PROVIDER_SPECS[self.cfg.provider]
        return OpenAICompatibleTransport(
            self.cfg,
            self.usage,
            provider_name=spec["provider_name"],
            default_base_url=spec["default_base_url"],
            default_api_key_env=spec["default_api_key_env"],
            requires_api_key=spec["requires_api_key"],
        )
