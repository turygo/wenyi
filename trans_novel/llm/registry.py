"""Provider-keyed lazy transport registry."""

from __future__ import annotations

import threading
from typing import Any

from trans_novel.config import LLMConfig
from trans_novel.llm.generation import GenerationOptions
from trans_novel.llm.providers.fake import FakeProviderTransport
from trans_novel.llm.providers.transport import OpenAICompatibleTransport, ProviderTransport
from trans_novel.llm.telemetry import CallTelemetrySink
from trans_novel.llm.usage import UsageTracker
from trans_novel.model_profiles import ModelCapabilities, capabilities_for

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
    """Lazily create one transport per provider; tests may inject keyed stubs."""

    def __init__(
        self,
        cfg: LLMConfig,
        usage: UsageTracker,
        *,
        transports: dict[str, ProviderTransport] | None = None,
        generation_options: GenerationOptions | None = None,
        telemetry_sink: CallTelemetrySink | None = None,
    ) -> None:
        self.cfg = cfg
        self.usage = usage
        self._generation_options = generation_options
        self._telemetry_sink = telemetry_sink
        self._injected = dict(transports or {})
        for transport in self._injected.values():
            transport.usage = usage
        self._built: dict[str, ProviderTransport] = {}
        self._lock = threading.Lock()

    @property
    def generation_options(self) -> GenerationOptions | None:
        return self._generation_options

    def set_generation_options(self, options: GenerationOptions | None) -> None:
        """Set request controls before materialization, or reject divergence."""
        with self._lock:
            if options == self._generation_options:
                return
            if self._built:
                raise ValueError("cannot change generation options after transport materialization")
            self._generation_options = options

    def transport(self, provider: str) -> ProviderTransport:
        with self._lock:
            injected = self._injected.get(provider)
            if injected is not None:
                return injected
            transport = self._built.get(provider)
            if transport is None:
                transport = self._build(provider)
                self._built[provider] = transport
            return transport

    @property
    def telemetry_sink(self) -> CallTelemetrySink | None:
        return self._telemetry_sink

    def set_telemetry_sink(self, sink: CallTelemetrySink | None) -> None:
        with self._lock:
            if sink is self._telemetry_sink:
                return
            if self._built:
                raise ValueError("cannot change telemetry sink after transport materialization")
            self._telemetry_sink = sink

    def capabilities_for(self, provider: str, model: str) -> ModelCapabilities:
        transport = self._injected.get(provider)
        if transport is not None:
            capabilities_method = getattr(transport, "capabilities_for", None)
            if callable(capabilities_method):
                return capabilities_method(model)
        return capabilities_for(provider, model)

    def _build(self, provider: str) -> ProviderTransport:
        if provider == "fake":
            return FakeProviderTransport(
                self.cfg,
                self.usage,
                provider=provider,
                generation_options=self.generation_options,
                telemetry_sink=self._telemetry_sink,
            )
        spec = _PROVIDER_SPECS[provider]
        return OpenAICompatibleTransport(
            self.cfg,
            self.usage,
            provider=provider,
            provider_name=spec["provider_name"],
            default_base_url=spec["default_base_url"],
            default_api_key_env=spec["default_api_key_env"],
            requires_api_key=spec["requires_api_key"],
            generation_options=self.generation_options,
            telemetry_sink=self._telemetry_sink,
        )
