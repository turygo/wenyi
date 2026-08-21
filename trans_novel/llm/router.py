"""将内部 Agent 映射到 primary / editor / fast 三个用户模型角色。"""

from __future__ import annotations

import time
from typing import Any

from trans_novel.config import PRODUCTION_AGENT_IDS, Config, ModelRef
from trans_novel.llm.base import LLMClient, Messages
from trans_novel.llm.errors import AllModelsFailedError, UnknownAgentError
from trans_novel.llm.generation import GenerationOptions
from trans_novel.llm.json_parser import parse_json_loose
from trans_novel.llm.providers.transport import validate_generation_options
from trans_novel.llm.registry import ProviderRegistry
from trans_novel.llm.retrying import classify_retry
from trans_novel.llm.telemetry import CallTelemetrySink
from trans_novel.model_profiles import parse_model_selection

_PRIMARY_AGENTS = frozenset({"translator", "analyst"})
_EDITOR_AGENTS = frozenset({"editor"})
_FAST_AGENTS = frozenset({"reviewer", "preparer", "light-translator"})


class AgentRouter(LLMClient):
    def __init__(
        self,
        config: Config,
        *,
        registry: ProviderRegistry | None = None,
        transports: dict | None = None,
        generation_options: GenerationOptions | None = None,
        telemetry_sink: CallTelemetrySink | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._primary_selection = parse_model_selection(config.llm.models.primary)
        self._editor_selection = parse_model_selection(config.llm.models.editor)
        self._fast_selection = parse_model_selection(config.llm.models.fast)
        if registry is not None:
            self._registry = registry
            if generation_options is not None:
                registry.set_generation_options(generation_options)
            if telemetry_sink is not None:
                registry.set_telemetry_sink(telemetry_sink)
            self.generation_options = registry.generation_options
            self.usage = registry.usage
        else:
            self.generation_options = generation_options
            self._registry = ProviderRegistry(
                config.llm,
                self.usage,
                transports=transports,
                generation_options=generation_options,
                telemetry_sink=telemetry_sink,
            )

    def _model(self, agent: str | None) -> ModelRef:
        if not agent or agent not in PRODUCTION_AGENT_IDS:
            raise UnknownAgentError(
                f"未知或缺失 Agent：{agent!r}（生产 LLM 调用必须提供内置 Agent 标识）"
            )
        if agent in _PRIMARY_AGENTS:
            selection = self._primary_selection
            thinking = selection.thinking or "high"
        elif agent in _EDITOR_AGENTS:
            selection = self._editor_selection
            thinking = selection.thinking or "high"
        elif agent in _FAST_AGENTS:
            selection = self._fast_selection
            thinking = selection.thinking or "off"
        else:  # pragma: no cover - PRODUCTION_AGENT_IDS 与映射必须同步
            raise UnknownAgentError(f"Agent 未绑定模型角色：{agent!r}")
        if thinking == "off":
            reasoning_enabled = False
            reasoning_effort = "high"
        else:
            reasoning_enabled = True
            reasoning_effort = thinking
        return ModelRef(
            provider=self.config.llm.provider,
            model=selection.model,
            reasoning_enabled=reasoning_enabled,
            reasoning_effort=reasoning_effort,
        )

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
        model_ref = self._model(agent)
        validate_generation_options(
            self._registry.capabilities_for(model_ref.model),
            model_ref,
            self.generation_options,
        )
        start = time.monotonic()
        try:
            try:
                return self._registry.transport().complete(
                    messages,
                    model_ref,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    stage=stage,
                    agent=agent,
                    operation=operation,
                )
            except Exception as error:
                reason = classify_retry(error)
                if reason is None:
                    raise
                raise AllModelsFailedError(((model_ref, reason),)) from error
        finally:
            self.usage.record_logical_call(agent, operation, (time.monotonic() - start) * 1000)

    def complete_json(
        self,
        messages: Messages,
        *,
        max_tokens: int | None = None,
        stage: str | None = None,
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
