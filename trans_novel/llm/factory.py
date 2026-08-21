"""根据配置构建生产 LLM 客户端（AgentRouter）。"""

from __future__ import annotations

from trans_novel.config import Config
from trans_novel.llm.base import LLMClient
from trans_novel.llm.generation import GenerationOptions
from trans_novel.llm.router import AgentRouter
from trans_novel.llm.telemetry import CallTelemetrySink


def build_client(
    config: Config,
    *,
    generation_options: GenerationOptions | None = None,
    telemetry_sink: CallTelemetrySink | None = None,
) -> LLMClient:
    """构建按内置 Agent 映射 primary / fast 模型的客户端。"""
    return AgentRouter(
        config,
        generation_options=generation_options,
        telemetry_sink=telemetry_sink,
    )
