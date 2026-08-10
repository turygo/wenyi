"""根据配置构建生产 LLM 客户端（AgentRouter）。"""

from __future__ import annotations

from ..config import Config
from .base import LLMClient
from .router import AgentRouter


def build_client(config: Config) -> LLMClient:
    """构建配置指定的 LLM 客户端：按 llm.agents 路由执行 primary/fallback。"""
    return AgentRouter(config)
