"""根据配置构建生产 LLM 客户端（AgentRouter）。"""

from __future__ import annotations

from ..config import Config
from .base import LLMClient
from .router import AgentRouter


def build_client(config: Config) -> LLMClient:
    """构建按内置 Agent 映射 primary / fast 模型的客户端。"""
    return AgentRouter(config)
