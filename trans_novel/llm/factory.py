"""根据配置构建生产 LLM 客户端（AgentRouter）。"""

from __future__ import annotations

from trans_novel.config import Config
from trans_novel.llm.base import LLMClient
from trans_novel.llm.router import AgentRouter


def build_client(config: Config) -> LLMClient:
    """构建按内置 Agent 映射 primary / fast 模型的客户端。"""
    return AgentRouter(config)
