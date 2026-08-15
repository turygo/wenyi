"""LLM 调用层的稳定公共接口。"""

from trans_novel.config import ModelRef
from trans_novel.llm.base import LLMClient, Messages
from trans_novel.llm.errors import AllModelsFailedError, UnknownAgentError
from trans_novel.llm.factory import build_client
from trans_novel.llm.json_parser import parse_json_loose
from trans_novel.llm.providers.fake import FakeClient
from trans_novel.llm.router import AgentRouter

__all__ = [
    "AgentRouter",
    "AllModelsFailedError",
    "FakeClient",
    "LLMClient",
    "Messages",
    "ModelRef",
    "UnknownAgentError",
    "build_client",
    "parse_json_loose",
]
