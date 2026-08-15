"""Workflow 具体节点实现包。

按独立输入/检查点/失败语义划分（不是每个 Agent 一个节点）：
- prepare.py:   prepare（解析/语言/身份/暂存）与 analyze（风格分析+术语播种）；
- prescan.py:   digest / mine_terms / name_terms / book_synopsis（全书预扫）;
- translate.py: translate（批翻译+lint+修复+检查点）与 polish（章末排干润色）;
- quality.py:   naturalize / review / backtranslate（章级质量环节）;
- finish.py:    titles / consistency_qa / report / assemble（书级收尾）。

组合根（bootstrap.py）是唯一 import/构造这些具体节点的生产位置；runner
只通过契约 NodeFactory 拿到节点实例。
"""

from __future__ import annotations

import threading
from typing import Any

from trans_novel.agents.analyzer import Analyzer
from trans_novel.agents.consistency import ConsistencyChecker
from trans_novel.agents.namer import CastNamer
from trans_novel.agents.naturalizer import Naturalizer
from trans_novel.agents.polisher import Polisher
from trans_novel.agents.reviewer import BackTranslator, Reviewer
from trans_novel.agents.synopsis import Synopsizer
from trans_novel.agents.translator import Translator
from trans_novel.config import Config
from trans_novel.glossary.extractor import GlossaryExtractor
from trans_novel.glossary.store import GlossaryStore
from trans_novel.llm.base import LLMClient
from trans_novel.pipeline.context import RollingContext

__all__ = ["AgentBundle", "RunShared"]


class AgentBundle:
    """一组按已解析语言对构造的 Agent（语言一经解析即不可变）。"""

    def __init__(self, *, client: LLMClient, config: Config, src: str, tgt: str):
        self.client = client
        self.config = config
        self.src = src
        self.tgt = tgt
        self.analyzer = Analyzer(client, config)
        self.synopsizer = Synopsizer(client, config)
        self.translator = Translator(client, config)
        self.reviewer = Reviewer(client, config)
        self.backtrans = BackTranslator(client, config)
        self.polisher = Polisher(client, config)
        self.extractor = GlossaryExtractor(client, config)
        self.namer = CastNamer(client, config)
        self.naturalizer = Naturalizer(client, config)
        self.consistency = ConsistencyChecker(client, config)
        for agent in (
            self.analyzer,
            self.synopsizer,
            self.translator,
            self.reviewer,
            self.backtrans,
            self.polisher,
            self.extractor,
            self.namer,
            self.naturalizer,
            self.consistency,
        ):
            agent.src = src
            agent.tgt = tgt


class RunShared:
    """一轮运行的共享**协调**状态：解析后的文档、惰性解析的 Agent、术语库连接、
    滚动上下文、进度计数、润色 future 与并行层锁。

    具体节点不再通过本类型获取 Agent/语言/术语库等具体依赖（那些走构造注入）；
    runner 只按不透明对象传递本类型，节点仅消费协调状态（计数/锁/在途 future）。
    """

    def __init__(self, *, store, config: Config, doc, agent_builder):
        self.store = store
        self.config = config
        self.doc = doc
        self._agent_builder = agent_builder
        self.resolved_source_lang: str | None = None  # prepare 解析后写入
        self._agents: AgentBundle | None = None
        self._context: RollingContext | None = None
        self._glossary: GlossaryStore | None = None
        self._style_brief: str | None = None
        self.segments_done = 0
        self.segments_total = 0
        # 并行层（digest ∥ mine_terms）节点在 worker 线程写 RunStore：锁串行化
        # 读改写，避免并发 save_state 互相覆盖（仓库既有约定：RunStore 读写主线程，
        # 并行场景退化为共享锁保护）。
        self.store_lock = threading.Lock()
        # translate 提交的润色 future（批起点 → Future），polish 节点排干时取用。
        self.polish_futures: dict[tuple[int, int], Any] = {}

    @property
    def agents(self) -> AgentBundle:
        if self._agents is None:
            src = self.resolved_source_lang or self.config.source_lang
            tgt = self.config.target_lang
            if self.store.exists():
                state = self.store.load_state()
                src = (
                    self.resolved_source_lang
                    or state.identity.source_lang
                    or state.source_lang
                    or src
                )
                tgt = state.identity.target_lang or state.target_lang or tgt
            self._agents = self._agent_builder(src, tgt)
        return self._agents

    def rolling_context(self) -> RollingContext:
        if self._context is None:
            self._context = RollingContext.from_dict(
                self.store.load_context() or {},
                min_recent_keep=max(40, self.config.pipeline.rolling_context_segments),
            )
        return self._context

    def glossary(self) -> GlossaryStore:
        if self._glossary is None:
            self._glossary = GlossaryStore(self.store.glossary_path)
        return self._glossary

    def style_brief(self) -> str:
        if self._style_brief is None:
            self._style_brief = self.agents.analyzer.style_brief(self.store.load_analysis() or {})
        return self._style_brief

    def close(self) -> None:
        if self._glossary is not None:
            self._glossary.close()
            self._glossary = None
