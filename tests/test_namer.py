"""CastNamer 并行定名的回归（离线，零网络）：分组并行输出与串行一致（按输入组序
合并）、进度按完成组数回调、单组异常整体冒泡（不得被兜成空定名）。"""

from __future__ import annotations

import json
import re
import time
import unittest

from trans_novel.agents.namer import CastNamer
from trans_novel.config import Config
from trans_novel.glossary.miner import Candidate
from trans_novel.llm.base import FakeClient

# 每个候选带一段超预算长 context → _group 里每候选单独成一组，组数 = 候选数。
_LONG = "x" * 6001


def _cfg() -> Config:
    return Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {
                "provider": "fake",
                "tiers": {"strong": {"model": "p"}, "cheap": {"model": "f"}},
            },
        }
    )


def _candidates(n: int) -> list[Candidate]:
    return [
        Candidate(surface=f"SURF{i}", count=3, chapters=[1], contexts=[_LONG]) for i in range(n)
    ]


def _surface_of(user: str) -> str:
    return re.search(r"\[0\] (\S+?)（", user).group(1)


class TestCastNamerConcurrency(unittest.TestCase):
    N = 5

    def _namer(self, handler) -> CastNamer:
        return CastNamer(FakeClient(handler=handler), _cfg())

    @staticmethod
    def _one_term_per_group(messages, tier, json_mode):
        """每组回一个 source=该组候选 surface、target=译<i> 的术语。"""
        i = int(_surface_of(messages[1]["content"])[4:])
        return json.dumps({"terms": [{"source": f"SURF{i}", "target": f"译{i}", "type": "术语"}]})

    def test_parallel_output_ordered_by_input_group(self):
        """乱序完成下仍按输入组序合并：让先提交的组睡得更久（完成顺序反转）。"""

        def handler(messages, tier, json_mode):
            i = int(_surface_of(messages[1]["content"])[4:])
            time.sleep((self.N - i) * 0.02)  # SURF0 最后完成
            return self._one_term_per_group(messages, tier, json_mode)

        out = self._namer(handler).name_terms(
            _candidates(self.N), "brief", ["d1"], concurrency=self.N
        )
        self.assertEqual([t.target for t in out], [f"译{i}" for i in range(self.N)])

    def test_concurrent_identical_to_serial(self):
        serial = self._namer(self._one_term_per_group).name_terms(
            _candidates(self.N), "brief", ["d1"], concurrency=1
        )
        parallel = self._namer(self._one_term_per_group).name_terms(
            _candidates(self.N), "brief", ["d1"], concurrency=self.N
        )

        def key(ts):
            return [(t.source, t.target) for t in ts]

        self.assertEqual(key(parallel), key(serial))

    def test_on_progress_counts_group_completions(self):
        calls: list[tuple[int, int]] = []
        self._namer(self._one_term_per_group).name_terms(
            _candidates(self.N),
            "brief",
            ["d1"],
            concurrency=2,
            on_progress=lambda i, n: calls.append((i, n)),
        )
        # 起点 (0,N) 一次，其后每完成一组回调一次，总数按输入组序无关的完成计数递增。
        self.assertEqual([i for i, _ in calls], list(range(self.N + 1)))
        self.assertTrue(all(n == self.N for _, n in calls))

    def test_single_group_failure_propagates(self):
        """一组抛异常 → name_terms 整体冒泡（交 orchestrator 放弃落 term_mining_done）。"""

        def handler(messages, tier, json_mode):
            if _surface_of(messages[1]["content"]) == "SURF2":
                raise ValueError("strong tier boom")
            return self._one_term_per_group(messages, tier, json_mode)

        with self.assertRaises(ValueError):
            self._namer(handler).name_terms(_candidates(self.N), "brief", ["d1"], concurrency=2)

    def test_empty_candidates_short_circuits(self):
        calls: list[tuple[int, int]] = []
        out = self._namer(self._one_term_per_group).name_terms(
            [], "brief", ["d1"], concurrency=4, on_progress=lambda i, n: calls.append((i, n))
        )
        self.assertEqual(out, [])
        self.assertEqual(calls, [])  # 无候选：不回调、不开线程池


if __name__ == "__main__":
    unittest.main()
