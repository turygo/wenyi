"""LLM 用量统计契约测试（离线，不发网络请求）。"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from tests.fake_llm import routing_handler
from tests.sample_data import write_sample_txt
from trans_novel.config import Config, LLMConfig, TierConfig
from trans_novel.llm.base import (
    DeepSeekClient,
    FakeClient,
    UsageTracker,
    merge_usage_summaries,
    usage_delta,
)
from trans_novel.pipeline.orchestrator import Orchestrator
from trans_novel.pipeline.runstore import RunStore


def _make_usage(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
) -> Any:
    """构造普通 class 实例作为 usage（非 dict）。"""
    u = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
    )
    if total_tokens is not None:
        u.total_tokens = total_tokens
    return u


def _make_response(content: str, usage: Any) -> Any:
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice], usage=usage)


class _CompletionsStub:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[dict[str, Any]] = []  # 记录请求 kwargs，供契约断言

    def create(self, **kwargs: Any) -> Any:
        if self._idx >= len(self._responses):
            raise AssertionError("stub 响应已耗尽")
        self.calls.append(kwargs)
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _ChatStub:
    def __init__(self, responses: list[Any]) -> None:
        self.completions = _CompletionsStub(responses)


class _ClientStub:
    """支持 stub.chat.completions.create(**kwargs) 的最小客户端。"""

    def __init__(self, responses: list[Any]) -> None:
        self.chat = _ChatStub(responses)


def _minimal_deepseek_cfg() -> LLMConfig:
    return LLMConfig(
        provider="deepseek",
        base_url="x",
        api_key_env="X",
        timeout=1,
        max_retries=0,
        tiers={
            "strong": TierConfig(model="m1"),
            "cheap": TierConfig(model="m2"),
        },
    )


class TestThinkingFlagWiring(unittest.TestCase):
    """thinking 开关必须显式下发：API 缺省是开思考，漏发 disabled 等于没关。"""

    def test_thinking_tiers_send_explicit_enable_or_disable(self):
        cfg = _minimal_deepseek_cfg()
        cfg.tiers["strong"] = TierConfig(model="m1", thinking=True, reasoning_effort="high")
        cfg.tiers["fast"] = TierConfig(model="m2", thinking=False)
        c = DeepSeekClient(cfg)
        stub = _ClientStub([_make_response("a", None), _make_response("b", None)])
        c._client = stub
        msgs = [{"role": "user", "content": "x"}]
        c.complete(msgs, tier="strong")
        c.complete(msgs, tier="fast")
        on, off = stub.chat.completions.calls
        self.assertEqual(on["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertEqual(on["reasoning_effort"], "high")
        self.assertEqual(off["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", off)


class TestDeepSeekUsageByTier(unittest.TestCase):
    def test_records_usage_and_splits_by_tier(self):
        cfg = _minimal_deepseek_cfg()
        c = DeepSeekClient(cfg)
        responses = [
            _make_response(
                "strong-out",
                _make_usage(
                    prompt_tokens=1000,
                    completion_tokens=200,
                    total_tokens=1200,
                    prompt_cache_hit_tokens=800,
                    prompt_cache_miss_tokens=200,
                ),
            ),
            _make_response(
                "cheap-out",
                _make_usage(
                    prompt_tokens=500,
                    completion_tokens=100,
                    total_tokens=600,
                    prompt_cache_hit_tokens=100,
                    prompt_cache_miss_tokens=400,
                ),
            ),
        ]
        msgs = [{"role": "user", "content": "hi"}]
        with patch.object(c, "_ensure_client", return_value=_ClientStub(responses)):
            self.assertEqual(c.complete(msgs, tier="strong", stage="Translator"), "strong-out")
            self.assertEqual(c.complete(msgs, tier="cheap"), "cheap-out")

        summary = c.usage_summary()
        totals = summary["totals"]
        self.assertEqual(totals["prompt_tokens"], 1500)
        self.assertEqual(totals["completion_tokens"], 300)
        self.assertEqual(totals["total_tokens"], 1800)
        self.assertEqual(totals["cache_hit_tokens"], 900)
        self.assertEqual(totals["cache_miss_tokens"], 600)
        self.assertEqual(totals["cache_hit_rate"], 0.6)
        self.assertEqual(totals["calls"], 2)

        by_tier = summary["by_tier"]
        self.assertEqual(by_tier["strong"]["cache_hit_rate"], 0.8)
        self.assertEqual(by_tier["cheap"]["cache_hit_rate"], 0.2)
        self.assertEqual(by_tier["strong"]["calls"], 1)
        self.assertEqual(by_tier["cheap"]["calls"], 1)
        self.assertEqual(by_tier["strong"]["prompt_tokens"], 1000)
        self.assertEqual(by_tier["cheap"]["prompt_tokens"], 500)

        # stage 归因：显式标注的调用进 by_stage；未标注的只计入 tier
        by_stage = summary["by_stage"]
        self.assertEqual(list(by_stage), ["Translator"])
        self.assertEqual(by_stage["Translator"]["calls"], 1)
        self.assertEqual(by_stage["Translator"]["prompt_tokens"], 1000)
        self.assertEqual(by_stage["Translator"]["cache_hit_rate"], 0.8)


class TestMissingUsage(unittest.TestCase):
    def test_none_usage_silently_skipped(self):
        tracker = UsageTracker()
        tracker.record("strong", None)
        summary = tracker.summary()
        self.assertEqual(summary["totals"]["calls"], 0)
        self.assertEqual(summary["totals"]["total_tokens"], 0)
        self.assertEqual(summary["by_tier"], {})

    def test_complete_with_none_usage_does_not_count(self):
        cfg = _minimal_deepseek_cfg()
        c = DeepSeekClient(cfg)
        with patch.object(
            c,
            "_ensure_client",
            return_value=_ClientStub([_make_response("ok", None)]),
        ):
            self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "ok")
        summary = c.usage_summary()
        self.assertEqual(summary["totals"]["calls"], 0)
        self.assertEqual(summary["totals"]["total_tokens"], 0)
        self.assertEqual(summary["by_tier"], {})

    def test_complete_with_missing_usage_attr_does_not_count(self):
        cfg = _minimal_deepseek_cfg()
        c = DeepSeekClient(cfg)
        msg = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=msg)
        # 无 usage 属性
        resp = SimpleNamespace(choices=[choice])
        with patch.object(c, "_ensure_client", return_value=_ClientStub([resp])):
            self.assertEqual(c.complete([{"role": "user", "content": "x"}]), "ok")
        summary = c.usage_summary()
        self.assertEqual(summary["totals"]["calls"], 0)
        self.assertEqual(summary["by_tier"], {})

    def test_missing_total_tokens_falls_back_to_prompt_plus_completion(self):
        tracker = UsageTracker()
        usage = _make_usage(prompt_tokens=40, completion_tokens=10)
        # 确认未设置 total_tokens
        self.assertFalse(hasattr(usage, "total_tokens"))
        tracker.record("cheap", usage)
        slot = tracker.summary()["by_tier"]["cheap"]
        self.assertEqual(slot["prompt_tokens"], 40)
        self.assertEqual(slot["completion_tokens"], 10)
        self.assertEqual(slot["total_tokens"], 50)
        self.assertEqual(slot["calls"], 1)


class TestEmptyCacheHitRate(unittest.TestCase):
    def test_fresh_client_zero_hit_rate_and_full_keys(self):
        c = FakeClient()
        totals = c.usage_summary()["totals"]
        self.assertEqual(totals["cache_hit_rate"], 0.0)
        self.assertEqual(totals["total_tokens"], 0)
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "cache_hit_rate",
        ):
            self.assertIn(key, totals)
        self.assertEqual(totals["calls"], 0)
        self.assertEqual(totals["prompt_tokens"], 0)
        self.assertEqual(totals["completion_tokens"], 0)
        self.assertEqual(totals["cache_hit_tokens"], 0)
        self.assertEqual(totals["cache_miss_tokens"], 0)
        self.assertEqual(c.usage_summary()["by_tier"], {})
        self.assertEqual(c.usage_summary()["by_stage"], {})


class TestUsageThreadSafety(unittest.TestCase):
    def test_concurrent_record_exact_counts(self):
        client = FakeClient()
        n_workers = 8
        per_worker = 25  # 8 * 25 = 200
        usage = _make_usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_cache_hit_tokens=3,
            prompt_cache_miss_tokens=7,
        )

        def _worker() -> None:
            for _ in range(per_worker):
                client.usage.record("strong", usage)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker) for _ in range(n_workers)]
            for f in concurrent.futures.as_completed(futs):
                f.result()

        total_calls = n_workers * per_worker
        summary = client.usage_summary()
        totals = summary["totals"]
        self.assertEqual(totals["calls"], total_calls)
        self.assertEqual(totals["prompt_tokens"], 10 * total_calls)
        self.assertEqual(totals["completion_tokens"], 5 * total_calls)
        self.assertEqual(totals["total_tokens"], 15 * total_calls)
        self.assertEqual(totals["cache_hit_tokens"], 3 * total_calls)
        self.assertEqual(totals["cache_miss_tokens"], 7 * total_calls)
        self.assertEqual(totals["cache_hit_rate"], 0.3)  # 3/(3+7)
        self.assertEqual(summary["by_tier"]["strong"]["calls"], total_calls)


class TestUsageIncrementalPersistence(unittest.TestCase):
    @staticmethod
    def _record(client: FakeClient, tier: str, *, prompt: int, completion: int) -> None:
        client.usage.record(
            tier,
            _make_usage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                prompt_cache_hit_tokens=prompt // 2,
                prompt_cache_miss_tokens=prompt - prompt // 2,
            ),
        )

    def test_delta_and_merge_do_not_double_count(self):
        client = FakeClient()
        self._record(client, "strong", prompt=100, completion=20)
        first = client.usage_summary()
        self._record(client, "strong", prompt=50, completion=10)
        self._record(client, "fast", prompt=30, completion=5)
        second = client.usage_summary()

        increment = usage_delta(second, first)
        self.assertEqual(increment["totals"]["total_tokens"], 95)
        merged = merge_usage_summaries(first, increment)
        self.assertEqual(merged, second)

    def test_usage_accumulates_across_orchestrators_for_one_book(self):
        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state", "book"))
            config = Config.from_dict({"llm": {"provider": "fake"}})

            first_client = FakeClient()
            first = Orchestrator(config, client=first_client)
            self._record(first_client, "strong", prompt=100, completion=20)
            cumulative = first._flush_usage(store, scope="translate")
            self.assertEqual(cumulative["totals"]["total_tokens"], 120)

            # 同一进程再次 flush 没有新增调用，不能重复累计。
            unchanged = first._flush_usage(store, scope="pipeline")
            self.assertEqual(unchanged["totals"]["total_tokens"], 120)

            # 模拟 resume：新 client / Orchestrator 的增量继续累加到同一本书。
            resumed_client = FakeClient()
            resumed = Orchestrator(config, client=resumed_client)
            self._record(resumed_client, "cheap", prompt=40, completion=10)
            cumulative = resumed._flush_usage(store, scope="translate")

            self.assertEqual(cumulative["totals"]["total_tokens"], 170)
            self.assertEqual(cumulative["totals"]["calls"], 2)
            self.assertEqual(cumulative["by_tier"]["strong"]["total_tokens"], 120)
            self.assertEqual(cumulative["by_tier"]["cheap"]["total_tokens"], 50)
            self.assertEqual(store.load_usage(), cumulative)
            self.assertTrue(os.path.isfile(store.usage_path))

    def test_report_omits_usage_and_usage_file_keeps_book_total(self):
        with tempfile.TemporaryDirectory() as d:
            source = os.path.join(d, "novel.txt")
            write_sample_txt(source)
            config = Config.from_dict(
                {
                    "language": {"source": "ja", "target": "zh"},
                    "llm": {"provider": "fake"},
                    "pipeline": {"book_understanding": False, "review": False},
                    "paths": {"state_dir": os.path.join(d, "state")},
                }
            )

            initial_client = FakeClient(handler=routing_handler)
            initial = Orchestrator(config, client=initial_client)
            store = initial.run_steps(source, {"translate"})["store"]
            self._record(initial_client, "strong", prompt=100, completion=20)
            initial._flush_usage(store, scope="translate")

            resumed_client = FakeClient(handler=routing_handler)
            resumed = Orchestrator(config, client=resumed_client)
            self._record(resumed_client, "cheap", prompt=40, completion=10)
            result = resumed.run_steps(source, {"report"})

            self.assertNotIn("usage", result["report"])
            usage = result["store"].load_usage()
            self.assertIsNotNone(usage)
            assert usage is not None
            self.assertEqual(usage["totals"]["total_tokens"], 170)
            self.assertEqual(usage["totals"]["calls"], 2)
            self.assertEqual(result["store"].load_usage(), usage)

    def test_operation_only_failure_persists_and_second_flush_does_not_duplicate(self):
        """由 Agent default 捕获的失败调用：by_tier/by_stage/totals.calls 全零
        （无成功响应），但 by_operation 的 attempts/failed_attempts/logical_calls
        真实增长——_flush_usage 不得因 totals.calls==0 就跳过持久化。"""
        from trans_novel.agents.base import Agent

        with tempfile.TemporaryDirectory() as d:
            store = RunStore(os.path.join(d, "state", "book"))
            config = Config.from_dict({"llm": {"provider": "fake"}})

            def _boom(messages, tier, json_mode):
                raise RuntimeError("model down")

            client = FakeClient(handler=_boom)
            orch = Orchestrator(config, client=client)
            agent = Agent(client, config)

            result = agent._ask_json(
                "sys", "user", tier="strong", default={}, operation="translate.batch"
            )
            self.assertEqual(result, {}, "default 应吞掉异常，Agent 调用方视角照常返回")

            op_before = client.usage_summary()["by_operation"]["translate.batch"]
            self.assertGreater(op_before["attempts"], 0)
            self.assertGreater(op_before["failed_attempts"], 0)
            self.assertGreater(op_before["logical_calls"], 0)
            self.assertEqual(op_before["calls"], 0)  # 无成功响应，token/calls 字段不动

            cumulative = orch._flush_usage(store, scope="translate")
            self.assertEqual(
                cumulative["by_operation"]["translate.batch"]["attempts"], op_before["attempts"]
            )
            self.assertEqual(
                cumulative["by_operation"]["translate.batch"]["failed_attempts"],
                op_before["failed_attempts"],
            )

            persisted = store.load_usage()
            self.assertIsNotNone(persisted)
            assert persisted is not None
            self.assertEqual(persisted, cumulative)
            self.assertEqual(
                persisted["by_operation"]["translate.batch"]["logical_calls"],
                op_before["logical_calls"],
            )

            # 第二次 flush：没有新调用，增量为 0，不得重复累加或再次写盘造成翻倍。
            unchanged = orch._flush_usage(store, scope="translate")
            self.assertEqual(unchanged, cumulative)
            self.assertEqual(store.load_usage(), cumulative)


class TestOperationTelemetry(unittest.TestCase):
    """by_operation：新增槽位、字段计数、旧快照向后兼容（decision 16/43/47）。"""

    def test_new_operation_slot_has_full_canonical_field_set(self):
        c = FakeClient()
        c.complete([{"role": "user", "content": "x"}], operation="translate.batch")
        op = c.usage_summary()["by_operation"]["translate.batch"]
        for key in (
            "calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "cache_hit_rate",
            "logical_calls",
            "attempts",
            "failed_attempts",
            "elapsed_ms",
            "reasoning_tokens",
            "accepted",
            "rejected",
        ):
            self.assertIn(key, op)
        # FakeClient 不产生真实 provider usage：token/cache 字段保持 0，
        # 但 operation 标签本身与 attempts/logical_calls 必须被记录。
        self.assertEqual(op["calls"], 0)
        self.assertEqual(op["prompt_tokens"], 0)
        self.assertEqual(op["attempts"], 1)
        self.assertEqual(op["logical_calls"], 1)
        self.assertEqual(op["failed_attempts"], 0)

    def test_operation_missing_from_old_snapshot_normalizes_to_zero(self):
        """旧快照没有 by_operation 键（含"没有该 operation 键"两种缺档形态）：
        merge/delta 后新增的 operation 槽位仍是全字段、缺失按 0 补齐，不报错。"""
        old_snapshot = {"totals": {}, "by_tier": {}, "by_stage": {}}  # 完全没有 by_operation 键
        c = FakeClient()
        c.usage.record(
            "strong", _make_usage(prompt_tokens=10, completion_tokens=5), operation="polish.batch"
        )
        merged = merge_usage_summaries(old_snapshot, c.usage_summary())
        self.assertIn("by_operation", merged)
        slot = merged["by_operation"]["polish.batch"]
        self.assertEqual(slot["prompt_tokens"], 10)
        self.assertEqual(slot["accepted"], 0)  # 旧快照没有的字段按 0 合并
        self.assertEqual(
            slot["attempts"], 0
        )  # record() 走 usage.record 而非 complete()，未记 attempts

    def test_delta_and_merge_round_trip_for_operation_fields(self):
        c = FakeClient()
        c.usage.record(
            "strong",
            _make_usage(prompt_tokens=100, completion_tokens=20),
            operation="translate.batch",
        )
        c.usage.record_outcome("translate.batch", accepted=True)
        first = c.usage_summary()
        c.usage.record(
            "strong",
            _make_usage(prompt_tokens=50, completion_tokens=10),
            operation="translate.batch",
        )
        c.usage.record_outcome("translate.batch", accepted=False)
        second = c.usage_summary()

        increment = usage_delta(second, first)
        self.assertEqual(increment["by_operation"]["translate.batch"]["prompt_tokens"], 50)
        self.assertEqual(increment["by_operation"]["translate.batch"]["rejected"], 1)
        self.assertEqual(increment["by_operation"]["translate.batch"]["accepted"], 0)
        merged = merge_usage_summaries(first, increment)
        self.assertEqual(merged["by_operation"], second["by_operation"])

    def test_reasoning_tokens_direct_field(self):
        tracker = UsageTracker()
        usage = _make_usage(prompt_tokens=10, completion_tokens=20)
        usage.reasoning_tokens = 7
        tracker.record("strong", usage, operation="naturalize.rewrite")
        slot = tracker.summary()["by_operation"]["naturalize.rewrite"]
        self.assertEqual(slot["reasoning_tokens"], 7)
        # 不叠加进 total_tokens
        self.assertEqual(slot["total_tokens"], 30)

    def test_reasoning_tokens_nested_completion_details(self):
        tracker = UsageTracker()
        usage = _make_usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=4)
        tracker.record("strong", usage, operation="naturalize.rewrite")
        slot = tracker.summary()["by_operation"]["naturalize.rewrite"]
        self.assertEqual(slot["reasoning_tokens"], 4)
        self.assertEqual(slot["total_tokens"], 30)

    def test_outcome_and_by_tier_by_stage_unaffected(self):
        """by_tier/by_stage 保持原字段集合不变，不被 operation 新字段污染。"""
        c = FakeClient()
        c.usage.record(
            "strong",
            _make_usage(prompt_tokens=1, completion_tokens=1),
            "Polisher",
            operation="polish.batch",
        )
        summary = c.usage_summary()
        self.assertNotIn("attempts", summary["by_tier"]["strong"])
        self.assertNotIn("attempts", summary["by_stage"]["Polisher"])
        self.assertIn("attempts", summary["by_operation"]["polish.batch"])


class TestDeepSeekAttemptTelemetry(unittest.TestCase):
    """底层 attempt/failed_attempt/logical_call/elapsed_ms 统计（decision 45）。"""

    def test_retry_then_success_counts_attempts_and_one_logical_call(self):
        cfg = _minimal_deepseek_cfg()
        cfg.max_retries = 2
        c = DeepSeekClient(cfg)

        # _ClientStub 只支持按响应列表顺序返回；异常项需要直接抛出而非当作响应，
        # 这里用一个最小自定义 stub 模拟"前两次抛异常，第三次成功"。
        class _FlakyCompletions:
            def __init__(self, items):
                self._items = list(items)
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                item = self._items.pop(0)
                if isinstance(item, Exception):
                    raise item
                return item

        flaky = SimpleNamespace(
            chat=SimpleNamespace(
                completions=_FlakyCompletions(
                    [RuntimeError("boom"), RuntimeError("boom"), _make_response("ok", None)]
                )
            )
        )
        with patch.object(c, "_ensure_client", return_value=flaky):
            result = c.complete(
                [{"role": "user", "content": "x"}], tier="strong", operation="translate.batch"
            )
        self.assertEqual(result, "ok")
        op = c.usage_summary()["by_operation"]["translate.batch"]
        self.assertEqual(op["attempts"], 3)
        self.assertEqual(op["failed_attempts"], 2)
        self.assertEqual(op["logical_calls"], 1)
        self.assertGreaterEqual(op["elapsed_ms"], 0)

    def test_terminal_failure_still_records_before_reraising(self):
        cfg = _minimal_deepseek_cfg()
        cfg.max_retries = 1  # 最多尝试 2 次
        c = DeepSeekClient(cfg)

        class _AlwaysFailCompletions:
            def create(self, **kwargs):
                raise RuntimeError("down")

        flaky = SimpleNamespace(chat=SimpleNamespace(completions=_AlwaysFailCompletions()))
        with patch.object(c, "_ensure_client", return_value=flaky):
            with self.assertRaises(RuntimeError):
                c.complete(
                    [{"role": "user", "content": "x"}], tier="strong", operation="translate.batch"
                )
        op = c.usage_summary()["by_operation"]["translate.batch"]
        self.assertEqual(op["attempts"], 2)
        self.assertEqual(op["failed_attempts"], 2)
        self.assertEqual(op["logical_calls"], 1)  # 终态失败仍只计一次逻辑调用
        self.assertEqual(op["calls"], 0)  # 无成功响应，token 字段不动

    def test_ensure_client_init_failure_records_logical_call_and_reraises(self):
        """_ensure_client 初始化失败（缺 API key）：即便从未发出 create() 请求
        （attempts=0），本次逻辑调用仍要计 logical_calls/elapsed_ms 后原样重抛。"""
        cfg = _minimal_deepseek_cfg()
        c = DeepSeekClient(cfg)
        env_backup = os.environ.pop(cfg.api_key_env, None)
        try:
            with self.assertRaises(RuntimeError):
                c.complete(
                    [{"role": "user", "content": "x"}], tier="strong", operation="translate.batch"
                )
        finally:
            if env_backup is not None:
                os.environ[cfg.api_key_env] = env_backup
        op = c.usage_summary()["by_operation"]["translate.batch"]
        self.assertEqual(op["logical_calls"], 1)
        self.assertEqual(op["attempts"], 0)
        self.assertEqual(op["failed_attempts"], 0)
        self.assertGreaterEqual(op["elapsed_ms"], 0)
        self.assertEqual(op["calls"], 0)

    def test_resolve_tier_failure_records_logical_call_and_reraises(self):
        """resolve_tier 缺 strong 档抛 KeyError：同样先记账（logical_calls=1，
        attempts=0，因为从未走到 create()）再原样重抛。"""
        cfg = _minimal_deepseek_cfg()
        cfg.tiers = {"cheap": cfg.tiers["cheap"]}  # 缺 strong 档
        c = DeepSeekClient(cfg)
        with self.assertRaises(KeyError):
            c.complete(
                [{"role": "user", "content": "x"}], tier="strong", operation="translate.batch"
            )
        op = c.usage_summary()["by_operation"]["translate.batch"]
        self.assertEqual(op["logical_calls"], 1)
        self.assertEqual(op["attempts"], 0)


class TestFakeClientOperationTelemetry(unittest.TestCase):
    """FakeClient 记录 operation 便于测试，不伪造 provider token（decision 17/52/61）。"""

    def test_records_operation_on_calls_without_faking_tokens(self):
        c = FakeClient(handler=lambda m, t, j: "ok")
        c.complete([{"role": "user", "content": "x"}], operation="review.chapter")
        self.assertEqual(c.calls[0]["operation"], "review.chapter")
        op = c.usage_summary()["by_operation"]["review.chapter"]
        self.assertEqual(op["prompt_tokens"], 0)
        self.assertEqual(op["total_tokens"], 0)
        self.assertEqual(op["logical_calls"], 1)

    def test_handler_exception_records_failed_attempt_then_reraises(self):
        def _boom(messages, tier, json_mode):
            raise ValueError("bad")

        c = FakeClient(handler=_boom)
        with self.assertRaises(ValueError):
            c.complete([{"role": "user", "content": "x"}], operation="review.chapter")
        op = c.usage_summary()["by_operation"]["review.chapter"]
        self.assertEqual(op["attempts"], 1)
        self.assertEqual(op["failed_attempts"], 1)
        self.assertEqual(op["logical_calls"], 1)

    def test_concurrent_calls_do_not_lose_or_corrupt_records(self):
        """FakeClient.calls 只用一把锁保护列表本身，never 持锁调用 handler；
        并发下 calls 长度与各 operation 的 logical_calls 计数必须精确，不丢不重。"""
        c = FakeClient(handler=lambda m, t, j: "ok")
        n = 64

        def _one(i):
            c.complete([{"role": "user", "content": str(i)}], operation="naturalize.pair")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_one, range(n)))

        self.assertEqual(len(c.calls), n)
        op = c.usage_summary()["by_operation"]["naturalize.pair"]
        self.assertEqual(op["logical_calls"], n)
        self.assertEqual(op["attempts"], n)


if __name__ == "__main__":
    unittest.main()
