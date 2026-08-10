"""AgentRouter 路由/降级与集中重试分类的离线契约测试。

确定性 stub 传输通过 AgentRouter 的 transports 参数注入，无需生产插件或 YAML hook。
路由只按 agent（六个功能 Agent 键之一）；operation 是内部业务标签，不参与路由，
只作用量/调试归因（by_operation 视图）。
"""

from __future__ import annotations

import unittest

import yaml

from trans_novel.config import (
    PRODUCTION_AGENT_IDS,
    AgentRouteConfig,
    Config,
    LLMConfig,
    ModelRef,
    ProviderConfig,
    ProviderModelConfig,
)
from trans_novel.llm import AgentRouter, build_client
from trans_novel.llm.errors import AllModelsFailedError, UnknownAgentError
from trans_novel.llm.retrying import (
    CONNECTION,
    EMPTY_RESPONSE,
    HTTP_408,
    HTTP_409,
    PROVIDER_RETRY,
    RATE_LIMIT,
    SERVER_ERROR,
    TIMEOUT,
    EmptyResponseError,
    classify_retry,
)


class _StatusError(Exception):
    status_code = None


class _Http429(_StatusError):
    status_code = 429


class _Http500(_StatusError):
    status_code = 500


class _Http400(_StatusError):
    status_code = 400


class _Http408(_StatusError):
    status_code = 408


class _Http409(_StatusError):
    status_code = 409


class _RemoteDisconnected(Exception):
    pass


class _ProxyError(Exception):
    pass


class _ConnectTimeoutError(Exception):
    pass


class _Gaierror(Exception):
    pass


class StubTransport:
    """确定性 stub 传输：按模型别名顺序消费计划项（str 成功 / Exception 失败）。

    物理记账与真实传输一致（record_attempt / record_attempt_failed）。
    """

    def __init__(self, usage, alias: str):
        self.usage = usage
        self.alias = alias
        self._plan: dict[str, list] = {}
        self.calls: list[tuple[str, dict]] = []
        self.fail_before_request: Exception | None = None

    def plan(self, model_ref: ModelRef, *items) -> "StubTransport":
        self._plan.setdefault(model_ref.full_name, []).extend(items)
        return self

    def complete(
        self,
        messages,
        model_ref,
        *,
        json_mode=False,
        max_tokens=None,
        stage=None,
        agent=None,
        operation=None,
    ):
        self.calls.append(
            (
                model_ref.full_name,
                {
                    "json_mode": json_mode,
                    "max_tokens": max_tokens,
                    "stage": stage,
                    "agent": agent,
                    "operation": operation,
                },
            )
        )
        if self.fail_before_request is not None:
            raise self.fail_before_request  # 模拟凭据/配置失败：无物理尝试
        self.usage.record_attempt(
            agent=agent, operation=operation, provider=self.alias, model_ref=model_ref
        )
        items = self._plan.get(model_ref.full_name)
        if not items:
            raise AssertionError(f"stub {self.alias}:{model_ref.full_name} 没有可消费的计划项")
        item = items.pop(0)
        if isinstance(item, Exception):
            self.usage.record_attempt_failed(
                agent=agent, operation=operation, provider=self.alias, model_ref=model_ref
            )
            raise item
        return item


def _providers() -> dict[str, ProviderConfig]:
    return {
        "p1": ProviderConfig(
            type="fake",
            models={
                "m1": ProviderModelConfig(id="m1"),
                "m2": ProviderModelConfig(id="m2"),
            },
        ),
        "p2": ProviderConfig(
            type="fake",
            models={"n1": ProviderModelConfig(id="n1")},
        ),
    }


def _build(agents: dict[str, AgentRouteConfig], transports: dict) -> AgentRouter:
    cfg = Config(llm=LLMConfig(providers=_providers(), agents=agents))
    return AgentRouter(cfg, transports=transports)


def _route(model: str, *fallback: str) -> AgentRouteConfig:
    refs = [ModelRef.parse(model), *(ModelRef.parse(f) for f in fallback)]
    return AgentRouteConfig(model=refs[0], fallback=tuple(refs[1:]))


class TestRetryClassifier(unittest.TestCase):
    def test_status_matrix(self):
        self.assertEqual(classify_retry(_Http408()), HTTP_408)
        self.assertEqual(classify_retry(_Http409()), HTTP_409)
        self.assertEqual(classify_retry(_Http429()), RATE_LIMIT)
        self.assertEqual(classify_retry(_Http500()), SERVER_ERROR)
        self.assertEqual(classify_retry(_Http400()), None)
        for code in (401, 403, 404, 422, 300):
            err = _StatusError()
            err.status_code = code
            self.assertIsNone(classify_retry(err), code)

    def test_decimal_string_status(self):
        err = _StatusError()
        err.code = "429"
        self.assertEqual(classify_retry(err), RATE_LIMIT)
        err2 = _StatusError()
        err2.response = type("R", (), {"status_code": 503})()
        self.assertEqual(classify_retry(err2), SERVER_ERROR)

    def test_empty_response(self):
        self.assertEqual(classify_retry(EmptyResponseError("x")), EMPTY_RESPONSE)

    def test_timeout_families(self):
        self.assertEqual(classify_retry(TimeoutError("slow")), TIMEOUT)
        self.assertEqual(classify_retry(_ConnectTimeoutError()), TIMEOUT)

    def test_connection_families(self):
        self.assertEqual(classify_retry(ConnectionError("reset")), CONNECTION)
        self.assertEqual(classify_retry(ConnectionResetError("reset")), CONNECTION)
        self.assertEqual(classify_retry(_RemoteDisconnected()), CONNECTION)
        self.assertEqual(classify_retry(_ProxyError()), CONNECTION)
        self.assertEqual(classify_retry(_Gaierror()), CONNECTION)

    def test_unknown_or_runtime_errors_are_permanent(self):
        self.assertIsNone(classify_retry(RuntimeError("down")))
        self.assertIsNone(classify_retry(ValueError("bad")))

    def test_explicit_header_false_wins_over_status(self):
        err = _Http429()
        err.response = type("R", (), {"headers": {"X-Should-Retry": "false"}})()
        self.assertIsNone(classify_retry(err))

    def test_explicit_header_true_authorizes_but_narrower_reason_wins(self):
        err = _Http429()
        err.response = type("R", (), {"headers": {"x-should-retry": "True"}})()
        self.assertEqual(classify_retry(err), RATE_LIMIT)  # 状态类别优先于 provider_retry
        plain = _Http400()
        plain.response = type("R", (), {"headers": {"x-should-retry": "true"}})()
        self.assertIsNone(classify_retry(plain))  # 4xx 的不可重试语义优先：显式 true 也不能覆盖
        unknown = RuntimeError("weird")
        unknown.headers = {"X-Should-Retry": True}
        self.assertEqual(classify_retry(unknown), PROVIDER_RETRY)

    def test_header_read_from_error_headers_and_response_headers_case_insensitive(self):
        err = _Http500()
        err.response = type("R", (), {"headers": {"X-SHOULD-RETRY": "false"}})()
        self.assertIsNone(classify_retry(err))
        err2 = _Http500()
        err2.headers = {"x-should-retry": "false"}
        self.assertIsNone(classify_retry(err2))


class TestAgentRouter(unittest.TestCase):
    def test_primary_success_usage(self):
        t1 = StubTransport(object(), "p1")
        router = _build({"op": _route("p1:m1")}, {"p1": t1})
        t1.plan(ModelRef("p1", "m1"), "ok")
        self.assertEqual(
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op"), "ok"
        )
        summary = router.usage_summary()
        agent = summary["by_agent"]["op"]
        self.assertEqual(agent["logical_calls"], 1)
        self.assertEqual(agent["attempts"], 1)
        self.assertEqual(agent["failed_attempts"], 0)
        self.assertEqual(agent["fallbacks"], 0)
        self.assertEqual(summary["by_provider"]["p1"]["attempts"], 1)
        self.assertEqual(summary["by_model"]["p1:m1"]["attempts"], 1)

    def test_same_provider_different_model_fallback(self):
        t1 = StubTransport(object(), "p1")
        router = _build({"op": _route("p1:m1", "p1:m2")}, {"p1": t1})
        t1.plan(ModelRef("p1", "m1"), TimeoutError("boom")).plan(ModelRef("p1", "m2"), "recovered")
        self.assertEqual(
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op"),
            "recovered",
        )
        agent = router.usage_summary()["by_agent"]["op"]
        self.assertEqual(agent["fallbacks"], 1)
        self.assertEqual(agent["attempts"], 2)
        self.assertEqual(agent["failed_attempts"], 1)
        models = router.usage_summary()["by_model"]
        self.assertEqual(models["p1:m1"]["failed_attempts"], 1)
        self.assertEqual(models["p1:m2"]["failed_attempts"], 0)
        self.assertEqual(
            [name for name, _ in t1.calls], ["p1:m1", "p1:m2"], "候选按 primary→fallback 顺序尝试"
        )

    def test_cross_provider_fallback(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), _Http429())
        t2.plan(ModelRef("p2", "n1"), "cross-ok")
        self.assertEqual(
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op"),
            "cross-ok",
        )
        summary = router.usage_summary()
        self.assertEqual(summary["by_provider"]["p1"]["failed_attempts"], 1, "失败计入 by_provider")
        self.assertEqual(summary["by_provider"]["p2"]["attempts"], 1)
        self.assertEqual(summary["by_agent"]["op"]["fallbacks"], 1)

    def test_fallback_increments_once_per_transition(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p1:m2", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), TimeoutError("a")).plan(
            ModelRef("p1", "m2"), TimeoutError("b")
        )
        t2.plan(ModelRef("p2", "n1"), "ok")
        self.assertEqual(
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op"), "ok"
        )
        self.assertEqual(router.usage_summary()["by_agent"]["op"]["fallbacks"], 2)

    def test_permanent_error_propagates_original_type_without_fallback(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), RuntimeError("provider down"))
        t2.plan(ModelRef("p2", "n1"), "should-not-run")
        with self.assertRaisesRegex(RuntimeError, "provider down"):
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op")
        self.assertEqual(t2.calls, [], "永久错误不触发降级")
        agent = router.usage_summary()["by_agent"]["op"]
        self.assertEqual(agent["fallbacks"], 0)
        self.assertEqual(agent["logical_calls"], 1)

    def test_http_400_is_permanent_and_not_hidden(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), _Http400())
        t2.plan(ModelRef("p2", "n1"), "x")
        with self.assertRaises(_Http400):
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op")
        self.assertEqual(t2.calls, [])

    def test_unknown_agent_fails_before_request_and_accounting(self):
        t1 = StubTransport(object(), "p1")
        router = _build({"op": _route("p1:m1")}, {"p1": t1})
        t1.plan(ModelRef("p1", "m1"), "ok")
        with self.assertRaisesRegex(UnknownAgentError, "Agent"):
            router.complete(
                [{"role": "user", "content": "hi"}], agent="missing", operation="translate.batch"
            )
        with self.assertRaises(UnknownAgentError):
            router.complete(
                [{"role": "user", "content": "hi"}], agent="", operation="translate.batch"
            )
        self.assertEqual(t1.calls, [], "未知 Agent 不得发起任何请求")
        summary = router.usage_summary()
        self.assertEqual(summary["by_agent"], {}, "未知 Agent 不做逻辑记账")
        self.assertEqual(summary["by_operation"], {}, "未知 Agent 不做逻辑记账")

    def test_router_selects_models_by_agent_only(self):
        """路由只认 agent：不同 operation 走同一 agent 时命中同一模型；operation 只进归因。"""
        t1 = StubTransport(object(), "p1")
        router = _build({"op": _route("p1:m1", "p1:m2")}, {"p1": t1})
        t1.plan(ModelRef("p1", "m1"), "a").plan(ModelRef("p1", "m1"), "b")
        self.assertEqual(
            router.complete(
                [{"role": "user", "content": "hi"}], agent="op", operation="translate.batch"
            ),
            "a",
        )
        self.assertEqual(
            router.complete(
                [{"role": "user", "content": "hi"}], agent="op", operation="review.chapter"
            ),
            "b",
        )
        self.assertEqual(
            [name for name, _ in t1.calls], ["p1:m1", "p1:m1"], "operation 不改变模型路由"
        )
        summary = router.usage_summary()
        self.assertEqual(
            summary["by_operation"]["translate.batch"]["logical_calls"],
            1,
            "operation 进 by_operation",
        )
        self.assertEqual(summary["by_operation"]["review.chapter"]["logical_calls"], 1)
        self.assertEqual(summary["by_agent"]["op"]["logical_calls"], 2, "同一 agent 汇总")

    def test_one_call_records_both_views_and_totals_once(self):
        """离线烟雾：一次逻辑调用在 by_agent 与 by_operation 记同一事件，fallback 各计一次。"""
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), TimeoutError("a"))
        t2.plan(ModelRef("p2", "n1"), "ok")
        self.assertEqual(
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op.detail"),
            "ok",
        )
        summary = router.usage_summary()
        for view in ("by_agent", "by_operation"):
            slot = summary[view]
            key = "op" if view == "by_agent" else "op.detail"
            self.assertEqual(slot[key]["logical_calls"], 1)
            self.assertEqual(slot[key]["attempts"], 2)
            self.assertEqual(slot[key]["failed_attempts"], 1)
            self.assertEqual(slot[key]["fallbacks"], 1)
        self.assertEqual(summary["by_provider"]["p2"]["attempts"], 1, "物理维度照旧")

    def test_all_models_failed_sanitized_error(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), TimeoutError("secret prompt content"))
        t2.plan(ModelRef("p2", "n1"), _Http500())
        with self.assertRaises(AllModelsFailedError) as ctx:
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op")
        err = ctx.exception
        self.assertEqual(
            [(ref.full_name, reason) for ref, reason in err.records],
            [("p1:m1", TIMEOUT), ("p2:n1", SERVER_ERROR)],
        )
        text = str(err)
        self.assertNotIn("secret prompt content", text)
        self.assertNotIn("hi", text)
        self.assertIsInstance(err.__cause__, _Http500)  # 链自最终错误

    def test_credential_failure_records_logical_call_zero_attempts(self):
        t1 = StubTransport(object(), "p1")
        router = _build({"op": _route("p1:m1")}, {"p1": t1})
        t1.fail_before_request = RuntimeError("未设置环境变量 X（API key）")
        with self.assertRaisesRegex(RuntimeError, "API key"):
            router.complete([{"role": "user", "content": "hi"}], agent="op", operation="op")
        agent = router.usage_summary()["by_agent"]["op"]
        self.assertEqual(agent["logical_calls"], 1)
        self.assertEqual(agent["attempts"], 0)
        self.assertEqual(agent["failed_attempts"], 0)

    def test_complete_json_parses_selected_response_once(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), '{"ok": true}')
        self.assertEqual(
            router.complete_json([{"role": "user", "content": "x"}], agent="op", operation="op"),
            {"ok": True},
        )
        self.assertEqual(t2.calls, [], "primary 成功即不再触碰 fallback")
        self.assertEqual(router.usage_summary()["by_agent"]["op"]["logical_calls"], 1)

    def test_malformed_json_records_usage_but_does_not_fallback(self):
        t1, t2 = StubTransport(object(), "p1"), StubTransport(object(), "p2")
        router = _build({"op": _route("p1:m1", "p2:n1")}, {"p1": t1, "p2": t2})
        t1.plan(ModelRef("p1", "m1"), "这不是 JSON")
        with self.assertRaises(ValueError):
            router.complete_json([{"role": "user", "content": "x"}], agent="op", operation="op")
        self.assertEqual(t2.calls, [], "malformed JSON 解析失败不触发模型降级")
        agent = router.usage_summary()["by_agent"]["op"]
        self.assertEqual(agent["logical_calls"], 1)
        self.assertEqual(agent["attempts"], 1)

    def test_fake_provider_build_client_no_credentials(self):
        from tests.fake_llm import fake_llm_dict

        cfg = Config.from_dict({"llm": fake_llm_dict()})
        router = build_client(cfg)
        self.assertEqual(
            router.complete(
                [{"role": "user", "content": "x"}], agent="preparer", operation="prescan.digest"
            ),
            "",
        )
        self.assertEqual(
            router.complete_json(
                [{"role": "user", "content": "x"}], agent="preparer", operation="prescan.digest"
            ),
            [],
        )
        agent = router.usage_summary()["by_agent"]["preparer"]
        self.assertEqual(agent["attempts"], 2)
        self.assertEqual(agent["failed_attempts"], 0, "fake 空输出是成功响应")
        self.assertEqual(agent["logical_calls"], 2)


class TestShippedConfigRouting(unittest.TestCase):
    """验证包内唯一的默认配置是否恰好声明六个 Agent 路由，并正确绑定模型别名。"""

    def setUp(self) -> None:
        self.cfg = Config.from_dict(yaml.safe_load(Config.default_config_text()))

    def test_all_production_agent_ids_bound_exactly(self):
        self.assertEqual(set(self.cfg.llm.agents), set(PRODUCTION_AGENT_IDS))
        self.assertEqual(len(PRODUCTION_AGENT_IDS), 6)

    def test_shipped_primary_refs_with_empty_fallback(self):
        expected = {
            "translator": "deepseek:pro",
            "editor": "deepseek:pro",
            "reviewer": "deepseek:flash-thinking",
            "analyst": "deepseek:pro",
            "preparer": "deepseek:flash-fast",
            "light-translator": "deepseek:flash-fast",
        }
        self.assertEqual(len(expected), 6, "六个可配置 Agent")
        for agent, primary in expected.items():
            route = self.cfg.llm.agents[agent]
            self.assertEqual(route.model.full_name, primary, agent)
            self.assertEqual(route.fallback, (), agent)

    def test_shipped_deepseek_model_aliases(self):
        models = self.cfg.llm.providers["deepseek"].models
        self.assertEqual(set(models), {"pro", "flash-thinking", "flash-fast"})


if __name__ == "__main__":
    unittest.main()
