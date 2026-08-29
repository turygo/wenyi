"""三模型角色路由与可重试错误归一化测试。"""

from __future__ import annotations

import json
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.config import Config, ModelRef
from trans_novel.llm import AgentRouter, GenerationOptions, build_client
from trans_novel.llm.errors import AllModelsFailedError, UnknownAgentError
from trans_novel.llm.retrying import CONNECTION, RATE_LIMIT, TIMEOUT, classify_retry
from trans_novel.llm.usage import UsageTracker


class _Http429(Exception):
    status_code = 429


class _Http400(Exception):
    status_code = 400


class StubTransport:
    def __init__(self, provider: str = "fake") -> None:
        self.provider = provider
        self.usage = UsageTracker()
        self.calls: list[dict] = []
        self.results: list[object] = []

    def plan(self, *results: object) -> StubTransport:
        self.results.extend(results)
        return self

    def complete(
        self,
        messages,
        model_ref: ModelRef,
        *,
        json_mode=False,
        max_tokens=None,
        stage=None,
        agent,
        operation,
    ):
        self.calls.append(
            {
                "messages": messages,
                "model_ref": model_ref,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
                "stage": stage,
                "agent": agent,
                "operation": operation,
            }
        )
        result = self.results.pop(0) if self.results else "ok"
        if isinstance(result, BaseException):
            raise result
        return result


def _router(transport: StubTransport) -> AgentRouter:
    cfg = Config.from_dict(
        {
            "llm": {
                "provider": "fake",
                "models": {
                    "primary": "primary-id",
                    "editor": "editor-id",
                    "fast": "fast-id",
                },
            }
        }
    )
    return AgentRouter(cfg, transports={"fake": transport})


class TestRetryClassifier(unittest.TestCase):
    def test_retryable_categories(self):
        self.assertEqual(classify_retry(TimeoutError("late")), TIMEOUT)
        self.assertEqual(classify_retry(ConnectionError("down")), CONNECTION)
        self.assertEqual(classify_retry(_Http429()), RATE_LIMIT)

    def test_permanent_error_not_retryable(self):
        self.assertIsNone(classify_retry(_Http400()))
        self.assertIsNone(classify_retry(ValueError("bad config")))


class TestAgentRouter(unittest.TestCase):
    def test_primary_agents_use_primary_model_with_reasoning(self):
        for agent in ("translator", "analyst"):
            with self.subTest(agent=agent):
                transport = StubTransport()
                router = _router(transport)
                router.complete([{"role": "user", "content": "x"}], agent=agent, operation="op")
                ref = transport.calls[0]["model_ref"]
                self.assertEqual(ref.full_name, "fake:primary-id")
                self.assertTrue(ref.reasoning_enabled)
                self.assertEqual(ref.reasoning_effort, "high")

    def test_editor_agent_uses_editor_model_with_reasoning(self):
        transport = StubTransport()
        router = _router(transport)
        router.complete([{"role": "user", "content": "x"}], agent="editor", operation="op")
        ref = transport.calls[0]["model_ref"]
        self.assertEqual(ref.full_name, "fake:editor-id")
        self.assertTrue(ref.reasoning_enabled)
        self.assertEqual(ref.reasoning_effort, "high")

    def test_fast_agents_use_fast_model_without_reasoning(self):
        for agent in ("preparer", "light-translator"):
            with self.subTest(agent=agent):
                transport = StubTransport()
                router = _router(transport)
                router.complete([{"role": "user", "content": "x"}], agent=agent, operation="op")
                ref = transport.calls[0]["model_ref"]
                self.assertEqual(ref.full_name, "fake:fast-id")
                self.assertFalse(ref.reasoning_enabled)

    def test_omitted_editor_inherits_primary_thinking_suffix(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "opencode-go",
                    "models": {
                        "primary": "deepseek-v4-flash:max",
                        "fast": "deepseek-v4-flash:off",
                    },
                }
            }
        )
        transport = StubTransport("opencode-go")
        router = AgentRouter(cfg, transports={"opencode-go": transport})
        router.complete([{"role": "user", "content": "x"}], agent="editor", operation="op")
        ref = transport.calls[0]["model_ref"]
        self.assertEqual(ref.model, "deepseek-v4-flash")
        self.assertTrue(ref.reasoning_enabled)
        self.assertEqual(ref.reasoning_effort, "max")

    def test_explicit_suffix_overrides_role_defaults(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "opencode-go",
                    "models": {
                        "primary": "deepseek-v4-flash:off",
                        "editor": "deepseek-v4-flash:max",
                        "fast": "deepseek-v4-flash:max",
                    },
                }
            }
        )
        transport = StubTransport("opencode-go")
        router = AgentRouter(cfg, transports={"opencode-go": transport})
        router.complete([{"role": "user", "content": "x"}], agent="translator", operation="op")
        router.complete([{"role": "user", "content": "x"}], agent="editor", operation="op")
        router.complete(
            [{"role": "user", "content": "x"}], agent="light-translator", operation="op"
        )
        primary, editor, fast = (call["model_ref"] for call in transport.calls)
        self.assertEqual(primary.model, "deepseek-v4-flash")
        self.assertFalse(primary.reasoning_enabled)
        self.assertEqual(editor.model, "deepseek-v4-flash")
        self.assertTrue(editor.reasoning_enabled)
        self.assertEqual(editor.reasoning_effort, "max")
        self.assertEqual(fast.model, "deepseek-v4-flash")
        self.assertTrue(fast.reasoning_enabled)
        self.assertEqual(fast.reasoning_effort, "max")

    def test_unknown_agent_fails_before_transport(self):
        transport = StubTransport()
        router = _router(transport)
        with self.assertRaises(UnknownAgentError):
            router.complete(
                [{"role": "user", "content": "x"}],
                agent="missing",
                operation="op",
            )
        self.assertEqual(transport.calls, [])

    def test_retryable_exhaustion_is_sanitized(self):
        transport = StubTransport().plan(TimeoutError("secret response body"))
        router = _router(transport)
        with self.assertRaises(AllModelsFailedError) as caught:
            router.complete(
                [{"role": "user", "content": "x"}],
                agent="translator",
                operation="translate.batch",
            )
        self.assertEqual(str(caught.exception), "fake:primary-id: timeout")
        self.assertNotIn("secret", str(caught.exception))

    def test_permanent_error_propagates(self):
        transport = StubTransport().plan(ValueError("bad request"))
        router = _router(transport)
        with self.assertRaisesRegex(ValueError, "bad request"):
            router.complete(
                [{"role": "user", "content": "x"}],
                agent="translator",
                operation="translate.batch",
            )

    def test_complete_json_routes_and_parses_once(self):
        transport = StubTransport().plan(json.dumps({"ok": True}))
        router = _router(transport)
        self.assertEqual(
            router.complete_json(
                [{"role": "user", "content": "x"}],
                agent="editor",
                operation="polish.batch",
            ),
            {"ok": True},
        )
        self.assertTrue(transport.calls[0]["json_mode"])

    def test_controlled_options_validate_before_stub_transport(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "bailian",
                    "models": {
                        "primary": "qwen3.8-max:off",
                        "editor": "deepseek-v4-pro-us:off",
                        "fast": "qwen3.7-flash-2026-07-15:off",
                    },
                }
            }
        )
        transport = StubTransport("bailian")
        router = AgentRouter(
            cfg,
            transports={"bailian": transport},
            generation_options=GenerationOptions(
                temperature=0.1,
                require_catalogued_model=True,
                require_thinking_disabled=True,
            ),
        )
        for agent in ("translator", "editor", "preparer"):
            with self.subTest(agent=agent):
                router.complete([{"role": "user", "content": "x"}], agent=agent, operation="op")
        self.assertEqual(len(transport.calls), 3)

    def test_unknown_model_is_rejected_before_stub_transport(self):
        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "bailian",
                    "models": {
                        "primary": "unknown-model:off",
                        "editor": "unknown-model:off",
                        "fast": "unknown-model:off",
                    },
                }
            }
        )
        transport = StubTransport("bailian")
        router = AgentRouter(
            cfg,
            transports={"bailian": transport},
            generation_options=GenerationOptions(
                temperature=0.1,
                require_catalogued_model=True,
            ),
        )
        with self.assertRaises(ValueError):
            router.complete([{"role": "user", "content": "x"}], agent="translator", operation="op")
        self.assertEqual(transport.calls, [])

    def test_generation_options_override_before_and_after_materialization(self):
        from trans_novel.llm.registry import ProviderRegistry

        cfg = Config.from_dict(
            {
                "llm": {
                    "provider": "fake",
                    "models": {
                        "primary": "primary-id",
                        "editor": "editor-id",
                        "fast": "fast-id",
                    },
                }
            }
        )
        registry = ProviderRegistry(cfg.llm, UsageTracker())
        options = GenerationOptions(temperature=0.1, seed=7)

        AgentRouter(cfg, registry=registry, generation_options=options)
        self.assertEqual(registry.generation_options, options)
        transport = registry.transport()
        self.assertEqual(transport.generation_options, options)

        AgentRouter(
            cfg,
            registry=registry,
            generation_options=GenerationOptions(temperature=0.1, seed=7),
        )
        with self.assertRaisesRegex(ValueError, "generation options"):
            AgentRouter(
                cfg,
                registry=registry,
                generation_options=GenerationOptions(temperature=0.2, seed=7),
            )

    def test_factory_returns_router(self):
        cfg = Config.from_dict({"llm": fake_llm_dict()})
        self.assertIsInstance(build_client(cfg), AgentRouter)


if __name__ == "__main__":
    unittest.main()
