"""Ordered provider routing and retry classification contracts."""

from __future__ import annotations

import json
import unittest
from typing import ClassVar

from trans_novel.config import Config, ModelRef
from trans_novel.llm import AgentRouter, GenerationOptions
from trans_novel.llm.errors import AllModelsFailedError, UnknownAgentError
from trans_novel.llm.retrying import (
    CONNECTION,
    MODEL_NOT_FOUND,
    RATE_LIMIT,
    SERVER_ERROR,
    TIMEOUT,
    classify_fallback,
    classify_retry,
)
from trans_novel.llm.usage import UsageTracker


class _Http429(Exception):
    status_code = 429


class _Http400(Exception):
    status_code = 400


class _Http401(Exception):
    status_code = 401


class _Http403(Exception):
    status_code = 403


class _Http404(Exception):
    status_code = 404


class _StructuredNotFound(Exception):
    body: ClassVar[dict[str, dict[str, str]]] = {"error": {"code": "model_not_found"}}


class _StructuredNotFoundRetryable(Exception):
    code = MODEL_NOT_FOUND
    status_code = 500
    headers: ClassVar[dict[str, str]] = {"x-should-retry": "true"}


class StubTransport:
    def __init__(self, provider: str = "fake") -> None:
        self.provider = provider
        self.usage = UsageTracker()
        self.calls: list[dict] = []
        self.results: list[object] = []

    def plan(self, *results: object) -> StubTransport:
        self.results.extend(results)
        return self

    def capabilities_for(self, model: str):
        from trans_novel.model_profiles import capabilities_for

        return capabilities_for(self.provider, model)

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
        logical_call_id=None,
        attempt_counter=None,
    ):
        if attempt_counter is not None:
            attempt_counter[0] += 1
        self.calls.append(
            {
                "model_ref": model_ref,
                "json_mode": json_mode,
                "logical_call_id": logical_call_id,
                "attempt_index": attempt_counter[0] if attempt_counter is not None else 1,
            }
        )
        result = self.results.pop(0) if self.results else "ok"
        if isinstance(result, BaseException):
            raise result
        return result


class _Sink:
    def record(self, attempt) -> None:
        pass


def _router(*, models=None, transports=None, options=None, telemetry_sink=None) -> AgentRouter:
    models = models or {
        "primary": ["fake/primary-id"],
        "editor": ["fake/editor-id"],
        "fast": ["fake/fast-id"],
    }
    cfg = Config.from_dict({"llm": {"models": models}})
    return AgentRouter(
        cfg,
        transports=transports or {"fake": StubTransport()},
        generation_options=options,
        telemetry_sink=telemetry_sink,
    )


class TestRetryClassifier(unittest.TestCase):
    def test_retryable_categories(self):
        self.assertEqual(classify_retry(TimeoutError("late")), TIMEOUT)
        self.assertEqual(classify_retry(ConnectionError("down")), CONNECTION)
        self.assertEqual(classify_retry(_Http429()), RATE_LIMIT)

    def test_permanent_error_not_retryable(self):
        self.assertIsNone(classify_retry(_Http400()))
        self.assertIsNone(classify_retry(ValueError("bad config")))

    def test_structured_model_not_found_overrides_retryable_status_and_header(self):
        self.assertEqual(classify_retry(_StructuredNotFoundRetryable()), SERVER_ERROR)
        self.assertEqual(
            classify_fallback(_StructuredNotFoundRetryable()),
            MODEL_NOT_FOUND,
        )

    def test_fallback_classifier_only_uses_structured_not_found(self):
        self.assertEqual(classify_fallback(_Http404()), MODEL_NOT_FOUND)
        self.assertEqual(classify_fallback(_StructuredNotFound()), MODEL_NOT_FOUND)
        self.assertIsNone(classify_fallback(ValueError("model_not_found in message")))


class TestAgentRouter(unittest.TestCase):
    def test_first_candidate_success_never_touches_backup(self):
        first = StubTransport("fake").plan("ok")
        backup = StubTransport("bailian").plan("backup")
        router = _router(
            models={
                "primary": ["fake/first", "bailian/second"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
            transports={"fake": first, "bailian": backup},
        )
        self.assertEqual(router.complete([], agent="translator", operation="op"), "ok")
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(backup.calls, [])

    def test_transport_does_not_retry_structured_model_not_found(self):
        from trans_novel.llm.providers.transport import OpenAICompatibleTransport

        class _Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise _StructuredNotFoundRetryable()

        class _Client:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": _Completions()})()

        cfg = Config.from_dict(
            {
                "llm": {
                    "models": {
                        "primary": ["fake/first", "bailian/second"],
                        "editor": ["fake/editor"],
                        "fast": ["fake/fast"],
                    }
                }
            }
        )
        first = OpenAICompatibleTransport(
            cfg.llm,
            UsageTracker(),
            provider="fake",
            provider_name="fake",
            default_base_url="https://fake.example",
            default_api_key_env=None,
            requires_api_key=False,
        )
        first._client = _Client()
        backup = StubTransport("bailian").plan("done")
        router = _router(
            models={
                "primary": ["fake/first", "bailian/second"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
            transports={"fake": first, "bailian": backup},
        )

        self.assertEqual(router.complete([], agent="translator", operation="op"), "done")
        self.assertEqual(first._client.chat.completions.calls, 1)

    def test_fallback_switches_in_order_and_sanitizes_records(self):
        first = StubTransport("fake").plan(TimeoutError("secret one"))
        second = StubTransport("bailian").plan(_Http404())
        third = StubTransport("openai").plan("done")
        router = _router(
            models={
                "primary": ["fake/one", "bailian/two", "openai/three"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
            transports={"fake": first, "bailian": second, "openai": third},
            telemetry_sink=_Sink(),
        )
        self.assertEqual(router.complete([], agent="translator", operation="op"), "done")
        self.assertEqual([c["model_ref"].full_name for c in first.calls], ["fake:one"])
        self.assertEqual([c["model_ref"].full_name for c in second.calls], ["bailian:two"])
        self.assertEqual([c["model_ref"].full_name for c in third.calls], ["openai:three"])
        attempts = first.calls + second.calls + third.calls
        self.assertEqual(len({call["logical_call_id"] for call in attempts}), 1)
        self.assertEqual([call["attempt_index"] for call in attempts], [1, 2, 3])
        usage = router.usage.summary()["by_agent"]["translator"]
        self.assertEqual(usage["logical_calls"], 1)
        self.assertEqual(usage["fallbacks"], 2)

    def test_permanent_error_does_not_switch(self):
        for error_type in (_Http400, _Http401, _Http403):
            with self.subTest(error=error_type):
                first = StubTransport("fake").plan(error_type())
                backup = StubTransport("bailian")
                router = _router(
                    models={
                        "primary": ["fake/one", "bailian/two"],
                        "editor": ["fake/editor"],
                        "fast": ["fake/fast"],
                    },
                    transports={"fake": first, "bailian": backup},
                )
                with self.assertRaises(error_type):
                    router.complete([], agent="translator", operation="op")
                self.assertEqual(backup.calls, [])

    def test_all_fallback_candidates_fail_in_order(self):
        first = StubTransport("fake").plan(TimeoutError("secret one"))
        second = StubTransport("bailian").plan(_StructuredNotFound())
        router = _router(
            models={
                "primary": ["fake/one", "bailian/two"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
            transports={"fake": first, "bailian": second},
        )
        with self.assertRaises(AllModelsFailedError) as caught:
            router.complete([], agent="translator", operation="op")
        self.assertEqual(
            caught.exception.records_data,
            (("fake:one", "timeout"), ("bailian:two", "model_not_found")),
        )
        self.assertNotIn("secret", str(caught.exception))

    def test_generation_options_validate_entire_chain_before_transport(self):
        first = StubTransport("bailian")
        second = StubTransport("bailian")
        router = _router(
            models={
                "primary": ["bailian/qwen3.7-flash:off", "bailian/unknown:off"],
                "editor": ["fake/editor"],
                "fast": ["fake/fast"],
            },
            transports={"bailian": first, "fake": second},
            options=GenerationOptions(require_catalogued_model=True),
        )
        with self.assertRaises(ValueError):
            router.complete([], agent="translator", operation="op")
        self.assertEqual(first.calls, [])
        usage = router.usage.summary()["by_agent"]["translator"]
        self.assertEqual(usage["logical_calls"], 1)
        self.assertEqual(usage["attempts"], 0)

    def test_roles_and_json_route(self):
        transport = StubTransport("fake").plan(json.dumps({"ok": True}))
        router = _router(transports={"fake": transport})
        self.assertEqual(router.complete_json([], agent="editor", operation="op"), {"ok": True})
        ref = transport.calls[0]["model_ref"]
        self.assertEqual(ref.full_name, "fake:editor-id")

    def test_unknown_agent_fails_before_transport(self):
        transport = StubTransport()
        with self.assertRaises(UnknownAgentError):
            _router(transports={"fake": transport}).complete([], agent="missing", operation="op")
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
