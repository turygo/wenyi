from __future__ import annotations

import json
import unittest

from trans_novel.benchmark.run.telemetry import CollectingCallTelemetrySink, JsonlCallTelemetrySink
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.llm.usage import has_response_usage, normalize_response_usage


def _attempt() -> CallAttemptTelemetry:
    return CallAttemptTelemetry(
        schema_version=1,
        logical_call_id="a" * 32,
        attempt_index=1,
        started_at="2026-01-01T00:00:00.000Z",
        elapsed_ms=1,
        stage=None,
        agent="translator",
        operation="translate",
        provider="test",
        requested_model="model-a",
        resolved_model="model-a",
        reasoning_enabled=False,
        reasoning_effort="high",
        temperature=None,
        seed=None,
        json_mode=False,
        max_tokens=None,
        status="success",
        retry_class=None,
        http_status=None,
        finish_reason="stop",
        response_id="r1",
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        cache_hit_tokens=4,
        cache_miss_tokens=6,
        reasoning_tokens=0,
        billed_usage_unknown=False,
        request_sha256="0" * 64,
        response_sha256="1" * 64,
    )


class TestBenchmarkTelemetry(unittest.TestCase):
    def test_usage_normalization_direct_and_nested(self) -> None:
        self.assertEqual(
            normalize_response_usage(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 4},
                }
            ),
            {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cache_hit_tokens": 4,
                "cache_miss_tokens": 6,
                "reasoning_tokens": 0,
            },
        )
        self.assertEqual(
            normalize_response_usage(
                {
                    "input_tokens": 20,
                    "output_tokens": 7,
                    "total_tokens": 27,
                    "input_tokens_details": {"cached_tokens": 5},
                    "output_tokens_details": {"reasoning_tokens": 3},
                }
            ),
            {
                "prompt_tokens": 20,
                "completion_tokens": 7,
                "total_tokens": 27,
                "cache_hit_tokens": 5,
                "cache_miss_tokens": 15,
                "reasoning_tokens": 3,
            },
        )
        self.assertTrue(has_response_usage({"input_tokens": 0}))
        self.assertTrue(has_response_usage({"total_tokens": 0}))
        self.assertFalse(has_response_usage({"prompt_tokens_details": {"cached_tokens": 1}}))


def test_collecting_and_jsonl_sinks(tmp_path) -> None:
    attempt = _attempt()
    collecting = CollectingCallTelemetrySink(benchmark_id="b", candidate_id="c", run_id="r")
    collecting.record(attempt)
    assert collecting.records[0]["benchmark_id"] == "b"
    path = tmp_path / "nested" / "calls.jsonl"
    JsonlCallTelemetrySink(path, benchmark_id="b", candidate_id="c", run_id="r").record(attempt)
    assert json.loads(path.read_text(encoding="utf-8"))["request_sha256"] == "0" * 64
