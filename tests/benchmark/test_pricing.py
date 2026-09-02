from __future__ import annotations

from decimal import Decimal

from trans_novel.benchmark.pricing import PriceSnapshot, quote_usage


def snapshot() -> PriceSnapshot:
    return PriceSnapshot.model_validate(
        {
            "schema_version": 1,
            "provider": "test",
            "region": "global",
            "currency": "USD",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "source_urls": ["https://example.com/prices"],
            "models": {
                "model-a": {
                    "model_id": "model-a",
                    "rules": [
                        {
                            "min_prompt_tokens": 0,
                            "max_prompt_tokens": None,
                            "time_band": "all",
                            "input_uncached_per_million": "1",
                            "input_cached_per_million": "0.5",
                            "output_per_million": "2",
                        }
                    ],
                }
            },
        }
    )


def test_decimal_quote_and_unknown() -> None:
    quote = quote_usage(
        snapshot(), "model-a", {"prompt_tokens": 10, "completion_tokens": 2, "cache_hit_tokens": 4}
    )
    assert quote.total_cost == Decimal("0.000012")
    unknown = quote_usage(snapshot(), "model-a", {}, billed_usage_unknown=True)
    assert unknown.reason == "billed_usage_unknown"
