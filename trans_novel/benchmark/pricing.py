"""Strict immutable price snapshots and deterministic usage costing."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_HEX64 = r"^[0-9a-f]{64}$"
_PRICE_FIELDS = ("input_uncached_per_million", "input_cached_per_million", "output_per_million")

_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")


def _valid_source_hostname(hostname: str, netloc: str) -> bool:
    try:
        host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if ":" in host:
            return False
        if host.endswith("."):
            host = host[:-1]
        if not host or len(host) > 253:
            return False
        labels = host.split(".")
        return all(0 < len(label) <= 63 and _DNS_LABEL.fullmatch(label) for label in labels)
    if isinstance(address, ipaddress.IPv6Address):
        return netloc.rsplit("@", 1)[-1].startswith("[")
    return True


def _strict_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must not be blank")
    return value


def _decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite nonnegative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"{name} must be a finite nonnegative decimal") from None
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    return result


class PricingRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_prompt_tokens: int = 0
    max_prompt_tokens: int | None = None
    time_band: str = "all"
    input_uncached_per_million: Decimal
    input_cached_per_million: Decimal
    output_per_million: Decimal

    @field_validator("min_prompt_tokens", "max_prompt_tokens", mode="before")
    @classmethod
    def _strict_int(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("token tier bounds must be exact integers")
        return value

    @field_validator("min_prompt_tokens")
    @classmethod
    def _min_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("min_prompt_tokens must be nonnegative")
        return value

    @field_validator("max_prompt_tokens")
    @classmethod
    def _max_valid(cls, value: int | None, info: Any) -> int | None:
        if value is not None and value < info.data.get("min_prompt_tokens", 0):
            raise ValueError("max_prompt_tokens must be >= min_prompt_tokens")
        return value

    @field_validator("time_band", mode="before")
    @classmethod
    def _band(cls, value: object) -> str:
        return _strict_text(value, "time_band")

    @field_validator(*_PRICE_FIELDS, mode="before")
    @classmethod
    def _price(cls, value: object, info: Any) -> Decimal:
        return _decimal(value, info.field_name)


class ModelPricing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    rules: list[PricingRule]

    @field_validator("model_id", mode="before")
    @classmethod
    def _model_id(cls, value: object) -> str:
        return _strict_text(value, "model_id")

    @field_validator("rules", mode="before")
    @classmethod
    def _rules_shape(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("rules must be a list")
        if not value:
            raise ValueError("rules must be a nonempty list")
        return value

    @field_validator("rules")
    @classmethod
    def _rules_nonempty(cls, value: list[PricingRule]) -> list[PricingRule]:
        if not value:
            raise ValueError("rules must be a nonempty list")
        return value

    @model_validator(mode="after")
    def _validate_tiers(self) -> ModelPricing:
        by_band: dict[str, list[PricingRule]] = {}
        for rule in self.rules:
            by_band.setdefault(rule.time_band, []).append(rule)
        for band, rules in by_band.items():
            expected = 0
            for rule in rules:
                if rule.min_prompt_tokens != expected:
                    raise ValueError(f"pricing tier gap/overlap in time_band {band!r}")
                if rule.max_prompt_tokens is None:
                    if rule is not rules[-1]:
                        raise ValueError("unbounded pricing tier must be final")
                    expected = -1
                else:
                    expected = rule.max_prompt_tokens + 1
            # A finite final tier is valid, but usage beyond it must fail closed.
        return self


class PriceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    provider: str
    region: str
    currency: str
    retrieved_at: datetime
    source_urls: list[str]
    models: dict[str, ModelPricing]

    @field_validator("provider", "region", "currency", mode="before")
    @classmethod
    def _text(cls, value: object, info: Any) -> str:
        return _strict_text(value, info.field_name)

    @field_validator("retrieved_at", mode="after")
    @classmethod
    def _utc_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include timezone")
        return value.astimezone(timezone.utc)

    @field_validator("source_urls", mode="before")
    @classmethod
    def _urls_shape(cls, value: object) -> object:
        if not isinstance(value, list):
            raise TypeError("source_urls must be a list")
        if not value:
            raise ValueError("source_urls must be a nonempty list")
        return value

    @field_validator("source_urls")
    @classmethod
    def _urls(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError("source_urls must be a nonempty list")
        result: list[str] = []
        for raw in value:
            url = _strict_text(raw, "source URL")
            if any(ord(char) <= 0x20 or ord(char) == 0x7F for char in url):
                raise ValueError("source URLs must not contain whitespace or control characters")
            try:
                parsed = urlsplit(url)
                hostname = parsed.hostname
                username = parsed.username
                password = parsed.password
            except ValueError:
                raise ValueError("source URLs must be valid HTTPS URLs") from None
            if (
                parsed.scheme != "https"
                or not hostname
                or username is not None
                or password is not None
                or not _valid_source_hostname(hostname, parsed.netloc)
            ):
                raise ValueError("source URLs must be HTTPS URLs with a valid hostname")
            try:
                _ = parsed.port
            except ValueError:
                raise ValueError("source URL has invalid port") from None
            result.append(url)
        return result

    @field_validator("models", mode="before")
    @classmethod
    def _models_shape(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise TypeError("models must be a mapping")
        if not value:
            raise ValueError("models must be a nonempty mapping")
        return value

    @field_validator("models")
    @classmethod
    def _models(cls, value: dict[str, ModelPricing]) -> dict[str, ModelPricing]:
        if not isinstance(value, dict) or not value:
            raise ValueError("models must be a nonempty mapping")
        for key, model in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("model keys must be nonblank strings")
            if key != model.model_id:
                raise ValueError("model map key must equal nested model_id")
        return value


def _snapshot_hash(snapshot: PriceSnapshot) -> str:
    payload = snapshot.model_dump(mode="json")
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class CostQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uncached_input_cost: Decimal
    cached_input_cost: Decimal
    output_cost: Decimal
    total_cost: Decimal
    currency: str
    model_id: str
    time_band: str
    snapshot_sha256: str
    min_prompt_tokens: int
    max_prompt_tokens: int | None

    @field_validator("min_prompt_tokens", "max_prompt_tokens", mode="before")
    @classmethod
    def _strict_tier_int(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("token tier bounds must be exact integers")
        return value

    @field_validator("min_prompt_tokens")
    @classmethod
    def _tier_min_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("min_prompt_tokens must be nonnegative")
        return value

    @field_validator("max_prompt_tokens")
    @classmethod
    def _tier_max_valid(cls, value: int | None, info: Any) -> int | None:
        if value is not None and value < info.data.get("min_prompt_tokens", 0):
            raise ValueError("max_prompt_tokens must be >= min_prompt_tokens")
        return value

    @field_validator("currency", "model_id", "time_band", mode="before")
    @classmethod
    def _text(cls, value: object, info: Any) -> str:
        return _strict_text(value, info.field_name)

    @field_validator(
        "uncached_input_cost", "cached_input_cost", "output_cost", "total_cost", mode="before"
    )
    @classmethod
    def _cost_decimal(cls, value: object, info: Any) -> Decimal:
        return _decimal(value, info.field_name)

    @field_validator("snapshot_sha256", mode="before")
    @classmethod
    def _hash(cls, value: object) -> str:
        import re

        value = _strict_text(value, "snapshot_sha256")
        if not re.fullmatch(_HEX64, value):
            raise ValueError("snapshot_sha256 must be lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def _total(self) -> CostQuote:
        if self.total_cost != self.uncached_input_cost + self.cached_input_cost + self.output_cost:
            raise ValueError("total_cost must equal component cost sum")
        return self


class UnknownCost(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason: Literal["billed_usage_unknown"]
    snapshot_sha256: str

    @field_validator("snapshot_sha256", mode="before")
    @classmethod
    def _hash(cls, value: object) -> str:
        import re

        value = _strict_text(value, "snapshot_sha256")
        if not re.fullmatch(_HEX64, value):
            raise ValueError("snapshot_sha256 must be lowercase SHA-256")
        return value


def _cost_bounds(count: int, rate: Decimal) -> tuple[int, int]:
    """Return highest/lowest decimal positions for count * rate / 1e6."""
    if count == 0 or rate.is_zero():
        return 0, 0
    digits = len(str(count)) + len(rate.as_tuple().digits) - 1
    high = rate.as_tuple().exponent + digits - 1 - 6
    low = rate.as_tuple().exponent - 6
    return high, low


def load_price_snapshot(path: str | Path) -> PriceSnapshot:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("price snapshot top level must be a mapping")
    return PriceSnapshot.model_validate(value)


def _usage_count(usage: Mapping[str, int], name: str) -> int:
    value = usage.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"usage field {name!r} must be a nonnegative exact int")
    return value


def quote_usage(
    snapshot: PriceSnapshot,
    model_id: str,
    usage: Mapping[str, int],
    *,
    time_band: str = "all",
    billed_usage_unknown: bool = False,
) -> CostQuote | UnknownCost:
    digest = _snapshot_hash(snapshot)
    if billed_usage_unknown:
        return UnknownCost(reason="billed_usage_unknown", snapshot_sha256=digest)
    if not isinstance(usage, Mapping):
        raise TypeError("usage must be a mapping")
    prompt = _usage_count(usage, "prompt_tokens")
    completion = _usage_count(usage, "completion_tokens")
    _usage_count(usage, "total_tokens")
    cache_hit = _usage_count(usage, "cache_hit_tokens")
    _usage_count(usage, "cache_miss_tokens")
    _usage_count(usage, "reasoning_tokens")
    if cache_hit > prompt:
        raise ValueError("cache_hit_tokens cannot exceed prompt_tokens")
    try:
        model = snapshot.models[model_id]
    except KeyError:
        raise ValueError(f"no pricing for model {model_id!r}") from None
    rules = [rule for rule in model.rules if rule.time_band == time_band]
    matches = [
        rule
        for rule in rules
        if prompt >= rule.min_prompt_tokens
        and (rule.max_prompt_tokens is None or prompt <= rule.max_prompt_tokens)
    ]
    if len(matches) != 1:
        raise ValueError("usage does not match exactly one pricing tier")
    rule = matches[0]
    uncached = prompt - cache_hit
    bounds = (
        _cost_bounds(uncached, rule.input_uncached_per_million),
        _cost_bounds(cache_hit, rule.input_cached_per_million),
        _cost_bounds(completion, rule.output_per_million),
    )
    precision = max(high for high, _ in bounds) - min(low for _, low in bounds) + 2
    with localcontext() as context:
        context.prec = max(2, precision)
        uncached_cost = Decimal(uncached) * rule.input_uncached_per_million / Decimal(1_000_000)
        cached_cost = Decimal(cache_hit) * rule.input_cached_per_million / Decimal(1_000_000)
        output_cost = Decimal(completion) * rule.output_per_million / Decimal(1_000_000)
        total_cost = uncached_cost + cached_cost + output_cost
        return CostQuote(
            uncached_input_cost=uncached_cost,
            cached_input_cost=cached_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            currency=snapshot.currency,
            model_id=model_id,
            time_band=time_band,
            snapshot_sha256=digest,
            min_prompt_tokens=rule.min_prompt_tokens,
            max_prompt_tokens=rule.max_prompt_tokens,
        )


__all__ = [
    "CostQuote",
    "ModelPricing",
    "PriceSnapshot",
    "PricingRule",
    "UnknownCost",
    "load_price_snapshot",
    "quote_usage",
]
