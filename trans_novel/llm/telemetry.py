"""Optional, content-safe telemetry for physical provider attempts."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CallAttemptTelemetry(BaseModel):
    """Immutable metadata for one physical provider attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    logical_call_id: str
    attempt_index: int = Field(ge=1)
    started_at: str
    elapsed_ms: int = Field(ge=0)
    stage: str | None
    agent: str
    operation: str
    provider: str
    requested_model: str
    resolved_model: str | None
    reasoning_enabled: bool
    reasoning_effort: str | None
    temperature: float | None
    seed: int | None
    json_mode: bool
    max_tokens: int | None = Field(default=None, ge=0)
    status: Literal["success", "error", "empty_response"]
    retry_class: str | None
    http_status: int | None
    finish_reason: str | None
    response_id: str | None
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_hit_tokens: int = Field(ge=0)
    cache_miss_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    billed_usage_unknown: bool
    request_sha256: str
    response_sha256: str | None

    @field_validator(
        "logical_call_id",
        "agent",
        "operation",
        "provider",
        "requested_model",
        "started_at",
        mode="before",
    )
    @classmethod
    def _required_string(cls, value: object) -> str:
        if not isinstance(value, str):
            raise TypeError("must be a string")
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")

        return value

    @field_validator("started_at")
    @classmethod
    def _utc_timestamp(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("started_at must be a UTC ISO timestamp ending in Z")
        try:
            parsed = datetime.fromisoformat(value[:-1] + "+00:00")
        except ValueError:
            raise ValueError("started_at must be a UTC ISO timestamp ending in Z") from None
        if parsed.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("started_at must be a UTC ISO timestamp ending in Z")
        return value

    @field_validator(
        "stage",
        "resolved_model",
        "reasoning_effort",
        "retry_class",
        "finish_reason",
        "response_id",
        mode="before",
    )
    @classmethod
    def _optional_string(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("must be a string or None")
        value = value.strip()
        return value or None

    @field_validator("request_sha256", "response_sha256", mode="before")
    @classmethod
    def _hash(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ValueError("must be a lowercase SHA-256 hex digest")
        return value

    @field_validator("temperature", mode="before")
    @classmethod
    def _temperature(cls, value: object) -> float | None:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("temperature must be finite")
        return float(value)

    @field_validator("seed", "max_tokens", "http_status", mode="before")
    @classmethod
    def _strict_int(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("must be an int or None")
        return value

    @field_validator(
        "attempt_index",
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_hit_tokens",
        "cache_miss_tokens",
        "reasoning_tokens",
        mode="before",
    )
    @classmethod
    def _strict_token_int(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("must be an exact int")
        return value


@runtime_checkable
class CallTelemetrySink(Protocol):
    def record(self, attempt: CallAttemptTelemetry) -> None: ...


__all__ = ["CallAttemptTelemetry", "CallTelemetrySink"]
