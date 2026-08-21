"""Strict schemas for Phase 8 report specifications."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

__all__ = ["PublicationGates", "ReportSpec", "load_report_spec"]


# Decimal conversion is deliberately narrower than Pydantic's normal Decimal
# coercion.  In particular, accepting floats here would make direct model
# construction differ from the YAML representation (where unquoted floats
# are Python floats).
def _decimal_before(value: object) -> Decimal:
    if isinstance(value, Decimal):
        result = value
    elif type(value) is int:
        result = Decimal(value)
    elif type(value) is str:
        if not value or value != value.strip():
            raise ValueError("decimal strings must be nonblank and have no surrounding whitespace")
        try:
            result = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("invalid decimal string") from exc
    else:
        raise ValueError("decimal values must be Decimal, int, or a decimal string")
    if not result.is_finite():
        raise ValueError("decimal values must be finite")
    return result


_DecimalInput = Annotated[Decimal, BeforeValidator(_decimal_before)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_default=True,
    )


class PublicationGates(_StrictModel):
    """Frozen publication thresholds used by the report integration layer."""

    critical_max: int = Field(default=0, ge=0)
    completion_min: _DecimalInput = Field(default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("1"))
    structure_errors_max: int = Field(default=0, ge=0)
    protocol_errors_max: int = Field(default=0, ge=0)
    required_node_failures_max: int = Field(default=0, ge=0)
    resume_duplicate_operations_max: int = Field(default=0, ge=0)
    reasoning_tokens_max: int = Field(default=0, ge=0)
    major_per_10k_upper95_max: _DecimalInput = Field(
        default=Decimal("1.0"), ge=Decimal("0"), le=Decimal("1")
    )
    per_book_major_per_10k_max: _DecimalInput = Field(default=Decimal("2.0"), ge=Decimal("0"))
    fidelity_mean_min: _DecimalInput = Field(
        default=Decimal("4.3"), ge=Decimal("1"), le=Decimal("5")
    )
    naturalness_mean_min: _DecimalInput = Field(
        default=Decimal("4.0"), ge=Decimal("1"), le=Decimal("5")
    )
    polish_major_semantic_harm_max: int = Field(default=0, ge=0)
    polish_harm_rate_upper95_max: _DecimalInput = Field(
        default=Decimal("0.01"), ge=Decimal("0"), le=Decimal("1")
    )
    krippendorff_alpha_min: _DecimalInput = Field(
        default=Decimal("0.67"), ge=Decimal("0"), le=Decimal("1")
    )


class ReportSpec(_StrictModel):
    """The complete, externally supplied Phase 8 report specification."""

    schema_version: Literal[1]
    benchmark_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    price_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_seed: int
    bootstrap_replicates: int = Field(default=2000, ge=1000, le=10000)
    editor_hourly_rates: list[_DecimalInput] = Field(
        default_factory=lambda: [Decimal("50"), Decimal("100"), Decimal("200")],
        min_length=1,
    )
    gates: PublicationGates = Field(default_factory=PublicationGates)

    @field_validator("benchmark_id", mode="before")
    @classmethod
    def _strip_benchmark_id(cls, value: object) -> object:
        if type(value) is str:
            return value.strip()
        return value

    @field_validator("editor_hourly_rates", mode="before")
    @classmethod
    def _require_editor_rate_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("editor_hourly_rates must be a list")
        return value

    @field_validator("editor_hourly_rates")
    @classmethod
    def _validate_editor_hourly_rates(cls, value: list[Decimal]) -> list[Decimal]:
        if not value:
            raise ValueError("editor_hourly_rates must not be empty")
        if any(rate <= 0 or not rate.is_finite() for rate in value):
            raise ValueError("editor_hourly_rates must contain positive finite values")
        if len(set(value)) != len(value):
            raise ValueError("editor_hourly_rates must contain unique values")
        return sorted(value)


def load_report_spec(path: Path) -> ReportSpec:
    """Load a UTF-8 YAML report specification from a :class:`Path`."""

    if not isinstance(path, Path):
        raise TypeError("load_report_spec requires a pathlib.Path")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return ReportSpec.model_validate(document)
