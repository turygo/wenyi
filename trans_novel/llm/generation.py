"""Constructor-injected generation controls for controlled model runs."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Validated request controls independent of public configuration."""

    temperature: float | int | None = None
    seed: int | None = None
    require_catalogued_model: bool = False
    require_thinking_disabled: bool = False

    def __post_init__(self) -> None:
        temperature = self.temperature
        if temperature is not None:
            if isinstance(temperature, bool) or not isinstance(temperature, numbers.Real):
                raise TypeError("temperature must be a finite real number or None")
            if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
                raise ValueError("temperature must be a finite real number from 0.0 to 2.0")

        seed = self.seed
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")

        for name in ("require_catalogued_model", "require_thinking_disabled"):
            value: Any = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
