"""Versioned values shared by benchmark execution and reporting phases."""

RUN_SCHEMA_VERSION = 3
GENERATION_FIELDS = {
    "temperature": 0.1,
    "seed": None,
    "require_catalogued_model": True,
    "require_thinking_disabled": False,
}

__all__ = ["GENERATION_FIELDS", "RUN_SCHEMA_VERSION"]
