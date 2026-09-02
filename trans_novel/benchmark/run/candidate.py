"""Candidate configuration, model policy, and client construction."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from trans_novel.benchmark.schema import Candidate, CandidateSpec
from trans_novel.config import Config, LLMConfig, ModelRoles
from trans_novel.llm import GenerationOptions, build_client
from trans_novel.llm.telemetry import CallTelemetrySink
from trans_novel.model_profiles import (
    capabilities_for,
    parse_provider_model,
    validate_model_selection,
)


class CandidateRuntimeError(ValueError):
    """Invalid candidate configuration or unsupported model capability."""


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise CandidateRuntimeError(f"cannot load YAML {source}: {error}") from error
    if not isinstance(value, dict):
        raise CandidateRuntimeError(f"YAML root must be an object: {source}")
    return value


def load_candidate_spec(path: str | os.PathLike[str]) -> CandidateSpec:
    try:
        return CandidateSpec.model_validate(_load_yaml(path))
    except Exception as error:
        raise CandidateRuntimeError(f"invalid CandidateSpec: {error}") from error


def candidate_models(candidate: Candidate) -> tuple[str, str, str, str]:
    return (
        candidate.translator_model,
        candidate.analyst_model,
        candidate.editor_model,
        candidate.fast_model,
    )


def _validate_model(value: str, options: GenerationOptions) -> None:
    try:
        provider, model = parse_provider_model(value)
        selection = validate_model_selection(provider, model)
    except Exception as error:
        raise CandidateRuntimeError(f"invalid model selection {value}: {error}") from error
    capabilities = capabilities_for(provider, selection.model)
    if options.require_catalogued_model and not capabilities.catalogued:
        raise CandidateRuntimeError(f"model is not catalogued: {provider}:{selection.model}")
    if options.require_thinking_disabled and not capabilities.supports_thinking_disabled:
        raise CandidateRuntimeError(
            f"model does not support thinking disabled: {provider}:{selection.model}"
        )
    if options.temperature is not None and not capabilities.supports_temperature:
        raise CandidateRuntimeError(
            f"model does not support temperature: {provider}:{selection.model}"
        )


def validate_candidate_capabilities(spec: CandidateSpec) -> GenerationOptions:
    options = GenerationOptions(
        temperature=spec.temperature,
        seed=spec.seed,
        require_catalogued_model=True,
        require_thinking_disabled=False,
    )
    for candidate in spec.candidates:
        for role in ("translator", "analyst", "editor", "fast"):
            _validate_model(getattr(candidate, f"{role}_model"), options)
    return options


def attach_telemetry_sink(
    client: Any, sink: CallTelemetrySink | None, *, required: bool = False
) -> None:
    if sink is None:
        return
    setter = getattr(client, "set_telemetry_sink", None)
    attached = False
    if callable(setter):
        try:
            setter(sink)
            attached = True
        except (AttributeError, TypeError, ValueError):
            pass
    if not attached and (
        hasattr(client, "telemetry_sink") or "telemetry_sink" in getattr(client, "__dict__", {})
    ):
        try:
            client.telemetry_sink = sink
            attached = getattr(client, "telemetry_sink", None) is sink
        except Exception:
            pass
    if required and not attached:
        raise CandidateRuntimeError(
            "client_factory returned a client without an attachable telemetry sink"
        )


def model_client(
    spec: CandidateSpec,
    model: str,
    role: str,
    options: GenerationOptions,
    factory: Callable[..., Any] | None,
    telemetry_sink: CallTelemetrySink | None = None,
    *,
    roles: ModelRoles | None = None,
) -> Any:
    provider, provider_model = parse_provider_model(model)
    models = roles or ModelRoles(translator=[model], analyst=[model], editor=[model], fast=[model])
    if factory is not None:
        attempts = (
            {
                "provider": provider,
                "model": provider_model,
                "role": role,
                "models": models,
                "generation_options": options,
                "telemetry_sink": telemetry_sink,
            },
            {
                "provider": provider,
                "model": provider_model,
                "role": role,
                "generation_options": options,
                "telemetry_sink": telemetry_sink,
            },
            {
                "provider": provider,
                "model": provider_model,
                "role": role,
                "generation_options": options,
            },
            {"provider": provider, "model": provider_model, "role": role},
        )
        for kwargs in attempts:
            try:
                client = factory(**kwargs)
                attach_telemetry_sink(client, telemetry_sink, required=True)
                return client
            except TypeError:
                continue
        for args in (
            (provider, provider_model, role, options, telemetry_sink),
            (provider, provider_model, role, options),
            (provider_model, role),
            (provider_model,),
        ):
            try:
                client = factory(*args)
                attach_telemetry_sink(client, telemetry_sink, required=True)
                return client
            except TypeError:
                continue
        raise CandidateRuntimeError("client_factory does not accept a supported signature")
    config = Config(llm=LLMConfig(models=models), source_lang="en", target_lang="zh")
    return build_client(config, generation_options=options, telemetry_sink=telemetry_sink)


__all__ = [
    "CandidateRuntimeError",
    "attach_telemetry_sink",
    "candidate_models",
    "load_candidate_spec",
    "model_client",
    "validate_candidate_capabilities",
]
