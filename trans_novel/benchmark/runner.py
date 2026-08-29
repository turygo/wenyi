"""Production-equivalent EPUB benchmark runner.

The benchmark owns isolation, immutable identities, telemetry, and resumability. Translation
itself is delegated to ``Application`` with the same ``Config`` and original EPUB input used
by the production CLI. No benchmark-specific prompt or frozen shared preparation exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from trans_novel.benchmark.corpus import (
    _resolve_books,
    canonical_json,
    load_book_spec,
    segment_id,
    sha256_bytes,
)
from trans_novel.benchmark.schema import Candidate, CandidateSpec
from trans_novel.config import Config, LLMConfig, ModelRoles, PipelineConfig
from trans_novel.ingest.epub_reader import read_epub
from trans_novel.llm import GenerationOptions, build_client
from trans_novel.llm.telemetry import CallAttemptTelemetry, CallTelemetrySink
from trans_novel.llm.usage import usage_delta
from trans_novel.model_profiles import capabilities_for, validate_model_selection
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.runstore import RunStore, clone_closed_runstore

RUN_SCHEMA_VERSION = 3
PROMPT_VERSION = "production-translator-v1"
_GENERATION_FIELDS = {
    "temperature": 0.1,
    "seed": None,
    "require_catalogued_model": True,
    "require_thinking_disabled": False,
}


class BenchmarkError(ValueError):
    """Invalid benchmark input or an integrity failure in a run directory."""


class _JsonlTelemetrySink:
    """Append immutable physical-call telemetry to one candidate/book artifact."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        if path.exists():
            try:
                self.records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as error:
                raise BenchmarkError(f"invalid telemetry artifact {path}: {error}") from error

    def record(self, attempt: CallAttemptTelemetry) -> None:
        value = (
            attempt.model_dump(mode="python") if hasattr(attempt, "model_dump") else dict(attempt)
        )
        self.records.append(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise BenchmarkError(f"invalid JSON artifact {path}: {error}") from error


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except Exception as error:
        raise BenchmarkError(f"cannot load YAML {source}: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"YAML root must be an object: {source}")
    return value


def load_candidate_spec(path: str | os.PathLike[str]) -> CandidateSpec:
    try:
        return CandidateSpec.model_validate(_load_yaml(path))
    except Exception as error:
        raise BenchmarkError(f"invalid CandidateSpec: {error}") from error


def _validate_model(provider: str, value: str, options: GenerationOptions) -> None:
    try:
        selection = validate_model_selection(provider, value)
    except Exception as error:
        raise BenchmarkError(f"invalid model selection {provider}:{value}: {error}") from error
    capabilities = capabilities_for(provider, selection.model)
    if options.require_catalogued_model and not capabilities.catalogued:
        raise BenchmarkError(f"model is not catalogued: {provider}:{selection.model}")
    if options.require_thinking_disabled and not capabilities.supports_thinking_disabled:
        raise BenchmarkError(
            f"model does not support thinking disabled: {provider}:{selection.model}"
        )
    if options.temperature is not None and not capabilities.supports_temperature:
        raise BenchmarkError(f"model does not support temperature: {provider}:{selection.model}")


def validate_candidate_capabilities(spec: CandidateSpec) -> GenerationOptions:
    options = GenerationOptions(
        temperature=spec.temperature,
        seed=spec.seed,
        require_catalogued_model=True,
        require_thinking_disabled=False,
    )
    _validate_model(spec.provider, spec.fast_model, options)
    for candidate in spec.candidates:
        _validate_model(spec.provider, candidate.primary_model, options)
        _validate_model(spec.provider, candidate.editor_model, options)
    return options


def _attach_sink(client: Any, sink: _JsonlTelemetrySink | None, *, required: bool = False) -> None:
    if sink is None:
        return
    setter = getattr(client, "set_telemetry_sink", None)
    attached = False
    if callable(setter):
        try:
            setter(sink)
            attached = True
        except (AttributeError, TypeError, ValueError):
            attached = False
    if not attached and (
        hasattr(client, "telemetry_sink") or "telemetry_sink" in getattr(client, "__dict__", {})
    ):
        try:
            client.telemetry_sink = sink
            attached = getattr(client, "telemetry_sink", None) is sink
        except Exception:
            attached = False
    if required and not attached:
        raise BenchmarkError(
            "client_factory returned a client without an attachable telemetry sink"
        )


def _model_client(
    spec: CandidateSpec,
    model: str,
    role: str,
    options: GenerationOptions,
    factory: Callable[..., Any] | None,
    telemetry_sink: CallTelemetrySink | None = None,
    *,
    roles: ModelRoles | None = None,
) -> Any:
    models = roles or ModelRoles(primary=model, editor=model, fast=spec.fast_model)
    if factory is not None:
        attempts = (
            {
                "provider": spec.provider,
                "model": model,
                "role": role,
                "models": models,
                "generation_options": options,
                "telemetry_sink": telemetry_sink,
            },
            {
                "provider": spec.provider,
                "model": model,
                "role": role,
                "generation_options": options,
                "telemetry_sink": telemetry_sink,
            },
            {
                "provider": spec.provider,
                "model": model,
                "role": role,
                "generation_options": options,
            },
            {"provider": spec.provider, "model": model, "role": role},
        )
        for kwargs in attempts:
            try:
                client = factory(**kwargs)
                _attach_sink(client, telemetry_sink, required=True)
                return client
            except TypeError:
                continue
        for args in (
            (spec.provider, model, role, options, telemetry_sink),
            (spec.provider, model, role, options),
            (model, role),
            (model,),
        ):
            try:
                client = factory(*args)
                _attach_sink(client, telemetry_sink, required=True)
                return client
            except TypeError:
                continue
        raise BenchmarkError("client_factory does not accept a supported signature")
    config = Config(
        llm=LLMConfig(provider=spec.provider, models=models),
        source_lang="en",
        target_lang="zh",
    )
    return build_client(config, generation_options=options, telemetry_sink=telemetry_sink)


def _usage_of(client: Any) -> dict[str, Any]:
    usage = getattr(client, "usage", None)
    return usage.summary() if usage is not None and hasattr(usage, "summary") else {}


def _safe_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _safe_book_id(value: str) -> str:
    safe = "".join(char if (char.isalnum() or char in "-_") else "_" for char in value).strip("_")
    return f"{safe or 'book'}-{_safe_id(value)[:12]}"


def _candidate_store(state_dir: Path) -> RunStore:
    stores = (
        [
            path
            for path in state_dir.iterdir()
            if path.is_dir() and (path / "manifest.json").exists()
        ]
        if state_dir.exists()
        else []
    )
    if len(stores) != 1:
        raise BenchmarkError("candidate Application state root is missing or ambiguous")
    return RunStore(str(stores[0]))


def _target_hash(rows: list[dict[str, Any]]) -> str:
    ordered = [
        {"chapter": row["chapter_index"], "index": row["segment_index"], "target": row["target"]}
        for row in rows
    ]
    return sha256_bytes(canonical_json(ordered).encode("utf-8"))


def _segment_rows(store: RunStore, source_sha256: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    state = store.load_state()
    qa_node = state.nodes.get("deterministic_qa")
    qa_findings = (qa_node.output or {}).get("issues", []) if qa_node else []
    for chapter in state.chapters:
        progress = store.load_progress(chapter.index)
        lint_findings = progress.lint_issues
        for segment in store.load_chapter(chapter.index).text_segments:
            target = segment.target or ""
            if not target.strip():
                raise BenchmarkError("completed candidate contains an empty translation")
            rows.append(
                {
                    "segment_id": segment_id(
                        source_sha256, chapter.index, segment.index, segment.source
                    ),
                    "chapter_index": chapter.index,
                    "chapter_title": chapter.title,
                    "segment_index": segment.index,
                    "kind": segment.kind,
                    "source": segment.source,
                    "target": target,
                    "lint_findings": [
                        item for item in lint_findings if item.get("index") in {None, segment.index}
                    ],
                    "deterministic_findings": [
                        item
                        for item in qa_findings
                        if item.get("chapter") == chapter.index
                        and item.get("index") == segment.index
                    ],
                    "source_sha256": sha256_bytes(segment.source.encode("utf-8")),
                    "target_sha256": sha256_bytes(target.encode("utf-8")),
                }
            )
    if not rows:
        raise BenchmarkError("completed candidate contains no translated segments")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


class FullRunner:
    """Run original formal chapter EPUBs through the production quality pipeline."""

    def __init__(
        self, *, client_factory: Callable[..., Any] | None = None, client: Any | None = None
    ) -> None:
        self.client_factory = client_factory
        self.client = client

    @staticmethod
    def _config(
        spec: CandidateSpec,
        primary: str,
        editor: str,
        *,
        pipeline_variant: str = "minimal",
        state_dir: str,
    ) -> Config:
        preset = "quality" if pipeline_variant == "polish" else "balanced"
        return Config(
            llm=LLMConfig(
                provider=spec.provider,
                models=ModelRoles(primary=primary, editor=editor, fast=spec.fast_model),
            ),
            quality=preset,
            source_lang="en",
            target_lang="zh",
            pipeline=PipelineConfig.for_quality(preset),
            state_dir=state_dir,
        )

    def _client(
        self,
        spec: CandidateSpec,
        candidate: Candidate,
        options: GenerationOptions,
        sink: _JsonlTelemetrySink,
    ) -> Any:
        if self.client is not None:
            _attach_sink(self.client, sink, required=True)
            return self.client
        roles = ModelRoles(
            primary=candidate.primary_model,
            editor=candidate.editor_model,
            fast=spec.fast_model,
        )
        return _model_client(
            spec,
            candidate.primary_model,
            "translator",
            options,
            self.client_factory,
            sink,
            roles=roles,
        )

    @staticmethod
    def _artifact_key(
        spec: CandidateSpec,
        candidate: Candidate,
        book_id: str,
        source_sha256: str,
        replicate: int,
    ) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "schema_version": RUN_SCHEMA_VERSION,
                    "benchmark_id": spec.benchmark_id,
                    "provider": spec.provider,
                    "candidate": candidate.model_dump(mode="python"),
                    "fast_model": spec.fast_model,
                    "generation": _GENERATION_FIELDS,
                    "pipeline_variant": candidate.pipeline_variant,
                    "quality": candidate.pipeline_variant,
                    "source_sha256": source_sha256,
                    "replicate": replicate,
                }
            ).encode("utf-8")
        )

    def _validate_artifact(
        self,
        out: Path,
        row: dict[str, Any],
        *,
        spec: CandidateSpec,
        candidate: Candidate,
        book_id: str,
        source_sha256: str,
        replicate: int,
        minimal_row: dict[str, Any] | None = None,
    ) -> None:
        required = {
            "artifact_key",
            "candidate_id",
            "pipeline_variant",
            "book_id",
            "replicate",
            "source_sha256",
            "state_path",
            "segments_path",
            "segments_sha256",
            "telemetry_path",
            "telemetry_sha256",
            "outputs",
            "output_hashes",
            "initial_targets_sha256",
            "final_targets_sha256",
            "usage",
            "polish_incremental_usage",
            "provider",
            "primary_model",
            "editor_model",
            "fast_model",
        }
        if set(row) != required:
            raise BenchmarkError("candidate artifact fields invalid")
        expected_key = self._artifact_key(spec, candidate, book_id, source_sha256, replicate)
        expected_values = {
            "artifact_key": expected_key,
            "candidate_id": candidate.candidate_id,
            "pipeline_variant": candidate.pipeline_variant,
            "book_id": book_id,
            "replicate": replicate,
            "source_sha256": source_sha256,
            "provider": spec.provider,
            "primary_model": candidate.primary_model,
            "editor_model": candidate.editor_model,
            "fast_model": spec.fast_model,
        }
        if any(row.get(field) != value for field, value in expected_values.items()):
            raise BenchmarkError("candidate artifact identity mismatch")
        for field, digest_field in (
            ("segments_path", "segments_sha256"),
            ("telemetry_path", "telemetry_sha256"),
        ):
            path = out / row[field]
            if not path.is_file() or sha256_bytes(path.read_bytes()) != row[digest_field]:
                raise BenchmarkError(f"candidate artifact {field} is missing or changed")
        segments_path = out / row["segments_path"]
        try:
            segment_rows = [
                json.loads(line)
                for line in segments_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as error:
            raise BenchmarkError("candidate segments artifact is invalid") from error
        if _target_hash(segment_rows) != row["final_targets_sha256"]:
            raise BenchmarkError("candidate final target hash mismatch")
        if minimal_row is None:
            if row["initial_targets_sha256"] != row["final_targets_sha256"]:
                raise BenchmarkError("minimal candidate initial target hash mismatch")
        elif row["initial_targets_sha256"] != minimal_row["final_targets_sha256"]:
            raise BenchmarkError("polish candidate initial target hash mismatch")
        state_root = out / row["state_path"]
        store = _candidate_store(state_root)
        usage = store.load_usage() or {}
        if usage != row["usage"]:
            raise BenchmarkError("candidate usage metadata mismatch")
        expected_increment = (
            usage_delta(usage, minimal_row["usage"])
            if minimal_row is not None
            else usage_delta({}, {})
        )
        if row["polish_incremental_usage"] != expected_increment:
            raise BenchmarkError("candidate polish usage metadata mismatch")
        for relative, digest in row["output_hashes"].items():
            path = out / relative
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                raise BenchmarkError("candidate output is missing or changed")

    def _run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        mode: str,
        selected_books: list[tuple[Any, Path, bytes]],
        replicate_count: int,
    ) -> dict[str, Any]:
        source_spec_path = Path(book_spec_path).expanduser().resolve()
        candidate_spec_path = Path(candidates_path).expanduser().resolve()
        spec = load_candidate_spec(candidate_spec_path)
        options = validate_candidate_capabilities(spec)
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        immutable = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_mode": mode,
            "benchmark_id": spec.benchmark_id,
            "book_spec_sha256": sha256_bytes(source_spec_path.read_bytes()),
            "candidate_spec_sha256": sha256_bytes(
                canonical_json(spec.model_dump(mode="python")).encode("utf-8")
            ),
            "generation": dict(_GENERATION_FIELDS),
            "pipeline_variants": {
                candidate.candidate_id: candidate.pipeline_variant for candidate in spec.candidates
            },
            "book_ids": [book.book_id for book, _, _ in selected_books],
            "replicates": replicate_count,
        }
        run_path = out / "run.json"
        if run_path.exists() and _read_json(run_path) != immutable:
            raise BenchmarkError("immutable benchmark run identity mismatch")
        if not run_path.exists():
            _atomic_json(run_path, immutable)
        state_path = out / "run_state.json"
        state = (
            _read_json(state_path)
            if state_path.exists()
            else {"schema_version": 1, "status": "pending", "artifacts": {}}
        )
        rows_path = out / "candidates.json"
        existing_rows = _read_json(rows_path) if rows_path.exists() else []
        if not isinstance(existing_rows, list):
            raise BenchmarkError("candidate artifact index must be a list")
        by_key = {row["artifact_key"]: row for row in existing_rows}
        expected = len(selected_books) * len(spec.candidates) * replicate_count
        candidate_by_id = {candidate.candidate_id: candidate for candidate in spec.candidates}
        book_by_id = {
            book.book_id: (book, sha256_bytes(source_bytes))
            for book, _source_path, source_bytes in selected_books
        }
        ordered_candidates = sorted(
            spec.candidates,
            key=lambda candidate: (candidate.pipeline_variant != "minimal", candidate.candidate_id),
        )
        minimal_by_pair = {
            (candidate.primary_model, candidate.editor_model): candidate
            for candidate in ordered_candidates
            if candidate.pipeline_variant == "minimal"
        }
        if state.get("status") == "completed":
            if len(by_key) != expected:
                raise BenchmarkError("completed benchmark candidate count mismatch")
            for row in by_key.values():
                candidate = candidate_by_id.get(row.get("candidate_id"))
                book_id = row.get("book_id")
                if candidate is None or book_id not in book_by_id:
                    raise BenchmarkError("completed benchmark candidate identity mismatch")
                minimal_row = None
                if candidate.pipeline_variant == "polish":
                    minimal = minimal_by_pair.get((candidate.primary_model, candidate.editor_model))
                    minimal_row = next(
                        (
                            value
                            for value in by_key.values()
                            if value.get("candidate_id")
                            == (minimal.candidate_id if minimal else None)
                            and value.get("book_id") == book_id
                            and value.get("replicate") == row.get("replicate")
                        ),
                        None,
                    )
                    if minimal_row is None:
                        raise BenchmarkError("polish candidate is missing its minimal arm")
                self._validate_artifact(
                    out,
                    row,
                    spec=spec,
                    candidate=candidate,
                    book_id=book_id,
                    source_sha256=book_by_id[book_id][1],
                    replicate=row.get("replicate"),
                    minimal_row=minimal_row,
                )
            return immutable | {"status": "completed", "branch_count": len(by_key)}
        _atomic_json(
            state_path,
            {"schema_version": 1, "status": "running", "artifacts": state.get("artifacts", {})},
        )
        try:
            for book, source_path, source_bytes in selected_books:
                source_sha = sha256_bytes(source_bytes)
                for replicate in range(1, replicate_count + 1):
                    for candidate in ordered_candidates:
                        key = self._artifact_key(
                            spec, candidate, book.book_id, source_sha, replicate
                        )
                        if key in by_key:
                            minimal_row = None
                            if candidate.pipeline_variant == "polish":
                                minimal = minimal_by_pair.get(
                                    (candidate.primary_model, candidate.editor_model)
                                )
                                minimal_row = next(
                                    (
                                        value
                                        for value in by_key.values()
                                        if minimal is not None
                                        and value.get("candidate_id") == minimal.candidate_id
                                        and value.get("book_id") == book.book_id
                                        and value.get("replicate") == replicate
                                    ),
                                    None,
                                )
                                if minimal_row is None:
                                    raise BenchmarkError(
                                        "polish candidate is missing its minimal arm"
                                    )
                            self._validate_artifact(
                                out,
                                by_key[key],
                                spec=spec,
                                candidate=candidate,
                                book_id=book.book_id,
                                source_sha256=source_sha,
                                replicate=replicate,
                                minimal_row=minimal_row,
                            )
                            continue
                        root = (
                            out
                            / "candidates"
                            / candidate.candidate_id
                            / _safe_book_id(book.book_id)
                            / f"r{replicate}"
                        )
                        state_root = root / "state"
                        minimal_row = None
                        if candidate.pipeline_variant == "polish":
                            minimal = minimal_by_pair.get(
                                (candidate.primary_model, candidate.editor_model)
                            )
                            if minimal is None:
                                raise BenchmarkError(
                                    "polish candidate requires a matching minimal candidate"
                                )
                            minimal_key = self._artifact_key(
                                spec, minimal, book.book_id, source_sha, replicate
                            )
                            minimal_row = by_key.get(minimal_key)
                            if minimal_row is None:
                                raise BenchmarkError(
                                    "minimal arm must complete before polish arm is created"
                                )
                            self._validate_artifact(
                                out,
                                minimal_row,
                                spec=spec,
                                candidate=minimal,
                                book_id=book.book_id,
                                source_sha256=source_sha,
                                replicate=replicate,
                            )
                            minimal_root = (
                                out
                                / "candidates"
                                / minimal.candidate_id
                                / _safe_book_id(book.book_id)
                                / f"r{replicate}"
                                / "state"
                            )
                            minimal_store = _candidate_store(minimal_root)
                            destination_store = state_root / Path(minimal_store.run_dir).name
                            if (
                                destination_store.is_dir()
                                and (destination_store / "manifest.json").is_file()
                            ):
                                cloned = _candidate_store(state_root)
                            else:
                                state_root.mkdir(parents=True, exist_ok=True)
                                clone_closed_runstore(minimal_store.run_dir, str(destination_store))
                                cloned = RunStore(str(destination_store))
                                cloned_state = cloned.load_state()
                                for node_key in list(cloned_state.nodes):
                                    if node_key.split(":", 1)[0] in {
                                        "polish",
                                        "titles",
                                        "deterministic_qa",
                                        "report",
                                        "assemble",
                                    }:
                                        from trans_novel.pipeline.state import NodeState

                                        cloned_state.nodes[node_key] = NodeState(node_id=node_key)
                                cloned.save_state(cloned_state)
                        telemetry_path = root / "telemetry.jsonl"
                        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                        telemetry_path.touch(exist_ok=True)
                        sink = _JsonlTelemetrySink(telemetry_path)
                        if candidate.pipeline_variant == "polish":
                            minimal_segments = out / minimal_row["segments_path"]
                            initial_rows = [
                                json.loads(line)
                                for line in minimal_segments.read_text(
                                    encoding="utf-8"
                                ).splitlines()
                                if line.strip()
                            ]
                            usage_before = minimal_row["usage"]
                        else:
                            initial_rows = None
                            usage_before = {}
                        client = self._client(spec, candidate, options, sink)
                        config = self._config(
                            spec,
                            candidate.primary_model,
                            candidate.editor_model,
                            pipeline_variant=candidate.pipeline_variant,
                            state_dir=str(state_root),
                        )
                        output_path = root / "outputs" / f"{book.book_id}.epub"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        result = Application(config, client=client).run_all(
                            str(source_path), out_format="epub", out_path=str(output_path)
                        )
                        store = _candidate_store(state_root)
                        final_rows = _segment_rows(store, source_sha)
                        if initial_rows is None:
                            initial_rows = final_rows
                        segments_path = root / "segments.jsonl"
                        _write_jsonl(segments_path, final_rows)
                        outputs = [
                            Path(value).expanduser().resolve()
                            for value in result.get("outputs", [])
                            if isinstance(value, str)
                        ]
                        if not outputs or any(not path.is_file() for path in outputs):
                            raise BenchmarkError("production benchmark did not create output EPUBs")
                        relative_outputs = [str(path.relative_to(out)) for path in outputs]
                        usage = store.load_usage() or {}
                        row = {
                            "artifact_key": key,
                            "candidate_id": candidate.candidate_id,
                            "pipeline_variant": candidate.pipeline_variant,
                            "book_id": book.book_id,
                            "replicate": replicate,
                            "source_sha256": source_sha,
                            "state_path": str(state_root.relative_to(out)),
                            "segments_path": str(segments_path.relative_to(out)),
                            "segments_sha256": sha256_bytes(segments_path.read_bytes()),
                            "telemetry_path": str(telemetry_path.relative_to(out)),
                            "telemetry_sha256": sha256_bytes(telemetry_path.read_bytes()),
                            "outputs": relative_outputs,
                            "output_hashes": {
                                relative: sha256_bytes((out / relative).read_bytes())
                                for relative in relative_outputs
                            },
                            "initial_targets_sha256": _target_hash(initial_rows),
                            "final_targets_sha256": _target_hash(final_rows),
                            "usage": usage,
                            "polish_incremental_usage": (
                                usage_delta(usage, usage_before)
                                if candidate.pipeline_variant == "polish"
                                else usage_delta({}, {})
                            ),
                            "provider": spec.provider,
                            "primary_model": candidate.primary_model,
                            "editor_model": candidate.editor_model,
                            "fast_model": spec.fast_model,
                        }
                        self._validate_artifact(
                            out,
                            row,
                            spec=spec,
                            candidate=candidate,
                            book_id=book.book_id,
                            source_sha256=source_sha,
                            replicate=replicate,
                            minimal_row=minimal_row,
                        )
                        by_key[key] = row
                        _atomic_json(
                            rows_path, sorted(by_key.values(), key=lambda x: x["artifact_key"])
                        )
                        artifacts = dict(state.get("artifacts", {}))
                        artifacts[key] = {
                            "candidate_id": candidate.candidate_id,
                            "pipeline_variant": candidate.pipeline_variant,
                            "book_id": book.book_id,
                            "replicate": replicate,
                        }
                        state = {"schema_version": 1, "status": "running", "artifacts": artifacts}
                        _atomic_json(state_path, state)
            if len(by_key) != expected:
                raise BenchmarkError("benchmark candidate count mismatch")
            completed = {
                "schema_version": 1,
                "status": "completed",
                "artifacts": state.get("artifacts", {}),
            }
            _atomic_json(state_path, completed)
            return immutable | {"status": "completed", "branch_count": len(by_key)}
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "artifacts": state.get("artifacts", {}),
                    "error": str(error),
                },
            )
            if isinstance(error, BenchmarkError):
                raise
            raise BenchmarkError(str(error)) from error

    def run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        book_spec = load_book_spec(book_spec_path)
        resolved = _resolve_books(book_spec, Path(book_spec_path).expanduser().resolve())
        formal = [(book, path, data) for book, path, data in resolved if book.split == "formal"]
        if len(formal) != 6:
            raise BenchmarkError("full benchmark requires exactly six formal chapter EPUBs")
        for book, path, _data in formal:
            document = read_epub(
                str(path),
                source_lang=book_spec.source_language,
                target_lang=book_spec.target_language,
            )
            if len(document.chapters) != 1:
                raise BenchmarkError(
                    f"{book.book_id} must contain exactly one chapter; "
                    f"found {len(document.chapters)}"
                )
        return self._run(
            book_spec_path,
            candidates_path,
            out_dir,
            mode="full",
            selected_books=formal,
            replicate_count=load_candidate_spec(candidates_path).replicates,
        )


class CanaryRunner(FullRunner):
    """Run one real screen chapter EPUB through the same production quality pipeline."""

    def run(
        self,
        book_spec_path: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        book_id: str | None = None,
    ) -> dict[str, Any]:
        book_spec = load_book_spec(book_spec_path)
        resolved = _resolve_books(book_spec, Path(book_spec_path).expanduser().resolve())
        screen = [(book, path, data) for book, path, data in resolved if book.split == "screen"]
        if book_id is not None:
            screen = [row for row in screen if row[0].book_id == book_id]
        if not screen:
            raise BenchmarkError("canary requires a matching screen chapter EPUB")
        return self._run(
            book_spec_path,
            candidates_path,
            out_dir,
            mode="canary",
            selected_books=[sorted(screen, key=lambda row: row[0].book_id)[0]],
            replicate_count=1,
        )


__all__ = [
    "PROMPT_VERSION",
    "BenchmarkError",
    "CanaryRunner",
    "FullRunner",
    "load_candidate_spec",
    "validate_candidate_capabilities",
]
