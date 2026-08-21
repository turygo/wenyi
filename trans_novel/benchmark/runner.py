"""Offline, reproducible layer-one translation/polish attribution runner."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from trans_novel.agents import prompts
from trans_novel.agents.polisher import Polisher
from trans_novel.agents.prompts import TranslationContextBundle
from trans_novel.assemble.translator import Translator
from trans_novel.benchmark.corpus import (
    canonical_json,
    load_book_spec,
    sha256_bytes,
    validate_corpus,
)
from trans_novel.benchmark.schema import (
    BookPreparation,
    CandidateSpec,
    ChapterSourceDigest,
    EmittedRunnerRecord,
    GlossaryPreparation,
    PreparationBundle,
    PreparationSpec,
)
from trans_novel.config import Config, LLMConfig, ModelRoles, PipelineConfig
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.ingest.models import Chapter
from trans_novel.ingest.segmenter import batch_segments
from trans_novel.llm import GenerationOptions, build_client
from trans_novel.llm.telemetry import CallAttemptTelemetry, CallTelemetrySink
from trans_novel.llm.usage import merge_usage_summaries, usage_delta
from trans_novel.model_profiles import capabilities_for, validate_model_selection
from trans_novel.pipeline import lint
from trans_novel.pipeline.context import RollingContext
from trans_novel.pipeline.frozen import FrozenBookPreparation, FrozenPreparationMap
from trans_novel.pipeline.runstore import RunStore, clone_closed_runstore
from trans_novel.pipeline.state import RunState

PROMPT_VERSION = "benchmark-context-v1"
RUN_SCHEMA_VERSION = 1
_GENERATION_FIELDS = {
    "temperature": 0.1,
    "seed": None,
    "require_catalogued_model": True,
    "require_thinking_disabled": True,
}


class BenchmarkError(ValueError):
    """Invalid benchmark input or an integrity failure in a run directory."""


class _JsonlTelemetrySink:
    """A real collecting sink shared by a client and its artifact file."""

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
            stream.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as error:
        raise BenchmarkError(f"invalid JSON artifact {path}: {error}") from error


def _load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception as error:
        raise BenchmarkError(f"cannot read benchmark YAML: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkError("benchmark YAML root must be a mapping")
    return value


def load_candidate_spec(path: str | os.PathLike[str]) -> CandidateSpec:
    try:
        return CandidateSpec.model_validate(_load_yaml(path))
    except Exception as error:
        raise BenchmarkError(f"invalid CandidateSpec: {error}") from error


def load_preparation_bundle(path: str | os.PathLike[str]) -> tuple[PreparationBundle, str]:
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "preparation.json"
    try:
        bundle = PreparationBundle.model_validate(_read_json(source))
    except Exception as error:
        raise BenchmarkError(f"invalid PreparationBundle: {error}") from error
    expected_spec = sha256_bytes(
        canonical_json(bundle.preparation_spec.model_dump(mode="python")).encode("utf-8")
    )
    if expected_spec != bundle.preparation_spec_sha256:
        raise BenchmarkError("preparation spec hash mismatch")
    expected_semantic = _preparation_hash(bundle)
    if expected_semantic != bundle.preparation_sha256:
        raise BenchmarkError("preparation semantic hash mismatch")
    return bundle, bundle.preparation_sha256


def load_preparation_spec(path: str | os.PathLike[str]) -> PreparationSpec:
    try:
        raw = _load_yaml(path)
        return PreparationSpec.model_validate(raw)
    except Exception as error:
        raise BenchmarkError(f"invalid PreparationSpec: {error}") from error


def _preparation_semantics(bundle: PreparationBundle) -> dict[str, Any]:
    books: list[dict[str, Any]] = []
    for key, book in sorted(bundle.books.items()):
        value = book.model_dump(mode="python")
        # Usage, telemetry references and paths are physical evidence, not
        # semantic preparation inputs. They must never perturb the shared hash.
        for field in ("usage", "telemetry_sha256", "telemetry_path"):
            value.pop(field, None)
        books.append({"book_id": key, **value})
    return {
        "schema_version": bundle.schema_version,
        "corpus_sha256": bundle.corpus_sha256,
        "preparation_spec": bundle.preparation_spec.model_dump(mode="python"),
        "books": books,
    }


def _preparation_hash(bundle: PreparationBundle) -> str:
    return sha256_bytes(canonical_json(_preparation_semantics(bundle)).encode("utf-8"))


def preparation_source(bundle: PreparationBundle) -> FrozenPreparationMap:
    books: dict[str, FrozenBookPreparation] = {}
    for key, value in bundle.books.items():
        books[key] = FrozenBookPreparation(
            book_id=value.book_id,
            source_sha256=value.source_sha256,
            analysis=value.analysis,
            style=value.style,
            style_brief=value.style_brief,
            book_synopsis=value.book_synopsis,
            chapter_digests=value.chapter_digests,
            source_digests=tuple(
                (row.chapter_index, row.source_sha256) for row in value.source_digests
            ),
            glossary=tuple(_glossary(value.glossary)),
            node_fingerprints=value.node_fingerprints,
        )
    return FrozenPreparationMap(bundle.preparation_sha256, books)


def _preparation_config(spec: PreparationSpec) -> Config:
    return Config(
        llm=LLMConfig(
            provider=spec.provider,
            models=ModelRoles(
                primary=spec.primary_model,
                editor=spec.editor_model,
                fast=spec.fast_model,
            ),
        ),
        source_lang="en",
        target_lang="zh",
    )


def freeze_preparation(
    corpus_dir: str | os.PathLike[str],
    book_spec_path: str | os.PathLike[str],
    preparation_spec_path: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    client=None,
) -> dict[str, Any]:
    """Create or resume a completed preparation export through Application."""
    corpus_root = Path(corpus_dir).expanduser().resolve()
    validate_corpus(corpus_root)
    corpus_value = _read_json(corpus_root / "corpus.json")
    spec = load_book_spec(book_spec_path)
    prep_spec = load_preparation_spec(preparation_spec_path)
    options = GenerationOptions(**_GENERATION_FIELDS)
    # Capability checks are deliberately before constructing an Application or
    # making a physical model call.
    for model in (prep_spec.primary_model, prep_spec.editor_model, prep_spec.fast_model):
        _validate_model(prep_spec.provider, model, options)

    books_input: list[tuple[Any, Path, str]] = []
    book_spec_root = Path(book_spec_path).expanduser().resolve().parent
    manifest = _read_json(corpus_root / "source_manifest.json")
    manifest_by_id = {row["book_id"]: row for row in manifest["books"]}
    for entry in spec.books:
        if entry.split not in {"screen", "formal"}:
            continue
        path = Path(entry.path).expanduser()
        if not path.is_absolute():
            path = book_spec_root / path
        path = path.resolve()
        source_sha = sha256_bytes(path.read_bytes())
        frozen = manifest_by_id.get(entry.book_id)
        if frozen is None or frozen["source_sha256"] != source_sha:
            raise BenchmarkError(f"source manifest identity mismatch: {entry.book_id}")
        books_input.append((entry, path, source_sha))
    if not books_input:
        raise BenchmarkError("no screen/formal books selected for preparation")

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    immutable = {
        "schema_version": 1,
        "corpus_sha256": corpus_value["corpus_sha256"],
        "book_spec_sha256": sha256_bytes(
            canonical_json(
                [
                    {"book_id": e.book_id, "source_sha256": source_sha, "split": e.split}
                    for e, _, source_sha in books_input
                ]
            ).encode("utf-8")
        ),
        "preparation_spec": prep_spec.model_dump(mode="python"),
    }
    immutable["immutable_sha256"] = sha256_bytes(canonical_json(immutable).encode("utf-8"))
    freeze_path = out / "freeze.json"
    if freeze_path.exists() and _read_json(freeze_path) != immutable:
        raise BenchmarkError("preparation immutable hash mismatch")
    if not freeze_path.exists():
        _atomic_json(freeze_path, immutable)
    state_path = out / "freeze_state.json"
    state = (
        _read_json(state_path)
        if state_path.exists()
        else {
            "status": "pending",
            "books": {entry.book_id: {"status": "pending"} for entry, _, _ in books_input},
        }
    )
    expected_ids = {entry.book_id for entry, _, _ in books_input}
    if set(state.get("books", {})) != expected_ids:
        raise BenchmarkError("preparation state book set mismatch")
    if state.get("status") == "completed":
        validate_preparation(out)
        return _read_json(out / "preparation.json")

    books_dir = out / "books"
    telemetry_dir = out / "telemetry"
    work_dir = out / "work"
    prepared: dict[str, BookPreparation] = {}
    for entry, path, source_sha in books_input:
        safe_id = _safe_book_id(entry.book_id)
        export_path = books_dir / f"{safe_id}.json"
        telemetry_path = telemetry_dir / f"{safe_id}.jsonl"
        previous = state["books"].get(entry.book_id, {})
        if previous.get("status") == "completed" and export_path.exists():
            try:
                exported = BookPreparation.model_validate(_read_json(export_path))
                if exported.source_sha256 != source_sha:
                    raise BenchmarkError(f"preparation source mismatch: {entry.book_id}")
                if exported.telemetry_sha256 != _relative_telemetry_hash(telemetry_path):
                    raise BenchmarkError(f"preparation telemetry mismatch: {entry.book_id}")
                prepared[entry.book_id] = exported
                continue
            except Exception as error:
                raise BenchmarkError(
                    f"invalid completed book export {entry.book_id}: {error}"
                ) from error

        state["status"] = "running"
        state["books"][entry.book_id] = {"status": "running", "export": f"books/{safe_id}.json"}
        _atomic_json(state_path, state)
        try:
            # Each book owns an isolated state root. The collision-resistant
            # suffix keeps equal parsed titles from sharing a RunStore.
            book_work = work_dir / safe_id
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_path.touch(exist_ok=True)
            sink = _JsonlTelemetrySink(telemetry_path)
            config = replace(_preparation_config(prep_spec), state_dir=str(book_work))
            persisted_before = (
                _read_json(Path(book_work) / "usage.json")
                if (Path(book_work) / "usage.json").exists()
                else {}
            )
            physical_client = client
            if physical_client is None:
                physical_client = build_client(
                    config, generation_options=options, telemetry_sink=sink
                )
            else:
                _attach_sink(physical_client, sink, required=True)
                # Injected clients used by offline tests expose these fields;
                # setting them makes the controlled contract observable without
                # changing the normal client interface.
                with suppress(Exception):
                    physical_client.generation_options = options
            before_usage = _usage_of(physical_client)
            from trans_novel.agents.analyzer import Analyzer
            from trans_novel.pipeline.bootstrap import Application

            app = Application(config, client=physical_client)
            store = app.prepare_for_translation(str(path))
            analysis = store.load_analysis() or {}
            style_brief = Analyzer(app.client, app.config).style_brief(analysis)
            chapter_digests: dict[str, str] = {}
            source_digests: list[ChapterSourceDigest] = []
            for chapter in store.load_state().chapters:
                chapter_store = store.load_chapter(chapter.index)
                source = "\n".join(s.source for s in chapter_store.text_segments)
                source_digests.append(
                    ChapterSourceDigest(
                        chapter_index=chapter.index,
                        source_sha256=sha256_bytes(source.encode("utf-8")),
                    )
                )
                chapter_digests[str(chapter.index)] = store.load_progress(
                    chapter.index
                ).source_digest
            glossary_store = GlossaryStore(store.glossary_path)
            try:
                terms = [
                    GlossaryPreparation.model_validate(term.__dict__)
                    for term in glossary_store.all_terms()
                ]
            finally:
                glossary_store.close()
            node_fingerprints = {
                key: node.input_fingerprint
                for key, node in store.load_state().nodes.items()
                if node.input_fingerprint
            }
            current_delta = usage_delta(_usage_of(app.client), before_usage)
            persisted_after = (
                _read_json(Path(store.usage_path)) if Path(store.usage_path).exists() else {}
            )
            expected_usage = merge_usage_summaries(persisted_before, current_delta)
            # RunStore usage.json is authoritative across process crashes. If a
            # client emitted a delta that was not checkpointed, reconcile just
            # that missing portion without double-counting persisted calls.
            usage = merge_usage_summaries(
                persisted_after,
                usage_delta(expected_usage, persisted_after),
            )
            telemetry_hash = _relative_telemetry_hash(telemetry_path)
            exported = BookPreparation(
                book_id=entry.book_id,
                source_sha256=source_sha,
                analysis=analysis,
                style=str(analysis.get("style") or ""),
                style_brief=style_brief,
                book_synopsis=str(analysis.get("book_synopsis") or ""),
                chapter_digests=chapter_digests,
                source_digests=source_digests,
                glossary=terms,
                node_fingerprints=node_fingerprints,
                usage=usage,
                telemetry_sha256=telemetry_hash,
                telemetry_path=f"telemetry/{safe_id}.jsonl",
            )
            _atomic_json(export_path, exported.model_dump(mode="python"))
            # Completion is recorded only after a strict byte/hash reread.
            checked = BookPreparation.model_validate(_read_json(export_path))
            if checked != exported or checked.telemetry_sha256 != _relative_telemetry_hash(
                telemetry_path
            ):
                raise BenchmarkError(f"book export validation failed: {entry.book_id}")
            prepared[entry.book_id] = checked
            state["books"][entry.book_id] = {
                "status": "completed",
                "export": f"books/{safe_id}.json",
            }
            _atomic_json(state_path, state)
        except Exception as error:
            state["status"] = "failed"
            state["books"][entry.book_id] = {"status": "failed"}
            _atomic_json(state_path, state)
            raise BenchmarkError(str(error)) from error
    if set(prepared) != expected_ids:
        raise BenchmarkError("preparation state is incomplete")
    bundle = PreparationBundle(
        schema_version=1,
        corpus_sha256=corpus_value["corpus_sha256"],
        preparation_spec=prep_spec,
        preparation_spec_sha256=sha256_bytes(
            canonical_json(prep_spec.model_dump(mode="python")).encode("utf-8")
        ),
        preparation_sha256="0" * 64,
        books=prepared,
    )
    bundle = bundle.model_copy(update={"preparation_sha256": _preparation_hash(bundle)})
    preparation_path = out / "preparation.json"
    _atomic_json(preparation_path, bundle.model_dump(mode="python"))
    usage_root = out / "usage"
    usage_root.mkdir(parents=True, exist_ok=True)
    usage_values: dict[str, Any] = {}
    book_evidence: dict[str, Any] = {}
    for book_id, value in sorted(prepared.items()):
        safe_id = _safe_book_id(book_id)
        usage_path = usage_root / f"{safe_id}.json"
        _atomic_json(usage_path, value.usage)
        usage_values[book_id] = value.usage
        export_path = out / "books" / f"{safe_id}.json"
        telemetry_path = out / value.telemetry_path
        book_evidence[book_id] = {
            "export_path": f"books/{safe_id}.json",
            "export_sha256": sha256_bytes(export_path.read_bytes()),
            "usage_path": f"usage/{safe_id}.json",
            "usage_sha256": sha256_bytes(usage_path.read_bytes()),
            "telemetry_path": value.telemetry_path,
            "telemetry_sha256": sha256_bytes(telemetry_path.read_bytes()),
        }
    usage_path = out / "usage.json"
    _atomic_json(usage_path, usage_values)
    completion = {
        "schema_version": 1,
        "preparation_sha256": bundle.preparation_sha256,
        "preparation_path": "preparation.json",
        "preparation_file_sha256": sha256_bytes(preparation_path.read_bytes()),
        "usage_path": "usage.json",
        "usage_file_sha256": sha256_bytes(usage_path.read_bytes()),
        "books": book_evidence,
    }
    completion["completion_sha256"] = sha256_bytes(canonical_json(completion).encode("utf-8"))
    _atomic_json(out / "preparation_complete.json", completion)
    state["status"] = "completed"
    state["completion_sha256"] = completion["completion_sha256"]
    _atomic_json(state_path, state)
    validate_preparation(out)
    return bundle.model_dump(mode="python")


def validate_preparation(preparation_dir: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(preparation_dir).expanduser().resolve()
    freeze = _read_json(root / "freeze.json")
    immutable_hash = freeze.get("immutable_sha256")
    freeze_copy = dict(freeze)
    freeze_copy.pop("immutable_sha256", None)
    if immutable_hash != sha256_bytes(canonical_json(freeze_copy).encode("utf-8")):
        raise BenchmarkError("preparation immutable manifest hash mismatch")
    state = _read_json(root / "freeze_state.json")
    if state.get("status") != "completed" or not isinstance(state.get("completion_sha256"), str):
        raise BenchmarkError("preparation export is incomplete")
    book_statuses = state.get("books")
    if not isinstance(book_statuses, dict) or any(
        not isinstance(value, dict) or value.get("status") != "completed"
        for value in book_statuses.values()
    ):
        raise BenchmarkError("preparation export is incomplete")
    bundle, _ = load_preparation_bundle(root / "preparation.json")
    if set(book_statuses) != set(bundle.books):
        raise BenchmarkError("preparation state/book export mismatch")
    if bundle.corpus_sha256 != freeze.get("corpus_sha256"):
        raise BenchmarkError("preparation corpus hash mismatch")
    expected_spec_hash = sha256_bytes(
        canonical_json(bundle.preparation_spec.model_dump(mode="python")).encode("utf-8")
    )
    if bundle.preparation_spec_sha256 != expected_spec_hash:
        raise BenchmarkError("preparation spec hash mismatch")
    if bundle.preparation_sha256 != _preparation_hash(bundle):
        raise BenchmarkError("preparation semantic hash mismatch")
    completion = _read_json(root / "preparation_complete.json")
    if not isinstance(completion, dict):
        raise BenchmarkError("preparation completion manifest is invalid")
    completion_hash = completion.get("completion_sha256")
    completion_copy = dict(completion)
    completion_copy.pop("completion_sha256", None)
    if completion_hash != sha256_bytes(canonical_json(completion_copy).encode("utf-8")):
        raise BenchmarkError("preparation completion manifest hash mismatch")
    if state.get("completion_sha256") != completion_hash:
        raise BenchmarkError("preparation completion state binding mismatch")
    if set(completion) != {
        "schema_version",
        "preparation_sha256",
        "preparation_path",
        "preparation_file_sha256",
        "usage_path",
        "usage_file_sha256",
        "books",
        "completion_sha256",
    }:
        raise BenchmarkError("preparation completion manifest schema mismatch")
    if (
        completion.get("schema_version") != 1
        or completion.get("preparation_sha256") != bundle.preparation_sha256
        or completion.get("preparation_path") != "preparation.json"
        or completion.get("usage_path") != "usage.json"
    ):
        raise BenchmarkError("preparation completion binding mismatch")
    prep_path = root / "preparation.json"
    usage_path = root / "usage.json"
    if completion.get("preparation_file_sha256") != sha256_bytes(prep_path.read_bytes()):
        raise BenchmarkError("preparation completion bundle hash mismatch")
    if completion.get("usage_file_sha256") != sha256_bytes(usage_path.read_bytes()):
        raise BenchmarkError("preparation completion usage hash mismatch")
    usage_values = _read_json(usage_path)
    if usage_values != {book_id: book.usage for book_id, book in sorted(bundle.books.items())}:
        raise BenchmarkError("preparation usage aggregate mismatch")
    evidence = completion.get("books")
    if not isinstance(evidence, dict) or set(evidence) != set(bundle.books):
        raise BenchmarkError("preparation completion book evidence mismatch")
    for book_id, book in bundle.books.items():
        relative = book.telemetry_path
        if relative is None or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise BenchmarkError(f"preparation telemetry path is not relative: {book_id}")
        telemetry = root / relative
        if book.telemetry_sha256 != _relative_telemetry_hash(telemetry):
            raise BenchmarkError(f"preparation telemetry hash mismatch: {book_id}")
        export = root / "books" / f"{_safe_book_id(book_id)}.json"
        book_evidence = evidence[book_id]
        usage_file = root / "usage" / f"{_safe_book_id(book_id)}.json"
        expected_evidence_keys = {
            "export_path",
            "export_sha256",
            "usage_path",
            "usage_sha256",
            "telemetry_path",
            "telemetry_sha256",
        }
        if (
            not isinstance(book_evidence, dict)
            or set(book_evidence) != expected_evidence_keys
            or book_evidence.get("export_path") != f"books/{_safe_book_id(book_id)}.json"
            or book_evidence.get("export_sha256") != sha256_bytes(export.read_bytes())
            or book_evidence.get("usage_path") != f"usage/{_safe_book_id(book_id)}.json"
            or book_evidence.get("usage_sha256") != sha256_bytes(usage_file.read_bytes())
            or book_evidence.get("telemetry_path") != relative
            or book_evidence.get("telemetry_sha256") != sha256_bytes(telemetry.read_bytes())
            or _read_json(usage_file) != book.usage
            or not export.exists()
        ):
            raise BenchmarkError(f"preparation completion evidence mismatch: {book_id}")
        if BookPreparation.model_validate(_read_json(export)) != book:
            raise BenchmarkError(f"preparation book export mismatch: {book_id}")
    return bundle.model_dump(mode="python")


def build_continuous_document(
    corpus_dir: str | os.PathLike[str],
    *,
    benchmark_id: str,
    book_id: str,
    replicate: int,
    identity_dir: str | os.PathLike[str],
    preparation: PreparationBundle | None = None,
):
    """Build the exact synthetic continuous-only Document and identity sidecar."""
    from trans_novel.ingest.models import Chapter, Document, Segment
    from trans_novel.pipeline.runstore import slugify

    _, rows = _load_corpus_rows(Path(corpus_dir).expanduser().resolve())
    selected = [row for row in rows if row["subset"] == "continuous" and row["book_id"] == book_id]
    if not selected:
        raise BenchmarkError(f"book has no continuous passages: {book_id}")
    chapters: list[Chapter] = []
    mapping: dict[str, str] = {}
    all_segment_ids: list[str] = []
    for ordinal, row in enumerate(selected, 1):
        segments: list[Segment] = []
        for segment in row["segments"]:
            meta = dict(segment.get("meta") or {})
            meta.update(
                {
                    "original_chapter_index": row["chapter_index"],
                    "original_segment_index": segment["index"],
                    "original_segment_id": segment["segment_id"],
                    "passage_id": row["passage_id"],
                }
            )
            segments.append(
                Segment(
                    index=len(segments),
                    source=segment["source"],
                    kind=segment.get("kind", "text"),
                    anchor=segment.get("anchor"),
                    resource_href=segment.get("resource_href"),
                    cont=segment.get("cont", False),
                    meta=meta,
                )
            )
            all_segment_ids.append(segment["segment_id"])
        passage_digest = sha256_bytes(
            "\n".join(segment["source"] for segment in row["segments"]).encode("utf-8")
        )
        chapter_digest = passage_digest
        if preparation is not None:
            book = preparation.books.get(book_id)
            if book is None:
                raise BenchmarkError(f"missing frozen book for synthetic document: {book_id}")
            original_index = row["chapter_index"]
            proof_matches = [
                proof for proof in book.source_digests if proof.chapter_index == original_index
            ]
            if len(proof_matches) != 1:
                raise BenchmarkError(
                    f"synthetic passage does not map to one original chapter: {book_id}"
                )
            chapter_digest = book.chapter_digests.get(str(original_index))
            if not chapter_digest:
                raise BenchmarkError(
                    f"frozen preparation has no chapter digest: {book_id}:{original_index}"
                )
            mapping[str(ordinal - 1)] = str(original_index)
        else:
            mapping[str(ordinal - 1)] = str(row["chapter_index"])
        chapters.append(
            Chapter(
                index=ordinal - 1,
                title=f"Benchmark {ordinal}",
                segments=segments,
                meta={
                    "benchmark_passage_id": row["passage_id"],
                    "original_chapter_index": row["chapter_index"],
                    "original_chapter_digest": chapter_digest,
                },
            )
        )
    title = f"{slugify(benchmark_id)}_{slugify(book_id)}_r{replicate}"
    corpus_value = _read_json(Path(corpus_dir).expanduser().resolve() / "corpus.json")
    manifest_books = _read_json(Path(corpus_dir).expanduser().resolve() / "source_manifest.json")[
        "books"
    ]
    original_source_sha = next(
        row["source_sha256"] for row in manifest_books if row["book_id"] == book_id
    )
    sidecar = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "book_id": book_id,
        "replicate": replicate,
        "corpus_sha256": corpus_value["corpus_sha256"],
        "original_source_sha256": original_source_sha,
        "segment_ids": all_segment_ids,
        "source_sha256": [
            sha256_bytes(segment["source"].encode("utf-8"))
            for row in selected
            for segment in row["segments"]
        ],
        "chapter_mapping": mapping,
    }
    root = Path(identity_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity_path = root / f"{title}.identity.json"
    _atomic_json(identity_path, sidecar)
    doc = Document(
        title=title,
        source_lang="en",
        target_lang="zh",
        fmt="text",
        source_path=str(identity_path),
        chapters=chapters,
        meta={
            "benchmark_book_id": book_id,
            "benchmark_id": benchmark_id,
            "replicate": replicate,
            "continuous_chapter_mapping": mapping,
            "source_sha256": original_source_sha,
        },
    )
    return doc, str(identity_path), mapping


def _validate_model(provider: str, value: str, options: GenerationOptions) -> None:
    try:
        selection = validate_model_selection(provider, value)
    except Exception as error:
        raise BenchmarkError(f"invalid model selection {provider}:{value}: {error}") from error
    if selection.thinking != "off":
        raise BenchmarkError(f"model must explicitly disable thinking: {provider}:{value}")
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
        require_thinking_disabled=True,
    )
    _validate_model(spec.provider, spec.fast_model, options)
    for candidate in spec.candidates:
        _validate_model(spec.provider, candidate.primary_model, options)
        if candidate.editor_model is not None:
            _validate_model(spec.provider, candidate.editor_model, options)
    return options


def _load_corpus_rows(corpus_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    validate_corpus(corpus_dir)
    corpus = _read_json(corpus_dir / "corpus.json")
    try:
        digest = corpus["corpus_sha256"]
        rows = [
            json.loads(line)
            for line in (corpus_dir / "runner_segments.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        parsed = [EmittedRunnerRecord.model_validate(row).model_dump(mode="python") for row in rows]
    except Exception as error:
        raise BenchmarkError(f"invalid corpus runner artifacts: {error}") from error
    return digest, parsed


def _glossary(rows: list[GlossaryPreparation]) -> list[GlossaryTerm]:
    return [
        GlossaryTerm(
            source=row.source,
            target=row.target,
            reading=row.reading,
            type=row.type,
            gender=row.gender,
            aliases=list(row.aliases),
            first_chapter=row.first_chapter,
            note=row.note,
            confidence=row.confidence,
            locked=row.locked,
            status=row.status,
        )
        for row in rows
    ]


def _locked_terms(terms: list[GlossaryTerm]) -> list[GlossaryTerm]:
    return [term for term in terms if term.locked]


def _prep_for(bundle: PreparationBundle, row: dict[str, Any]) -> BookPreparation:
    try:
        book = bundle.books[row["book_id"]]
    except KeyError as error:
        raise BenchmarkError(f"preparation missing book {row.get('book_id')}") from error
    chapter = str(row["chapter_index"])
    if chapter not in book.chapter_digests:
        raise BenchmarkError(f"preparation missing chapter {row['book_id']}:{chapter}")
    return book


def _context_for(row: dict[str, Any], strategy: str) -> TranslationContextBundle:
    if strategy == "c0" or row.get("context") is None:
        return TranslationContextBundle()
    context = row["context"]
    before = "\n".join(f"[{x['segment_id']}] {x['source']}" for x in context["source_before"])
    after = "\n".join(f"[{x['segment_id']}] {x['source']}" for x in context["source_after"])
    if strategy == "c1":
        return TranslationContextBundle(source_before=before, source_after=after)
    target = "\n".join(
        f"[{x['segment_id']}] {x['target']}" for x in context["frozen_target_before"]
    )
    return TranslationContextBundle(source_before=before, target_before=target, source_after=after)


def _safe_id(value: str) -> str:
    """Canonical opaque artifact identifier used by phase-five artifacts."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _safe_book_id(value: str) -> str:
    """Readable, collision-resistant relative preparation directory name."""
    safe = "".join(char if (char.isalnum() or char in "-_") else "_" for char in value).strip("_")
    safe = safe or "book"
    return f"{safe}-{_safe_id(value)[:12]}"


def _relative_telemetry_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes()) if path.exists() else ""


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
    cfg = Config(
        llm=LLMConfig(provider=spec.provider, models=models),
        source_lang="en",
        target_lang="zh",
    )
    return build_client(cfg, generation_options=options, telemetry_sink=telemetry_sink)


def _attach_sink(client: Any, sink: _JsonlTelemetrySink | None, *, required: bool = False) -> None:
    if sink is None:
        return
    setter = getattr(client, "set_telemetry_sink", None)
    attached = False
    if callable(setter):
        try:
            setter(sink)
            attached = True
        except (AttributeError, ValueError, TypeError):
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


def _usage_of(client: Any) -> dict[str, Any]:
    usage = getattr(client, "usage", None)
    return usage.summary() if usage is not None and hasattr(usage, "summary") else {}


def _passage_usage_delta(client: Any, before: dict[str, Any]) -> dict[str, Any]:
    after = _usage_of(client)
    if not after and not before:
        return usage_delta({}, {})
    return usage_delta(after, before)


def _source_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_bytes(
        canonical_json(
            [{"segment_id": r["segment_id"], "source": r["source"]} for r in rows]
        ).encode()
    )


def _stage_manifest(
    *,
    artifact_key: str,
    kind: str,
    spec: CandidateSpec,
    model: str,
    corpus_sha: str,
    preparation_sha: str,
    replicate: int,
    subset: str,
    strategy: str,
    passage_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "artifact_key": artifact_key,
        "kind": kind,
        "prompt_version": PROMPT_VERSION,
        "provider": spec.provider,
        "model": model,
        "corpus_sha256": corpus_sha,
        "preparation_sha256": preparation_sha,
        "replicate": replicate,
        "scope": {"subset": subset, "context_strategy": strategy, "passage_ids": passage_ids},
    }


def _context_dict(value: TranslationContextBundle) -> dict[str, str]:
    return {
        "source_before": value.source_before,
        "target_before": value.target_before,
        "source_after": value.source_after,
    }


def _context_hash(
    contexts: list[TranslationContextBundle],
    prep: BookPreparation | None,
    *,
    chapter_digest: str = "",
    glossary_terms: list[GlossaryTerm] | None = None,
) -> str:
    terms = (
        glossary_terms
        if glossary_terms is not None
        else (_glossary(prep.glossary) if prep is not None else [])
    )
    preparation = {
        "style": prep.style if prep is not None else "",
        "book_synopsis": prep.book_synopsis if prep is not None else "",
        "chapter_digest": chapter_digest if prep is not None else "",
        "glossary": prompts.render_glossary(terms),
    }
    return sha256_bytes(
        canonical_json(
            {"preparation": preparation, "batches": [_context_dict(x) for x in contexts]}
        ).encode()
    )


def _batch_context_hash(context: TranslationContextBundle) -> str:
    return sha256_bytes(canonical_json(_context_dict(context)).encode())


def _hash_record(record: dict[str, Any]) -> None:
    raw = record["translation_raw"]
    after_lint = record["translation_after_lint"]
    final = record["final"]
    record["translation_raw_sha256"] = sha256_bytes(raw.encode())
    record["translation_after_lint_sha256"] = sha256_bytes(after_lint.encode())
    record["final_sha256"] = sha256_bytes(final.encode())
    record["output_sha256"] = record["final_sha256"]


def _validate_usage_delta(value: Any) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise BenchmarkError("invalid or missing normalized passage usage delta")
    required = {
        "schema_version",
        "totals",
        "by_agent",
        "by_operation",
        "by_provider",
        "by_model",
        "by_stage",
    }
    if set(value) != required:
        raise BenchmarkError("invalid normalized passage usage delta")
    try:
        merge_usage_summaries({}, value)
    except Exception as error:
        raise BenchmarkError("invalid normalized passage usage delta") from error


def _validate_segment_hashes(record: dict[str, Any], *, editor: bool) -> None:
    required = (
        "segment_id",
        "source",
        "translation_raw",
        "translation_after_lint",
        "translation_lint_issues",
        "polish_proposal",
        "polish_accepted",
        "polish_rejection_reasons",
        "final",
        "translation_raw_sha256",
        "translation_after_lint_sha256",
        "final_sha256",
        "output_sha256",
    )
    if any(key not in record for key in required):
        raise BenchmarkError("incomplete benchmark segment record")
    if any(
        not isinstance(record[key], str) or not record[key].strip()
        for key in ("segment_id", "source", "translation_raw", "translation_after_lint", "final")
    ):
        raise BenchmarkError("benchmark segment contains empty text")
    if not isinstance(record["translation_lint_issues"], list) or not isinstance(
        record["polish_rejection_reasons"], list
    ):
        raise BenchmarkError("benchmark segment issue fields are invalid")
    expected = {
        "translation_raw_sha256": sha256_bytes(record["translation_raw"].encode()),
        "translation_after_lint_sha256": sha256_bytes(record["translation_after_lint"].encode()),
        "final_sha256": sha256_bytes(record["final"].encode()),
    }
    if record.get("output_sha256") != expected["final_sha256"] or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise BenchmarkError("benchmark segment output hash mismatch")
    if editor:
        if (
            not isinstance(record.get("polish_proposal"), str)
            or not record["polish_proposal"].strip()
        ):
            raise BenchmarkError("editor proposal is missing")
        if not isinstance(record.get("polish_accepted"), bool):
            raise BenchmarkError("editor acceptance is missing")
        if not isinstance(record.get("polish_rejection_reasons"), list):
            raise BenchmarkError("editor rejection reasons are missing")
    else:
        if (
            record["translation_after_lint"] != record["translation_raw"]
            or record["final"] != record["translation_raw"]
        ):
            raise BenchmarkError("translation artifact changed its baseline text")
        if record.get("polish_proposal") is not None or record.get("polish_accepted") is not None:
            raise BenchmarkError("translation artifact contains editor output")


def _validate_passage_common(
    passage: dict[str, Any],
    row: dict[str, Any],
    *,
    artifact_key: str,
    spec: CandidateSpec,
    replicate: int,
    subset: str,
    strategy: str,
    model: str,
    context_hash: str,
    preparation_sha256: str | None = None,
    editor: bool = False,
    editor_key: str | None = None,
    baseline_segments: list[dict[str, Any]] | None = None,
    prep: BookPreparation | None = None,
) -> None:
    expected = {
        "status": "complete",
        "artifact_key": artifact_key,
        "passage_id": row["passage_id"],
        "subset": subset,
        "book_id": row["book_id"],
        "chapter_index": row["chapter_index"],
        "replicate": replicate,
        "context_strategy": strategy,
        "provider": spec.provider,
        "source_hash": _source_hash(row["segments"]),
        "context_hash": context_hash,
    }
    expected_preparation = (
        preparation_sha256 if preparation_sha256 is not None else row.get("preparation_sha256")
    )
    if expected_preparation is not None:
        expected["preparation_sha256"] = expected_preparation
    if passage.get("kind") != ("editor" if editor else "translation"):
        raise BenchmarkError("benchmark passage kind mismatch")
    for key, value in expected.items():
        if passage.get(key) != value:
            raise BenchmarkError(
                f"invalid or incomplete benchmark passage {row['passage_id']}: {key}"
            )
    if editor:
        if prep is None:
            raise BenchmarkError("editor passage preparation is missing")
        if (
            passage.get("artifact_key") != editor_key
            or passage.get("editor_model") != model
            or passage.get("translation_artifact_key")
            != row.get("artifact_key", passage.get("translation_artifact_key"))
        ):
            raise BenchmarkError("editor passage provenance mismatch")
    elif passage.get("primary_model") != model:
        raise BenchmarkError("translation passage model mismatch")
    segments = passage.get("segments")
    if not isinstance(segments, list) or len(segments) != len(row["segments"]):
        raise BenchmarkError("benchmark passage segment count mismatch")
    expected_ids = [x["segment_id"] for x in row["segments"]]

    if [x.get("segment_id") for x in segments] != expected_ids:
        raise BenchmarkError("benchmark passage segment order mismatch")
    for index, (expected_segment, record) in enumerate(zip(row["segments"], segments, strict=True)):
        if record.get("source") != expected_segment["source"]:
            raise BenchmarkError("benchmark passage source mismatch")
        _validate_segment_hashes(record, editor=editor)
        if editor and baseline_segments is not None:
            baseline = baseline_segments[index]
            for field in (
                "segment_id",
                "source",
                "translation_raw",
                "translation_after_lint",
                "translation_lint_issues",
                "translation_raw_sha256",
                "translation_after_lint_sha256",
            ):
                if record.get(field) != baseline.get(field):
                    raise BenchmarkError("editor passage changed its translation baseline")
            gate = lint.polish_gate(
                record["source"],
                record["translation_raw"],
                record["polish_proposal"],
                locked_terms=_locked_terms(_glossary(prep.glossary)),
                src_lang="en",
            )
            expected_gate = {
                "polish_proposal": gate.proposal,
                "polish_accepted": gate.accepted,
                "polish_rejection_reasons": list(gate.rejection_reasons),
                "final": gate.selected,
            }
            if any(record.get(key) != value for key, value in expected_gate.items()):
                raise BenchmarkError("editor polish gate evidence mismatch")
    _validate_usage_delta(passage.get("usage_delta"))


def _segment_findings(items: list[dict[str, Any]], segment_index: int) -> list[dict[str, Any]]:
    return [
        item for item in items if item.get("index") is None or item.get("index") == segment_index
    ]


class FullRunner:
    """Run the production quality DAG over continuous synthetic documents."""

    def __init__(
        self, *, client_factory: Callable[..., Any] | None = None, client: Any | None = None
    ):
        self.client_factory = client_factory
        self.client = client

    def _client(
        self,
        spec: CandidateSpec,
        model: str,
        role: str,
        options: GenerationOptions,
        sink: _JsonlTelemetrySink,
        *,
        roles: ModelRoles | None = None,
    ):
        if self.client is not None:
            _attach_sink(self.client, sink, required=True)
            return self.client
        return _model_client(spec, model, role, options, self.client_factory, sink, roles=roles)

    @staticmethod
    def _config(spec: CandidateSpec, primary: str, editor: str, *, quality: bool, state_dir: str):
        p = PipelineConfig.for_quality("quality" if quality else "economy").model_copy(
            update={
                "book_understanding": True,
                "polish": quality,
                "naturalize": quality,
                "review": quality,
                "autofix_severe": quality,
                "backtranslate_sample": 0.05 if quality else 0,
                "consistency_qa": quality,
            }
        )
        return Config(
            llm=LLMConfig(
                provider=spec.provider,
                models=ModelRoles(primary=primary, editor=editor, fast=spec.fast_model),
            ),
            source_lang="en",
            target_lang="zh",
            pipeline=p,
            state_dir=state_dir,
        )

    @staticmethod
    def _raw_key(
        spec: CandidateSpec,
        primary: str,
        corpus_sha: str,
        preparation_sha: str,
        document,
        identity_path: str,
        book_id: str,
        replicate: int,
    ) -> str:
        identity_sha = sha256_bytes(Path(identity_path).read_bytes())
        semantics = document.model_dump(mode="python")
        semantics["source_path"] = ""
        value = {
            "schema_version": 1,
            "provider": spec.provider,
            "primary_model": primary,
            "generation": dict(_GENERATION_FIELDS),
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
            "document": semantics,
            "identity_sha256": identity_sha,
            "book_id": book_id,
            "replicate": replicate,
        }
        return sha256_bytes(canonical_json(value).encode("utf-8"))

    @staticmethod
    def _preparation_allocation_id(
        candidate_id: str,
        book_id: str,
        preparation_sha256: str,
    ) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "candidate_id": candidate_id,
                    "book_id": book_id,
                    "preparation_sha256": preparation_sha256,
                }
            ).encode("utf-8")
        )

    @staticmethod
    def _stage_records(store: RunStore) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for chapter in store.load_state().chapters:
            progress = store.load_progress(chapter.index)
            issues = progress.review_issue_dicts()
            review = [item for item in issues if item.get("stage") in {None, "review"}]
            lint_findings = [item for item in issues if item.get("stage") == "lint"]
            for segment in store.load_chapter(chapter.index).text_segments:
                segment_id = segment.meta.get("original_segment_id")
                if not segment_id:
                    raise BenchmarkError("synthetic segment has no original segment_id")
                rows.append(
                    {
                        "segment_id": segment_id,
                        "source": segment.source,
                        "raw_after_translation_lint": segment.target or "",
                        "final_after_full_pipeline": segment.target or "",
                        "review_findings": _segment_findings(review, segment.index),
                        "lint_findings": _segment_findings(lint_findings, segment.index),
                        "backtranslation_findings": _segment_findings(
                            progress.backtranslation_issue_dicts(), segment.index
                        ),
                        "final_sha256": sha256_bytes((segment.target or "").encode()),
                    }
                )
        return rows

    @classmethod
    def _readonly_store(
        cls, run_dir: Path
    ) -> tuple[RunState, list[dict[str, Any]], dict[str, Any]]:
        try:
            state = RunState.model_validate(_read_json(run_dir / "manifest.json"))
            if state.run_state_schema != 2:
                raise BenchmarkError("completed RunStore schema is invalid")
            if not state.chapters:
                raise BenchmarkError("completed RunStore has no chapters")
            for name in ("context.json", "analysis.json", "report.json"):
                artifact = run_dir / name
                if artifact.exists() and not isinstance(_read_json(artifact), dict):
                    raise BenchmarkError(f"completed {name} artifact is invalid")
            usage = _read_json(run_dir / "usage.json")
            if not isinstance(usage, dict):
                raise BenchmarkError("completed usage artifact is invalid")
            _validate_usage_delta(usage)
            events = run_dir / "events.jsonl"
            if events.exists():
                for line in events.read_text(encoding="utf-8").splitlines():
                    if line.strip() and not isinstance(json.loads(line), dict):
                        raise BenchmarkError("completed events artifact is invalid")
            rows: list[dict[str, Any]] = []
            for chapter in state.chapters:
                progress = state.progress.get(chapter.index)
                if progress is None or progress.status != "done":
                    raise BenchmarkError("completed RunStore progress is incomplete")
                chapter_path = run_dir / "chapters_v2" / f"ch{chapter.index}.json"
                chapter_value = Chapter.from_dict(_read_json(chapter_path))
                issues = progress.review_issue_dicts()
                review = [item for item in issues if item.get("stage") in {None, "review"}]
                lint_findings = [item for item in issues if item.get("stage") == "lint"]
                for segment in chapter_value.text_segments:
                    segment_id = segment.meta.get("original_segment_id")
                    if not segment_id:
                        raise BenchmarkError("synthetic segment has no original segment_id")
                    rows.append(
                        {
                            "segment_id": segment_id,
                            "source": segment.source,
                            "raw_after_translation_lint": segment.target or "",
                            "final_after_full_pipeline": segment.target or "",
                            "review_findings": _segment_findings(review, segment.index),
                            "lint_findings": _segment_findings(lint_findings, segment.index),
                            "backtranslation_findings": _segment_findings(
                                progress.backtranslation_issue_dicts(), segment.index
                            ),
                            "final_sha256": sha256_bytes((segment.target or "").encode()),
                        }
                    )
            if any(node.status not in {"succeeded", "skipped"} for node in state.nodes.values()):
                raise BenchmarkError("completed RunStore nodes are incomplete")
            return state, rows, usage
        except BenchmarkError:
            raise
        except Exception as error:
            raise BenchmarkError("completed RunStore artifact is invalid") from error

    @staticmethod
    def _store_ready(store: RunStore) -> bool:
        try:
            state = store.load_state()
            if any(progress.status != "done" for progress in state.progress.values()):
                return False
            if any(node.status not in {"succeeded", "skipped"} for node in state.nodes.values()):
                return False
            return bool(state.chapters) and all(
                Path(store.chapter_path(chapter.index)).exists() for chapter in state.chapters
            )
        except Exception:
            return False

    @classmethod
    def _validate_completed_output(
        cls,
        out: Path,
        immutable: dict[str, Any],
        spec: CandidateSpec,
        bundle: PreparationBundle,
        *,
        expected_count: int,
        expected_tuples: set[tuple[str, str, int]],
    ) -> tuple[int, list[dict[str, Any]]]:
        raw_root = out / "raw"
        branch_root = out / "branches"
        candidates_path = out / "candidates.json"
        payload = _read_json(candidates_path)
        if not isinstance(payload, list):
            raise BenchmarkError("completed candidates artifact must be a list")
        if len(payload) != expected_count:
            raise BenchmarkError("completed candidate branch set is incomplete")
        actual_tuples = {
            (row.get("candidate_id"), row.get("book_id"), row.get("replicate"))
            for row in payload
            if isinstance(row, dict)
        }
        if len(actual_tuples) != len(payload) or actual_tuples != expected_tuples:
            raise BenchmarkError("completed candidate branch tuple set is invalid")
        allocation_rows: dict[str, list[dict[str, Any]]] = {}
        actual: dict[str, Any] = {}
        counted_raw: set[str] = set()
        counted_branch: set[str] = set()
        for row in payload:
            if not isinstance(row, dict):
                raise BenchmarkError("completed candidate record is invalid")
            required = {
                "candidate_id",
                "book_id",
                "replicate",
                "provider",
                "corpus_sha256",
                "preparation_sha256",
                "primary_model",
                "editor_model",
                "fast_model",
                "raw_artifact_id",
                "branch_artifact_id",
                "preparation_allocation_id",
                "allocated_usage",
                "stage",
                "raw_telemetry_sha256",
                "branch_telemetry_sha256",
                "zero_retranslation",
            }
            if set(row) < required or not isinstance(row["stage"], list):
                raise BenchmarkError("completed candidate record is incomplete")
            candidate_def = next(
                (
                    candidate
                    for candidate in spec.candidates
                    if candidate.candidate_id == row["candidate_id"]
                ),
                None,
            )
            if (
                candidate_def is None
                or row["provider"] != spec.provider
                or row["fast_model"] != spec.fast_model
                or row["primary_model"] != candidate_def.primary_model
                or row["editor_model"] != candidate_def.editor_model
                or row["corpus_sha256"] != immutable["corpus_sha256"]
                or row["preparation_sha256"] != immutable["preparation_sha256"]
                or row["zero_retranslation"] is not True
                or row["editor_model"] is None
            ):
                raise BenchmarkError("completed candidate identity mismatch")
            raw_manifest_path = raw_root / row["raw_artifact_id"] / "manifest.json"
            branch_manifest_path = branch_root / row["branch_artifact_id"] / "manifest.json"
            if not raw_manifest_path.exists() or not branch_manifest_path.exists():
                raise BenchmarkError("completed RunStore manifest is missing")
            raw_manifest = _read_json(raw_manifest_path)
            branch_manifest = _read_json(branch_manifest_path)
            if not isinstance(raw_manifest, dict) or not isinstance(branch_manifest, dict):
                raise BenchmarkError("completed RunStore manifest is invalid")
            raw_required = {
                "schema_version": 1,
                "kind": "raw",
                "artifact_key": row["raw_artifact_id"],
                "provider": spec.provider,
                "primary_model": row["primary_model"],
                "editor_model": None,
                "fast_model": spec.fast_model,
                "generation": dict(_GENERATION_FIELDS),
                "corpus_sha256": immutable["corpus_sha256"],
                "preparation_sha256": immutable["preparation_sha256"],
                "book_id": row["book_id"],
                "replicate": row["replicate"],
            }
            if any(raw_manifest.get(key) != value for key, value in raw_required.items()):
                raise BenchmarkError("completed raw manifest semantics mismatch")
            branch_required = {
                "schema_version": 1,
                "kind": "branch",
                "artifact_key": row["branch_artifact_id"],
                "raw_artifact": row["raw_artifact_id"],
                "provider": spec.provider,
                "primary_model": row["primary_model"],
                "editor_model": row["editor_model"],
                "fast_model": spec.fast_model,
                "generation": dict(_GENERATION_FIELDS),
                "corpus_sha256": immutable["corpus_sha256"],
                "preparation_sha256": immutable["preparation_sha256"],
                "book_id": row["book_id"],
                "replicate": row["replicate"],
            }
            if any(branch_manifest.get(key) != value for key, value in branch_required.items()):
                raise BenchmarkError("completed branch manifest semantics mismatch")
            if raw_manifest.get("document") != branch_manifest.get("document"):
                raise BenchmarkError("completed manifest document linkage mismatch")
            if raw_manifest.get("identity_sha256") != branch_manifest.get("identity_sha256"):
                raise BenchmarkError("completed manifest identity linkage mismatch")
            if not isinstance(raw_manifest.get("document"), dict) or not isinstance(
                raw_manifest.get("identity_sha256"), str
            ):
                raise BenchmarkError("completed manifest document identity is invalid")
            if set(raw_manifest) != set(raw_required) | {"document", "identity_sha256"} or set(
                branch_manifest
            ) != set(branch_required) | {"document", "identity_sha256"}:
                raise BenchmarkError("completed manifest schema keys are invalid")
            document_value = raw_manifest["document"]
            if (
                document_value.get("title")
                != f"{spec.benchmark_id}_{row['book_id']}_r{row['replicate']}"
                or document_value.get("source_path") != ""
            ):
                raise BenchmarkError("completed manifest document semantics mismatch")
            expected_raw_key = sha256_bytes(
                canonical_json(
                    {
                        "schema_version": 1,
                        "provider": spec.provider,
                        "primary_model": row["primary_model"],
                        "generation": dict(_GENERATION_FIELDS),
                        "corpus_sha256": immutable["corpus_sha256"],
                        "preparation_sha256": immutable["preparation_sha256"],
                        "document": document_value,
                        "identity_sha256": raw_manifest["identity_sha256"],
                        "book_id": row["book_id"],
                        "replicate": row["replicate"],
                    }
                ).encode("utf-8")
            )
            expected_branch_key = sha256_bytes(
                canonical_json(
                    {
                        "raw_artifact": row["raw_artifact_id"],
                        "editor_model": row["editor_model"],
                        "fast_model": spec.fast_model,
                    }
                ).encode("utf-8")
            )
            if (
                expected_raw_key != row["raw_artifact_id"]
                or expected_branch_key != row["branch_artifact_id"]
            ):
                raise BenchmarkError("completed artifact key hash mismatch")
            raw_dirs = [p for p in (raw_root / row["raw_artifact_id"]).iterdir() if p.is_dir()]
            branch_dirs = [
                p for p in (branch_root / row["branch_artifact_id"]).iterdir() if p.is_dir()
            ]
            if len(raw_dirs) != 1 or len(branch_dirs) != 1:
                raise BenchmarkError("completed RunStore layout is invalid")
            _, raw_records, raw_usage = cls._readonly_store(raw_dirs[0])
            _, branch_records_stage, branch_usage = cls._readonly_store(branch_dirs[0])
            raw_stage = {entry["segment_id"]: entry for entry in raw_records}
            branch_stage = {entry["segment_id"]: entry for entry in branch_records_stage}
            if set(raw_stage) != set(branch_stage):
                raise BenchmarkError("completed stage segment set mismatch")
            raw_telemetry = raw_root / row["raw_artifact_id"] / "telemetry.jsonl"
            branch_telemetry = branch_root / row["branch_artifact_id"] / "telemetry.jsonl"
            if not raw_telemetry.exists() or not branch_telemetry.exists():
                raise BenchmarkError("completed telemetry artifact is missing")
            if row["raw_telemetry_sha256"] != sha256_bytes(raw_telemetry.read_bytes()):
                raise BenchmarkError("completed raw telemetry hash mismatch")
            if row["branch_telemetry_sha256"] != sha256_bytes(branch_telemetry.read_bytes()):
                raise BenchmarkError("completed branch telemetry hash mismatch")
            try:
                raw_telemetry_records = [
                    json.loads(line)
                    for line in raw_telemetry.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                branch_telemetry_records = [
                    json.loads(line)
                    for line in branch_telemetry.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception as error:
                raise BenchmarkError("completed telemetry is invalid") from error
            if any(
                not isinstance(record, dict)
                for record in (*raw_telemetry_records, *branch_telemetry_records)
            ):
                raise BenchmarkError("completed telemetry record is invalid")
            if any(
                str(record.get("operation", "")) in {"translate.batch", "translate.lint_fix"}
                for record in branch_telemetry_records
            ):
                raise BenchmarkError("completed branch contains first-pass translation")
            allocated = row["allocated_usage"]
            if not isinstance(allocated, dict) or set(allocated) != {
                "preparation",
                "raw",
                "branch_increment",
            }:
                raise BenchmarkError("completed usage allocation is invalid")
            expected_allocation_id = cls._preparation_allocation_id(
                row["candidate_id"],
                row["book_id"],
                immutable["preparation_sha256"],
            )
            expected_preparation = (
                bundle.books[row["book_id"]].usage if row["replicate"] == 1 else {}
            )
            if (
                row["preparation_allocation_id"] != expected_allocation_id
                or allocated["preparation"] != expected_preparation
                or allocated["raw"] != raw_usage
                or allocated["branch_increment"] != usage_delta(branch_usage, raw_usage)
            ):
                raise BenchmarkError("completed usage allocation does not match artifacts")
            allocation_rows.setdefault(row["preparation_allocation_id"], []).append(row)
            stage_ids = [
                stage.get("segment_id") for stage in row["stage"] if isinstance(stage, dict)
            ]
            if len(stage_ids) != len(set(stage_ids)) or set(stage_ids) != set(raw_stage):
                raise BenchmarkError("completed stage segment set is invalid")
            if row["raw_artifact_id"] not in counted_raw:
                actual = merge_usage_summaries(actual, allocated["raw"])
                counted_raw.add(row["raw_artifact_id"])
            if row["branch_artifact_id"] not in counted_branch:
                actual = merge_usage_summaries(actual, allocated["branch_increment"])
                counted_branch.add(row["branch_artifact_id"])
            for stage in row["stage"]:
                if not isinstance(stage, dict):
                    raise BenchmarkError("completed stage record is invalid")
                required_stage = (
                    "segment_id",
                    "source",
                    "raw_after_translation_lint",
                    "final_after_full_pipeline",
                    "review_findings",
                    "lint_findings",
                    "backtranslation_findings",
                    "final_after_full_pipeline",
                    "raw_artifact_id",
                    "branch_artifact_id",
                    "preparation_sha256",
                )
                if any(field not in stage for field in required_stage):
                    raise BenchmarkError("completed stage mapping is incomplete")
                baseline_raw = raw_stage.get(stage["segment_id"])
                baseline_branch = branch_stage.get(stage["segment_id"])
                if baseline_raw is None or baseline_branch is None:
                    raise BenchmarkError("completed stage segment is missing")
                if (
                    stage["source"] != baseline_raw["source"]
                    or stage["raw_after_translation_lint"]
                    != baseline_raw["raw_after_translation_lint"]
                    or stage["final_after_full_pipeline"]
                    != baseline_branch["final_after_full_pipeline"]
                    or stage["review_findings"] != baseline_branch["review_findings"]
                    or stage["lint_findings"] != baseline_branch["lint_findings"]
                    or stage["backtranslation_findings"]
                    != baseline_branch["backtranslation_findings"]
                ):
                    raise BenchmarkError("completed stage content mismatch")
                if (
                    stage["raw_artifact_id"] != row["raw_artifact_id"]
                    or stage["branch_artifact_id"] != row["branch_artifact_id"]
                ):
                    raise BenchmarkError("completed stage provenance mismatch")
                if stage["preparation_sha256"] != immutable["preparation_sha256"]:
                    raise BenchmarkError("completed stage preparation mismatch")
                if stage.get("final_sha256") != sha256_bytes(
                    str(stage["final_after_full_pipeline"]).encode()
                ):
                    raise BenchmarkError("completed stage hash mismatch")
        if not (out / "actual_usage.json").exists():
            raise BenchmarkError("completed actual usage artifact is missing")
        expected_allocations = {
            cls._preparation_allocation_id(
                candidate.candidate_id,
                book_id,
                immutable["preparation_sha256"],
            )
            for candidate in spec.candidates
            if candidate.editor_model is not None
            for book_id in {book for _, book, _ in expected_tuples}
        }
        if set(allocation_rows) != expected_allocations:
            raise BenchmarkError("completed preparation allocation set is invalid")
        for rows_for_allocation in allocation_rows.values():
            if {row["replicate"] for row in rows_for_allocation} != set(
                range(1, spec.replicates + 1)
            ):
                raise BenchmarkError("completed preparation allocation replicates are invalid")
            owners = [row for row in rows_for_allocation if row["replicate"] == 1]
            if len(owners) != 1:
                raise BenchmarkError("completed preparation allocation is not owned once")
            if (
                owners[0]["allocated_usage"]["preparation"]
                != bundle.books[owners[0]["book_id"]].usage
            ):
                raise BenchmarkError("completed preparation allocation value is invalid")
            actual = merge_usage_summaries(actual, owners[0]["allocated_usage"]["preparation"])
        expected_actual = _read_json(out / "actual_usage.json")
        if not isinstance(expected_actual, dict):
            raise BenchmarkError("completed actual usage artifact is invalid")
        if expected_actual != actual:
            raise BenchmarkError("completed actual usage does not match unique artifacts")
        return len(payload), payload

    def run(
        self,
        corpus_dir: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        preparation_dir: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
    ) -> dict[str, Any]:
        corpus_root = Path(corpus_dir).expanduser().resolve()
        corpus_sha, rows = _load_corpus_rows(corpus_root)
        spec = load_candidate_spec(candidates_path)
        options = validate_candidate_capabilities(spec)
        bundle, preparation_sha = load_preparation_bundle(preparation_dir)
        if bundle.corpus_sha256 != corpus_sha:
            raise BenchmarkError("preparation corpus_sha256 mismatch")
        formal_books = sorted({row["book_id"] for row in rows if row["subset"] == "continuous"})
        if not formal_books:
            raise BenchmarkError("full run requires continuous formal passages")
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        raw_root = out / "raw"
        branch_root = out / "branches"
        identity_root = out / "identities"
        immutable = {
            "schema_version": 1,
            "run_mode": "full",
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
            "benchmark_id": spec.benchmark_id,
            "spec_sha256": sha256_bytes(
                canonical_json(spec.model_dump(mode="python")).encode("utf-8")
            ),
            "replicates": spec.replicates,
        }
        if (out / "run.json").exists() and _read_json(out / "run.json") != immutable:
            raise BenchmarkError("immutable full run hash mismatch")
        if not (out / "run.json").exists():
            _atomic_json(out / "run.json", immutable)
        state_path = out / "run_state.json"
        state = (
            _read_json(state_path)
            if state_path.exists()
            else {"status": "pending", "artifacts": {}}
        )
        if state.get("status") == "completed":
            expected_count = (
                len(formal_books)
                * spec.replicates
                * sum(candidate.editor_model is not None for candidate in spec.candidates)
            )
            expected_tuples = {
                (candidate.candidate_id, book_id, replicate)
                for candidate in spec.candidates
                if candidate.editor_model is not None
                for book_id in formal_books
                for replicate in range(1, spec.replicates + 1)
            }
            branch_count, _ = self._validate_completed_output(
                out,
                immutable,
                spec,
                bundle,
                expected_count=expected_count,
                expected_tuples=expected_tuples,
            )
            return immutable | {"status": "completed", "branch_count": branch_count}
        _atomic_json(state_path, {"status": "running", "artifacts": state.get("artifacts", {})})
        source = preparation_source(bundle)
        raw_stores: dict[tuple[str, int, str], RunStore] = {}
        branch_outputs: list[dict[str, Any]] = []
        try:
            for book_id in formal_books:
                for replicate in range(1, spec.replicates + 1):
                    document, identity_path, _ = build_continuous_document(
                        corpus_root,
                        benchmark_id=spec.benchmark_id,
                        book_id=book_id,
                        replicate=replicate,
                        identity_dir=identity_root,
                        preparation=bundle,
                    )
                    for primary in sorted(
                        {candidate.primary_model for candidate in spec.candidates}
                    ):
                        key = self._raw_key(
                            spec,
                            primary,
                            corpus_sha,
                            preparation_sha,
                            document,
                            identity_path,
                            book_id,
                            replicate,
                        )
                        raw_dir = raw_root / key / document.title
                        raw_dir.parent.mkdir(parents=True, exist_ok=True)
                        (raw_dir.parent / "telemetry.jsonl").touch(exist_ok=True)
                        sink = _JsonlTelemetrySink(raw_dir.parent / "telemetry.jsonl")
                        client = self._client(spec, primary, "translator", options, sink)
                        config = self._config(
                            spec,
                            primary,
                            primary,
                            quality=False,
                            state_dir=str(raw_dir.parent),
                        )
                        from trans_novel.pipeline.bootstrap import Application
                        from trans_novel.pipeline.contracts import GOAL_TRANSLATE

                        # Always reopen and run the production goal. Planner/
                        # RunStore checkpoints skip completed chapters and
                        # recover a crash-partial raw store under its advisory lock.
                        app = Application(config, client=client, frozen_preparation=source)
                        _, raw_store = app.run_document_goal(
                            document, identity_path, GOAL_TRANSLATE
                        )
                        _atomic_json(
                            raw_dir.parent / "manifest.json",
                            {
                                "schema_version": 1,
                                "kind": "raw",
                                "artifact_key": key,
                                "provider": spec.provider,
                                "primary_model": primary,
                                "editor_model": None,
                                "fast_model": spec.fast_model,
                                "generation": dict(_GENERATION_FIELDS),
                                "corpus_sha256": corpus_sha,
                                "preparation_sha256": preparation_sha,
                                "document": document.model_copy(
                                    update={"source_path": ""}
                                ).model_dump(mode="python"),
                                "identity_sha256": sha256_bytes(Path(identity_path).read_bytes()),
                                "book_id": book_id,
                                "replicate": replicate,
                            },
                        )
                        if Path(raw_store.journal_path).exists():
                            raise BenchmarkError("raw translation store has a pending journal")
                        raw_stores[(book_id, replicate, primary)] = raw_store
                        raw_records = self._stage_records(raw_store)
                        for candidate in spec.candidates:
                            if candidate.primary_model != primary or candidate.editor_model is None:
                                continue
                            branch_key = sha256_bytes(
                                canonical_json(
                                    {
                                        "raw_artifact": key,
                                        "editor_model": candidate.editor_model,
                                        "fast_model": spec.fast_model,
                                    }
                                ).encode("utf-8")
                            )
                            branch_dir = branch_root / branch_key / document.title
                            if not branch_dir.exists():
                                clone_closed_runstore(raw_store.run_dir, str(branch_dir))
                            (branch_dir.parent / "telemetry.jsonl").touch(exist_ok=True)
                            sink = _JsonlTelemetrySink(branch_dir.parent / "telemetry.jsonl")
                            branch_client = self._client(
                                spec,
                                candidate.editor_model,
                                "editor",
                                options,
                                sink,
                                roles=ModelRoles(
                                    primary=primary,
                                    editor=candidate.editor_model,
                                    fast=spec.fast_model,
                                ),
                            )
                            config = self._config(
                                spec,
                                primary,
                                candidate.editor_model,
                                quality=True,
                                state_dir=str(branch_dir.parent),
                            )
                            from trans_novel.pipeline.bootstrap import Application
                            from trans_novel.pipeline.contracts import GOAL_RUN_ALL, ExecutionGoal

                            app = Application(
                                config,
                                client=branch_client,
                                frozen_preparation=source,
                                backtranslation_sample_scope=canonical_json(
                                    {
                                        "benchmark_id": spec.benchmark_id,
                                        "book_id": book_id,
                                        "replicate": replicate,
                                    }
                                ),
                            )
                            branch_goal = ExecutionGoal(
                                name="run_all",
                                phases=GOAL_RUN_ALL.phases,
                                out_format="txt",
                                out_path=str(branch_dir / "output.txt"),
                            )
                            _, branch_store = app.run_document_goal(
                                document,
                                identity_path,
                                branch_goal,
                            )
                            if not self._store_ready(branch_store):
                                raise BenchmarkError("quality branch store is incomplete")
                            _atomic_json(
                                branch_dir.parent / "manifest.json",
                                {
                                    "schema_version": 1,
                                    "kind": "branch",
                                    "artifact_key": branch_key,
                                    "raw_artifact": key,
                                    "provider": spec.provider,
                                    "primary_model": primary,
                                    "editor_model": candidate.editor_model,
                                    "fast_model": spec.fast_model,
                                    "generation": dict(_GENERATION_FIELDS),
                                    "corpus_sha256": corpus_sha,
                                    "preparation_sha256": preparation_sha,
                                    "document": document.model_copy(
                                        update={"source_path": ""}
                                    ).model_dump(mode="python"),
                                    "identity_sha256": sha256_bytes(
                                        Path(identity_path).read_bytes()
                                    ),
                                    "book_id": book_id,
                                    "replicate": replicate,
                                },
                            )
                            if any(
                                str(record.get("operation", ""))
                                in {"translate.batch", "translate.lint_fix"}
                                for record in sink.records
                            ):
                                raise BenchmarkError(
                                    "quality branch retranslated synthetic document"
                                )
                            final_records = self._stage_records(branch_store)
                            final_by_id = {record["segment_id"]: record for record in final_records}
                            raw_snapshot = (
                                _read_json(Path(raw_store.usage_path))
                                if Path(raw_store.usage_path).exists()
                                else {}
                            )
                            stage = []
                            for raw in raw_records:
                                final = final_by_id.get(raw["segment_id"])
                                if final is None:
                                    raise BenchmarkError("branch lost a synthetic segment")
                                stage_record = {
                                    **raw,
                                    "final_after_full_pipeline": final["final_after_full_pipeline"],
                                    "review_findings": final["review_findings"],
                                    "lint_findings": final["lint_findings"],
                                    "backtranslation_findings": final["backtranslation_findings"],
                                    "raw_artifact_id": key,
                                    "branch_artifact_id": branch_key,
                                    "preparation_sha256": preparation_sha,
                                    "primary_model": primary,
                                    "editor_model": candidate.editor_model,
                                    "fast_model": spec.fast_model,
                                }
                                stage_record["final_sha256"] = sha256_bytes(
                                    str(stage_record["final_after_full_pipeline"]).encode()
                                )
                                stage.append(stage_record)
                            branch_usage = (
                                _read_json(Path(branch_store.usage_path))
                                if Path(branch_store.usage_path).exists()
                                else {}
                            )
                            branch_increment = usage_delta(branch_usage, raw_snapshot)
                            preparation_allocation_id = self._preparation_allocation_id(
                                candidate.candidate_id,
                                book_id,
                                preparation_sha,
                            )
                            branch_outputs.append(
                                {
                                    "candidate_id": candidate.candidate_id,
                                    "book_id": book_id,
                                    "replicate": replicate,
                                    "provider": spec.provider,
                                    "corpus_sha256": corpus_sha,
                                    "preparation_sha256": preparation_sha,
                                    "primary_model": primary,
                                    "editor_model": candidate.editor_model,
                                    "fast_model": spec.fast_model,
                                    "raw_artifact_id": key,
                                    "branch_artifact_id": branch_key,
                                    "preparation_allocation_id": preparation_allocation_id,
                                    "raw_telemetry_sha256": sha256_bytes(
                                        (raw_dir.parent / "telemetry.jsonl").read_bytes()
                                    ),
                                    "branch_telemetry_sha256": sha256_bytes(
                                        (branch_dir.parent / "telemetry.jsonl").read_bytes()
                                    ),
                                    "zero_retranslation": True,
                                    "stage": stage,
                                    "allocated_usage": {
                                        "preparation": bundle.books[book_id].usage
                                        if replicate == 1
                                        else {},
                                        "raw": raw_snapshot,
                                        "branch_increment": branch_increment,
                                    },
                                }
                            )
                            state["artifacts"][branch_key] = "completed"
                            _atomic_json(
                                state_path,
                                {"status": "running", "artifacts": state["artifacts"]},
                            )
            actual: dict[str, Any] = {}
            counted_preparation: set[str] = set()
            counted_raw: set[str] = set()
            counted_branch: set[str] = set()
            for row in branch_outputs:
                if row["preparation_allocation_id"] not in counted_preparation:
                    actual = merge_usage_summaries(actual, row["allocated_usage"]["preparation"])
                    counted_preparation.add(row["preparation_allocation_id"])
                if row["raw_artifact_id"] not in counted_raw:
                    actual = merge_usage_summaries(actual, row["allocated_usage"]["raw"])
                    counted_raw.add(row["raw_artifact_id"])
                if row["branch_artifact_id"] not in counted_branch:
                    actual = merge_usage_summaries(
                        actual, row["allocated_usage"]["branch_increment"]
                    )
                    counted_branch.add(row["branch_artifact_id"])
            _atomic_json(out / "actual_usage.json", actual)
            _atomic_json(out / "candidates.json", branch_outputs)
            _atomic_json(state_path, {"status": "completed", "artifacts": state["artifacts"]})
            return immutable | {"status": "completed", "branch_count": len(branch_outputs)}
        except Exception as error:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "last_error_kind": type(error).__name__,
                    "artifacts": state["artifacts"],
                },
            )
            if isinstance(error, BenchmarkError):
                raise
            raise BenchmarkError(str(error)) from error


class AttributionRunner:
    def __init__(
        self, *, client_factory: Callable[..., Any] | None = None, client: Any | None = None
    ) -> None:
        self.client_factory = client_factory
        self.client = client

    def _client(
        self,
        spec: CandidateSpec,
        model: str,
        role: str,
        options: GenerationOptions,
        sink: _JsonlTelemetrySink,
    ) -> Any:
        if self.client is not None:
            client = self.client
            _attach_sink(client, sink, required=True)
            return client
        return _model_client(spec, model, role, options, self.client_factory, sink)

    @contextmanager
    def _run_lock(self, out: Path) -> Iterator[None]:
        out.mkdir(parents=True, exist_ok=True)
        lock_path = out / ".run.lock"
        with lock_path.open("a+b") as lock_file:
            if os.name == "nt":  # pragma: no cover
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def run(
        self,
        corpus_dir: str | os.PathLike[str],
        candidates_path: str | os.PathLike[str],
        preparation_path: str | os.PathLike[str],
        out_dir: str | os.PathLike[str],
        *,
        mode: str = "attribution",
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        if mode not in {"attribution", "canary"}:
            raise BenchmarkError(f"unknown run mode: {mode}")
        corpus_root = Path(corpus_dir).expanduser().resolve()
        out = Path(out_dir).expanduser().resolve()
        spec = load_candidate_spec(candidates_path)
        options = validate_candidate_capabilities(spec)
        corpus_sha, all_rows = _load_corpus_rows(corpus_root)
        rows = [
            row
            for row in all_rows
            if row["subset"] in {"screen", "continuous", "stratified", "context"}
        ]
        bundle, preparation_sha = load_preparation_bundle(preparation_path)
        if bundle.corpus_sha256 != corpus_sha:
            raise BenchmarkError("preparation corpus_sha256 mismatch")
        selected_sample_id: str | None = None
        if mode == "canary":
            screens = [row for row in rows if row["subset"] == "screen"]
            selected_sample_id = sample_id or (screens[0]["passage_id"] if screens else None)
            if selected_sample_id is None or not any(
                row["passage_id"] == selected_sample_id for row in screens
            ):
                raise BenchmarkError("canary sample_id must resolve to a screen passage")
        for row in rows:
            _prep_for(bundle, row)
        spec_value = spec.model_dump(mode="python")
        spec_hash_value = (
            {"spec": spec_value, "canary_sample_id": selected_sample_id}
            if mode == "canary"
            else spec_value
        )
        immutable = {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_mode": mode,
            "prompt_version": PROMPT_VERSION,
            "benchmark_id": spec.benchmark_id,
            "spec_sha256": sha256_bytes(canonical_json(spec_hash_value).encode()),
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
            "canary_sample_id": selected_sample_id,
        }
        with self._run_lock(out):
            run_path, state_path = out / "run.json", out / "run_state.json"
            if run_path.exists():
                if _read_json(run_path) != immutable:
                    raise BenchmarkError("immutable run/spec hash mismatch")
            else:
                _atomic_json(run_path, immutable)
            state = _read_json(state_path) if state_path.exists() else {"status": "pending"}
            if state.get("status") == "completed":
                self._validate_completed(
                    out,
                    spec,
                    options,
                    bundle,
                    corpus_sha,
                    preparation_sha,
                    rows,
                    mode,
                    selected_sample_id,
                )
                completed_result = immutable | {"status": "completed"}
                if mode == "attribution":
                    manifests = _read_json(out / "candidates.json")
                    completed_result["candidate_count"] = len(manifests)
                return completed_result
            _atomic_json(state_path, {"status": "running"})
            try:
                result = (
                    self._canary(
                        out,
                        spec,
                        options,
                        bundle,
                        corpus_sha,
                        preparation_sha,
                        rows,
                        selected_sample_id,
                    )
                    if mode == "canary"
                    else self._attribution(
                        out, spec, options, bundle, corpus_sha, preparation_sha, rows
                    )
                )
                _atomic_json(state_path, {"status": "completed"})
                return immutable | result | {"status": "completed"}
            except Exception as error:
                _atomic_json(
                    state_path, {"status": "failed", "last_error_kind": type(error).__name__}
                )
                if isinstance(error, BenchmarkError):
                    raise
                raise BenchmarkError(str(error)) from error

    def _passage_set(
        self, rows: list[dict[str, Any]], subset: str, strategy: str
    ) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["subset"] == subset and (strategy == "c2" or subset == "context")
        ]

    def _artifact_key(
        self,
        spec: CandidateSpec,
        options: GenerationOptions,
        corpus_sha: str,
        preparation_sha: str,
        primary: str,
        replicate: int,
        subset: str,
        strategy: str,
        passage_ids: list[str],
    ) -> str:
        value = {
            "schema_version": 1,
            "prompt_version": PROMPT_VERSION,
            "provider": spec.provider,
            "primary_model": primary,
            "generation": dict(_GENERATION_FIELDS),
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
            "replicate": replicate,
            "scope": {"subset": subset, "context_strategy": strategy, "passage_ids": passage_ids},
        }
        return sha256_bytes(canonical_json(value).encode())

    def _context_for_batch(
        self, row: dict[str, Any], strategy: str, subset: str, rolling: RollingContext | None
    ) -> TranslationContextBundle:
        if strategy in {"c0", "c1"}:
            return _context_for(row, strategy)
        if subset == "continuous":
            return TranslationContextBundle(
                target_before=rolling.render(6) if rolling is not None else ""
            )
        return _context_for(row, strategy)

    def _contexts_for_passage(
        self,
        row: dict[str, Any],
        strategy: str,
        subset: str,
        prep: BookPreparation,
        rolling: RollingContext | None,
    ) -> tuple[list[TranslationContextBundle], list[dict[str, Any]]]:
        contexts: list[TranslationContextBundle] = []
        segment_objects = [SimpleNamespace(**segment) for segment in row["segments"]]
        for _ in batch_segments(segment_objects, 1800):
            context = self._context_for_batch(row, strategy, subset, rolling)
            contexts.append(context)
        prep_value = prep if strategy == "c2" else None
        return contexts, [
            {
                "context": _context_dict(x),
                "preparation": prep_value.model_dump(mode="python") if prep_value else None,
            }
            for x in contexts
        ]

    def _compute_usage(self, artifact: Path) -> dict[str, Any]:
        total: dict[str, Any] = {}
        passages = artifact / "passages"
        if passages.exists():
            for path in sorted(passages.glob("*.json")):
                passage = _read_json(path)
                if passage.get("status") != "complete":
                    continue
                delta = passage.get("usage_delta")
                _validate_usage_delta(delta)
                total = merge_usage_summaries(total, delta)
        journal_path = artifact / "journal.json"
        if journal_path.exists():
            journal = _read_json(journal_path)
            batches = journal.get("batches") if isinstance(journal, dict) else None
            if not isinstance(batches, list):
                raise BenchmarkError("corrupt or incomplete benchmark journal")
            for batch in batches:
                delta = batch.get("usage_delta") if isinstance(batch, dict) else None
                _validate_usage_delta(delta)
                total = merge_usage_summaries(total, delta)
        return total

    def _rebuild_usage(self, artifact: Path) -> dict[str, Any]:
        total = self._compute_usage(artifact)
        _atomic_json(artifact / "usage.json", total)
        return total

    def _hydrate_validated_rolling(
        self,
        artifact: Path,
        rows: list[dict[str, Any]],
        target_passage_id: str,
        spec: CandidateSpec,
        bundle: PreparationBundle,
        artifact_key: str,
        replicate: int,
        subset: str,
        strategy: str,
        model: str,
        preparation_sha256: str,
    ) -> tuple[RollingContext, str | None]:
        rolling = RollingContext(max_recent_keep=6)
        current_book: str | None = None
        for row in rows:
            if row["passage_id"] == target_passage_id:
                break
            if subset == "continuous" and row["book_id"] != current_book:
                rolling = RollingContext(max_recent_keep=6)
                current_book = row["book_id"]
            path = artifact / "passages" / f"{_safe_id(row['passage_id'])}.json"
            if not path.exists():
                raise BenchmarkError(f"missing translation passage {row['passage_id']}")
            passage = _read_json(path)
            prep = _prep_for(bundle, row)
            segment_objects = [SimpleNamespace(**segment) for segment in row["segments"]]
            contexts: list[TranslationContextBundle] = []
            offset = 0
            for batch in batch_segments(segment_objects, 1800):
                context = self._context_for_batch(
                    row, strategy, subset, rolling if subset == "continuous" else None
                )
                contexts.append(context)
                offset += len(batch)
                if subset == "continuous":
                    rolling.add_targets(
                        [
                            f"[{record['segment_id']}] {record['final']}"
                            for record in passage["segments"][offset - len(batch) : offset]
                        ]
                    )
            terms = _glossary(prep.glossary) if strategy == "c2" else []
            digest = prep.chapter_digests[str(row["chapter_index"])] if strategy == "c2" else ""
            expected_context_hash = _context_hash(
                contexts,
                prep if strategy == "c2" else None,
                chapter_digest=digest,
                glossary_terms=terms,
            )
            if passage.get("batch_context_hashes") != [
                _batch_context_hash(context) for context in contexts
            ]:
                raise BenchmarkError("translation batch context provenance mismatch")
            _validate_passage_common(
                passage,
                row,
                artifact_key=artifact_key,
                spec=spec,
                replicate=replicate,
                subset=subset,
                strategy=strategy,
                model=model,
                context_hash=expected_context_hash,
                preparation_sha256=preparation_sha256,
            )
        return rolling, current_book

    def _validate_translation_journal_prefix(
        self,
        artifact: Path,
        journal: dict[str, Any],
        rows: list[dict[str, Any]],
        spec: CandidateSpec,
        bundle: PreparationBundle,
        artifact_key: str,
        replicate: int,
        subset: str,
        strategy: str,
        model: str,
        preparation_sha256: str,
    ) -> None:
        passage_id = journal.get("passage_id")
        row = next((candidate for candidate in rows if candidate["passage_id"] == passage_id), None)
        batches = journal.get("batches")
        if row is None or not isinstance(batches, list):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        expected_batches = batch_segments([SimpleNamespace(**x) for x in row["segments"]], 1800)
        if len(batches) > len(expected_batches):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        rolling, current_book = self._hydrate_validated_rolling(
            artifact,
            rows,
            passage_id,
            spec,
            bundle,
            artifact_key,
            replicate,
            subset,
            strategy,
            model,
            preparation_sha256,
        )
        if subset == "continuous" and row["book_id"] != current_book:
            rolling = RollingContext(max_recent_keep=6)
        prep = _prep_for(bundle, row)
        expected_preparation = prep.model_dump(mode="python") if strategy == "c2" else None
        for index, batch in enumerate(batches):
            if (
                not isinstance(batch, dict)
                or batch.get("index") != index
                or not isinstance(batch.get("segments"), list)
            ):
                raise BenchmarkError("corrupt or incomplete benchmark journal")
            expected_context = self._context_for_batch(
                row, strategy, subset, rolling if subset == "continuous" else None
            )
            if (
                batch.get("context") != _context_dict(expected_context)
                or batch.get("preparation") != expected_preparation
            ):
                raise BenchmarkError("translation journal context provenance mismatch")
            expected_segments = expected_batches[index]
            records = batch["segments"]
            if len(records) != len(expected_segments):
                raise BenchmarkError("corrupt or incomplete benchmark journal")
            for segment, record in zip(expected_segments, records, strict=True):
                if (
                    record.get("segment_id") != segment.segment_id
                    or record.get("source") != segment.source
                ):
                    raise BenchmarkError("translation journal source mismatch")
                _validate_segment_hashes(record, editor=False)

    def _recover_translation_journal(
        self,
        artifact: Path,
        artifact_key: str,
        rows: list[dict[str, Any]],
        spec: CandidateSpec,
        replicate: int,
        subset: str,
        strategy: str,
        model: str,
        bundle: PreparationBundle,
        preparation_sha256: str,
    ) -> None:
        journal_path = artifact / "journal.json"
        if not journal_path.exists():
            return
        journal = _read_json(journal_path)
        if not isinstance(journal, dict):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        passage_id = journal.get("passage_id")
        rows_by_id = {row["passage_id"]: row for row in rows}
        row = rows_by_id.get(passage_id)
        batches = journal.get("batches")
        if journal.get("kind") != "translation" or row is None or not isinstance(batches, list):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        expected_batches = batch_segments([SimpleNamespace(**x) for x in row["segments"]], 1800)
        if (
            journal.get("batch_count") != len(expected_batches)
            or journal.get("artifact_key") != artifact_key
        ):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        self._validate_translation_journal_prefix(
            artifact,
            journal,
            rows,
            spec,
            bundle,
            artifact_key,
            replicate,
            subset,
            strategy,
            model,
            preparation_sha256,
        )
        batches = journal.get("batches") if isinstance(journal, dict) else None
        if journal.get("kind") != "translation" or row is None or not isinstance(batches, list):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        expected_batches = batch_segments([SimpleNamespace(**x) for x in row["segments"]], 1800)
        if (
            journal.get("batch_count") != len(expected_batches)
            or journal.get("artifact_key") != artifact_key
        ):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        if [b.get("index") for b in batches] != list(range(len(batches))):
            raise BenchmarkError("corrupt or incomplete benchmark journal")
        for index, batch in enumerate(batches):
            if not isinstance(batch, dict) or not isinstance(batch.get("segments"), list):
                raise BenchmarkError("corrupt or incomplete benchmark journal")
            expected_ids = [x.segment_id for x in expected_batches[index]]
            if [x.get("segment_id") for x in batch["segments"]] != expected_ids:
                raise BenchmarkError("corrupt or incomplete benchmark journal")
            for record in batch["segments"]:
                _validate_segment_hashes(record, editor=False)
            _validate_usage_delta(batch.get("usage_delta"))
        if len(batches) < len(expected_batches):
            self._rebuild_usage(artifact)
            return
        prep = _prep_for(bundle, row)
        contexts = [TranslationContextBundle(**batch["context"]) for batch in batches]
        digest = prep.chapter_digests[str(row["chapter_index"])] if strategy == "c2" else ""
        terms = _glossary(prep.glossary) if strategy == "c2" else []
        context_hash = _context_hash(
            contexts,
            prep if strategy == "c2" else None,
            chapter_digest=digest,
            glossary_terms=terms,
        )
        records = [record for batch in batches for record in batch["segments"]]
        delta: dict[str, Any] = {}
        for batch in batches:
            delta = merge_usage_summaries(delta, batch["usage_delta"])
        passage = {
            "status": "complete",
            "kind": "translation",
            "artifact_key": artifact_key,
            "passage_id": row["passage_id"],
            "subset": subset,
            "book_id": row["book_id"],
            "chapter_index": row["chapter_index"],
            "replicate": replicate,
            "context_strategy": strategy,
            "provider": spec.provider,
            "primary_model": model,
            "source_hash": _source_hash(row["segments"]),
            "context_hash": context_hash,
            "preparation_sha256": preparation_sha256,
            "batch_context_hashes": [_batch_context_hash(context) for context in contexts],
            "segments": records,
            "usage_delta": delta,
        }
        _validate_passage_common(
            passage,
            row,
            artifact_key=artifact_key,
            spec=spec,
            replicate=replicate,
            subset=subset,
            strategy=strategy,
            model=model,
            context_hash=context_hash,
            preparation_sha256=preparation_sha256,
        )
        _atomic_json(artifact / "passages" / f"{_safe_id(passage_id)}.json", passage)
        journal_path.unlink(missing_ok=True)
        self._rebuild_usage(artifact)

    def _validate_translation_artifact(
        self,
        artifact: Path,
        spec: CandidateSpec,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        primary: str,
        replicate: int,
        subset: str,
        strategy: str,
        rows: list[dict[str, Any]],
        artifact_key: str,
    ) -> None:
        manifest = _stage_manifest(
            artifact_key=artifact_key,
            kind="translation",
            spec=spec,
            model=primary,
            corpus_sha=corpus_sha,
            preparation_sha=preparation_sha,
            replicate=replicate,
            subset=subset,
            strategy=strategy,
            passage_ids=[r["passage_id"] for r in rows],
        )
        if (
            not artifact.exists()
            or _read_json(artifact / "manifest.json") != manifest
            or (artifact / "journal.json").exists()
        ):
            raise BenchmarkError("translation artifact integrity failure")
        rolling = RollingContext(max_recent_keep=6)
        current_book: str | None = None
        for row in rows:
            if subset == "continuous" and row["book_id"] != current_book:
                rolling = RollingContext(max_recent_keep=6)
                current_book = row["book_id"]
            path = artifact / "passages" / f"{_safe_id(row['passage_id'])}.json"
            if not path.exists():
                raise BenchmarkError(f"missing translation passage {row['passage_id']}")
            prep = _prep_for(bundle, row)
            passage = _read_json(path)
            segment_objects = [SimpleNamespace(**segment) for segment in row["segments"]]
            contexts: list[TranslationContextBundle] = []
            offset = 0
            for batch in batch_segments(segment_objects, 1800):
                contexts.append(
                    self._context_for_batch(
                        row, strategy, subset, rolling if subset == "continuous" else None
                    )
                )
                offset += len(batch)
                if subset == "continuous":
                    rolling.add_targets(
                        [
                            f"[{record['segment_id']}] {record['final']}"
                            for record in passage["segments"][offset - len(batch) : offset]
                        ]
                    )
            terms = _glossary(prep.glossary) if strategy == "c2" else []
            digest = prep.chapter_digests[str(row["chapter_index"])] if strategy == "c2" else ""
            expected_context_hash = _context_hash(
                contexts,
                prep if strategy == "c2" else None,
                chapter_digest=digest,
                glossary_terms=terms,
            )
            if passage.get("batch_context_hashes") != [
                _batch_context_hash(context) for context in contexts
            ]:
                raise BenchmarkError("translation batch context provenance mismatch")
            _validate_passage_common(
                passage,
                row,
                artifact_key=artifact_key,
                spec=spec,
                replicate=replicate,
                subset=subset,
                strategy=strategy,
                model=primary,
                context_hash=expected_context_hash,
                preparation_sha256=preparation_sha,
            )
        stored_usage = _read_json(artifact / "usage.json")
        _validate_usage_delta(stored_usage)
        rebuilt = self._compute_usage(artifact)
        if rebuilt != stored_usage:
            raise BenchmarkError("translation usage does not match completed passage deltas")

    def _translate_artifact(
        self,
        root: Path,
        spec: CandidateSpec,
        options: GenerationOptions,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        primary: str,
        replicate: int,
        subset: str,
        strategy: str,
        rows: list[dict[str, Any]],
        artifact_key: str,
    ) -> Path:
        artifact = root / "translation" / artifact_key
        artifact.mkdir(parents=True, exist_ok=True)
        manifest = _stage_manifest(
            artifact_key=artifact_key,
            kind="translation",
            spec=spec,
            model=primary,
            corpus_sha=corpus_sha,
            preparation_sha=preparation_sha,
            replicate=replicate,
            subset=subset,
            strategy=strategy,
            passage_ids=[r["passage_id"] for r in rows],
        )
        if (artifact / "manifest.json").exists() and _read_json(
            artifact / "manifest.json"
        ) != manifest:
            raise BenchmarkError("translation manifest mismatch")
        _atomic_json(artifact / "manifest.json", manifest)
        if not (artifact / "usage.json").exists():
            _atomic_json(artifact / "usage.json", usage_delta({}, {}))
        sink = _JsonlTelemetrySink(artifact / "telemetry.jsonl")
        self._recover_translation_journal(
            artifact,
            artifact_key,
            rows,
            spec,
            replicate,
            subset,
            strategy,
            primary,
            bundle,
            preparation_sha,
        )
        client = self._client(spec, primary, "translator", options, sink)
        translator = Translator(
            client,
            Config(
                llm=LLMConfig(
                    provider=spec.provider,
                    models=ModelRoles(primary=primary, editor=primary, fast=spec.fast_model),
                ),
                source_lang="en",
                target_lang="zh",
            ),
        )
        rolling = RollingContext(max_recent_keep=6)
        current_book: str | None = None
        for row in rows:
            if subset == "continuous" and row["book_id"] != current_book:
                rolling = RollingContext(max_recent_keep=6)
                current_book = row["book_id"]
            passage_path = artifact / "passages" / f"{_safe_id(row['passage_id'])}.json"
            prep = _prep_for(bundle, row)
            segment_objects = [SimpleNamespace(**segment) for segment in row["segments"]]
            batches = batch_segments(segment_objects, 1800)
            if passage_path.exists():
                existing = _read_json(passage_path)
                contexts: list[TranslationContextBundle] = []
                offset = 0
                for batch in batches:
                    contexts.append(
                        self._context_for_batch(
                            row, strategy, subset, rolling if subset == "continuous" else None
                        )
                    )
                    offset += len(batch)
                    if subset == "continuous":
                        rolling.add_targets(
                            [
                                f"[{record['segment_id']}] {record['final']}"
                                for record in existing["segments"][offset - len(batch) : offset]
                            ]
                        )
                terms = _glossary(prep.glossary) if strategy == "c2" else []
                digest = prep.chapter_digests[str(row["chapter_index"])] if strategy == "c2" else ""
                context_hash = _context_hash(
                    contexts,
                    prep if strategy == "c2" else None,
                    chapter_digest=digest,
                    glossary_terms=terms,
                )
                if existing.get("batch_context_hashes") != [
                    _batch_context_hash(context) for context in contexts
                ]:
                    raise BenchmarkError("translation batch context provenance mismatch")
                _validate_passage_common(
                    existing,
                    row,
                    artifact_key=artifact_key,
                    spec=spec,
                    replicate=replicate,
                    subset=subset,
                    strategy=strategy,
                    model=primary,
                    context_hash=context_hash,
                    preparation_sha256=preparation_sha,
                )
                continue
            journal_path = artifact / "journal.json"
            committed: list[dict[str, Any]] = []
            if journal_path.exists():
                journal = _read_json(journal_path)
                if (
                    journal.get("passage_id") != row["passage_id"]
                    or journal.get("batch_count") != len(batches)
                    or journal.get("artifact_key") != artifact_key
                ):
                    raise BenchmarkError("corrupt or incomplete benchmark journal")
                committed = journal.get("batches", [])
            all_contexts = [TranslationContextBundle(**entry["context"]) for entry in committed]
            all_records = [record for entry in committed for record in entry["segments"]]
            for entry in committed:
                for record in entry["segments"]:
                    _validate_segment_hashes(record, editor=False)
                if subset == "continuous":
                    rolling.add_targets(
                        [
                            f"[{record['segment_id']}] {record['final']}"
                            for record in entry["segments"]
                        ]
                    )
            terms = _glossary(prep.glossary) if strategy == "c2" else []
            style = prep.style if strategy == "c2" else ""
            synopsis = prep.book_synopsis if strategy == "c2" else ""
            digest = prep.chapter_digests[str(row["chapter_index"])] if strategy == "c2" else ""
            for batch_index, batch in enumerate(batches):
                if batch_index < len(committed):
                    continue
                context = self._context_for_batch(
                    row, strategy, subset, rolling if subset == "continuous" else None
                )
                before_usage = _usage_of(client)
                targets = translator.translate_batch(
                    [s.source for s in batch],
                    agent="translator",
                    operation="translate.batch",
                    glossary_terms=terms,
                    style=style,
                    book_synopsis=synopsis,
                    chapter_digest=digest,
                    context_bundle=context,
                )
                delta = _passage_usage_delta(client, before_usage)
                issues = lint.lint_targets(
                    [s.source for s in batch],
                    targets,
                    locked_terms=_locked_terms(terms),
                    src_lang="en",
                )
                by_index: dict[int, list[dict[str, Any]]] = {}
                for issue in issues:
                    by_index.setdefault(issue.index, []).append(
                        {"type": issue.type, "detail": issue.detail}
                    )
                batch_records: list[dict[str, Any]] = []
                for index, (segment, target) in enumerate(zip(batch, targets, strict=True)):
                    record = {
                        "segment_id": segment.segment_id,
                        "source": segment.source,
                        "translation_raw": target,
                        "translation_after_lint": target,
                        "translation_lint_issues": by_index.get(index, []),
                        "polish_proposal": None,
                        "polish_accepted": None,
                        "polish_rejection_reasons": [],
                        "final": target,
                    }
                    _hash_record(record)
                    batch_records.append(record)
                _validate_usage_delta(delta)
                committed.append(
                    {
                        "index": batch_index,
                        "context": _context_dict(context),
                        "segments": batch_records,
                        "usage_delta": delta,
                    }
                )
                _atomic_json(
                    journal_path,
                    {
                        "kind": "translation",
                        "artifact_key": artifact_key,
                        "passage_id": row["passage_id"],
                        "batch_count": len(batches),
                        "batches": [
                            {
                                **entry,
                                "preparation": prep.model_dump(mode="python")
                                if strategy == "c2"
                                else None,
                            }
                            for entry in committed
                        ],
                    },
                )
                self._rebuild_usage(artifact)
                all_contexts.append(context)
                all_records.extend(batch_records)
                if subset == "continuous":
                    rolling.add_targets(
                        [
                            f"[{segment.segment_id}] {target}"
                            for segment, target in zip(batch, targets, strict=True)
                        ]
                    )
            total_delta: dict[str, Any] = {}
            for entry in committed:
                total_delta = merge_usage_summaries(total_delta, entry["usage_delta"])
            context_hash = _context_hash(
                all_contexts,
                prep if strategy == "c2" else None,
                chapter_digest=digest,
                glossary_terms=terms,
            )
            passage = {
                "status": "complete",
                "kind": "translation",
                "artifact_key": artifact_key,
                "passage_id": row["passage_id"],
                "subset": subset,
                "book_id": row["book_id"],
                "chapter_index": row["chapter_index"],
                "replicate": replicate,
                "context_strategy": strategy,
                "provider": spec.provider,
                "primary_model": primary,
                "source_hash": _source_hash(row["segments"]),
                "context_hash": context_hash,
                "preparation_sha256": preparation_sha,
                "batch_context_hashes": [_batch_context_hash(context) for context in all_contexts],
                "segments": all_records,
                "usage_delta": total_delta,
            }
            _validate_passage_common(
                passage,
                row,
                artifact_key=artifact_key,
                spec=spec,
                replicate=replicate,
                subset=subset,
                strategy=strategy,
                model=primary,
                context_hash=context_hash,
                preparation_sha256=preparation_sha,
            )
            _atomic_json(passage_path, passage)
            journal_path.unlink(missing_ok=True)
            self._rebuild_usage(artifact)
        self._rebuild_usage(artifact)
        return artifact

    def _editor_key(self, options: GenerationOptions, editor: str, translation_key: str) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "schema_version": 1,
                    "editor_model": editor,
                    "translation_artifact_key": translation_key,
                    "generation": dict(_GENERATION_FIELDS),
                }
            ).encode()
        )

    def _recover_editor_journal(
        self,
        artifact: Path,
        translation_artifact: Path,
        key: str,
        editor: str,
        spec: CandidateSpec,
        bundle: PreparationBundle,
        translation_rows: dict[str, dict[str, Any]],
    ) -> None:
        journal_path = artifact / "journal.json"
        if not journal_path.exists():
            return
        journal = _read_json(journal_path)
        if not isinstance(journal, dict):
            raise BenchmarkError("corrupt or incomplete editor journal")
        passage_id = journal.get("passage_id")
        source = translation_rows.get(passage_id)
        batches = journal.get("batches")
        if journal.get("kind") != "editor" or source is None or not isinstance(batches, list):
            raise BenchmarkError("corrupt or incomplete editor journal")
        expected_batches = batch_segments([SimpleNamespace(**x) for x in source["segments"]], 1800)
        if (
            journal.get("batch_count") != len(expected_batches)
            or journal.get("artifact_key") != key
        ):
            raise BenchmarkError("corrupt or incomplete editor journal")
        self._validate_editor_journal_prefix(journal, source, spec, bundle, key, editor)
        if len(batches) < len(expected_batches):
            self._rebuild_usage(artifact)
            return
        records = [record for batch in batches for record in batch["segments"]]
        delta: dict[str, Any] = {}
        for batch in batches:
            delta = merge_usage_summaries(delta, batch["usage_delta"])
        row = dict(source)
        row["artifact_key"] = translation_artifact.name
        output = dict(source)
        output.update(
            {
                "kind": "editor",
                "artifact_key": key,
                "translation_artifact_key": translation_artifact.name,
                "editor_model": editor,
                "segments": records,
                "usage_delta": delta,
            }
        )
        prep = _prep_for(bundle, row)
        _validate_passage_common(
            output,
            row,
            artifact_key=key,
            spec=spec,
            replicate=source["replicate"],
            subset=source["subset"],
            strategy=source["context_strategy"],
            model=editor,
            context_hash=source["context_hash"],
            editor=True,
            editor_key=key,
            baseline_segments=source["segments"],
            prep=prep,
        )
        _atomic_json(artifact / "passages" / f"{_safe_id(passage_id)}.json", output)
        journal_path.unlink(missing_ok=True)
        self._rebuild_usage(artifact)

    def _validate_editor_journal_prefix(
        self,
        journal: dict[str, Any],
        source: dict[str, Any],
        spec: CandidateSpec,
        bundle: PreparationBundle,
        key: str,
        editor: str,
    ) -> None:
        batches = journal.get("batches")
        if not isinstance(batches, list):
            raise BenchmarkError("corrupt or incomplete editor journal")
        expected_batches = batch_segments([SimpleNamespace(**x) for x in source["segments"]], 1800)
        if len(batches) > len(expected_batches):
            raise BenchmarkError("corrupt or incomplete editor journal")
        prep = _prep_for(bundle, source)
        baseline_by_id = {record["segment_id"]: record for record in source["segments"]}
        for index, batch in enumerate(batches):
            if (
                not isinstance(batch, dict)
                or batch.get("index") != index
                or not isinstance(batch.get("segments"), list)
            ):
                raise BenchmarkError("corrupt or incomplete editor journal")
            expected_segments = expected_batches[index]
            records = batch["segments"]
            if len(records) != len(expected_segments):
                raise BenchmarkError("corrupt or incomplete editor journal")
            for segment, record in zip(expected_segments, records, strict=True):
                baseline = baseline_by_id.get(segment.segment_id)
                if (
                    baseline is None
                    or record.get("segment_id") != segment.segment_id
                    or record.get("source") != segment.source
                ):
                    raise BenchmarkError("editor journal source mismatch")
                for field in (
                    "segment_id",
                    "source",
                    "translation_raw",
                    "translation_after_lint",
                    "translation_lint_issues",
                    "translation_raw_sha256",
                    "translation_after_lint_sha256",
                ):
                    if record.get(field) != baseline.get(field):
                        raise BenchmarkError("editor journal changed its translation baseline")
                _validate_segment_hashes(record, editor=True)
                gate = lint.polish_gate(
                    record["source"],
                    record["translation_raw"],
                    record["polish_proposal"],
                    locked_terms=_locked_terms(_glossary(prep.glossary)),
                    src_lang="en",
                )
                if (
                    record.get("polish_proposal") != gate.proposal
                    or record.get("polish_accepted") != gate.accepted
                    or record.get("polish_rejection_reasons") != list(gate.rejection_reasons)
                    or record.get("final") != gate.selected
                ):
                    raise BenchmarkError("editor journal polish gate evidence mismatch")
            _validate_usage_delta(batch.get("usage_delta"))

    def _editor_artifact(
        self,
        root: Path,
        spec: CandidateSpec,
        options: GenerationOptions,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        editor: str,
        translation_artifact: Path,
        translation_key: str,
    ) -> Path:
        key = self._editor_key(options, editor, translation_key)
        artifact = root / "candidates" / "_shared" / key
        artifact.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "kind": "editor",
            "artifact_key": key,
            "translation_artifact_key": translation_key,
            "provider": spec.provider,
            "editor_model": editor,
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
        }
        if (artifact / "manifest.json").exists() and _read_json(
            artifact / "manifest.json"
        ) != manifest:
            raise BenchmarkError("editor manifest mismatch")
        _atomic_json(artifact / "manifest.json", manifest)
        if not (artifact / "usage.json").exists():
            _atomic_json(artifact / "usage.json", usage_delta({}, {}))
        sink = _JsonlTelemetrySink(artifact / "telemetry.jsonl")
        translation_manifest = _read_json(translation_artifact / "manifest.json")
        if (
            translation_manifest.get("artifact_key") != translation_key
            or translation_manifest.get("preparation_sha256") != preparation_sha
        ):
            raise BenchmarkError("editor translation artifact provenance mismatch")
        translation_passages: dict[str, dict[str, Any]] = {}
        for passage_file in sorted((translation_artifact / "passages").glob("*.json")):
            passage = _read_json(passage_file)
            if (
                passage.get("artifact_key") != translation_key
                or passage.get("preparation_sha256") != preparation_sha
            ):
                raise BenchmarkError("editor translation passage provenance mismatch")
            translation_passages[passage["passage_id"]] = passage
        self._recover_editor_journal(
            artifact, translation_artifact, key, editor, spec, bundle, translation_passages
        )
        client = self._client(spec, editor, "editor", options, sink)
        polisher = Polisher(
            client,
            Config(
                llm=LLMConfig(
                    provider=spec.provider,
                    models=ModelRoles(primary=editor, editor=editor, fast=spec.fast_model),
                ),
                source_lang="en",
                target_lang="zh",
            ),
        )
        for passage_file in sorted((translation_artifact / "passages").glob("*.json")):
            passage = _read_json(passage_file)
            records = passage.get("segments")
            if not isinstance(records, list) or not records:
                raise BenchmarkError("invalid translation passage for editor")
            source_row = {
                "passage_id": passage["passage_id"],
                "subset": passage["subset"],
                "book_id": passage["book_id"],
                "chapter_index": passage["chapter_index"],
                "replicate": passage["replicate"],
                "context_strategy": passage["context_strategy"],
                "segments": [
                    {"segment_id": x["segment_id"], "source": x["source"]} for x in records
                ],
                "context_hash": passage["context_hash"],
                "preparation_sha256": passage["preparation_sha256"],
                "artifact_key": translation_key,
            }
            prep = _prep_for(bundle, source_row)
            row_path = artifact / "passages" / passage_file.name
            if row_path.exists():
                existing = _read_json(row_path)
                _validate_passage_common(
                    existing,
                    source_row,
                    artifact_key=key,
                    spec=spec,
                    replicate=passage["replicate"],
                    subset=passage["subset"],
                    strategy=passage["context_strategy"],
                    model=editor,
                    context_hash=passage["context_hash"],
                    editor=True,
                    editor_key=key,
                    baseline_segments=records,
                    prep=prep,
                )
                continue
            segment_objects = [SimpleNamespace(**record) for record in records]
            batches = batch_segments(segment_objects, 1800)
            journal_path = artifact / "journal.json"
            committed: list[dict[str, Any]] = []
            if journal_path.exists():
                journal = _read_json(journal_path)
                if journal.get("passage_id") != passage["passage_id"] or journal.get(
                    "batch_count"
                ) != len(batches):
                    raise BenchmarkError("corrupt or incomplete editor journal")
                committed = journal.get("batches", [])
            edited_by_id = {record["segment_id"]: dict(record) for record in records}
            for entry in committed:
                for record in entry["segments"]:
                    edited_by_id[record["segment_id"]] = record
            terms = _glossary(prep.glossary)
            for batch_index, batch in enumerate(batches):
                if batch_index < len(committed):
                    continue
                batch_records = [edited_by_id[x.segment_id] for x in batch]
                before = _usage_of(client)
                proposals = polisher.polish(
                    [x["translation_raw"] for x in batch_records],
                    [x["source"] for x in batch_records],
                    glossary_terms=terms,
                    style=prep.style,
                    strict=True,
                )
                for record, proposal in zip(batch_records, proposals, strict=True):
                    gate = lint.polish_gate(
                        record["source"],
                        record["translation_raw"],
                        proposal,
                        locked_terms=_locked_terms(terms),
                        src_lang="en",
                    )
                    record["polish_proposal"] = gate.proposal
                    record["polish_accepted"] = gate.accepted
                    record["polish_rejection_reasons"] = list(gate.rejection_reasons)
                    record["final"] = gate.selected
                    _hash_record(record)
                    usage = getattr(client, "usage", None)
                    if usage is not None and hasattr(usage, "record_outcome"):
                        usage.record_outcome("editor", "polish.batch", accepted=gate.accepted)
                delta = _passage_usage_delta(client, before)
                batch_output = [dict(edited_by_id[x.segment_id]) for x in batch]
                _validate_usage_delta(delta)
                committed.append(
                    {"index": batch_index, "segments": batch_output, "usage_delta": delta}
                )
                _atomic_json(
                    journal_path,
                    {
                        "kind": "editor",
                        "artifact_key": key,
                        "passage_id": passage["passage_id"],
                        "batch_count": len(batches),
                        "batches": committed,
                    },
                )
                self._rebuild_usage(artifact)
                for record in batch_output:
                    edited_by_id[record["segment_id"]] = record
            edited = [edited_by_id[record["segment_id"]] for record in records]
            total_delta: dict[str, Any] = {}
            for entry in committed:
                total_delta = merge_usage_summaries(total_delta, entry["usage_delta"])
            output = dict(passage)
            output.update(
                {
                    "kind": "editor",
                    "artifact_key": key,
                    "translation_artifact_key": translation_key,
                    "editor_model": editor,
                    "segments": edited,
                    "usage_delta": total_delta,
                }
            )
            _validate_passage_common(
                output,
                source_row,
                artifact_key=key,
                spec=spec,
                replicate=passage["replicate"],
                subset=passage["subset"],
                strategy=passage["context_strategy"],
                model=editor,
                context_hash=passage["context_hash"],
                editor=True,
                editor_key=key,
                baseline_segments=records,
                prep=prep,
            )
            _atomic_json(row_path, output)
            journal_path.unlink(missing_ok=True)
            self._rebuild_usage(artifact)
        self._rebuild_usage(artifact)
        return artifact

    def _attribution(
        self,
        out: Path,
        spec: CandidateSpec,
        options: GenerationOptions,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        translations: dict[tuple[str, int, str, str], Path] = {}
        candidate_manifests: list[dict[str, Any]] = []
        groups = [
            ("screen", "c2"),
            ("continuous", "c2"),
            ("stratified", "c2"),
            ("context", "c0"),
            ("context", "c1"),
            ("context", "c2"),
        ]
        all_artifacts: dict[str, Path] = {}
        for candidate in spec.candidates:
            for replicate in range(1, spec.replicates + 1):
                editor_artifacts: dict[str, str] = {}
                for subset, strategy in groups:
                    selected = self._passage_set(rows, subset, strategy)
                    if not selected:
                        continue
                    key = (candidate.primary_model, replicate, subset, strategy)
                    if key not in translations:
                        passage_ids = [r["passage_id"] for r in selected]
                        artifact_key = self._artifact_key(
                            spec,
                            options,
                            corpus_sha,
                            preparation_sha,
                            candidate.primary_model,
                            replicate,
                            subset,
                            strategy,
                            passage_ids,
                        )
                        translations[key] = self._translate_artifact(
                            out,
                            spec,
                            options,
                            bundle,
                            corpus_sha,
                            preparation_sha,
                            candidate.primary_model,
                            replicate,
                            subset,
                            strategy,
                            selected,
                            artifact_key,
                        )
                        all_artifacts[translations[key].name] = translations[key]
                    translation = translations[key]
                    scope = f"{subset}:{strategy}"
                    if candidate.editor_model is not None and strategy == "c2":
                        editor = self._editor_artifact(
                            out,
                            spec,
                            options,
                            bundle,
                            corpus_sha,
                            preparation_sha,
                            candidate.editor_model,
                            translation,
                            translation.name,
                        )
                        editor_artifacts[scope] = str(editor.relative_to(out))
                        all_artifacts[editor.name] = editor
                translation_refs = {
                    f"{s}:{c}": str(p.relative_to(out))
                    for (m, r, s, c), p in translations.items()
                    if m == candidate.primary_model and r == replicate
                }
                refs = list(translation_refs.values()) + list(editor_artifacts.values())
                allocated: dict[str, Any] = {}
                for reference in refs:
                    allocated = merge_usage_summaries(
                        allocated, _read_json(out / reference / "usage.json")
                    )
                manifest_row = {
                    "candidate_id": candidate.candidate_id,
                    "replicate": replicate,
                    "primary_model": candidate.primary_model,
                    "editor_model": candidate.editor_model,
                    "translation_artifacts": translation_refs,
                    "editor_artifacts": editor_artifacts,
                    "editor_artifact_id": next(iter(editor_artifacts.values()), None),
                    "allocated_usage": allocated,
                }
                candidate_dir = out / "candidates" / candidate.candidate_id / str(replicate)
                candidate_dir.mkdir(parents=True, exist_ok=True)
                _atomic_json(candidate_dir / "manifest.json", manifest_row)
                _atomic_json(candidate_dir / "allocated_usage.json", allocated)
                candidate_manifests.append(manifest_row)
        actual: dict[str, Any] = {}
        for artifact in all_artifacts.values():
            actual = merge_usage_summaries(actual, _read_json(artifact / "usage.json"))
        _atomic_json(out / "candidates.json", candidate_manifests)
        _atomic_json(out / "actual_usage.json", actual)
        return {
            "translation_artifacts": len(translations),
            "candidate_count": len(candidate_manifests),
            "actual_usage": actual,
        }

    @staticmethod
    def _check_canary_telemetry(
        client: Any,
        expected_model: str,
        sink: _JsonlTelemetrySink | None = None,
        *,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        records = sink.records if sink is not None else getattr(client, "telemetry_records", None)
        if records is None:
            existing_sink = getattr(client, "telemetry_sink", None)
            records = getattr(existing_sink, "records", None) if existing_sink is not None else None
        values = list(records or [])[start_index:]
        if not values:
            raise BenchmarkError("canary telemetry has no attempts")
        parsed: list[dict[str, Any]] = []
        for value in values:
            try:
                telemetry = (
                    value
                    if isinstance(value, CallAttemptTelemetry)
                    else CallAttemptTelemetry.model_validate(value)
                )
            except Exception as error:
                raise BenchmarkError("canary telemetry record is incomplete or invalid") from error
            if telemetry.status != "success":
                raise BenchmarkError("canary response was not successful")
            if (
                telemetry.requested_model != expected_model
                or telemetry.resolved_model != expected_model
            ):
                raise BenchmarkError("canary requested/resolved model mismatch")
            if telemetry.temperature != 0.1:
                raise BenchmarkError("canary temperature mismatch")
            if telemetry.reasoning_tokens != 0:
                raise BenchmarkError("canary reasoning tokens must be zero")
            if telemetry.billed_usage_unknown or telemetry.response_sha256 is None:
                raise BenchmarkError("canary response usage or content hash is unknown")
            parsed.append(telemetry.model_dump(mode="python"))
        return parsed

    def _validate_completed(
        self,
        out: Path,
        spec: CandidateSpec,
        options: GenerationOptions,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        rows: list[dict[str, Any]],
        mode: str,
        sample_id: str | None,
    ) -> None:
        if mode == "canary":
            canary = _read_json(out / "canary.json")
            if (
                canary.get("status") != "passed"
                or canary.get("sample_id") != sample_id
                or canary.get("temperature") != 0.1
                or canary.get("reasoning_tokens") != 0
            ):
                raise BenchmarkError("completed canary artifact identity/evidence is invalid")
            canary_row = next(
                (x for x in rows if x["subset"] == "screen" and x["passage_id"] == sample_id), None
            )
            if canary_row is None or canary.get("source_hash") != _source_hash(
                canary_row["segments"]
            ):
                raise BenchmarkError("completed canary sample evidence is invalid")
            expected_segments = [
                {"segment_id": x["segment_id"], "source": x["source"]}
                for x in canary_row["segments"]
            ]
            if canary.get("segments") != expected_segments:
                raise BenchmarkError("completed canary alignment evidence is invalid")
            first_batch = batch_segments(
                [SimpleNamespace(**x) for x in canary_row["segments"]], 1800
            )[0]
            canary_prep = _prep_for(bundle, canary_row)
            expected_primaries = sorted({candidate.primary_model for candidate in spec.candidates})
            expected_pairs = {
                (candidate.primary_model, candidate.editor_model)
                for candidate in spec.candidates
                if candidate.editor_model is not None
            }
            results = canary.get("results")
            if (
                not isinstance(results, list)
                or [x.get("primary_model") for x in results] != expected_primaries
            ):
                raise BenchmarkError("completed canary model result set is invalid")
            actual_pairs: set[tuple[str, str]] = set()
            for result in results:
                primary = result.get("primary_model")
                selection = validate_model_selection(spec.provider, primary)
                outputs = result.get("outputs")
                if (
                    not isinstance(outputs, list)
                    or len(outputs) != len(first_batch)
                    or any(not isinstance(value, str) or not value.strip() for value in outputs)
                ):
                    raise BenchmarkError("completed canary output evidence is invalid")
                if result.get("segments") != len(first_batch) or result.get(
                    "output_sha256"
                ) != sha256_bytes(canonical_json(outputs).encode()):
                    raise BenchmarkError("completed canary output evidence is invalid")
                telemetry_path = out / "canary" / _safe_id(primary) / "telemetry.jsonl"
                sink = _JsonlTelemetrySink(telemetry_path)
                count = result.get("attempt_count")
                if not isinstance(count, int) or count <= 0 or count > len(sink.records):
                    raise BenchmarkError("completed canary telemetry evidence is invalid")
                current = self._check_canary_telemetry(
                    None, selection.model, sink, start_index=len(sink.records) - count
                )
                if result.get("telemetry_sha256") != sha256_bytes(
                    telemetry_path.read_bytes()
                ) or result.get("telemetry_records_sha256") != sha256_bytes(
                    canonical_json(current).encode()
                ):
                    raise BenchmarkError("completed canary telemetry evidence is invalid")
                expected_primary_pairs = {
                    editor for model, editor in expected_pairs if model == primary
                }
                pairs = result.get("editor_pairs")
                if (
                    not isinstance(pairs, list)
                    or len(pairs) != len(expected_primary_pairs)
                    or {pair.get("editor_model") for pair in pairs} != expected_primary_pairs
                ):
                    raise BenchmarkError("completed canary editor pair set is invalid")
                for pair in pairs:
                    editor_name = pair.get("editor_model")
                    if not isinstance(editor_name, str):
                        raise BenchmarkError("completed canary editor pair evidence is invalid")
                    actual_pairs.add((primary, editor_name))
                    editor_selection = validate_model_selection(spec.provider, editor_name)
                    proposals = pair.get("proposals")
                    finals = pair.get("finals")
                    if (
                        not isinstance(proposals, list)
                        or not isinstance(finals, list)
                        or len(proposals) != len(first_batch)
                        or len(finals) != len(first_batch)
                        or any(
                            not isinstance(value, str) or not value.strip()
                            for value in proposals + finals
                        )
                    ):
                        raise BenchmarkError("completed canary editor output evidence is invalid")
                    expected_gates = [
                        lint.polish_gate(
                            source,
                            raw,
                            proposal,
                            locked_terms=_locked_terms(_glossary(canary_prep.glossary)),
                            src_lang="en",
                        )
                        for source, raw, proposal in zip(
                            [x.source for x in first_batch],
                            next(
                                result["outputs"]
                                for result in results
                                if result.get("primary_model") == primary
                            ),
                            proposals,
                            strict=True,
                        )
                    ]
                    if finals != [gate.selected for gate in expected_gates] or proposals != [
                        gate.proposal for gate in expected_gates
                    ]:
                        raise BenchmarkError("completed canary editor gate evidence is invalid")
                    if (
                        pair.get("outputs") != finals
                        or pair.get("proposal_sha256")
                        != sha256_bytes(canonical_json(proposals).encode())
                        or pair.get("output_sha256")
                        != sha256_bytes(canonical_json(finals).encode())
                    ):
                        raise BenchmarkError("completed canary editor output evidence is invalid")
                    editor_path = (
                        out / "canary" / _safe_id(f"{primary}:{editor_name}") / "telemetry.jsonl"
                    )
                    editor_sink = _JsonlTelemetrySink(editor_path)
                    editor_count = pair.get("attempt_count")
                    if (
                        not isinstance(editor_count, int)
                        or editor_count <= 0
                        or editor_count > len(editor_sink.records)
                    ):
                        raise BenchmarkError(
                            "completed canary editor telemetry evidence is invalid"
                        )
                    current_editor = self._check_canary_telemetry(
                        None,
                        editor_selection.model,
                        editor_sink,
                        start_index=len(editor_sink.records) - editor_count,
                    )
                    if pair.get("telemetry_sha256") != sha256_bytes(
                        editor_path.read_bytes()
                    ) or pair.get("telemetry_records_sha256") != sha256_bytes(
                        canonical_json(current_editor).encode()
                    ):
                        raise BenchmarkError(
                            "completed canary editor telemetry evidence is invalid"
                        )
            if actual_pairs != expected_pairs:
                raise BenchmarkError("completed canary editor pair set is invalid")
            return
        groups = [
            ("screen", "c2"),
            ("continuous", "c2"),
            ("stratified", "c2"),
            ("context", "c0"),
            ("context", "c1"),
            ("context", "c2"),
        ]
        expected_candidates: list[dict[str, Any]] = []
        unique_artifacts: dict[str, Path] = {}
        actual: dict[str, Any] = {}
        for candidate in spec.candidates:
            for replicate in range(1, spec.replicates + 1):
                translations: dict[str, str] = {}
                editors: dict[str, str] = {}
                for subset, strategy in groups:
                    selected = self._passage_set(rows, subset, strategy)
                    if not selected:
                        continue
                    ids = [x["passage_id"] for x in selected]
                    key = self._artifact_key(
                        spec,
                        options,
                        corpus_sha,
                        preparation_sha,
                        candidate.primary_model,
                        replicate,
                        subset,
                        strategy,
                        ids,
                    )
                    artifact = out / "translation" / key
                    self._validate_translation_artifact(
                        artifact,
                        spec,
                        bundle,
                        corpus_sha,
                        preparation_sha,
                        candidate.primary_model,
                        replicate,
                        subset,
                        strategy,
                        selected,
                        key,
                    )
                    unique_artifacts[key] = artifact
                    translations[f"{subset}:{strategy}"] = str(artifact.relative_to(out))
                    if candidate.editor_model is not None and strategy == "c2":
                        editor_key = self._editor_key(options, candidate.editor_model, key)
                        editor = out / "candidates" / "_shared" / editor_key
                        self._validate_editor_artifact(
                            editor,
                            spec,
                            bundle,
                            corpus_sha,
                            preparation_sha,
                            candidate.editor_model,
                            artifact,
                            key,
                        )
                        unique_artifacts[editor_key] = editor
                        editors[f"{subset}:{strategy}"] = str(editor.relative_to(out))
                refs = list(translations.values()) + list(editors.values())
                allocated: dict[str, Any] = {}
                for reference in refs:
                    allocated = merge_usage_summaries(
                        allocated, _read_json(out / reference / "usage.json")
                    )
                expected_candidates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "replicate": replicate,
                        "primary_model": candidate.primary_model,
                        "editor_model": candidate.editor_model,
                        "translation_artifacts": translations,
                        "editor_artifacts": editors,
                        "editor_artifact_id": next(iter(editors.values()), None),
                        "allocated_usage": allocated,
                    }
                )
        stored_candidates = _read_json(out / "candidates.json")
        if not isinstance(stored_candidates, list) or stored_candidates != expected_candidates:
            raise BenchmarkError("completed candidate manifest set is invalid")
        for expected in expected_candidates:
            candidate_dir = (
                out / "candidates" / expected["candidate_id"] / str(expected["replicate"])
            )
            if (
                _read_json(candidate_dir / "manifest.json") != expected
                or _read_json(candidate_dir / "allocated_usage.json") != expected["allocated_usage"]
            ):
                raise BenchmarkError("completed candidate artifact integrity failure")
        for artifact in unique_artifacts.values():
            actual = merge_usage_summaries(actual, _read_json(artifact / "usage.json"))
        if _read_json(out / "actual_usage.json") != actual:
            raise BenchmarkError("actual benchmark usage integrity failure")

    def _validate_editor_artifact(
        self,
        artifact: Path,
        spec: CandidateSpec,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        editor: str,
        translation_artifact: Path,
        translation_key: str,
    ) -> None:
        key = self._editor_key(
            GenerationOptions(
                temperature=0.1,
                seed=None,
                require_catalogued_model=True,
                require_thinking_disabled=True,
            ),
            editor,
            translation_key,
        )
        manifest = {
            "schema_version": 1,
            "kind": "editor",
            "artifact_key": key,
            "translation_artifact_key": translation_key,
            "provider": spec.provider,
            "editor_model": editor,
            "corpus_sha256": corpus_sha,
            "preparation_sha256": preparation_sha,
        }
        if (
            _read_json(artifact / "manifest.json") != manifest
            or (artifact / "journal.json").exists()
        ):
            raise BenchmarkError("editor artifact integrity failure")
        translation_manifest = _read_json(translation_artifact / "manifest.json")
        if (
            translation_manifest.get("artifact_key") != translation_key
            or translation_manifest.get("provider") != spec.provider
            or translation_manifest.get("corpus_sha256") != corpus_sha
            or translation_manifest.get("preparation_sha256") != preparation_sha
        ):
            raise BenchmarkError("editor translation artifact provenance mismatch")
        for path in sorted((translation_artifact / "passages").glob("*.json")):
            source = _read_json(path)
            if (
                source.get("artifact_key") != translation_key
                or source.get("preparation_sha256") != preparation_sha
            ):
                raise BenchmarkError("editor translation passage provenance mismatch")
            row = {
                "passage_id": source["passage_id"],
                "subset": source["subset"],
                "book_id": source["book_id"],
                "chapter_index": source["chapter_index"],
                "replicate": source["replicate"],
                "context_strategy": source["context_strategy"],
                "context_hash": source["context_hash"],
                "preparation_sha256": preparation_sha,
                "artifact_key": translation_key,
                "segments": [
                    {"segment_id": x["segment_id"], "source": x["source"]}
                    for x in source["segments"]
                ],
            }
            editor_path = artifact / "passages" / path.name
            if not editor_path.exists():
                raise BenchmarkError("missing editor passage")
            _validate_passage_common(
                _read_json(editor_path),
                row,
                artifact_key=key,
                spec=spec,
                replicate=row["replicate"],
                subset=row["subset"],
                strategy=row["context_strategy"],
                model=editor,
                context_hash=row["context_hash"],
                editor=True,
                editor_key=key,
                baseline_segments=source["segments"],
                prep=_prep_for(bundle, row),
            )
        stored_usage = _read_json(artifact / "usage.json")
        _validate_usage_delta(stored_usage)
        if self._compute_usage(artifact) != stored_usage:
            raise BenchmarkError("editor usage does not match completed passage deltas")

    def _canary(
        self,
        out: Path,
        spec: CandidateSpec,
        options: GenerationOptions,
        bundle: PreparationBundle,
        corpus_sha: str,
        preparation_sha: str,
        rows: list[dict[str, Any]],
        sample_id: str | None,
    ) -> dict[str, Any]:
        row = next(
            (x for x in rows if x["subset"] == "screen" and x["passage_id"] == sample_id), None
        )
        if row is None:
            raise BenchmarkError("canary sample_id must resolve to a screen passage")
        results: list[dict[str, Any]] = []
        seen_editors: set[tuple[str, str]] = set()
        first_batch = batch_segments([SimpleNamespace(**x) for x in row["segments"]], 1800)[0]
        prep = _prep_for(bundle, row)
        for primary in sorted({c.primary_model for c in spec.candidates}):
            selection = validate_model_selection(spec.provider, primary)
            sink = _JsonlTelemetrySink(out / "canary" / _safe_id(primary) / "telemetry.jsonl")
            start_index = len(sink.records)
            client = self._client(spec, primary, "translator", options, sink)
            translator = Translator(
                client,
                Config(
                    llm=LLMConfig(
                        provider=spec.provider,
                        models=ModelRoles(primary=primary, editor=primary, fast=spec.fast_model),
                    ),
                    source_lang="en",
                    target_lang="zh",
                ),
            )
            targets = translator.translate_batch(
                [x.source for x in first_batch],
                agent="translator",
                operation="translate.batch",
                glossary_terms=_glossary(prep.glossary),
                style=prep.style,
                book_synopsis=prep.book_synopsis,
                chapter_digest=prep.chapter_digests[str(row["chapter_index"])],
                context_bundle=TranslationContextBundle(),
            )
            primary_records = self._check_canary_telemetry(
                client, selection.model, sink, start_index=start_index
            )
            if len(targets) != len(first_batch) or any(not t.strip() for t in targets):
                raise BenchmarkError("canary returned empty or misaligned output")
            editor_pairs: list[dict[str, Any]] = []
            for candidate in spec.candidates:
                if candidate.primary_model != primary or candidate.editor_model is None:
                    continue
                pair = (primary, candidate.editor_model)
                if pair in seen_editors:
                    continue
                seen_editors.add(pair)
                editor_selection = validate_model_selection(spec.provider, candidate.editor_model)
                editor_sink = _JsonlTelemetrySink(
                    out
                    / "canary"
                    / _safe_id(f"{primary}:{candidate.editor_model}")
                    / "telemetry.jsonl"
                )
                editor_start = len(editor_sink.records)
                editor_client = self._client(
                    spec, candidate.editor_model, "editor", options, editor_sink
                )
                polisher = Polisher(
                    editor_client,
                    Config(
                        llm=LLMConfig(
                            provider=spec.provider,
                            models=ModelRoles(
                                primary=candidate.editor_model,
                                editor=candidate.editor_model,
                                fast=spec.fast_model,
                            ),
                        ),
                        source_lang="en",
                        target_lang="zh",
                    ),
                )
                polished = polisher.polish(
                    targets,
                    [x.source for x in first_batch],
                    glossary_terms=_glossary(prep.glossary),
                    style=prep.style,
                    strict=True,
                )
                editor_records = self._check_canary_telemetry(
                    editor_client, editor_selection.model, editor_sink, start_index=editor_start
                )
                if len(polished) != len(first_batch) or any(not x.strip() for x in polished):
                    raise BenchmarkError("canary polish returned empty or misaligned output")
                gates = [
                    lint.polish_gate(
                        source,
                        raw,
                        proposal,
                        locked_terms=_locked_terms(_glossary(prep.glossary)),
                        src_lang="en",
                    )
                    for source, raw, proposal in zip(
                        [x.source for x in first_batch], targets, polished, strict=True
                    )
                ]
                proposals = [gate.proposal for gate in gates]
                finals = [gate.selected for gate in gates]
                editor_pairs.append(
                    {
                        "editor_model": candidate.editor_model,
                        "proposals": proposals,
                        "finals": finals,
                        "outputs": finals,
                        "telemetry_sha256": sha256_bytes(editor_sink.path.read_bytes()),
                        "telemetry_records_sha256": sha256_bytes(
                            canonical_json(editor_records).encode()
                        ),
                        "attempt_count": len(editor_records),
                        "proposal_sha256": sha256_bytes(canonical_json(proposals).encode()),
                        "output_sha256": sha256_bytes(canonical_json(finals).encode()),
                    }
                )
            results.append(
                {
                    "primary_model": primary,
                    "segments": len(targets),
                    "outputs": list(targets),
                    "output_sha256": sha256_bytes(canonical_json(targets).encode()),
                    "telemetry_sha256": sha256_bytes(sink.path.read_bytes()),
                    "telemetry_records_sha256": sha256_bytes(
                        canonical_json(primary_records).encode()
                    ),
                    "attempt_count": len(primary_records),
                    "editor_pairs": editor_pairs,
                }
            )
        segments = [{"segment_id": x["segment_id"], "source": x["source"]} for x in row["segments"]]
        _atomic_json(
            out / "canary.json",
            {
                "status": "passed",
                "sample_id": row["passage_id"],
                "source_hash": _source_hash(row["segments"]),
                "segments": segments,
                "temperature": spec.temperature,
                "reasoning_tokens": 0,
                "results": results,
            },
        )
        return {"canary": "passed"}


__all__ = [
    "PROMPT_VERSION",
    "AttributionRunner",
    "BenchmarkError",
    "FrozenBookPreparation",
    "FrozenPreparationMap",
    "FullRunner",
    "load_candidate_spec",
    "load_preparation_bundle",
    "preparation_source",
    "validate_candidate_capabilities",
    "validate_preparation",
]
