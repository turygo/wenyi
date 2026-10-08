from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.fixtures.migration_accounting import _assert_protocol_accounting
from tests.fixtures.migration_artifacts import canonical_json, sha256_file
from tests.fixtures.migration_boundary import RecordingTransport, RejectingClient, RequestBoundary
from tests.fixtures.migration_bundle import RecordingError
from tests.fixtures.migration_epub import write_structured_epub
from tests.fixtures.migration_observation import observe_run
from tests.fixtures.migration_provenance import (
    _runtime_environment,
    _wire_versions,
    verify_baseline_inputs,
)

_SOURCE_RELATIVE = "source/book.epub"
_CONFIG_RELATIVE = "config.yaml"


@dataclass(frozen=True)
class ReferenceCandidate:
    workspace: Path
    run_relative: str
    source_relative: str
    source_sha256: str
    run_slug: str
    outputs: tuple[dict[str, str], ...]
    source_first_output: dict[str, str]
    source_first_run_relative: str
    protocol: dict[str, Any]
    projection: dict[str, Any]
    comparison: dict[str, Any]
    validated_config: dict[str, Any]
    explicit_fields: dict[str, Any]
    expanded_policy: dict[str, Any]
    environment: dict[str, Any]
    wire_versions: dict[str, Any]
    executed_checks: tuple[str, ...]


def _working_directory(path: Path):
    class WorkingDirectory:
        def __enter__(self):
            self.previous = Path.cwd()
            os.chdir(path)
            return self

        def __exit__(self, exc_type, exc, traceback):
            os.chdir(self.previous)

    return WorkingDirectory()


def _config_data(*, source_first: bool = False) -> dict[str, Any]:
    return {
        "llm": {
            "models": {
                "translator": ["deepseek/deepseek-chat"],
                "analyst": ["deepseek/deepseek-chat"],
                "editor": ["deepseek/deepseek-chat"],
                "fast": ["deepseek/deepseek-chat"],
            }
        },
        "quality": "balanced",
        "output": {
            "mono": not source_first,
            "bilingual": {
                "enabled": True,
                "order": "source_first" if source_first else "target_first",
            },
            "override_theme": None,
            "bilingual_styles": "builtin:bilingual",
        },
    }


def _write_config(path: Path, *, source_first: bool = False) -> dict[str, Any]:
    raw = _config_data(source_first=source_first)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return raw


def _tree_hashes(root: Path) -> dict[str, str]:
    records = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RecordingError(f"unexpected symlink in reference workspace: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RecordingError(f"unexpected non-regular file in reference workspace: {relative}")
        if path.name == ".run.lock":
            continue
        if path.name.endswith(("-wal", "-shm")) and path.stat().st_size == 0:
            continue
        records[relative] = sha256_file(path)
    return records


def _build_transport(config: Any, boundary: RequestBoundary):
    from trans_novel.llm.providers.transport import OpenAICompatibleTransport
    from trans_novel.llm.registry import ProviderRegistry
    from trans_novel.llm.router import AgentRouter
    from trans_novel.llm.usage import UsageTracker

    tracker = UsageTracker()
    sdk = boundary.sdk
    transport = OpenAICompatibleTransport(
        config.llm,
        tracker,
        provider="deepseek",
        provider_name="DeepSeek",
        default_base_url="https://api.deepseek.com",
        default_api_key_env="DEEPSEEK_API_KEY",
        requires_api_key=True,
    )
    transport._client = sdk
    boundary.attach(transport)
    adapter = RecordingTransport(transport, boundary)
    registry = ProviderRegistry(config.llm, tracker, transports={"deepseek": adapter})
    return AgentRouter(config, registry=registry)


def _output_records(
    paths: list[str], verification_path: str, roles: tuple[str, ...]
) -> tuple[dict[str, str], ...]:
    if len(paths) != len(roles):
        raise RecordingError("production full flow returned an unexpected output count")
    records = []
    for path, role in zip(paths, roles, strict=True):
        relative = Path(path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or role
            not in {
                "mono",
                "bilingual_target_first",
                "bilingual_source_first",
            }
        ):
            raise RecordingError(
                "production output escaped the candidate workspace or has an invalid role"
            )
        records.append(
            {
                "role": role,
                "file": relative.as_posix(),
                "proof": verification_path,
                "receipt": verification_path,
            }
        )
    return tuple(records)


def _expected_targets(projection: dict[str, Any]) -> list[str]:
    return [
        segment["target"]
        for chapter in projection["run"]["manifest"]["chapters"]
        for segment in chapter["segments"]
    ]


def _source_first_workspace(
    workspace: Path,
    run_slug: str,
    source_path: Path,
    original_run: Path,
) -> tuple[Path, Path, dict[str, str], dict[str, str]]:
    clone = workspace / "assembly" / "source-first"
    clone.mkdir(parents=True)
    (clone / "source").mkdir()
    shutil.copyfile(source_path, clone / "source" / "book.epub")
    _write_config(clone / "config.yaml", source_first=True)
    before = _tree_hashes(original_run)
    clone_run = clone / "state" / run_slug
    shutil.copytree(original_run, clone_run, ignore=shutil.ignore_patterns(".run.lock"))
    copied_before = _tree_hashes(clone_run)
    if copied_before != before:
        raise RecordingError("source-first copy does not preserve durable run artifacts")
    (clone / "outputs").mkdir()
    return clone, clone_run, before, copied_before


def _run_primary_flow(workspace: Path):
    from trans_novel.assemble.epub.rendering.note_recovery import bind_note_references
    from trans_novel.assemble.epub.richtext_styles import enrich_rich_sources
    from trans_novel.config import Config
    from trans_novel.ingest import load_document
    from trans_novel.pipeline import Application
    from trans_novel.pipeline.planning.prescan import sample_text

    with _working_directory(workspace):
        config = Config.load(_CONFIG_RELATIVE)
        config.source_lang = "en"
        config.target_lang = "zh"
        document = load_document(
            _SOURCE_RELATIVE,
            "en",
            "zh",
            split_segments=config.segment.max_chars_per_segment,
        )
        enrich_rich_sources(_SOURCE_RELATIVE, document.chapters)
        bind_note_references(_SOURCE_RELATIVE, document.chapters)
        boundary = RequestBoundary(document, sample_text(document))
        router = _build_transport(config, boundary)
        application = Application(config, client=router)
        result = application.run_all(
            _SOURCE_RELATIVE,
            out_format="epub",
            out_path="outputs/structured-v1.epub",
        )
        boundary.assert_complete()
        store = result["store"]
        run_slug = Path(store.run_dir).name
        run_relative = Path(store.run_dir).as_posix()
        verification = (Path(run_relative) / "epub_verification.json").as_posix()
        outputs = _output_records(
            result["outputs"],
            verification,
            ("mono", "bilingual_target_first"),
        )
        if result.get("report") != store.read_json(store.report_path):
            raise RecordingError("run_all report return value differs from its persisted report")
        projection = observe_run(
            workspace,
            run_relative,
            _SOURCE_RELATIVE,
            list(outputs),
        )
        _assert_protocol_accounting(projection["run"]["usage"], boundary.evidence())
    return config, boundary, run_slug, run_relative, outputs, projection


def _assemble_source_first(
    workspace: Path,
    run_slug: str,
    run_relative: str,
    initial_projection: dict[str, Any],
    source_path: Path,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    from trans_novel.config import Config
    from trans_novel.pipeline import Application
    from trans_novel.pipeline.state import RunStore

    clone, clone_run, original_before, copied_before = _source_first_workspace(
        workspace, run_slug, source_path, workspace / run_relative
    )
    config = Config.load(str(clone / "config.yaml"))
    config.source_lang = "en"
    config.target_lang = "zh"
    reject = RejectingClient()
    with _working_directory(clone):
        store = RunStore(str(Path("state") / run_slug))
        application = Application(config, client=reject)
        paths = application.assemble(
            store,
            _SOURCE_RELATIVE,
            out_format="epub",
            out_path="outputs/structured-v1-source-first.epub",
        )
    if reject.calls:
        raise RecordingError("source-first assembly unexpectedly called the model boundary")
    verification = (Path("state") / run_slug / "epub_verification.json").as_posix()
    if len(paths) != 1:
        raise RecordingError("source-first assembly did not produce exactly one bilingual output")
    output = {
        "role": "bilingual_source_first",
        "file": Path(paths[0]).as_posix(),
        "proof": verification,
        "receipt": verification,
    }
    source_projection = observe_run(
        clone,
        (Path("state") / run_slug).as_posix(),
        _SOURCE_RELATIVE,
        [output],
        source_origin=initial_projection["run"]["manifest"]["source_path"],
    )
    original_state = initial_projection["run"]
    copied_state = source_projection["run"]
    _assert_source_first_preservation(original_state, copied_state)
    original_after = _tree_hashes(workspace / run_relative)
    if original_after != original_before:
        raise RecordingError("source-first assembly mutated the frozen original run")
    copied_after = _tree_hashes(clone_run)
    preserved = {
        relative: digest
        for relative, digest in copied_before.items()
        if relative.startswith("chapters_v2/")
        or relative in {"usage.json", "glossary.db", "analysis.json", "context.json", "report.json"}
    }
    if any(copied_after.get(relative) != digest for relative, digest in preserved.items()):
        raise RecordingError("source-first assembly modified paid content or durable accounting")
    comparison = _source_first_comparison(
        reject,
        original_state,
        targets=_expected_targets(initial_projection),
        preserved=preserved,
        raw_usage_before=sha256_file(workspace / run_relative / "usage.json"),
        raw_usage_after=sha256_file(clone_run / "usage.json"),
    )
    (clone / "comparison.json").write_bytes(canonical_json(comparison) + b"\n")
    initial_projection["outputs"].extend(source_projection["outputs"])
    initial_projection["outputs"].sort(key=lambda item: item["role"])
    initial_projection["source_first_comparison"] = comparison
    volatile = initial_projection.setdefault("volatile", {})
    volatile.update(
        {
            "original_run_file_sha256": original_before,
            "original_run_file_sha256_after": original_after,
            "copied_run_file_sha256_before": copied_before,
            "copied_run_file_sha256_after": copied_after,
            "source_first_observation": source_projection.get("volatile", {}),
        }
    )
    return output, comparison, _configuration(config)


def _assert_source_first_preservation(original: dict[str, Any], copied: dict[str, Any]) -> None:
    if original["manifest"]["chapters"] != copied["manifest"]["chapters"]:
        raise RecordingError(
            "source-first assembly changed persisted chapter content or fingerprints"
        )
    if _content_fingerprints(original) != _content_fingerprints(copied):
        raise RecordingError("source-first assembly changed upstream content-node fingerprints")
    if original["usage"] != copied["usage"]:
        raise RecordingError("source-first assembly changed usage accounting")
    if original["sqlite"] != copied["sqlite"]:
        raise RecordingError("source-first assembly changed the reopened glossary database")
    if original["report"] != copied["report"]:
        raise RecordingError("source-first assembly changed the persisted QA report")
    if original["manifest"]["title_translated"] != copied["manifest"]["title_translated"]:
        raise RecordingError("source-first assembly changed the stored translated title")


def _content_fingerprints(state: dict[str, Any]) -> dict[str, str]:
    return {
        key: node["fingerprint"]
        for key, node in state["manifest"]["nodes"].items()
        if key.partition(":")[0] not in {"assemble", "report", "layout"}
    }


def _source_first_comparison(
    reject: RejectingClient,
    state: dict[str, Any],
    *,
    targets: list[str],
    preserved: dict[str, str],
    raw_usage_before: str,
    raw_usage_after: str,
) -> dict[str, Any]:
    if raw_usage_before != raw_usage_after:
        raise RecordingError("source-first assembly changed raw usage bytes")
    return {
        "model_calls": len(reject.calls),
        "original_unchanged": True,
        "copied_preserved_files": sorted(preserved),
        "copied_preserved_targets": targets,
        "copied_usage": state["usage"],
        "copied_sqlite": state["sqlite"],
        "content_fingerprints": _content_fingerprints(state),
        "copied_content_fingerprints_unchanged": True,
        "raw_usage_bytes_unchanged": True,
        "copy_output_role": "bilingual_source_first",
    }


def _configuration(config: Any) -> dict[str, Any]:
    return {
        "llm": config.llm.model_dump(mode="json"),
        "quality": config.quality,
        "output": config.output.model_dump(mode="json"),
        "source_lang": config.source_lang,
        "target_lang": config.target_lang,
        "state_dir": config.state_dir,
    }


def build_candidate(
    workspace: str | Path,
    repo_root: str | Path,
    *,
    test_identifier: str,
) -> ReferenceCandidate:
    root = Path(repo_root).resolve(strict=True)
    verify_baseline_inputs(root)
    if (
        test_identifier
        != "tests.pipeline.test_reference_snapshot.TestReferenceSnapshot.test_full_flow_reference_candidate"
    ):
        raise RecordingError("candidate must be produced by the approved full-flow reference test")
    workspace_path = Path(workspace).resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    if any(workspace_path.iterdir()):
        raise RecordingError("candidate workspace must be empty")
    (workspace_path / "source").mkdir()
    (workspace_path / "outputs").mkdir()
    (workspace_path / "protocol").mkdir()
    source_path = workspace_path / _SOURCE_RELATIVE
    write_structured_epub(source_path)
    source_hash = sha256_file(source_path)
    _write_config(workspace_path / _CONFIG_RELATIVE)
    config, boundary, run_slug, run_relative, outputs, projection = _run_primary_flow(
        workspace_path
    )
    source_first_output, comparison, source_first_config = _assemble_source_first(
        workspace_path,
        run_slug,
        run_relative,
        projection,
        source_path,
    )
    protocol = boundary.evidence()
    (workspace_path / "protocol/structured-v1-responses.json").write_bytes(
        canonical_json(protocol) + b"\n"
    )
    explicit_fields = {
        "yaml": {
            "llm.models.translator": True,
            "llm.models.analyst": True,
            "llm.models.editor": True,
            "llm.models.fast": True,
            "quality": True,
            "output.mono": True,
            "output.bilingual.enabled": True,
            "output.bilingual.order": True,
            "output.override_theme": True,
            "output.bilingual_styles": True,
        },
        "runtime_overrides": {"source_lang": "en", "target_lang": "zh"},
    }
    validated_config = _configuration(config)
    policy = {
        "pipeline": config.pipeline.model_dump(mode="json"),
        "segment": config.segment.model_dump(mode="json"),
        "honorific_strategy": config.honorific_strategy,
        "punctuation_normalize": config.punctuation_normalize,
    }
    candidate = ReferenceCandidate(
        workspace=workspace_path,
        run_relative=run_relative,
        source_relative=_SOURCE_RELATIVE,
        source_sha256=source_hash,
        run_slug=run_slug,
        outputs=outputs,
        source_first_output=source_first_output,
        source_first_run_relative=(Path("assembly/source-first/state") / run_slug).as_posix(),
        protocol=protocol,
        projection=projection,
        comparison=comparison,
        validated_config={"initial": validated_config, "source_first": source_first_config},
        explicit_fields=explicit_fields,
        expanded_policy=policy,
        environment=_runtime_environment(root),
        wire_versions=_wire_versions(projection),
        executed_checks=(
            "baseline_inputs_match_fixed_commit",
            "real_application_run_all_epub",
            "reopened_json_sqlite_source_and_outputs",
            "required_lifecycle_and_no_pending_recovery",
            "independent_epub_target_scope_object_resource_reference_observation",
            "persisted_publication_proof_and_receipt_admitted",
            "source_first_application_assemble_with_rejecting_client",
            "source_first_targets_fingerprints_and_usage_preserved",
            "frozen_original_run_unchanged",
        ),
    )
    return candidate
