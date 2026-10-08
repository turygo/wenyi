from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from tests.fixtures.migration_admission import admit_fixture
from tests.fixtures.migration_artifacts import (
    _origin_bound_fields,
    canonical_json,
    sha256_bytes,
    sha256_file,
)
from tests.fixtures.migration_epub import BASELINE_COMMIT, FIXTURE_ID
from tests.fixtures.migration_errors import ObservationError
from tests.fixtures.migration_observation import assert_semantic_match

_RECORD_ENV = "WENYI_R0_RECORD"
_REFERENCE_RELATIVE = PurePosixPath("fixtures/migration")
_INVENTORIES = ("surface-inventory.json", "state-inventory.json")
_INVENTORY_PATHS = tuple(f"fixtures/migration/{name}" for name in _INVENTORIES)
_TEST_IDENTIFIER = "tests.pipeline.test_reference_snapshot.TestReferenceSnapshot.test_full_flow_reference_candidate"


class RecordingError(RuntimeError):
    pass


def _support_hashes(repo_root: Path) -> dict[str, str]:
    paths = sorted(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "tests/fixtures").glob("migration_*.py")
    )
    paths.append("tests/pipeline/test_reference_snapshot.py")
    return {path: sha256_file(repo_root / path) for path in paths}


def _copy_artifact(
    source: Path,
    stage_root: Path,
    workspace_map: dict[str, str | None],
    fixture_relative: str,
    workspace_relative: str | None,
) -> None:
    target = stage_root / fixture_relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    workspace_map[fixture_relative] = workspace_relative


def _candidate_files(
    candidate: Any, stage_root: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], str]:
    workspace_map: dict[str, str | None] = {}
    run_prefix = f"state/{FIXTURE_ID}/run/state/{candidate.run_slug}"
    copy_prefix = f"state/{FIXTURE_ID}/assembly/source-first/run/state/{candidate.run_slug}"
    _copy_artifact(
        candidate.workspace / candidate.source_relative,
        stage_root,
        workspace_map,
        "epub/structured-v1.epub",
        "source/book.epub",
    )
    _copy_artifact(
        candidate.workspace / "protocol/structured-v1-responses.json",
        stage_root,
        workspace_map,
        "protocol/structured-v1-responses.json",
        None,
    )
    _copy_artifact(
        candidate.workspace / "config.yaml",
        stage_root,
        workspace_map,
        f"state/{FIXTURE_ID}/config.yaml",
        "config.yaml",
    )
    _copy_artifact(
        candidate.workspace / "assembly/source-first/config.yaml",
        stage_root,
        workspace_map,
        f"state/{FIXTURE_ID}/assembly/source-first/config.yaml",
        "assembly/source-first/config.yaml",
    )
    _copy_artifact(
        candidate.workspace / "assembly/source-first/source/book.epub",
        stage_root,
        workspace_map,
        f"state/{FIXTURE_ID}/assembly/source-first/source/book.epub",
        "assembly/source-first/source/book.epub",
    )
    _copy_run_tree(
        candidate.workspace / candidate.run_relative,
        stage_root / run_prefix,
        workspace_map,
        run_prefix,
        lambda name: f"state/{candidate.run_slug}/{name}",
    )
    _copy_run_tree(
        candidate.workspace / "assembly/source-first/state" / candidate.run_slug,
        stage_root / copy_prefix,
        workspace_map,
        copy_prefix,
        lambda name: f"assembly/source-first/state/{candidate.run_slug}/{name}",
    )
    _copy_outputs(candidate, stage_root, workspace_map)
    expected = canonical_json(candidate.projection) + b"\n"
    expected_path = stage_root / f"state/{FIXTURE_ID}/expected.json"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_bytes(expected)
    workspace_map[f"state/{FIXTURE_ID}/expected.json"] = None
    for relative in _INVENTORY_PATHS:
        if not (repo_root / relative).is_file():
            raise RecordingError(f"required inventory is missing: {relative}")
        workspace_map[PurePosixPath(relative).name] = None
    records = []
    for path, workspace_relative in workspace_map.items():
        artifact = (
            repo_root / "fixtures/migration" / path if path in _INVENTORIES else stage_root / path
        )
        records.append(
            {
                "path": path,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
                "workspace_path": workspace_relative,
            }
        )
    return sorted(records, key=lambda item: item["path"]), sha256_bytes(expected)


def _copy_run_tree(
    source: Path,
    destination: Path,
    workspace_map: dict[str, str | None],
    fixture_prefix: str,
    workspace_destination,
) -> None:
    for current, dirs, files in os.walk(source):
        current_path = Path(current)
        for directory in dirs:
            if (current_path / directory).is_symlink():
                raise RecordingError(f"symlink in persisted run: {current_path / directory}")
        for name in files:
            path = current_path / name
            relative = path.relative_to(source).as_posix()
            if name == ".run.lock":
                continue
            if name.endswith(("-wal", "-shm")) and path.stat().st_size == 0:
                continue
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise RecordingError(f"non-regular persisted run artifact: {path}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            workspace_map[f"{fixture_prefix}/{relative}"] = workspace_destination(relative)


def _copy_outputs(candidate: Any, stage_root: Path, workspace_map: dict[str, str | None]) -> None:
    for output in candidate.outputs:
        name = Path(output["file"]).name
        _copy_artifact(
            candidate.workspace / output["file"],
            stage_root,
            workspace_map,
            f"state/{FIXTURE_ID}/outputs/{name}",
            f"outputs/{name}",
        )
    output = candidate.source_first_output
    name = Path(output["file"]).name
    _copy_artifact(
        candidate.workspace / "assembly/source-first" / output["file"],
        stage_root,
        workspace_map,
        f"state/{FIXTURE_ID}/assembly/source-first/outputs/{name}",
        f"assembly/source-first/outputs/{name}",
    )
    _copy_artifact(
        candidate.workspace / "assembly/source-first/comparison.json",
        stage_root,
        workspace_map,
        f"state/{FIXTURE_ID}/assembly/source-first/comparison.json",
        None,
    )


def _output_records(candidate: Any) -> list[dict[str, str]]:
    records = [
        {
            "role": output["role"],
            "file": f"state/{FIXTURE_ID}/outputs/{Path(output['file']).name}",
            "proof": f"state/{FIXTURE_ID}/run/state/{candidate.run_slug}/epub_verification.json",
            "receipt": f"state/{FIXTURE_ID}/run/state/{candidate.run_slug}/epub_verification.json",
        }
        for output in candidate.outputs
    ]
    output = candidate.source_first_output
    records.append(
        {
            "role": output["role"],
            "file": f"state/{FIXTURE_ID}/assembly/source-first/outputs/{Path(output['file']).name}",
            "proof": f"state/{FIXTURE_ID}/assembly/source-first/run/state/{candidate.run_slug}/epub_verification.json",
            "receipt": f"state/{FIXTURE_ID}/assembly/source-first/run/state/{candidate.run_slug}/epub_verification.json",
        }
    )
    return records


def _record_destination(repo_root: Path) -> Path:
    if os.environ.get(_RECORD_ENV) != "fixtures/migration":
        raise RecordingError("recording requires WENYI_R0_RECORD=fixtures/migration")
    root = repo_root.resolve(strict=True)
    destination = root
    for part in _REFERENCE_RELATIVE.parts:
        destination = destination / part
        if destination.is_symlink():
            raise RecordingError("recording root must not contain symlinked path components")
    try:
        resolved = destination.resolve(strict=True)
    except OSError as error:
        raise RecordingError("recording fixture directory is unavailable") from error
    if resolved != destination or not resolved.is_dir():
        raise RecordingError("recording root must be the real fixtures/migration directory")
    return resolved


def _reject_existing(destination: Path) -> None:
    state_parent = destination / "state"
    if state_parent.is_symlink() or (state_parent.exists() and not state_parent.is_dir()):
        raise RecordingError("recording state root is not a safe directory")
    for relative in ("epub", "protocol", f"state/{FIXTURE_ID}"):
        target = destination / relative
        if target.exists() or target.is_symlink():
            raise RecordingError(f"recording destination already exists: {relative}")


def _reference_inputs(
    root: Path,
    candidate: Any,
    baseline: dict[str, Any],
    support: dict[str, str],
) -> dict[str, Any]:
    return {
        **baseline,
        "support_sha256": support,
        "exporter_sha256": sha256_bytes(canonical_json(support)),
        "inventory_sha256": {
            name: sha256_file(root / f"fixtures/migration/{name}") for name in _INVENTORIES
        },
        "source_sha256": candidate.source_sha256,
        "protocol_sha256": sha256_file(
            candidate.workspace / "protocol/structured-v1-responses.json"
        ),
        "configuration": candidate.validated_config,
        "explicit_field_presence": candidate.explicit_fields,
        "expanded_policy": candidate.expanded_policy,
    }


def _envelope(
    candidate: Any,
    root: Path,
    stage: Path,
    baseline: dict[str, Any],
    support: dict[str, str],
) -> dict[str, Any]:
    files, projection_hash = _candidate_files(candidate, stage, root)
    return {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "baseline_commit": BASELINE_COMMIT,
        "reference_inputs": _reference_inputs(root, candidate, baseline, support),
        "environment": candidate.environment,
        "wire_versions": candidate.wire_versions,
        "files": files,
        "outputs": _output_records(candidate),
        "semantic_projection": {
            "path": f"state/{FIXTURE_ID}/expected.json",
            "sha256": projection_hash,
            "format_version": 1,
        },
        "generation": {
            "test_identifier": _TEST_IDENTIFIER,
            "mode": "record",
            "executed_checks": list(candidate.executed_checks),
            "omissions": [
                {
                    "artifact": "verified-empty SQLite sidecars",
                    "reason": "no durable data remains outside the database",
                },
                {
                    "artifact": "run locks and temporary handles",
                    "reason": "transient, not persisted run inputs",
                },
                {
                    "artifact": "wall-clock values",
                    "reason": "retained separately under expected.json volatile observations",
                },
            ],
            "origin_bound_fields": _origin_bound_fields(stage, files),
        },
    }


def _validate_stage(stage: Path, envelope: dict[str, Any], repo_root: Path, candidate: Any) -> None:
    for name in _INVENTORIES:
        shutil.copyfile(repo_root / "fixtures/migration" / name, stage / name)
    for record in envelope["files"]:
        path = stage.joinpath(*record["path"].split("/"))
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise RecordingError(f"staged artifact failed integrity validation: {record['path']}")
    expected = _load_object(stage / envelope["semantic_projection"]["path"])
    if expected.get("format_version") != 1 or not isinstance(expected.get("source"), dict):
        raise RecordingError("staged semantic projection is incomplete")
    try:
        admitted = admit_fixture(stage)
    except (OSError, ObservationError) as error:
        raise RecordingError(
            "staged bundle failed the shared fixture admission contract"
        ) from error
    if admitted["fixture_id"] != FIXTURE_ID:
        raise RecordingError("staged fixture identity changed during admission")
    from tests.fixtures.migration_admission import materialize_fixture
    from tests.fixtures.migration_observation import observe_run
    from tests.fixtures.migration_reference import _assert_source_first_preservation
    from tests.fixtures.migration_run_observer import _workspace_file

    with tempfile.TemporaryDirectory(prefix="wenyi-r0-staged-") as temporary:
        workspace = materialize_fixture(stage, Path(temporary) / "workspace")
        for prefix, hash_map in (
            (candidate.run_relative, "original_run_file_sha256_after"),
            (f"assembly/source-first/state/{candidate.run_slug}", "copied_run_file_sha256_after"),
        ):
            for relative, digest in expected["volatile"][hash_map].items():
                if sha256_file(_workspace_file(workspace, f"{prefix}/{relative}")) != digest:
                    raise RecordingError("staged durable bytes differ from the qualified run")
        actual = observe_run(
            workspace,
            candidate.run_relative,
            candidate.source_relative,
            list(candidate.outputs),
            source_origin=expected["run"]["manifest"]["source_path"],
        )
        copied = observe_run(
            workspace / "assembly/source-first",
            f"state/{candidate.run_slug}",
            candidate.source_relative,
            [candidate.source_first_output],
            source_origin=expected["run"]["manifest"]["source_path"],
        )
        _assert_source_first_preservation(actual["run"], copied["run"])
        actual["outputs"].extend(copied["outputs"])
        actual["outputs"].sort(key=lambda item: item["role"])
        actual["source_first_comparison"] = expected["source_first_comparison"]
        assert_semantic_match(actual, expected)


def record_qualified_candidate(repo_root: str | Path, candidate: Any) -> Path:
    root = Path(repo_root).resolve(strict=True)
    destination = _record_destination(root)
    _reject_existing(destination)
    if candidate.protocol.get("fixture_id") != FIXTURE_ID:
        raise RecordingError("candidate protocol does not match the approved fixture")
    if not candidate.environment or not candidate.source_sha256:
        raise RecordingError("candidate is missing runtime or source provenance")
    from tests.fixtures.migration_provenance import _runtime_environment, verify_baseline_inputs

    baseline = verify_baseline_inputs(root)
    if _runtime_environment(root) != candidate.environment:
        raise RecordingError("runtime environment changed after candidate qualification")
    support = _support_hashes(root)
    stage = Path(tempfile.mkdtemp(prefix=".r0-stage-", dir=destination))
    published = []
    state_parent = destination / "state"
    created_state_parent = False
    try:
        envelope = _envelope(candidate, root, stage, baseline, support)
        metadata_path = stage / f"state/{FIXTURE_ID}/reference.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_bytes(canonical_json(envelope) + b"\n")
        _validate_stage(stage, envelope, root, candidate)
        for directory in ("epub", "protocol"):
            target = destination / directory
            if target.exists() or target.is_symlink():
                raise RecordingError(f"recording destination already exists: {directory}")
            os.rename(stage / directory, target)
            published.append(target)
        if not state_parent.exists():
            state_parent.mkdir()
            created_state_parent = True
        state_target = state_parent / FIXTURE_ID
        if state_target.exists() or state_target.is_symlink():
            raise RecordingError("recording state destination already exists")
        os.rename(stage / f"state/{FIXTURE_ID}", state_target)
        published.append(state_target)
        return state_target / "reference.json"
    except BaseException:
        for path in reversed(published):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                elif path.exists() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass
        if created_state_parent:
            with suppress(OSError):
                state_parent.rmdir()
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecordingError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise RecordingError(f"JSON artifact must be an object: {path}")
    return value


def qualify_candidate_against_fixture(candidate: Any, fixture_root: str | Path) -> dict[str, Any]:
    from tests.fixtures.migration_provenance import verify_baseline_inputs

    root = Path(fixture_root).resolve(strict=True)
    envelope = admit_fixture(root)
    expected = _load_object(root / envelope["semantic_projection"]["path"])
    assert_semantic_match(candidate.projection, expected)
    repo_root = root.parent.parent
    inputs = envelope["reference_inputs"]
    baseline = verify_baseline_inputs(repo_root)
    if any(inputs.get(key) != value for key, value in baseline.items()):
        raise ObservationError(
            "production source or dependency lock differs from the accepted baseline"
        )
    if _support_hashes(repo_root) != inputs.get("support_sha256"):
        raise ObservationError(
            "exporter or regression test source differs from the accepted reference"
        )
    inventories = {name: sha256_file(root / name) for name in _INVENTORIES}
    if inventories != inputs.get("inventory_sha256"):
        raise ObservationError("frozen surface or state inventory changed")
    if sha256_file(candidate.workspace / "protocol/structured-v1-responses.json") != inputs.get(
        "protocol_sha256"
    ):
        raise ObservationError(
            "synthetic request/response protocol differs from the accepted reference"
        )
    if sha256_file(candidate.workspace / candidate.source_relative) != inputs.get("source_sha256"):
        raise ObservationError("candidate synthetic EPUB input differs from the accepted reference")
    frozen_protocol = _load_object(root / "protocol/structured-v1-responses.json")
    if candidate.protocol != frozen_protocol:
        raise ObservationError(
            "candidate call identities differ from the accepted protocol evidence"
        )
    return envelope


def _accepted_bundle_hashes(root: Path) -> dict[str, str]:
    envelope = admit_fixture(root)
    paths = [record["path"] for record in envelope["files"]]
    paths.append(f"state/{FIXTURE_ID}/reference.json")
    return {path: sha256_file(root / path) for path in sorted(paths)}


def assert_failed_recording_preserves_bundle(repo_root: str | Path, candidate: Any) -> None:
    fixture_root = Path(repo_root).resolve(strict=True) / "fixtures/migration"
    before = _accepted_bundle_hashes(fixture_root)
    try:
        record_qualified_candidate(fixture_root.parent.parent, candidate)
    except RecordingError:
        pass
    else:
        raise AssertionError("recording overwrote an accepted reference bundle")
    if _accepted_bundle_hashes(fixture_root) != before:
        raise AssertionError("failed recording changed accepted fixture bundle bytes")
