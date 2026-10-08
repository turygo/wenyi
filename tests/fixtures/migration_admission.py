from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

from tests.fixtures.migration_artifacts import _origin_bound_fields, sha256_file
from tests.fixtures.migration_epub import BASELINE_COMMIT, FIXTURE_ID
from tests.fixtures.migration_errors import ObservationError

_REFERENCE_PATH = PurePosixPath("state") / FIXTURE_ID / "reference.json"
_ROLES = {"mono", "bilingual_target_first", "bilingual_source_first"}
_HEX = re.compile(r"^[0-9a-f]{64}$")
_TEST_IDENTIFIER = "tests.pipeline.test_reference_snapshot.TestReferenceSnapshot.test_full_flow_reference_candidate"


def _portable_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ObservationError("invalid fixture path")
    if value.startswith("/") or "\\" in value or ":" in value:
        raise ObservationError(f"unsafe fixture path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ObservationError(f"unsafe fixture path: {value!r}")
    if PurePosixPath(value).is_absolute():
        raise ObservationError(f"unsafe fixture path: {value!r}")
    return value


def _contained_file(root: Path, relative: str) -> Path:
    root_real = root.resolve(strict=True)
    current = root_real
    for part in _portable_path(relative).split("/"):
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise ObservationError(f"missing fixture artifact: {relative}") from error
        if stat.S_ISLNK(info.st_mode):
            raise ObservationError(f"symlink fixture artifact: {relative}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ObservationError(f"fixture artifact is not a regular file: {relative}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root_real):
        raise ObservationError(f"fixture artifact escapes its root: {relative}")
    return resolved


def _fixture_root(root: str | Path) -> Path:
    if not isinstance(root, str | Path) or (isinstance(root, str) and not root):
        raise ObservationError("invalid fixture root")
    try:
        candidate = Path(root)
        info = candidate.lstat()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ObservationError("fixture root is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ObservationError("fixture root must be a real directory")
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ObservationError("fixture root is unavailable") from error


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ObservationError(f"JSON artifact must be an object: {path}")
    return value


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _validate_evidence(envelope: dict[str, Any]) -> None:
    reference = envelope["reference_inputs"]
    required_inputs = {
        "baseline_commit",
        "production_tree_sha256",
        "production_file_sha256",
        "pyproject_sha256",
        "lockfile_sha256",
        "support_sha256",
        "exporter_sha256",
        "inventory_sha256",
        "source_sha256",
        "protocol_sha256",
        "configuration",
        "explicit_field_presence",
        "expanded_policy",
    }
    if not required_inputs <= set(reference) or reference["baseline_commit"] != BASELINE_COMMIT:
        raise ObservationError("reference input provenance is incomplete or baseline-mismatched")
    for key in (
        "production_tree_sha256",
        "pyproject_sha256",
        "lockfile_sha256",
        "exporter_sha256",
        "source_sha256",
        "protocol_sha256",
    ):
        if not _valid_digest(reference.get(key)):
            raise ObservationError(f"invalid reference input digest: {key}")
    for key in ("production_file_sha256", "support_sha256"):
        values = reference.get(key)
        if (
            not isinstance(values, dict)
            or not values
            or any(
                not isinstance(path, str) or not path or not _valid_digest(digest)
                for path, digest in values.items()
            )
        ):
            raise ObservationError(f"invalid reference input digest map: {key}")
    inventories = reference.get("inventory_sha256")
    if (
        not isinstance(inventories, dict)
        or set(inventories) != {"surface-inventory.json", "state-inventory.json"}
        or any(not _valid_digest(digest) for digest in inventories.values())
    ):
        raise ObservationError("reference inventory digests are incomplete or invalid")
    support_bytes = json.dumps(
        reference["support_sha256"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(support_bytes).hexdigest() != reference["exporter_sha256"]:
        raise ObservationError("exporter digest does not bind the support files")
    for key in ("configuration", "explicit_field_presence", "expanded_policy"):
        if not isinstance(reference.get(key), dict):
            raise ObservationError(f"reference input {key} must be an object")


def _validate_environment(environment: dict[str, Any]) -> None:
    environment_strings = (
        "python",
        "platform",
        "system",
        "machine",
        "unicode_database",
        "sqlite",
        "working_directory_convention",
    )
    if any(
        not isinstance(environment.get(key), str) or not environment[key]
        for key in environment_strings
    ):
        raise ObservationError("runtime environment evidence is incomplete")
    for key in ("lxml", "libxml2"):
        if (
            not isinstance(environment.get(key), list)
            or not environment[key]
            or any(type(version) is not int for version in environment[key])
        ):
            raise ObservationError(f"runtime {key} version is invalid")
    dependencies = environment.get("dependencies")
    deterministic = environment.get("deterministic_environment")
    if (
        not isinstance(dependencies, dict)
        or not dependencies
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in dependencies.items()
        )
        or not isinstance(deterministic, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in deterministic.items()
        )
    ):
        raise ObservationError(
            "runtime dependency or deterministic environment evidence is invalid"
        )


def _validate_wire_versions(versions: dict[str, Any]) -> None:
    version_fields = {
        "run_state_schema",
        "run_input_schema",
        "translation_policy",
        "epub_source_slot_schema",
        "usage_schema",
        "event_schema",
    }
    if not version_fields <= set(versions) or any(
        type(versions.get(key)) is not int or versions[key] < 1 for key in version_fields
    ):
        raise ObservationError("wire schema version evidence is incomplete")
    for key in ("rich_source", "rich_target"):
        observed = versions.get(key)
        if (
            not isinstance(observed, list)
            or not observed
            or any(type(item) is not int for item in observed)
        ):
            raise ObservationError(f"wire {key} versions are invalid")


def _validate_generation(generation: dict[str, Any]) -> None:
    generation_fields = {
        "test_identifier",
        "mode",
        "executed_checks",
        "omissions",
        "origin_bound_fields",
    }
    if (
        set(generation) != generation_fields
        or generation.get("test_identifier") != _TEST_IDENTIFIER
        or generation.get("mode") != "record"
    ):
        raise ObservationError("reference generation evidence is incomplete or unqualified")
    checks = generation["executed_checks"]
    omissions = generation["omissions"]
    origin_fields = generation["origin_bound_fields"]
    if (
        not isinstance(checks, list)
        or not checks
        or any(not isinstance(item, str) or not item for item in checks)
        or len(checks) != len(set(checks))
        or not isinstance(omissions, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"artifact", "reason"}
            or any(
                not isinstance(item[key], str) or not item[key] for key in ("artifact", "reason")
            )
            for item in omissions
        )
        or not isinstance(origin_fields, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"artifact", "pointer", "value"}
            or any(
                not isinstance(item[key], str) or not item[key]
                for key in ("artifact", "pointer", "value")
            )
            for item in origin_fields
        )
    ):
        raise ObservationError(
            "reference generation checks, omissions or origin fields are invalid"
        )


def _validate_file_bindings(reference: dict[str, Any], paths: dict[str, dict[str, Any]]) -> None:
    for name, digest in reference["inventory_sha256"].items():
        if name not in paths or paths[name]["sha256"] != digest:
            raise ObservationError(f"inventory provenance does not match its file: {name}")
    for name, digest in (
        ("epub/structured-v1.epub", reference["source_sha256"]),
        ("protocol/structured-v1-responses.json", reference["protocol_sha256"]),
    ):
        if name not in paths or paths[name]["sha256"] != digest:
            raise ObservationError(f"reference input digest does not match its file: {name}")


def _admit_files(
    root_path: Path,
    metadata_path: Path,
    files: list[Any],
) -> dict[str, dict[str, Any]]:
    paths: dict[str, dict[str, Any]] = {}
    canonical_paths: set[Path] = {metadata_path}
    fixture_paths: list[tuple[str, ...]] = []
    workspace_paths: list[tuple[str, ...]] = []
    previous = ""
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "size_bytes",
            "sha256",
            "workspace_path",
        }:
            raise ObservationError("invalid file record")
        relative = _portable_path(record.get("path"))
        if relative <= previous or relative in paths:
            raise ObservationError("reference files are not strictly path-sorted and unique")
        folded_path = tuple(
            unicodedata.normalize("NFC", part).casefold() for part in relative.split("/")
        )
        if any(
            folded_path[: len(prior)] == prior or prior[: len(folded_path)] == folded_path
            for prior in fixture_paths
        ):
            raise ObservationError("duplicate or overlapping fixture path")
        fixture_paths.append(folded_path)
        previous = relative
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not _HEX.fullmatch(digest)
        ):
            raise ObservationError(f"invalid file size or hash: {relative}")
        workspace_path = record.get("workspace_path")
        if workspace_path is not None:
            workspace_path = _portable_path(workspace_path)
            folded = tuple(
                unicodedata.normalize("NFC", part).casefold() for part in workspace_path.split("/")
            )
            if any(
                folded[: len(prior)] == prior or prior[: len(folded)] == folded
                for prior in workspace_paths
            ):
                raise ObservationError("duplicate or overlapping workspace destination")
            workspace_paths.append(folded)
            record["workspace_path"] = workspace_path
        artifact = _contained_file(root_path, relative)
        if artifact.stat().st_size != size:
            raise ObservationError(f"fixture size mismatch: {relative}")
        if sha256_file(artifact) != digest:
            raise ObservationError(f"fixture hash mismatch: {relative}")
        if artifact in canonical_paths:
            raise ObservationError("fixture paths resolve to the same artifact")
        canonical_paths.add(artifact)
        paths[relative] = record
    return paths


def admit_fixture(root: str | Path) -> dict[str, Any]:
    root_path = _fixture_root(root)
    metadata_path = _contained_file(root_path, _REFERENCE_PATH.as_posix())
    envelope = _json_object(metadata_path)
    expected_fields = {
        "schema_version",
        "fixture_id",
        "baseline_commit",
        "reference_inputs",
        "environment",
        "wire_versions",
        "files",
        "outputs",
        "semantic_projection",
        "generation",
    }
    if set(envelope) != expected_fields:
        raise ObservationError("invalid reference envelope fields")
    if (
        type(envelope.get("schema_version")) is not int
        or envelope.get("schema_version") != 1
        or envelope.get("fixture_id") != FIXTURE_ID
        or envelope.get("baseline_commit") != BASELINE_COMMIT
    ):
        raise ObservationError("unsupported fixture identity or baseline")
    for key in ("reference_inputs", "environment", "wire_versions", "generation"):
        if not isinstance(envelope.get(key), dict):
            raise ObservationError(f"reference {key} must be an object")
    _validate_evidence(envelope)
    _validate_environment(envelope["environment"])
    _validate_wire_versions(envelope["wire_versions"])
    _validate_generation(envelope["generation"])
    files = envelope.get("files")
    if not isinstance(files, list) or not files:
        raise ObservationError("reference files must be a nonempty array")
    paths = _admit_files(root_path, metadata_path, files)
    for origin in envelope["generation"]["origin_bound_fields"]:
        artifact = _portable_path(origin["artifact"])
        if artifact not in paths or not origin["pointer"].startswith("/"):
            raise ObservationError("origin field does not refer to a listed JSON artifact")
    if envelope["generation"]["origin_bound_fields"] != _origin_bound_fields(root_path, files):
        raise ObservationError("origin-bound field declarations do not match persisted artifacts")

    _validate_file_bindings(envelope["reference_inputs"], paths)
    outputs = envelope.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 3:
        raise ObservationError("reference must contain exactly three output roles")
    roles: set[str] = set()
    for output in outputs:
        if not isinstance(output, dict) or set(output) != {"role", "file", "proof", "receipt"}:
            raise ObservationError("invalid output record")
        role = output.get("role")
        if not isinstance(role, str) or role not in _ROLES or role in roles:
            raise ObservationError("invalid or duplicate output role")
        roles.add(role)
        for key in ("file", "proof", "receipt"):
            path = _portable_path(output.get(key))
            if path not in paths:
                raise ObservationError(f"output {key} is not listed in files")
            output[key] = path
    if roles != _ROLES:
        raise ObservationError("required output roles are incomplete")
    projection = envelope.get("semantic_projection")
    if not isinstance(projection, dict) or set(projection) != {"path", "sha256", "format_version"}:
        raise ObservationError("invalid semantic projection record")
    projection_path = _portable_path(projection.get("path"))
    if type(projection.get("format_version")) is not int or projection.get("format_version") != 1:
        raise ObservationError("unsupported semantic projection")
    if projection_path not in paths or projection.get("sha256") != paths[projection_path]["sha256"]:
        raise ObservationError("semantic projection hash does not match its file record")
    return envelope


def materialize_fixture(root: str | Path, workspace: str | Path) -> Path:
    envelope = admit_fixture(root)
    root_path = _fixture_root(root)
    destination = Path(workspace)
    if destination.exists() or destination.is_symlink():
        raise ObservationError("fixture workspace must be fresh")
    destination.mkdir(parents=True)
    try:
        for record in envelope["files"]:
            relative = record["workspace_path"]
            if relative is None:
                continue
            source = _contained_file(root_path, record["path"])
            if (
                source.stat().st_size != record["size_bytes"]
                or sha256_file(source) != record["sha256"]
            ):
                raise ObservationError(f"fixture changed before materialization: {record['path']}")
            target = destination.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise ObservationError("fixture workspace destination already exists")
            shutil.copyfile(source, target)
            if (
                target.stat().st_size != record["size_bytes"]
                or sha256_file(target) != record["sha256"]
            ):
                raise ObservationError(f"materialized artifact mismatch: {relative}")
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return destination
