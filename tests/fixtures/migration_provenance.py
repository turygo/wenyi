from __future__ import annotations

import os
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from lxml import etree

from tests.fixtures.migration_artifacts import canonical_json, sha256_bytes, sha256_file
from tests.fixtures.migration_bundle import RecordingError
from tests.fixtures.migration_epub import BASELINE_COMMIT


def verify_baseline_inputs(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve(strict=True)
    command = [
        "git",
        "ls-tree",
        "-r",
        "--name-only",
        BASELINE_COMMIT,
        "--",
        "trans_novel",
        "pyproject.toml",
        "uv.lock",
    ]
    try:
        result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RecordingError("fixed Python reference baseline is unavailable") from error
    tracked = sorted(line for line in result.stdout.splitlines() if line)
    expected: dict[str, str] = {}
    for relative in tracked:
        try:
            old = subprocess.run(
                ["git", "show", f"{BASELINE_COMMIT}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise RecordingError(f"unable to read baseline artifact: {relative}") from error
        path = root / relative
        if not path.is_file() or sha256_file(path) != sha256_bytes(old):
            raise RecordingError(f"production baseline input changed: {relative}")
        expected[relative] = sha256_bytes(old)
    actual_source = {
        path.relative_to(root).as_posix()
        for path in (root / "trans_novel").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    baseline_source = {path for path in tracked if path.startswith("trans_novel/")}
    if actual_source != baseline_source:
        raise RecordingError("production tree contains files outside the fixed reference baseline")
    tree_payload = {key: expected[key] for key in sorted(expected)}
    return {
        "baseline_commit": BASELINE_COMMIT,
        "production_tree_sha256": sha256_bytes(canonical_json(tree_payload)),
        "production_file_sha256": {
            key: value for key, value in tree_payload.items() if key.startswith("trans_novel/")
        },
        "pyproject_sha256": expected["pyproject.toml"],
        "lockfile_sha256": expected["uv.lock"],
    }


def _runtime_environment(repo_root: Path) -> dict[str, Any]:
    dependencies = metadata.requires("trans-novel") or []
    versions = {}
    for requirement in dependencies:
        name = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
        if not name:
            raise RecordingError(f"unrecognized dependency declaration: {requirement!r}")
        distribution = name.group(1)
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError as error:
            raise RecordingError(f"required dependency is not installed: {distribution}") from error
    return {
        "python": __import__("sys").version,
        "platform": __import__("platform").platform(),
        "system": __import__("platform").system(),
        "machine": __import__("platform").machine(),
        "unicode_database": __import__("unicodedata").unidata_version,
        "sqlite": __import__("sqlite3").sqlite_version,
        "lxml": list(etree.LXML_VERSION),
        "libxml2": list(etree.LIBXML_VERSION),
        "dependencies": versions,
        "package": {"name": "trans-novel", "version": metadata.version("trans-novel")},
        "installed_distributions": dict(
            sorted(
                (distribution.metadata["Name"], distribution.version)
                for distribution in metadata.distributions()
            )
        ),
        "working_directory_convention": "workspace-root; source, state, and outputs are relative paths",
        "deterministic_environment": {
            key: os.environ[key]
            for key in ("LANG", "LC_ALL", "TZ", "PYTHONHASHSEED")
            if key in os.environ
        },
    }


def _wire_versions(projection: dict[str, Any]) -> dict[str, Any]:
    manifest = projection["run"]["manifest"]
    source_segments = [
        segment for chapter in manifest["chapters"] for segment in chapter["segments"]
    ]
    rich_versions = sorted(
        {
            segment["rich_source"]["version"]
            for segment in source_segments
            if isinstance(segment.get("rich_source"), dict)
        }
    )
    target_versions = sorted(
        {
            segment["rich_target"]["version"]
            for segment in source_segments
            if isinstance(segment.get("rich_target"), dict)
        }
    )
    event_versions = projection["run"]["events"]["schema_versions"]
    if len(event_versions) != 1:
        raise RecordingError("reference run contains inconsistent event schema versions")
    return {
        "run_state_schema": manifest["run_state_schema"],
        "run_input_schema": manifest["run_input_schema_version"],
        "translation_policy": manifest["translation_policy_version"],
        "epub_source_slot_schema": manifest["meta"]["epub_schema"],
        "rich_source": rich_versions,
        "rich_target": target_versions,
        "usage_schema": projection["run"]["usage"]["schema_version"],
        "event_schema": event_versions[0],
    }
