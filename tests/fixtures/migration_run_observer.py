from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from tests.fixtures.migration_admission import _portable_path
from tests.fixtures.migration_artifacts import canonical_json, sha256_bytes, sha256_file
from tests.fixtures.migration_errors import ObservationError


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ObservationError(f"JSON artifact must be an object: {path}")
    return value


def _read_sqlite(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ObservationError("glossary SQLite integrity check failed")
        schema = [
            dict(row)
            for row in connection.execute(
                "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table','index','view','trigger') ORDER BY type,name"
            )
        ]
        table_names = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        raw_rows = {}
        for table in table_names:
            escaped = table.replace('"', '""')
            columns = [row["name"] for row in connection.execute(f'PRAGMA table_info("{escaped}")')]
            order = ", ".join('"' + column.replace('"', '""') + '"' for column in columns)
            query = f'SELECT * FROM "{escaped}"'
            if order:
                query += f" ORDER BY {order}"
            raw_rows[table] = [dict(row) for row in connection.execute(query)]
    except (sqlite3.Error, OSError) as error:
        raise ObservationError("glossary database could not be reopened read-only") from error
    finally:
        if "connection" in locals():
            connection.close()
    volatile = {
        table: [
            {key: row[key] for key in row if key in {"updated_at", "created_at"}} for row in rows
        ]
        for table, rows in raw_rows.items()
    }
    stable_rows = {
        table: [
            {key: value for key, value in row.items() if key not in {"updated_at", "created_at"}}
            for row in rows
        ]
        for table, rows in raw_rows.items()
    }
    return {"schema": schema, "rows": stable_rows, "integrity": result}, volatile


def _stable_usage(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _stable_usage(item)
            for key, item in value.items()
            if key not in {"elapsed_ms", "elapsed_ms_total"}
        }
    if isinstance(value, list):
        return [_stable_usage(item) for item in value]
    return value


def _event_observation(path: Path) -> dict[str, Any]:
    versions: set[int] = set()
    events: Counter[str] = Counter()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or type(row.get("event_schema")) is not int
                or not isinstance(row.get("event"), str)
            ):
                raise ObservationError("persisted event has an invalid schema or identity")
            versions.add(row["event_schema"])
            events[row["event"]] += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservationError("persisted event log is invalid") from error
    if not events:
        raise ObservationError("required persisted event log is empty")
    return {"schema_versions": sorted(versions), "event_counts": dict(sorted(events.items()))}


def _elapsed_fields(value: Any, path: tuple[str, ...] = ()) -> dict[str, Any]:
    if isinstance(value, dict):
        found = {}
        for key, item in value.items():
            if key in {"elapsed_ms", "elapsed_ms_total"}:
                found["/".join((*path, key))] = item
            else:
                found.update(_elapsed_fields(item, (*path, key)))
        return found
    if isinstance(value, list):
        return {
            key: item
            for index, value_item in enumerate(value)
            for key, item in _elapsed_fields(value_item, (*path, str(index))).items()
        }
    return {}


def _translation_source(segment: dict[str, Any]) -> str:
    epub_state = segment.get("epub_state")
    source = epub_state.get("rich_source") if isinstance(epub_state, dict) else None
    if not isinstance(source, dict):
        return str(segment.get("source", ""))
    breaks = {
        atom.get("id")
        for atom in source.get("atoms", [])
        if isinstance(atom, dict) and atom.get("kind") == "linebreak"
    }
    if not breaks:
        return str(segment.get("source", ""))
    runs = source.get("runs", [])
    return "".join(
        "\n" if isinstance(run, dict) and run.get("atom") in breaks else run.get("text", "")
        for run in runs
        if isinstance(run, dict)
    ).strip()


def _stable_node_output(value: Any, path: tuple[str, ...], volatile: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        stable = {}
        for key, item in value.items():
            if key == "output_sha256":
                volatile["/".join((*path, key))] = item
            else:
                stable[key] = _stable_node_output(item, (*path, key), volatile)
        return stable
    if isinstance(value, list):
        return [
            _stable_node_output(item, (*path, str(index)), volatile)
            for index, item in enumerate(value)
        ]
    return value


def _chapter_observations(
    run_dir: Path,
    chapter_rows: list[Any],
    progress: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    chapters = []
    indices = []
    for entry in chapter_rows:
        if not isinstance(entry, dict) or type(entry.get("index")) is not int:
            raise ObservationError("invalid persisted chapter index")
        index = entry["index"]
        chapter = _json_object(run_dir / "chapters_v2" / f"ch{index}.json")
        segments = chapter.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ObservationError("persisted chapter has no segments")
        projected_segments = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ObservationError("invalid persisted segment")
            target = segment.get("target")
            if not isinstance(target, str) or not target.strip():
                raise ObservationError("completed source segment has no target")
            epub_state = segment.get("epub_state")
            rich_source = epub_state.get("rich_source") if isinstance(epub_state, dict) else None
            projected_segments.append(
                {
                    "index": segment.get("index"),
                    "source": segment.get("source"),
                    "translation_source": _translation_source(segment),
                    "kind": segment.get("kind"),
                    "target": target,
                    "target_scalar_length": len(target),
                    "anchor": segment.get("anchor"),
                    "resource_href": segment.get("resource_href"),
                    "source_fingerprint": (
                        sha256_bytes(canonical_json(rich_source))
                        if isinstance(rich_source, dict)
                        else None
                    ),
                    "epub_state": epub_state,
                    "rich_source": rich_source,
                    "rich_target": segment.get("rich_target"),
                    "meta": segment.get("meta", {}),
                }
            )
        chapter_progress = progress.get(str(index), {})
        if not isinstance(chapter_progress, dict) or chapter_progress.get("status") != "done":
            raise ObservationError("a chapter did not reach the completed state")
        if chapter_progress.get("pending_polish"):
            raise ObservationError("a chapter retains pending polish work")
        if any(
            isinstance(issue, dict) and issue.get("status") in {"pending", "repairing"}
            for issue in chapter_progress.get("repair_ledger", {}).values()
        ):
            raise ObservationError("a chapter retains unfinished repair work")
        chapters.append(
            {
                "index": index,
                "title": chapter.get("title"),
                "href": chapter.get("href"),
                "processing": chapter.get("processing"),
                "segments": projected_segments,
                "progress": {
                    "status": chapter_progress.get("status"),
                    "pending_polish": chapter_progress.get("pending_polish", []),
                    "lint_issues": chapter_progress.get("lint_issues", []),
                    "repair_ledger": chapter_progress.get("repair_ledger", {}),
                },
            }
        )
        indices.append(index)
    return chapters, indices


def _node_observations(
    state: dict[str, Any], chapter_indices: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"prepare", "analyze", "titles", "deterministic_qa", "repair", "report", "assemble"}
    for key in required:
        node = state.get(key)
        if not isinstance(node, dict) or node.get("status") != "succeeded":
            raise ObservationError(f"required workflow node did not succeed: {key}")
    nodes = {}
    volatile = {"times": {}, "output_hashes": {}}
    for key, node in sorted(state.items()):
        if not isinstance(node, dict) or node.get("status") not in {"succeeded", "skipped"}:
            raise ObservationError(f"unfinished workflow node: {key}")
        for timestamp in ("started_at", "finished_at"):
            volatile["times"].setdefault(key, {})[timestamp] = node.get(timestamp)
        stable_output = _stable_node_output(
            node.get("output", {}), (key,), volatile["output_hashes"]
        )
        nodes[key] = {
            "status": node.get("status"),
            "attempts": node.get("attempts"),
            "fingerprint": str(node.get("input_fingerprint", "")),
            "failure": node.get("failure"),
            "output": stable_output,
        }
    for index in chapter_indices:
        key = f"translate:{index}"
        if nodes.get(key, {}).get("status") != "succeeded":
            raise ObservationError(f"chapter translation node did not succeed: {key}")
    return nodes, volatile


def _state_observation(
    workspace: Path, run_relative: str, source_relative: str, source_origin: str
) -> dict[str, Any]:
    run_dir = _workspace_directory(workspace, run_relative)
    manifest = _json_object(run_dir / "manifest.json")
    if manifest.get("run_state_schema") != 4:
        raise ObservationError("run-state schema is not version 4")
    identity = manifest.get("identity")
    meta = manifest.get("meta")
    if not isinstance(identity, dict) or identity.get("run_input_schema_version") != 2:
        raise ObservationError("run-input schema is not version 2")
    if (
        identity.get("translation_policy_version") != 5
        or not isinstance(meta, dict)
        or meta.get("epub_schema") != 4
    ):
        raise ObservationError(
            "run policy or EPUB source schema does not match the admitted baseline"
        )
    if manifest.get("source_path") != source_origin:
        raise ObservationError("persisted source origin differs from the declared origin")
    source_path = _workspace_file(workspace, source_relative)
    if sha256_file(source_path) != identity.get("source_bytes_sha256"):
        raise ObservationError("persisted run source identity does not match the source archive")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            run_dir / "journal.json",
            run_dir / "note_migration.json",
            run_dir / "usage.wal.json",
        )
    ):
        raise ObservationError("run contains a pending durable journal")
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise ObservationError("run contains an unexpected symlink")
        if path.is_file() and path.name.endswith((".pending", ".tmp")):
            raise ObservationError("run contains a pending transaction artifact")
    wal_path = run_dir / "glossary.db-wal"
    if wal_path.is_symlink() or (wal_path.exists() and wal_path.stat().st_size):
        raise ObservationError("glossary database depends on an uncollected WAL")
    state = manifest.get("nodes")
    progress = manifest.get("progress")
    chapter_rows = manifest.get("chapters")
    if (
        not isinstance(state, dict)
        or not isinstance(progress, dict)
        or not isinstance(chapter_rows, list)
    ):
        raise ObservationError("run manifest lifecycle data is incomplete")
    chapters, chapter_indices = _chapter_observations(run_dir, chapter_rows, progress)
    nodes, volatile_nodes = _node_observations(state, chapter_indices)
    event_path = _workspace_file(workspace, f"{_portable_path(run_relative)}/events.jsonl")
    events = _event_observation(event_path)
    usage = _json_object(run_dir / "usage.json")
    if usage.get("schema_version") != 2:
        raise ObservationError("usage schema is not version 2")
    report = _json_object(run_dir / "report.json")
    if not report:
        raise ObservationError("required persisted report is empty")
    database, volatile_sqlite = _read_sqlite(run_dir / "glossary.db")
    return {
        "manifest": {
            "run_state_schema": manifest["run_state_schema"],
            "run_input_schema_version": identity["run_input_schema_version"],
            "translation_policy_version": identity["translation_policy_version"],
            "source_bytes_sha256": identity["source_bytes_sha256"],
            "source_path": manifest["source_path"],
            "source_workspace_path": source_relative,
            "source_lang": manifest.get("source_lang"),
            "target_lang": manifest.get("target_lang"),
            "fmt": manifest.get("fmt"),
            "meta": {
                "epub_schema": meta.get("epub_schema"),
                "resources": meta.get("epub_resources"),
                "toc_entries": meta.get("toc_entries"),
                "notes": meta.get("epub_notes"),
            },
            "nodes": nodes,
            "chapters": chapters,
            "title": manifest.get("title"),
            "title_translated": manifest.get("title_translated"),
        },
        "events": events,
        "usage": _stable_usage(usage),
        "sqlite": database,
        "report": report,
        "volatile": {
            "node_times": volatile_nodes["times"],
            "node_output_hashes": volatile_nodes["output_hashes"],
            "event_log_sha256": sha256_file(event_path),
            "usage_elapsed_ms": _elapsed_fields(usage),
            "sqlite_times": volatile_sqlite,
        },
    }


def _workspace_path(workspace: Path, relative: str) -> Path:
    try:
        workspace_path = Path(workspace)
        if workspace_path.is_symlink():
            raise ObservationError("workspace root is a symlink")
        root = workspace_path.resolve(strict=True)
        current = root
        for part in _portable_path(relative).split("/"):
            current = current / part
            if current.is_symlink() or not current.exists():
                raise ObservationError(f"workspace artifact is missing or unsafe: {relative}")
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ObservationError(f"workspace artifact escapes the workspace: {relative}")
        return resolved
    except ObservationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ObservationError(f"workspace artifact is unavailable: {relative}") from error


def _workspace_file(workspace: Path, relative: str) -> Path:
    path = _workspace_path(workspace, relative)
    if not path.is_file():
        raise ObservationError(f"workspace artifact is not a regular file: {relative}")
    return path


def _workspace_directory(workspace: Path, relative: str) -> Path:
    path = _workspace_path(workspace, relative)
    if not path.is_dir():
        raise ObservationError(f"workspace artifact is not a directory: {relative}")
    return path
