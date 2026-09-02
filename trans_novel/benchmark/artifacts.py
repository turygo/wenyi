"""Canonical JSON, JSONL, path containment, and hash primitives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class ArtifactError(ValueError):
    """A benchmark artifact is unreadable or outside its declared root."""


def atomic_json(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Any], *, sync: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        if sync:
            stream.flush()
            os.fsync(stream.fileno())


def read_json(path: str | os.PathLike[str], *, error_type: type[Exception] = ArtifactError) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"invalid JSON artifact {source}: {error}") from error


def read_jsonl(
    path: str | os.PathLike[str], *, error_type: type[Exception] = ArtifactError
) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise error_type(f"cannot read JSONL {source}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise error_type(f"blank JSONL line {source}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise error_type(f"invalid JSONL {source}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise error_type(f"JSONL row is not an object {source}:{line_number}")
        rows.append(row)
    return rows


def read_canonical_json(
    path: str | os.PathLike[str], *, error_type: type[Exception] = ArtifactError
) -> Any:
    source = Path(path)
    try:
        raw = source.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise error_type(f"invalid JSON artifact {source}: {error}") from error
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise error_type(f"non-canonical JSON artifact: {source}")
    return value


def relative_path(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> str:
    try:
        return (
            Path(path)
            .expanduser()
            .resolve()
            .relative_to(Path(root).expanduser().resolve())
            .as_posix()
        )
    except ValueError as error:
        raise ArtifactError(f"path escapes declared root: {path}") from error


def artifact_sha256(path: str | os.PathLike[str]) -> str:
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError as error:
        raise ArtifactError(f"cannot hash artifact {path}: {error}") from error


__all__ = [
    "ArtifactError",
    "artifact_sha256",
    "atomic_json",
    "canonical_json",
    "read_canonical_json",
    "read_json",
    "read_jsonl",
    "relative_path",
    "sha256_bytes",
    "write_jsonl",
]
