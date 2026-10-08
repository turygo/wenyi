from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _origin_bound_fields(stage: Path, files: list[dict[str, Any]]) -> list[dict[str, str]]:
    records = []

    def visit(artifact: str, value: Any, pointer: str) -> None:
        if isinstance(value, str) and Path(value).is_absolute():
            records.append({"artifact": artifact, "pointer": pointer, "value": value})
        elif isinstance(value, dict):
            for key, child in value.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                visit(artifact, child, f"{pointer}/{escaped}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(artifact, child, f"{pointer}/{index}")

    for file in files:
        path = stage / file["path"]
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.suffix == ".jsonl":
            value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
        visit(file["path"], value, "")
    return sorted(records, key=lambda item: (item["artifact"], item["pointer"]))
