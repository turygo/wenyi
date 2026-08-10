#!/usr/bin/env python3
"""Require CHANGELOG.md whenever a commit contains code changes."""

from __future__ import annotations

import sys
from pathlib import PurePosixPath

CODE_ROOTS = (
    ".github/workflows/",
    "scripts/",
    "tests/",
    "trans_novel/",
)
CODE_FILES = {
    ".pre-commit-config.yaml",
    "pyproject.toml",
    "uv.lock",
}


def is_code_path(path: str) -> bool:
    normalized = PurePosixPath(path).as_posix()
    return normalized in CODE_FILES or normalized.startswith(CODE_ROOTS)


def changelog_is_required(paths: list[str]) -> bool:
    return any(is_code_path(path) for path in paths)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--null-from-stdin"]:
        paths = [path for path in sys.stdin.read().split("\0") if path]
    else:
        paths = arguments

    if changelog_is_required(paths) and "CHANGELOG.md" not in paths:
        print(
            "Code changes require a CHANGELOG.md update in the same change.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
