#!/usr/bin/env python3
"""Validate release metadata and extract notes for a version tag."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path


class ReleaseMetadataError(ValueError):
    """Raised when a release tag and repository metadata do not agree."""


def extract_release_notes(changelog: str, version: str) -> str:
    """Return the body of an exact, dated CHANGELOG section."""
    prefix = f"## [{version}] - "
    lines = changelog.splitlines()
    start: int | None = None

    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        date_text = line.removeprefix(prefix)
        try:
            dt.date.fromisoformat(date_text)
        except ValueError as exc:
            raise ReleaseMetadataError(
                f"CHANGELOG section for {version} must use YYYY-MM-DD, got {date_text!r}"
            ) from exc
        start = index + 1
        break

    if start is None:
        raise ReleaseMetadataError(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section")

    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    notes = "\n".join(lines[start:end]).strip()
    if not notes:
        raise ReleaseMetadataError(f"CHANGELOG section for {version} is empty")
    return f"{notes}\n"


def prepare_release(
    tag: str,
    pyproject_path: Path,
    changelog_path: Path,
    output_path: Path,
) -> None:
    """Validate tag/version consistency and write release notes."""
    import tomllib

    with pyproject_path.open("rb") as file:
        version = tomllib.load(file)["project"]["version"]

    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ReleaseMetadataError(
            f"tag {tag!r} does not match project version {version!r}; expected {expected_tag!r}"
        )

    notes = extract_release_notes(changelog_path.read_text(encoding="utf-8"), version)
    output_path.write_text(notes, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Git tag to validate, for example v1.2.3")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--output", type=Path, default=Path("release-notes.md"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_release(args.tag, args.pyproject, args.changelog, args.output)
    except (KeyError, OSError, ReleaseMetadataError) as exc:
        print(f"Release metadata validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
