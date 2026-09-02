from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_changelog import changelog_is_required
from scripts.check_changelog import main as check_changelog
from scripts.prepare_release import (
    ReleaseMetadataError,
    extract_release_notes,
    prepare_release,
)


class ChangelogCheckTests(unittest.TestCase):
    def test_code_change_requires_changelog(self) -> None:
        self.assertTrue(changelog_is_required(["trans_novel/cli.py"]))
        self.assertEqual(check_changelog(["trans_novel/cli.py"]), 1)

    def test_code_change_with_changelog_passes(self) -> None:
        self.assertEqual(check_changelog(["trans_novel/cli.py", "CHANGELOG.md"]), 0)

    def test_documentation_only_change_does_not_require_changelog(self) -> None:
        self.assertFalse(changelog_is_required(["README.md"]))
        self.assertEqual(check_changelog(["README.md"]), 0)


class ReleaseNotesTests(unittest.TestCase):
    def test_extracts_only_requested_version(self) -> None:
        changelog = """# Changelog

## [Unreleased]

- Later work.

## [1.2.3] - 2026-08-10

### Added

- Shipped feature.

## [1.2.2] - 2026-08-01

- Previous feature.
"""

        self.assertEqual(
            extract_release_notes(changelog, "1.2.3"),
            "### Added\n\n- Shipped feature.\n",
        )

    def test_rejects_undated_version_section(self) -> None:
        changelog = "## [1.2.3]\n\n- Shipped feature.\n"

        with self.assertRaisesRegex(ReleaseMetadataError, "has no"):
            extract_release_notes(changelog, "1.2.3")

    @unittest.skipIf(sys.version_info < (3, 11), "tomllib is available in Python 3.11+")
    def test_prepare_release_requires_tag_to_match_project_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pyproject = root / "pyproject.toml"
            changelog = root / "CHANGELOG.md"
            output = root / "release-notes.md"
            pyproject.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
            changelog.write_text(
                "## [1.2.3] - 2026-08-10\n\n- Shipped feature.\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ReleaseMetadataError, "expected 'v1.2.3'"):
                prepare_release("v1.2.2", pyproject, changelog, output)

            prepare_release("v1.2.3", pyproject, changelog, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "- Shipped feature.\n")


if __name__ == "__main__":
    unittest.main()
