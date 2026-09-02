from __future__ import annotations

import stat
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from tests.fixtures.books import write_phase9_epub
from trans_novel.benchmark.epub_check import validate_epub


def _copy_epub(
    source: Path, target: Path, replacements: dict[str, bytes | None] | None = None
) -> None:
    replacements = replacements or {}
    with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(target, "w") as zout:
        existing = set()
        for info in zin.infolist():
            existing.add(info.filename)
            if info.filename in replacements and replacements[info.filename] is None:
                continue
            replacement = replacements.get(info.filename)
            data = replacement if replacement is not None else zin.read(info.filename)
            compression = (
                zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
            )
            zout.writestr(info.filename, data, compression)
        for name, data in replacements.items():
            if name not in existing and data is not None:
                zout.writestr(name, data, zipfile.ZIP_DEFLATED)


class Phase9EpubFixtureTests(unittest.TestCase):
    def test_actual_corrupt_compressed_member_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                info = zin.getinfo("OEBPS/text/chapter-1.xhtml")
                raw = bytearray(source.read_bytes())
                name_size, extra_size = struct.unpack_from("<HH", raw, info.header_offset + 26)
                data_offset = info.header_offset + 30 + name_size + extra_size
            raw[data_offset + max(1, info.compress_size // 2)] ^= 0xFF
            source.write_bytes(raw)
            codes = {item["code"] for item in validate_epub(source)["failures"]}
            self.assertTrue({"crc_error", "member_read"} & codes)

    def test_corrupt_member_read_and_crc_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            write_phase9_epub(str(source))
            with patch.object(zipfile.ZipFile, "open", side_effect=zipfile.BadZipFile("crc")):
                result = validate_epub(source)
            self.assertTrue(
                any(item["code"] in {"crc_error", "member_read"} for item in result["failures"])
            )

    def test_large_member_guard_is_reported_without_reading_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            write_phase9_epub(str(source))
            with patch("trans_novel.benchmark.epub_check._MAX_MEMBER_BYTES", 1):
                result = validate_epub(source)
            self.assertIn("member_too_large", {item["code"] for item in result["failures"]})

    def test_zip_preflight_rejects_duplicate_unsafe_symlink_and_special_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                members = [(info.filename, zin.read(info.filename)) for info in zin.infolist()]
            cases = (
                ("duplicate", [*members, members[-1]], "duplicate_entry"),
                ("unsafe", [*members, ("../escape", b"x")], "unsafe_entry"),
            )
            for label, extra_members, expected in cases:
                output = Path(directory) / f"{label}.epub"
                with zipfile.ZipFile(output, "w") as zout:
                    for name, data in extra_members:
                        zout.writestr(
                            name,
                            data,
                            zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED,
                        )
                self.assertIn(
                    expected, {item["code"] for item in validate_epub(output)["failures"]}
                )
            for label, mode in (
                ("symlink", stat.S_IFLNK | 0o777),
                ("special", stat.S_IFIFO | 0o600),
            ):
                output = Path(directory) / f"{label}.epub"
                with zipfile.ZipFile(output, "w") as zout:
                    for name, data in members:
                        zout.writestr(
                            name,
                            data,
                            zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED,
                        )
                    info = zipfile.ZipInfo("payload")
                    info.external_attr = mode << 16
                    zout.writestr(info, b"x")
                self.assertIn(
                    "special_entry", {item["code"] for item in validate_epub(output)["failures"]}
                )

    def test_relocated_identical_bytes_have_identical_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            one = Path(first) / "book.epub"
            two = Path(second) / "book.epub"
            write_phase9_epub(str(one))
            two.write_bytes(one.read_bytes())
            self.assertEqual(validate_epub(one), validate_epub(two))
