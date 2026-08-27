from __future__ import annotations

import stat
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from bs4 import BeautifulSoup

from tests.sample_data import write_phase9_epub
from trans_novel.assemble.bilingual_dom import BILINGUAL_CSS as _BILINGUAL_CSS
from trans_novel.benchmark.epub_check import validate_epub, validate_epub_triplet


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
    def test_valid_two_spine_fixture_has_required_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.epub"
            write_phase9_epub(str(path))
            result = validate_epub(path)
            self.assertTrue(result["structural_pass"], result["failures"])
            self.assertEqual(result["failures"], [])
            self.assertEqual(result["generated_resources"], [])

    def test_triplet_and_tn_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            mono = Path(directory) / "mono.epub"
            bilingual = Path(directory) / "bilingual.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter_two = BeautifulSoup(zin.read("OEBPS/text/chapter-2.xhtml"), "xml")
                body = chapter_two.find(id="body-two")
                assert body is not None
                body.string = "Translated second chapter."
                mono_chapter = str(chapter_two).encode("utf-8")
            _copy_epub(source, mono, {"OEBPS/text/chapter-2.xhtml": mono_chapter})
            chapter_two_soup = BeautifulSoup(mono_chapter, "xml")
            body = chapter_two_soup.find(id="body-two")
            assert body is not None
            source_node = chapter_two_soup.new_tag("p", attrs={"class": "tn-source"})
            source_node.string = "Second chapter."
            body.insert_after(source_node)
            style = chapter_two_soup.new_tag("style", id="tn-bilingual-style")
            style.string = _BILINGUAL_CSS
            chapter_two_soup.head.append(style)
            with zipfile.ZipFile(source) as zin:
                chapter_one_soup = BeautifulSoup(zin.read("OEBPS/text/chapter-1.xhtml"), "xml")
                chapter_one_style = chapter_one_soup.new_tag("style", id="tn-bilingual-style")
                chapter_one_style.string = _BILINGUAL_CSS
                chapter_one_soup.head.append(chapter_one_style)
                replacements = {
                    "OEBPS/text/chapter-1.xhtml": str(chapter_one_soup).encode("utf-8"),
                    "OEBPS/text/chapter-2.xhtml": str(chapter_two_soup).encode("utf-8"),
                }
            _copy_epub(source, bilingual, replacements)
            result = validate_epub_triplet(source, mono, bilingual)
            self.assertTrue(result["structural_pass"], result)
            self.assertTrue(result["mono"]["structural_pass"])
            self.assertTrue(result["bilingual"]["structural_pass"])
            standalone = validate_epub(bilingual, source_path=source, bilingual=True)
            self.assertGreater(
                result["bilingual"]["counts"]["bilingual_source"]["checked"],
                standalone["counts"]["bilingual_source"]["checked"],
            )

    def test_changed_and_missing_assets_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            changed = Path(directory) / "changed.epub"
            missing = Path(directory) / "missing.epub"
            write_phase9_epub(str(source))
            _copy_epub(source, changed, {"OEBPS/images/figure.png": b"tampered"})
            _copy_epub(source, missing, {"OEBPS/images/figure.png": None})
            self.assertTrue(
                any(
                    i["code"] == "changed_asset"
                    for i in validate_epub(changed, source_path=source)["failures"]
                )
            )
            self.assertTrue(
                any(
                    i["code"] == "missing_asset"
                    for i in validate_epub(missing, source_path=source)["failures"]
                )
            )

    def test_missing_links_and_fragments_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"#chapter-two", b"#does-not-exist"
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            codes = {item["code"] for item in validate_epub(broken, source_path=source)["failures"]}
            self.assertIn("missing_fragment", codes)

    def test_spine_sequence_mismatch_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                original_opf = zin.read("OEBPS/content.opf")
                soup = BeautifulSoup(original_opf, "xml")
                spine = soup.find("spine")
                assert spine is not None
                refs = spine.find_all("itemref", recursive=False)
                assert len(refs) == 2
                refs[0]["idref"], refs[1]["idref"] = refs[1]["idref"], refs[0]["idref"]
                opf = str(soup).encode("utf-8")
            self.assertNotEqual(original_opf, opf)
            _copy_epub(source, broken, {"OEBPS/content.opf": opf})
            codes = {item["code"] for item in validate_epub(broken, source_path=source)["failures"]}
            self.assertIn("sequence_mismatch", codes)

    def test_internal_attributes_placeholder_bad_nesting_and_external_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml")
                chapter = chapter.replace(
                    b"<ul><li>First item</li>",
                    b'<ul><p data-tn-id="x">spine-fallback</p><li>First item</li>',
                )
                chapter = chapter.replace(
                    b"</body>", b'<a href="https://example.test/no-fetch">external</a></body>'
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(broken)
            codes = {item["code"] for item in result["failures"]}
            self.assertIn("internal_attribute", codes)
            self.assertNotIn("marker", codes)
            self.assertIn("illegal_nesting", codes)
            self.assertTrue(
                all("example.test" not in item["detail"] for item in result["warnings"])
            )
            self.assertTrue(any(item["code"] == "external_skipped" for item in result["warnings"]))

    def test_marker_words_in_prose_and_attributes_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            prose = Path(directory) / "prose.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"First item", b"open journal.json and spine-fallback"
                )
                chapter = chapter.replace(
                    b'class="chapter"', b'class="chapter" title="journal.json"'
                )
            _copy_epub(source, prose, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(prose)
            self.assertTrue(result["structural_pass"])
            self.assertNotIn("marker", {item["code"] for item in result["failures"]})

    def test_malformed_zip_and_malformed_xml_fail_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed_zip = Path(directory) / "bad.epub"
            malformed_zip.write_bytes(b"not a zip")
            self.assertFalse(validate_epub(malformed_zip)["structural_pass"])
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            _copy_epub(source, broken, {"OEBPS/content.opf": b"<package>"})
            result = validate_epub(broken)
            self.assertFalse(result["structural_pass"])
            self.assertTrue(any(item["category"] == "parse" for item in result["failures"]))

    def test_determinism_traversal_and_missing_resource(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml")
                chapter = chapter.replace(
                    b'href="chapter-2.xhtml#chapter-two"',
                    b'href="../../../outside.xhtml#chapter-two"',
                )
                chapter = chapter.replace(
                    b'src="../images/figure.png"', b'src="../images/missing.png"'
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            first = validate_epub(broken)
            second = validate_epub(broken)
            self.assertEqual(first, second)
            codes = {item["code"] for item in first["failures"]}
            self.assertIn("unsafe_reference", codes)
            self.assertIn("missing_resource", codes)
            self.assertNotIn("Chapter One", repr(first))

    def test_corrupt_content_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</body>", b"<broken></body>"
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            self.assertIn(
                "malformed_content", {item["code"] for item in validate_epub(broken)["failures"]}
            )

    def test_missing_source_and_output_source_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            _copy_epub(source, output)
            result = validate_epub(
                output, source_path=Path(directory) / "missing.epub", bilingual=False
            )
            self.assertIn("source_missing", {item["code"] for item in result["failures"]})
            self.assertTrue(
                any(
                    item["code"].startswith("reopen")
                    for item in validate_epub(output, bilingual=False)["failures"]
                )
                is False
            )

    def test_missing_footnote_backlink_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-2.xhtml").replace(
                    b'<a href="chapter-1.xhtml#ref-1">back</a>', b"<span>back</span>"
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-2.xhtml": chapter})
            result = validate_epub(broken)
            self.assertIn("missing_backlink", {item["code"] for item in result["failures"]})

    def test_bilingual_missing_dummy_and_misplaced_nodes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            mono = Path(directory) / "mono.epub"
            missing = Path(directory) / "missing.epub"
            dummy = Path(directory) / "dummy.epub"
            misplaced = Path(directory) / "misplaced.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter_two = BeautifulSoup(zin.read("OEBPS/text/chapter-2.xhtml"), "xml")
                body = chapter_two.find(id="body-two")
                assert body is not None
                source_text = body.get_text(" ", strip=True)
                body.string = "Translated second chapter."
                translated_two = str(chapter_two).encode("utf-8")
                original_two = zin.read("OEBPS/text/chapter-2.xhtml")
                chapter_one = BeautifulSoup(zin.read("OEBPS/text/chapter-1.xhtml"), "xml")
                dummy_node = chapter_one.new_tag("p", attrs={"class": "tn-source"})
                dummy_node.string = source_text
                chapter_one.body.append(dummy_node)
                misplaced_one = str(chapter_one).encode("utf-8")
                dummy_soup = BeautifulSoup(translated_two, "xml")
                dummy_tag = dummy_soup.new_tag("p", attrs={"class": "tn-source"})
                dummy_tag.string = "not source"
                dummy_soup.body.append(dummy_tag)
                dummy_two = str(dummy_soup).encode("utf-8")
            _copy_epub(source, mono, {"OEBPS/text/chapter-2.xhtml": translated_two})
            _copy_epub(source, missing, {"OEBPS/text/chapter-2.xhtml": original_two})
            _copy_epub(source, dummy, {"OEBPS/text/chapter-2.xhtml": dummy_two})
            _copy_epub(
                source,
                misplaced,
                {
                    "OEBPS/text/chapter-2.xhtml": translated_two,
                    "OEBPS/text/chapter-1.xhtml": misplaced_one,
                },
            )
            self.assertFalse(validate_epub_triplet(source, mono, missing)["structural_pass"])
            self.assertFalse(validate_epub_triplet(source, mono, dummy)["structural_pass"])
            self.assertFalse(validate_epub_triplet(source, mono, misplaced)["structural_pass"])

    def test_zero_standalone_bilingual_source_nodes_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            write_phase9_epub(str(source))
            result = validate_epub(source, bilingual=True)
            self.assertIn("missing_source_nodes", {item["code"] for item in result["failures"]})

    def test_reference_tag_attribute_and_multiplicity_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b'<img src="../images/figure.png"',
                    b'<img src="../images/figure.png"/><img src="../images/figure.png"',
                    1,
                )
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(output, source_path=source)
            self.assertIn("reference_graph_mismatch", {item["code"] for item in result["failures"]})
            self.assertGreater(result["counts"]["internal_links"]["checked"], 0)

    def test_missing_nav_and_ncx_are_independent_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            no_nav = Path(directory) / "no-nav.epub"
            no_ncx = Path(directory) / "no-ncx.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf_no_nav = zin.read("OEBPS/content.opf").replace(
                    b'<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                    b"",
                )
                opf_no_ncx = zin.read("OEBPS/content.opf").replace(
                    b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>', b""
                )
            _copy_epub(source, no_nav, {"OEBPS/content.opf": opf_no_nav})
            _copy_epub(source, no_ncx, {"OEBPS/content.opf": opf_no_ncx})
            self.assertIn(
                "missing_source_nav",
                {item["code"] for item in validate_epub(no_nav, source_path=source)["failures"]},
            )
            self.assertIn(
                "missing_source_ncx",
                {item["code"] for item in validate_epub(no_ncx, source_path=source)["failures"]},
            )

    def test_standalone_nav_only_and_ncx_only_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            nav_only = Path(directory) / "nav-only.epub"
            ncx_only = Path(directory) / "ncx-only.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf_nav = (
                    zin.read("OEBPS/content.opf")
                    .replace(
                        b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                        b"",
                    )
                    .replace(b'<spine toc="ncx">', b"<spine>")
                )
                opf_ncx = zin.read("OEBPS/content.opf").replace(
                    b'<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                    b"",
                )
            _copy_epub(source, nav_only, {"OEBPS/content.opf": opf_nav, "OEBPS/toc.ncx": None})
            _copy_epub(source, ncx_only, {"OEBPS/content.opf": opf_ncx, "OEBPS/nav.xhtml": None})
            self.assertTrue(validate_epub(nav_only)["structural_pass"])
            self.assertTrue(validate_epub(ncx_only)["structural_pass"])

    def test_nav_and_ncx_semantics_are_checked_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            bad_nav = Path(directory) / "bad-nav.epub"
            bad_ncx = Path(directory) / "bad-ncx.epub"
            bad_media = Path(directory) / "bad-media.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                original_nav = zin.read("OEBPS/nav.xhtml")
                nav_soup = BeautifulSoup(original_nav, "xml")
                nav_node = nav_soup.find("nav")
                assert nav_node is not None
                nav_node.decompose()
                nav = str(nav_soup).encode("utf-8")
                original_ncx = zin.read("OEBPS/toc.ncx")
                ncx_soup = BeautifulSoup(original_ncx, "xml")
                nav_map = ncx_soup.find("navMap")
                assert nav_map is not None
                nav_map.name = "tocMap"
                ncx = str(ncx_soup).encode("utf-8")
                opf = zin.read("OEBPS/content.opf").replace(
                    b'media-type="application/x-dtbncx+xml"',
                    b'media-type="text/plain"',
                )
            self.assertNotEqual(original_nav, nav)
            self.assertNotEqual(original_ncx, ncx)
            _copy_epub(source, bad_nav, {"OEBPS/nav.xhtml": nav})
            _copy_epub(source, bad_ncx, {"OEBPS/toc.ncx": ncx})
            _copy_epub(source, bad_media, {"OEBPS/content.opf": opf})
            self.assertIn(
                "nav_toc_missing", {item["code"] for item in validate_epub(bad_nav)["failures"]}
            )
            self.assertIn(
                "ncx_navmap_missing", {item["code"] for item in validate_epub(bad_ncx)["failures"]}
            )
            media_codes = {item["code"] for item in validate_epub(bad_media)["failures"]}
            self.assertTrue({"ncx_manifest_media", "spine_toc_unresolved"} <= media_codes)

    def test_nav_safety_checks_apply_to_attributes_markers_duplicates_and_nesting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                original_nav = zin.read("OEBPS/nav.xhtml")
            nav_soup = BeautifulSoup(original_nav, "xml")
            heading = nav_soup.find("h1")
            toc = nav_soup.find("ol")
            assert heading is not None and toc is not None
            heading["id"] = "dup"
            heading.string = "spine-fallback"
            bad_item = nav_soup.new_tag("p", attrs={"id": "dup", "data-tn-id": "x"})
            bad_item.string = "bad"
            toc.insert(0, bad_item)
            nav = str(nav_soup).encode("utf-8")
            self.assertNotEqual(original_nav, nav)
            _copy_epub(source, broken, {"OEBPS/nav.xhtml": nav})
            codes = {item["code"] for item in validate_epub(broken)["failures"]}
            self.assertTrue({"internal_attribute", "duplicate_anchor", "illegal_nesting"} <= codes)

    def test_unmanifested_and_manifested_output_extras_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = zin.read("OEBPS/content.opf").replace(
                    b"</manifest>",
                    b'<item id="extra" href="extra.css" media-type="text/css"/></manifest>',
                )
            _copy_epub(source, output, {"OEBPS/content.opf": opf, "extra.css": b"extra"})
            codes = {item["code"] for item in validate_epub(output, source_path=source)["failures"]}
            self.assertTrue({"extra_manifest_resource", "unmanifested_resource"} <= codes)

    def test_directory_entries_are_legal_but_output_meta_inf_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            _copy_epub(source, output, {"OEBPS/extra/": b"", "META-INF/extra.xml": b"extra"})
            result = validate_epub(output, source_path=source)
            codes = {item["code"] for item in result["failures"]}
            self.assertIn("unmanifested_resource", codes)

    def test_external_unsupported_and_invalid_urls_are_private_warnings_or_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</body>",
                    b'<a href="https://private.example/x">x</a>'
                    b'<a href="foo:bar">y</a>'
                    b'<a href="abcdefghijklmnopq:bar">long</a>'
                    b'<a href="bad%zz">z</a></body>',
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(broken)
            self.assertTrue(any(i["code"] == "external_skipped" for i in result["warnings"]))
            self.assertIn("unsupported_scheme", {i["code"] for i in result["failures"]})
            self.assertNotIn("private.example", repr(result))
            unsupported = [
                item for item in result["failures"] if item["code"] == "unsupported_scheme"
            ]
            self.assertTrue(unsupported)
            self.assertTrue(all(len(item["detail"]) <= 64 for item in unsupported))
            self.assertTrue(all("abcdefghijklmnopq" not in item["detail"] for item in unsupported))

    def test_same_fragment_is_a_valid_local_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b'href="chapter-2.xhtml#chapter-two"', b'href="#intro"'
                )
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": chapter})
            self.assertNotIn(
                "missing_fragment", {i["code"] for i in validate_epub(output)["failures"]}
            )

    def test_text_html_valid_and_malformed_are_format_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            valid = Path(directory) / "valid.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = zin.read("OEBPS/content.opf").replace(
                    b'<item id="ch1" href="text/chapter-1.xhtml" media-type="application/xhtml+xml"',
                    b'<item id="ch1" href="text/chapter-1.xhtml" media-type="text/html"',
                )
                body = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</body>", b"<em>html</em></body>"
                )
                malformed = body.replace(b"</html>", b"\x00</html>")
            _copy_epub(
                source, valid, {"OEBPS/content.opf": opf, "OEBPS/text/chapter-1.xhtml": body}
            )
            _copy_epub(
                source, broken, {"OEBPS/content.opf": opf, "OEBPS/text/chapter-1.xhtml": malformed}
            )
            self.assertTrue(validate_epub(valid)["structural_pass"])
            self.assertIn(
                "malformed_content", {i["code"] for i in validate_epub(broken)["failures"]}
            )

    def test_duplicate_and_idless_manifest_items_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = (
                    zin.read("OEBPS/content.opf")
                    .replace(
                        b'<item id="style"',
                        b'<item id="ch1" href="style.css" media-type="text/css"/><item id="style"',
                    )
                    .replace(
                        b'<item id="image"',
                        b'<item href="missing.png" media-type="image/png"/><item id="image"',
                    )
                )
            _copy_epub(source, broken, {"OEBPS/content.opf": opf})
            codes = {i["code"] for i in validate_epub(broken)["failures"]}
            self.assertTrue({"manifest_id_duplicate", "manifest_id_missing"} <= codes)

    def test_generated_style_is_reported_without_path_leak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</head>",
                    b'<style id="tn-bilingual-style">'
                    + _BILINGUAL_CSS.encode("utf-8")
                    + b"</style></head>",
                )
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(output)
            self.assertTrue(
                any("tn-bilingual-style" in value for value in result["generated_resources"])
            )
            self.assertNotIn(directory, repr(result))

    def test_legal_image_only_body_is_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            image_only = Path(directory) / "image-only.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml")
                start, end = chapter.index(b"<body>"), chapter.index(b"</body>") + len(b"</body>")
                chapter = (
                    chapter[:start]
                    + b'<body><img src="../images/figure.png"/></body>'
                    + chapter[end:]
                )
            _copy_epub(source, image_only, {"OEBPS/text/chapter-1.xhtml": chapter})
            self.assertNotIn(
                "empty_content", {i["code"] for i in validate_epub(image_only)["failures"]}
            )

    def test_anonymous_noteref_fails_even_with_arbitrary_backlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter_one = zin.read("OEBPS/text/chapter-1.xhtml").replace(b' id="ref-1"', b"")
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter_one})
            self.assertIn(
                "missing_backlink", {i["code"] for i in validate_epub(broken)["failures"]}
            )

    def test_checked_counts_include_failed_and_warned_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            broken = Path(directory) / "broken.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</body>",
                    b'<a href="https://example.test/x">external</a><a href="missing.xhtml">missing</a></body>',
                )
            _copy_epub(source, broken, {"OEBPS/text/chapter-1.xhtml": chapter})
            result = validate_epub(broken)
            self.assertGreaterEqual(result["counts"]["internal_links"]["checked"], 2)
            self.assertGreater(result["counts"]["internal_links"]["warnings"], 0)

    def test_changed_manifest_media_and_properties_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = zin.read("OEBPS/content.opf").replace(
                    b'id="style" href="style.css" media-type="text/css"',
                    b'id="style" href="style.css" media-type="text/plain" properties="cover"',
                )
            _copy_epub(source, output, {"OEBPS/content.opf": opf})
            self.assertIn(
                "manifest_metadata_mismatch",
                {i["code"] for i in validate_epub(output, source_path=source)["failures"]},
            )

    def test_mono_source_and_reserved_style_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = (
                    zin.read("OEBPS/text/chapter-1.xhtml")
                    .replace(b"</head>", b'<style id="tn-bilingual-style">bad</style></head>')
                    .replace(b'<p id="intro">', b'<p class="tn-source">source</p><p id="intro">')
                )
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": chapter})
            codes = {
                item["code"]
                for item in validate_epub(output, source_path=source, bilingual=False)["failures"]
            }
            self.assertIn("unexpected_source_nodes", codes)
            self.assertIn("generated_resource_mismatch", codes)

    def test_illegal_table_direct_child_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"<table>", b"<table><p>bad</p>", 1
                )
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": chapter})
            self.assertIn(
                "illegal_nesting", {item["code"] for item in validate_epub(output)["failures"]}
            )

    def test_graph_removal_and_duplicate_multiplicity_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            removed = Path(directory) / "removed.epub"
            duplicated = Path(directory) / "duplicated.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml")
                no_image = chapter.replace(b'<img src="../images/figure.png" alt="Figure"/>', b"")
                many_images = chapter.replace(
                    b'<img src="../images/figure.png" alt="Figure"/>',
                    b'<img src="../images/figure.png" alt="Figure"/><img src="../images/figure.png" alt="Figure"/>',
                )
            _copy_epub(source, removed, {"OEBPS/text/chapter-1.xhtml": no_image})
            _copy_epub(source, duplicated, {"OEBPS/text/chapter-1.xhtml": many_images})
            self.assertIn(
                "reference_graph_mismatch",
                {item["code"] for item in validate_epub(removed, source_path=source)["failures"]},
            )
            self.assertIn(
                "reference_graph_mismatch",
                {
                    item["code"]
                    for item in validate_epub(duplicated, source_path=source)["failures"]
                },
            )

    def test_direct_br_line_wrappers_validate_both_orders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.epub"
            source = Path(directory) / "source.epub"
            mono = Path(directory) / "mono.epub"
            write_phase9_epub(str(original))
            with zipfile.ZipFile(original) as zin:
                source_soup = BeautifulSoup(zin.read("OEBPS/text/chapter-2.xhtml"), "xml")
                chapter_one_soup = BeautifulSoup(zin.read("OEBPS/text/chapter-1.xhtml"), "xml")
                chapter_one_style = chapter_one_soup.new_tag("style", id="tn-bilingual-style")
                chapter_one_style.string = _BILINGUAL_CSS
                chapter_one_soup.head.append(chapter_one_style)
                chapter_one_data = str(chapter_one_soup).encode("utf-8")
            source_body = source_soup.find(id="body-two")
            assert source_body is not None
            source_body.clear()
            source_body.append("Line one")
            source_body.append(source_soup.new_tag("br"))
            source_body.append("Line two")
            source_data = str(source_soup).encode("utf-8")
            mono_soup = BeautifulSoup(source_data, "xml")
            mono_body = mono_soup.find(id="body-two")
            assert mono_body is not None
            mono_body.clear()
            mono_body.append("Translated one")
            mono_body.append(mono_soup.new_tag("br"))
            mono_body.append("Translated two")
            mono_data = str(mono_soup).encode("utf-8")
            _copy_epub(original, source, {"OEBPS/text/chapter-2.xhtml": source_data})
            _copy_epub(original, mono, {"OEBPS/text/chapter-2.xhtml": mono_data})
            for order in ("target_first", "source_first"):
                bilingual = Path(directory) / f"bi-{order}.epub"
                bi_soup = BeautifulSoup(mono_data, "xml")
                bi_body = bi_soup.find(id="body-two")
                assert bi_body is not None
                bi_body.clear()
                for index, (target_text, source_text) in enumerate(
                    (("Translated one", "Line one"), ("Translated two", "Line two"))
                ):
                    if index:
                        bi_body.append(bi_soup.new_tag("br"))
                    target = bi_soup.new_tag("span")
                    target.string = target_text
                    source_span = bi_soup.new_tag("span", attrs={"class": "tn-source"})
                    source_span.string = source_text
                    if order == "target_first":
                        bi_body.extend([target, source_span])
                    else:
                        bi_body.extend([source_span, target])
                style = bi_soup.new_tag("style", id="tn-bilingual-style")
                style.string = _BILINGUAL_CSS
                bi_soup.head.append(style)
                _copy_epub(
                    original,
                    bilingual,
                    {
                        "OEBPS/text/chapter-1.xhtml": chapter_one_data,
                        "OEBPS/text/chapter-2.xhtml": str(bi_soup).encode("utf-8"),
                    },
                )
                self.assertTrue(validate_epub_triplet(source, mono, bilingual)["structural_pass"])

    def test_inline_style_script_attribute_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            styled_source = Path(directory) / "styled-source.epub"
            preserved = Path(directory) / "preserved.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</head>", b'<style id="source-style" media="screen">x</style></head>'
                )
                tampered = chapter.replace(
                    b"</head>", b'<script type="text/javascript">x</script></head>'
                )
            _copy_epub(source, styled_source, {"OEBPS/text/chapter-1.xhtml": chapter})
            _copy_epub(styled_source, preserved)
            self.assertTrue(validate_epub(preserved, source_path=styled_source)["structural_pass"])
            _copy_epub(source, output, {"OEBPS/text/chapter-1.xhtml": tampered})
            codes = {item["code"] for item in validate_epub(output, source_path=source)["failures"]}
            self.assertIn("inline_resource_mismatch", codes)

    def test_manifest_declared_typeless_nav_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "typeless-nav.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = (
                    zin.read("OEBPS/content.opf")
                    .replace(
                        b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                        b"",
                    )
                    .replace(b'<spine toc="ncx">', b"<spine>")
                )
                nav = zin.read("OEBPS/nav.xhtml").replace(b'epub:type="toc"', b"")
            _copy_epub(
                source,
                output,
                {
                    "OEBPS/content.opf": opf,
                    "OEBPS/nav.xhtml": nav,
                    "OEBPS/toc.ncx": None,
                },
            )
            self.assertTrue(validate_epub(output)["structural_pass"])

    def test_typed_toc_precedes_untyped_landmark_nav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "typed-toc-landmark.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                nav = zin.read("OEBPS/nav.xhtml").replace(
                    b"</nav></body>",
                    b'</nav><nav epub:type="landmarks"><p>Landmarks</p></nav></body>',
                    1,
                )
            _copy_epub(source, output, {"OEBPS/nav.xhtml": nav})
            self.assertTrue(validate_epub(output)["structural_pass"])

    def test_only_first_untyped_nav_is_toc_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "two-untyped-navs.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                nav = (
                    zin.read("OEBPS/nav.xhtml")
                    .replace(b'epub:type="toc"', b"")
                    .replace(
                        b"</nav></body>",
                        b"</nav><nav><p>Secondary navigation</p></nav></body>",
                        1,
                    )
                )
            _copy_epub(source, output, {"OEBPS/nav.xhtml": nav})
            self.assertTrue(validate_epub(output)["structural_pass"])

    def test_arbitrary_content_nav_is_not_toc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "arbitrary-nav.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = (
                    zin.read("OEBPS/content.opf")
                    .replace(b' properties="nav"', b"")
                    .replace(
                        b'<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                        b"",
                    )
                    .replace(b'<spine toc="ncx">', b"<spine>")
                )
                chapter = zin.read("OEBPS/text/chapter-1.xhtml").replace(
                    b"</body>",
                    b'<nav><ol><li><a href="chapter-2.xhtml#chapter-two">Local</a></li></ol></nav></body>',
                )
            _copy_epub(
                source,
                output,
                {
                    "OEBPS/content.opf": opf,
                    "OEBPS/text/chapter-1.xhtml": chapter,
                    "OEBPS/toc.ncx": None,
                },
            )
            self.assertIn(
                "missing_toc", {item["code"] for item in validate_epub(output)["failures"]}
            )

    def test_recoverable_html5_and_no_fetch_external(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            output = Path(directory) / "output.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                opf = zin.read("OEBPS/content.opf").replace(
                    b'<item id="ch1" href="text/chapter-1.xhtml" media-type="application/xhtml+xml"',
                    b'<item id="ch1" href="text/chapter-1.xhtml" media-type="text/html"',
                )
                html = b'<!doctype html><html><body><p>ok<a href="https://example.invalid/x">x</a>'
            _copy_epub(
                source, output, {"OEBPS/content.opf": opf, "OEBPS/text/chapter-1.xhtml": html}
            )
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                result = validate_epub(output)
            self.assertNotIn("malformed_content", {item["code"] for item in result["failures"]})
            self.assertTrue(any(item["code"] == "external_skipped" for item in result["warnings"]))

    def test_wrong_target_pair_with_equal_source_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.epub"
            mono = Path(directory) / "mono.epub"
            bilingual = Path(directory) / "bilingual.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                mono_soup = BeautifulSoup(zin.read("OEBPS/text/chapter-2.xhtml"), "xml")
                target = mono_soup.find(id="body-two")
                assert target is not None
                target.string = "Translated"
                mono_data = str(mono_soup).encode("utf-8")
                bi_soup = BeautifulSoup(mono_data, "xml")
                source_node = bi_soup.new_tag("p", attrs={"class": "tn-source"})
                source_node.string = "Second chapter."
                heading = bi_soup.find(id="chapter-two")
                assert heading is not None
                heading.insert_before(source_node)
                bi_data = str(bi_soup).encode("utf-8")
            _copy_epub(source, mono, {"OEBPS/text/chapter-2.xhtml": mono_data})
            _copy_epub(source, bilingual, {"OEBPS/text/chapter-2.xhtml": bi_data})
            self.assertIn(
                "source_target_pair_mismatch",
                {
                    item["code"]
                    for item in validate_epub_triplet(source, mono, bilingual)["bilingual"][
                        "failures"
                    ]
                },
            )

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


if __name__ == "__main__":
    unittest.main()
