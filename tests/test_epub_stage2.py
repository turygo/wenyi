from __future__ import annotations

import hashlib
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lxml import etree

from tests.sample_data import write_phase9_epub
from trans_novel.assemble.epub_verifier import (
    EpubPublishError,
    EpubVerificationError,
    _bilingual_proof,
    _nav_label_locations,
    _source_subtree_signature,
    publish_epub,
    verify_epub,
)
from trans_novel.assemble.writer import _BILINGUAL_CSS, _rewrite_html_document
from trans_novel.pipeline.runstore import RunStore


class TestEpubStage2(unittest.TestCase):
    def test_failed_verification_preserves_existing_final_and_persists_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = RunStore(str(root / "state"))
            final = root / "book.epub"
            final.write_bytes(b"previous-final")

            def corrupt(path: str) -> None:
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("mimetype", b"not-an-epub", zipfile.ZIP_STORED)

            with self.assertRaises(EpubVerificationError) as raised:
                publish_epub(state, None, final, mode="generated", writer=corrupt)
            self.assertFalse(raised.exception.published)
            self.assertEqual(final.read_bytes(), b"previous-final")
            report = state.load_epub_verification()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertFalse(report["passed"])
            self.assertFalse(report["published"])
            self.assertEqual(report["output_label"], "book.epub")
            self.assertEqual(raised.exception.report, report)
            self.assertFalse(list(root.glob(".book.epub.epub-verify-*.tmp")))
            self.assertTrue(state.report_path != state.epub_verification_path)

    def test_success_publishes_report_and_stable_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            state = RunStore(str(root / "state"))
            final = root / "published.epub"
            publish_epub(
                state,
                None,
                final,
                mode="generated",
                writer=lambda path: shutil.copyfile(source, path),
            )
            report = state.load_epub_verification()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertTrue(report["passed"])
            self.assertTrue(report["published"])
            self.assertEqual(report["mode"], "generated")
            self.assertIsNone(report["source_sha256"])
            self.assertEqual(report["output_label"], "published.epub")
            event = Path(state.event_log_path).read_text(encoding="utf-8").splitlines()[-1]
            self.assertIn('"event": "epub_verification_passed"', event)
            self.assertIn('"output": "published.epub"', event)

    def test_identical_bytes_have_relocation_stable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one"
            second = Path(directory) / "two"
            first.mkdir()
            second.mkdir()
            one = first / "book.epub"
            two = second / "book.epub"
            write_phase9_epub(str(one))
            shutil.copyfile(one, two)
            report_one = verify_epub(one, mode="generated")
            report_two = verify_epub(two, mode="generated")
            self.assertEqual(report_one, report_two)

    def test_real_schema3_mono_report_counts_actual_authorized_changes(self) -> None:
        from tests.test_assemble import _run
        from trans_novel.assemble.writer import assemble

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            store, _ = _run(str(source), str(root / "state"))
            output = root / "mono.epub"
            assemble(store, str(source), str(output), out_format="epub")
            report = store.load_epub_verification()
            assert report is not None
            self.assertTrue(report["passed"])
            self.assertEqual(
                report["authorized_differences"],
                {"text_slots": 19, "toc_labels": 4, "language_fields": 1, "bilingual_nodes": 0},
            )

    def test_real_schema3_bilingual_report_proves_both_orders(self) -> None:
        from tests.test_assemble import _run
        from trans_novel.assemble.writer import assemble

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            store, _ = _run(str(source), str(root / "state"))
            for order in ("target_first", "source_first"):
                output = root / f"{order}.epub"
                assemble(
                    store,
                    str(source),
                    str(output),
                    out_format="epub",
                    bilingual=True,
                    order=order,
                )
                report = store.load_epub_verification()
                assert report is not None
                self.assertTrue(report["passed"])
                self.assertEqual(report["authorized_differences"]["bilingual_nodes"], 12)

    def test_style_only_rewrite_preserves_vertical_root_and_language_keys(self) -> None:
        source = b'<html class="vrtl keep" xml:lang="ja"><head></head><body><p>Original</p></body></html>'
        output = _rewrite_html_document(
            source,
            lang="zh-Hans",
            force_horizontal=False,
            bilingual=True,
            rewrite_language=False,
        )
        source_root = etree.fromstring(source)
        output_root = etree.fromstring(output)
        self.assertEqual(set(source_root.attrib), set(output_root.attrib))
        self.assertEqual(output_root.get("class"), "vrtl keep")
        self.assertIsNotNone(output_root.xpath(".//style[@id='tn-bilingual-style']"))

    def test_direct_br_pairing_order_is_checked_for_both_orders(self) -> None:
        source = etree.fromstring(b"<html><head></head><body><p>One<br/>Two</p></body></html>")
        state_one = SimpleNamespace(
            block_path=(1, 0),
            slots=[
                SimpleNamespace(
                    element_path=(),
                    field="text",
                    source_value="One",
                    source_core="One",
                    target_core="Uno",
                )
            ],
        )
        state_two = SimpleNamespace(
            block_path=(1, 0),
            slots=[
                SimpleNamespace(
                    element_path=(0,),
                    field="tail",
                    source_value="Two",
                    source_core="Two",
                    target_core="Dos",
                )
            ],
        )
        segments = [
            SimpleNamespace(kind="text", source="One", target="Uno", epub_state=state_one),
            SimpleNamespace(kind="text", source="Two", target="Dos", epub_state=state_two),
        ]

        def check(
            order: str,
            invert_first_pair: bool,
            source_root: etree._Element = source,
            block_tag: str = "p",
        ) -> list[dict[str, str]]:
            target_first = (
                '<span>Uno</span><span class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">One</span>'
            )
            source_first = (
                '<span class="tn-source ibooks-dark-theme-use-custom-text-color">'
                "One</span><span>Uno</span>"
            )
            second = (
                '<span>Dos</span><span class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">Two</span>'
                if order == "target_first"
                else '<span class="tn-source ibooks-dark-theme-use-custom-text-color">'
                "Two</span><span>Dos</span>"
            )
            first = (
                source_first if (order == "target_first" and invert_first_pair) else target_first
            )
            if order == "source_first":
                first = target_first if invert_first_pair else source_first
            markup = (
                f"<html><head><style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style></head>"
                f"<body><{block_tag}>{first}<br/>{second}</{block_tag}></body></html>"
            ).encode()
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source_root,
                etree.fromstring(markup),
                segments,
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            return failures

        for order in ("source_first", "target_first"):
            self.assertNotIn("source_node_order", {item["code"] for item in check(order, False)})
            self.assertIn("source_node_order", {item["code"] for item in check(order, True)})
        div_source = etree.fromstring(
            b"<html><head></head><body><div>One<br/>Two</div></body></html>"
        )
        for order in ("source_first", "target_first"):
            self.assertNotIn(
                "source_node_order",
                {item["code"] for item in check(order, False, div_source, "div")},
            )
            self.assertIn(
                "source_node_order",
                {item["code"] for item in check(order, True, div_source, "div")},
            )

    def test_attributed_wrapped_japanese_ruby_source_passes_canonical_proof(self) -> None:
        source = etree.fromstring(
            "<html><head></head><body><p><em>前<ruby id='drop' class='keep'>"
            "<rb data-note='keep'>漢</rb><rt>かん</rt><rp>（</rp></ruby>後</em></p>"
            "</body></html>".encode()
        )
        output = etree.fromstring(
            f"<html><head><style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style></head>"
            '<body><p>译文</p><p class="tn-source ibooks-dark-theme-use-custom-text-color">'
            "前<ruby class='keep'><rb data-note='keep'>漢</rb><rt>かん</rt><rp>（</rp></ruby>後"
            "</p></body></html>".encode()
        )
        segment = SimpleNamespace(
            kind="text",
            source="前漢後",
            target="译文",
            epub_state=SimpleNamespace(block_path=(1, 0)),
        )
        from trans_novel.assemble.writer import _bilingual_source_copy

        expected = _bilingual_source_copy(
            source.find(".//p"),
            source.find(".//p"),
            source_lang="ja",
            source_tag="p",
        )
        actual = output.xpath(".//p[@class]")[0]
        self.assertEqual(_source_subtree_signature(expected), _source_subtree_signature(actual))
        failures: list[dict[str, str]] = []
        _bilingual_proof(
            source,
            output,
            [segment],
            source_lang="ja",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertNotIn("source_node_subtree_mismatch", {item["code"] for item in failures})
        self.assertNotIn("source_node_ruby_mismatch", {item["code"] for item in failures})

    def test_bilingual_source_active_media_and_reserved_style_are_checked(self) -> None:
        source = etree.fromstring(b"<html><body><p>Original</p></body></html>")
        output = etree.fromstring(
            b'<html><body><p>Translated</p><p class="tn-source">'
            b"<img src='active.png'/>Original</p></body></html>"
        )
        state = SimpleNamespace(block_path=(0, 0))
        segment = SimpleNamespace(
            kind="text",
            source="Original",
            target="Translated",
            epub_state=state,
        )
        failures: list[dict[str, str]] = []
        count = _bilingual_proof(
            source,
            output,
            [segment],
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertEqual(count, 1)
        self.assertIn("source_node_attributes", {item["code"] for item in failures})
        self.assertIn("source_node_active_media", {item["code"] for item in failures})
        self.assertIn("bilingual_style_count", {item["code"] for item in failures})

    def test_nested_bilingual_order_is_checked_in_parent_text(self) -> None:
        source = etree.fromstring(b"<html><body><li>Original</li></body></html>")
        state = SimpleNamespace(block_path=(0, 0))
        segment = SimpleNamespace(
            kind="text",
            source="Original",
            target="Translated",
            epub_state=state,
        )
        output = etree.fromstring(
            b'<html><body><li><div class="tn-source '
            b'ibooks-dark-theme-use-custom-text-color">Original</div>Translated</li></body></html>'
        )
        failures: list[dict[str, str]] = []
        _bilingual_proof(
            source,
            output,
            [segment],
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertIn("source_node_order", {item["code"] for item in failures})

    def test_nested_nav_label_locator_matches_writer_clear_contract(self) -> None:
        root = etree.fromstring(
            b'<html xmlns:epub="http://www.idpf.org/2007/ops"><body>'
            b'<nav epub:type="toc"><ol><li><a href="chapter.xhtml">'
            b"Old <span><em>Nested</em></span></a></li></ol></nav></body></html>"
        )
        locations = _nav_label_locations(root, is_ncx=False)
        self.assertEqual(len(locations), 1)
        label, path = locations[0]
        self.assertEqual(label.tag.rsplit("}", 1)[-1], "a")
        self.assertEqual(path, (0, 0, 0, 0, 0))

    def test_corrupt_ruby_source_subtree_fails_exactly(self) -> None:
        source = etree.fromstring(
            b"<html><body><p><ruby>\xe6\xbc\xa2<rt>\xe3\x81\x8b\xe3\x82\x93</rt></ruby></p></body></html>"
        )
        state = SimpleNamespace(block_path=(0, 0))
        segment = SimpleNamespace(
            kind="text",
            source="\u6f22",
            target="\u8bd1",
            epub_state=state,
        )
        outputs = (
            b'<html><body><p>\xe8\xaf\x91</p><p class="tn-source ibooks-dark-theme-use-custom-text-color">'
            b"<ruby>\xe6\xbc\xa2<rt>\xe6\x94\xb9</rt></ruby></p></body></html>",
            b'<html><body><p>\xe8\xaf\x91</p><p class="tn-source ibooks-dark-theme-use-custom-text-color">'
            b"<ruby><rb>\xe6\xbc\xa2</rb><rt>\xe3\x81\x8b\xe3\x82\x93</rt></ruby></p></body></html>",
        )
        for data in outputs:
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                etree.fromstring(data),
                [segment],
                source_lang="ja",
                order="target_first",
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertTrue(
                {
                    "source_node_ruby_mismatch",
                    "source_node_subtree_mismatch",
                }
                & {item["code"] for item in failures}
            )

    def test_source_dom_slot_tag_attr_and_nav_mutations_fail_exactly(self) -> None:
        from tests.test_assemble import _run

        def mutate(source: Path, output: Path, member: str, transform) -> None:
            with zipfile.ZipFile(source) as zin, zipfile.ZipFile(output, "w") as zout:
                zout.comment = zin.comment
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    if info.filename == member:
                        data = transform(data)
                    zout.writestr(info, data)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            store, _ = _run(str(source), str(root / "state"))
            cases = (
                (
                    "OEBPS/text/chapter-1.xhtml",
                    lambda data: data.replace(b'id="intro"', b'id="changed"'),
                    "unauthorized_dom_change",
                ),
                (
                    "OEBPS/text/chapter-1.xhtml",
                    lambda data: data.replace(b'<p id="intro">', b'<div id="intro">'),
                    "unauthorized_dom_change",
                ),
                (
                    "OEBPS/text/chapter-1.xhtml",
                    lambda data: data.replace("润".encode(), "错误".encode(), 1),
                    "slot_value_mismatch",
                ),
                (
                    "OEBPS/nav.xhtml",
                    lambda data: data.replace(b"Chapter One", b"Wrong label", 1),
                    "label_value_mismatch",
                ),
            )
            for index, (member, transform, expected_code) in enumerate(cases):
                output = root / f"mutated-{index}.epub"
                mutate(source, output, member, transform)
                report = verify_epub(
                    output,
                    source_path=source,
                    store=store,
                    mode="monolingual",
                )
                self.assertFalse(report["passed"], report)
                self.assertIn(expected_code, {item["code"] for item in report["failures"]})

    def test_tools_assemble_uses_the_same_publication_gate(self) -> None:
        from typer.testing import CliRunner

        from tests.sample_data import write_sample_txt
        from tests.test_bilingual import _run
        from trans_novel.cli import app

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "novel.txt"
            write_sample_txt(str(source))
            _, config = _run(str(source), str(Path(directory) / "state"))
            with patch("trans_novel.cli._load_config", return_value=config):
                result = CliRunner().invoke(app, ["tools", "assemble", str(source), "--mono"])
            self.assertEqual(result.exit_code, 0, result.output)
            output = Path(directory) / "novel.zh.epub"
            self.assertTrue(output.is_file())
            verification_files = list((Path(directory) / "state").rglob("epub_verification.json"))
            self.assertEqual(len(verification_files), 1)
            report = __import__("json").loads(verification_files[0].read_text(encoding="utf-8"))
            self.assertTrue(report["passed"])
            event_files = list((Path(directory) / "state").rglob("events.jsonl"))
            self.assertTrue(event_files)
            self.assertIn(
                '"event": "epub_verification_passed"',
                event_files[0].read_text(encoding="utf-8"),
            )

    def test_normal_finish_assemble_node_reaches_the_publication_gate(self) -> None:
        from tests.sample_data import write_sample_txt
        from tests.test_bilingual import _run
        from trans_novel.pipeline.contracts import NodeRequest
        from trans_novel.pipeline.nodes.finish import AssembleNode

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "novel.txt"
            write_sample_txt(str(source))
            store, config = _run(str(source), str(Path(directory) / "state"))
            outcome = AssembleNode(config=config, out_format="epub").execute(
                NodeRequest(
                    store=store,
                    node_id="assemble",
                    key="assemble",
                    ci=None,
                    scope="book",
                    input_path=str(source),
                    progress=None,
                )
            )
            self.assertEqual(len(outcome.artifacts["outputs"]), 2)
            self.assertTrue(all(Path(path).is_file() for path in outcome.artifacts["outputs"]))
            report = store.load_epub_verification()
            self.assertIsNotNone(report)
            assert report is not None
            self.assertTrue(report["passed"])

    def test_report_persistence_failure_is_the_exception_cause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            store = RunStore(str(root / "state"))
            persistence_error = RuntimeError("persist failed")
            with (
                patch.object(store, "save_epub_verification", side_effect=persistence_error),
                self.assertRaises(EpubPublishError) as raised,
            ):
                publish_epub(
                    store,
                    None,
                    root / "published.epub",
                    mode="generated",
                    writer=lambda path: shutil.copyfile(source, path),
                )
            self.assertIs(raised.exception.__cause__, persistence_error)
            self.assertFalse(raised.exception.published)

    def test_preflight_report_persistence_failure_is_chained(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RunStore(str(root / "state"))
            persistence_error = RuntimeError("persist failed")
            with (
                patch.object(store, "save_epub_verification", side_effect=persistence_error),
                self.assertRaises(EpubPublishError) as raised,
            ):
                publish_epub(
                    store,
                    None,
                    root / "missing" / "published.epub",
                    mode="generated",
                    writer=lambda path: None,
                )
            self.assertIs(raised.exception.__cause__, persistence_error)
            self.assertFalse(raised.exception.published)

    def test_replace_failure_persists_failed_event_and_preserves_final(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            final = root / "published.epub"
            final.write_bytes(b"existing")
            state = RunStore(str(root / "state"))
            with (
                patch("trans_novel.assemble.epub_verifier.os.replace", side_effect=OSError("no")),
                self.assertRaisesRegex(Exception, "EPUB publication failed") as raised,
            ):
                publish_epub(
                    state,
                    None,
                    final,
                    mode="generated",
                    writer=lambda path: shutil.copyfile(source, path),
                )
            self.assertEqual(final.read_bytes(), b"existing")
            report = getattr(raised.exception, "report", None)
            assert report is not None
            self.assertFalse(report["published"])

    def test_post_replace_fsync_failure_reports_published_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            final = root / "published.epub"
            state = RunStore(str(root / "state"))
            with (
                patch(
                    "trans_novel.assemble.epub_verifier._fsync_file",
                    side_effect=OSError("fsync"),
                ),
                self.assertRaises(Exception) as raised,
            ):
                publish_epub(
                    state,
                    None,
                    final,
                    mode="generated",
                    writer=lambda path: shutil.copyfile(source, path),
                )
            self.assertTrue(raised.exception.published)
            report = state.load_epub_verification()
            assert report is not None
            self.assertTrue(report["published"])
            self.assertFalse(report["passed"])
            self.assertTrue(any(item["code"] == "durability_failed" for item in report["failures"]))

    def test_recovered_schema3_resource_reports_recovered_assurance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "recovered.epub"
            chapter = b"<html xmlns='http://www.w3.org/1999/xhtml'><body><p>broken"
            container = (
                b"<container xmlns='urn:oasis:names:tc:opendocument:xmlns:container'>"
                b"<rootfiles><rootfile full-path='content.opf'/></rootfiles></container>"
            )
            opf = (
                b"<package xmlns='http://www.idpf.org/2007/opf'><metadata>"
                b"<dc:language xmlns:dc='http://purl.org/dc/elements/1.1/'>en</dc:language>"
                b"</metadata><manifest><item id='c' href='chapter.xhtml' "
                b"media-type='application/xhtml+xml'/></manifest><spine><itemref idref='c'/>"
                b"</spine></package>"
            )
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("mimetype", b"application/epub+zip", zipfile.ZIP_STORED)
                archive.writestr("META-INF/container.xml", container)
                archive.writestr("content.opf", opf)
                archive.writestr("chapter.xhtml", chapter)

            from trans_novel.ingest.epub_reader import _resource_parser

            parser_diagnostics = _resource_parser(chapter)[2]

            class Store:
                def load_manifest(self):
                    return {
                        "target_lang": "zh",
                        "chapters": [],
                        "meta": {
                            "epub_schema": 3,
                            "epub_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "epub_resources": [
                                {
                                    "href": "chapter.xhtml",
                                    "resource_sha256": hashlib.sha256(chapter).hexdigest(),
                                    "parse_mode": "recovered",
                                    "parser_diagnostics": parser_diagnostics,
                                }
                            ],
                        },
                    }

                def load_chapter(self, index):
                    raise AssertionError(index)

            report = verify_epub(
                source,
                source_path=source,
                store=Store(),
                mode="monolingual",
            )
            self.assertFalse(report["passed"])
            self.assertTrue(
                any(item["code"].startswith("recovered_diagnostic_") for item in report["warnings"])
            )
            mismatch_store = Store()
            original_load_manifest = mismatch_store.load_manifest

            def load_mismatch():
                value = original_load_manifest()
                value["meta"]["epub_resources"][0]["parser_diagnostics"] = []
                return value

            mismatch_store.load_manifest = load_mismatch
            mismatch_report = verify_epub(
                source,
                source_path=source,
                store=mismatch_store,
                mode="monolingual",
            )
            self.assertTrue(
                any(
                    item["code"] == "recovered_diagnostic_mismatch"
                    for item in mismatch_report["failures"]
                )
            )


if __name__ == "__main__":
    unittest.main()
