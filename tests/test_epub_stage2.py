from __future__ import annotations

import hashlib
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lxml import etree

from tests.sample_data import write_phase9_epub
from trans_novel.assemble.bilingual_dom import (
    BILINGUAL_CSS as _BILINGUAL_CSS,
)
from trans_novel.assemble.bilingual_dom import (
    BILINGUAL_DIRECT_TARGET_CLASS,
    BILINGUAL_SOURCE_CLASS,
)
from trans_novel.assemble.epub_verifier import (
    EpubPublishError,
    EpubVerificationError,
    _bilingual_proof,
    _compare_dom,
    _nav_label_locations,
    _source_subtree_signature,
    publish_epub,
    verify_epub,
)
from trans_novel.assemble.writer import _add_bilingual_sources
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

    def test_input_output_aliases_are_rejected_before_any_writer(self) -> None:
        from trans_novel.assemble.writer import assemble

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            before = source.read_bytes()
            store = SimpleNamespace(
                load_manifest=lambda: {"fmt": "text", "target_lang": "zh"},
            )
            with (
                patch("trans_novel.pipeline.readiness.ensure_assemble_ready"),
                self.assertRaises(ValueError),
            ):
                assemble(store, str(source), str(source), out_format="txt")
            self.assertEqual(source.read_bytes(), before)

            generated_store = RunStore(str(root / "generated-state"))
            called = []
            with self.assertRaises(EpubPublishError) as raised:
                publish_epub(
                    generated_store,
                    None,
                    source,
                    mode="generated",
                    source_identity_path=source,
                    writer=lambda path: called.append(path),
                )
            self.assertEqual(raised.exception.report["failures"][0]["code"], "input_output_alias")
            self.assertEqual(called, [])
            self.assertEqual(source.read_bytes(), before)

    def test_generated_assemble_alias_is_rejected_before_writer(self) -> None:
        from trans_novel.assemble.writer import assemble

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "book.txt"
            source.write_text("source", encoding="utf-8")

            class Store:
                def load_manifest(self):
                    return {"fmt": "text", "target_lang": "zh"}

            with (
                patch("trans_novel.pipeline.readiness.ensure_assemble_ready"),
                self.assertRaisesRegex(ValueError, "paths must differ"),
            ):
                assemble(Store(), str(source), str(source), out_format="epub")

    def test_source_assemble_rejects_unsafe_zip_before_output_write(self) -> None:
        from trans_novel.assemble.writer import _assemble_source_epub

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unsafe.epub"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("META-INF/x/../container.xml", b"<container/>")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()

            class Store:
                def load_manifest(self):
                    return {
                        "meta": {
                            "epub_schema": 4,
                            "epub_sha256": digest,
                            "epub_resources": [],
                        },
                        "source_lang": "en",
                        "chapters": [],
                    }

            output = Path(directory) / "output.epub"
            with self.assertRaisesRegex(ValueError, "unsafe_entry"):
                _assemble_source_epub(Store(), str(source), str(output), target_lang="zh")
            self.assertFalse(output.exists())

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

    def test_real_schema4_mono_report_counts_actual_authorized_changes(self) -> None:
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

    def test_real_schema4_bilingual_report_proves_both_orders(self) -> None:
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

    def test_direct_br_pairing_order_is_checked_for_both_orders(self) -> None:
        source = etree.fromstring(b"<html><head></head><body><p>One<br/>Two</p></body></html>")
        state_one = SimpleNamespace(
            block_path=(1, 0),
            slots=[
                SimpleNamespace(
                    element_path=(),
                    field="text",
                    source_value="One",
                    target_value="Uno",
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
                    target_value="Dos",
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
                '<span class="tn-bilingual-target">Uno</span><span class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">One</span>'
            )
            source_first = (
                '<span class="tn-source ibooks-dark-theme-use-custom-text-color">'
                'One</span><span class="tn-bilingual-target">Uno</span>'
            )
            second = (
                '<span class="tn-bilingual-target">Dos</span><span class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">Two</span>'
                if order == "target_first"
                else '<span class="tn-source ibooks-dark-theme-use-custom-text-color">'
                'Two</span><span class="tn-bilingual-target">Dos</span>'
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
            "前<ruby class='keep'><rb>漢</rb><rt>かん</rt><rp>（</rp></ruby>後"
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

    def test_container_mixed_content_accepts_both_orders_and_rejects_inversion(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><li><em>Original</em> tail</li></body></html>"
        )
        segment = SimpleNamespace(
            kind="text",
            source="Original tail",
            target="Translated tail",
            epub_state=SimpleNamespace(block_path=(1, 0)),
        )
        source_markup = (
            '<div class="tn-source ibooks-dark-theme-use-custom-text-color">'
            "<em>Original</em> tail</div>"
        )
        for order, body in (
            (
                "target_first",
                f"<li><em>Translated</em> tail{source_markup}</li>",
            ),
            (
                "source_first",
                f"<li>{source_markup}<em>Translated</em> tail</li>",
            ),
        ):
            output = etree.fromstring(
                (
                    f"<html><head><style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style>"
                    f"</head><body>{body}</body></html>"
                ).encode()
            )
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertNotIn("source_node_order", {item["code"] for item in failures})

        inverted = etree.fromstring(
            (
                f"<html><head><style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style>"
                f"</head><body><li>{source_markup}<em>Translated</em> tail</li></body></html>"
            ).encode()
        )
        failures = []
        _bilingual_proof(
            source,
            inverted,
            [segment],
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertIn("source_node_order", {item["code"] for item in failures})

    def test_repeated_source_blocks_match_distinct_adjacent_targets(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><p>Same</p><p>Middle</p><p>Same</p></body></html>"
        )
        output = etree.fromstring(etree.tostring(source))
        paths = ((1, 0), (1, 1), (1, 2))
        segments = [
            SimpleNamespace(
                kind="text",
                source=text,
                target=target,
                epub_state=SimpleNamespace(block_path=path, slots=[]),
            )
            for path, text, target in zip(
                paths, ("Same", "Middle", "Same"), ("同一", "中间", "同一"), strict=True
            )
        ]
        blocks = source.xpath(".//p")
        output_blocks = output.xpath(".//p")
        _add_bilingual_sources(
            output,
            segments,
            order="target_first",
            source_blocks={
                path: etree.fromstring(etree.tostring(block))
                for path, block in zip(paths, blocks, strict=True)
            },
            block_refs=dict(zip(paths, output_blocks, strict=True)),
        )
        style = etree.Element("style", id="tn-bilingual-style")
        style.text = _BILINGUAL_CSS
        output.find("head").append(style)
        failures: list[dict[str, str]] = []
        _bilingual_proof(
            source,
            output,
            segments,
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertEqual(failures, [])

    def test_container_direct_br_uses_one_nested_source_div_in_both_orders(self) -> None:
        source = etree.fromstring(b"<html><head></head><body><li>One<br/>Two</li></body></html>")
        block = source.xpath(".//li")[0]
        state = SimpleNamespace(block_path=(1, 0), slots=[])
        segments = [
            SimpleNamespace(kind="text", source="One", target="Uno", epub_state=state),
            SimpleNamespace(kind="text", source="Two", target="Dos", epub_state=state),
        ]
        for order in ("target_first", "source_first"):
            output = etree.fromstring(etree.tostring(source))
            output_block = output.xpath(".//li")[0]
            added = _add_bilingual_sources(
                output,
                segments,
                order=order,
                source_blocks={(1, 0): etree.fromstring(etree.tostring(block))},
                block_refs={(1, 0): output_block},
            )
            self.assertEqual(added, 1)
            source_nodes = output.xpath(".//*[contains(@class, 'tn-source')]")
            self.assertEqual(len(source_nodes), 1)
            self.assertEqual(source_nodes[0].tag.rsplit("}", 1)[-1], "div")
            style = etree.Element("style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            output.find("head").append(style)
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                output,
                segments,
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertNotIn("source_node_order", {item["code"] for item in failures})
            if order == "target_first":
                corrupt = etree.fromstring(etree.tostring(source))
                corrupt_block = corrupt.xpath(".//li")[0]
                _add_bilingual_sources(
                    corrupt,
                    segments,
                    order=order,
                    source_blocks={(1, 0): etree.fromstring(etree.tostring(block))},
                    block_refs={(1, 0): corrupt_block},
                )
                corrupt_source = corrupt_block[-1]
                corrupt_block.remove(corrupt_source)
                corrupt_block.insert(0, corrupt_source)
                corrupt_style = etree.Element("style", id="tn-bilingual-style")
                corrupt_style.text = _BILINGUAL_CSS
                corrupt.find("head").append(corrupt_style)
                corrupt_failures: list[dict[str, str]] = []
                _bilingual_proof(
                    source,
                    corrupt,
                    segments,
                    source_lang="en",
                    order=order,
                    resource="chapter.xhtml",
                    failures=corrupt_failures,
                )
                self.assertIn("source_node_order", {item["code"] for item in corrupt_failures})

    def test_direct_br_nested_inline_slots_pair_at_each_actual_owner(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><p>One <em>two</em> tail<br/>Next</p></body></html>"
        )
        slots = [
            SimpleNamespace(
                element_path=(),
                field="text",
                source_value="One ",
                target_value="Uno",
            ),
            SimpleNamespace(
                element_path=(0,),
                field="text",
                source_value="two",
                target_value="dos",
            ),
            SimpleNamespace(
                element_path=(0,),
                field="tail",
                source_value=" tail",
                target_value="cola",
            ),
            SimpleNamespace(
                element_path=(1,),
                field="tail",
                source_value="Next",
                target_value="Siguiente",
            ),
        ]
        state = SimpleNamespace(block_path=(1, 0), slots=slots)
        segment = SimpleNamespace(
            kind="text",
            source="One two tail Next",
            target="Uno dos cola Siguiente",
            epub_state=state,
        )
        for order in ("target_first", "source_first"):
            output = etree.fromstring(etree.tostring(source))
            output_block = output.xpath(".//p")[0]
            self.assertEqual(
                _add_bilingual_sources(
                    output,
                    [segment],
                    order=order,
                    source_blocks={(1, 0): etree.fromstring(etree.tostring(output_block))},
                    block_refs={(1, 0): output_block},
                ),
                4,
            )
            self.assertEqual(len(output.xpath(".//p//br")), 1)
            self.assertEqual(len(output.xpath(".//p//span[contains(@class, 'tn-source')]")), 4)
            style = etree.Element("style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            output.find("head").append(style)
            failures: list[dict[str, str]] = []
            count = _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertEqual(count, 4)
            self.assertEqual(failures, [])
        corrupt = etree.fromstring(etree.tostring(source))
        corrupt_block = corrupt.xpath(".//p")[0]
        _add_bilingual_sources(
            corrupt,
            [segment],
            order="target_first",
            source_blocks={(1, 0): etree.fromstring(etree.tostring(corrupt_block))},
            block_refs={(1, 0): corrupt_block},
        )
        first_target, first_source = corrupt_block[0], corrupt_block[1]
        corrupt_block.remove(first_target)
        corrupt_block.remove(first_source)
        corrupt_block.insert(0, first_source)
        corrupt_block.insert(1, first_target)
        corrupt_style = etree.Element("style", id="tn-bilingual-style")
        corrupt_style.text = _BILINGUAL_CSS
        corrupt.find("head").append(corrupt_style)
        corrupt_failures: list[dict[str, str]] = []
        _bilingual_proof(
            source,
            corrupt,
            [segment],
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=corrupt_failures,
        )
        self.assertIn("source_node_order", {item["code"] for item in corrupt_failures})

    def test_direct_br_whitespace_source_is_coalesced_and_paired(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><p>One<em>two</em> "
            b"<span>Next</span><br/>After</p></body></html>"
        )
        slots = [
            SimpleNamespace(element_path=(), field="text", source_value="One", target_value="Uno"),
            SimpleNamespace(
                element_path=(0,), field="text", source_value="two", target_value="dos"
            ),
            SimpleNamespace(element_path=(0,), field="tail", source_value=" ", target_value=""),
            SimpleNamespace(
                element_path=(1,), field="text", source_value="Next", target_value="Siguiente"
            ),
        ]
        segment = SimpleNamespace(
            kind="text",
            source="One two Next",
            target="Uno dos Siguiente",
            epub_state=SimpleNamespace(block_path=(1, 0), slots=slots),
        )
        for order in ("target_first", "source_first"):
            output = etree.fromstring(etree.tostring(source))
            output_block = output.xpath(".//p")[0]
            added = _add_bilingual_sources(
                output,
                [segment],
                order=order,
                source_blocks={(1, 0): etree.fromstring(etree.tostring(output_block))},
                block_refs={(1, 0): output_block},
            )
            self.assertEqual(added, 3)
            self.assertEqual(
                [node.text for node in output.xpath(".//p//span[contains(@class, 'tn-source')]")],
                ["One", "two ", "Next"],
            )
            self.assertEqual(
                len(output.xpath(".//p//span[@class='tn-bilingual-target']")),
                3,
            )
            failures: list[dict[str, str]] = []
            count = _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertEqual(count, 3)
            self.assertNotIn("source_node_empty", {item["code"] for item in failures})

    def test_direct_run_active_link_and_original_span_both_orders(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><p><a href='next.xhtml'>One</a>"
            b"<span>keep</span><br/>Two</p></body></html>"
        )
        block = source.xpath(".//p")[0]
        slots = [
            SimpleNamespace(
                element_path=(0,),
                field="text",
                source_value="One",
                target_value="Uno",
            ),
            SimpleNamespace(
                element_path=(2,),
                field="tail",
                source_value="Two",
                target_value="Dos",
            ),
        ]
        segment = SimpleNamespace(
            kind="text",
            source="One Two",
            target="Uno Dos",
            epub_state=SimpleNamespace(block_path=(1, 0), slots=slots),
        )
        for order in ("target_first", "source_first"):
            output = etree.fromstring(etree.tostring(source))
            output_block = output.xpath(".//p")[0]
            self.assertEqual(
                _add_bilingual_sources(
                    output,
                    [segment],
                    order=order,
                    source_blocks={(1, 0): etree.fromstring(etree.tostring(block))},
                    block_refs={(1, 0): output_block},
                ),
                2,
            )
            original_span = next(
                node for node in output_block.xpath("./span") if node.text == "keep"
            )
            self.assertEqual(dict(original_span.attrib), {})
            self.assertEqual(original_span.text, "keep")
            for node in output.xpath(
                ".//*[contains(concat(' ', normalize-space(@class), ' '), ' tn-source ')]"
            ):
                self.assertFalse(
                    any(
                        ancestor.tag.rsplit("}", 1)[-1] in {"a", "ruby"}
                        for ancestor in node.iterancestors()
                    )
                )
            self.assertEqual(
                len(output.xpath(f".//span[@class='{BILINGUAL_DIRECT_TARGET_CLASS}']")),
                2,
            )
            style = etree.Element("style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            output.find("head").append(style)
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertNotIn("source_target_pair_mismatch", {item["code"] for item in failures})

    def test_container_atomic_media_source_is_sanitized_both_orders(self) -> None:
        source = etree.fromstring(
            b"<html><head></head><body><li>Before<svg><text>DROP</text></svg>After</li></body></html>"
        )
        block = source.xpath(".//li")[0]
        segment = SimpleNamespace(
            kind="text",
            source="BeforeAfter",
            target="Translated",
            epub_state=SimpleNamespace(block_path=(1, 0), slots=[]),
        )
        for order in ("target_first", "source_first"):
            output = etree.fromstring(etree.tostring(source))
            output_block = output.xpath(".//li")[0]
            output_block.text = "Translated"
            self.assertEqual(
                _add_bilingual_sources(
                    output,
                    [segment],
                    order=order,
                    source_blocks={(1, 0): etree.fromstring(etree.tostring(block))},
                    block_refs={(1, 0): output_block},
                ),
                1,
            )
            source_node = output.xpath(
                ".//*[contains(concat(' ', normalize-space(@class), ' '), ' tn-source ')]"
            )[0]
            self.assertEqual("".join(source_node.itertext()), "BeforeAfter")
            self.assertFalse(source_node.xpath(".//*[local-name()='svg' or local-name()='text']"))
            style = etree.Element("style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            output.find("head").append(style)
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertNotIn("source_target_pair_mismatch", {item["code"] for item in failures})

    def test_direct_run_active_button_onclick_and_ruby_tail_both_orders(self) -> None:
        cases = (
            (
                "<html><head></head><body><p><button onclick='go()'>One</button>"
                "<br/>Two</p></body></html>",
                "en",
                [
                    SimpleNamespace(
                        element_path=(0,),
                        field="text",
                        source_value="One",
                        target_value="Uno",
                    ),
                    SimpleNamespace(
                        element_path=(1,),
                        field="tail",
                        source_value="Two",
                        target_value="Dos",
                    ),
                ],
            ),
            (
                "<html><head></head><body><p><span onclick='go()'>One</span>"
                "<br/>Two</p></body></html>",
                "en",
                [
                    SimpleNamespace(
                        element_path=(0,),
                        field="text",
                        source_value="One",
                        target_value="Uno",
                    ),
                    SimpleNamespace(
                        element_path=(1,),
                        field="tail",
                        source_value="Two",
                        target_value="Dos",
                    ),
                ],
            ),
            (
                "<html><head></head><body><p><ruby>漢<rt>かん</rt></ruby>"
                " tail<br/>Next</p></body></html>",
                "ja",
                [
                    SimpleNamespace(
                        element_path=(0,),
                        field="tail",
                        source_value=" tail",
                        target_value=" 尾",
                    ),
                    SimpleNamespace(
                        element_path=(1,),
                        field="tail",
                        source_value="Next",
                        target_value="下",
                    ),
                ],
            ),
        )
        for markup, source_lang, slots in cases:
            source = etree.fromstring(markup.encode())
            segment = SimpleNamespace(
                kind="text",
                source="source",
                target="target",
                epub_state=SimpleNamespace(block_path=(1, 0), slots=slots),
            )
            for order in ("target_first", "source_first"):
                output = etree.fromstring(markup.encode())
                block = output.xpath(".//p")[0]
                _add_bilingual_sources(
                    output,
                    [segment],
                    order=order,
                    source_blocks={(1, 0): etree.fromstring(markup.encode()).xpath(".//p")[0]},
                    block_refs={(1, 0): block},
                )
                source_nodes = output.xpath(
                    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' tn-source ')]"
                )
                self.assertEqual(len(source_nodes), len(slots))
                for node in source_nodes:
                    self.assertFalse(
                        any(
                            ancestor.tag.rsplit("}", 1)[-1] in {"a", "ruby", "button"}
                            or any(key.lower().startswith("on") for key in ancestor.attrib)
                            for ancestor in node.iterancestors()
                        )
                    )
                if source_lang == "ja":
                    source_tail = next(node for node in source_nodes if node.text == " tail")
                    target_tail = next(
                        node
                        for node in output.xpath(".//span")
                        if node.get("class") == BILINGUAL_DIRECT_TARGET_CLASS and node.text == " 尾"
                    )
                    ruby = output.xpath(".//ruby")[0]
                    if order == "target_first":
                        self.assertGreater(block.index(source_tail), block.index(target_tail))
                        self.assertGreater(block.index(source_tail), block.index(ruby))
                    else:
                        self.assertLess(block.index(source_tail), block.index(ruby))
                        self.assertLess(block.index(source_tail), block.index(target_tail))
                    self.assertFalse(source_tail.xpath(".//*[local-name()='ruby']"))
                    corrupt = etree.fromstring(etree.tostring(output))
                    corrupt_block = corrupt.xpath(".//p")[0]
                    corrupt_source = next(
                        node
                        for node in corrupt_block
                        if node.get("class") == BILINGUAL_SOURCE_CLASS and node.text == " tail"
                    )
                    corrupt_target = next(
                        node
                        for node in corrupt_block
                        if node.get("class") == BILINGUAL_DIRECT_TARGET_CLASS and node.text == " 尾"
                    )
                    corrupt_block.remove(corrupt_source)
                    corrupt_block.remove(corrupt_target)
                    ruby_index = corrupt_block.index(corrupt_block.xpath("./ruby")[0])
                    corrupt_block.insert(
                        ruby_index + 1,
                        corrupt_source if order == "target_first" else corrupt_target,
                    )
                    corrupt_block.insert(
                        ruby_index + 2,
                        corrupt_target if order == "target_first" else corrupt_source,
                    )
                    corrupt_style = etree.Element("style", id="tn-bilingual-style")
                    corrupt_style.text = _BILINGUAL_CSS
                    corrupt.find("head").append(corrupt_style)
                    corrupt_failures: list[dict[str, str]] = []
                    _bilingual_proof(
                        source,
                        corrupt,
                        [segment],
                        source_lang=source_lang,
                        order=order,
                        resource="chapter.xhtml",
                        failures=corrupt_failures,
                    )
                    self.assertIn("source_node_order", {item["code"] for item in corrupt_failures})
                style = etree.Element("style", id="tn-bilingual-style")
                style.text = _BILINGUAL_CSS
                output.find("head").append(style)
                failures: list[dict[str, str]] = []
                _bilingual_proof(
                    source,
                    output,
                    [segment],
                    source_lang=source_lang,
                    order=order,
                    resource="chapter.xhtml",
                    failures=failures,
                )
                self.assertNotIn("source_node_active_ancestor", {item["code"] for item in failures})
                self.assertNotIn("source_target_pair_mismatch", {item["code"] for item in failures})

            # Moving a verified source wrapper back under the active button is
            # deliberately rejected rather than being silently normalized.
            if "button" in markup:
                corrupt = etree.fromstring(markup.encode())
                corrupt_block = corrupt.xpath(".//p")[0]
                _add_bilingual_sources(
                    corrupt,
                    [segment],
                    order="target_first",
                    source_blocks={(1, 0): etree.fromstring(markup.encode()).xpath(".//p")[0]},
                    block_refs={(1, 0): corrupt_block},
                )
                source_node = corrupt.xpath(
                    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' tn-source ')]"
                )[0]
                corrupt_block.remove(source_node)
                corrupt.xpath(".//button")[0].append(source_node)
                style = etree.Element("style", id="tn-bilingual-style")
                style.text = _BILINGUAL_CSS
                corrupt.find("head").append(style)
                failures = []
                _bilingual_proof(
                    etree.fromstring(markup.encode()),
                    corrupt,
                    [segment],
                    source_lang="en",
                    order="target_first",
                    resource="chapter.xhtml",
                    failures=failures,
                )
                self.assertTrue(failures)

    def test_direct_run_sources_keep_two_slots_in_same_active_boundary_order(self) -> None:
        markup = b"<html><head></head><body><p><a href='x'><em>One</em><em>Two</em></a><br/>Tail</p></body></html>"
        source = etree.fromstring(markup)
        slots = [
            SimpleNamespace(
                element_path=(0, 0),
                field="text",
                source_value="One",
                target_value="Uno",
            ),
            SimpleNamespace(
                element_path=(0, 1),
                field="text",
                source_value="Two",
                target_value="Dos",
            ),
            SimpleNamespace(
                element_path=(1,),
                field="tail",
                source_value="Tail",
                target_value="尾",
            ),
        ]
        segment = SimpleNamespace(
            kind="text",
            source="One Two Tail",
            target="Uno Dos 尾",
            epub_state=SimpleNamespace(block_path=(1, 0), slots=slots),
        )
        for order in ("target_first", "source_first"):
            output = etree.fromstring(markup)
            output_block = output.xpath(".//p")[0]
            _add_bilingual_sources(
                output,
                [segment],
                order=order,
                source_blocks={(1, 0): etree.fromstring(markup).xpath(".//p")[0]},
                block_refs={(1, 0): output_block},
            )
            source_texts = [
                node.text
                for node in output.xpath(
                    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' tn-source ')]"
                )
            ]
            self.assertEqual(source_texts, ["One", "Two", "Tail"])
            style = etree.Element("style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            output.find("head").append(style)
            failures: list[dict[str, str]] = []
            _bilingual_proof(
                source,
                output,
                [segment],
                source_lang="en",
                order=order,
                resource="chapter.xhtml",
                failures=failures,
            )
            self.assertNotIn("source_node_order", {item["code"] for item in failures})

        corrupt = etree.fromstring(markup)
        corrupt_block = corrupt.xpath(".//p")[0]
        _add_bilingual_sources(
            corrupt,
            [segment],
            order="target_first",
            source_blocks={(1, 0): etree.fromstring(markup).xpath(".//p")[0]},
            block_refs={(1, 0): corrupt_block},
        )
        first = next(
            node
            for node in corrupt_block
            if node.get("class") == BILINGUAL_SOURCE_CLASS and node.text == "One"
        )
        second = next(
            node
            for node in corrupt_block
            if node.get("class") == BILINGUAL_SOURCE_CLASS and node.text == "Two"
        )
        corrupt_block.remove(first)
        corrupt_block.remove(second)
        anchor = corrupt_block.index(corrupt_block.xpath("./a")[0])
        corrupt_block.insert(anchor + 1, second)
        corrupt_block.insert(anchor + 2, first)
        style = etree.Element("style", id="tn-bilingual-style")
        style.text = _BILINGUAL_CSS
        corrupt.find("head").append(style)
        failures = []
        _bilingual_proof(
            source,
            corrupt,
            [segment],
            source_lang="en",
            order="target_first",
            resource="chapter.xhtml",
            failures=failures,
        )
        self.assertIn("source_node_order", {item["code"] for item in failures})

    def test_unknown_namespace_source_wrapper_preserves_visible_pairing_text(self) -> None:
        source = etree.fromstring(
            b"<html xmlns:u='urn:unknown'><head></head><body><p>Before"
            b"<u:wrapper>Visible<em>Inner</em></u:wrapper>After</p></body></html>"
        )
        segment = SimpleNamespace(
            kind="text",
            source="BeforeVisibleInnerAfter",
            target="Translated",
            epub_state=SimpleNamespace(block_path=(1, 0)),
        )
        output = etree.fromstring(
            (
                f"<html><head><style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style></head>"
                '<body><p>Translated</p><p class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">BeforeVisible<em>Inner</em>After</p>'
                "</body></html>"
            ).encode()
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
        self.assertNotIn("source_node_subtree_mismatch", {item["code"] for item in failures})

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

    def test_reserved_style_outside_head_is_rejected(self) -> None:
        source = etree.fromstring(b"<html><head></head><body><p>Original</p></body></html>")
        output = etree.fromstring(
            (
                '<html><head></head><body><p>Translated</p><p class="tn-source '
                'ibooks-dark-theme-use-custom-text-color">Original</p>'
                f"<style id='tn-bilingual-style'>{_BILINGUAL_CSS}</style></body></html>"
            ).encode()
        )
        segment = SimpleNamespace(
            kind="text",
            source="Original",
            target="Translated",
            epub_state=SimpleNamespace(block_path=(1, 0)),
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
        self.assertIn("bilingual_style_mismatch", {item["code"] for item in failures})

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

    def test_recovered_schema4_resource_reports_recovered_assurance(self) -> None:
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
                            "epub_schema": 4,
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

    def test_generated_mono_rejects_stray_tn_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generated.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}
            entries["OEBPS/text/chapter-1.xhtml"] = entries["OEBPS/text/chapter-1.xhtml"].replace(
                b"</body>", b'<p class="tn-source">stray</p></body>'
            )
            with zipfile.ZipFile(source, "w") as zout:
                for name, data in entries.items():
                    zout.writestr(
                        name,
                        data,
                        zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED,
                    )
            report = verify_epub(source, mode="generated", bilingual=False)
            self.assertIn("unexpected_source_nodes", {item["code"] for item in report["failures"]})

    def test_generated_bilingual_rejects_missing_source_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "generated.epub"
            write_phase9_epub(str(source))
            report = verify_epub(source, mode="generated", bilingual=True)
            self.assertTrue(report["failures"])
            self.assertIn(
                "missing_source_nodes",
                {item["code"] for item in report["failures"]},
            )

    def test_multi_base_ruby_direct_run_does_not_duplicate_source_values(self) -> None:
        source = etree.fromstring(
            "<html><head></head><body><p><ruby><rb>漢</rb><rt>かん</rt>"
            "<rb>字</rb><rt>じ</rt></ruby>です<br/>次</p></body></html>".encode()
        )
        slots = [
            SimpleNamespace(
                element_path=(0, 0),
                field="text",
                source_value="漢",
                target_value="Kan",
            ),
            SimpleNamespace(
                element_path=(0, 2),
                field="text",
                source_value="字",
                target_value="Ji",
            ),
            SimpleNamespace(
                element_path=(0,),
                field="tail",
                source_value="です",
                target_value="Desu",
            ),
            SimpleNamespace(
                element_path=(1,),
                field="tail",
                source_value="次",
                target_value="Next",
            ),
        ]
        segment = SimpleNamespace(
            kind="text",
            source="source",
            target="target",
            epub_state=SimpleNamespace(block_path=(1, 0), slots=slots),
        )
        template = etree.tostring(source)
        for order in ("target_first", "source_first"):
            output = etree.fromstring(template)
            block = output.xpath(".//p")[0]
            _add_bilingual_sources(
                output,
                [segment],
                order=order,
                source_blocks={(1, 0): etree.fromstring(template).xpath(".//p")[0]},
                block_refs={(1, 0): block},
            )
            source_nodes = output.xpath(".//*[contains(@class, 'tn-source')]")
            style = etree.SubElement(output.xpath(".//head")[0], "style", id="tn-bilingual-style")
            style.text = _BILINGUAL_CSS
            self.assertEqual(len(source_nodes), 3)
            values = ["".join(node.itertext()) for node in source_nodes]
            combined = "".join(values)
            self.assertIn("です", values)
            self.assertEqual(len(source.xpath(".//ruby")), 1)
            self.assertEqual(sum(len(node.xpath(".//ruby")) for node in source_nodes), 1)
            self.assertEqual(sum(len(node.xpath(".//rt")) for node in source_nodes), 2)
            self.assertEqual(combined.count("漢"), 1)
            self.assertEqual(combined.count("字"), 1)
            failures = []
            _bilingual_proof(
                etree.fromstring(template),
                output,
                [segment],
                source_lang="ja",
                order=order,
                resource="test.xhtml",
                failures=failures,
            )
            self.assertEqual(failures, [])

    def test_descriptor_flags_publish_and_secondary_opf_stays_identical(self) -> None:
        from tests.test_assemble import _run
        from trans_novel.assemble.writer import assemble

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.epub"
            write_phase9_epub(str(source))
            with zipfile.ZipFile(source) as zin:
                entries = {info.filename: zin.read(info.filename) for info in zin.infolist()}
            backup = (
                b'<package xmlns="http://www.idpf.org/2007/opf"><metadata '
                b'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:language>ja</dc:language>'
                b"</metadata></package>"
            )
            entries["META-INF/backup.opf"] = backup
            with zipfile.ZipFile(source, "w") as zout:
                for name, data in entries.items():
                    zout.writestr(
                        name,
                        data,
                        zipfile.ZIP_STORED if name == "mimetype" else zipfile.ZIP_DEFLATED,
                    )
            raw = bytearray(source.read_bytes())
            with zipfile.ZipFile(source) as zin:
                info = zin.getinfo("OEBPS/text/chapter-1.xhtml")
            descriptor = struct.unpack_from("<H", raw, info.header_offset + 6)[0] | 0x08
            struct.pack_into("<H", raw, info.header_offset + 6, descriptor)
            marker = b"OEBPS/text/chapter-1.xhtml"
            central_offset = raw.find(b"PK\x01\x02", info.header_offset + 30)
            while central_offset >= 0:
                name_length = struct.unpack_from("<H", raw, central_offset + 28)[0]
                if raw[central_offset + 46 : central_offset + 46 + name_length] == marker:
                    break
                central_offset = raw.find(b"PK\x01\x02", central_offset + 4)
            self.assertGreaterEqual(central_offset, 0)
            struct.pack_into("<H", raw, central_offset + 8, descriptor)
            store, _ = _run(str(source), str(root / "state"))
            output = root / "output.epub"
            assemble(store, str(source), str(output), out_format="epub")
            with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(output) as output_zip:
                self.assertEqual(
                    [info.flag_bits for info in source_zip.infolist()],
                    [info.flag_bits for info in output_zip.infolist()],
                )
                self.assertTrue(output_zip.read("OEBPS/text/chapter-1.xhtml"))
                self.assertEqual(output_zip.read("META-INF/backup.opf"), backup)

    def test_ncx_root_language_and_comment_pi_contents_are_immutable(self) -> None:
        from trans_novel.assemble.writer import _rewrite_toc_lxml

        data = (
            b'<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" xml:lang="ja">'
            b"<navMap><navPoint><navLabel><text>Old<!--keep-->"
            b' One<?keep pi?> tail</text></navLabel><content src="ch.xhtml"/>'
            b"</navPoint></navMap></ncx>"
        )
        result = _rewrite_toc_lxml(
            data,
            [
                {
                    "toc_path": "toc.ncx",
                    "node_index": 0,
                    "raw_href": "ch.xhtml",
                    "title_translated": "New",
                }
            ],
            is_ncx=True,
            toc_path="toc.ncx",
            target_lang="zh-Hans",
        )
        root = etree.fromstring(result)
        self.assertEqual(root.get("{http://www.w3.org/XML/1998/namespace}lang"), "ja")
        label = root.xpath("//*[local-name()='text']")[0]
        self.assertEqual(label.text, "New")
        self.assertEqual(label[0].text, "keep")
        self.assertEqual(label[1].text, "pi")

    def test_toc_label_outer_tail_is_preserved_and_inner_tail_corruption_fails(self) -> None:
        for label_name in ("a", "text"):
            source = etree.fromstring(
                f"<root><{label_name}>Old<!--keep--></{label_name}> OUT</root>".encode()
            )
            output = etree.fromstring(
                f"<root><{label_name}>New<!--keep--></{label_name}> OUT</root>".encode()
            )
            self.assertTrue(
                _compare_dom(
                    source,
                    output,
                    {((0,), "text"): {"kind": "toc", "expected": "New", "count": True}},
                    toc_label_paths={(0,)},
                )
            )
            output[0][0].tail = " stale"
            self.assertFalse(
                _compare_dom(
                    source,
                    output,
                    {((0,), "text"): {"kind": "toc", "expected": "New", "count": True}},
                    toc_label_paths={(0,)},
                )
            )


if __name__ == "__main__":
    unittest.main()
