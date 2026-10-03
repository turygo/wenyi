from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from lxml import etree

from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.source_css import (
    collect_source_stylesheets,
    source_semantic_styles,
    source_style_evidence,
)

_XHTML = "http://www.w3.org/1999/xhtml"


def _root(head: str, body: str = "") -> etree._Element:
    return etree.fromstring(
        f'<html xmlns="{_XHTML}"><head>{head}</head><body>{body}</body></html>'.encode()
    )


def _collect(
    members: dict[str, bytes],
    head: str,
    body: str = "",
    *,
    resource: str = "OPS/text/chapter.xhtml",
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "book.epub"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        with zipfile.ZipFile(path) as archive:
            result = collect_source_stylesheets(archive, _root(head, body), resource)
            after = {name: archive.read(name) for name in members}
    return result, after


class TestSourceStylesheetCollection(unittest.TestCase):
    def test_collects_in_document_and_depth_first_order_without_mutating_sources(self) -> None:
        base = b'@charset "windows-1252";@import "nested/one.css" screen;p#caf\xe9 { color: red }'
        one = b'@import "../../shared.css"; article#\xe9lan {}'
        members = {
            "OPS/css/base.css": base,
            "OPS/css/nested/one.css": one,
            "OPS/shared.css": b"#shared {}",
        }
        sheets, after = _collect(
            members,
            """
            <style type="application/json">ignored</style>
            <style>@import /* lead */ url( "../css/base.css" ) screen; #inline {}</style>
            <link rel="alternate StyleSheet" href="../css/base.css"/>
            <style type=" ">#second {}</style>
            <link rel="stylesheet" type="text/plain" href="https://example.invalid/x.css"/>
            """,
            body="<section><style>#descendant {}</style></section>",
        )

        self.assertEqual(
            list(sheets),
            [
                "file:OPS/shared.css",
                "file:OPS/css/nested/one.css",
                "file:OPS/css/base.css",
                "inline:1",
                "file:OPS/shared.css#occurrence:2",
                "file:OPS/css/nested/one.css#occurrence:2",
                "file:OPS/css/base.css#occurrence:2",
                "inline:2",
                "inline:3",
            ],
        )
        self.assertIn("élan", sheets["file:OPS/css/nested/one.css"].decode())
        self.assertIn("café", sheets["file:OPS/css/base.css"].decode())
        self.assertTrue(sheets["file:OPS/shared.css"].startswith(b"@media screen{@media screen{"))
        self.assertTrue(sheets["file:OPS/shared.css#occurrence:2"].startswith(b"@media screen{"))
        self.assertTrue(all(b"@charset" not in css.lower() for css in sheets.values()))
        self.assertTrue(all(b"@import" not in css.lower() for css in sheets.values()))
        self.assertEqual(after, members)

    def test_accepts_string_url_token_and_quoted_url_function_imports(self) -> None:
        members = {
            "OPS/text/a.css": b"#a {}",
            "OPS/text/b.css": b"#b {}",
            "OPS/text/c.css": b"#c {}",
        }
        sheets, _after = _collect(
            members,
            """
            <style>
              @import /* comment */ "a.css";
              @import URL(b.css) print;
              @import url( "c.css" ) supports(display: grid);
              #owner {}
            </style>
            """,
        )
        self.assertEqual(
            list(sheets),
            [
                "file:OPS/text/a.css",
                "file:OPS/text/b.css",
                "file:OPS/text/c.css",
                "inline:0",
            ],
        )
        self.assertEqual(sheets["file:OPS/text/a.css"], b"#a {}")
        self.assertEqual(sheets["file:OPS/text/b.css"], b"@media print{#b {}}")
        self.assertEqual(sheets["file:OPS/text/c.css"], b"@supports (display: grid){#c {}}")

    def test_preserves_each_import_condition_and_source_order(self) -> None:
        sheets, after = _collect(
            {"OPS/text/a.css": b"p{font-style:italic}"},
            '<style>@import "a.css" print;@import "a.css" screen;p{font-weight:500}</style>'
            '<link rel="stylesheet" href="a.css"/>',
        )
        self.assertEqual(
            list(sheets),
            [
                "file:OPS/text/a.css",
                "file:OPS/text/a.css#occurrence:2",
                "inline:0",
                "file:OPS/text/a.css#occurrence:3",
            ],
        )
        self.assertEqual(
            list(sheets.values()),
            [
                b"@media print{p{font-style:italic}}",
                b"@media screen{p{font-style:italic}}",
                b"p{font-weight:500}",
                b"p{font-style:italic}",
            ],
        )
        self.assertEqual(after["OPS/text/a.css"], b"p{font-style:italic}")

    def test_nested_imports_keep_supports_and_media_context(self) -> None:
        sheets, _ = _collect(
            {
                "OPS/text/a.css": b'@import "b.css" supports(selector(p)) print;p{}',
                "OPS/text/b.css": b"p{font-style:italic}",
            },
            '<style media="screen">@import "a.css" supports(display: grid);</style>',
        )
        self.assertEqual(
            sheets["file:OPS/text/b.css"],
            b"@media screen{@supports (display: grid){@supports selector(p){@media print{p{font-style:italic}}}}}",
        )

    def test_style_and_link_media_attributes_are_wrapped(self) -> None:
        sheets, _ = _collect(
            {"OPS/text/a.css": b"p{font-style:italic}"},
            '<link rel="stylesheet" href="a.css" media="print"/>'
            '<style media="screen and (min-width: 1px)">p{font-weight:500}</style>'
            '<style media="all">p{font-style:normal}</style>',
        )
        self.assertEqual(sheets["file:OPS/text/a.css"], b"@media print{p{font-style:italic}}")
        self.assertEqual(
            sheets["inline:0"], b"@media screen and (min-width: 1px){p{font-weight:500}}"
        )
        self.assertEqual(sheets["inline:1"], b"p{font-style:normal}")

    def test_conditional_import_cannot_be_assumed_to_apply_to_semantics(self) -> None:
        root = _root("", "<p>Text</p>")
        paragraph = root.xpath("//*[local-name()='p']")[0]
        for condition in ("print", "supports(display: grid)"):
            with self.subTest(condition=condition):
                sheets, _ = _collect(
                    {"OPS/text/a.css": b"p{font-style:italic}"},
                    f'<style>@import "a.css" {condition};</style>',
                )
                with self.assertRaisesRegex(ThemeError, "^ambiguous_source_semantics$"):
                    source_semantic_styles(
                        root, (paragraph,), sheets, resource="OPS/text/chapter.xhtml"
                    )

    def test_nested_import_is_left_for_the_source_bound_validator(self) -> None:
        sheets, _after = _collect({}, "<style>@media screen { @import 'nested.css'; }</style>")
        self.assertIn(b"@import", sheets["inline:0"])

    def test_rejects_cycle_external_missing_unsafe_and_invalid_imports(self) -> None:
        cases = (
            (
                "cycle",
                '<link rel="stylesheet" href="../css/a.css"/>',
                {
                    "OPS/css/a.css": b'@import "b.css";',
                    "OPS/css/b.css": b'@import "a.css";',
                },
            ),
            ("external", "<style>@import 'https://example.invalid/x.css';</style>", {}),
            ("external_link", '<link rel="stylesheet" href="//example.invalid/x.css"/>', {}),
            ("missing", "<style>@import '../css/missing.css';</style>", {}),
            ("missing_href", '<link rel="stylesheet"/>', {}),
            ("unsafe", "<style>@import '../../../outside.css';</style>", {}),
            ("fragment", "<style>@import 'a.css#x';</style>", {}),
            ("query", "<style>@import 'a.css?v=1';</style>", {}),
            ("layer", "<style>@import 'a.css' layer;</style>", {"OPS/text/a.css": b""}),
            (
                "layer_function",
                "<style>@import 'a.css' layer(base);</style>",
                {"OPS/text/a.css": b""},
            ),
            ("malformed", "<style>@import url(a.css;</style>", {}),
            ("invalid_form", "<style>@import url(a.css b.css);</style>", {}),
            ("malformed_sheet", "<style>p</style>", {}),
        )
        for label, head, members in cases:
            detail = {
                "cycle": "source_import_cycle",
                "layer": "source_layers",
                "layer_function": "source_layers",
            }.get(label, "source_import_failed")
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(ThemeError, f"^{detail}$") as raised,
            ):
                _collect(members, head)
            self.assertEqual(raised.exception.code, "theme_css")
            self.assertEqual(raised.exception.resource, "OPS/text/chapter.xhtml")
            self.assertIsNone(raised.exception.__cause__)

    def test_rejects_base_urls_that_change_reader_stylesheet_resolution(self) -> None:
        members = {
            "OPS/text/a.css": b"",
            "OPS/css/a.css": b"#x#x { color:red !important }",
        }
        for base in (
            '<base href="../css/"/>',
            '<base href="https://example.invalid/"/>',
            '<link xml:base="../css/" rel="stylesheet" href="a.css"/>',
        ):
            with (
                self.subTest(base=base),
                self.assertRaisesRegex(ThemeError, "^source_base_unsupported$"),
            ):
                _collect(members, base + '<link rel="stylesheet" href="a.css"/>')

    def test_rejects_invalid_resource_path_even_without_stylesheets(self) -> None:
        with self.assertRaisesRegex(ThemeError, "^source_import_failed$"):
            _collect({}, "", resource="../chapter.xhtml")


class TestSourceStyleEvidence(unittest.TestCase):
    def test_layout_preserves_rule_text_declaration_order_and_conditions(self) -> None:
        root = _root("", '<p style="color: navy;">Text</p>')
        paragraph = root.xpath("//*[local-name()='p']")[0]
        sheets = {"book": b"@media print { p { color: red;\ncolor: blue !important; } }"}
        self.assertEqual(
            source_style_evidence(
                root, (paragraph,), sheets, resource="chapter", preserve_rule_text=True
            ),
            (("@media print{p{color: red;\ncolor: blue !important;}}", "@inline p{color: navy;}"),),
        )
        self.assertEqual(
            source_style_evidence(root, (paragraph,), sheets, resource="chapter"),
            (("@media print{p{color:red;color:blue !important}}", "@inline p{color: navy;}"),),
        )

    def test_layout_preserves_one_ordinary_selector_list_with_matched_union(self) -> None:
        root = _root("", '<section><p class="quote">Text</p></section><div>Other</div>')
        nodes = root.xpath("//*[local-name()='p' or local-name()='div']")
        sheets = {"book": b"section, p.quote, div { color: red; }"}
        self.assertEqual(
            source_style_evidence(root, nodes, sheets, resource="chapter", preserve_rule_text=True),
            (("section, p.quote, div{color: red;}",), ("section, p.quote, div{color: red;}",)),
        )
        self.assertEqual(
            source_style_evidence(root, nodes, sheets, resource="chapter"),
            (("section{color:red}", "p.quote{color:red}"), ("div{color:red}",)),
        )

    def test_layout_mixed_pseudo_rule_keeps_only_ordinary_branch(self) -> None:
        root = _root("", "<p>Text</p>")
        paragraph = root.xpath("//*[local-name()='p']")[0]
        sheets = {"book": b"p, p::first-letter { font-size: 120%; }"}
        self.assertEqual(
            source_style_evidence(
                root, (paragraph,), sheets, resource="chapter", preserve_rule_text=True
            ),
            (("p{font-size: 120%;}",),),
        )
        self.assertEqual(
            source_style_evidence(root, (paragraph,), sheets, resource="chapter"),
            (("p{font-size:120%}",),),
        )

    def test_layout_excludes_nonmatching_dynamic_branch_from_selector_list(self) -> None:
        root = _root("", "<p>Text</p>")
        paragraph = root.xpath("//*[local-name()='p']")[0]
        sheets = {"book": b"p, .missing:hover { font-weight: bold; }"}
        for preserve in (False, True):
            with self.subTest(preserve_rule_text=preserve):
                self.assertEqual(
                    source_style_evidence(
                        root, (paragraph,), sheets, resource="chapter", preserve_rule_text=preserve
                    ),
                    (("p{font-weight: bold;}" if preserve else "p{font-weight:bold}",),),
                )

    def test_matches_node_and_ancestors_with_media_and_inline_context(self) -> None:
        root = _root(
            "",
            '<section class="chapter" style="writing-mode: vertical-rl">'
            '<p class="quote" style="color: navy">Text</p></section>',
        )
        paragraph = root.xpath("//*[local-name()='p']")[0]
        evidence = source_style_evidence(
            root,
            (paragraph,),
            {
                "book": (
                    b".chapter { margin: 1em }"
                    b"@media print { .chapter > p { color: red } }"
                    b".unrelated { display: none }"
                )
            },
            resource="OPS/text/chapter.xhtml",
        )[0]

        self.assertEqual(len(evidence), 4)
        self.assertTrue(any(item.startswith(".chapter{") for item in evidence))
        self.assertTrue(any(item.startswith("@media print{") for item in evidence))
        self.assertIn(
            "@inline section{writing-mode: vertical-rl}",
            evidence,
        )
        self.assertIn("@inline p{color: navy}", evidence)
        self.assertFalse(any(".unrelated" in item for item in evidence))

    def test_reuses_strict_source_selector_and_inline_validation(self) -> None:
        root = _root("", '<p style="color: red; broken">Text</p>')
        paragraph = root.xpath("//*[local-name()='p']")[0]
        with self.assertRaisesRegex(ThemeError, "^unsupported_source_selector$"):
            source_style_evidence(
                root,
                (paragraph,),
                {"book": b"p||span { color: red }"},
                resource="OPS/text/chapter.xhtml",
            )
        with self.assertRaisesRegex(ThemeError, "^invalid_inline_css$"):
            source_style_evidence(
                root,
                (paragraph,),
                {},
                resource="OPS/text/chapter.xhtml",
            )
