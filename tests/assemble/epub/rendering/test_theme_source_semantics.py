"""源文字语义、重置节点与中文首字装饰的确定性证明。"""

import unittest

from bs4 import BeautifulSoup
from lxml import etree

from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.source_css import (
    chinese_dropcap_overrides,
    source_semantic_styles,
    source_style_evidence,
    source_style_requirements,
)
from trans_novel.assemble.epub.rendering.theme.source_selectors import source_node_map
from trans_novel.epub.markup import resource_parser


def _fixture(body, css):
    root = etree.fromstring(
        (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head/><body>' + body + "</body></html>"
        ).encode()
    )
    nodes = root.xpath("//*[@id]")
    return root, nodes, {"book": css.encode()}


class TestSourceSemanticStyles(unittest.TestCase):
    def test_dtd_serialization_does_not_fabricate_a_node_or_change_source_context(self):
        data = (
            b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
            b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
            b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title></head>'
            b'<body><p><i id="scope">Text</i></p></body></html>'
        )
        tree, mode, _ = resource_parser(data)
        self.assertEqual(mode, "xml")
        root = tree.getroot()
        original_nodes = list(root.iter())
        doctype = tree.docinfo.doctype
        soup, reverse = source_node_map(root, resource="c")
        self.assertEqual(len(soup.find_all(True)), len(original_nodes))
        self.assertEqual(set(reverse.values()), set(original_nodes))
        self.assertEqual(list(root.iter()), original_nodes)
        self.assertEqual(tree.docinfo.doctype, doctype)
        self.assertEqual(
            source_semantic_styles(
                root,
                root.xpath("//*[@id]"),
                {},
                resource="c",
            ),
            (("italic",),),
        )

    def test_dynamic_styles_without_possible_nodes_are_skipped_without_guessing_state(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a">text</span></p>',
            ".absent a,.absent a:visited{font-weight:bold}.absent:hover > span{font-style:italic}",
        )
        self.assertEqual(source_style_evidence(root, nodes, sheets, resource="c"), ((),))
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), ((),))
        for selector in ("span:hover", "span:visited", "span:not(:visited)", "span:future"):
            with self.subTest(selector=selector):
                sheets["book"] = f"{selector}{{font-style:italic}}".encode()
                with self.assertRaisesRegex(ThemeError, "unsupported_source_semantic_selector"):
                    source_semantic_styles(root, nodes, sheets, resource="c")

    def test_weight_roles_compare_to_parent_and_inherited_weight_is_not_extra_emphasis(self):
        root, nodes, sheets = _fixture(
            '<p><span id="heavy"><span id="inherited">text</span></span>'
            '<span id="medium">medium</span><span id="reset">normal</span></p>',
            "#heavy{font-weight:700}#medium{font-weight:500}#reset{font-weight:normal}",
        )
        self.assertEqual(
            source_semantic_styles(root, nodes, sheets, resource="c"),
            (("bold",), (), ("bold",), ()),
        )
        self.assertEqual(
            source_style_requirements(root, nodes, sheets, resource="c"), (True, False, True, False)
        )

    def test_specificity_importance_order_inline_and_normal_reset(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a" class="x" style="font-style:normal">a</span>'
            '<span id="b" class="x">b</span><span id="c" class="x">c</span></p>',
            ".x{font-style:italic!important}#b{font-style:normal!important}"
            "#c{font-style:italic!important}#c{font-style:normal!important}",
        )
        self.assertEqual(
            source_semantic_styles(root, nodes, sheets, resource="c"), (("italic",), (), ())
        )
        root, nodes, sheets = _fixture(
            '<p><span id="a" class="x">text</span></p>',
            ":where(#a){font-style:normal}.x{font-style:italic}",
        )
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), (("italic",),))

    def test_native_tag_styles_and_resets_require_the_actual_wrapper(self):
        root, nodes, sheets = _fixture(
            '<p><i id="i">italic<span id="roman">normal</span></i>'
            '<b id="b">bold<span id="light">normal</span></b>'
            '<sub id="sub">90</sub><sup id="sup">2</sup></p>',
            "#roman{font-style:normal}#light{font-weight:normal}",
        )
        self.assertEqual(
            source_semantic_styles(root, nodes, sheets, resource="c"),
            (("italic",), (), ("bold",), (), ("subscript",), ("superscript",)),
        )
        self.assertEqual(source_style_requirements(root, nodes, sheets, resource="c"), (True,) * 6)

    def test_explicit_vertical_reset_and_global_inherit_initial_unset(self):
        root, nodes, sheets = _fixture(
            '<p style="font-weight:700;font-style:italic"><span id="a">a</span>'
            '<span id="b">b</span><span id="c">c</span><span id="d">d</span></p>',
            "#a{font:inherit}#b{all:initial}#c{font:unset}#d{vertical-align:baseline}",
        )
        self.assertEqual(
            source_semantic_styles(root, nodes, sheets, resource="c"),
            (("italic",), (), ("italic",), ("italic",)),
        )
        self.assertEqual(
            source_style_requirements(root, nodes, sheets, resource="c"), (False, True, False, True)
        )

    def test_condition_that_can_change_semantics_fails_but_overridden_rule_does_not(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a">a</span></p>', "@media print{#a{font-style:italic}}"
        )
        with self.assertRaisesRegex(ThemeError, "^ambiguous_source_semantics$") as raised:
            source_semantic_styles(root, nodes, sheets, resource="c")
        self.assertEqual(raised.exception.resource, "c")
        sheets["book"] += b"#a{font-style:normal!important}"
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), ((),))

    def test_unknown_semantic_values_fail_only_when_relevant_and_not_overridden(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a">a</span><span class="other">b</span></p>',
            ".other{font:var(--font)}#a{font-weight:400}",
        )
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), ((),))
        sheets["book"] = b"#a{font:italic 12px serif}"
        with self.assertRaisesRegex(ThemeError, "^ambiguous_source_semantics$"):
            source_semantic_styles(root, nodes, sheets, resource="c")
        sheets["book"] += b"#a{font-style:normal!important;font-weight:400!important}"
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), ((),))


class TestSourcePseudoElements(unittest.TestCase):
    def test_mixed_selector_list_keeps_ordinary_branch_and_never_styles_body_as_first_letter(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a" class="drop">A</span>bc</p>',
            "span.drop,p::first-letter{font-weight:bold;font-size:2em}",
        )
        evidence = source_style_evidence(root, nodes, sheets, resource="c")[0]
        self.assertEqual(len(evidence), 1)
        self.assertTrue(evidence[0].startswith("span.drop{"))
        self.assertNotIn("first-letter", evidence[0])
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), (("bold",),))

    def test_pseudo_only_and_unrelated_pseudo_rule_do_not_style_normal_nodes(self):
        root, nodes, sheets = _fixture(
            '<p id="a">text</p>',
            "p:first-letter{font-weight:bold}.unused::first-line{font-style:italic}",
        )
        self.assertEqual(source_style_evidence(root, nodes, sheets, resource="c"), ((),))
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), ((),))

    def test_top_level_commas_respect_function_and_attribute_tokens(self):
        root, nodes, sheets = _fixture(
            '<p><span id="a" data-value="a,b">text</span></p>',
            ':is(span,em)[data-value="a,b"],p::first-letter{font-style:italic}',
        )
        self.assertEqual(source_semantic_styles(root, nodes, sheets, resource="c"), (("italic",),))

    def test_unsupported_ordinary_selector_skips_layout_but_rejects_necessary_semantics(self):
        root, nodes, sheets = _fixture('<p id="a">text</p>', "p:future{margin:1em}")
        self.assertEqual(source_style_evidence(root, nodes, sheets, resource="c"), ((),))
        sheets["book"] = b"p:future{font-style:italic}"
        with self.assertRaisesRegex(ThemeError, "^unsupported_source_semantic_selector$"):
            source_semantic_styles(root, nodes, sheets, resource="c")

    def test_dropcap_override_is_chinese_only_preserves_conditions_and_never_cancels_first_line(
        self,
    ):
        css = {
            "book": b"@media print{p:first-letter{font-size:150%}}p::first-line{font-size:3em}.small::first-letter{font-size:1em}.float::first-letter{float:left}"
        }
        root = etree.fromstring(
            b'<html><body><p>target</p><div class="small">small</div><div class="float">float</div></body></html>'
        )
        output = chinese_dropcap_overrides(css, resource="c", root=root)
        self.assertIn(
            "@media print{p:lang(zh):not(.tn-source):not(.tn-source *)::first-letter{", output
        )
        self.assertIn(".float:lang(zh):not(.tn-source):not(.tn-source *)::first-letter{", output)
        self.assertNotIn("first-line", output)
        self.assertNotIn(".small", output)
        self.assertIn("font-weight:inherit!important", output)
        self.assertIn("initial-letter:normal!important", output)
        self.assertIn("all:unset!important", output)

    def test_dropcap_override_matches_only_chinese_target_and_excludes_source_descendants(self):
        output = chinese_dropcap_overrides(
            {"book": b"p::first-letter{font-size:2em}"}, resource="c"
        )
        base = output.split("::first-letter", 1)[0]
        soup = BeautifulSoup(
            '<html lang="zh"><body><p id="target">中文</p>'
            '<p id="source" class="tn-source">Source</p>'
            '<div class="tn-source"><p id="descendant">Source</p></div>'
            '<p id="english" lang="en">English</p></body></html>',
            "html.parser",
        )
        self.assertEqual([node["id"] for node in soup.select(base)], ["target"])

    def test_dropcap_proof_respects_property_resets_and_important(self):
        css = {
            "book": b"a::first-letter{font-size:2em;font:inherit}b::first-letter{font-size:2em!important;font:inherit}c::first-letter{float:left;all:initial}d::first-letter{initial-letter:2}"
        }
        output = chinese_dropcap_overrides(css, resource="c")
        self.assertNotIn("a:lang", output)
        self.assertNotIn("c:lang", output)
        self.assertIn("b:lang", output)
        self.assertIn("d:lang", output)

    def test_more_specific_rule_that_cancels_decoration_preserves_its_semantics_by_rejecting_override(
        self,
    ):
        css = {
            "book": b"p::first-letter{font-size:2em}p.note::first-letter{font-size:1em;font-style:italic}"
        }
        root = etree.fromstring(b'<html><body><p class="note">text</p></body></html>')
        for source_root in (None, root):
            with (
                self.subTest(root=source_root is not None),
                self.assertRaisesRegex(ThemeError, "^ambiguous_source_decoration$") as raised,
            ):
                chinese_dropcap_overrides(css, resource="c", root=source_root)
            self.assertEqual(raised.exception.resource, "c")

    def test_multiple_rules_that_keep_the_same_base_decorative_are_proven(self):
        css = {"book": b"p::first-letter{font-size:2em}p::first-letter{font-size:3em}"}
        output = chinese_dropcap_overrides(css, resource="c")
        self.assertEqual(output.count("::first-letter"), 1)

    def test_original_root_proves_nonoverlapping_normal_rule_is_irrelevant(self):
        css = {
            "book": b".drop::first-letter{font-size:2em}.normal::first-letter{font-size:1em;font-style:italic}"
        }
        root = etree.fromstring(
            b'<html><body><p class="drop">decorative</p><p class="normal">normal</p></body></html>'
        )
        output = chinese_dropcap_overrides(css, resource="c", root=root)
        self.assertIn(".drop:lang", output)
        self.assertNotIn(".normal:lang", output)
        with self.assertRaisesRegex(ThemeError, "^ambiguous_source_decoration$"):
            chinese_dropcap_overrides(css, resource="c")

    def test_conditional_rule_that_may_cancel_decoration_is_not_guessed_inactive(self):
        css = {
            "book": b"p::first-letter{font-size:2em}@media print{p::first-letter{font-size:1em}}"
        }
        root = etree.fromstring(b"<html><body><p>text</p></body></html>")
        with self.assertRaisesRegex(ThemeError, "^ambiguous_source_decoration$"):
            chinese_dropcap_overrides(css, resource="c", root=root)
