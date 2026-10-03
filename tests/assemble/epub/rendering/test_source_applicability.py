"""证明无效动画不会阻塞源读取，也不会因生成节点而误放行主题。"""

import unittest

from lxml import etree

from trans_novel.assemble.epub.rendering.theme.cascade import source_specificity_bound
from trans_novel.assemble.epub.rendering.theme.contracts import ThemeError
from trans_novel.assemble.epub.rendering.theme.source_applicability import (
    project_nonmatching_animations,
)
from trans_novel.assemble.epub.rendering.theme.source_css import source_style_evidence
from trans_novel.assemble.epub.rendering.theme.source_semantics import source_semantic_styles


class TestSourceApplicability(unittest.TestCase):
    def setUp(self):
        self.root = etree.fromstring(b'<html><head/><body><p id="p">Text</p></body></html>')
        self.node = self.root.find(".//p")

    def test_unmatched_animation_projection_preserves_original_css_and_other_declarations(self):
        original = (
            b"@media screen{.unused a:hover img,.unused a:focus img{"
            b"-webkit-animation:scale .2s;animation:scale .2s;font-weight:bold}}"
            b"@keyframes scale{0%,100%{width:100%}50%{width:85%}}"
            b"p{font-style:italic}"
        )
        sheets = {"book": original}
        projected = project_nonmatching_animations(sheets, self.root, resource="c")
        self.assertNotIn(b"animation:", projected["book"])
        self.assertIn(b"font-weight:bold", projected["book"])
        self.assertIn(b"@keyframes", projected["book"])
        self.assertEqual(sheets["book"], original)
        self.assertEqual(source_specificity_bound(sheets, resource="c", root=self.root), 1)
        with self.assertRaisesRegex(ThemeError, "source_animation"):
            source_specificity_bound(sheets, resource="c")
        self.assertEqual(
            source_semantic_styles(self.root, [self.node], sheets, resource="c"), (("italic",),)
        )
        self.assertEqual(
            source_style_evidence(self.root, [self.node], sheets, resource="c"),
            (("p{font-style:italic}",),),
        )

    def test_any_possible_branch_or_unknown_state_retains_strict_animation_rejection(self):
        for selector in ("p", ".absent,p:hover", "p:not(:hover)", "p:future"):
            with self.subTest(selector=selector):
                sheets = {"book": f"{selector}{{animation:fade 1s}}".encode()}
                with self.assertRaisesRegex(ThemeError, "source_animation"):
                    source_semantic_styles(self.root, [self.node], sheets, resource="c")
                with self.assertRaisesRegex(ThemeError, "source_animation"):
                    source_specificity_bound(sheets, resource="c", root=self.root)

    def test_theme_does_not_assume_future_node_types_or_reserved_identities_remain_absent(self):
        for selector in ("span:hover", ".tn-note", '[data-tn-role="body"]', ".tn-source"):
            with self.subTest(selector=selector):
                sheets = {"book": f"{selector}{{animation:fade 1s}}".encode()}
                self.assertEqual(
                    source_semantic_styles(self.root, [self.node], sheets, resource="c"), ((),)
                )
                with self.assertRaisesRegex(ThemeError, "source_animation"):
                    source_specificity_bound(sheets, resource="c", root=self.root)

    def test_negative_identity_does_not_prove_future_safe_absence_and_inline_animation_fails(self):
        sheets = {"book": b"span:not(.absent){animation:fade 1s}"}
        with self.assertRaisesRegex(ThemeError, "source_animation"):
            source_specificity_bound(sheets, resource="c", root=self.root)
        self.node.set("style", "animation:fade 1s")
        with self.assertRaisesRegex(ThemeError, "source_animation"):
            source_semantic_styles(self.root, [self.node], {}, resource="c")
