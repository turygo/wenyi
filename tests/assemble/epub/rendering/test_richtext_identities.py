"""空身份记录与实际内联内容共享同一个源节点身份。"""

from __future__ import annotations

import unittest
from copy import deepcopy

from lxml import etree

from tests.assemble.epub.rendering.test_richtext import _fixture
from trans_novel.assemble.epub.rendering.richtext import render_rich_block
from trans_novel.assemble.epub.verification.richtext import prove_rich_target
from trans_novel.assemble.epub.verification.structure import check_links, html_soup
from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.epub.slots import normalized_source_text, slot_contract_digest
from trans_novel.ingest.epub.richtext import extract_rich_sources


def _graph(block):
    soup, _ = html_soup(etree.tostring(block), "application/xhtml+xml")
    return check_links(
        soup, "c.xhtml", {"c.xhtml"}, {"c.xhtml": {"n"}}, [], [], {"internal_links": 0}
    )


class TestRichTextIdentities(unittest.TestCase):
    def test_empty_link_identity_with_different_nesting_uses_one_link(self):
        for empty_first in (True, False):
            with self.subTest(empty_first=empty_first):
                _, _, block, segments = _fixture(
                    '<p>Before<i><sup><a id="ref" name="ref-name" href="#n">5</a></sup></i>'
                    "After</p>"
                )
                source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
                inventory = segments[0].epub_state.rich_source
                italic, superscript, link = inventory.marks
                original_anchor = block.find(".//{*}a")
                empty = InlineRun(marks=(link.id,))
                label = InlineRun(text="5", marks=(italic.id, superscript.id, link.id))
                middle = [empty, label] if empty_first else [label, empty]
                segments[0].assign_translation(
                    RichTarget(runs=[InlineRun(text="正文"), *middle, InlineRun(text="。")])
                )
                replacements = render_rich_block(block, segments)
                anchors = block.findall(".//{*}a")
                self.assertEqual(len(anchors), 1)
                self.assertEqual(anchors[0].attrib, link.attributes)
                self.assertEqual(anchors[0].text, "5")
                self.assertEqual(anchors[0].getparent().tag.rsplit("}", 1)[-1], "sup")
                self.assertIs(replacements[original_anchor], anchors[0])
                self.assertEqual(_graph(source), _graph(block))
                prove_rich_target(source, block, segments)

    def test_independent_empty_identity_and_original_anchor_are_retained(self):
        _, _, block, segments = _fixture(
            '<p><span id="independent">Original</span><a id="untouched"/>'
            '<a id="ref" href="#n">5</a></p>'
        )
        source = deepcopy(block.getroottree().getroot()).find(".//{*}p")
        inventory = segments[0].epub_state.rich_source
        span, link = inventory.marks
        anchor = inventory.atoms[0]
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(marks=(span.id,)),
                    InlineRun(text="正文"),
                    InlineRun(atom=anchor.id),
                    InlineRun(marks=(link.id,)),
                    InlineRun(text="5", marks=(link.id,)),
                ]
            )
        )
        render_rich_block(block, segments)
        standalone = block.find(".//{*}span")
        self.assertEqual(standalone.get("id"), "independent")
        self.assertEqual(standalone.text or "", "")
        self.assertEqual(len(block.xpath('.//*[@id="untouched"]')), 1)
        self.assertEqual(len(block.xpath('.//*[@id="ref"]')), 1)
        self.assertEqual(_graph(source), _graph(block))
        prove_rich_target(source, block, segments)

    def test_meaningful_identity_is_found_across_legacy_br_segments(self):
        _, root, block, original_segments = _fixture('<p><span id="shared">One<br/>Two</span></p>')
        source = deepcopy(root).find(".//{*}p")
        original_state = original_segments[0].epub_state
        states = []
        for slot in original_state.slots:
            state = original_state.model_copy(deep=True)
            state.slots = [slot]
            state.slot_contract_sha256 = slot_contract_digest(state.slots)
            states.append(state)
        segments = []
        for state, rich in zip(states, extract_rich_sources(root, block, states), strict=True):
            state.rich_source = rich
            segment = original_segments[0].model_copy(deep=True)
            segment.epub_state = state
            segment.source = normalized_source_text(state.slots)
            segments.append(segment)
        mark = segments[0].epub_state.rich_source.marks[0]
        atom = segments[1].epub_state.rich_source.atoms[0]
        segments[0].assign_translation(
            RichTarget(runs=[InlineRun(marks=(mark.id,)), InlineRun(text="一")])
        )
        segments[1].assign_translation(
            RichTarget(runs=[InlineRun(atom=atom.id), InlineRun(text="二", marks=(mark.id,))])
        )
        render_rich_block(block, segments)
        spans = block.findall(".//{*}span")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].get("id"), "shared")
        self.assertEqual(spans[0].text, "二")
        self.assertEqual(block.text, "一")
        self.assertEqual(len(block.findall(".//{*}br")), 1)
        prove_rich_target(source, block, segments)

    def test_atom_scope_is_meaningful_and_keeps_identity(self):
        _, root, block, segments = _fixture(
            '<p><span id="media">Caption<img src="picture.png"/></span></p>'
        )
        source = deepcopy(root).find(".//{*}p")
        inventory = segments[0].epub_state.rich_source
        mark, atom = inventory.marks[0], inventory.atoms[0]
        segments[0].assign_translation(
            RichTarget(
                runs=[
                    InlineRun(marks=(mark.id,)),
                    InlineRun(text="图片"),
                    InlineRun(atom=atom.id, marks=(mark.id,)),
                ]
            )
        )
        render_rich_block(block, segments)
        spans = block.findall(".//{*}span")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].get("id"), "media")
        self.assertEqual(spans[0][0].get("src"), "picture.png")
        prove_rich_target(source, block, segments)
