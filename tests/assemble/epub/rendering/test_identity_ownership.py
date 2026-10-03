"""独立证明拒绝同样式身份在不同中文位置或链接语境间互换。"""

import unittest
from copy import deepcopy

from tests.assemble.epub.rendering.test_richtext import _fixture
from trans_novel.assemble.epub.rendering.richtext import render_rich_block
from trans_novel.assemble.epub.verification.richtext import prove_rich_target
from trans_novel.epub.richtext import InlineRun, RichTarget


class TestIdentityOwnership(unittest.TestCase):
    def test_swapped_ids_are_rejected_even_for_equal_wrapper_styles(self):
        for linked in (False, True):
            for identity in ("id", "name"):
                with self.subTest(linked=linked, identity=identity):
                    first = f'<span class="same" {identity}="first">One</span>'
                    second = f'<span class="same" {identity}="second">Two</span>'
                    if linked:
                        first = '<a href="#one">' + first + "</a>"
                        second = '<a href="#two">' + second + "</a>"
                    _, root, block, segments = _fixture("<p>" + first + second + "</p>")
                    source = deepcopy(root).find(".//{*}p")
                    inventory = segments[0].epub_state.rich_source
                    segments[0].assign_translation(
                        RichTarget(
                            runs=[
                                run.model_copy(update={"slot_id": None}) for run in inventory.runs
                            ]
                        )
                    )
                    render_rich_block(block, segments)
                    prove_rich_target(source, block, segments)
                    nodes = block.findall(".//{*}span")
                    nodes[0].set(identity, "second")
                    nodes[1].set(identity, "first")
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        prove_rich_target(source, block, segments)

    def test_eat_kuku_implicit_subject_retains_boundary_and_other_semantic_ranges(self):
        _, root, block, segments = _fixture(
            '<p>Eat <i id="kuku">kuku</i>! <i id="you">You</i> <b id="eat">eat</b> it!</p>'
        )
        source = deepcopy(root).find(".//{*}p")
        kuku, subject, verb = segments[0].epub_state.rich_source.marks
        target = RichTarget(
            version=3,
            runs=[
                InlineRun(text="吃"),
                InlineRun(text="库库", marks=(kuku.id,)),
                InlineRun(text="！"),
                InlineRun(
                    marks=(subject.id,),
                    omission="implicit",
                    explanation="The second Chinese imperative implies its subject.",
                ),
                InlineRun(text="把它"),
                InlineRun(text="吃", marks=(verb.id,)),
                InlineRun(text="了！"),
            ],
        )
        segments[0].assign_translation(target)
        render_rich_block(block, segments)
        self.assertEqual("".join(block.itertext()), "吃库库！把它吃了！")
        self.assertEqual(block.xpath('.//*[@id="kuku"]')[0].text, "库库")
        self.assertEqual(block.xpath('.//*[@id="eat"]')[0].text, "吃")
        subject_node = block.xpath('.//*[@id="you"]')[0]
        self.assertFalse(subject_node.text)
        self.assertEqual(subject_node.getprevious().tail, "！")
        self.assertEqual(subject_node.tail, "把它")
        prove_rich_target(source, block, segments)
        for tamper in ("swap", "delete", "duplicate"):
            with self.subTest(tamper=tamper):
                altered = deepcopy(block)
                subject_node = altered.xpath('.//*[@id="you"]')[0]
                if tamper == "swap":
                    altered.xpath('.//*[@id="kuku"]')[0].set("id", "you")
                    subject_node.set("id", "kuku")
                elif tamper == "delete":
                    subject_node.attrib.pop("id")
                else:
                    subject_node.addnext(deepcopy(subject_node))
                with self.assertRaisesRegex(ValueError, "mismatch|identity missing or duplicated"):
                    prove_rich_target(source, altered, segments)

    def test_multiple_implicit_identities_share_one_boundary_without_changing_text(self):
        _, root, block, segments = _fixture(
            '<p>Eat kuku! <i id="you">You</i> do it <i name="yourself">yourself</i>.</p>'
        )
        source = deepcopy(root).find(".//{*}p")
        subject, reflexive = segments[0].epub_state.rich_source.marks
        segments[0].assign_translation(
            RichTarget(
                version=3,
                runs=[
                    InlineRun(text="吃库库！"),
                    InlineRun(
                        marks=(subject.id, reflexive.id),
                        omission="implicit",
                        explanation="The imperative implies the listener performing the action.",
                    ),
                    InlineRun(text="动手吧。"),
                ],
            )
        )
        render_rich_block(block, segments)
        self.assertEqual("".join(block.itertext()), "吃库库！动手吧。")
        self.assertEqual(len(block.xpath('.//*[@id="you"]')), 1)
        self.assertEqual(len(block.xpath('.//*[@name="yourself"]')), 1)
        prove_rich_target(source, block, segments)

    def test_merged_source_identities_share_meaningful_target_range_in_version_two(self):
        _, root, block, segments = _fixture(
            '<p><i id="first">firm</i> and <i name="second">decisive</i>.</p>'
        )
        source = deepcopy(root).find(".//{*}p")
        first, second = segments[0].epub_state.rich_source.marks
        segments[0].assign_translation(
            RichTarget(runs=[InlineRun(text="坚决。", marks=(first.id, second.id))])
        )
        render_rich_block(block, segments)
        self.assertEqual("".join(block.itertext()), "坚决。")
        self.assertEqual(segments[0].rich_target.version, 2)
        self.assertEqual(block.xpath('.//*[@name="second"]')[0].text, "坚决。")
        prove_rich_target(source, block, segments)
