"""页码锚点局部移动时不改变文字、格式范围或其他注释。"""

import unittest

from trans_novel.epub.richtext import InlineMark, InlineRun, RichSource, RichTarget
from trans_novel.epub.richtext_edits import place_boundary_atoms


def _atom_positions(target):
    offset = 0
    result = []
    for run in target.runs:
        if run.atom is not None:
            result.append((run.atom, offset, run.marks))
        offset += len(run.text)
    return result


def _character_marks(target):
    return [(char, run.marks) for run in target.runs for char in run.text]


class TestPageBoundaryAtoms(unittest.TestCase):
    def test_nearest_sentence_end_keeps_italics_notes_and_identity(self):
        target = RichTarget(
            runs=[
                InlineRun(text="甲乙", marks=("i",)),
                InlineRun(atom="page", marks=("i",)),
                InlineRun(text="丙", marks=("i",)),
                InlineRun(atom="note"),
                InlineRun(text="丁。", marks=("i",)),
                InlineRun(text="", marks=("identity",)),
                InlineRun(text="之后。"),
            ]
        )
        updated, count = place_boundary_atoms(target, {"page"})
        self.assertEqual(count, 1)
        self.assertEqual(updated.text, target.text)
        self.assertEqual(_character_marks(updated), _character_marks(target))
        self.assertEqual(_atom_positions(updated), [("note", 3, ()), ("page", 5, ("i",))])
        self.assertTrue(any(not run.text and run.marks == ("identity",) for run in updated.runs))
        self.assertEqual(place_boundary_atoms(updated, {"page"}), (updated, 0))

    def test_equal_distance_moves_forward(self):
        target = RichTarget(
            runs=[InlineRun(text="甲。乙丙"), InlineRun(atom="page"), InlineRun(text="丁。戊")]
        )
        updated, count = place_boundary_atoms(target, {"page"})
        self.assertEqual(count, 1)
        self.assertEqual(_atom_positions(updated), [("page", 6, ())])

    def test_multiple_pages_keep_order_and_start_anchor_is_kept(self):
        target = RichTarget(
            runs=[
                InlineRun(atom="start"),
                InlineRun(text="甲乙丙"),
                InlineRun(atom="p1"),
                InlineRun(text="丁"),
                InlineRun(atom="p2"),
                InlineRun(text="。后文。"),
            ]
        )
        updated, count = place_boundary_atoms(target, {"start", "p1", "p2"})
        self.assertEqual(count, 2)
        self.assertEqual(_atom_positions(updated), [("start", 0, ()), ("p1", 5, ()), ("p2", 5, ())])
        self.assertEqual(place_boundary_atoms(updated, {"start", "p1", "p2"}), (updated, 0))

    def test_closing_quotes_and_ellipsis_are_part_of_boundary(self):
        target = RichTarget(
            runs=[InlineRun(text="他说“停！"), InlineRun(atom="p"), InlineRun(text="”后文……再说")]
        )
        updated, _ = place_boundary_atoms(target, {"p"})
        self.assertEqual(_atom_positions(updated), [("p", 6, ())])

    def test_urls_code_and_internal_ascii_period_are_not_boundaries(self):
        source = RichSource(marks=[InlineMark(id="code", path=(0,), tag="code", source_text="x?")])
        for runs in [
            [
                InlineRun(text="代码x?", marks=("code",)),
                InlineRun(atom="p"),
                InlineRun(text="以及完整句。"),
            ],
            [
                InlineRun(text="网址https://a.test/x?"),
                InlineRun(atom="p"),
                InlineRun(text="q=1 以及完整句。"),
            ],
            [
                InlineRun(text="数字9.8与缩写Mr."),
                InlineRun(atom="p"),
                InlineRun(text=" Smith以及完整句。"),
            ],
        ]:
            with self.subTest(runs=runs):
                target = RichTarget(runs=runs)
                updated, count = place_boundary_atoms(target, {"p"}, source=source)
                self.assertEqual(count, 1)
                self.assertEqual(_atom_positions(updated)[0][1], len(target.text))
                self.assertEqual(updated.text, target.text)
