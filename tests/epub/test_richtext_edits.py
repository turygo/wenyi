"""译文局部编辑保留强调、链接和脚注语义范围。"""

import unittest

from trans_novel.epub.richtext import InlineMark, InlineRun, RichSource, RichTarget
from trans_novel.epub.richtext_edits import (
    apply_rich_edits,
    normalize_rich_target,
    omit_decorations,
    replace_rich_terms,
)
from trans_novel.postprocess.text_edits import TextEdit


class TestRichTextEdits(unittest.TestCase):
    def test_decoration_removed_with_identity_and_semantic_bold_preserved(self):
        source = RichSource(
            marks=[
                InlineMark(
                    id="drop",
                    path=(0,),
                    tag="span",
                    source_text="I",
                    kind="decoration",
                    attributes={"id": "opening"},
                ),
                InlineMark(id="bold", path=(1,), tag="b", source_text="important", kind="bold"),
            ],
        )
        target = RichTarget(
            runs=[
                InlineRun(text="我认为", marks=("drop",)),
                InlineRun(text="很重要。", marks=("bold",)),
            ]
        )
        cleaned = omit_decorations(target, source)
        self.assertEqual(cleaned.text, target.text)
        self.assertEqual(cleaned.runs[0], InlineRun(marks=("drop",)))
        self.assertEqual(cleaned.runs[1], InlineRun(text="我认为"))
        self.assertEqual(cleaned.runs[2], target.runs[1])
        self.assertEqual(omit_decorations(cleaned, source), cleaned)

    def test_length_change_preserves_later_bold_and_atom(self):
        target = RichTarget(
            runs=[
                InlineRun(text="格雷厄姆"),
                InlineRun(atom="n1"),
                InlineRun(text="全部完整保留。", marks=("b1",)),
            ]
        )
        updated, conflicts = replace_rich_terms(target, {"格雷厄姆": "本杰明·格雷厄姆"})
        self.assertEqual(conflicts, [])
        self.assertEqual(updated.text, "本杰明·格雷厄姆全部完整保留。")
        self.assertEqual(
            [(r.text, r.marks) for r in updated.runs if r.marks], [("全部完整保留。", ("b1",))]
        )
        self.assertEqual([r.atom for r in updated.runs if r.atom], ["n1"])

    def test_replacement_crossing_format_or_atom_is_reported(self):
        for runs in [
            [InlineRun(text="格雷", marks=("b",)), InlineRun(text="厄姆")],
            [InlineRun(text="格雷"), InlineRun(atom="n"), InlineRun(text="厄姆")],
        ]:
            with self.subTest(runs=runs):
                target = RichTarget(runs=runs)
                updated, conflicts = replace_rich_terms(target, {"格雷厄姆": "格雷厄穆"})
                self.assertEqual(updated, target)
                self.assertEqual(len(conflicts), 1)
                with self.assertRaises(ValueError):
                    apply_rich_edits(target, [TextEdit(0, 4, "新词")])

    def test_same_marks_allow_fragment_collapse_and_empty_runs(self):
        target = RichTarget(
            runs=[
                InlineRun(text="甲.", marks=("b",)),
                InlineRun(text=".", marks=("b",)),
                InlineRun(text=".乙", marks=("b",)),
            ]
        )
        updated = normalize_rich_target(target)
        self.assertEqual(updated.text, "甲……乙")
        self.assertTrue(all(run.marks == ("b",) for run in updated.runs))
        self.assertEqual(normalize_rich_target(updated), updated)

    def test_different_mark_collapse_is_skipped_but_local_punctuation_changes(self):
        target = RichTarget(runs=[InlineRun(text="甲.", marks=("b",)), InlineRun(text="..乙,")])
        updated = normalize_rich_target(target)
        self.assertEqual(updated.text, "甲...乙，")
        self.assertEqual(updated.runs[0].marks, ("b",))

    def test_literal_scope_and_atom_barrier(self):
        source = RichSource(marks=[InlineMark(id="c", path=(0,), tag="code", source_text="x")])
        target = RichTarget(
            runs=[
                InlineRun(text="甲...", marks=("c",)),
                InlineRun(text="."),
                InlineRun(atom="n"),
                InlineRun(text="..乙,"),
            ]
        )
        updated = normalize_rich_target(target, source=source)
        self.assertEqual(updated.text, "甲......乙，")
        self.assertEqual([r.atom for r in updated.runs if r.atom], ["n"])

    def test_disjoint_sequential_term_replacements_keep_link_scope(self):
        target = RichTarget(runs=[InlineRun(text="前AB后", marks=("link",))])
        updated, conflicts = replace_rich_terms(target, {"AB": "CD", "CD": "E"})
        self.assertEqual(updated.text, "前E后")
        self.assertEqual(conflicts, [])
        self.assertTrue(all(r.marks == ("link",) for r in updated.runs))


class TestLiteralTermProtection(unittest.TestCase):
    def test_term_replacements_preserve_code_and_urls(self):
        from trans_novel.epub.richtext import InlineMark, RichSource

        source = RichSource(
            marks=[InlineMark(id="code", path=(0,), tag="code", source_text="Smith")],
            runs=[InlineRun(text="Smith", marks=("code",))],
        )
        target = RichTarget(
            runs=[
                InlineRun(text="代码Smith", marks=("code",)),
                InlineRun(text="，https://example.com/Smith，人物Smith。"),
            ]
        )
        edited, conflicts = replace_rich_terms(target, {"Smith": "史密斯"}, source=source)
        self.assertEqual(edited.text, "代码Smith，https://example.com/Smith，人物史密斯。")
        self.assertEqual(len(conflicts), 2)
        self.assertEqual(edited.runs[0].marks, ("code",))
