"""译文局部编辑保留强调、链接和脚注语义范围。"""

import unittest

from trans_novel.epub.richtext import (
    InlineAtom,
    InlineMark,
    InlineRun,
    RichSource,
    RichTarget,
    validate_rich_target,
)
from trans_novel.epub.richtext_edits import (
    apply_rich_edits,
    normalize_rich_target,
    omit_decorations,
    place_boundary_atoms,
    replace_rich_terms,
)
from trans_novel.postprocess.text_edits import TextEdit


class TestRichTextEdits(unittest.TestCase):
    def test_insertions_inside_literals_are_rejected_and_declared_role_is_protected(self):
        for tag, semantics in (("code", ()), ("span", ("literal",))):
            with self.subTest(tag=tag):
                source = RichSource(
                    marks=[
                        InlineMark(
                            id="literal",
                            path=(0,),
                            tag=tag,
                            source_text="x,y",
                            semantics=semantics,
                        )
                    ]
                )
                target = RichTarget(runs=[InlineRun(text="x,y", marks=("literal",))])
                with self.assertRaisesRegex(ValueError, "protected literal"):
                    apply_rich_edits(target, [TextEdit(1, 1, "新字")], source=source)
                updated, conflicts = replace_rich_terms(target, {"x": "新词"}, source=source)
                self.assertEqual(updated, target)
                self.assertEqual(len(conflicts), 1)
        email = RichTarget(runs=[InlineRun(text="name@example.com")])
        with self.assertRaisesRegex(ValueError, "protected literal"):
            apply_rich_edits(email, [TextEdit(6, 6, "新字")])

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
    def test_email_text_and_mailto_link_labels_are_immutable(self):
        source = RichSource(
            marks=[
                InlineMark(
                    id="email",
                    path=(0,),
                    tag="{http://www.w3.org/1999/xhtml}a",
                    attributes={"href": "mailto:business@simonandschuster.com"},
                    source_text="business@simonandschuster.com",
                    kind="link",
                )
            ]
        )
        target = RichTarget(
            runs=[
                InlineRun(text="联系business@simonandschuster.com，"),
                InlineRun(text="商务联系人...", marks=("email",)),
                InlineRun(text="另有business。"),
            ]
        )
        edited, conflicts = replace_rich_terms(
            target, {"business": "商务", "联系人": "联络人"}, source=source
        )
        self.assertEqual(edited.text, "联系business@simonandschuster.com，商务联系人...另有商务。")
        self.assertEqual(len(conflicts), 2)
        normalized = normalize_rich_target(edited, source=source)
        self.assertEqual(normalized, edited)
        validate_rich_target(source, normalized)

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


class TestRichIdentityEdits(unittest.TestCase):
    def test_deleted_identity_whitespace_keeps_original_mark_chain(self):
        for identity in ("id", "name"):
            with self.subTest(identity=identity):
                source = RichSource(
                    marks=[
                        InlineMark(id="outer", path=(0,), tag="span", source_text=" "),
                        InlineMark(
                            id="anchor",
                            path=(0, 0),
                            tag="span",
                            attributes={identity: "bookmark"},
                            source_text=" ",
                        ),
                    ]
                )
                target = RichTarget(
                    runs=[
                        InlineRun(text="甲，"),
                        InlineRun(text=" ", marks=("outer", "anchor")),
                        InlineRun(text="乙。"),
                    ]
                )
                validate_rich_target(source, target)
                edited = normalize_rich_target(target, source=source)
                self.assertEqual(edited.text, "甲，乙。")
                self.assertIn(InlineRun(marks=("outer", "anchor")), edited.runs)
                validate_rich_target(source, edited)
                self.assertEqual(normalize_rich_target(edited, source=source), edited)

    def test_deleted_pure_emphasis_does_not_create_empty_mark(self):
        source = RichSource(
            marks=[InlineMark(id="bold", path=(0,), tag="b", kind="bold", source_text=" ")]
        )
        target = RichTarget(runs=[InlineRun(text=" ", marks=("bold",))])
        for supplied_source in (None, source):
            with self.subTest(source=supplied_source):
                edited = apply_rich_edits(target, [TextEdit(0, 1, "")], source=supplied_source)
                self.assertEqual(edited.runs, [])
                validate_rich_target(source, edited)

    def test_same_marks_fragmentation_keeps_text_and_atom_mapping(self):
        text = "甲...…乙"
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                source = RichSource(
                    marks=[InlineMark(id="italic", path=(0,), tag="i", source_text=text)],
                    atoms=[InlineAtom(id="note", path=(1,), kind="note")],
                    runs=[InlineRun(text=text), InlineRun(atom="note")],
                )
                target = RichTarget(
                    runs=[
                        InlineRun(text=text[:split], marks=("italic",)) if split else InlineRun(),
                        InlineRun(text=text[split:], marks=("italic",))
                        if split < len(text)
                        else InlineRun(),
                        InlineRun(atom="note"),
                    ]
                )
                edited = normalize_rich_target(target, source=source)
                self.assertEqual(edited.text, "甲……乙")
                self.assertEqual([run.atom for run in edited.runs if run.atom], ["note"])
                self.assertEqual(edited.runs[-1], InlineRun(atom="note"))
                validate_rich_target(source, edited)
                self.assertEqual(normalize_rich_target(edited, source=source), edited)


class TestOmissionEdits(unittest.TestCase):
    def test_local_edits_preserve_version_and_boundary_declaration(self):
        declaration = InlineRun(
            marks=("subject",), omission="implicit", explanation="The subject is implied."
        )
        target = RichTarget(
            version=3,
            runs=[InlineRun(text="吃库库!"), declaration, InlineRun(text="动手吧。")],
        )
        edited = apply_rich_edits(target, [TextEdit(1, 3, "米饭")])
        self.assertEqual(edited.version, 3)
        self.assertEqual(edited.text, "吃米饭!动手吧。")
        self.assertIn(declaration, edited.runs)
        normalized = normalize_rich_target(target)
        self.assertEqual(normalized.version, 3)
        self.assertIn(declaration, normalized.runs)
        replaced, conflicts = replace_rich_terms(target, {"库库": "米饭"})
        self.assertEqual(conflicts, [])
        self.assertEqual(replaced.version, 3)
        self.assertIn(declaration, replaced.runs)
        self.assertEqual(apply_rich_edits(target, []), target)

    def test_cross_boundary_replacements_and_insertions_are_rejected(self):
        target = RichTarget(
            version=3,
            runs=[
                InlineRun(text="吃"),
                InlineRun(marks=("subject",), omission="implicit", explanation="Implied."),
                InlineRun(text="库库！"),
            ],
        )
        for edit in (TextEdit(0, 3, "吃饭"), TextEdit(1, 1, "米饭")):
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, "immutable"):
                apply_rich_edits(target, [edit])
        replaced, conflicts = replace_rich_terms(target, {"吃库库": "吃饭"})
        self.assertEqual(replaced, target)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(apply_rich_edits(target, [TextEdit(1, 3, "米饭")]).text, "吃米饭！")

    def test_decoration_removal_and_page_anchor_reconstruction_preserve_declaration(self):
        declaration = InlineRun(
            marks=("subject",), omission="merged", explanation="Merged into the imperative."
        )
        source = RichSource(
            marks=[
                InlineMark(id="drop", path=(0,), tag="span", source_text="E", kind="decoration"),
                InlineMark(id="subject", path=(1,), tag="i", source_text="you"),
            ],
            atoms=[InlineAtom(id="page", path=(2,), kind="anchor")],
            runs=[InlineRun(atom="page")],
        )
        target = RichTarget(
            version=3,
            runs=[
                InlineRun(text="吃", marks=("drop",)),
                InlineRun(atom="page"),
                InlineRun(text="库库。"),
                declaration,
                InlineRun(text="动手吧。"),
            ],
        )
        cleaned = omit_decorations(target, source)
        self.assertEqual(cleaned.version, 3)
        self.assertIn(declaration, cleaned.runs)
        relocated, changed = place_boundary_atoms(cleaned, {"page"}, source=source)
        self.assertEqual(changed, 1)
        self.assertEqual(relocated.version, 3)
        self.assertEqual(relocated.text, target.text)
        self.assertIn(declaration, relocated.runs)
        validate_rich_target(source, relocated)
