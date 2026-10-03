"""恢复引用对象与冻结译文的独立边界。"""

from __future__ import annotations

import unittest
from copy import deepcopy

from tests.assemble.epub.rendering.test_richtext import _fixture
from trans_novel.assemble.epub.rendering.richtext import render_rich_block
from trans_novel.assemble.epub.verification.richtext import prove_rich_target
from trans_novel.epub.richtext import (
    InlineRun,
    NoteReference,
    RichSource,
    RichTarget,
    validate_rich_target,
)


def _note_fixture():
    _, root, block, segments = _fixture(
        '<p>Original<span id="ref"><a href="notes.xhtml#n10"><sup>10</sup></a></span>.</p>'
    )
    original = deepcopy(root).find(".//{*}p")
    source = segments[0].epub_state.rich_source
    owner, link, superscript = source.marks
    reference = NoteReference(
        mark_id=link.id,
        label="10",
        kind="noteref",
        target_resource="notes.xhtml",
        target_path=(1, 0),
        content_marks=(link.id, superscript.id),
    )
    source.note_references = [reference]
    return original, block, segments, source, owner, link, superscript


class TestRichTextNoteObjects(unittest.TestCase):
    def test_restore_label_without_changing_paid_text_and_prove_boundary(self):
        original, block, segments, source, owner, link, superscript = _note_fixture()
        target = RichTarget(
            version=3,
            runs=[
                InlineRun(text="完整译文"),
                InlineRun(note=link.id, marks=(owner.id, link.id, superscript.id)),
                InlineRun(text="。"),
            ],
        )
        validate_rich_target(source, target, expected_text="完整译文。")
        segments[0].assign_translation(target)
        render_rich_block(block, segments)
        self.assertEqual(segments[0].target, "完整译文。")
        self.assertEqual("".join(block.itertext()), "完整译文10。")
        self.assertEqual(block.find(".//{*}a").get("href"), "notes.xhtml#n10")
        prove_rich_target(original, block, segments)
        block.find(".//{*}sup").text = "11"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            prove_rich_target(original, block, segments)

    def test_existing_superscript_glyph_is_preserved_without_restoration(self):
        original, block, segments, source, owner, link, superscript = _note_fixture()
        target = RichTarget(
            runs=[
                InlineRun(text="译文"),
                InlineRun(text="¹⁰", marks=(owner.id, link.id, superscript.id)),
                InlineRun(text="。"),
            ]
        )
        validate_rich_target(source, target, expected_text="译文¹⁰。")
        segments[0].assign_translation(target)
        render_rich_block(block, segments)
        self.assertEqual("".join(block.itertext()), "译文¹⁰。")
        prove_rich_target(original, block, segments)

    def test_unknown_duplicate_and_double_representation_are_rejected(self):
        _, _, _, source, owner, link, superscript = _note_fixture()
        note = InlineRun(note=link.id, marks=(owner.id, link.id, superscript.id))
        variants = [
            [InlineRun(note="missing", marks=note.marks)],
            [note, note],
            [note, InlineRun(text="10", marks=note.marks)],
            [InlineRun(note=link.id, marks=(owner.id, link.id))],
            [InlineRun(text="unrelated Chinese", marks=note.marks)],
        ]
        for runs in variants:
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                validate_rich_target(source, RichTarget(version=3, runs=runs))

    def test_object_requires_v3_and_recovery_proof_and_cannot_carry_prose(self):
        _, _, _, source, owner, link, superscript = _note_fixture()
        marks = (owner.id, link.id, superscript.id)
        with self.assertRaises(ValueError):
            RichTarget(runs=[InlineRun(note=link.id, marks=marks)])
        with self.assertRaises(ValueError):
            InlineRun(note=link.id, marks=marks, text="10")
        source.note_references = []
        with self.assertRaises(ValueError):
            validate_rich_target(
                source, RichTarget(version=3, runs=[InlineRun(note=link.id, marks=marks)])
            )

    def test_source_reference_inventory_rejects_wrong_label_and_unknown_marks(self):
        _, _, _, source, _, _, _ = _note_fixture()
        for patch in ({"label": "11"}, {"content_marks": ("unknown",)}):
            raw = source.model_dump(mode="json")
            raw["note_references"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                RichSource.model_validate(raw)
