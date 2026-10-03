from __future__ import annotations

import unittest

from trans_novel.epub.slots import (
    EpubSegmentState,
    EpubTextSlot,
    distribute_slot_translation,
    source_passthrough_transport,
)
from trans_novel.ingest.models import (
    InlineMark,
    InlineRun,
    RichSource,
    RichTarget,
    Segment,
    sanitize_generated_text,
)


def _segment(source_parts: list[str]) -> Segment:
    slots = [
        EpubTextSlot(
            id=f"slot-{index}",
            element_path=() if index == 0 else (index,),
            field="text" if index == 0 else "tail",
            source_value=part,
        )
        for index, part in enumerate(source_parts)
    ]
    return Segment(
        index=0,
        source="".join(source_parts),
        epub_state=EpubSegmentState(
            resource_href="OEBPS/chapter.xhtml",
            resource_sha256="resource",
            block_fingerprint="block",
            parse_mode="xml",
            slots=slots,
            slot_contract_sha256="contract",
            rich_source=RichSource(
                runs=[InlineRun(text=slot.source_value, slot_id=slot.id) for slot in slots]
            ),
        ),
    )


class TestIndependentRichTarget(unittest.TestCase):
    def test_target_run_lengths_and_order_are_independent_of_source_slots(self):
        segment = _segment(
            ["a" * 292, "b" * 135, "c" * 14, "d" * 708, "e" * 114, "f" * 22, "g" * 14, "h" * 169]
        )
        translation = (
            "母亲去上班，女儿去上学。母亲回到家，把手提包扔到桌上。女儿写作业，母亲在厨房唱歌。"
        )

        state = segment.epub_state
        state.rich_source.marks = [
            InlineMark(id="bold", path=(2,), tag="b", kind="bold", source_text="c" * 14)
        ]
        before = [slot.model_dump() for slot in state.slots]
        target = RichTarget(
            runs=[InlineRun(text="母亲", marks=("bold",)), InlineRun(text=translation[2:])]
        )
        segment.assign_translation(target)
        self.assertEqual(segment.target, translation)
        self.assertEqual(segment.rich_target.runs[0].text, "母亲")
        self.assertEqual([slot.model_dump() for slot in state.slots], before)
        self.assertEqual(len(segment.rich_target.runs), 2)
        self.assertEqual(len(state.slots), 8)

    def test_structured_source_rejects_the_legacy_distribution_helper(self):
        for source_parts in (["long source", "another source"], ["single source"]):
            with self.subTest(source_parts=source_parts):
                segment = _segment(source_parts)
                if len(source_parts) == 1:
                    segment.epub_state.rich_source.marks = [
                        InlineMark(
                            id="bold", path=(), tag="b", kind="bold", source_text=source_parts[0]
                        )
                    ]
                with self.assertRaisesRegex(ValueError, "requires semantic rich-text annotation"):
                    distribute_slot_translation(segment.epub_state, "译")
                self.assertIsNone(segment.target)

    def test_short_translation_allows_empty_presentation_identity_runs(self):
        segment = _segment(["long source", "another source"])
        segment.epub_state.rich_source.marks = [
            InlineMark(
                id="decoration",
                path=(0,),
                tag="span",
                source_text="long source",
                kind="decoration",
                attributes={"id": "original-anchor"},
            )
        ]
        segment.assign_translation(
            RichTarget(runs=[InlineRun(text="", marks=("decoration",)), InlineRun(text="译")])
        )
        self.assertEqual(segment.target, "译")
        self.assertEqual(segment.rich_target.runs[0].marks, ("decoration",))
        self.assertTrue(all(slot.target_value is None for slot in segment.epub_state.slots))

    def test_source_passthrough_preserves_whitespace_slot_exactly(self):
        segment = _segment(["你好", " ", "世界"])
        segment.assign_translation(source_passthrough_transport(segment.epub_state))
        self.assertEqual(
            [slot.target_value for slot in segment.epub_state.slots],
            ["你好", " ", "世界"],
        )
        self.assertEqual(segment.target, "你好 世界")

    def test_target_whitespace_is_exact_and_source_whitespace_is_evidence(self):
        segment = _segment(["你好", " ", "世界"])
        translation = "甲 乙"
        segment.assign_translation(RichTarget(runs=[InlineRun(text=translation)]))
        self.assertEqual(segment.target, translation)
        self.assertEqual(segment.epub_state.slots[1].source_value, " ")
        self.assertIsNone(segment.epub_state.slots[1].target_value)

    def test_assignment_removes_forbidden_controls_before_deriving_target(self):
        segment = _segment(["source"])
        segment.assign_translation([{"id": "slot-0", "value": "前\x02后"}])

        self.assertEqual(segment.target, "前后")
        self.assertEqual(segment.epub_state.slots[0].target_value, "前后")

    def test_legacy_targets_are_cleaned_on_reconstruction(self):
        segment = Segment.model_validate(
            {
                "index": 0,
                "source": "前后",
                "target": "前\x02后",
                "epub_state": {
                    "resource_href": "OEBPS/chapter.xhtml",
                    "resource_sha256": "resource",
                    "block_fingerprint": "block",
                    "parse_mode": "xml",
                    "slots": [
                        {
                            "id": "slot-0",
                            "field": "text",
                            "source_value": "前",
                            "target_value": "前\x02",
                        },
                        {
                            "id": "slot-1",
                            "element_path": [1],
                            "field": "tail",
                            "source_value": "后",
                            "target_value": "后",
                        },
                    ],
                    "slot_contract_sha256": "contract",
                },
            }
        )

        self.assertEqual(segment.target, "前后")
        self.assertEqual([slot.target_value for slot in segment.epub_state.slots], ["前", "后"])
        self.assertEqual(segment.source, "前后")
        self.assertIsNone(segment.rich_target)
        self.assertEqual(sanitize_generated_text("\t\n\r中😀"), "\t\n\r中😀")


if __name__ == "__main__":
    unittest.main()
