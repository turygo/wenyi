from __future__ import annotations

import unittest

from trans_novel.epub.slots import (
    EpubSegmentState,
    EpubTextSlot,
    distribute_slot_translation,
    source_passthrough_transport,
)
from trans_novel.ingest.models import Segment


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
        ),
    )


class TestSlotDistribution(unittest.TestCase):
    def test_distributes_by_source_length_without_changing_translation(self):
        segment = _segment(
            ["a" * 292, "b" * 135, "c" * 14, "d" * 708, "e" * 114, "f" * 22, "g" * 14, "h" * 169]
        )
        translation = (
            "母亲去上班，女儿去上学。母亲回到家，把手提包扔到桌上。女儿写作业，母亲在厨房唱歌。"
        )

        transport = distribute_slot_translation(segment.epub_state, translation)

        self.assertEqual("".join(item["value"] for item in transport), translation)
        self.assertTrue(all(item["value"] for item in transport))
        segment.assign_translation(transport)
        self.assertEqual(
            [slot.target_value for slot in segment.epub_state.slots],
            [item["value"] for item in transport],
        )

    def test_short_translation_remains_lossless_even_when_a_slot_is_empty(self):
        segment = _segment(["long source", "another source"])

        transport = distribute_slot_translation(segment.epub_state, "译")

        self.assertEqual("".join(item["value"] for item in transport), "译")
        segment.assign_translation(transport)
        self.assertEqual(segment.target, "译")

    def test_source_passthrough_preserves_whitespace_slot_exactly(self):
        segment = _segment(["你好", " ", "世界"])
        segment.assign_translation(source_passthrough_transport(segment.epub_state))
        self.assertEqual(
            [slot.target_value for slot in segment.epub_state.slots],
            ["你好", " ", "世界"],
        )
        self.assertEqual(segment.target, "你好 世界")

    def test_whitespace_slots_are_empty_and_target_spaces_are_preserved(self):
        segment = _segment(["你好", " ", "世界"])
        translation = "甲 乙"
        transport = distribute_slot_translation(segment.epub_state, translation)
        self.assertEqual("".join(item["value"] for item in transport), translation)
        segment.assign_translation(transport)
        self.assertEqual(segment.target, translation)
        self.assertEqual(segment.epub_state.slots[1].target_value, "")


if __name__ == "__main__":
    unittest.main()
