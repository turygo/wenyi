from __future__ import annotations

import json
import unittest

from tests.fake_llm import fake_llm_dict
from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.agents.boundary_aligner import TextBoundaryAligner
from trans_novel.config import Config
from trans_novel.ingest.models import (
    SLOT_BOUNDARY_MARKER,
    EpubSegmentState,
    EpubTextSlot,
    Segment,
    assign_segment_translation,
)
from trans_novel.llm import FakeClient


def _segment() -> Segment:
    slots = [
        EpubTextSlot(
            id="private-a",
            field="text",
            source_value="The ",
            source_core="The",
            trailing_whitespace=" ",
        ),
        EpubTextSlot(
            id="private-b",
            field="tail",
            source_value="book",
            source_core="book",
        ),
    ]
    return Segment(
        index=0,
        source="The book",
        epub_state=EpubSegmentState(
            resource_href="OEBPS/chapter.xhtml",
            resource_sha256="resource",
            block_fingerprint="block",
            parse_mode="xml",
            slots=slots,
            slot_contract_sha256="contract",
        ),
    )


class TestTextBoundaryAligner(unittest.TestCase):
    def test_retries_then_assigns_without_exposing_epub_coordinates(self):
        responses = iter(
            [
                {"aligned": ["这本书"]},
                {"aligned": [f"这本{SLOT_BOUNDARY_MARKER}书"]},
            ]
        )
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                next(responses), ensure_ascii=False
            )
        )
        config = Config.from_dict({"llm": fake_llm_dict()})
        config.pipeline.align_retry_limit = 1
        aligned = TextBoundaryAligner(client, config).align_batch([["The", "book"]], ["这本书"])

        segment = _segment()
        assign_segment_translation(segment, aligned[0])
        self.assertEqual(segment.target, "这本 书")
        self.assertEqual([slot.target_core for slot in segment.epub_state.slots], ["这本", "书"])
        prompt = client.calls[0]["messages"][-1]["content"]
        self.assertNotIn("private-a", prompt)
        self.assertNotIn("OEBPS/chapter.xhtml", prompt)
        self.assertEqual(
            [call["agent"] for call in client.calls],
            ["analyst", "analyst"],
        )

    def test_rejects_translation_mutation(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"aligned": [f"这一本{SLOT_BOUNDARY_MARKER}书"]}, ensure_ascii=False
            )
        )
        config = Config.from_dict({"llm": fake_llm_dict()})
        config.pipeline.align_retry_limit = 0
        with self.assertRaisesRegex(WorkflowProtocolError, "边界对齐失败"):
            TextBoundaryAligner(client, config).align_batch([["The", "book"]], ["这本书"])

    def test_rejects_empty_translation_part(self):
        client = FakeClient(
            handler=lambda messages, agent, operation, json_mode: json.dumps(
                {"aligned": [f"{SLOT_BOUNDARY_MARKER}这本书"]}, ensure_ascii=False
            )
        )
        config = Config.from_dict({"llm": fake_llm_dict()})
        config.pipeline.align_retry_limit = 0
        with self.assertRaisesRegex(WorkflowProtocolError, "边界对齐失败"):
            TextBoundaryAligner(client, config).align_batch([["The", "book"]], ["这本书"])


if __name__ == "__main__":
    unittest.main()
