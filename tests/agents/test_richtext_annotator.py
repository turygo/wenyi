"""语义格式标注的严格协议与原译文守恒回归。"""

from __future__ import annotations

import json
import unittest

from tests.fixtures.fake_llm import fake_llm_dict
from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.agents.richtext_annotator import RichTextAnnotator
from trans_novel.config import Config
from trans_novel.ingest.models import InlineAtom, InlineMark, InlineRun, RichSource
from trans_novel.llm import FakeClient


def _config(retries=0):
    config = Config.from_dict({"llm": fake_llm_dict()})
    config.pipeline.protocol_retry_limit = retries
    return config


def _source():
    return RichSource(
        marks=[InlineMark(id="bold", path=(0,), tag="b", kind="bold", source_text="boldface type")],
        atoms=[InlineAtom(id="note", path=(1,), kind="note", label="1")],
        runs=[
            InlineRun(text="All original chapters are retained."),
            InlineRun(atom="note"),
            InlineRun(text=" I highlighted many passages in "),
            InlineRun(text="boldface type", marks=("bold",)),
            InlineRun(text="."),
        ],
    )


def _runs():
    return [
        {"text": "格雷厄姆的全部原始章节均完整保留。"},
        {"atom": "note"},
        {"text": "我还用"},
        {"text": "粗体", "marks": ["bold"]},
        {"text": "标出了格雷厄姆的许多段落。"},
    ]


def _response(runs, record_id=0):
    return json.dumps({"annotated": [{"id": record_id, "runs": runs}]}, ensure_ascii=False)


class TestRichTextAnnotator(unittest.TestCase):
    def test_reorders_emphasis_and_places_note_after_the_complete_statement(self):
        target = "格雷厄姆的全部原始章节均完整保留。我还用粗体标出了格雷厄姆的许多段落。"
        source = _source()
        before = source.model_dump()
        client = FakeClient(handler=lambda *_: _response(_runs()))
        result = RichTextAnnotator(client, _config()).annotate_batch([source], [target])
        self.assertEqual(result.targets[0].text, target)
        self.assertEqual(result.targets[0].runs[0].text[-5:], "完整保留。")
        self.assertEqual(result.targets[0].runs[1].atom, "note")
        self.assertEqual(result.targets[0].runs[3].text, "粗体")
        self.assertEqual(result.targets[0].runs[3].marks, ("bold",))
        self.assertEqual(source.model_dump(), before)
        self.assertEqual(result.request_count, 1)
        self.assertEqual(client.calls[0]["agent"], "analyst")
        self.assertEqual(client.calls[0]["operation"], "richtext.annotate")

    def test_plain_inventory_requires_no_model_call(self):
        client = FakeClient(handler=lambda *_: self.fail("plain target must not call a provider"))
        result = RichTextAnnotator(client, _config()).annotate_batch(
            [RichSource(runs=[InlineRun(text="Plain text.")])], ["普通译文。"]
        )
        self.assertEqual(result.targets[0].text, "普通译文。")
        self.assertEqual(result.request_count, 0)
        self.assertEqual(client.calls, [])

    def test_one_mark_can_cover_reordered_discontinuous_target_ranges(self):
        source = RichSource(
            marks=[
                InlineMark(
                    id="bold",
                    path=(0,),
                    tag="strong",
                    kind="bold",
                    source_text="two important parts",
                )
            ],
            runs=[InlineRun(text="two important parts", marks=("bold",))],
        )
        runs = [
            {"text": "要点甲", "marks": ["bold"]},
            {"text": "连接文字"},
            {"text": "要点乙", "marks": ["bold"]},
        ]
        client = FakeClient(handler=lambda *_: _response(runs))
        result = RichTextAnnotator(client, _config()).annotate_batch(
            [source], ["要点甲连接文字要点乙"]
        )
        self.assertEqual(
            [run.text for run in result.targets[0].runs if run.marks], ["要点甲", "要点乙"]
        )

    def test_reordered_response_ids_preserve_batch_order_and_skip_plain_inventory(self):
        source = _source()
        target = "格雷厄姆的全部原始章节均完整保留。我还用粗体标出了格雷厄姆的许多段落。"
        client = FakeClient(
            handler=lambda *_: json.dumps(
                {"annotated": [{"id": 2, "runs": _runs()}, {"id": 0, "runs": _runs()}]},
                ensure_ascii=False,
            )
        )
        result = RichTextAnnotator(client, _config()).annotate_batch(
            [source, RichSource(), source], [target, "中间普通文字", target]
        )
        self.assertEqual([value.text for value in result.targets], [target, "中间普通文字", target])
        request = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual([item["id"] for item in request["segments"]], [0, 2])

    def test_protocol_retry_cannot_change_the_saved_target(self):
        source = _source()
        target = "格雷厄姆的全部原始章节均完整保留。我还用粗体标出了格雷厄姆的许多段落。"
        changed = _runs()
        changed[0] = {"text": "改写了原来的译文。"}
        responses = iter([_response(changed), _response(_runs())])
        client = FakeClient(handler=lambda *_: next(responses))
        result = RichTextAnnotator(client, _config(1)).annotate_batch([source], [target])
        self.assertEqual(result.targets[0].text, target)
        self.assertEqual(result.request_count, 2)
        self.assertEqual(client.calls[0]["messages"], client.calls[1]["messages"])

    def test_invalid_text_marks_atoms_coordinates_and_ids_fail_explicitly(self):
        target = "格雷厄姆的全部原始章节均完整保留。我还用粗体标出了格雷厄姆的许多段落。"
        cases = [
            _response([{**item, "marks": []} for item in _runs()]),
            _response([item for item in _runs() if "atom" not in item]),
            _response([*_runs(), {"atom": "note"}]),
            _response([*_runs(), {"text": "多余译文"}]),
            _response([{"atom": "unknown"}, *_runs()]),
            _response([{**item, "marks": ["unknown"]} for item in _runs()]),
            _response([{**item, "slot_id": None} for item in _runs()]),
            _response(_runs(), True),
            json.dumps({"annotated": [{"id": 0, "runs": _runs()}] * 2}),
            json.dumps({"annotated": []}),
        ]
        for response in cases:
            with self.subTest(response=response):
                client = FakeClient(handler=lambda *_, response=response: response)
                with self.assertRaises(WorkflowProtocolError):
                    RichTextAnnotator(client, _config()).annotate_batch([_source()], [target])
                self.assertEqual(len(client.calls), 1)

    def test_length_mismatch_fails_before_call(self):
        client = FakeClient(handler=lambda *_: self.fail("invalid input must not call a provider"))
        with self.assertRaisesRegex(ValueError, "source/target count mismatch"):
            RichTextAnnotator(client, _config()).annotate_batch([_source()], [])


if __name__ == "__main__":
    unittest.main()
