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
    def test_whitespace_rejection_reports_exact_difference_without_changing_input(self):
        source = RichSource(
            marks=[InlineMark(id="amount", path=(0,), tag="b", source_text="200")],
            runs=[InlineRun(text="200", marks=("amount",)), InlineRun(text=" grams")],
        )
        expected = "200 克 煮熟"
        responses = iter(
            [
                _response([{"text": "200", "marks": ["amount"]}, {"text": " 克煮熟"}]),
                _response([{"text": "200", "marks": ["amount"]}, {"text": " 克 煮熟"}]),
            ]
        )
        client = FakeClient(handler=lambda *_: next(responses))
        result = RichTextAnnotator(client, _config(retries=1)).annotate_batch([source], [expected])
        self.assertEqual(result.targets[0].text, expected)
        first, second = (json.loads(call["messages"][-1]["content"]) for call in client.calls)
        self.assertEqual(first["segments"], second["segments"])
        detail = second["retry_feedback"]["detail"]
        self.assertIn('"offset": 5', detail)
        self.assertIn('"expected_code_point": "U+0020"', detail)
        self.assertIn('"actual_code_point": "U+716E"', detail)
        self.assertIn('"expected_context": "200 克 煮熟"', detail)

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
        self.assertEqual(result.targets[0].version, 2)
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

    def test_eat_kuku_implicit_subject_selects_version_three_without_rewriting(self):
        source = RichSource(
            marks=[
                InlineMark(id="subject", path=(0,), tag="i", source_text="You"),
                InlineMark(id="food", path=(1,), tag="i", source_text="kuku"),
            ],
            runs=[
                InlineRun(text="You", marks=("subject",)),
                InlineRun(text=" eat "),
                InlineRun(text="kuku", marks=("food",)),
                InlineRun(text="!"),
            ],
        )
        runs = [
            {
                "marks": ["subject"],
                "omission": "implicit",
                "explanation": "The imperative implies the second-person subject.",
            },
            {"text": "吃"},
            {"text": "库库", "marks": ["food"]},
            {"text": "！"},
        ]
        client = FakeClient(handler=lambda *_: _response(runs))
        result = RichTextAnnotator(client, _config()).annotate_batch([source], ["吃库库！"])
        self.assertEqual(result.targets[0].version, 3)
        self.assertEqual(result.targets[0].text, "吃库库！")
        self.assertEqual(result.targets[0].runs[0].omission, "implicit")
        self.assertEqual(result.targets[0].runs[2].marks, ("food",))
        request = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual(request["segments"][0]["source"], source.model_dump(mode="json"))

    def test_implicit_retry_rejects_empty_explanation_then_preserves_target(self):
        source = RichSource(marks=[InlineMark(id="subject", path=(0,), tag="i", source_text="You")])
        invalid = [
            {"marks": ["subject"], "omission": "implicit", "explanation": " "},
            {"text": "吃库库！"},
        ]
        valid = [{**invalid[0], "explanation": "The subject is implied."}, invalid[1]]
        responses = iter([_response(invalid), _response(valid)])
        client = FakeClient(handler=lambda *_: next(responses))
        result = RichTextAnnotator(client, _config(1)).annotate_batch([source], ["吃库库！"])
        self.assertEqual(result.targets[0].version, 3)
        self.assertEqual(result.targets[0].text, "吃库库！")
        self.assertEqual(result.request_count, 2)
        retry = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertIn("requires an explanation", retry["retry_feedback"]["detail"])

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
        first = json.loads(client.calls[0]["messages"][-1]["content"])
        retried = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertEqual(first["segments"], retried["segments"])
        self.assertNotIn("retry_feedback", first)
        self.assertIn("record 0", retried["retry_feedback"]["detail"])
        self.assertEqual(client.calls[0]["messages"][0], client.calls[1]["messages"][0])

    def test_optional_style_identity_is_required_and_retry_receives_missing_mark_id(self):
        source = RichSource(
            marks=[
                InlineMark(
                    id="opening",
                    path=(0,),
                    tag="span",
                    kind="style",
                    attributes={"id": "opening-quote"},
                    source_text="“",
                )
            ],
            runs=[InlineRun(text="“", marks=("opening",)), InlineRun(text="Book")],
        )
        responses = iter(
            [
                _response([{"text": "《书》"}]),
                _response([{"text": "", "marks": ["opening"]}, {"text": "《书》"}]),
            ]
        )
        client = FakeClient(handler=lambda *_: next(responses))
        result = RichTextAnnotator(client, _config(1)).annotate_batch([source], ["《书》"])
        self.assertEqual(result.targets[0].text, "《书》")
        self.assertEqual(result.targets[0].runs[0].marks, ("opening",))
        retry = json.loads(client.calls[1]["messages"][-1]["content"])
        self.assertIn("inline anchor identity", retry["retry_feedback"]["detail"])
        self.assertIn("opening", retry["retry_feedback"]["detail"])
        self.assertEqual(result.request_count, 2)

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
