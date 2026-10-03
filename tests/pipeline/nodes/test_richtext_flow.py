"""翻译、润色和目标编辑保留语义范围的离线行为回归。"""

from __future__ import annotations

import json
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from tests.fixtures.fake_llm import fake_llm_dict
from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.agents.polisher import PolishResult
from trans_novel.agents.richtext_annotator import RichTextAnnotator
from trans_novel.agents.translator import TranslationBatchResult
from trans_novel.config import Config
from trans_novel.epub.slots import EpubSegmentState, EpubTextSlot
from trans_novel.ingest.models import (
    Chapter,
    InlineMark,
    InlineRun,
    RichSource,
    RichTarget,
    Segment,
)
from trans_novel.llm import FakeClient
from trans_novel.pipeline.execution.readiness import assemble_readiness_problems
from trans_novel.pipeline.nodes.common import normalize_batch
from trans_novel.pipeline.nodes.finish import TitlesNode
from trans_novel.pipeline.nodes.polish import PolishNode
from trans_novel.pipeline.nodes.translation_batch import align_epub_translations, translate_batch
from trans_novel.pipeline.planning.fingerprints import (
    assembly_target_fingerprint_part,
    translation_structure_fingerprint_part,
)
from trans_novel.pipeline.state import ChapterProgress, PolishBatch, RollingContext


def _config():
    config = Config.from_dict({"llm": fake_llm_dict()})
    config.source_lang = "en"
    config.pipeline.protocol_retry_limit = 0
    config.punctuation_normalize = False
    return config


def _segment(source="There is no sure and easy path to riches anywhere.", *, formatted=True):
    marks = [InlineMark(id="bold", path=(0,), tag="b", kind="bold", source_text=source)]
    inventory = RichSource(
        marks=marks if formatted else [],
        runs=[InlineRun(text=source, marks=("bold",) if formatted else (), slot_id="s")],
    )
    return Segment(
        index=0,
        source=source,
        epub_state=EpubSegmentState(
            resource_href="chapter.xhtml",
            resource_sha256="archive",
            block_fingerprint="block",
            parse_mode="xml",
            slots=[EpubTextSlot(id="s", field="text", source_value=source)],
            slot_contract_sha256="contract",
            rich_source=inventory,
        ),
    )


def _annotator(config):
    def handler(messages, *_):
        records = json.loads(messages[-1]["content"])["segments"]
        return json.dumps(
            {
                "annotated": [
                    {"id": item["id"], "runs": [{"text": item["target"], "marks": ["bold"]}]}
                    for item in records
                ]
            },
            ensure_ascii=False,
        )

    client = FakeClient(handler=handler)
    return RichTextAnnotator(client, config), client


class TestRichTextFlow(unittest.TestCase):
    def test_full_wall_street_sentence_remains_bold_including_its_punctuation(self):
        segment = _segment()
        target = "在华尔街或任何其他地方，都不存在确定而轻松的致富之路。"
        annotator, client = _annotator(_config())
        transport = align_epub_translations([segment], [target], annotator=annotator)[0]
        segment.assign_translation(transport)
        self.assertEqual(segment.target, target)
        self.assertEqual(segment.rich_target.runs[0].marks, ("bold",))
        self.assertEqual(segment.rich_target.runs[0].text, target)
        self.assertEqual(segment.epub_state.slots[0].source_value, segment.source)
        self.assertIsNone(segment.epub_state.slots[0].target_value)
        self.assertEqual(len(client.calls), 1)

    def test_unchanged_target_reuses_format_without_model_dependency(self):
        segment = _segment()
        segment.assign_translation(RichTarget(runs=[InlineRun(text="整句中文。", marks=("bold",))]))
        transport = align_epub_translations([segment], ["整句中文。"])[0]
        self.assertEqual(transport, segment.rich_target)
        self.assertIsNot(transport, segment.rich_target)

    def test_plain_inventory_has_no_annotation_dependency(self):
        transport = align_epub_translations([_segment(formatted=False)], ["普通中文。"])[0]
        self.assertEqual(transport.text, "普通中文。")

    def test_missing_formatted_annotation_dependency_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "requires a rich text annotator"):
            align_epub_translations([_segment()], ["整句中文。"])

    def test_formatted_canonical_title_is_annotated_before_assignment(self):
        segment = _segment(source="A Source Title")
        segment.kind = "heading"
        chapter = Chapter(index=0, title=segment.source, segments=[segment])
        store = Mock()
        store.load_chapter.return_value = chapter
        annotator, client = _annotator(_config())
        skipped = TitlesNode._overwrite_canonical_title_segments(
            store,
            {"chapters": [{"index": 0, "title": segment.source}]},
            {"chapter:0": "中文标题"},
            {0: "chapter:0"},
            {},
            annotator=annotator,
        )
        self.assertEqual(skipped, [])
        self.assertEqual(segment.target, "中文标题")
        self.assertEqual(segment.rich_target.runs[0].marks, ("bold",))
        self.assertEqual(segment.meta["canonical_title_id"], "chapter:0")
        self.assertEqual(len(client.calls), 1)
        store.save_chapter.assert_called_once_with(chapter)

    def test_translation_counts_annotation_calls_and_normalizes_after_annotation(self):
        segment = _segment()
        annotator, client = _annotator(_config())
        translator = Mock()
        translator.translate_batch.return_value = TranslationBatchResult(("中文...",), 2)
        result, count = translate_batch(
            translator,
            [segment],
            [],
            RollingContext(),
            "",
            chapter_segments=[segment],
            start_index=0,
            chapter_title="",
            n_recent=2,
            annotator=annotator,
            punctuation_normalize=True,
        )
        self.assertEqual(result[0].text, "中文……")
        self.assertEqual(count, 3)
        request = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual(request["segments"][0]["target"], "中文...")

    def test_annotation_precedes_literal_url_and_decimal_safe_local_normalization(self):
        source = 'Explain "x--y", https://example.com/a?x=1.2 and 9.11.'
        segment = _segment(source=source)
        segment.epub_state.rich_source.marks.append(
            InlineMark(id="code", path=(1,), tag="code", kind="style", source_text='"x--y"')
        )
        target = '他说"中文"，代码 "x--y"，网址 https://example.com/a?x=1.2，数字9.11...'
        runs = [
            {"text": '他说"中文"，代码 ', "marks": ["bold"]},
            {"text": '"x--y"', "marks": ["code", "bold"]},
            {"text": "，网址 https://example.com/a?x=1.2，数字9.11..."},
        ]
        client = FakeClient(
            handler=lambda *_: json.dumps(
                {"annotated": [{"id": 0, "runs": runs}]}, ensure_ascii=False
            )
        )
        translator = Mock()
        translator.translate_batch.return_value = TranslationBatchResult((target,), 1)
        result, _ = translate_batch(
            translator,
            [segment],
            [],
            RollingContext(),
            "",
            chapter_segments=[segment],
            start_index=0,
            chapter_title="",
            n_recent=2,
            annotator=RichTextAnnotator(client, _config()),
            punctuation_normalize=True,
        )
        self.assertEqual(
            result[0].text, '他说“中文”，代码 "x--y"，网址 https://example.com/a?x=1.2，数字9.11……'
        )
        self.assertEqual([run.text for run in result[0].runs if "code" in run.marks], ['"x--y"'])
        request = json.loads(client.calls[0]["messages"][-1]["content"])
        self.assertEqual(request["segments"][0]["target"], target)

    def test_annotation_protocol_failure_does_not_become_successful_source_fallback(self):
        segment = _segment()
        client = FakeClient(handler=lambda *_: '{"annotated":[]}')
        translator = Mock()
        translator.translate_batch.return_value = TranslationBatchResult(("中文译文。",), 1)
        with self.assertRaises(WorkflowProtocolError):
            translate_batch(
                translator,
                [segment],
                [],
                RollingContext(),
                "",
                chapter_segments=[segment],
                start_index=0,
                chapter_title="",
                n_recent=2,
                annotator=RichTextAnnotator(client, _config()),
            )
        self.assertIsNone(segment.target)

    def test_local_normalization_does_not_move_the_bold_range(self):
        segment = _segment()
        segment.assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="前言, "),
                    InlineRun(text="核心", marks=("bold",)),
                    InlineRun(text="。"),
                ]
            )
        )
        normalize_batch([segment], [segment.target], punctuation_normalize=True)
        self.assertEqual(segment.target, "前言，核心。")
        self.assertEqual([run.text for run in segment.rich_target.runs if run.marks], ["核心"])

    def test_rejected_polish_keeps_existing_ranges_without_reannotation(self):
        segment = _segment(source='"Hello," she said.')
        original = RichTarget(
            runs=[
                InlineRun(text="“"),
                InlineRun(text="你好", marks=("bold",)),
                InlineRun(text="，”她说。"),
            ]
        )
        segment.assign_translation(original)
        chapter = Chapter(index=0, segments=[segment])
        progress = ChapterProgress(pending_polish=[PolishBatch(start=0, count=1)])
        future = Future()
        future.set_result(PolishResult(["你好，她说。"], {}, {0: "你好，她说。"}))
        polisher = SimpleNamespace(src="en", client=FakeClient())
        annotator = Mock()
        annotator.annotate_batch.side_effect = AssertionError(
            "rejected proposal must not reannotate"
        )
        node = PolishNode(
            polisher=polisher,
            extractor=None,
            glossary=None,
            config=_config(),
            style_brief="",
            annotator=annotator,
        )
        node._drain_chapter_polish(
            chapter, progress, [segment], {(0, 0): (future, ())}, None, "", [], Mock(), 0
        )
        self.assertEqual(segment.rich_target, original)
        self.assertEqual(progress.pending_polish, [])
        annotator.annotate_batch.assert_not_called()

    def test_source_and_target_format_changes_affect_their_fingerprints(self):
        segment = _segment()
        segment.assign_translation(RichTarget(runs=[InlineRun(text="中文", marks=("bold",))]))
        source_fp = translation_structure_fingerprint_part([segment])
        output_fp = assembly_target_fingerprint_part([segment])
        segment.epub_state.rich_source.marks[0].style_evidence = ["font-weight: 700"]
        self.assertNotEqual(source_fp, translation_structure_fingerprint_part([segment]))
        segment.rich_target.runs = [InlineRun(text="中", marks=("bold",)), InlineRun(text="文")]
        self.assertNotEqual(output_fp, assembly_target_fingerprint_part([segment]))

    def test_output_readiness_requires_valid_rich_targets_but_allows_machine_passthrough(self):
        segment = _segment()
        segment.target = "中文译文"
        chapter = Chapter(index=0, segments=[segment])
        keys = (
            "prepare",
            "analyze",
            "titles",
            "deterministic_qa",
            "repair",
            "report",
            "translate:0",
            "polish:0",
        )
        state = SimpleNamespace(
            chapters=[SimpleNamespace(index=0, processing=None)],
            progress={0: ChapterProgress(status="done")},
            nodes={key: SimpleNamespace(status="succeeded", node_id=key) for key in keys},
        )
        store = Mock()
        store.load_state.return_value = state
        store.load_chapter.return_value = chapter
        self.assertTrue(
            any("未建立格式标注" in problem for problem in assemble_readiness_problems(store))
        )
        segment.assign_translation(RichTarget(runs=[InlineRun(text="中文译文", marks=("bold",))]))
        self.assertEqual(assemble_readiness_problems(store), [])
        segment.rich_target.runs[0].marks = ("unknown",)
        self.assertTrue(
            any("无效译文格式标注" in problem for problem in assemble_readiness_problems(store))
        )
        segment.rich_target = None
        source = segment.epub_state.rich_source
        segment.epub_state.rich_source = None
        self.assertTrue(
            any("缺少可信源格式清单" in problem for problem in assemble_readiness_problems(store))
        )
        segment.epub_state.rich_source = source
        segment.source = segment.target = "245"
        self.assertEqual(assemble_readiness_problems(store), [])


if __name__ == "__main__":
    unittest.main()


class TestWholeBlockBreakInput(unittest.TestCase):
    def test_translator_receives_one_complete_block_with_linebreak_evidence(self):
        from trans_novel.ingest.models import InlineAtom

        segment = Segment(
            index=0,
            source="AlphaBeta",
            epub_state=EpubSegmentState(
                resource_href="c.xhtml",
                resource_sha256="hash",
                block_fingerprint="block",
                parse_mode="xml",
                slots=[
                    EpubTextSlot(id="s0", field="text", source_value="Alpha"),
                    EpubTextSlot(id="s1", field="tail", element_path=(0,), source_value="Beta"),
                ],
                slot_contract_sha256="contract",
                rich_source=RichSource(
                    atoms=[InlineAtom(id="br", path=(0,), kind="linebreak")],
                    runs=[
                        InlineRun(text="Alpha", slot_id="s0"),
                        InlineRun(atom="br"),
                        InlineRun(text="Beta", slot_id="s1"),
                    ],
                ),
            ),
        )
        translator = Mock()
        translator.translate_batch.return_value = TranslationBatchResult(("甲乙",), 1)
        client = FakeClient(
            handler=lambda *_args: (
                '{"annotated":[{"id":0,"runs":[{"text":"甲"},{"atom":"br"},{"text":"乙"}]}]}'
            )
        )
        targets, count = translate_batch(
            translator,
            [segment],
            [],
            RollingContext(),
            "",
            chapter_segments=[segment],
            start_index=0,
            chapter_title="",
            n_recent=0,
            annotator=RichTextAnnotator(client, _config()),
        )
        self.assertEqual(translator.translate_batch.call_args.args[0], ["Alpha\nBeta"])
        self.assertEqual(targets[0].text, "甲乙")
        self.assertEqual(count, 2)
        self.assertEqual(segment.source, "AlphaBeta")
