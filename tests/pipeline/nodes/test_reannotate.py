"""既有中文译文标注迁移的原文绑定、守恒和恢复契约。"""

import json
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch

from tests.fixtures.books import write_sample_epub
from trans_novel.epub.richtext import InlineRun, RichSource, RichTarget, rich_source_digest
from trans_novel.epub.slots import slot_contract_digest
from trans_novel.ingest.epub.reader import read_epub
from trans_novel.ingest.models import (
    CANONICAL_TITLE_ID_META,
    ChapterProcessing,
    chapter_source_digest,
    segment_preserves_source,
)
from trans_novel.pipeline.nodes.reannotate import reannotate_store
from trans_novel.pipeline.state import (
    ChapterIndex,
    ChapterProgress,
    RunIdentity,
    RunState,
    RunStore,
)
from trans_novel.pipeline.state.models import TRANSLATION_POLICY_VERSION, PolishBatch, stable_digest


class _Annotator:
    def __init__(self, *, mode="valid", fail_on=None):
        self.calls = 0
        self.mode = mode
        self.fail_on = fail_on

    def annotate_batch(self, sources, texts):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("provider interrupted")
        targets = []
        for source, text in zip(sources, texts, strict=True):
            marks = tuple(mark.id for mark in source.marks)
            runs = [InlineRun(text=text + ("改" if self.mode == "changed" else ""), marks=marks)]
            atoms = [InlineRun(atom=atom.id) for atom in source.atoms]
            if self.mode == "missing":
                atoms = []
            if self.mode == "duplicate":
                atoms *= 2
            targets.append(RichTarget(runs=[*runs, *atoms]))
        return SimpleNamespace(targets=tuple(targets), request_count=1)


class TestReannotateStore(unittest.TestCase):
    def _fixture(self, directory, *, page=False, dropcap=False, body_override=None):
        original = os.path.join(directory, "original.epub")
        path = os.path.join(directory, "book.epub")
        write_sample_epub(original)
        semantics = 'role="doc-pagebreak"' if page else ""
        opening = '<span class="drop">I</span> have ' if dropcap else ""
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            "<style>.drop{font-size:1.5em;font-weight:bold}</style></head><body>"
            f'<p>{opening}<b>Important first.</b><a id="zero" {semantics}/></p>'
            "<p><i>Important second.</i></p></body></html>"
        )
        if body_override is not None:
            body = body_override
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, "w") as target:
            for item in source.infolist():
                target.writestr(
                    item, body if item.filename == "OEBPS/ch1.xhtml" else source.read(item.filename)
                )
        document = read_epub(path, "en", "zh")
        store = RunStore(os.path.join(directory, "state"))
        state = RunState(
            fmt="epub",
            source_path=path,
            meta=document.meta,
            identity=RunIdentity(
                source_bytes_sha256=document.meta["epub_sha256"], translation_policy_version=3
            ),
            chapters=[ChapterIndex(index=chapter.index) for chapter in document.chapters],
            progress={
                chapter.index: ChapterProgress(status="done") for chapter in document.chapters
            },
        )
        store.save_state(state)
        for chapter in document.chapters:
            for segment in chapter.segments:
                text = f"原有中文 {chapter.index}-{segment.index}，完整保留。"
                segment.epub_state.rich_source = None
                first = next(
                    index
                    for index, slot in enumerate(segment.epub_state.slots)
                    if slot.source_value.strip()
                )
                for index, slot in enumerate(segment.epub_state.slots):
                    slot.target_value = text if index == first else ""
                segment.target = text
            store.save_chapter(chapter)
        return path, store

    def test_trusted_decoration_reclassification_reuses_paid_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, dropcap=True)
            annotator = _Annotator()
            first = reannotate_store(store, path, annotator=annotator)
            chapter = store.load_chapter(0)
            segment = chapter.segments[0]
            mark = next(m for m in segment.epub_state.rich_source.marks if m.kind == "decoration")
            # 模拟旧识别器把独立成词的首字视为 CSS 强调的已支付状态。
            mark.kind = "style"
            segment.rich_target.runs[0].marks += (mark.id,)
            store.save_chapter(chapter)
            calls = annotator.calls
            repeated = reannotate_store(store, path, annotator=annotator)
            saved = store.load_chapter(0).segments[0]
            self.assertEqual(repeated["model_requests"], 0)
            self.assertEqual(annotator.calls, calls)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])
            self.assertFalse(any(mark.id in r.marks and r.text for r in saved.rich_target.runs))

    def test_migration_settles_model_page_position_and_reuses_result(self):
        class MidSentence(_Annotator):
            def annotate_batch(self, sources, texts):
                result = super().annotate_batch(sources, texts)
                for source, target in zip(sources, result.targets, strict=True):
                    if source.atoms:
                        text = target.text
                        target.runs = [
                            InlineRun(text=text[:2], marks=target.runs[0].marks),
                            InlineRun(atom=source.atoms[0].id),
                            InlineRun(text=text[2:], marks=target.runs[0].marks),
                        ]
                return result

        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, page=True)
            annotator = MidSentence()
            report = reannotate_store(store, path, annotator=annotator)
            segment = store.load_chapter(0).segments[0]
            self.assertEqual(report["page_anchors_moved"], 1)
            self.assertEqual(
                segment.rich_target.runs[-1].atom, segment.epub_state.rich_source.atoms[0].id
            )
            calls = annotator.calls
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(repeated["page_anchors_moved"], 0)
            self.assertEqual(annotator.calls, calls)
            self.assertEqual(repeated["target_sha256"], report["target_sha256"])

    def test_target_text_and_resume_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            old = [store.load_chapter(c.index).to_dict() for c in store.load_state().chapters]
            annotator = _Annotator()
            report = reannotate_store(store, path, annotator=annotator, batch_size=1)
            saved = [store.load_chapter(c.index) for c in store.load_state().chapters]
            self.assertEqual(
                [s.target for c in saved for s in c.segments],
                [s["target"] for c in old for s in c["segments"]],
            )
            self.assertEqual(
                store.load_state().identity.translation_policy_version, TRANSLATION_POLICY_VERSION
            )
            self.assertTrue(all(s.rich_target is not None for c in saved for s in c.segments))
            calls = annotator.calls
            repeat = reannotate_store(store, path, annotator=annotator, batch_size=1)
            self.assertEqual(annotator.calls, calls)
            self.assertEqual(repeat["annotated"], 0)
            self.assertEqual(repeat["target_sha256"], report["target_sha256"])
            first = saved[0].segments[0]
            for slot in first.epub_state.slots:
                slot.target_value = "已失效的旧槽位"
            store.save_chapter(saved[0])
            self.assertEqual(store.load_chapter(saved[0].index).segments[0].target, first.target)

    def test_plain_source_inventory_is_assigned_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)

            class FormattedOnly(_Annotator):
                def annotate_batch(self, sources, texts):
                    self.sources = getattr(self, "sources", []) + sources
                    if any(not source.marks and not source.atoms for source in sources):
                        raise AssertionError("plain paragraph must not enter paid batch")
                    return super().annotate_batch(sources, texts)

            annotator = FormattedOnly()
            report = reannotate_store(store, path, annotator=annotator, batch_size=8)
            self.assertEqual(report["deterministic"], 2)
            self.assertEqual(len(annotator.sources), 2)
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(report["annotated"], report["segments"])

    def test_invalid_annotation_does_not_advance_policy(self):
        for mode in ["changed", "missing", "duplicate"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, store = self._fixture(directory)
                with self.assertRaises(ValueError):
                    reannotate_store(store, path, annotator=_Annotator(mode=mode), batch_size=1)
                self.assertEqual(store.load_state().identity.translation_policy_version, 3)
                self.assertIsNone(store.load_chapter(0).segments[0].rich_target)
                with open(os.path.join(store.run_dir, "events.jsonl"), encoding="utf-8") as events:
                    interrupted = next(
                        item
                        for line in events
                        if (item := json.loads(line))["event"] == "richtext_migration_interrupted"
                    )
                self.assertEqual(interrupted["failed_batch"]["chapter_index"], 0)
                self.assertEqual(interrupted["failed_batch"]["segment_indices"], [0])
                self.assertEqual(interrupted["failed_batch"]["error_type"], "ValueError")
                self.assertTrue(interrupted["failed_batch"]["error_detail"])

    def test_partial_checkpoint_does_not_repeat_completed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                reannotate_store(store, path, annotator=_Annotator(fail_on=2), batch_size=1)
            self.assertIsNotNone(store.load_chapter(0).segments[0].rich_target)
            self.assertEqual(store.load_state().identity.translation_policy_version, 3)
            resumed = _Annotator()
            result = reannotate_store(store, path, annotator=resumed, batch_size=1)
            self.assertGreaterEqual(result["reused"], 1)
            self.assertEqual(resumed.calls, result["segments"] - result["reused"])

    def test_successful_later_futures_are_checkpointed_after_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                reannotate_store(store, path, annotator=_Annotator(fail_on=1), batch_size=1)
            segments = [
                segment
                for entry in store.load_state().chapters
                for segment in store.load_chapter(entry.index).segments
            ]
            self.assertIsNone(segments[0].rich_target)
            self.assertTrue(all(segment.rich_target is not None for segment in segments[1:4]))
            resumed = _Annotator()
            result = reannotate_store(store, path, annotator=resumed, batch_size=1)
            self.assertGreaterEqual(result["reused"], 3)
            self.assertEqual(resumed.calls, len(segments) - result["reused"])

    def test_unsupported_policy_and_pending_polish_prevent_calls(self):
        for policy, pending in [(2, False), (6, False), (3, True)]:
            with (
                self.subTest(policy=policy, pending=pending),
                tempfile.TemporaryDirectory() as directory,
            ):
                path, store = self._fixture(directory)
                state = store.load_state()
                state.identity.translation_policy_version = policy
                if pending:
                    state.progress[0].pending_polish = [PolishBatch(start=0, count=1)]
                store.save_state(state)
                annotator = _Annotator()
                with self.assertRaises(ValueError):
                    reannotate_store(store, path, annotator=annotator)
                self.assertEqual(annotator.calls, 0)

    def test_missing_source_slot_is_rejected_even_with_recomputed_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            chapter = store.load_chapter(0)
            state = chapter.segments[0].epub_state
            state.slots = []
            state.slot_contract_sha256 = slot_contract_digest(state.slots)
            chapter.segments[0].target = None
            # 验证覆盖检查而非仅仅目标完整性检查。
            store.save_chapter(chapter)
            from trans_novel.assemble.epub.richtext_sources import hydrate_rich_sources

            with self.assertRaisesRegex(ValueError, "slot coverage mismatch"):
                hydrate_rich_sources(path, [chapter])

    def test_source_mirrors_with_distinct_slot_ids_share_trusted_geometry(self):
        from trans_novel.assemble.epub.richtext_sources import hydrate_rich_sources

        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            chapter = store.load_chapter(0)
            mirror = chapter.model_copy(deep=True)
            mirror.index = 20
            for segment in mirror.segments:
                for slot in segment.epub_state.slots:
                    slot.id = "mirror:" + slot.id
                segment.epub_state.slot_contract_sha256 = slot_contract_digest(
                    segment.epub_state.slots
                )
            hydrate_rich_sources(path, [chapter, mirror])
            for segment in mirror.segments:
                source = segment.epub_state.rich_source
                self.assertTrue(
                    all(
                        run.slot_id is None or run.slot_id.startswith("mirror:")
                        for run in source.runs
                    )
                )

    def test_wrong_source_hash_prevents_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            state = store.load_state()
            state.identity.source_bytes_sha256 = "wrong"
            store.save_state(state)
            annotator = _Annotator()
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)


class TestPreservedCanonicalTitleMigration(unittest.TestCase):
    def test_canonical_title_in_preserved_chapter_keeps_translation_and_gets_rich_target(self):
        for title, translated, italic in (("Notes", "注释", False), ("Index", "索引", True)):
            with self.subTest(italic=italic), tempfile.TemporaryDirectory() as directory:
                heading = f"<i>{title}</i>" if italic else title
                body = (
                    '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
                    f"<h1>{heading}</h1><p>Original reference text.</p></body></html>"
                )
                path, store = TestReannotateStore()._fixture(directory, body_override=body)
                state = store.load_state()
                self.assertEqual(state.identity.translation_policy_version, 3)
                original = []
                for entry in state.chapters:
                    chapter = store.load_chapter(entry.index)
                    for segment in chapter.segments:
                        for slot in segment.epub_state.slots:
                            slot.target_value = slot.source_value
                        segment.target = segment.source
                        if chapter.index == 0 and segment.kind == "heading":
                            segment.meta[CANONICAL_TITLE_ID_META] = f"canonical-{title.lower()}"
                            segment.target = translated
                            for index, slot in enumerate(segment.epub_state.slots):
                                slot.target_value = translated if index == 0 else ""
                    chapter.set_processing(
                        ChapterProcessing(
                            action="preserve",
                            review_required=False,
                            reason="reference-only",
                            source_sha256=chapter_source_digest(chapter),
                            strategy_version="chapter_semantics_v2",
                        )
                    )
                    store.save_chapter(chapter)
                    original.extend(
                        [chapter.index, segment.index, segment.target]
                        for segment in chapter.segments
                    )
                expected_digest = stable_digest(original)
                annotator = _Annotator()
                report = reannotate_store(store, path, annotator=annotator)
                self.assertEqual(report["target_sha256"], expected_digest)
                self.assertEqual(report["annotated"], 1)
                self.assertEqual(report["deterministic"], 0 if italic else 1)
                self.assertEqual(report["model_requests"], int(italic))
                self.assertEqual(annotator.calls, int(italic))
                persisted = []
                for entry in state.chapters:
                    chapter = store.load_chapter(entry.index)
                    self.assertTrue(chapter.preserve_source)
                    for segment in chapter.segments:
                        self.assertTrue(segment.preserve_source)
                        self.assertEqual(segment.epub_state.rich_source.version, 2)
                        persisted.append([chapter.index, segment.index, segment.target])
                        if CANONICAL_TITLE_ID_META in segment.meta:
                            self.assertFalse(segment_preserves_source(segment))
                            self.assertEqual(segment.target, translated)
                            self.assertIsNotNone(segment.rich_target)
                            self.assertEqual(segment.rich_target.version, 2)
                            self.assertEqual(segment.rich_target.text, translated)
                            marks = segment.epub_state.rich_source.marks
                            self.assertEqual(bool(marks), italic)
                            if italic:
                                self.assertIn(marks[0].id, segment.rich_target.runs[0].marks)
                        else:
                            self.assertTrue(segment_preserves_source(segment))
                            self.assertEqual(segment.target, segment.source)
                            self.assertIsNone(segment.rich_target)
                self.assertEqual(persisted, original)
                repeated = reannotate_store(store, path, annotator=annotator)
                self.assertEqual(repeated["target_sha256"], expected_digest)
                self.assertEqual(repeated["annotated"], 0)
                self.assertEqual(repeated["model_requests"], 0)
                self.assertEqual(annotator.calls, int(italic))


class TestReannotationOriginalText(unittest.TestCase):
    def _fixture(self, directory):
        return TestReannotateStore()._fixture(directory)

    def test_raw_legacy_target_mismatch_rejected_before_any_chapter_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            raw = store.read_json(store.chapter_path_v2(0))
            raw["segments"][0]["target"] = "完整的另一个目标字符串"
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            with (
                patch.object(store, "load_chapter", side_effect=AssertionError("load too early")),
                self.assertRaisesRegex(ValueError, "raw target disagrees"),
            ):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)
            self.assertEqual(store.read_json(store.chapter_path_v2(0)), raw)

    def test_raw_rich_target_mismatch_rejected_before_any_chapter_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            reannotate_store(store, path, annotator=_Annotator())
            raw = store.read_json(store.chapter_path_v2(0))
            raw["segments"][0]["rich_target"]["runs"][0]["text"] += "改"
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            with (
                patch.object(store, "load_chapter", side_effect=AssertionError("load too early")),
                self.assertRaisesRegex(ValueError, "raw target disagrees"),
            ):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)

    def test_model_xml_cleanup_cannot_hide_changed_original_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            raw = store.read_json(store.chapter_path_v2(0))
            raw["segments"][0]["target"] += "\x01"
            raw["segments"][0]["epub_state"]["slots"][0]["target_value"] += "\x01"
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            with self.assertRaisesRegex(ValueError, "model loading changed"):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)
            self.assertEqual(store.read_json(store.chapter_path_v2(0)), raw)

    def test_unassigned_legacy_slot_without_pending_proof_remains_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            raw = store.read_json(store.chapter_path_v2(0))
            raw["segments"][0]["epub_state"]["slots"][0]["target_value"] = None
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            with self.assertRaisesRegex(ValueError, "proven pending"):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)

    def test_pending_proof_also_requires_the_frozen_full_original_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            reannotate_store(store, path, annotator=_Annotator())
            raw = store.read_json(store.chapter_path_v2(0))
            segment = raw["segments"][0]
            source = RichSource.model_validate(segment["epub_state"]["rich_source"])
            segment["rich_target"] = None
            for slot in segment["epub_state"]["slots"]:
                slot["target_value"] = None
            segment["meta"]["richtext_pending"] = {
                "version": 2,
                "source_sha256": rich_source_digest(source),
                "target_sha256": stable_digest(segment["target"]),
            }
            store.write_json(store.chapter_path_v2(0), raw)
            state = store.load_state()
            state.meta.pop("richtext_migration")
            store.save_state(state)
            annotator = _Annotator()
            with self.assertRaisesRegex(ValueError, "frozen full digest"):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)


class TestPaidRichContractUpgrade(unittest.TestCase):
    def _fixture(self, directory, **kwargs):
        return TestReannotateStore()._fixture(directory, **kwargs)

    def _downgrade_paid_annotations(self, store, *, omit_literal=False):
        state = store.load_state()
        state.identity.translation_policy_version = 4
        store.save_state(state)
        for chapter in state.chapters:
            raw = store.read_json(store.chapter_path_v2(chapter.index))
            for segment in raw["segments"]:
                source = segment["epub_state"]["rich_source"]
                source["version"] = 1
                for mark in source["marks"]:
                    mark["semantics"] = []
                target = segment.get("rich_target")
                if target is None:
                    continue
                target["version"] = 1
                if omit_literal:
                    literal_ids = {
                        mark["id"] for mark in source["marks"] if mark["tag"].endswith("}code")
                    }
                    for run in target["runs"]:
                        run["marks"] = [value for value in run["marks"] if value not in literal_ids]
            store.write_json(store.chapter_path_v2(chapter.index), raw)

    def test_paid_v1_annotations_upgrade_after_fresh_source_proof_without_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            first = reannotate_store(store, path, annotator=_Annotator())
            self._downgrade_paid_annotations(store)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])
            self.assertEqual(repeated["annotated"], 0)
            for chapter in store.load_state().chapters:
                for segment in store.load_chapter(chapter.index).segments:
                    self.assertEqual(segment.epub_state.rich_source.version, 2)
                    self.assertEqual(segment.rich_target.version, 2)

    def test_v1_css_evidence_serialization_whitespace_does_not_repeat_paid_annotation(self):
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            "<style>.accent { font-style: italic; font-weight: bold; }</style>"
            '</head><body><p><span class="accent">Important first.</span><a id="zero"/></p>'
            "<p><i>Important second.</i></p></body></html>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, body_override=body)
            first = reannotate_store(store, path, annotator=_Annotator())
            original_source = store.load_chapter(0).segments[0].epub_state.rich_source
            original_evidence = original_source.marks[0].style_evidence
            self.assertTrue(original_evidence)
            self._downgrade_paid_annotations(store)
            raw = store.read_json(store.chapter_path_v2(0))
            mark = raw["segments"][0]["epub_state"]["rich_source"]["marks"][0]
            mark["style_evidence"] = [
                value.replace("{", "{\n  ").replace(":", " : ").replace(";", ";\n  ")
                for value in mark["style_evidence"]
            ]
            self.assertNotEqual(mark["style_evidence"], original_evidence)
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            saved = store.load_chapter(0).segments[0]
            self.assertEqual(annotator.calls, 0)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])
            self.assertEqual(saved.rich_target.version, 2)
            self.assertEqual(
                saved.epub_state.rich_source.marks[0].style_evidence, original_evidence
            )

    def test_v1_old_css_evidence_cannot_hide_a_fresh_required_reset_wrapper(self):
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
            "<style>p { font-style: italic; } .normal { font-style: normal; }</style>"
            '</head><body><p><span class="normal">Important first.</span><a id="zero"/></p>'
            "<p><i>Important second.</i></p></body></html>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, body_override=body)
            first = reannotate_store(store, path, annotator=_Annotator())
            fresh = store.load_chapter(0).segments[0].epub_state.rich_source.marks[0]
            self.assertTrue(fresh.required)
            self.assertEqual(fresh.semantics, ())
            self._downgrade_paid_annotations(store)
            raw = store.read_json(store.chapter_path_v2(0))
            segment = raw["segments"][0]
            mark = segment["epub_state"]["rich_source"]["marks"][0]
            mark["required"] = False
            mark["style_evidence"] = [".normal { font-style : normal; }"]
            for run in segment["rich_target"]["runs"]:
                run["marks"] = [value for value in run["marks"] if value != mark["id"]]
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            saved = store.load_chapter(0).segments[0]
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(repeated["annotated"], 1)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])
            self.assertTrue(saved.epub_state.rich_source.marks[0].required)
            self.assertIn(mark["id"], saved.rich_target.runs[0].marks)

    def test_v2_evidence_change_still_requires_full_fresh_source_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            first = reannotate_store(store, path, annotator=_Annotator())
            raw = store.read_json(store.chapter_path_v2(0))
            source = raw["segments"][0]["epub_state"]["rich_source"]
            self.assertEqual(source["version"], 2)
            source["marks"][0]["style_evidence"].append("b { font-weight: bold; }")
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(repeated["annotated"], 1)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])

    def test_rich_target_is_authoritative_when_legacy_slot_targets_are_unassigned(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            first = reannotate_store(store, path, annotator=_Annotator())
            self._downgrade_paid_annotations(store)
            for chapter in store.load_state().chapters:
                raw = store.read_json(store.chapter_path_v2(chapter.index))
                for segment in raw["segments"]:
                    for slot in segment["epub_state"]["slots"]:
                        slot["target_value"] = None
                store.write_json(store.chapter_path_v2(chapter.index), raw)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])

    def test_legacy_geometry_change_reannotates_only_the_affected_paid_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            first = reannotate_store(store, path, annotator=_Annotator())
            self._downgrade_paid_annotations(store)
            raw = store.read_json(store.chapter_path_v2(0))
            raw["segments"][0]["epub_state"]["rich_source"]["marks"][0]["source_text"] = (
                "changed evidence"
            )
            store.write_json(store.chapter_path_v2(0), raw)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(repeated["annotated"], 1)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])

    def test_only_v1_target_failing_new_literal_semantics_is_reannotated(self):
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<p><code>Important first.</code><a id="zero"/></p>'
            "<p><i>Important second.</i></p></body></html>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, body_override=body)
            first = reannotate_store(store, path, annotator=_Annotator())
            self._downgrade_paid_annotations(store, omit_literal=True)
            annotator = _Annotator()
            repeated = reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 1)
            self.assertEqual(repeated["annotated"], 1)
            self.assertEqual(repeated["target_sha256"], first["target_sha256"])

    def test_failed_v1_upgrade_persists_bound_pending_text_and_resumes(self):
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<p><code>Important first.</code><a id="zero"/></p>'
            "<p><i>Important second.</i></p></body></html>"
        )
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory, body_override=body)
            first = reannotate_store(store, path, annotator=_Annotator())
            self._downgrade_paid_annotations(store, omit_literal=True)
            raw = store.read_json(store.chapter_path_v2(0))
            target = raw["segments"][0]["target"]
            for slot in raw["segments"][0]["epub_state"]["slots"]:
                slot["target_value"] = "stale old mapping"
            store.write_json(store.chapter_path_v2(0), raw)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                reannotate_store(store, path, annotator=_Annotator(fail_on=1))
            pending = store.read_json(store.chapter_path_v2(0))["segments"][0]
            self.assertIsNone(pending["rich_target"])
            self.assertTrue(
                all(slot["target_value"] is None for slot in pending["epub_state"]["slots"])
            )
            self.assertEqual(pending["target"], target)
            self.assertEqual(pending["meta"]["richtext_pending"]["version"], 2)
            resumed = _Annotator()
            result = reannotate_store(store, path, annotator=resumed)
            self.assertEqual(resumed.calls, 1)
            self.assertEqual(result["target_sha256"], first["target_sha256"])
            accepted = store.load_chapter(0).segments[0]
            self.assertEqual(accepted.target, target)
            self.assertEqual(accepted.rich_target.version, 2)
            self.assertNotIn("richtext_pending", accepted.meta)

    def test_resume_rejects_changed_saved_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path, store = self._fixture(directory)
            reannotate_store(store, path, annotator=_Annotator())
            chapter = store.load_chapter(0)
            chapter.segments[0].rich_target.runs[0].text += "修改"
            chapter.segments[0].target = chapter.segments[0].rich_target.text
            store.save_chapter(chapter)
            annotator = _Annotator()
            with self.assertRaisesRegex(ValueError, "changed during resume"):
                reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, 0)
