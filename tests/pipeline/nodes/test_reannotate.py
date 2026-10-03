"""既有中文译文标注迁移的原文绑定、守恒和恢复契约。"""

import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

from tests.fixtures.books import write_sample_epub
from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.epub.slots import slot_contract_digest
from trans_novel.ingest.epub.reader import read_epub
from trans_novel.pipeline.nodes.reannotate import reannotate_store
from trans_novel.pipeline.state import (
    ChapterIndex,
    ChapterProgress,
    RunIdentity,
    RunState,
    RunStore,
)
from trans_novel.pipeline.state.models import PolishBatch


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
    def _fixture(self, directory, *, page=False, dropcap=False):
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
                for index, slot in enumerate(segment.epub_state.slots):
                    slot.target_value = text if index == 0 else ""
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
            self.assertEqual(store.load_state().identity.translation_policy_version, 4)
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
        for policy, pending in [(2, False), (5, False), (3, True)]:
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
