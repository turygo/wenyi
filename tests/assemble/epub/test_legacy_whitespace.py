"""旧 BR 纯空白字段只能在可重放证明下恢复，默认来源覆盖保持严格。"""

import tempfile
import unittest

from tests.pipeline.nodes import test_reannotate
from trans_novel.assemble.epub.richtext_sources import hydrate_rich_sources
from trans_novel.epub.slots import slot_contract_digest
from trans_novel.pipeline.nodes.reannotate import reannotate_store
from trans_novel.pipeline.state.models import TRANSLATION_POLICY_VERSION


class TestLegacyWhitespace(unittest.TestCase):
    def _fixture(self, directory, body):
        xhtml = '<html xmlns="http://www.w3.org/1999/xhtml"><body>' + body + "</body></html>"
        path, store = test_reannotate.TestReannotateStore()._fixture(directory, body_override=xhtml)
        chapter = store.load_chapter(0)
        return path, store, chapter

    def _discard_whitespace(self, chapter):
        segment = chapter.segments[0]
        source = segment.epub_state
        source.slots = [slot for slot in source.slots if slot.source_value.strip()]
        source.slot_contract_sha256 = slot_contract_digest(source.slots)
        source.rich_source = None
        return segment

    def test_table_media_trailing_run_restored_and_mirrors_receive_same_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, chapter = self._fixture(
                directory, '<td><a id="page36"/><b>TABLE 1</b><br/><p><img/></p>\n\t</td>'
            )
            original = self._discard_whitespace(chapter)
            text = original.target
            mirror = chapter.model_copy(deep=True)
            mirror.index = 20
            for segment in mirror.segments:
                segment.anchor = "mirror:" + segment.anchor
                for slot in segment.epub_state.slots:
                    slot.id = "mirror:" + slot.id
                segment.epub_state.slot_contract_sha256 = slot_contract_digest(
                    segment.epub_state.slots
                )
            with self.assertRaisesRegex(ValueError, "slot coverage mismatch"):
                hydrate_rich_sources(path, [chapter])
            records = hydrate_rich_sources(path, [chapter, mirror], restore_legacy_whitespace=True)
            self.assertEqual(len(records), 2)
            self.assertEqual({record["chapter_index"] for record in records}, {0, 20})
            for restored in (chapter.segments[0], mirror.segments[0]):
                state = restored.epub_state
                self.assertEqual(restored.target, text)
                self.assertEqual(
                    [(slot.element_path, slot.field) for slot in state.slots],
                    [((1,), "text"), ((3,), "tail")],
                )
                self.assertEqual(state.slots[-1].source_value, "\n\t")
                self.assertEqual(state.slots[-1].target_value, "")
                self.assertEqual(state.slot_contract_sha256, slot_contract_digest(state.slots))
                self.assertEqual(state.rich_source.text, "TABLE 1\n\t")
                self.assertEqual(len(state.rich_source.atoms), 3)
            self.assertEqual(
                hydrate_rich_sources(path, [chapter, mirror], restore_legacy_whitespace=True), []
            )
            self.assertEqual(hydrate_rich_sources(path, [chapter]), [])

    def test_leading_pure_run_is_owned_by_next_existing_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, chapter = self._fixture(directory, "<p> \t<br/><b>TEXT</b></p>")
            segment = self._discard_whitespace(chapter)
            records = hydrate_rich_sources(path, [chapter], restore_legacy_whitespace=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["segment_index"], segment.index)
            self.assertEqual(segment.epub_state.slots[0].element_path, ())
            self.assertEqual(segment.epub_state.slots[0].source_value, " \t")
            self.assertEqual(segment.epub_state.slots[0].target_value, "")

    def test_partial_nonwhite_extra_and_nbsp_source_fields_remain_rejected(self):
        cases = [
            ("<p><b>TEXT</b> <br/>\t</p>", "partial"),
            ("<p><b>TEXT</b><br/><i>OTHER</i></p>", "nonwhite"),
            ("<p><b>TEXT</b><br/>\u00a0</p>", "nbsp"),
            ("<p><b>TEXT</b><br/>\t</p>", "extra"),
        ]
        for body, mode in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, _, chapter = self._fixture(directory, body)
                segment = self._discard_whitespace(chapter)
                if mode == "nonwhite":
                    segment.epub_state.slots.pop()
                if mode == "extra":
                    segment.epub_state.slots[0].source_value = "TAMPERED"
                segment.epub_state.slot_contract_sha256 = slot_contract_digest(
                    segment.epub_state.slots
                )
                with self.assertRaisesRegex(ValueError, "legacy whitespace restoration"):
                    hydrate_rich_sources(path, [chapter], restore_legacy_whitespace=True)

    def test_migration_records_recovery_once_and_only_policy3_can_restore(self):
        body = '<td><a id="page36"/><b>TABLE 1</b><br/><p><img/></p>\n\t</td>'
        with tempfile.TemporaryDirectory() as directory:
            path, store, chapter = self._fixture(directory, body)
            segment = self._discard_whitespace(chapter)
            original = segment.target
            store.save_chapter(chapter)
            annotator = test_reannotate._Annotator()
            reannotate_store(store, path, annotator=annotator)
            state = store.load_state()
            self.assertEqual(state.identity.translation_policy_version, TRANSLATION_POLICY_VERSION)
            audit = state.meta["richtext_migration"]["legacy_whitespace_restored"]
            self.assertEqual(len(audit), 1)
            self.assertEqual(store.load_chapter(0).segments[0].target, original)
            calls = annotator.calls
            reannotate_store(store, path, annotator=annotator)
            self.assertEqual(annotator.calls, calls)
            self.assertEqual(
                store.load_state().meta["richtext_migration"]["legacy_whitespace_restored"], audit
            )
        for policy in (4, TRANSLATION_POLICY_VERSION):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as directory:
                path, store, chapter = self._fixture(directory, body)
                self._discard_whitespace(chapter)
                store.save_chapter(chapter)
                state = store.load_state()
                state.identity.translation_policy_version = policy
                store.save_state(state)
                annotator = test_reannotate._Annotator()
                with self.assertRaisesRegex(ValueError, "slot coverage mismatch"):
                    reannotate_store(store, path, annotator=annotator)
                self.assertEqual(annotator.calls, 0)
