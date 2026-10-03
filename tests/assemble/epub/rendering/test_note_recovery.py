"""引用恢复证明独立于旧脚注槽位，绑定与重证不能改写译文。"""

import copy
import os
import tempfile
import unittest
import zipfile

from tests.fixtures.books import write_sample_epub
from trans_novel.assemble.epub.rendering.note_recovery import (
    bind_note_references,
    note_references_for_source,
    validate_note_references,
)
from trans_novel.assemble.epub.richtext_sources import hydrate_rich_sources
from trans_novel.epub.archive import ZipSafetyError
from trans_novel.epub.notes import NoteRelations
from trans_novel.ingest.epub.reader import read_epub


class TestNoteRecovery(unittest.TestCase):
    def _fixture(self, directory, *, content="<sup>10</sup>", second=True):
        original = os.path.join(directory, "original.epub")
        path = os.path.join(directory, "book.epub")
        write_sample_epub(original)
        reference = (
            f'<span class="xrefInternal" id="en_10"><a href="ch2.xhtml#en10">{content}</a></span>'
        )
        note = (
            '<p class="endnote"><span id="en10">'
            '<a href="ch1.xhtml#en_10">10.</a></span><span>Ibid.</span></p>'
        )
        if second:
            reference += (
                '<span class="xrefInternal" id="en_11">'
                '<a href="ch2.xhtml#en11"><sup>11</sup></a></span>'
            )
            note += (
                '<p class="endnote"><span id="en11">'
                '<a href="ch1.xhtml#en_11">11.</a></span><span>Another note.</span></p>'
            )
        prefix = '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        bodies = {
            "OEBPS/ch1.xhtml": prefix + "<p>Original " + reference + " sentence.</p></body></html>",
            "OEBPS/ch2.xhtml": prefix + note + "</body></html>",
        }
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, "w") as output:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "OEBPS/content.opf":
                    data = data.decode().replace(
                        "</package>",
                        '<guide><reference type="endnotes" href="ch2.xhtml"/></guide></package>',
                    )
                output.writestr(item, bodies.get(item.filename, data))
        document = read_epub(path, "en", "zh")
        for chapter in document.chapters:
            for segment in chapter.segments:
                segment.target = "冻结译文。"
                for index, slot in enumerate(segment.epub_state.slots):
                    slot.target_value = segment.target if index == 0 else ""
        return path, document

    def _source(self, document):
        return document.chapters[0].segments[0].epub_state.rich_source

    def _segments_without_proof(self, document):
        values = [
            segment.model_dump(mode="json")
            for chapter in document.chapters
            for segment in chapter.segments
        ]
        for value in values:
            value["epub_state"]["rich_source"].pop("note_references", None)
        return values

    def test_binding_preserves_slots_runs_targets_and_old_note_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = self._fixture(directory)
            before = self._segments_without_proof(document)
            meta = copy.deepcopy(document.meta)
            bind_note_references(path, document.chapters)
            source = self._source(document)
            self.assertEqual(
                [reference.label for reference in source.note_references], ["10", "11"]
            )
            self.assertEqual(source.note_references[0].kind, "noteref")
            self.assertEqual(source.note_references[0].target_resource, "OEBPS/ch2.xhtml")
            link = next(
                mark for mark in source.marks if mark.id == source.note_references[0].mark_id
            )
            self.assertEqual(link.source_text, "10")
            self.assertEqual(len(source.note_references[0].content_marks), 2)
            backlink = document.chapters[1].segments[0].epub_state.rich_source.note_references[0]
            self.assertEqual(backlink.kind, "backlink")
            self.assertEqual(backlink.label, "10.")
            self.assertEqual(self._segments_without_proof(document), before)
            self.assertEqual(document.meta, meta)
            validate_note_references(path, document.chapters)
            original = source.model_dump(mode="json")
            bind_note_references(path, document.chapters)
            self.assertEqual(source.model_dump(mode="json"), original)

    def test_hydration_binds_proof_without_changing_frozen_text_or_slot_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = self._fixture(directory)
            before = [
                (segment.target, segment.epub_state.slots, segment.epub_state.rich_source.runs)
                for chapter in document.chapters
                for segment in chapter.segments
            ]
            meta = copy.deepcopy(document.meta)
            self.assertEqual(hydrate_rich_sources(path, document.chapters), [])
            after = [
                (segment.target, segment.epub_state.slots, segment.epub_state.rich_source.runs)
                for chapter in document.chapters
                for segment in chapter.segments
            ]
            self.assertEqual(after, before)
            self.assertEqual(document.meta, meta)
            self.assertEqual(len(self._source(document).note_references), 2)

    def test_changed_extra_and_partially_lost_proof_are_rejected(self):
        cases = {
            "label": {"label": "12"},
            "kind": {"kind": "backlink"},
            "target_resource": {"target_resource": "OEBPS/ch1.xhtml"},
            "target_path": {"target_path": (99,)},
            "content_marks": {"content_marks": ()},
        }
        for mode in [*cases, "extra", "missing"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, document = self._fixture(directory)
                bind_note_references(path, document.chapters)
                source = self._source(document)
                if mode == "extra":
                    source.note_references.append(source.note_references[0].model_copy(deep=True))
                elif mode == "missing":
                    source.note_references.pop()
                else:
                    source.note_references[0] = source.note_references[0].model_copy(
                        update=cases[mode]
                    )
                before = source.model_dump(mode="json")
                with self.assertRaisesRegex(ValueError, "note reference proof mismatch"):
                    validate_note_references(path, document.chapters)
                self.assertEqual(source.model_dump(mode="json"), before)

    def test_historical_empty_proof_remains_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = self._fixture(directory)
            self.assertEqual(self._source(document).note_references, [])
            validate_note_references(path, document.chapters)
            validate_note_references("missing.epub", document.chapters)
            self.assertEqual(self._source(document).note_references, [])

    def test_branches_unsupported_tags_and_internal_atoms_are_not_bound(self):
        for content in ("<sup>1</sup><span>0</span>", "<em>10</em>", "<sup>10</sup><br/>"):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path, document = self._fixture(directory, content=content, second=False)
                bind_note_references(path, document.chapters)
                self.assertEqual(self._source(document).note_references, [])

    def test_exact_label_and_source_run_coverage_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = self._fixture(directory)
            bind_note_references(path, document.chapters)
            state = document.chapters[0].segments[0].epub_state
            source = state.rich_source
            reference = source.note_references[0]
            link_path = next(mark.path for mark in source.marks if mark.id == reference.mark_id)
            relations: NoteRelations = {
                "version": 1,
                "targets": [],
                "markers": [
                    {
                        "resource_href": state.resource_href,
                        "path": [*state.block_path, *link_path],
                        "kind": reference.kind,
                        "label": reference.label,
                        "target_resource": reference.target_resource,
                        "target_path": list(reference.target_path),
                    }
                ],
            }
            self.assertEqual(
                note_references_for_source(
                    source, state.resource_href, state.block_path, relations
                ),
                [reference],
            )
            broken = source.model_copy(deep=True)
            next(run for run in broken.runs if reference.mark_id in run.marks).text = "wrong"
            self.assertEqual(
                note_references_for_source(
                    broken, state.resource_href, state.block_path, relations
                ),
                [],
            )
            relations["markers"][0]["label"] = " 10 "
            self.assertEqual(
                note_references_for_source(
                    source, state.resource_href, state.block_path, relations
                ),
                [],
            )

    def test_unsafe_zip_member_is_rejected_before_any_proof_is_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document = self._fixture(directory)
            before = self._segments_without_proof(document)
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("../outside.xhtml", "unsafe")
            with self.assertRaises(ZipSafetyError):
                bind_note_references(path, document.chapters)
            self.assertEqual(self._segments_without_proof(document), before)

    def test_source_without_rich_inventory_skips_proof_io(self):
        with tempfile.TemporaryDirectory() as directory:
            _, document = self._fixture(directory)
            for chapter in document.chapters:
                for segment in chapter.segments:
                    segment.epub_state.rich_source = None
            bind_note_references("missing.epub", document.chapters)
            validate_note_references("missing.epub", document.chapters)
