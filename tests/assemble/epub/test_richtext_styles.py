"""源 CSS 证据与授权取消首字装饰的保守识别契约。"""

import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from tests.fixtures.books import write_sample_epub
from trans_novel.assemble.epub.rendering.theme.source_css import collect_source_stylesheets
from trans_novel.assemble.epub.richtext_styles import enrich_rich_sources
from trans_novel.ingest.epub.reader import read_epub


class TestRichTextStyles(unittest.TestCase):
    def _book(self, directory: str, body: str, css: str) -> str:
        original = os.path.join(directory, "original.epub")
        path = os.path.join(directory, "book.epub")
        write_sample_epub(original)
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "OEBPS/ch1.xhtml":
                    data = (
                        '<html xmlns="http://www.w3.org/1999/xhtml" '
                        'xmlns:epub="http://www.idpf.org/2007/ops">'
                        '<head><link rel="stylesheet" href="style.css"/></head>'
                        f"<body>{body}</body></html>"
                    ).encode()
                target.writestr(item, data)
            target.writestr("OEBPS/style.css", css)
        return path

    def _marks(self, document):
        return [
            mark
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.epub_state is not None and segment.epub_state.rich_source is not None
            for mark in segment.epub_state.rich_source.marks
        ]

    def test_dropcap_cancelled_without_removing_semantic_emphasis_or_page_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory,
                '<p><span class="drop">T</span>here is '
                '<b>important text</b><span epub:type="pagebreak" id="page1"/></p>',
                ".drop{font-size:1.5em;font-weight:bold}",
            )
            document = read_epub(path, "en", "zh")
            slots = [
                segment.epub_state.slots
                for chapter in document.chapters
                for segment in chapter.segments
                if segment.epub_state
            ]
            before = [[slot.model_dump() for slot in group] for group in slots]
            enrich_rich_sources(path, document.chapters)
            marks = {mark.source_text: mark for mark in self._marks(document)}
            self.assertEqual(marks["T"].kind, "decoration")
            self.assertEqual(marks["important text"].kind, "bold")
            self.assertTrue(marks["T"].style_evidence)
            self.assertEqual(before, [[slot.model_dump() for slot in group] for group in slots])
            atoms = [
                atom
                for chapter in document.chapters
                for segment in chapter.segments
                if segment.epub_state and segment.epub_state.rich_source
                for atom in segment.epub_state.rich_source.atoms
            ]
            self.assertTrue(any(atom.kind == "anchor" for atom in atoms))

    def test_single_letter_words_and_quoted_opening_letters_are_dropcaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory,
                '<p><span class="drop">I</span> have a plan.</p>'
                '<p><span class="drop">A</span> long story.</p>'
                '<p><span class="drop">“T</span>here is a story.”</p>'
                '<p>“<span class="drop">T</span>here is another.”</p>',
                ".drop{font-size:1.5em;font-weight:bold}",
            )
            document = read_epub(path, "en", "zh")
            enrich_rich_sources(path, document.chapters)
            self.assertEqual([mark.kind for mark in self._marks(document)], ["decoration"] * 4)

    def test_nonopening_letter_or_single_letter_entire_block_is_not_dropcap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory,
                '<p>Before <span class="drop">A</span> sentence.</p>'
                '<p><span class="drop">I</span></p>'
                '<p><span class="drop">Whole phrase</span> continues.</p>'
                '<p><strong class="drop">A whole phrase</strong> continues.</p>'
                "<p><span>A</span> has no local CSS.</p>",
                ".drop{font-size:1.5em;font-weight:bold}",
            )
            document = read_epub(path, "en", "zh")
            enrich_rich_sources(path, document.chapters)
            self.assertEqual(
                [mark.kind for mark in self._marks(document)],
                ["style", "style", "style", "bold", "style"],
            )

    def test_inherited_style_and_whole_phrase_are_not_dropcaps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory,
                '<p class="large"><span>T</span>here is text.</p>'
                '<p><span class="drop">This phrase</span> remains.</p>'
                '<p><b class="drop">T</b>here is semantic emphasis.</p>',
                ".large,.drop{font-size:2em;font-weight:bold}",
            )
            document = read_epub(path, "en", "zh")
            enrich_rich_sources(path, document.chapters)
            self.assertEqual(
                [mark.kind for mark in self._marks(document)], ["style", "style", "bold"]
            )

    def test_css_backed_spans_get_evidence_once_per_resource(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory,
                '<p><span class="bb">important</span> and <span class="font-sid">other</span>.</p>',
                ".bb{font-weight:bold}.font-sid{font-style:italic}",
            )
            document = read_epub(path, "en", "zh")
            with patch(
                "trans_novel.assemble.epub.richtext_styles.collect_source_stylesheets",
                wraps=collect_source_stylesheets,
            ) as collect:
                enrich_rich_sources(path, document.chapters)
            self.assertEqual(collect.call_count, 1)
            marks = self._marks(document)
            self.assertTrue(all(mark.style_evidence for mark in marks))
            self.assertTrue(all(mark.kind == "style" for mark in marks))

    def test_source_hash_mismatch_fails_without_mutating_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._book(
                directory, '<p><span class="bb">important</span>.</p>', ".bb{font-weight:bold}"
            )
            document = read_epub(path, "en", "zh")
            state = next(
                segment.epub_state
                for chapter in document.chapters
                for segment in chapter.segments
                if segment.epub_state
            )
            state.resource_sha256 = "wrong"
            with self.assertRaisesRegex(ValueError, "source resource mismatch"):
                enrich_rich_sources(path, document.chapters)
            self.assertEqual(self._marks(document)[0].style_evidence, [])

    def test_no_marks_does_not_open_source_archive(self):
        enrich_rich_sources("missing.epub", [])
