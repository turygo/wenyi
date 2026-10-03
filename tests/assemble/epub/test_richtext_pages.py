"""页码位置维护必须核对真实源身份，保持源证据和其他锚点。"""

import os
import tempfile
import unittest
import zipfile

from tests.fixtures.books import write_sample_epub
from trans_novel.assemble.epub.richtext_pages import settle_page_anchors
from trans_novel.epub.richtext import InlineRun, RichTarget, rich_source_digest
from trans_novel.ingest.epub.reader import read_epub


class TestSettlePageAnchors(unittest.TestCase):
    def _fixture(self, directory, *, page=True):
        original = os.path.join(directory, "original.epub")
        path = os.path.join(directory, "book.epub")
        write_sample_epub(original)
        semantics = 'epub:type="pagebreak"' if page else 'class="pagebreak"'
        body = (
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            'xmlns:epub="http://www.idpf.org/2007/ops"><body><p>'
            "<i>All original words.</i>"
            f'<span id="page" {semantics}/><a id="ordinary"/>'
            "</p></body></html>"
        )
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, "w") as target:
            for item in source.infolist():
                target.writestr(
                    item, body if item.filename == "OEBPS/ch1.xhtml" else source.read(item.filename)
                )
        document = read_epub(path, "en", "zh")
        segment = document.chapters[0].segments[0]
        source = segment.epub_state.rich_source
        mark = source.marks[0].id
        page_id, ordinary_id = [atom.id for atom in source.atoms]
        segment.assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="甲乙", marks=(mark,)),
                    InlineRun(atom=page_id),
                    InlineRun(text="丙", marks=(mark,)),
                    InlineRun(atom=ordinary_id),
                    InlineRun(text="丁。", marks=(mark,)),
                ]
            )
        )
        return path, document, segment, page_id, ordinary_id

    def _positions(self, segment):
        offset = 0
        positions = {}
        for run in segment.rich_target.runs:
            if run.atom:
                positions[run.atom] = offset
            offset += len(run.text)
        return positions

    def test_page_moves_and_ordinary_anchor_stays_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, segment, page, ordinary = self._fixture(directory)
            before = rich_source_digest(segment.epub_state.rich_source)
            text = segment.target
            self.assertEqual(settle_page_anchors(path, document.chapters), 1)
            self.assertEqual(self._positions(segment), {ordinary: 3, page: 5})
            self.assertEqual(segment.target, text)
            self.assertEqual(rich_source_digest(segment.epub_state.rich_source), before)
            self.assertEqual(settle_page_anchors(path, document.chapters), 0)

    def test_page_like_class_is_not_explicit_page_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, segment, _, _ = self._fixture(directory, page=False)
            before = segment.rich_target.model_copy(deep=True)
            self.assertEqual(settle_page_anchors(path, document.chapters), 0)
            self.assertEqual(segment.rich_target, before)

    def test_fake_atom_path_is_rejected_even_with_real_page_in_same_block(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, segment, _, _ = self._fixture(directory)
            atoms = segment.epub_state.rich_source.atoms
            atoms[1].path = atoms[0].path
            with self.assertRaisesRegex(ValueError, "source inventory mismatch"):
                settle_page_anchors(path, document.chapters)

    def test_preserved_source_has_no_page_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, document, segment, _, _ = self._fixture(directory)
            segment.preserve_source = True
            before = segment.rich_target.model_copy(deep=True)
            self.assertEqual(settle_page_anchors(path, document.chapters), 0)
            self.assertEqual(segment.rich_target, before)
