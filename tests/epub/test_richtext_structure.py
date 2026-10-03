"""独立译文格式与整块源文单元的端到端契约。"""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile

from lxml import etree

from tests.epub.test_stage1 import _CONTAINER, _OPF, _Store
from trans_novel.assemble.epub.rendering import assemble_source_epub
from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.ingest.epub.reader import read_epub


class TestRichTextSourceContracts(unittest.TestCase):
    def _book(self, xhtml):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = os.path.join(directory.name, "source.epub")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
            archive.writestr("META-INF/container.xml", _CONTAINER)
            archive.writestr("O/content.opf", _OPF)
            archive.writestr("O/c.xhtml", xhtml)
        return path

    def test_chinese_translation_keeps_semantically_aligned_italics(self):
        source = (
            b"<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            b"<p>On board the <i>Mustin</i>, radar was active.</p></body></html>"
        )
        path = self._book(source)
        document = read_epub(path, "en", "zh")
        segment = document.chapters[0].segments[0]
        translation = "在马斯廷号驱逐舰上，雷达正在运转。"
        mark = segment.epub_state.rich_source.marks[0]
        segment.assign_translation(
            RichTarget(
                runs=[
                    InlineRun(text="在"),
                    InlineRun(text="马斯廷号", marks=(mark.id,)),
                    InlineRun(text="驱逐舰上，雷达正在运转。"),
                ]
            )
        )
        output = path + ".italics.epub"
        self.addCleanup(os.unlink, output)

        assemble_source_epub(_Store(document), path, output, target_lang="zh-Hans")

        with zipfile.ZipFile(output) as archive:
            root = etree.fromstring(archive.read("O/c.xhtml"))
        paragraph = root.find(".//{http://www.w3.org/1999/xhtml}p")
        self.assertIsNotNone(paragraph)
        assert paragraph is not None
        italic = paragraph.find("{http://www.w3.org/1999/xhtml}i")
        self.assertIsNotNone(italic)
        assert italic is not None
        self.assertEqual("".join(paragraph.itertext()), translation)
        self.assertEqual(italic.text, "马斯廷号")

    def test_br_does_not_split_url_translation_unit(self):
        path = self._book(
            b"<html xmlns='http://www.w3.org/1999/xhtml'><body><p>GJOpen<br/>.com</p></body></html>"
        )
        document = read_epub(path, "en", "zh")
        self.assertEqual(len(document.chapters[0].segments), 1)
        segment = document.chapters[0].segments[0]
        self.assertEqual(segment.source, "GJOpen.com")
        atoms = segment.epub_state.rich_source.atoms
        self.assertEqual([atom.kind for atom in atoms], ["linebreak"])
        segment.assign_translation(
            RichTarget(
                runs=[InlineRun(text="GJOpen"), InlineRun(atom=atoms[0].id), InlineRun(text=".com")]
            )
        )
        output = path + ".url.epub"
        self.addCleanup(os.unlink, output)
        assemble_source_epub(_Store(document), path, output, target_lang="zh")
        with zipfile.ZipFile(output) as archive:
            root = etree.fromstring(archive.read("O/c.xhtml"))
        self.assertEqual(root.find(".//{*}p").find("{*}br").tail, ".com")

    def test_semantic_emphasis_on_anchor_alone_is_rejected(self):
        path = self._book(
            b"<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            b"<p><b>Important words.</b><a id='page'/></p></body></html>"
        )
        segment = read_epub(path, "en", "zh").chapters[0].segments[0]
        source = segment.epub_state.rich_source
        for empty_text in ["", " "]:
            with (
                self.subTest(text=empty_text),
                self.assertRaisesRegex(ValueError, "lost a required semantic mark"),
            ):
                segment.assign_translation(
                    RichTarget(
                        runs=[
                            InlineRun(text="重要表述。"),
                            InlineRun(text=empty_text, marks=(source.marks[0].id,))
                            if empty_text
                            else InlineRun(atom=source.atoms[0].id, marks=(source.marks[0].id,)),
                            *([InlineRun(atom=source.atoms[0].id)] if empty_text else []),
                        ]
                    )
                )
