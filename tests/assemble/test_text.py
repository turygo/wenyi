"""回填（TXT / EPUB）、报告、一致性 的测试（离线）。"""

from __future__ import annotations

import os
import tempfile
import unittest
import zipfile

from bs4 import BeautifulSoup

from tests.fixtures.books import (
    write_sample_txt,
)
from tests.fixtures.fake_llm import fake_llm_dict, routing_handler
from trans_novel.assemble import assemble
from trans_novel.config import Config
from trans_novel.ingest.segmenter import load_document
from trans_novel.llm import FakeClient
from trans_novel.pipeline import Application

_FB2_WITH_IMAGES = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:xlink="http://www.w3.org/1999/xlink">
<description><title-info>
  <book-title>Illustrated Book</book-title>
  <coverpage><image xlink:href="#cover.jpg"/></coverpage>
</title-info></description>
<body><section><title><p>Chapter</p></title>
  <image xlink:href="#inside.png"/><p>Illustrated text.</p>
</section></body>
<binary id="cover.jpg" content-type="image/jpeg">Y292ZXItYnl0ZXM=</binary>
<binary id="inside.png" content-type="image/png">aW5zaWRlLWJ5dGVz</binary>
</FictionBook>
"""


def _write_vertical_epub(path: str) -> None:
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    opf = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>縦書き小説</dc:title>
    <dc:language>ja</dc:language>
  </metadata>
  <manifest>
    <item id="style" href="style.css" media-type="text/css"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine page-progression-direction="rtl">
    <itemref idref="ch1"/>
  </spine>
</package>
"""
    ch1 = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" class="vrtl"><head>
<title>第一章</title><link rel="stylesheet" href="style.css"/>
</head><body>
<h1>第一章　出会い</h1>
<p>綾小路は教室の窓際に座っていた。</p>
</body></html>
"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/style.css", "html { writing-mode: vertical-rl; }")
        zf.writestr("OEBPS/ch1.xhtml", ch1)


def _config(state_dir: str):
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
    config.source_lang = "ja"
    config.state_dir = state_dir
    return config


def _run(input_path, state_dir):
    cfg = _config(state_dir)
    orch = Application(cfg, client=FakeClient(handler=routing_handler))
    store = orch.run(input_path)
    _stamp_formal_prereqs(store)
    return store, cfg


def _stamp_formal_prereqs(store):
    """Direct writer tests stamp title, QA, Repair, and report prerequisites."""
    from trans_novel.pipeline.state import NODE_DETERMINISTIC_QA, NODE_REPAIR, NodeState

    state = store.load_state()
    for node_id in ("titles", NODE_DETERMINISTIC_QA, NODE_REPAIR, "report"):
        state.nodes.setdefault(node_id, NodeState(node_id=node_id, status="succeeded"))
    store.save_state(state)
    return store


class TestAssembleText(unittest.TestCase):
    def test_fb2_images_and_cover_are_preserved_in_generated_epub(self):
        with tempfile.TemporaryDirectory() as directory:
            fb2 = os.path.join(directory, "illustrated.fb2")
            with open(fb2, "w", encoding="utf-8") as file:
                file.write(_FB2_WITH_IMAGES)
            store, _ = _run(fb2, os.path.join(directory, "state"))

            output = assemble(store, fb2, out_format="epub")

            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                cover_name = next(name for name in names if name.endswith("images/cover.jpg"))
                inside_name = next(name for name in names if name.endswith("images/inside.png"))
                chapter_name = next(name for name in names if name.endswith("/ch0.xhtml"))
                package_name = next(name for name in names if name.endswith("content.opf"))
                chapter = BeautifulSoup(archive.read(chapter_name), "html.parser")
                package = BeautifulSoup(archive.read(package_name), "xml")

                self.assertEqual(archive.read(cover_name), b"cover-bytes")
                self.assertEqual(archive.read(inside_name), b"inside-bytes")

        self.assertIsNotNone(chapter.find("img", src="images/inside.png"))
        self.assertIsNotNone(package.find("item", properties="cover-image"))

    def test_txt_input_to_txt(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _ = _run(txt, os.path.join(d, "state"))
            out = assemble(store, txt, out_format="txt")
            self.assertTrue(out.endswith(".txt"))
            self.assertEqual(os.path.basename(out), "novel.zh.txt")
            with open(out, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("润0", content)  # 译文已写入

    def test_txt_input_to_epub(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _ = _run(txt, os.path.join(d, "state"))
            out = assemble(store, txt, out_format="epub")
            self.assertTrue(out.endswith(".epub"))
            self.assertEqual(os.path.basename(out), "novel.zh.epub")
            self.assertTrue(zipfile.is_zipfile(out))
            # 重新解析生成的 EPUB，应能读出章节且含译文
            doc = load_document(out, "ja", "zh")
            self.assertGreaterEqual(len(doc.chapters), 2)
            alltext = "".join(s.source for c in doc.chapters for s in c.text_segments)
            self.assertIn("润", alltext)


class TestHeadingNumberInWriter(unittest.TestCase):
    """章节标题编号数字风格（阿拉伯 → 汉字）在槽位分配前统一。"""

    def test_txt_heading_normalized(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _ = _run(txt, os.path.join(d, "state"))
            ch = store.load_chapter(0)
            ch.segments[0].target = "第5章 相遇"
            store.save_chapter(ch)

            out_path = os.path.join(d, "novel.zh.txt")
            from trans_novel.assemble.text import assemble_text

            assemble_text(store, out_path)
            with open(out_path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("第五章 相遇", text)
            self.assertNotIn("第5章", text)


if __name__ == "__main__":
    unittest.main()
