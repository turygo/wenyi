"""摄取与切分的冒烟测试。"""

from __future__ import annotations

import base64
import os
import tempfile
import unittest

from trans_novel.ingest.fb2 import read_fb2_binaries
from trans_novel.ingest.models import KIND_HEADING
from trans_novel.ingest.segmenter import (
    load_document,
)

_FB2_FLAT = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>平铺之书</book-title></title-info></description>
<body>
  <section><title><p>第一章</p></title><p>第一段。</p><p>第二段。</p></section>
  <section><title><p>第二章</p></title><p>仅一段。</p></section>
</body>
<body name="notes"><section><p>这是注释，应被跳过。</p></section></body>
</FictionBook>
"""

_FB2_BODY_TITLE = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>正文标题之书</book-title></title-info></description>
<body>
  <title><p>作者姓名</p><p>正文标题之书</p></title>
  <section><title><p>第一章</p></title><p>第一段。</p></section>
</body>
</FictionBook>
"""

# 嵌套：部 → 章（section 套 section）。容器节正文不得丢失。
_FB2_NESTED = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>嵌套之书</book-title></title-info></description>
<body>
  <section>
    <title><p>第一部</p></title>
    <section><title><p>第一章</p></title><p>一章首段。</p><p>一章次段。</p></section>
    <section><title><p>第二章</p></title><p>二章仅一段。</p></section>
  </section>
</body>
</FictionBook>
"""


# subtitle / poem / cite / text-author 等正文块不得丢字
_FB2_BLOCKS = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">
<description><title-info><book-title>块之书</book-title></title-info></description>
<body>
  <section>
    <title><p>第一章</p></title>
    <epigraph><p>题记一行。</p><text-author>题记作者</text-author></epigraph>
    <p>普通段落。</p>
    <subtitle>场景小标题</subtitle>
    <poem><title><p>诗名</p></title>
      <stanza><v>第一诗行。</v><v>第二诗行。</v></stanza>
      <text-author>诗人</text-author></poem>
    <cite><p>引文段落。</p><text-author>引文作者</text-author></cite>
    <p>结尾段落。</p>
  </section>
</body>
</FictionBook>
"""


_FB2_IMAGES = """\
<?xml version="1.0" encoding="utf-8"?>
<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"
             xmlns:xlink="http://www.w3.org/1999/xlink">
<description><title-info>
  <book-title>插图之书</book-title>
  <coverpage><image xlink:href="#cover.jpg"/></coverpage>
</title-info></description>
<body>
  <section><title><p>第一章</p></title>
    <image xlink:href="#inside.png"/>
    <p>带插图的正文。</p>
  </section>
</body>
<binary id="cover.jpg" content-type="image/jpeg">Y292ZXItYnl0ZXM=</binary>
<binary id="inside.png" content-type="image/png">aW5zaWRlLWJ5dGVz</binary>
</FictionBook>
"""


class TestFb2Ingest(unittest.TestCase):
    def _load(self, content: str):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "novel.fb2")
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return load_document(p, "ja", "zh")

    def test_flat_sections_and_notes_skipped(self):
        doc = self._load(_FB2_FLAT)
        self.assertEqual(doc.fmt, "fb2")
        self.assertEqual(doc.title, "平铺之书")
        self.assertEqual(len(doc.chapters), 2)  # notes body 不计入
        ch1 = doc.chapters[0]
        self.assertEqual(ch1.title, "第一章")
        self.assertEqual(ch1.segments[0].kind, KIND_HEADING)
        self.assertEqual(len(ch1.text_segments), 3)  # 标题 + 2 段
        # 注释正文不应出现在任何章中
        all_src = [s.source for ch in doc.chapters for s in ch.segments]
        self.assertNotIn("这是注释，应被跳过。", all_src)

    def test_namespace_variants_are_supported(self):
        variants = {
            "2.1": _FB2_FLAT.replace("fictionbook/2.0", "fictionbook/2.1"),
            "none": _FB2_FLAT.replace(' xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"', ""),
        }
        for name, content in variants.items():
            with self.subTest(namespace=name):
                doc = self._load(content)
                self.assertEqual(doc.title, "平铺之书")
                self.assertEqual([ch.title for ch in doc.chapters], ["第一章", "第二章"])
                self.assertEqual(
                    [s.source for s in doc.chapters[0].text_segments],
                    ["第一章", "第一段。", "第二段。"],
                )

    def test_single_quoted_windows_1251_declaration(self):
        content = """<?xml version='1.0' encoding='windows-1251'?>
<FictionBook>
  <description><title-info><book-title>Детство</book-title></title-info></description>
  <body><section><title><p>Глава</p></title><p>Текст</p></section></body>
</FictionBook>"""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "book.fb2")
            with open(path, "wb") as f:
                f.write(content.encode("windows-1251"))

            doc = load_document(path, "ru", "zh")

        self.assertEqual(doc.title, "Детство")
        self.assertEqual(doc.chapters[0].segments[0].source, "Глава")

    def test_body_title_becomes_a_separate_chapter(self):
        doc = self._load(_FB2_BODY_TITLE)
        self.assertEqual(len(doc.chapters), 2)
        title_page, first_chapter = doc.chapters
        self.assertEqual(title_page.title, "正文标题之书")
        self.assertEqual(
            [s.source for s in title_page.segments],
            ["作者姓名", "正文标题之书"],
        )
        self.assertTrue(all(s.kind == KIND_HEADING for s in title_page.segments))
        self.assertEqual([s.index for s in title_page.segments], [0, 1])
        self.assertEqual(
            [s.anchor for s in title_page.segments],
            ["tn0_0", "tn0_1"],
        )
        self.assertEqual(first_chapter.index, 1)
        self.assertEqual(first_chapter.title, "第一章")
        self.assertEqual(
            [s.anchor for s in first_chapter.segments],
            ["tn1_0", "tn1_1"],
        )

    def test_block_types_not_lost(self):
        doc = self._load(_FB2_BLOCKS)
        ch = doc.chapters[0]
        texts = [s.source for s in ch.segments]
        for expect in [
            "题记一行。",
            "题记作者",
            "普通段落。",
            "诗名",
            "第一诗行。",
            "第二诗行。",
            "诗人",
            "引文段落。",
            "引文作者",
            "结尾段落。",
        ]:
            self.assertIn(expect, texts)
        # subtitle 作为 heading
        headings = [s.source for s in ch.segments if s.kind == KIND_HEADING]
        self.assertIn("场景小标题", headings)

    def test_nested_sections_not_lost(self):
        doc = self._load(_FB2_NESTED)
        # 部标题成一章 + 两个子章，正文一段不丢
        titles = [ch.title for ch in doc.chapters]
        self.assertEqual(titles, ["第一部", "第一章", "第二章"])
        all_text = [
            s.source for ch in doc.chapters for s in ch.text_segments if s.kind != KIND_HEADING
        ]
        self.assertIn("一章首段。", all_text)
        self.assertIn("一章次段。", all_text)
        self.assertIn("二章仅一段。", all_text)

    def test_images_and_cover_are_recorded_without_persisting_binary_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "images.fb2")
            with open(path, "w", encoding="utf-8") as file:
                file.write(_FB2_IMAGES)

            document = load_document(path, "ru", "zh")
            binaries = read_fb2_binaries(path)

        self.assertEqual(document.meta["fb2_cover_image"], "cover.jpg")
        self.assertEqual(
            document.meta["fb2_resources"],
            [
                {"id": "cover.jpg", "content_type": "image/jpeg"},
                {"id": "inside.png", "content_type": "image/png"},
            ],
        )
        self.assertEqual(
            document.chapters[0].meta["fb2_images"],
            [{"id": "inside.png", "position": 1}],
        )
        self.assertNotIn(base64.b64encode(b"cover-bytes").decode(), str(document.meta))
        self.assertEqual(binaries["cover.jpg"], ("image/jpeg", b"cover-bytes"))
        self.assertEqual(binaries["inside.png"], ("image/png", b"inside-bytes"))


if __name__ == "__main__":
    unittest.main()
