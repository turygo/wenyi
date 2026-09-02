"""摄取与切分的冒烟测试。"""

from __future__ import annotations

import unittest

from trans_novel.ingest.models import KIND_HEADING, KIND_TEXT, Chapter, Segment
from trans_novel.ingest.segmenter import (
    _split_text,
    split_long_segments,
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


class TestSplitLongSegments(unittest.TestCase):
    def test_split_by_sentence_and_cont_flag(self):
        long_src = "第一句。" * 10  # 40 字符
        ch = Chapter(
            index=0,
            title="章",
            segments=[
                Segment(index=0, source="标题", kind=KIND_HEADING, anchor="a0"),
                Segment(index=1, source=long_src, kind=KIND_TEXT, anchor="a1"),
                Segment(index=2, source="短。", kind=KIND_TEXT, anchor="a2"),
            ],
        )
        split_long_segments([ch], max_chars=30)
        # 长段被拆成多段：首段保留 anchor，续段 cont=True 且无 anchor
        conts = [s.cont for s in ch.segments]
        self.assertIn(True, conts)
        long_parts = [s for s in ch.segments if not s.cont and s.anchor == "a1"]
        self.assertEqual(len(long_parts), 1)  # 首段唯一带 a1
        cont_parts = [s for s in ch.segments if s.cont]
        self.assertTrue(all(s.anchor is None for s in cont_parts))
        # index 连续重排
        self.assertEqual([s.index for s in ch.segments], list(range(len(ch.segments))))
        # 拼回去等于原文
        joined = "".join(s.source for s in ch.segments if s.anchor == "a1" or s.cont)
        self.assertEqual(joined, long_src)

    def test_no_split_when_short(self):
        ch = Chapter(
            index=0,
            title="章",
            segments=[Segment(index=0, source="短句。", kind=KIND_TEXT, anchor="a0")],
        )
        split_long_segments([ch], max_chars=100)
        self.assertEqual(len(ch.segments), 1)
        self.assertFalse(ch.segments[0].cont)

    def test_oversized_single_sentence_hard_split(self):
        chunks = _split_text("あ" * 50, 20)  # 无句末标点的超长串
        self.assertTrue(all(len(c) <= 20 for c in chunks))
        self.assertEqual("".join(chunks), "あ" * 50)

    def test_english_splits_on_sentence_punctuation(self):
        text = "Alpha beta gamma. Delta epsilon zeta! Eta theta iota?"
        chunks = _split_text(text, 25)
        self.assertEqual(chunks, ["Alpha beta gamma.", " Delta epsilon zeta!", " Eta theta iota?"])
        self.assertEqual("".join(chunks), text)

    def test_oversized_english_sentence_does_not_split_words(self):
        text = "alphabet bravo charlie delta"
        chunks = _split_text(text, 18)
        self.assertEqual(chunks, ["alphabet bravo", " charlie delta"])
        self.assertEqual("".join(chunks), text)
        self.assertNotIn("char", chunks[0])
        self.assertEqual(chunks[1].split()[0], "charlie")


if __name__ == "__main__":
    unittest.main()
