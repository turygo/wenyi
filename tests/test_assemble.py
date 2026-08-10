"""回填（TXT / EPUB）、报告、一致性 的测试（离线）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from bs4 import BeautifulSoup, Tag

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import (
    write_cross_resource_toc_epub,
    write_epub_type_less_nav_epub,
    write_inline_sample_epub,
    write_nested_toc_epub,
    write_sample_epub,
    write_sample_txt,
)
from trans_novel.assemble.report import build_report
from trans_novel.assemble.writer import (
    _inject_bilingual_style,
    _render_chapter_html,
    _rewrite_html_document,
    assemble,
)
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.epub_reader import annotate_epub_resource
from trans_novel.ingest.models import Chapter
from trans_novel.ingest.segmenter import load_document
from trans_novel.llm import FakeClient
from trans_novel.pipeline.orchestrator import Orchestrator
from trans_novel.pipeline.runstore import RunStore

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
    return Config.from_dict(
        {
            "language": {"source": "ja", "target": "zh"},
            "llm": fake_llm_dict(),
            "pipeline": {"review": True, "polish": True, "backtranslate_sample": 0.0},
            "paths": {"state_dir": state_dir},
        }
    )


def _run(input_path, state_dir):
    cfg = _config(state_dir)
    orch = Orchestrator(cfg, client=FakeClient(handler=routing_handler))
    return orch.run(input_path), cfg


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

    def test_bilingual_rewrite_removes_temporary_file_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "book.epub")
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "ch0.xhtml",
                    "<html><head></head><body><p>text</p></body></html>",
                )

            with (
                patch(
                    "trans_novel.assemble.writer.os.replace",
                    side_effect=OSError("replace failed"),
                ),
                self.assertRaisesRegex(OSError, "replace failed"),
            ):
                _inject_bilingual_style(path, {"ch0.xhtml"}, "zh-Hans")

            self.assertFalse(os.path.exists(path + ".tmp"))

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


class TestAssembleEpub(unittest.TestCase):
    def test_rewrite_html_honors_declared_encoding_and_emits_utf8(self):
        source = (
            '<?xml version="1.0" encoding="Shift_JIS"?><html><body><p>日本語</p></body></html>'
        ).encode("shift_jis")

        output = _rewrite_html_document(
            source,
            lang="zh-Hans",
            force_horizontal=False,
        )
        decoded = output.decode("utf-8")

        self.assertIn("日本語", decoded)
        self.assertIn('encoding="utf-8"', decoded)
        self.assertIn('lang="zh-Hans"', decoded)

    def test_epub_export_restores_inline_image_from_persisted_meta(self):
        with tempfile.TemporaryDirectory() as d:
            epub = os.path.join(d, "inline.epub")
            write_inline_sample_epub(epub)
            store, _ = _run(epub, os.path.join(d, "state"))

            persisted = store.load_chapter(0)
            inline_segments = [s for s in persisted.segments if "epub_inline" in s.meta]
            self.assertEqual(len(inline_segments), 1)

            output = assemble(store, epub, out_format="epub")
            with zipfile.ZipFile(output) as archive:
                rendered = BeautifulSoup(
                    archive.read("OEBPS/ch1.xhtml"),
                    "html.parser",
                )
                image_data = archive.read("OEBPS/image.jpg")

        paragraph = rendered.find("p", class_="Textbody")
        self.assertIsInstance(paragraph, Tag)
        assert isinstance(paragraph, Tag)
        image = paragraph.find("img")
        self.assertIsInstance(image, Tag)
        assert isinstance(image, Tag)
        self.assertEqual(image.get("src"), "image.jpg")
        self.assertEqual(image_data, b"inline-image")
        self.assertIsNone(rendered.find(attrs={"data-tn-inline-id": True}))

    def test_epub_render_restores_inline_images_and_breaks(self):
        html = """<html><body>
<p class="Textbody"><img src="before.jpg"/>Avant<br/>Après<img src="after.jpg"/></p>
<p class="illustration"><img src="standalone.jpg"/></p>
</body></html>"""
        title, segments, template = annotate_epub_resource(
            html,
            0,
            "chapter.xhtml",
        )
        segments[0].target = "甲乙"
        segments[1].target = "丙丁"
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="chapter.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(_render_chapter_html(chapter), "html.parser")

        paragraph = rendered.find("p", class_="Textbody")
        self.assertIsInstance(paragraph, Tag)
        assert isinstance(paragraph, Tag)
        self.assertEqual(paragraph.get_text(), "甲乙丙丁")
        self.assertEqual(
            [image.get("src") for image in paragraph.find_all("img")],
            ["before.jpg", "after.jpg"],
        )
        self.assertIsNotNone(paragraph.find("br"))
        self.assertEqual(
            [
                child.name if getattr(child, "name", None) else str(child)
                for child in paragraph.children
            ],
            ["img", "甲乙", "br", "丙丁", "img"],
        )
        self.assertIsNone(rendered.find(attrs={"data-tn-inline-id": True}))
        standalone = rendered.find("p", class_="illustration")
        self.assertIsInstance(standalone, Tag)
        assert isinstance(standalone, Tag)
        standalone_image = standalone.find("img")
        self.assertIsInstance(standalone_image, Tag)
        assert isinstance(standalone_image, Tag)
        self.assertEqual(standalone_image.get("src"), "standalone.jpg")

    def test_epub_render_restores_footnote_markers_at_scaled_offsets(self):
        html = """<html><body><p class="Textbody"><sup id="note-wrapper-1" class="native"><a href="../notes.xhtml?edition=1#note-1" id="note-link-1" name="n1"><i>1</i></a></sup>Source <span class="xref superscript" title="footnote two"><a href="#footnote-2" id="note-link-2"><em>2</em></a></span>text<span style="vertical-align: sub; color: red" aria-label="endnote 3"><a href="#endnote-3" name="n3"><b>3</b></a></span></p></body></html>"""
        title, segments, template = annotate_epub_resource(html, 0, "chapter.xhtml")
        segments[0].target = "甲乙丙丁"
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="chapter.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(_render_chapter_html(chapter), "html.parser")

        paragraph = rendered.find("p", class_="Textbody")
        self.assertIsInstance(paragraph, Tag)
        assert isinstance(paragraph, Tag)
        self.assertEqual(paragraph.get_text(), "1甲乙丙2丁3")
        self.assertEqual(
            [child.name if isinstance(child, Tag) else str(child) for child in paragraph.children],
            ["sup", "甲乙丙", "span", "丁", "span"],
        )
        native = paragraph.find("sup", id="note-wrapper-1")
        self.assertIsInstance(native, Tag)
        assert isinstance(native, Tag)
        self.assertEqual(native.a.get("href"), "../notes.xhtml?edition=1#note-1")
        self.assertEqual(native.a.get("id"), "note-link-1")
        self.assertEqual(native.a.get("name"), "n1")
        self.assertIsNotNone(native.a.find("i"))
        inline = paragraph.find("span", class_="superscript")
        self.assertIsInstance(inline, Tag)
        assert isinstance(inline, Tag)
        self.assertEqual(inline.get("title"), "footnote two")
        self.assertEqual(inline.a.get("href"), "#footnote-2")
        self.assertEqual(inline.a.get("id"), "note-link-2")
        self.assertIsNotNone(inline.a.find("em"))
        ending = paragraph.find("span", attrs={"aria-label": "endnote 3"})
        self.assertIsInstance(ending, Tag)
        assert isinstance(ending, Tag)
        self.assertEqual(ending.get("style"), "vertical-align: sub; color: red")
        self.assertEqual(ending.a.get("href"), "#endnote-3")
        self.assertEqual(ending.a.get("name"), "n3")
        self.assertIsNotNone(ending.a.find("b"))
        self.assertIsNone(rendered.find(attrs={"data-tn-inline-id": True}))

    def test_epub_render_preserves_nested_list_links_and_blockquote_lines(self):
        html = """<html><body>
<ul><li><a href="#author">Author</a><ul>
<li><a href="chapter.xhtml#one">Chapter One</a></li>
<li><a href="chapter.xhtml#two">Chapter Two</a></li>
</ul></li></ul>
<blockquote><div>Dedication One</div><div>Dedication Two</div></blockquote>
</body></html>"""
        title, segments, template = annotate_epub_resource(html, 0, "contents.xhtml")
        for segment, target in zip(
            segments,
            ["作者", "第一章", "第二章", "献词一", "献词二"],
        ):
            segment.target = target
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="contents.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(_render_chapter_html(chapter), "html.parser")

        links = rendered.find_all("a")
        self.assertEqual(
            [link.get_text() for link in links],
            ["作者", "第一章", "第二章"],
        )
        self.assertEqual(
            [link.get("href") for link in links],
            ["#author", "chapter.xhtml#one", "chapter.xhtml#two"],
        )
        self.assertEqual(len(rendered.find_all("li")), 3)
        quote = rendered.find("blockquote")
        self.assertIsInstance(quote, Tag)
        assert isinstance(quote, Tag)
        self.assertEqual(
            [line.get_text() for line in quote.find_all("div", recursive=False)],
            ["献词一", "献词二"],
        )

    def test_epub_render_rebuilds_heading_breaks_from_translated_lines(self):
        html = """<html><body><h1>
Isaac Asimov<br/><br/>Tales of the Black Widowers<br/>
</h1></body></html>"""
        title, segments, template = annotate_epub_resource(html, 0, "title.xhtml")
        self.assertEqual(
            [segment.source for segment in segments],
            ["Isaac Asimov", "Tales of the Black Widowers"],
        )
        segments[0].target = "艾萨克·阿西莫夫"
        segments[1].target = "《黑鳏夫俱乐部故事》"
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="title.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(_render_chapter_html(chapter), "html.parser")
        heading = rendered.find("h1")
        self.assertIsInstance(heading, Tag)
        assert isinstance(heading, Tag)
        self.assertEqual(len(heading.find_all("br")), 3)
        self.assertIsNone(rendered.select_one("[data-tn-line]"))
        self.assertEqual(
            [
                child.name if isinstance(child, Tag) else str(child)
                for child in heading.children
                if isinstance(child, Tag) or str(child).strip()
            ],
            ["艾萨克·阿西莫夫", "br", "br", "《黑鳏夫俱乐部故事》", "br"],
        )

    def test_bilingual_break_lines_keep_valid_paragraph_structure(self):
        html = "<html><body><p>First<br/>Second</p></body></html>"
        title, segments, template = annotate_epub_resource(html, 0, "lines.xhtml")
        segments[0].target = "第一"
        segments[1].target = "第二"
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="lines.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(
            _render_chapter_html(chapter, bilingual=True),
            "html.parser",
        )
        paragraph = rendered.find("p")
        self.assertIsInstance(paragraph, Tag)
        assert isinstance(paragraph, Tag)
        self.assertIsNone(paragraph.find("p"))
        self.assertEqual(
            [source.get_text() for source in paragraph.select("span.tn-source")],
            ["First", "Second"],
        )
        self.assertEqual(
            [child.name if isinstance(child, Tag) else str(child) for child in paragraph.children],
            ["第一", "br", "span", "br", "第二", "br", "span"],
        )

    def test_bilingual_render_does_not_duplicate_inline_images(self):
        html = """<html><body>
<p><img src="illustration.jpg"/>Texte original.</p>
</body></html>"""
        title, segments, template = annotate_epub_resource(
            html,
            0,
            "chapter.xhtml",
        )
        segments[0].target = "译文。"
        chapter = Chapter(
            index=0,
            title=title,
            segments=segments,
            href="chapter.xhtml",
            template=template,
        )

        rendered = BeautifulSoup(
            _render_chapter_html(chapter, bilingual=True),
            "html.parser",
        )

        self.assertEqual(len(rendered.find_all("img")), 1)
        source = rendered.find(class_="tn-source")
        self.assertIsInstance(source, Tag)
        assert isinstance(source, Tag)
        self.assertIsNone(source.find("img"))

    def test_epub_template_rebuild(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            out = assemble(store, ep, out_format="epub")
            self.assertTrue(zipfile.is_zipfile(out))
            with zipfile.ZipFile(out) as z:
                html = z.read("OEBPS/ch1.xhtml").decode("utf-8")
            self.assertIn("润0", html)  # 译文已替换
            self.assertNotIn("data-tn-id", html)  # 占位标记已清除
            self.assertNotIn("綾小路は教室", html)  # 原文已被替换

    def test_vertical_epub_is_exported_as_horizontal_chinese(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "vertical.epub")
            _write_vertical_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                opf = z.read("OEBPS/content.opf").decode("utf-8")
                html = z.read("OEBPS/ch1.xhtml").decode("utf-8")
            self.assertIn("<dc:language>zh-Hans</dc:language>", opf)
            self.assertIn('page-progression-direction="ltr"', opf)
            self.assertIn("writing-mode: horizontal-tb", html)
            self.assertIn('lang="zh-Hans"', html)
            self.assertNotIn('class="vrtl"', html)


class TestTitleTranslation(unittest.TestCase):
    def test_manifest_keeps_book_title_and_translates_chapter_titles(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            # 书名不翻译；章节标题译出并写回 manifest（fake：标题0/1）
            m = store.load_manifest()
            self.assertNotIn("title_translated", m)
            self.assertTrue(all(c.get("title_translated") for c in m["chapters"]))
            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                opf = z.read("OEBPS/content.opf").decode("utf-8")
            self.assertIn("サンプル小説", opf)  # OPF 书名保持原文
            self.assertIn("<dc:language>zh-Hans</dc:language>", opf)
            self.assertEqual(os.path.basename(out), "novel.zh.epub")

    def test_rewrite_targets_propagates_to_titles(self):
        from trans_novel.agents.glossary_auditor import GlossaryAuditor

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, cfg = _run(txt, os.path.join(d, "state"))
            # 手动写入含变体的标题译名
            m = store.load_manifest()
            m["title_translated"] = "佳穂传"
            m["chapters"][0]["title_translated"] = "佳穂登场"
            store.save_manifest(m)
            g = GlossaryStore(store.glossary_path)
            GlossaryAuditor._rewrite_targets(store, g, {"佳穂": "佳穗"})
            g.close()
            m2 = store.load_manifest()
            self.assertNotIn("title_translated", m2)  # 书名译名字段被清理
            self.assertEqual(m2["chapters"][0]["title_translated"], "佳穗登场")  # 章名已规范

    def test_rewrite_nav_and_ncx_labels(self):
        from trans_novel.assemble.writer import _rewrite_toc

        nav = (
            b'<html xmlns:epub="http://www.idpf.org/2007/ops"><body>'
            b'<nav epub:type="toc"><ol>'
            b'<li><a href="ch1.xhtml">\xe7\xac\xac\xe4\xb8\x80\xe7\xab\xa0</a></li>'
            b"</ol></nav></body></html>"
        )
        out = _rewrite_toc(nav, {"ch1.xhtml": "第一章译名"}, is_ncx=False)
        self.assertIn("第一章译名", out.decode("utf-8"))

        ncx = (
            b'<?xml version="1.0"?><ncx><navMap><navPoint>'
            b"<navLabel><text>old</text></navLabel>"
            b'<content src="text/ch1.xhtml#x"/></navPoint></navMap></ncx>'
        )
        out2 = _rewrite_toc(ncx, {"ch1.xhtml": "第一章译名"}, is_ncx=True)
        dec = out2.decode("utf-8")
        self.assertIn("第一章译名", dec)
        self.assertNotIn(">old<", dec)


class TestHeadingNumberInWriter(unittest.TestCase):
    """章节标题编号数字风格（阿拉伯 → 汉字）在回填输出侧统一。"""

    def test_epub_heading_and_toc_normalized(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            ch = store.load_chapter(0)
            ch.segments[0].target = "第5章 迫击炮"  # 正文首段（heading 段）落成阿拉伯数字
            store.save_chapter(ch)
            m = store.load_manifest()
            m["chapters"][0]["title_translated"] = "第5章 迫击炮"  # 目录/nav 用的标题译名
            store.save_manifest(m)

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                html = z.read("OEBPS/ch1.xhtml").decode("utf-8")
            self.assertIn("<h1>第五章 迫击炮</h1>", html)
            self.assertNotIn("第5章", html)

    def test_txt_heading_normalized(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _ = _run(txt, os.path.join(d, "state"))
            ch = store.load_chapter(0)
            ch.segments[0].target = "第5章 相遇"
            store.save_chapter(ch)

            out_path = os.path.join(d, "novel.zh.txt")
            from trans_novel.assemble.writer import _assemble_text

            _assemble_text(store, out_path)
            with open(out_path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("第五章 相遇", text)
            self.assertNotIn("第5章", text)

    def test_toc_entries_title_translated_normalized_in_nav(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            with zipfile.ZipFile(ep, "a", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "OEBPS/nav.xhtml",
                    '<html xmlns:epub="http://www.idpf.org/2007/ops">'
                    '<body><nav epub:type="toc"><ol>'
                    '<li><a href="ch2.xhtml">old</a></li>'
                    "</ol></nav></body></html>",
                )
            store, _ = _run(ep, os.path.join(d, "state"))
            m = store.load_manifest()
            meta = m.setdefault("meta", {})
            meta["toc_entries"] = [{"href": "ch2.xhtml", "title_translated": "第8章 尾声"}]
            store.save_manifest(m)

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                nav = z.read("OEBPS/nav.xhtml").decode("utf-8")
            self.assertIn("第八章 尾声", nav)
            self.assertNotIn("第8章", nav)


class TestReport(unittest.TestCase):
    def test_report_summary(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _ = _run(txt, os.path.join(d, "state"))
            g = GlossaryStore(store.glossary_path)
            report = build_report(store, g)
            g.close()
            s = report["summary"]
            self.assertEqual(s["chapters_done"], s["chapters_total"])
            self.assertEqual(s["empty_targets"], 0)  # 全部段都有译文
            self.assertGreaterEqual(s["terms"], 1)


class TestConsistency(unittest.TestCase):
    def test_consistency_reports_issues(self):
        from trans_novel.agents.consistency import ConsistencyChecker

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, cfg = _run(txt, os.path.join(d, "state"))

            def handler(messages, agent, operation, json_mode):
                if "一致性审查员" in messages[0]["content"]:
                    return json.dumps(
                        {
                            "issues": [
                                {"type": "terminology", "detail": "X 译法不一致", "where": "第1章"}
                            ]
                        },
                        ensure_ascii=False,
                    )
                return "{}"

            g = GlossaryStore(store.glossary_path)
            client = FakeClient(handler=handler)
            checker = ConsistencyChecker(client, cfg)
            issues = checker.check(store, g)
            g.close()
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0]["type"], "terminology")
            self.assertEqual(client.calls[0]["agent"], "reviewer")
            self.assertEqual(client.calls[0]["operation"], "consistency.check")


class TestAssembleEpubPhysicalResourceGrouping(unittest.TestCase):
    """schema 2：物理资源按 href 聚合渲染，覆盖“一文件多逻辑章”和“一章跨多文件”。"""

    def test_two_logical_chapters_sharing_one_physical_file_both_translated(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "nested.epub")
            write_nested_toc_epub(ep, toc_kind="ncx")
            store, _ = _run(ep, os.path.join(d, "state"))
            m = store.load_manifest()
            self.assertEqual(len(m["chapters"]), 2)
            chapters = [store.load_chapter(c["index"]) for c in m["chapters"]]
            # 两个逻辑章共享同一物理资源
            self.assertEqual({ch.href for ch in chapters}, {"OEBPS/body.xhtml"})

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
                self.assertEqual(names.count("OEBPS/body.xhtml"), 1)  # 该物理文件只写一次
                html = z.read("OEBPS/body.xhtml").decode("utf-8")
            # 两章的全部译文都必须出现在这一份文件里，不能互相覆盖
            for ch in chapters:
                for seg in ch.segments:
                    self.assertIn(seg.target, html)
            self.assertNotIn("Part I intro.", html)
            self.assertNotIn("Part II intro.", html)
            self.assertNotIn("data-tn-id", html)

    def test_chapter_spanning_two_files_backfills_both(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "cross.epub")
            write_cross_resource_toc_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            m = store.load_manifest()
            self.assertEqual(len(m["chapters"]), 2)
            first_chapter = store.load_chapter(m["chapters"][0]["index"])
            resources_touched = {seg.resource_href for seg in first_chapter.segments}
            # 第一个逻辑章跨 one.xhtml 与 two.xhtml 两个物理文件
            self.assertEqual(resources_touched, {"OEBPS/one.xhtml", "OEBPS/two.xhtml"})

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                one_html = z.read("OEBPS/one.xhtml").decode("utf-8")
                two_html = z.read("OEBPS/two.xhtml").decode("utf-8")
            for seg in first_chapter.segments:
                target_html = one_html if seg.resource_href == "OEBPS/one.xhtml" else two_html
                self.assertIn(seg.target, target_html)
            self.assertNotIn("One.", one_html)
            self.assertNotIn("Two.", two_html)


class TestRewriteTocExactMode(unittest.TestCase):
    """`_rewrite_toc` 精确模式：按 toc_path + node_index 定位，同一文件中的多个 fragment 分别使用对应译名。"""

    def test_ncx_multi_fragment_nodes_get_distinct_titles(self):
        from trans_novel.assemble.writer import _rewrite_toc

        ncx = (
            b'<?xml version="1.0"?><ncx><navMap>'
            b"<navPoint><navLabel><text>old-a</text></navLabel>"
            b'<content src="chapter.xhtml#a"/></navPoint>'
            b"<navPoint><navLabel><text>old-b</text></navLabel>"
            b'<content src="chapter.xhtml#b"/></navPoint>'
            b"</navMap></ncx>"
        )
        entries = [
            {
                "toc_path": "OEBPS/toc.ncx",
                "node_index": 0,
                "raw_href": "chapter.xhtml#a",
                "title": "old-a",
                "title_translated": "译名甲",
            },
            {
                "toc_path": "OEBPS/toc.ncx",
                "node_index": 1,
                "raw_href": "chapter.xhtml#b",
                "title": "old-b",
                "title_translated": "译名乙",
            },
        ]
        out = _rewrite_toc(ncx, entries, is_ncx=True, toc_path="OEBPS/toc.ncx")
        soup = BeautifulSoup(out, "xml")
        nav_points = soup.find_all("navPoint")
        labels = [np.find("text").get_text() for np in nav_points]
        self.assertEqual(labels, ["译名甲", "译名乙"])
        # src 属性原样保留
        srcs = [np.find("content").get("src") for np in nav_points]
        self.assertEqual(srcs, ["chapter.xhtml#a", "chapter.xhtml#b"])

    def test_nav_multi_fragment_nodes_get_distinct_titles_and_href_mismatch_is_skipped(self):
        from trans_novel.assemble.writer import _rewrite_toc

        nav = (
            b'<html xmlns:epub="http://www.idpf.org/2007/ops"><body>'
            b'<nav epub:type="toc"><ol>'
            b'<li><a href="chapter.xhtml#a">old-a</a></li>'
            b'<li><a href="chapter.xhtml#b">old-b</a></li>'
            b"</ol></nav></body></html>"
        )
        entries = [
            {
                "toc_path": "OEBPS/nav.xhtml",
                "node_index": 0,
                "raw_href": "chapter.xhtml#a",
                "title_translated": "译名甲",
            },
            {
                # raw_href 与源文件实际 href 不一致，回填时须跳过，不能改错节点
                "toc_path": "OEBPS/nav.xhtml",
                "node_index": 1,
                "raw_href": "other.xhtml#b",
                "title_translated": "译名乙",
            },
        ]
        out = _rewrite_toc(nav, entries, is_ncx=False, toc_path="OEBPS/nav.xhtml")
        html = out.decode("utf-8")
        self.assertIn("译名甲", html)
        self.assertIn("old-b", html)
        self.assertNotIn("译名乙", html)


class TestAssembleEpubLegacySchema(unittest.TestCase):
    """schema 1 旧状态（章内嵌 template、无 resource_templates.json）导出兼容性。"""

    def test_legacy_chapter_template_without_resource_templates_still_exports(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "legacy.epub")
            write_sample_epub(ep)
            html = (
                "<html><body>"
                '<h1 data-tn-id="tn0_0">占位标题</h1>'
                '<p data-tn-id="tn0_1">占位正文</p>'
                "</body></html>"
            )
            title, segments, template = annotate_epub_resource(html, 0, "OEBPS/ch1.xhtml")
            segments[0].target = "旧状态标题"
            segments[1].target = "旧状态正文"
            chapter = Chapter(
                index=0,
                title=title,
                segments=segments,
                href="OEBPS/ch1.xhtml",
                template=template,
            )
            store = RunStore(os.path.join(d, "state"))
            store.save_chapter(chapter)
            manifest = {
                "title": "Legacy",
                "fmt": "epub",
                "source_path": ep,
                "source_lang": "ja",
                "target_lang": "zh",
                "meta": {},  # 无 epub_schema：模拟 schema 1 旧状态
                "chapters": [
                    {"index": 0, "title": chapter.title, "href": chapter.href, "status": "done"}
                ],
            }
            store.save_manifest(manifest)
            self.assertEqual(store.load_resource_templates(), {})  # 无 resource_templates.json

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                out_html = z.read("OEBPS/ch1.xhtml").decode("utf-8")
            self.assertIn("旧状态标题", out_html)
            self.assertIn("旧状态正文", out_html)
            self.assertNotIn("data-tn-id", out_html)


class TestTocRoutingAndSchemaFindings(unittest.TestCase):
    """评审修复：schema 2 缺少模板时须立即报错；目录回填须按 toc_entries 路由，不能只靠后缀或内容探测。"""

    def test_schema2_without_resource_templates_raises(self):
        """schema 2 状态缺少 resource_templates.json 或文件内容为空时，必须立即报错，不能静默回退并导出未翻译的原书。"""
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            chapter = Chapter(index=0, title="第一章", href="OEBPS/ch1.xhtml")
            store = RunStore(os.path.join(d, "state"))
            store.save_chapter(chapter)
            manifest = {
                "title": "Schema2NoTemplates",
                "fmt": "epub",
                "source_path": ep,
                "source_lang": "ja",
                "target_lang": "zh",
                "meta": {
                    "epub_schema": 2,
                    "toc_entries": [],
                },  # schema 2 状态，但缺少 resource_templates.json
                "chapters": [
                    {"index": 0, "title": chapter.title, "href": chapter.href, "status": "done"}
                ],
            }
            store.save_manifest(manifest)
            self.assertEqual(store.load_resource_templates(), {})

            with self.assertRaises(ValueError):
                assemble(store, ep, out_format="epub")

    def test_ncx_named_with_non_ncx_extension_still_backfills(self):
        """NCX 文件名不是 .ncx（如 toc.xml）时，须按 toc_entries 中的 toc_path + kind 路由并回填；
        否则，仅按后缀判断会漏掉该文件（解析端已能根据根节点将其识别为 NCX 目录）。"""
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "toc-xml.epub")
            write_nested_toc_epub(ep, ncx_filename="toc.xml")
            store, _ = _run(ep, os.path.join(d, "state"))
            m = store.load_manifest()
            meta = m.setdefault("meta", {})
            for entry in meta["toc_entries"]:
                entry["title_translated"] = f"译-{entry['title']}"
            store.save_manifest(m)

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                toc_xml = z.read("OEBPS/toc.xml").decode("utf-8")
            self.assertIn("译-PART I", toc_xml)
            self.assertIn("译-PART II", toc_xml)
            self.assertNotIn(">PART I<", toc_xml)

    def test_nav_without_epub_type_attribute_still_backfills(self):
        """NAV 缺少 epub:type="toc"（解析端 nav_toc_scopes 可兼容此情况）时，只要 toc_entries 中有
        toc_path 与该文件精确匹配的目录项，就必须执行精确回填，不能因 _is_nav 未识别出 NAV 而跳过。"""
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "navtypeless.epub")
            write_epub_type_less_nav_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            m = store.load_manifest()
            meta = m.setdefault("meta", {})
            for entry in meta["toc_entries"]:
                entry["title_translated"] = "译-One"
            store.save_manifest(m)

            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as z:
                nav_html = z.read("OEBPS/nav.xhtml").decode("utf-8")
            self.assertIn("译-One", nav_html)
            self.assertNotIn(">One<", nav_html)


if __name__ == "__main__":
    unittest.main()
