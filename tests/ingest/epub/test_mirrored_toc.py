from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

from tests.fixtures.fake_llm import fake_llm_dict
from trans_novel.config import Config
from trans_novel.epub.slots import distribute_slot_translation, target_slot_transport
from trans_novel.ingest import CANONICAL_TITLE_ID_META
from trans_novel.ingest.epub.reader import read_epub
from trans_novel.llm import FakeClient
from trans_novel.pipeline.contracts import NodeRequest
from trans_novel.pipeline.nodes.finish import TitlesNode
from trans_novel.pipeline.state import RunIdentity, RunStore


def _write_book(
    path: Path,
    *,
    root_title: str = "The Psychology of Money",
    document_title: str = (
        "The Psychology of Money: Timeless Lessons on Wealth, Greed, and Happiness"
    ),
) -> None:
    container = """<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
<rootfiles><rootfile full-path="O/content.opf"/></rootfiles></container>"""
    opf = f"""<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{document_title}</dc:title></metadata>
<manifest>
<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
<item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>
<item id="mirror" href="mirror.xhtml" media-type="application/xhtml+xml"/>
<item id="missing" href="missing.xhtml" media-type="application/xhtml+xml"/>
<item id="prefix" href="prefix.xhtml" media-type="application/xhtml+xml"/>
<item id="suffix" href="suffix.xhtml" media-type="application/xhtml+xml"/>
<item id="sibling" href="sibling.xhtml" media-type="application/xhtml+xml"/>
<item id="comment-tail" href="comment-tail.xhtml" media-type="application/xhtml+xml"/>
<item id="pi-tail" href="pi-tail.xhtml" media-type="application/xhtml+xml"/>
<item id="second-anchor" href="second-anchor.xhtml" media-type="application/xhtml+xml"/>
<item id="comment-only" href="comment-only.xhtml" media-type="application/xhtml+xml"/>
<item id="nested" href="nested.xhtml" media-type="application/xhtml+xml"/>
<item id="one" href="one.xhtml" media-type="application/xhtml+xml"/>
<item id="two" href="two.xhtml" media-type="application/xhtml+xml"/>
<item id="three" href="three.xhtml" media-type="application/xhtml+xml"/>
</manifest><spine toc="ncx"><itemref idref="title"/><itemref idref="mirror"/>
<itemref idref="missing"/><itemref idref="prefix"/><itemref idref="suffix"/>
<itemref idref="sibling"/><itemref idref="comment-tail"/><itemref idref="pi-tail"/>
<itemref idref="second-anchor"/><itemref idref="comment-only"/><itemref idref="nested"/>
<itemref idref="one"/><itemref idref="two"/><itemref idref="three"/></spine></package>"""
    ncx = f"""<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
<navPoint id="book"><navLabel><text>{root_title}</text></navLabel>
<content src="title.xhtml"/>
<navPoint id="one"><navLabel><text>Chapter One</text></navLabel>
<content src="one.xhtml#one"/></navPoint>
<navPoint id="two"><navLabel><text>Chapter Two</text></navLabel>
<content src="two.xhtml#two"/></navPoint>
<navPoint id="three"><navLabel><text>Chapter Three</text></navLabel>
<content src="three.xhtml#three"/></navPoint>
</navPoint></navMap></ncx>"""
    title = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Book</h1></body></html>"""
    mirror = """<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Contents</h1>
<p><a href="one.xhtml#one">Chapter One</a></p>
<p><a href="two.xhtml#two">Chapter Two</a></p>
<p><a href="three.xhtml#three">Chapter Three</a></p></body></html>"""
    missing = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p><a href="one.xhtml#one">Chapter One</a></p>
<p><a href="two.xhtml#two">Chapter Two</a></p></body></html>"""
    prefix = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li>For details see <a href="one.xhtml#one">Chapter One</a></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    suffix = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><a href="one.xhtml#one">Chapter One</a> for details</li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    sibling = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><a href="one.xhtml#one">Chapter One</a><span>New</span></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    comment_tail = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><!--note-->For details see <a href="one.xhtml#one">Chapter One</a></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    pi_tail = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><a href="one.xhtml#one">Chapter One</a><?note preserved?> for details</li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    second_anchor = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><a href="one.xhtml#one">Chapter One</a><a href="one.xhtml#one">Again</a></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    comment_only = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><!--note--><a href="one.xhtml#one">Chapter One</a></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    nested = """<html xmlns="http://www.w3.org/1999/xhtml"><body><ol>
<li><a href="one.xhtml#one">Chapter One</a><ol><li></li></ol><ul><li></li></ul></li>
<li><a href="two.xhtml#two">Chapter Two</a></li>
<li><a href="three.xhtml#three">Chapter Three</a></li></ol></body></html>"""
    one = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="one">Chapter One</h1><p>First body.</p></body></html>"""
    two = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="two">Chapter Two</h1><p>Second body.</p></body></html>"""
    three = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="three">Chapter Three</h1><p>Third body.</p></body></html>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("O/content.opf", opf)
        archive.writestr("O/toc.ncx", ncx)
        archive.writestr("O/title.xhtml", title)
        archive.writestr("O/mirror.xhtml", mirror)
        archive.writestr("O/missing.xhtml", missing)
        archive.writestr("O/prefix.xhtml", prefix)
        archive.writestr("O/suffix.xhtml", suffix)
        archive.writestr("O/sibling.xhtml", sibling)
        archive.writestr("O/comment-tail.xhtml", comment_tail)
        archive.writestr("O/pi-tail.xhtml", pi_tail)
        archive.writestr("O/second-anchor.xhtml", second_anchor)
        archive.writestr("O/comment-only.xhtml", comment_only)
        archive.writestr("O/nested.xhtml", nested)
        archive.writestr("O/one.xhtml", one)
        archive.writestr("O/two.xhtml", two)
        archive.writestr("O/three.xhtml", three)


class TestMirroredToc(unittest.TestCase):
    def test_subtitled_book_root_omission_marks_only_complete_owned_links(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "book.epub")
            _write_book(source)
            document = read_epub(str(source), "en", "zh")

        mirror = [
            segment
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.resource_href == "O/mirror.xhtml" and segment.kind != "heading"
        ]
        accepted_shapes = [
            segment
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.resource_href in {"O/comment-only.xhtml", "O/nested.xhtml"}
        ]
        rejected = [
            segment
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.resource_href
            in {
                "O/missing.xhtml",
                "O/prefix.xhtml",
                "O/suffix.xhtml",
                "O/sibling.xhtml",
                "O/comment-tail.xhtml",
                "O/pi-tail.xhtml",
                "O/second-anchor.xhtml",
            }
        ]
        self.assertEqual(
            [segment.meta.get("mirrored_toc_entry_id") for segment in mirror],
            ["O/toc.ncx:1", "O/toc.ncx:2", "O/toc.ncx:3"],
        )
        self.assertEqual(
            [segment.meta.get("mirrored_toc_entry_id") for segment in accepted_shapes],
            [
                "O/toc.ncx:1",
                "O/toc.ncx:2",
                "O/toc.ncx:3",
                "O/toc.ncx:1",
                "O/toc.ncx:2",
                "O/toc.ncx:3",
            ],
        )
        self.assertEqual(
            {segment.resource_href for segment in rejected},
            {
                "O/missing.xhtml",
                "O/prefix.xhtml",
                "O/suffix.xhtml",
                "O/sibling.xhtml",
                "O/comment-tail.xhtml",
                "O/pi-tail.xhtml",
                "O/second-anchor.xhtml",
            },
        )
        self.assertTrue(all("mirrored_toc_entry_id" not in segment.meta for segment in rejected))

    def test_book_root_omission_requires_matching_document_title(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "book.epub")
            _write_book(source, root_title="Different Book")
            document = read_epub(str(source), "en", "zh")

        mirror = [
            segment
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.resource_href == "O/mirror.xhtml"
        ]
        self.assertTrue(all("mirrored_toc_entry_id" not in segment.meta for segment in mirror))

    def test_book_root_omission_rejects_plain_title_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "book.epub")
            _write_book(source, root_title="The Psychology")
            document = read_epub(str(source), "en", "zh")

        mirror = [
            segment
            for chapter in document.chapters
            for segment in chapter.segments
            if segment.resource_href == "O/mirror.xhtml"
        ]
        self.assertTrue(all("mirrored_toc_entry_id" not in segment.meta for segment in mirror))

    def test_titles_node_overwrites_mirror_text_and_slot_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "book.epub")
            _write_book(source)
            document = read_epub(str(source), "en", "zh")
            store = RunStore(str(Path(directory, "state")))
            manifest = store.stage_document(
                document,
                RunIdentity(source_bytes_sha256="source", source_lang="en", target_lang="zh"),
            )
            manifest["initialized"] = True
            store.save_manifest(manifest)
            for chapter in document.chapters:
                for segment in chapter.segments:
                    if "mirrored_toc_entry_id" in segment.meta:
                        segment.assign_translation(
                            distribute_slot_translation(segment.epub_state, "ordinary translation")
                        )
                store.save_chapter(chapter)
                store.set_chapter_status(chapter.index, "done")

            def handler(messages, *_args):
                payload = json.loads(
                    messages[-1]["content"]
                    .split("【全书有序标题体系（JSON）】", 1)[-1]
                    .split("\n\n", 1)[0]
                )
                translations = {
                    "book": "书名",
                    "O/toc.ncx:0": "书名",
                    "O/toc.ncx:1": "第一章",
                    "O/toc.ncx:2": "第二章",
                    "O/toc.ncx:3": "第三章",
                }
                return json.dumps(
                    {
                        "titles": [
                            {
                                "id": item["id"],
                                "target": translations.get(item["id"], f"译-{item['source']}"),
                            }
                            for item in payload["titles"]
                        ]
                    },
                    ensure_ascii=False,
                )

            TitlesNode(
                client=FakeClient(handler=handler),
                config=Config.from_dict({"llm": fake_llm_dict()}),
                src="en",
                tgt="zh",
                glossary=SimpleNamespace(all_terms=list),
            ).execute(
                NodeRequest(
                    store=store,
                    node_id="titles",
                    key="titles",
                    ci=None,
                    scope="book",
                    input_path=str(source),
                )
            )

            mirrored_by_resource = {}
            for segment in (
                segment
                for chapter_meta in store.load_manifest()["chapters"]
                for segment in store.load_chapter(chapter_meta["index"]).segments
                if "mirrored_toc_entry_id" in segment.meta
            ):
                mirrored_by_resource.setdefault(segment.resource_href, []).append(segment)

            expected = {
                "O/mirror.xhtml": ["第一章", "第二章", "第三章"],
                "O/comment-only.xhtml": ["第一章", "第二章", "第三章"],
                "O/nested.xhtml": ["第一章", "第二章", "第三章"],
            }
            self.assertEqual(set(mirrored_by_resource), set(expected))
            for resource_href, titles in expected.items():
                with self.subTest(resource_href=resource_href):
                    segments = mirrored_by_resource[resource_href]
                    self.assertEqual([segment.target for segment in segments], titles)
                    self.assertEqual(
                        [segment.meta[CANONICAL_TITLE_ID_META] for segment in segments],
                        ["O/toc.ncx:1", "O/toc.ncx:2", "O/toc.ncx:3"],
                    )
                    self.assertEqual(
                        [
                            "".join(
                                item["value"] for item in target_slot_transport(segment.epub_state)
                            )
                            for segment in segments
                        ],
                        titles,
                    )


if __name__ == "__main__":
    unittest.main()
