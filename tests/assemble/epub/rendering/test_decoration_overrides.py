"""生成首字覆盖保持原CSS、原文副本，并由独立资源门禁验证。"""

import hashlib
import io
import unittest
import zipfile
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from bs4 import BeautifulSoup
from lxml import etree

from tests.epub.test_stage1 import _CONTAINER, _OPF, _Store
from trans_novel.assemble.epub.rendering import assemble_source_epub
from trans_novel.assemble.epub.rendering.decorations import render_decoration_overrides
from trans_novel.assemble.epub.rendering.theme import ThemeBundle
from trans_novel.assemble.epub.rendering.theme.service import ThemeService
from trans_novel.assemble.epub.verification import verify_epub
from trans_novel.assemble.epub.verification.decorations import (
    prove_and_remove_decoration_style,
    prove_soup_decoration_style,
)
from trans_novel.epub.layout import LayoutAssignment, LayoutProfile, source_node_digest
from trans_novel.epub.richtext import InlineRun, RichTarget
from trans_novel.ingest.epub.reader import read_epub


class TestDecorationOverrides(unittest.TestCase):
    def test_complete_epub_round_trip_with_bilingual_source_copy(self):
        original = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
            b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
            b'<html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head>'
            b"<style>p::first-letter{font-size:2em;float:left}"
            b".unused a:hover img{animation:scale .2s}"
            b"@keyframes scale{from{width:100%}to{width:85%}}</style></head>"
            b'<body><p>Opening <i id="emphasis">phrase</i>.</p></body></html>'
        )
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.epub"
            with zipfile.ZipFile(source_path, "w") as archive:
                archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
                archive.writestr("META-INF/container.xml", _CONTAINER)
                archive.writestr("O/content.opf", _OPF)
                archive.writestr("O/c.xhtml", original)
            document = read_epub(str(source_path), "en", "zh")
            segment = document.chapters[0].segments[0]
            mark = segment.epub_state.rich_source.marks[0]
            segment.assign_translation(
                RichTarget(
                    runs=[
                        InlineRun(text="开篇"),
                        InlineRun(text="短语", marks=(mark.id,)),
                        InlineRun(text="。"),
                    ]
                )
            )
            store = _Store(document)
            theme = ThemeService(
                ThemeBundle(
                    general_css=b'[data-tn-role="body"]{font-family:serif}',
                    bilingual_css=None,
                    digest="test",
                    policy_version="test",
                    provenance=(),
                    note_markers=False,
                ),
                layout=LayoutProfile(
                    source_sha256=document.meta["epub_sha256"],
                    inventory_digest=hashlib.sha256(original).hexdigest(),
                    policy_version="1",
                    assignments=(
                        LayoutAssignment(
                            node_id="paragraph",
                            resource_href="O/c.xhtml",
                            path=(1, 0),
                            source_sha256=source_node_digest("p", {}, "Opening phrase."),
                            role="body",
                        ),
                    ),
                    provenance={},
                ),
            )
            for bilingual, themed in product((False, True), repeat=2):
                with self.subTest(bilingual=bilingual, themed=themed):
                    output = Path(directory) / f"output-{bilingual}-{themed}.epub"
                    plan = assemble_source_epub(
                        store,
                        str(source_path),
                        str(output),
                        target_lang="zh",
                        bilingual=bilingual,
                        theme=theme if themed else None,
                    )
                    report = verify_epub(
                        str(output),
                        source_path=str(source_path),
                        store=store,
                        mode="bilingual" if bilingual else "monolingual",
                        bilingual=bilingual,
                        target_lang="zh",
                        theme_plan=plan,
                    )
                    self.assertTrue(report["passed"], report["failures"])
                    with zipfile.ZipFile(output) as archive:
                        root = etree.fromstring(archive.read("O/c.xhtml"))
                    generated = root.xpath('//*[@data-tn-richtext="decorations"]')
                    self.assertEqual(len(generated), 1)
                    if bilingual:
                        source_nodes = root.xpath('//*[contains(@class,"tn-source")]')
                        self.assertEqual(len(source_nodes), 1)
                        self.assertEqual("".join(source_nodes[0].itertext()), "Opening phrase.")

    def test_override_is_source_bound_and_not_an_optional_theme(self):
        original = (
            b'<html xmlns="http://www.w3.org/1999/xhtml" lang="en"><head>'
            b"<style>p::first-letter{font-size:2em;float:left}</style></head>"
            b"<body><p>Original words.</p></body></html>"
        )
        translated = original.replace(b'lang="en"', b'lang="zh-Hans"')
        segment = SimpleNamespace(rich_target=RichTarget(), preserve_source=False, meta={})
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("c.xhtml", original)
        with zipfile.ZipFile(stream) as archive:
            rendered = render_decoration_overrides(
                archive,
                "c.xhtml",
                original,
                translated,
                [segment],
                "zh-Hans",
            )
            source = etree.fromstring(original)
            output = etree.fromstring(rendered)
            self.assertEqual(output.find(".//{*}style").text, source.find(".//{*}style").text)
            generated = output.xpath('//*[@data-tn-richtext="decorations"]')[0]
            self.assertIn("all:unset!important", generated.text)
            self.assertIn(":not(.tn-source *)", generated.text)
            soup = BeautifulSoup(rendered, "xml")
            prove_soup_decoration_style(archive, "c.xhtml", source, soup)
            self.assertEqual(len(soup.find_all("style")), 1)
            prove_and_remove_decoration_style(archive, "c.xhtml", source, output, required=True)
            self.assertEqual(len(output.findall(".//{*}style")), 1)

    def test_missing_tampered_or_duplicated_override_is_rejected(self):
        original = b"<html><head><style>p::first-letter{font-size:2em}</style></head><body><p>Text</p></body></html>"
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("c.xhtml", original)
        segment = SimpleNamespace(rich_target=RichTarget(), preserve_source=False, meta={})
        with zipfile.ZipFile(stream) as archive:
            rendered = render_decoration_overrides(
                archive, "c.xhtml", original, original, [segment], "zh"
            )
            for mutation in ("missing", "css", "attributes", "duplicate"):
                with self.subTest(mutation=mutation):
                    root = etree.fromstring(rendered)
                    style = root.xpath('//*[@data-tn-richtext="decorations"]')[0]
                    if mutation == "missing":
                        style.getparent().remove(style)
                    elif mutation == "css":
                        style.text += "p{font-style:normal}"
                    elif mutation == "attributes":
                        style.set("class", "fabricated")
                    else:
                        from copy import deepcopy

                        style.getparent().append(deepcopy(style))
                    with self.assertRaises(ValueError):
                        prove_and_remove_decoration_style(
                            archive,
                            "c.xhtml",
                            etree.fromstring(original),
                            root,
                            required=True,
                        )
