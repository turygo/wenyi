"""无状态整块双语副本与原书、单语输出的独立匹配证明。"""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

from tests.fixtures.books import write_phase9_epub
from trans_novel.assemble.epub.verification import validate_epub_triplet

_RESOURCE = "OEBPS/text/chapter-2.xhtml"
_NS = "{http://www.w3.org/1999/xhtml}"


def _rewrite(source, output, data):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(output, "w") as archive:
        archive.comment = original.comment
        for info in original.infolist():
            archive.writestr(info, data if info.filename == _RESOURCE else original.read(info))


def _books(directory, order):
    original = directory / "original.epub"
    source, mono, bilingual = (
        directory / name for name in ("source.epub", "mono.epub", "bilingual.epub")
    )
    write_phase9_epub(str(original))
    with zipfile.ZipFile(original) as archive:
        root = etree.fromstring(archive.read(_RESOURCE))
    body = root.xpath("//*[@id='body-two']")[0]
    body.text = "Line one"
    br = etree.SubElement(body, _NS + "br")
    br.tail = "Line two"
    extra = etree.Element(_NS + "p", attrib={"id": "extra"})
    extra.text = "Another paragraph."
    body.addnext(extra)
    _rewrite(original, source, etree.tostring(root))
    body.text, br.tail, extra.text = "译文", None, "另一段"
    _rewrite(source, mono, etree.tostring(root))
    for target, text, tail in ((body, "Line one", "Line two"), (extra, "Another paragraph.", None)):
        copied = etree.Element(_NS + "p", attrib={"class": "tn-source"})
        copied.text = text
        if tail:
            etree.SubElement(copied, _NS + "br").tail = tail
        if order == "source_first":
            target.addprevious(copied)
        else:
            copied.tail, target.tail = target.tail, None
            target.addnext(copied)
    _rewrite(source, bilingual, etree.tostring(root))
    return source, mono, bilingual, root


class TestWholeBilingualBlocks(unittest.TestCase):
    def test_whole_source_copy_matches_both_orders_and_merged_target_prose(self):
        for order in ("source_first", "target_first"):
            with self.subTest(order=order), tempfile.TemporaryDirectory() as directory:
                source, mono, bilingual, _ = _books(Path(directory), order)
                report = validate_epub_triplet(source, mono, bilingual)
                self.assertTrue(report["structural_pass"], report)

    def test_whole_mode_rejects_text_subtree_duplicate_and_pair_order_tampering(self):
        for mutation in (
            "target_text",
            "source_text",
            "source_shape",
            "duplicate",
            "copy_order",
            "mixed_slot",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                source, mono, bilingual, root = _books(Path(directory), "target_first")
                copies = root.xpath("//*[@class='tn-source']")
                if mutation == "target_text":
                    root.xpath("//*[@id='body-two']")[0].text = "译"
                elif mutation == "source_text":
                    copies[0].text = "Wrong source"
                elif mutation == "source_shape":
                    copies[0].find(_NS + "br").tag = _NS + "em"
                elif mutation == "duplicate":
                    copies[0].addnext(deepcopy(copies[0]))
                elif mutation == "copy_order":
                    first, second = copies
                    first.addprevious(second)
                else:
                    copies[1].tag = _NS + "span"
                    copies[1].text = "Another"
                mutated = Path(directory) / "mutated.epub"
                _rewrite(bilingual, mutated, etree.tostring(root))
                report = validate_epub_triplet(source, mono, mutated)
                self.assertFalse(report["structural_pass"], report)
