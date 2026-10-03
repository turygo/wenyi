"""DTD 文档的生产序列化守恒，不接受解析器凭空添加节点。"""

import unittest

from lxml import etree

from trans_novel.assemble.epub.rendering.source_dom import (
    parse_source_markup,
    serialize_source_tree,
)
from trans_novel.assemble.epub.verification.slots import compare_dom


class TestSourceSerialization(unittest.TestCase):
    def test_dtd_document_preserves_nodes_encoding_and_document_siblings(self):
        for declaration in (False, True):
            with self.subTest(declaration=declaration):
                header = b'<?xml version="1.0" encoding="UTF-8"?>' if declaration else b""
                data = header + (
                    b'<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" '
                    b'"http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">'
                    b"<!--before--><?before keep?>"
                    b'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>T</title>'
                    b'<meta name="original" content="keep"/></head><body><p>Text</p></body></html>'
                    b"<?after keep?><!--after-->"
                )
                tree, mode = parse_source_markup(data)
                original_root = tree.getroot()
                original_count = len(list(original_root.iter()))
                encoded = serialize_source_tree(tree, data, mode)
                output, output_mode = parse_source_markup(encoded)
                self.assertEqual(output_mode, mode)
                self.assertEqual(output.docinfo.doctype, tree.docinfo.doctype)
                self.assertEqual(len(list(output.getroot().iter())), original_count)
                self.assertEqual(len(list(original_root.iter())), original_count)
                self.assertEqual(encoded.startswith(b"<?xml"), declaration)
                self.assertTrue(compare_dom(original_root, output.getroot(), {}))
                self.assertEqual(len(output.findall(".//{*}meta")), 1)
                for original, actual in zip(
                    [*original_root.itersiblings(preceding=True), *original_root.itersiblings()],
                    [
                        *output.getroot().itersiblings(preceding=True),
                        *output.getroot().itersiblings(),
                    ],
                    strict=True,
                ):
                    self.assertEqual(etree.tostring(actual), etree.tostring(original))
