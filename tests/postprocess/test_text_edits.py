"""标点编辑的上下文、字面量与片段无关契约。"""

import unittest

from trans_novel.postprocess.punct import normalize_zh, normalize_zh_parts
from trans_novel.postprocess.text_edits import TextEdit, apply_text_edits, punctuation_edits


class TestTextEdits(unittest.TestCase):
    def test_fragmentation_does_not_change_apostrophe_or_ellipsis(self):
        for parts in [["don", "'", "t"], ["甲", ".", ".", ".", "乙"], ["甲, ", " 乙"]]:
            with self.subTest(parts=parts):
                result = normalize_zh_parts(parts)
                self.assertEqual("".join(result), normalize_zh("".join(parts)))
                self.assertEqual(normalize_zh_parts(result), result)
        self.assertEqual(normalize_zh_parts([".", ".", "."]), ["……", "", ""])

    def test_urls_literals_and_numbers_are_preserved(self):
        text = '中文https://example.com/a--b?q="x",以及`a...b`与9.11。'
        self.assertEqual(normalize_zh(text), text)

    def test_quote_pairing_and_idempotence(self):
        text = '"他说,don\'t..." -- 「走了」'
        once = normalize_zh(text)
        self.assertEqual(once, "“他说，don’t……”—— “走了”")
        self.assertEqual(normalize_zh(once), once)

    def test_ranges_are_explicit_and_validated(self):
        edits = punctuation_edits("甲, 乙...")
        self.assertEqual(apply_text_edits("甲, 乙...", edits), "甲，乙……")
        for invalid in [
            [TextEdit(-1, 0, "")],
            [TextEdit(2, 1, "")],
            [TextEdit(0, 2, "x"), TextEdit(1, 3, "y")],
        ]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                apply_text_edits("abc", invalid)
