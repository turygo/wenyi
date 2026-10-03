"""标点局部编辑的幂等性、原始坐标和字面量保护契约。"""

import unittest

from trans_novel.postprocess.punct import normalize_zh, normalize_zh_parts
from trans_novel.postprocess.text_edits import (
    apply_text_edits,
    literal_ranges,
    punctuation_edits,
)


class TestPunctuationLiterals(unittest.TestCase):
    def test_mixed_collapses_and_removed_spaces_are_idempotent(self):
        cases = {
            "甲...…乙": "甲……乙",
            "甲....……乙": "甲……乙",
            "甲--—乙": "甲——乙",
            "甲。 。 。乙": "甲……乙",
            "甲. . .乙": "甲……乙",
            "甲。 . 。乙": "甲……乙",
            "甲。 . ,乙": "甲。。，乙",
            "甲。……乙": "甲。……乙",
            "甲.……乙": "甲。……乙",
            "甲-—乙": "甲-——乙",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                actual = apply_text_edits(original, punctuation_edits(original))
                self.assertEqual(actual, expected)
                self.assertEqual(normalize_zh(actual), actual)
                self.assertEqual(normalize_zh(original), actual)

    def test_mixed_punctuation_is_independent_of_fragmentation(self):
        text = '甲...…乙，  丙--—丁。 。 。戊"己"'
        expected = normalize_zh(text)
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                self.assertEqual(
                    "".join(normalize_zh_parts([text[:split], text[split:]])), expected
                )
        self.assertEqual("".join(normalize_zh_parts(list(text))), expected)

    def test_edits_keep_original_coordinates_and_unaffected_characters(self):
        text = "左甲...…乙右，  尾"
        edits = punctuation_edits(text)
        self.assertEqual(apply_text_edits(text, edits), "左甲……乙右，尾")
        previous = 0
        result = ""
        for edit in edits:
            result += text[previous : edit.start] + edit.replacement
            self.assertGreaterEqual(edit.start, previous)
            previous = edit.end
        result += text[previous:]
        self.assertEqual(result, normalize_zh(text))

    def test_complete_email_and_existing_literals_are_protected(self):
        literals = [
            "business@simonandschuster.com",
            "MacmillanSpecialMarkets@macmillan.com",
            "mailto:first.last+books@example.co.uk",
            "MAILTO:person@example.com",
            "https://example.com/a...b--c",
            "ftp://example.com/a...b",
            "www.example.com/a--b",
            "`x...--y`",
        ]
        for literal in literals:
            with self.subTest(literal=literal):
                text = "甲， " + literal + "，  乙...…丙"
                ranges = literal_ranges(text)
                self.assertIn(literal, [text[start:end] for start, end in ranges])
                edits = punctuation_edits(text)
                self.assertEqual(apply_text_edits(text, edits), "甲，" + literal + "，乙……丙")
                self.assertFalse(
                    any(
                        max(edit.start, start) < min(edit.end, end)
                        for edit in edits
                        for start, end in ranges
                    )
                )

    def test_collapses_do_not_cross_explicit_literal_ranges(self):
        text = "甲...…乙"
        actual = apply_text_edits(text, punctuation_edits(text, protected_ranges=[(1, 4)]))
        self.assertEqual(actual, "甲...……乙")
        self.assertEqual(
            apply_text_edits(actual, punctuation_edits(actual, protected_ranges=[(1, 4)])), actual
        )
