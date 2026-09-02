"""标点和标题编号规范化测试。"""

import unittest

from trans_novel.postprocess.punct import normalize_heading_numbering, normalize_zh


class TestPunct(unittest.TestCase):
    def test_japanese_quotes(self):
        self.assertEqual(normalize_zh("「你好」"), "“你好”")
        self.assertEqual(normalize_zh("『书名』"), "‘书名’")

    def test_halfwidth_to_full_in_cjk(self):
        self.assertEqual(normalize_zh("他说,真的吗?"), "他说，真的吗？")

    def test_no_harm_to_english_numbers(self):
        self.assertEqual(normalize_zh("9.11 vs 9.8"), "9.11 vs 9.8")

    def test_ellipsis_and_dash(self):
        self.assertEqual(normalize_zh("等等...走了--他笑了"), "等等……走了——他笑了")


class TestHeadingNumbering(unittest.TestCase):
    def test_boundary_numbers(self):
        self.assertEqual(normalize_heading_numbering("第5章 迫击炮"), "第五章 迫击炮")
        self.assertEqual(normalize_heading_numbering("第10章"), "第十章")
        self.assertEqual(normalize_heading_numbering("第22章"), "第二十二章")
        self.assertEqual(normalize_heading_numbering("第100章"), "第一百章")
        self.assertEqual(normalize_heading_numbering("第105章"), "第一百零五章")
        self.assertEqual(normalize_heading_numbering("第110章"), "第一百一十章")
        self.assertEqual(normalize_heading_numbering("第1024章"), "第一千零二十四章")

    def test_fullwidth_digits(self):
        self.assertEqual(normalize_heading_numbering("第５章 全角"), "第五章 全角")

    def test_idempotent(self):
        once = normalize_heading_numbering("第5章 迫击炮")
        self.assertEqual(normalize_heading_numbering(once), once)

    def test_already_hanzi_unchanged(self):
        self.assertEqual(normalize_heading_numbering("第五章 迫击炮"), "第五章 迫击炮")

    def test_non_matching_text_unchanged(self):
        self.assertEqual(normalize_heading_numbering("迫击炮与大规模生产"), "迫击炮与大规模生产")

    def test_quantifier_variants(self):
        self.assertEqual(normalize_heading_numbering("第3部 序曲"), "第三部 序曲")
        self.assertEqual(normalize_heading_numbering("第7节"), "第七节")
        self.assertEqual(normalize_heading_numbering("第2卷"), "第二卷")
        self.assertEqual(normalize_heading_numbering("第9回"), "第九回")

    def test_mid_string_number_not_touched(self):
        self.assertEqual(
            normalize_heading_numbering("番外：来自第5章的回忆"),
            "番外：来自第5章的回忆",
        )

    def test_out_of_range_unchanged(self):
        self.assertEqual(normalize_heading_numbering("第0章"), "第0章")
        self.assertEqual(normalize_heading_numbering("第10000章"), "第10000章")

    def test_empty_and_none_like(self):
        self.assertEqual(normalize_heading_numbering(""), "")


if __name__ == "__main__":
    unittest.main()
