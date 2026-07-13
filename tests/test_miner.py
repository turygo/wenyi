"""源文侧术语候选挖掘（miner.py）的确定性算法测试（离线，零 LLM）。"""

from __future__ import annotations

import unittest

from trans_novel.glossary.miner import Candidate, mine_candidates, mine_candidates_en


class TestMineCandidatesEn(unittest.TestCase):
    def test_grabs_multi_word_and_allcap_drops_sentence_initial_common_word(self):
        """抓多词序列（Morris Chang）、全大写缩写（TSMC）、句中单词名（Phoebe）；
        丢弃只在句首出现过的常见大写词（The/But）；count 统计正确。"""
        text = (
            "The comet approached. Morris Chang met with engineers. But she refused.\n"
            "Phoebe watched Morris Chang carefully. Phoebe smiled and left.\n"
            "TSMC and TSMC engineers gathered around Phoebe as Morris Chang spoke.\n"
            "Phoebe waved to Phoebe again.\n"
        )
        result = mine_candidates_en([(1, text)])
        by_surface = {c.surface: c for c in result}

        self.assertIn("Morris Chang", by_surface)
        self.assertEqual(by_surface["Morris Chang"].count, 3)
        self.assertIn("TSMC", by_surface)
        self.assertIn("Phoebe", by_surface)
        self.assertEqual(by_surface["Phoebe"].count, 5)

        # 只在句首出现过的常见大写词：从未在句中位置以大写出现，丢弃
        self.assertNotIn("The", by_surface)
        self.assertNotIn("But", by_surface)

        # 按 count 降序排列
        counts = [c.count for c in result]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_multi_word_sequence_kept_even_at_single_occurrence(self):
        """多词序列（如 "University of Tokyo"）只出现 1 次也保留。"""
        text = "He once studied at the University of Tokyo before joining TSMC.\n"
        result = mine_candidates_en([(2, text)])
        by_surface = {c.surface: c for c in result}
        self.assertIn("University of Tokyo", by_surface)
        self.assertEqual(by_surface["University of Tokyo"].count, 1)

    def test_single_word_sentence_initial_only_is_dropped_below_threshold(self):
        """纯句首孤词（从未在句中位置大写出现）即使出现多次也丢弃；
        全大写缩写出现 <2 次同样丢弃。"""
        text = (
            "The morning was quiet. The city slept.\n"
            "DARPA funded one obscure pilot program only once.\n"
        )
        result = mine_candidates_en([(1, text)])
        by_surface = {c.surface: c for c in result}
        self.assertNotIn("The", by_surface)  # 句首孤词，从未在句中位置出现
        self.assertNotIn("DARPA", by_surface)  # 全大写缩写但只出现 1 次

    def test_count_and_chapter_tracking_across_chapters(self):
        """跨章合并计数：同一 surface 在不同章各出现一次时 count 累加、chapters 记两章。"""
        text1 = "Chris met Nova once. Nova had a long talk with Chris that day.\n"
        text2 = "Nova returned home. Chris was waiting for Nova there.\n"
        result = mine_candidates_en([(1, text1), (2, text2)])
        by_surface = {c.surface: c for c in result}
        self.assertEqual(by_surface["Nova"].chapters, [1, 2])
        self.assertGreaterEqual(by_surface["Nova"].count, 4)

    def test_context_capped_at_two_snippets_of_80_chars(self):
        """contexts 最多 2 条样例，每条 ≤80 字符。"""
        long_sentence = "The team met Nova in the extremely long corridor " + "x" * 60 + "."
        text = f"{long_sentence} Nova paused. Nova looked back. Nova left quietly.\n"
        result = mine_candidates_en([(1, text)])
        by_surface = {c.surface: c for c in result}
        nova = by_surface["Nova"]
        self.assertLessEqual(len(nova.contexts), 2)
        for ctx in nova.contexts:
            self.assertLessEqual(len(ctx), 80)

    def test_unicode_letters_in_names_not_truncated(self):
        """带附加符号的姓名（Brontë/García Márquez）不得被 ASCII 分词拦腰截断。"""
        text1 = (
            "Scholars praised Charlotte Brontë widely. "
            "Charlotte Brontë wrote Jane Eyre in solitude.\n"
        )
        text2 = "The award committee honored García Márquez for lifetime achievement.\n"
        r1 = {c.surface: c for c in mine_candidates_en([(1, text1)])}
        r2 = {c.surface: c for c in mine_candidates_en([(2, text2)])}

        self.assertIn("Charlotte Brontë", r1)
        self.assertEqual(r1["Charlotte Brontë"].count, 2)
        self.assertNotIn("Bront", r1)  # 不得被截断成 ASCII 前缀

        self.assertIn("García Márquez", r2)
        self.assertEqual(r2["García Márquez"].count, 1)  # 多词序列单次出现也保留

    def test_dialogue_quote_boundary_drops_stopwords_after_close_quote(self):
        """句末标点+闭引号才是句子边界：闭引号后紧跟的 Do/She/But 不得被误判成句中大写。
        即使句子切分正确识别了边界，代词/虚词类（_EN_STOP）本身也永不入候选。"""
        text = '"Are you sure?" Do you think… She says. "But no."\n'
        result = mine_candidates_en([(1, text)])
        by_surface = {c.surface: c for c in result}
        self.assertNotIn("Do", by_surface)
        self.assertNotIn("She", by_surface)
        self.assertNotIn("But", by_surface)

    def test_leading_stopword_stripped_from_multi_word_sequence(self):
        """多词序列首尾剥离停用词："The Cornwall Inn" 与 "the Cornwall Inn" 应汇聚成同一
        候选（剥离 The 后为 "Cornwall Inn"，或保留 The 视实现取舍，二者兼容断言）。"""
        text = "He visits the Cornwall Inn twice. The Cornwall Inn is old.\n"
        result = mine_candidates_en([(1, text)])
        surfaces = {c.surface for c in result}
        self.assertTrue(
            any(s in ("Cornwall Inn", "The Cornwall Inn") for s in surfaces),
            f"应产出 Cornwall Inn（剥离或保留首部 The 均可），实际={surfaces}",
        )

    def test_sentence_initial_repeated_names_known_limitation(self):
        """已知限制：句首过滤是全局判据（该词从未在句中位置大写出现则丢弃），纯人名罗列
        （每句仅一词、恰好总在句首）即使重复出现也会被漏掉——这是"句首孤词过滤"的
        代价，为保住对话体虚词过滤（本轮修复的核心目标）而接受，不为此破坏该判据。"""
        text = "Jack. Jeff. Stan. Jack. Jeff. Stan.\n"
        result = mine_candidates_en([(1, text)])
        surfaces = {c.surface for c in result}
        self.assertEqual(surfaces, set())

    def test_question_words_never_enter_candidates(self):
        """疑问词（What/How/Why/Who/Where）即使句中位置多次出现（满足其它两条过滤的放行
        条件），也永不入候选——Wedding People 全书复测暴露的 Top40 残留项。"""
        text = (
            "What do you want? She wondered What he meant. "
            '"How are you?" How odd, she thought. '
            "Why now? I wonder Why this happened. "
            "Who is he? They asked Who could help. "
            "Where are we? Nobody knew Where to go.\n"
        )
        result = mine_candidates_en([(1, text)])
        surfaces = {c.surface for c in result}
        for word in ("What", "How", "Why", "Who", "Where"):
            self.assertNotIn(word, surfaces, f"{word} 是疑问词，不得入候选")

    def test_candidate_is_plain_dataclass(self):
        c = Candidate(surface="X", count=2, chapters=[1], contexts=["ctx"])
        self.assertEqual(c.surface, "X")
        self.assertEqual(c.count, 2)


class TestMineCandidatesDualChannel(unittest.TestCase):
    """mine_candidates 入口：en 走"确定性大写通道 ∪ fast 档 LLM 通道"双通道合并。"""

    class _FakeMinerAgent:
        """最小 stub：满足 mine_candidates_llm 所需接口（.src/.tgt + _ask_json），
        固定返回同一份候选列表（每章调用一次），不发真实网络请求。"""

        def __init__(self, src: str, candidates: list[str]):
            self.src = src
            self.tgt = "zh"
            self._candidates = candidates

        def _ask_json(
            self, system, user, *, tier, key=None, default=None, max_tokens=None, operation=None
        ):
            return list(self._candidates)

    def test_en_merges_llm_lowercase_terms_missed_by_uppercase_channel(self):
        """大写通道抓不到反复出现的小写领域术语（lithography）——完全没有大写形式出现
        过，靠 LLM 通道补上；两通道都命中的专名（Morris Chang）合并成一条，保留大写
        通道的原样 surface，count 取两通道较大者。"""
        text = (
            "Morris Chang met engineers twice. Morris Chang discussed lithography today.\n"
            "The lithography process improved. Lithography remains critical.\n"
        )
        agent = self._FakeMinerAgent("en", ["lithography", "Morris Chang"])
        result = mine_candidates("en", [(1, text)], agent)
        by_lower = {c.surface.lower(): c for c in result}

        # 小写领域术语：确定性通道天生抓不到（从未以大写形式出现），靠 LLM 通道补上
        self.assertIn("lithography", by_lower)

        # 两通道都命中同一 surface（大小写不敏感）→ 合并成一条，不重复
        morris = [c for c in result if c.surface.lower() == "morris chang"]
        self.assertEqual(len(morris), 1, "两通道命中同一候选应合并为一条，不重复")
        # 保留大写通道产物的原样 surface（而非 LLM 通道可能返回的其它大小写形式）
        self.assertEqual(morris[0].surface, "Morris Chang")
        # count 取两通道较大者：确定性通道统计出 2（正文出现两次），LLM 通道每章贡献 1
        self.assertEqual(morris[0].count, 2)

    def test_non_en_language_unaffected_single_llm_channel(self):
        """非 en 语言：mine_candidates 行为不变，只走 LLM 通道（不触发确定性正则）。"""
        agent = self._FakeMinerAgent("ja", ["堀北"])
        result = mine_candidates("ja", [(1, "本文だよ。")], agent)
        self.assertEqual([c.surface for c in result], ["堀北"])


class TestMineCandidatesLlmConcurrency(unittest.TestCase):
    """LLM 通道并行化：输出与串行完全一致（按输入章序合并）、进度按完成数回调、
    单章异常照旧冒泡（不得被兜成空列表）。"""

    class _PerChapterAgent:
        """按 user prompt 里的章号返回该章候选；值为异常实例时抛出（模拟单章失败）。"""

        def __init__(self, per_chapter: dict):
            self.src, self.tgt = "ja", "zh"
            self._per = per_chapter

        def _ask_json(
            self, system, user, *, tier, key=None, default=None, max_tokens=None, operation=None
        ):
            import re

            ci = int(re.search(r"第(\d+)章", user).group(1))
            val = self._per[ci]
            if isinstance(val, Exception):
                raise val
            return list(val)

    _CHAPTERS = [(1, "一章目。"), (2, "二章目。"), (3, "三章目。")]
    _PER = {1: ["堀北", "綾小路"], 2: ["堀北"], 3: ["綾小路", "堀北"]}

    def _mine(self, concurrency: int, on_progress=None):
        from trans_novel.glossary.miner import mine_candidates_llm

        agent = self._PerChapterAgent(self._PER)
        return mine_candidates_llm(
            self._CHAPTERS, agent, concurrency=concurrency, on_progress=on_progress
        )

    def test_concurrent_output_identical_to_serial(self):
        serial = self._mine(1)
        parallel = self._mine(3)

        def key(cands):
            return [(c.surface, c.count, c.chapters) for c in cands]

        self.assertEqual(key(parallel), key(serial))
        by_surface = {c.surface: c for c in parallel}
        self.assertEqual(by_surface["堀北"].count, 3)
        self.assertEqual(by_surface["堀北"].chapters, [1, 2, 3])  # 输入章序，与完成顺序无关

    def test_on_progress_counts_completions(self):
        calls: list[tuple[int, int]] = []
        self._mine(2, on_progress=lambda i, n: calls.append((i, n)))
        self.assertEqual([i for i, _ in calls], [1, 2, 3])
        self.assertTrue(all(n == 3 for _, n in calls))

    def test_single_chapter_failure_propagates(self):
        from trans_novel.glossary.miner import mine_candidates_llm

        per = {**self._PER, 2: ValueError("boom")}
        agent = self._PerChapterAgent(per)
        with self.assertRaises(ValueError):
            mine_candidates_llm(self._CHAPTERS, agent, concurrency=2)


if __name__ == "__main__":
    unittest.main()
