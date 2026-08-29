"""Crash-consistency tests for translation and polish checkpoint recovery."""

from __future__ import annotations

import os
import tempfile
import unittest

from trans_novel.ingest.models import Chapter, Segment
from trans_novel.pipeline import checkpoint
from trans_novel.pipeline.runstore import RunStore
from trans_novel.pipeline.state import (
    RUN_STATE_SCHEMA_VERSION,
    ChapterIndex,
    ChapterProgress,
    PolishBatch,
    RunIdentity,
    RunState,
)


def _store(d: str, *, segments: int = 2) -> RunStore:
    store = RunStore(os.path.join(d, "book"))
    store.save_state(
        RunState(
            run_state_schema=RUN_STATE_SCHEMA_VERSION,
            identity=RunIdentity(source_lang="ja", target_lang="zh"),
            title="T",
            fmt="text",
            source_lang="ja",
            target_lang="zh",
            chapters=[ChapterIndex(index=0, title="第一章")],
            progress={0: ChapterProgress()},
        )
    )
    store.save_chapter(
        Chapter(
            index=0,
            title="第一章",
            segments=[Segment(index=i, source=f"S{i}") for i in range(segments)],
        )
    )
    return store


def _reopen(run_dir: str) -> RunStore:
    """模拟崩溃后重新打开：锁内触发一次性迁移 + 检查点恢复。"""
    reopened = RunStore(run_dir)
    reopened.load_state()
    return reopened


class TestTranslateCheckpoint(unittest.TestCase):
    def test_crash_after_targets_before_marker_repairs_marker(self):
        """不变量 (a)：译文已落盘、标记缺失 → 恢复补回标记（续跑不再跳过润色）。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            ch = store.load_chapter(0)
            ch.text_segments[0].target = "译0"
            ch.text_segments[1].target = "译1"
            store.save_chapter(ch)
            checkpoint.begin_translate(store, 0, 0, 2)

            reopened = _reopen(store.run_dir)
            pg = reopened.load_progress(0)
            self.assertEqual([(p.start, p.count) for p in pg.pending_polish], [(0, 2)])
            self.assertFalse(os.path.isfile(reopened.journal_path))

    def test_crash_before_targets_rolls_back(self):
        """译文未落盘 → 未提交：清记录、不留标记，批按未译重跑。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            checkpoint.begin_translate(store, 0, 0, 2)

            reopened = _reopen(store.run_dir)
            self.assertEqual(reopened.load_progress(0).pending_polish, [])
            self.assertFalse(os.path.isfile(reopened.journal_path))
            self.assertTrue(all(not s.target for s in reopened.load_chapter(0).text_segments))

    def test_committed_translate_just_clears_journal(self):
        """译文与标记都已落盘 → 只清记录。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            ch = store.load_chapter(0)
            ch.text_segments[0].target = "译0"
            ch.text_segments[1].target = "译1"
            store.save_chapter(ch)
            pg = store.load_progress(0)
            pg.pending_polish = [PolishBatch(start=0, count=2)]
            store.save_progress(0, pg)
            checkpoint.begin_translate(store, 0, 0, 2)

            reopened = _reopen(store.run_dir)
            self.assertEqual(len(reopened.load_progress(0).pending_polish), 1)
            self.assertFalse(os.path.isfile(reopened.journal_path))


class TestPolishCheckpoint(unittest.TestCase):
    def _polished_state(self, d: str, polished: list[str]) -> RunStore:
        store = _store(d)
        pg = store.load_progress(0)
        pg.pending_polish = [PolishBatch(start=0, count=2)]
        store.save_progress(0, pg)
        ch = store.load_chapter(0)
        for i, t in enumerate(polished):
            ch.text_segments[i].target = t
        store.save_chapter(ch)
        return store

    def test_crash_after_polish_before_marker_clear_removes_marker(self):
        """不变量 (b)：润色结果已落盘、标记还在 → 恢复清标记（不重复润色）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._polished_state(d, ["润0", "润1"])
            checkpoint.begin_polish(store, 0, 0, 2, ["润0", "润1"])

            reopened = _reopen(store.run_dir)
            self.assertEqual(reopened.load_progress(0).pending_polish, [])
            self.assertFalse(os.path.isfile(reopened.journal_path))
            self.assertEqual(
                [s.target for s in reopened.load_chapter(0).text_segments],
                ["润0", "润1"],
                "已提交的润色结果必须保留",
            )

    def test_crash_before_polish_save_keeps_marker(self):
        """润色结果未落盘 → 未提交：保留标记，润色节点下次重跑该批。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._polished_state(d, ["译0", "译1"])  # 章节文件仍是润色前译文
            checkpoint.begin_polish(store, 0, 0, 2, ["润0", "润1"])

            reopened = _reopen(store.run_dir)
            pg = reopened.load_progress(0)
            self.assertEqual([(p.start, p.count) for p in pg.pending_polish], [(0, 2)])
            self.assertFalse(os.path.isfile(reopened.journal_path))

    def test_committed_polish_just_clears_journal(self):
        """标记已清 → 只清记录。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._polished_state(d, ["润0", "润1"])
            pg = store.load_progress(0)
            pg.pending_polish = []
            store.save_progress(0, pg)
            checkpoint.begin_polish(store, 0, 0, 2, ["润0", "润1"])

            reopened = _reopen(store.run_dir)
            self.assertEqual(reopened.load_progress(0).pending_polish, [])
            self.assertFalse(os.path.isfile(reopened.journal_path))


class TestCheckpointIdempotence(unittest.TestCase):
    def test_recovery_is_idempotent(self):
        """恢复可重复执行：第二次恢复不再改变状态（幂等重放）。"""
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            ch = store.load_chapter(0)
            ch.text_segments[0].target = "译0"
            ch.text_segments[1].target = "译1"
            store.save_chapter(ch)
            checkpoint.begin_translate(store, 0, 0, 2)

            first = _reopen(store.run_dir)
            self.assertEqual(len(first.load_progress(0).pending_polish), 1)
            # 第二次恢复：状态不变、无副作用
            second = _reopen(first.run_dir)
            self.assertEqual(len(second.load_progress(0).pending_polish), 1)
            self.assertFalse(os.path.isfile(second.journal_path))
            self.assertEqual(
                [s.target for s in second.load_chapter(0).text_segments],
                ["译0", "译1"],
            )

    def test_no_journal_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            store = _store(d)
            reopened = _reopen(store.run_dir)
            self.assertEqual(reopened.load_progress(0).pending_polish, [])
            self.assertFalse(os.path.isfile(reopened.journal_path))


if __name__ == "__main__":
    unittest.main()
