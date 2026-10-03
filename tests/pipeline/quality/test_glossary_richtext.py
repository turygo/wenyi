"""术语维护必须保留独立译文格式并报告被跳过的命中。"""

import json
import os
import tempfile
import unittest

from trans_novel.epub.richtext import InlineAtom, InlineMark, InlineRun, RichSource, RichTarget
from trans_novel.epub.slots import EpubSegmentState
from trans_novel.glossary.store import GlossaryStore, GlossaryTerm
from trans_novel.ingest.models import Chapter, Segment
from trans_novel.pipeline.quality.glossary import fix_latin_residue, rewrite_targets
from trans_novel.pipeline.state import ChapterIndex, RunState, RunStore


def _segment(index: int, runs: list[InlineRun]) -> Segment:
    mark_ids = {mark for run in runs for mark in run.marks}
    atom_ids = {run.atom for run in runs if run.atom is not None}
    source = RichSource(
        marks=[
            InlineMark(id=mark, path=(i,), tag="b", source_text="source", kind="bold")
            for i, mark in enumerate(sorted(mark_ids))
        ],
        atoms=[
            InlineAtom(id=atom, path=(i,), kind="note") for i, atom in enumerate(sorted(atom_ids))
        ],
        runs=runs,
    )
    evidence = EpubSegmentState(
        resource_href="chapter.xhtml",
        resource_sha256="hash",
        block_fingerprint="hash",
        parse_mode="xml",
        slot_contract_sha256="hash",
        rich_source=source,
    )
    return Segment(
        index=index, source="source", epub_state=evidence, rich_target=RichTarget(runs=runs)
    )


class TestRichGlossary(unittest.TestCase):
    def _store(self, directory: str, segments: list[Segment]) -> RunStore:
        store = RunStore(os.path.join(directory, "state"))
        store.save_state(RunState(fmt="text", chapters=[ChapterIndex(index=0)]))
        store.save_chapter(Chapter(index=0, segments=segments))
        return store

    def _events(self, store: RunStore) -> list[dict]:
        with open(store.event_log_path, encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def test_safe_rewrite_and_conflict_are_recorded_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            safe = _segment(
                0,
                [
                    InlineRun(text="格雷厄姆"),
                    InlineRun(atom="n"),
                    InlineRun(text="整句强调。", marks=("b",)),
                ],
            )
            unsafe = _segment(1, [InlineRun(text="格雷", marks=("b",)), InlineRun(text="厄姆")])
            store = self._store(directory, [safe, unsafe])
            glossary = GlossaryStore(store.glossary_path)
            self.addCleanup(glossary.close)
            self.assertEqual(rewrite_targets(store, glossary, {"格雷厄姆": "本杰明·格雷厄姆"}), 1)
            chapter = store.load_chapter(0)
            self.assertEqual(chapter.segments[0].target, "本杰明·格雷厄姆整句强调。")
            self.assertEqual(chapter.segments[1].target, "格雷厄姆")
            self.assertEqual(
                [r.text for r in chapter.segments[0].rich_target.runs if r.marks], ["整句强调。"]
            )
            events = self._events(store)
            self.assertEqual(sum(e["event"] == "glossary_rewrite_applied" for e in events), 1)
            skipped = [e for e in events if e["event"] == "glossary_rewrite_skipped"]
            self.assertEqual(len(skipped), 1)
            self.assertTrue(skipped[0]["conflicts"])

    def test_latin_residue_preserves_link_or_emphasis_and_reports_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            safe = _segment(
                0,
                [InlineRun(text="Smith", marks=("b",)), InlineRun(text="走了。Smithson没有变化。")],
            )
            unsafe = _segment(1, [InlineRun(text="Sm", marks=("b",)), InlineRun(text="ith走了。")])
            store = self._store(directory, [safe, unsafe])
            glossary = GlossaryStore(store.glossary_path)
            self.addCleanup(glossary.close)
            glossary.upsert_term(GlossaryTerm(source="Smith", target="史密斯", locked=True))
            applied = fix_latin_residue(store, glossary)
            self.assertEqual(len(applied), 1)
            chapter = store.load_chapter(0)
            self.assertEqual(chapter.segments[0].target, "史密斯走了。Smithson没有变化。")
            self.assertEqual(chapter.segments[1].target, "Smith走了。")
            self.assertEqual(chapter.segments[0].rich_target.runs[0].marks, ("b",))
            self.assertTrue(
                any(e["event"] == "glossary_latin_residue_skipped" for e in self._events(store))
            )
