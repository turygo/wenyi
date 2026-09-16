"""回填（TXT / EPUB）、报告、一致性 的测试（离线）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from types import SimpleNamespace

from tests.fixtures.books import (
    write_sample_epub,
    write_sample_txt,
)
from tests.fixtures.fake_llm import fake_llm_dict, routing_handler
from trans_novel.assemble import assemble
from trans_novel.config import Config
from trans_novel.epub.slots import EpubSegmentState, EpubTextSlot, target_slot_transport
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import (
    CANONICAL_TITLE_ID_META,
    Chapter,
    ChapterProcessing,
    Segment,
)
from trans_novel.llm import FakeClient
from trans_novel.pipeline import Application
from trans_novel.pipeline.contracts import NodeRequest
from trans_novel.pipeline.nodes.finish import TitlesNode


def _config(state_dir: str):
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
    config.source_lang = "ja"
    config.state_dir = state_dir
    return config


def _run(input_path, state_dir):
    cfg = _config(state_dir)
    orch = Application(cfg, client=FakeClient(handler=routing_handler))
    store = orch.run(input_path)
    _stamp_formal_prereqs(store)
    return store, cfg


def _stamp_formal_prereqs(store):
    """Direct writer tests stamp title, QA, Repair, and report prerequisites."""
    from trans_novel.pipeline.state import NODE_DETERMINISTIC_QA, NODE_REPAIR, NodeState

    state = store.load_state()
    for node_id in ("titles", NODE_DETERMINISTIC_QA, NODE_REPAIR, "report"):
        state.nodes.setdefault(node_id, NodeState(node_id=node_id, status="succeeded"))
    store.save_state(state)
    return store


class _DuplicateTargetStore:
    def __init__(self):
        self.manifest = {
            "title": "Book",
            "chapters": [{"index": 0, "title": "Chapter", "toc_entry_id": "nav:1"}],
            "meta": {
                "epub_split_toc_path": "nav",
                "toc_entries": [
                    {
                        "entry_id": "nav:0",
                        "toc_path": "nav",
                        "node_index": 0,
                        "depth": 0,
                        "title": "Part",
                        "target_key": "part.xhtml",
                    },
                    {
                        "entry_id": "nav:1",
                        "toc_path": "nav",
                        "node_index": 1,
                        "parent_index": 0,
                        "depth": 1,
                        "title": "Chapter",
                        "target_key": "chapter.xhtml",
                    },
                    {
                        "entry_id": "nav:2",
                        "toc_path": "nav",
                        "node_index": 2,
                        "parent_index": 0,
                        "depth": 1,
                        "title": "Duplicate Chapter",
                        "target_key": "chapter.xhtml",
                    },
                    {
                        "entry_id": "nav:3",
                        "toc_path": "nav",
                        "node_index": 3,
                        "parent_index": 2,
                        "depth": 2,
                        "title": "Child",
                        "target_key": "child.xhtml",
                    },
                    {
                        "entry_id": "ncx:0",
                        "toc_path": "toc.ncx",
                        "node_index": 0,
                        "depth": 0,
                        "title": "NCX Chapter",
                        "target_key": "chapter.xhtml",
                    },
                ],
            },
        }
        self.chapter = Chapter(index=0, title="Chapter")
        self.events = []

    @staticmethod
    def pending_chapters():
        return []

    def load_manifest(self):
        return self.manifest

    def save_manifest(self, manifest):
        self.manifest = manifest

    def load_chapter(self, _index):
        return self.chapter

    @staticmethod
    def save_chapter(_chapter):
        pass

    def log_event(self, name, **payload):
        self.events.append((name, payload))


class _HeadingStore:
    def __init__(self):
        self.manifest = {
            "title": "Book",
            "chapters": [
                {"index": 3, "title": "Missing"},
                {"index": 2, "title": "Duplicate"},
                {"index": 0, "title": "Chapter One"},
                {"index": 1, "title": "Appendix"},
            ],
            "meta": {},
        }
        epub_heading = Segment(
            index=0,
            source="Appendix",
            kind="heading",
            target="旧附录",
            epub_state=EpubSegmentState(
                resource_href="appendix.xhtml",
                resource_sha256="source",
                block_path=(0,),
                block_fingerprint="block",
                parse_mode="xml",
                slots=[
                    EpubTextSlot(
                        id="first",
                        field="text",
                        source_value="Appen",
                        target_value="旧",
                    ),
                    EpubTextSlot(
                        id="second",
                        field="tail",
                        source_value="dix",
                        target_value="附录",
                    ),
                ],
                slot_contract_sha256="contract",
            ),
        )
        self.chapters = {
            0: Chapter(
                index=0,
                title="Chapter One",
                segments=[
                    Segment(
                        index=0,
                        source="  chapter   ONE ",
                        kind="heading",
                        target="旧标题",
                    ),
                    Segment(
                        index=1,
                        source="Chapter One: Details",
                        kind="heading",
                        target="小标题",
                    ),
                ],
            ),
            1: Chapter(
                index=1,
                title="Appendix",
                segments=[epub_heading],
                processing=ChapterProcessing(
                    action="preserve",
                    review_required=False,
                    reason="reference-only",
                    source_sha256="source",
                    strategy_version="chapter_semantics_v2",
                ),
            ),
            2: Chapter(
                index=2,
                title="Duplicate",
                segments=[
                    Segment(index=0, source="Duplicate", kind="heading", target="旧标题一"),
                    Segment(index=1, source="Duplicate", kind="heading", target="旧标题二"),
                ],
            ),
            3: Chapter(
                index=3,
                title="Missing",
                segments=[
                    Segment(index=0, source="Other heading", kind="heading", target="旧标题")
                ],
            ),
        }
        self.events = []

    @staticmethod
    def pending_chapters():
        return []

    def load_manifest(self):
        return self.manifest

    def save_manifest(self, manifest):
        self.manifest = manifest

    def load_chapter(self, index):
        return self.chapters[index]

    def save_chapter(self, chapter):
        self.chapters[chapter.index] = chapter

    def log_event(self, name, **payload):
        self.events.append((name, payload))


class TestTitleTranslation(unittest.TestCase):
    def test_translated_book_title_is_persisted_and_written_to_epub_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "novel.epub")
            write_sample_epub(ep)
            store, _ = _run(ep, os.path.join(d, "state"))
            manifest = store.load_manifest()
            self.assertEqual(manifest["title_translated"], "标题0")
            self.assertTrue(all(c.get("title_translated") for c in manifest["chapters"]))
            out = assemble(store, ep, out_format="epub")
            with zipfile.ZipFile(out) as archive:
                opf = archive.read("OEBPS/content.opf").decode("utf-8")
            self.assertIn("<dc:title>标题0</dc:title>", opf)
            self.assertNotIn("<dc:title>サンプル小説</dc:title>", opf)
            self.assertIn("<dc:language>zh-Hans</dc:language>", opf)
            self.assertEqual(os.path.basename(out), "novel.zh.epub")

    def test_duplicate_canonical_and_secondary_targets_share_first_request(self):
        captured = []

        def handler(messages, *_args):
            payload = json.loads(
                messages[-1]["content"]
                .split("【全书有序标题体系（JSON）】", 1)[-1]
                .split("\n\n", 1)[0]
            )
            captured.append((messages, payload))
            return json.dumps(
                {
                    "titles": [
                        {
                            "id": item["id"],
                            "target": (
                                "第5章 Chapter" if item["id"] == "nav:1" else f"译-{item['source']}"
                            ),
                        }
                        for item in payload["titles"]
                    ]
                },
                ensure_ascii=False,
            )

        store = _DuplicateTargetStore()
        TitlesNode(
            client=FakeClient(handler=handler),
            config=_config("state"),
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
                input_path="input.epub",
            )
        )

        self.assertEqual(len(captured), 1)
        requested = captured[0][1]["titles"]
        self.assertEqual(
            [item["id"] for item in requested],
            ["book", "nav:0", "nav:1", "nav:3"],
        )
        self.assertEqual(requested[2]["parent_id"], "nav:0")
        self.assertEqual(requested[2]["chapter"], 0)
        self.assertEqual(requested[3]["parent_id"], "nav:1")
        entries = {entry["entry_id"]: entry for entry in store.manifest["meta"]["toc_entries"]}
        self.assertEqual(entries["nav:1"]["title_translated"], "第五章 Chapter")
        self.assertEqual(entries["nav:2"]["title_translated"], "第五章 Chapter")
        self.assertEqual(entries["nav:3"]["title_translated"], "译-Child")
        self.assertEqual(entries["ncx:0"]["title_translated"], "第五章 Chapter")
        self.assertEqual(store.manifest["chapters"][0]["title_translated"], "第五章 Chapter")

    def test_unique_exact_heading_syncs_normal_and_preserved_chapters(self):
        def handler(messages, *_args):
            payload = json.loads(
                messages[-1]["content"]
                .split("【全书有序标题体系（JSON）】", 1)[-1]
                .split("\n\n", 1)[0]
            )
            return json.dumps(
                {
                    "titles": [
                        {
                            "id": item["id"],
                            "target": (
                                "第5章 Chapter One"
                                if item["id"] == "chapter:0"
                                else f"译-{item['source']}"
                            ),
                        }
                        for item in payload["titles"]
                    ]
                },
                ensure_ascii=False,
            )

        store = _HeadingStore()
        TitlesNode(
            client=FakeClient(handler=handler),
            config=_config("state"),
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
                input_path="input.epub",
            )
        )

        ordinary = store.load_chapter(0)
        self.assertEqual(ordinary.segments[0].target, "第五章 Chapter One")
        self.assertEqual(ordinary.segments[1].target, "小标题")
        self.assertEqual(
            ordinary.segments[0].meta[CANONICAL_TITLE_ID_META],
            "chapter:0",
        )
        preserved = store.load_chapter(1).segments[0]
        self.assertEqual(preserved.target, "译-Appendix")
        self.assertEqual(preserved.meta[CANONICAL_TITLE_ID_META], "chapter:1")
        self.assertEqual(
            "".join(item["value"] for item in target_slot_transport(preserved.epub_state)),
            "译-Appendix",
        )
        duplicate = store.load_chapter(2)
        self.assertEqual(
            [segment.target for segment in duplicate.segments],
            ["旧标题一", "旧标题二"],
        )
        self.assertEqual(store.load_chapter(3).segments[0].target, "旧标题")
        event = next(payload for name, payload in store.events if name == "titles_translated")
        self.assertEqual(event["skipped_chapter_heading_sync"], [2, 3])
        self.assertEqual(
            store.manifest["chapters"][2]["title_translated"],
            "第五章 Chapter One",
        )

    def test_invalid_batch_protocol_retries_identical_full_request_without_partial_save(self):
        valid = {
            "titles": [
                {"id": "book", "target": "书名"},
                {"id": "chapter:0", "target": "章节"},
            ]
        }
        invalid_cases = {
            "duplicate": {
                "titles": [
                    {"id": "book", "target": "书名"},
                    {"id": "book", "target": "重复"},
                ]
            },
            "missing": {"titles": [{"id": "book", "target": "书名"}]},
            "extra": {"titles": [*valid["titles"], {"id": "extra", "target": "额外"}]},
            "empty": {
                "titles": [
                    {"id": "book", "target": "书名"},
                    {"id": "chapter:0", "target": " "},
                ]
            },
        }

        for name, invalid in invalid_cases.items():
            with self.subTest(name=name):

                class Store:
                    def __init__(self):
                        self.manifest = {
                            "title": "Book",
                            "chapters": [{"index": 0, "title": "Chapter"}],
                            "meta": {},
                        }
                        self.saved = 0

                    @staticmethod
                    def pending_chapters():
                        return []

                    def load_manifest(self):
                        return self.manifest

                    def save_manifest(self, manifest):
                        self.saved += 1
                        self.manifest = manifest

                    @staticmethod
                    def load_chapter(_index):
                        return Chapter(index=0, title="Chapter")

                    @staticmethod
                    def save_chapter(_chapter):
                        pass

                    @staticmethod
                    def log_event(*_args, **_kwargs):
                        pass

                responses = [invalid, valid]
                requests = []

                def handler(messages, *_args, requests=requests, responses=responses):
                    requests.append(messages)
                    return json.dumps(responses.pop(0), ensure_ascii=False)

                store = Store()
                TitlesNode(
                    client=FakeClient(handler=handler),
                    config=_config("state"),
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
                        input_path="input.epub",
                    )
                )
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(store.saved, 1)
                self.assertEqual(store.manifest["title_translated"], "书名")

    def test_rewrite_targets_updates_book_and_chapter_titles(self):
        from trans_novel.pipeline.quality import rewrite_targets

        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            store, _cfg = _run(txt, os.path.join(d, "state"))
            manifest = store.load_manifest()
            manifest["title_translated"] = "佳穂传"
            manifest["chapters"][0]["title_translated"] = "佳穂登场"
            store.save_manifest(manifest)
            glossary = GlossaryStore(store.glossary_path)
            rewrite_targets(store, glossary, {"佳穂": "佳穗"})
            glossary.close()
            rewritten = store.load_manifest()
            self.assertEqual(rewritten["title_translated"], "佳穗传")
            self.assertEqual(rewritten["chapters"][0]["title_translated"], "佳穗登场")


if __name__ == "__main__":
    unittest.main()
