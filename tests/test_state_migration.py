"""V2 状态基础契约测试：一次性迁移、运行身份、节点指纹失效、回填就绪。

覆盖 wave 1 的可观察契约：
- V1 状态在锁内一次性迁移，V2 根 manifest 最后原子切换，中断可重试；
- 已完成的译文与全部恢复标记原样保留，Chapter.meta 不再持有流水线字段；
- 源文件/语言失配在任何翻译或回填复用前被拒绝；
- 节点输入指纹失配只失效该节点及其后代，不波及其余节点；
- 进程崩溃遗留的 running 节点恢复为 pending 并记录中断；
- 正式回填（writer.assemble）对不完整状态一律拒绝。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

from tests.fake_llm import fake_llm_dict, routing_handler
from tests.sample_data import write_sample_txt
from trans_novel.assemble.writer import assemble
from trans_novel.config import Config
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.llm import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.readiness import ReadinessError, assemble_readiness_problems
from trans_novel.pipeline.runstore import STATUS_DONE, STATUS_PENDING, RunStore
from trans_novel.pipeline.state import (
    NODE_ANALYZE,
    NODE_BACKTRANSLATE,
    NODE_BOOK_SYNOPSIS,
    NODE_CONSISTENCY_QA,
    NODE_DIGEST,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    NODE_NATURALIZE,
    NODE_POLISH,
    NODE_PREPARE,
    NODE_REPORT,
    NODE_REVIEW,
    NODE_TITLES,
    NODE_TRANSLATE,
    RUN_STATE_SCHEMA_VERSION,
    IdentityMismatchError,
    NodeState,
    PolishBatch,
    RunIdentity,
    chapter_node_key,
    input_fingerprint,
)


def _write_v1_state(
    run_dir: str,
    source_path: str,
    *,
    chapters: list[dict],
    chapter_files: dict[int, dict],
    analysis: dict | None = None,
    manifest_extra: dict | None = None,
) -> None:
    """直接写 V1 磁盘布局（模拟迁移前的旧版本产物）。"""
    os.makedirs(os.path.join(run_dir, "chapters"), exist_ok=True)
    manifest = {
        "title": "Book",
        "fmt": "text",
        "source_path": source_path,
        "source_lang": "ja",
        "target_lang": "zh",
        "initialized": True,
        "meta": {"epub_schema": 2, "toc_entries": [{"entry_id": "toc.ncx:0", "title": "T"}]},
        "chapters": chapters,
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    if analysis is not None:
        with open(os.path.join(run_dir, "analysis.json"), "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False)
    for ci, raw in chapter_files.items():
        with open(os.path.join(run_dir, "chapters", f"ch{ci}.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False)


def _sample_v1(run_dir: str, source_path: str) -> None:
    """一份带全部恢复标记的 V1 状态：完成章 + 附属章。"""
    _write_v1_state(
        run_dir,
        source_path,
        chapters=[
            {
                "index": 0,
                "title": "第一章",
                "href": "a.xhtml",
                "toc_entry_id": "toc.ncx:0",
                "status": "done",
                "review_pending": True,
            },
            {"index": 1, "title": "Notes", "href": "b.xhtml", "status": "done"},
        ],
        chapter_files={
            0: {
                "index": 0,
                "title": "第一章",
                "href": "a.xhtml",
                "template": None,
                "segments": [
                    {
                        "index": 0,
                        "source": "綾小路は教室にいた。",
                        "target": "绫小路在教室里。",
                        "kind": "text",
                        "cont": False,
                        "meta": {"epub_inline": {"version": 1}},
                    },
                    {
                        "index": 1,
                        "source": "空は青い。",
                        "target": None,
                        "kind": "text",
                        "cont": False,
                        "meta": {},
                    },
                ],
                "meta": {
                    "source_digest": "本章梗概",
                    "pending_polish": [{"start": 0, "count": 1}],
                    "naturalized": True,
                    "review_issues": [
                        {
                            "index": 0,
                            "type": "missing",
                            "detail": "漏了一句",
                            "suggestion": "补上",
                            "quote": "extra-field",
                        }
                    ],
                    "backtranslation_issues": [{"index": 1, "detail": "偏离"}],
                    "back_matter_mode": None,
                    "epub_split_strategy": "toc",
                    "toc_entry_id": "toc.ncx:0",
                },
            },
            1: {
                "index": 1,
                "title": "Notes",
                "href": "b.xhtml",
                "template": None,
                "segments": [
                    {
                        "index": 0,
                        "source": "Note.",
                        "target": "Note.",
                        "kind": "text",
                        "cont": False,
                        "meta": {},
                    }
                ],
                "meta": {"back_matter_mode": "skip", "epub_split_strategy": "toc"},
            },
        },
        analysis={"term_mining_done": True, "book_synopsis": "全书概览", "style": {"tone": "cool"}},
    )


def _config(state_dir: str) -> Config:
    config = Config.from_dict({"llm": fake_llm_dict(), "quality": "quality"})
    config.source_lang = "ja"
    config.state_dir = state_dir
    config.pipeline.backtranslate_sample = 0
    return config


class TestV1Migration(unittest.TestCase):
    def test_migration_preserves_targets_and_markers(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "novel.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("あ" * 500)
            run_dir = os.path.join(d, "state")
            _sample_v1(run_dir, src)

            store = RunStore(run_dir)
            state = store.load_state()  # 打开即迁移

            # V2 根状态：schema 标记 + 身份（含源文件哈希与解析后的语言）
            self.assertEqual(state.run_state_schema, RUN_STATE_SCHEMA_VERSION)
            self.assertEqual(state.identity.source_lang, "ja")
            self.assertEqual(state.identity.target_lang, "zh")
            with open(src, "rb") as f:
                src_bytes = f.read()
            self.assertEqual(
                state.identity.source_bytes_sha256,
                hashlib.sha256(src_bytes).hexdigest(),
            )

            # 章索引元数据保留（含译名/目录归属），status 不在其中
            self.assertEqual(state.chapters[0].toc_entry_id, "toc.ncx:0")
            self.assertNotIn("status", state.chapters[0].model_dump())

            # 进度：状态 / 恢复标记 / 章级产物全部保留
            pg0 = state.progress[0]
            self.assertEqual(pg0.status, STATUS_DONE)
            self.assertTrue(pg0.review_pending)
            self.assertEqual(pg0.source_digest, "本章梗概")
            self.assertTrue(pg0.naturalized)
            self.assertEqual(
                [p.model_dump() for p in pg0.pending_polish], [{"start": 0, "count": 1}]
            )
            self.assertEqual(pg0.review_issues[0].type, "missing")
            self.assertEqual(pg0.review_issues[0].to_dict()["quote"], "extra-field")  # 未知字段保留
            self.assertEqual(pg0.backtranslation_issues[0].to_dict()["detail"], "偏离")
            self.assertEqual(state.progress[1].back_matter_mode, "skip")

            # 完成标志归 V2；analysis.json 产物（概览/风格）原样保留
            self.assertTrue(state.analysis_flags.term_mining_done)
            analysis = store.load_analysis() or {}
            self.assertEqual(analysis["book_synopsis"], "全书概览")

            # Chapter.meta 只留 ingest 元数据，流水线字段已搬走；译文原样保留
            ch0 = store.load_chapter(0)
            self.assertEqual(ch0.meta, {"epub_split_strategy": "toc", "toc_entry_id": "toc.ncx:0"})
            self.assertEqual(ch0.segments[0].target, "绫小路在教室里。")
            self.assertIsNone(ch0.segments[1].target)

            # 旧 V1 文件保留为备份，未被修改
            with open(os.path.join(run_dir, "chapters", "ch0.json"), encoding="utf-8") as f:
                v1_chapter = json.load(f)
            self.assertIn("source_digest", v1_chapter["meta"])

            # 二次打开不再迁移（幂等）
            state2 = RunStore(run_dir).load_state()
            self.assertEqual(state2.run_state_schema, RUN_STATE_SCHEMA_VERSION)

    def test_migration_synthesizes_naturalize_only_when_v1_marker_set(self):
        """V1 naturalized 标记语义保留：已自然化章合成 succeeded；未自然化章留空，
        由当前策略（启用时）调度补跑。"""
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "novel.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("あ" * 500)
            run_dir = os.path.join(d, "state")
            _write_v1_state(
                run_dir,
                src,
                chapters=[
                    {"index": 0, "title": "第一章", "status": "done"},
                    {"index": 1, "title": "第二章", "status": "done"},
                ],
                chapter_files={
                    0: {
                        "index": 0,
                        "title": "第一章",
                        "template": None,
                        "segments": [
                            {
                                "index": 0,
                                "source": "綾小路は教室にいた。",
                                "target": "绫小路在教室里。",
                                "kind": "text",
                                "cont": False,
                                "meta": {},
                            }
                        ],
                        "meta": {"naturalized": True},
                    },
                    1: {
                        "index": 1,
                        "title": "第二章",
                        "template": None,
                        "segments": [
                            {
                                "index": 0,
                                "source": "空は青い。",
                                "target": "天空很蓝。",
                                "kind": "text",
                                "cont": False,
                                "meta": {},
                            }
                        ],
                        "meta": {"naturalized": False},
                    },
                },
            )
            state = RunStore(run_dir).load_state()
            self.assertEqual(state.nodes[chapter_node_key(NODE_NATURALIZE, 0)].status, "succeeded")
            self.assertNotIn(
                chapter_node_key(NODE_NATURALIZE, 1),
                state.nodes,
                "V1 naturalized=false 的章不得合成 succeeded（策略可调度补跑）",
            )
            self.assertFalse(state.progress[1].naturalized)
            # 可选质量环节无 V1 完成证据：合成 skipped（启用新策略时会被重新规划执行，
            # 不得凭“无待办标记”永久视为已满足）。
            for ci in (0, 1):
                for node_id in (NODE_POLISH, NODE_REVIEW, NODE_BACKTRANSLATE):
                    self.assertEqual(
                        state.nodes[chapter_node_key(node_id, ci)].status,
                        "skipped",
                        f"{node_id}:{ci} 应合成为 skipped（无 V1 完成证据）",
                    )

    def test_interrupted_migration_leaves_v1_usable_and_retries(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "novel.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("あ" * 500)
            run_dir = os.path.join(d, "state")
            _sample_v1(run_dir, src)
            # 模拟迁移在切换前崩溃：chapters_v2 只写了一部分，manifest 仍是 V1
            os.makedirs(os.path.join(run_dir, "chapters_v2"), exist_ok=True)
            with open(os.path.join(run_dir, "chapters_v2", "ch0.json"), "w", encoding="utf-8") as f:
                json.dump({"partial": True}, f, ensure_ascii=False)
            # 切换前 V1 原封未动：manifest 仍无 V2 标记
            with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as f:
                v1 = json.load(f)
            self.assertNotIn("run_state_schema", v1)

            # 重新打开 → 迁移重试并完成，译文不丢
            store = RunStore(run_dir)
            state = store.load_state()
            self.assertEqual(state.run_state_schema, RUN_STATE_SCHEMA_VERSION)
            self.assertEqual(store.load_chapter(0).segments[0].target, "绫小路在教室里。")
            self.assertEqual(store.load_chapter(1).segments[0].target, "Note.")


class TestRunIdentity(unittest.TestCase):
    def test_source_mismatch_rejected_before_translation_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            # 同路径原地改写源文件（basename 不变 → 命中同一 run slug）：
            # 正文内容不同 → 拒绝复用，而非静默续跑。
            with open(txt, encoding="utf-8") as f:
                original = f.read()
            with open(txt, "w", encoding="utf-8") as f:
                f.write(
                    original.replace(
                        "綾小路は教室の窓際に座っていた。", "別の人物は校舎の屋上に立っていた。"
                    )
                )
            with self.assertRaises(IdentityMismatchError):
                Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)

            # 恢复原文 → 正常续跑
            with open(txt, "w", encoding="utf-8") as f:
                f.write(original)
            store2 = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            self.assertTrue(all(store2.chapter_status(i) == STATUS_DONE for i in range(2)))

    def test_language_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            with self.assertRaisesRegex(IdentityMismatchError, "源语言不一致"):
                with open(txt, "rb") as f:
                    txt_bytes = f.read()
                store.verify_identity(
                    source_bytes_sha256=hashlib.sha256(txt_bytes).hexdigest(),
                    source_lang="en",
                    target_lang="zh",
                )
            # 一致的身份通过
            with open(txt, "rb") as f:
                txt_bytes = f.read()
            store.verify_identity(
                source_bytes_sha256=hashlib.sha256(txt_bytes).hexdigest(),
                source_lang="ja",
                target_lang="zh",
            )

    def test_mismatch_fails_before_assemble_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            txt = os.path.join(d, "novel.txt")
            write_sample_txt(txt)
            cfg = _config(os.path.join(d, "state"))
            store = Application(cfg, client=FakeClient(handler=routing_handler)).run(txt)
            other = os.path.join(d, "other.txt")
            with open(other, "w", encoding="utf-8") as f:
                f.write("別の本" * 200)
            with self.assertRaises(IdentityMismatchError):
                assemble(store, other, out_format="txt")


class TestNodeFingerprints(unittest.TestCase):
    def _store_with_run(self, d: str) -> RunStore:
        store = RunStore(os.path.join(d, "state"))
        identity = RunIdentity(
            source_bytes_sha256="h" * 64,
            run_input_schema_version=1,
            source_lang="en",
            target_lang="zh",
        )
        state = store.stage_document(
            Document(
                title="T",
                fmt="text",
                source_lang="en",
                target_lang="zh",
                source_path="",
                chapters=[Chapter(index=0, title="C0", segments=[])],
            ),
            identity,
        )
        store.save_manifest(state)
        return store

    def test_mismatch_invalidates_only_affected_node_and_descendants(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store_with_run(d)
            fp_d = input_fingerprint("src text", "en")
            fp_m = input_fingerprint(["src text"], "en", 4)
            store.record_node_fingerprint(chapter_node_key(NODE_DIGEST, 0), fp_d)
            store.record_node_fingerprint(NODE_MINE_TERMS, fp_m)
            store.record_node_fingerprint(
                NODE_BOOK_SYNOPSIS, input_fingerprint(["digest"], "cool", "en", "zh")
            )
            store.record_node_fingerprint(
                chapter_node_key(NODE_TRANSLATE, 0), input_fingerprint("cfg")
            )
            pg = store.load_progress(0)
            pg.source_digest = "digest"
            store.save_progress(0, pg)
            state = store.load_state()
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            store.save_analysis({"book_synopsis": "syn"})

            # 只失配 book_synopsis → 失效它自己 + 全部下游（translate fan-in →
            # 章链 → titles → consistency/report/assemble 传递闭包）；digest/mine
            # 不受波及。
            inv = store.reconcile_fingerprints(
                {
                    chapter_node_key(NODE_DIGEST, 0): fp_d,
                    NODE_MINE_TERMS: fp_m,
                    NODE_BOOK_SYNOPSIS: input_fingerprint(["different"], "cool", "en", "zh"),
                }
            )
            self.assertEqual(
                inv,
                {
                    NODE_BOOK_SYNOPSIS,
                    chapter_node_key(NODE_TRANSLATE, 0),
                    chapter_node_key(NODE_POLISH, 0),
                    chapter_node_key(NODE_NATURALIZE, 0),
                    chapter_node_key(NODE_REVIEW, 0),
                    chapter_node_key(NODE_BACKTRANSLATE, 0),
                    "titles",
                    "consistency_qa",
                    "report",
                    "assemble",
                },
            )
            self.assertEqual(store.load_progress(0).source_digest, "digest")
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)
            self.assertNotIn("book_synopsis", store.load_analysis() or {})
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), store.load_state().nodes)

            # digest 失配 → 连带书级 book_synopsis 与全部下游；无关节点不受波及
            inv2 = store.reconcile_fingerprints(
                {chapter_node_key(NODE_DIGEST, 0): input_fingerprint("new src")}
            )
            self.assertEqual(
                inv2,
                {
                    chapter_node_key(NODE_DIGEST, 0),
                    NODE_BOOK_SYNOPSIS,
                    chapter_node_key(NODE_TRANSLATE, 0),
                    chapter_node_key(NODE_POLISH, 0),
                    chapter_node_key(NODE_NATURALIZE, 0),
                    chapter_node_key(NODE_REVIEW, 0),
                    chapter_node_key(NODE_BACKTRANSLATE, 0),
                    "titles",
                    "consistency_qa",
                    "report",
                    "assemble",
                },
            )
            self.assertEqual(store.load_progress(0).source_digest, "")
            self.assertTrue(store.load_state().analysis_flags.term_mining_done)  # mine_terms 不动
            self.assertIn(chapter_node_key(NODE_TRANSLATE, 0), store.load_state().nodes)

    def test_chapter_scoped_descendants_keep_chapter_scope(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store_with_run(d)
            store.record_node_fingerprint(
                chapter_node_key(NODE_NATURALIZE, 0), input_fingerprint("final text", "en")
            )
            pg = store.load_progress(0)
            pg.naturalized = True
            pg.set_review_issue_dicts(
                [{"index": 0, "type": "missing", "detail": "x", "fixed": False}]
            )
            pg.set_backtranslation_issue_dicts([{"index": 1, "detail": "y"}])
            store.save_progress(0, pg)

            inv = store.reconcile_fingerprints(
                {chapter_node_key(NODE_NATURALIZE, 0): input_fingerprint("changed")}
            )
            self.assertEqual(
                inv,
                {
                    chapter_node_key(NODE_NATURALIZE, 0),
                    chapter_node_key(NODE_REVIEW, 0),
                    chapter_node_key(NODE_BACKTRANSLATE, 0),
                    "titles",
                    "consistency_qa",
                    "report",
                    "assemble",
                },
            )
            pg2 = store.load_progress(0)
            self.assertFalse(pg2.naturalized)
            self.assertEqual(pg2.review_issues, [])
            self.assertEqual(pg2.backtranslation_issues, [])

    def test_transitive_invalidation_reaches_finish_chain(self):
        """传递闭包：translate 输入失配 → 章链 + titles + consistency/report/assemble
        一并失效（重译后必须重扫/重出报告/重新回填，不能跳过 QA 与输出再生成）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._store_with_run(d)
            for node_id in (
                NODE_DIGEST,
                NODE_TRANSLATE,
                NODE_POLISH,
                NODE_NATURALIZE,
                NODE_REVIEW,
                NODE_BACKTRANSLATE,
            ):
                key = chapter_node_key(node_id, 0)
                store.record_node_fingerprint(key, input_fingerprint("fp"))
            for node_id in ("titles", "consistency_qa", "report", "assemble"):
                store.record_node_fingerprint(node_id, input_fingerprint("fp"))
            inv = store.reconcile_fingerprints(
                {chapter_node_key(NODE_TRANSLATE, 0): input_fingerprint("changed")}
            )
            expected = {
                chapter_node_key(node_id, 0)
                for node_id in (
                    NODE_TRANSLATE,
                    NODE_POLISH,
                    NODE_NATURALIZE,
                    NODE_REVIEW,
                    NODE_BACKTRANSLATE,
                )
            }
            expected |= {"titles", "consistency_qa", "report", "assemble"}
            self.assertEqual(inv, expected)
            self.assertEqual(store.load_state().nodes["titles"].status, "pending")
            self.assertEqual(store.load_state().nodes["assemble"].status, "pending")

    def test_skip_clears_polish_and_review_recovery_markers(self):
        """策略跳过 polish/review 必须原子清除对应恢复标记（否则章永久卡未完成/
        就绪门禁永久拒绝）。"""
        from trans_novel.pipeline.state import NODE_POLISH, NODE_REVIEW, PolishBatch

        with tempfile.TemporaryDirectory() as d:
            store = self._store_with_run(d)
            pg = store.load_progress(0)
            pg.pending_polish = [PolishBatch(start=0, count=1)]
            pg.review_pending = True
            store.save_progress(0, pg)

            store.mark_node_skipped(chapter_node_key(NODE_POLISH, 0))
            store.mark_node_skipped(chapter_node_key(NODE_REVIEW, 0))

            pg2 = store.load_progress(0)
            self.assertEqual(pg2.pending_polish, [], "skip polish 必须清 pending_polish")
            self.assertFalse(pg2.review_pending, "skip review 必须清 review_pending")
            self.assertEqual(
                store.load_state().nodes[chapter_node_key(NODE_POLISH, 0)].status, "skipped"
            )

    def test_glossary_semantic_change_invalidates_consistency_fingerprint(self):
        """一致性 QA 指纹必须覆盖术语语义（target/aliases/type/lock/status），
        仅改既有词条语义（tools resolve/lock/audit）就应触发重扫。"""
        from trans_novel.glossary.store import GlossaryTerm
        from trans_novel.pipeline.fingerprints import (
            consistency_input_fingerprint,
            glossary_semantic_fingerprint_part,
        )

        t1 = GlossaryTerm(source="堀北", target="堀北", type="人物")
        t2 = GlossaryTerm(source="堀北", target="堀北铃音", type="人物", locked=True)
        base = "译文摘要"
        self.assertNotEqual(
            consistency_input_fingerprint(base, glossary_semantic_fingerprint_part([t1])),
            consistency_input_fingerprint(base, glossary_semantic_fingerprint_part([t2])),
            "术语语义变化（target/lock）必须改变 QA 指纹",
        )
        # 仅新增 source 也变化
        t3 = GlossaryTerm(source="绫小路", target="绫小路", type="人物")
        self.assertNotEqual(
            consistency_input_fingerprint(base, glossary_semantic_fingerprint_part([t1])),
            consistency_input_fingerprint(base, glossary_semantic_fingerprint_part([t1, t3])),
        )

    def test_running_node_recovers_to_pending_with_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._store_with_run(d)
            store.mark_node_running("digest:0")
            reopened = RunStore(store.run_dir)
            node = reopened.load_state().nodes["digest:0"]
            self.assertEqual(node.status, "pending")
            self.assertEqual(node.failure.kind, "interrupted")
            self.assertEqual(node.attempts, 1)


class TestAssembleReadiness(unittest.TestCase):
    def _complete_state(self, d: str, *, pending: bool = False) -> RunStore:
        store = RunStore(os.path.join(d, "state"))
        identity = RunIdentity(
            source_bytes_sha256="h" * 64,
            run_input_schema_version=1,
            source_lang="en",
            target_lang="zh",
        )
        segs = [Segment(index=0, source="Hi.", target="你好。")]
        state = store.stage_document(
            Document(
                title="T",
                fmt="text",
                source_lang="en",
                target_lang="zh",
                source_path="",
                chapters=[
                    Chapter(index=0, title="C0", segments=list(segs)),
                    Chapter(index=1, title="C1", segments=list(segs)),
                ],
            ),
            identity,
        )
        store.save_manifest(state)
        for i in range(2):
            pg = store.load_progress(i)
            pg.status = STATUS_PENDING if (pending and i == 1) else STATUS_DONE
            pg.source_digest = f"digest{i}"
            store.save_progress(i, pg)
        self._stamp_all_nodes(store)
        return store

    @staticmethod
    def _stamp_all_nodes(store: RunStore) -> None:
        """就绪测试 fixture：补全“已完整运行”的节点状态（必需上游 succeeded）。"""
        state = store.load_state()
        for node_id in (
            NODE_PREPARE,
            NODE_ANALYZE,
            NODE_MINE_TERMS,
            NODE_NAME_TERMS,
            NODE_BOOK_SYNOPSIS,
            NODE_TITLES,
            NODE_CONSISTENCY_QA,
            NODE_REPORT,
        ):
            state.nodes[node_id] = NodeState(node_id=node_id, status="succeeded")
        for ci in (0, 1):
            for node_id in (
                NODE_DIGEST,
                NODE_TRANSLATE,
                NODE_POLISH,
                NODE_NATURALIZE,
                NODE_REVIEW,
                NODE_BACKTRANSLATE,
            ):
                key = chapter_node_key(node_id, ci)
                state.nodes[key] = NodeState(node_id=key, status="succeeded")
        state.analysis_flags.term_mining_done = True
        store.save_state(state)

    def test_ready_state_passes(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            self.assertEqual(assemble_readiness_problems(store), [])

    def test_backtranslate_skipped_legal_and_failed_illegal(self):
        """backtranslate_sample 允许策略抽样不执行：sample=0 时 planner 的 skipped
        是 policy-authorized optional skip → 就绪通过；被选中后 failed/pending
        必须拒绝（不能把可选抽检误改成全书强制回译，也不能放行失败抽检）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            # sample=0 的策略跳过（与 planner skip 语义一致）：合法
            state = store.load_state()
            state.nodes[chapter_node_key(NODE_BACKTRANSLATE, 0)] = NodeState(
                node_id=chapter_node_key(NODE_BACKTRANSLATE, 0), status="skipped"
            )
            store.save_state(state)
            self.assertEqual(assemble_readiness_problems(store), [])

            # 抽检被选中后失败：必须拒绝
            state = store.load_state()
            state.nodes[chapter_node_key(NODE_BACKTRANSLATE, 0)] = NodeState(
                node_id=chapter_node_key(NODE_BACKTRANSLATE, 0), status="failed_retryable"
            )
            store.save_state(state)
            problems = assemble_readiness_problems(store)
            self.assertTrue(
                any("backtranslate:0" in p for p in problems), "被选中的回译失败必须拒绝回填"
            )

            # 抽检被选中后 pending：必须拒绝
            state = store.load_state()
            state.nodes[chapter_node_key(NODE_BACKTRANSLATE, 0)] = NodeState(
                node_id=chapter_node_key(NODE_BACKTRANSLATE, 0), status="pending"
            )
            store.save_state(state)
            problems = assemble_readiness_problems(store)
            self.assertTrue(
                any("backtranslate:0" in p for p in problems), "被选中的回译 pending 必须拒绝回填"
            )

    def test_missing_titles_or_report_blocks(self):
        """正式 writer 门禁：titles/report 缺失即拒绝（formal assemble goal 会执行
        它们；直接 writer 单测需 stamp/执行这两个正式前置）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            state = store.load_state()
            state.nodes.pop("titles", None)
            state.nodes.pop("report", None)
            store.save_state(state)
            problems = assemble_readiness_problems(store)
            self.assertTrue(any("titles 未执行" in p for p in problems))
            self.assertTrue(any("report 未执行" in p for p in problems))

    def test_missing_required_upstream_node_blocks(self):
        """就绪门禁要求适用的必需上游节点 succeeded（缺失即拒绝，不只看失败态）。"""
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            state = store.load_state()
            state.nodes.pop(chapter_node_key(NODE_BACKTRANSLATE, 0), None)
            store.save_state(state)
            problems = assemble_readiness_problems(store)
            self.assertTrue(
                any("backtranslate:0 未执行" in p for p in problems),
                "缺失的必需上游节点必须阻塞回填",
            )

    def test_pending_chapter_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d, pending=True)
            self.assertIn("未完成翻译", assemble_readiness_problems(store)[0])

    def test_empty_required_target_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            ch = store.load_chapter(0)
            ch.segments[0].target = None
            store.save_chapter(ch)
            self.assertTrue(any("未翻译段落" in p for p in assemble_readiness_problems(store)))

    def test_pending_polish_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            pg = store.load_progress(0)
            pg.pending_polish = [PolishBatch(start=0, count=1)]
            store.save_progress(0, pg)
            self.assertTrue(any("润色待完成" in p for p in assemble_readiness_problems(store)))

    def test_review_pending_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            pg = store.load_progress(0)
            pg.review_pending = True
            store.save_progress(0, pg)
            self.assertTrue(any("审校待完成" in p for p in assemble_readiness_problems(store)))

    def test_failed_required_node_blocks_but_best_effort_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d)
            state = store.load_state()
            state.nodes["digest:0"] = NodeState(node_id="digest:0", status="failed_permanent")
            store.save_state(state)
            self.assertTrue(any("节点 digest:0" in p for p in assemble_readiness_problems(store)))

            # mine_terms 是尽力而为节点：失败不阻塞正式产出（其余必需上游仍满足）
            state = store.load_state()
            state.nodes["digest:0"] = NodeState(node_id="digest:0", status="succeeded")
            state.nodes[NODE_MINE_TERMS] = NodeState(
                node_id=NODE_MINE_TERMS, status="failed_retryable"
            )
            store.save_state(state)
            self.assertEqual(assemble_readiness_problems(store), [])

    def test_writer_assemble_rejects_incomplete_state(self):
        with tempfile.TemporaryDirectory() as d:
            store = self._complete_state(d, pending=True)
            src = os.path.join(d, "book.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("Hi." * 100)
            state = store.load_state()
            with open(src, "rb") as f:
                src_bytes = f.read()
            state.identity.source_bytes_sha256 = hashlib.sha256(src_bytes).hexdigest()
            store.save_state(state)
            with self.assertRaises(ReadinessError):
                assemble(store, src, out_format="txt")
            self.assertFalse(os.path.exists(os.path.join(d, "book.zh.txt")))


if __name__ == "__main__":
    unittest.main()
