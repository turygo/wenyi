"""书级收尾节点：titles / consistency_qa / report / assemble。

- titles：章标题与目录项翻译（书名保持原文；TOC entry 驱动同步；正文 heading 复用），
  幂等（已全部译过则跳过）；provider 失败原样冒泡（必需节点，不得伪装成功空结果）；
- consistency_qa：跨章一致性扫描（尽力而为输出：空 issue 列表 = 成功）；
- report：QA 报告（含一致性扫描结果，若本轮已跑）；
- assemble：正式回填（writer 内部先过就绪门禁，不完整状态拒绝产出）。
"""

from __future__ import annotations

from trans_novel.agents import prompts
from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.assemble.report import build_report
from trans_novel.assemble.writer import assemble, bilingual_out_path
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.ingest.models import KIND_HEADING
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    assemble_input_fingerprint,
    consistency_input_fingerprint,
    fast_model_profile,
    glossary_semantic_fingerprint_part,
    primary_model_profile,
    report_input_fingerprint,
    titles_input_fingerprint,
)
from trans_novel.pipeline.nodes.common import terms_matching_text
from trans_novel.pipeline.runstore import stable_digest
from trans_novel.pipeline.state import (
    NODE_ASSEMBLE,
    NODE_CONSISTENCY_QA,
    NODE_REPORT,
    NODE_TITLES,
    SCOPE_BOOK,
)


class TitlesNode:
    """章标题/目录项翻译（schema 2 TOC entry 驱动；幂等）。"""

    node_id = NODE_TITLES
    scope = SCOPE_BOOK

    def __init__(
        self,
        *,
        client,
        config: Config,
        src: str,
        tgt: str,
        glossary: GlossaryStore,
    ):
        self.client = client
        self.config = config
        self.src = src
        self.tgt = tgt
        self.glossary = glossary

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if store.pending_chapters():
            return NodeOutcome()  # 还有未完成章 → 不译标题（与旧 run() 条件一致）
        src = self.src
        tgt = self.tgt
        m = store.load_manifest()
        chapters = m.get("chapters", [])
        glossary = self.glossary

        def _flat(s: object) -> str:
            return " ".join(str(s or "").split())

        raw_meta = m.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        raw_toc_entries = meta.get("toc_entries", [])
        toc_entry_list = raw_toc_entries if isinstance(raw_toc_entries, list) else []
        entries_by_id = {
            e["entry_id"]: e
            for e in toc_entry_list
            if isinstance(e, dict) and isinstance(e.get("entry_id"), str)
        }
        toc_entries_pending = [
            e
            for e in toc_entry_list
            if isinstance(e, dict)
            and not e.get("external")
            and _flat(e.get("title", ""))
            and not e.get("title_translated")
        ]

        toc_covered_chapters = []
        other_chapters = []
        for c in chapters:
            entry_id = c.get("toc_entry_id")
            if entry_id and entry_id in entries_by_id:
                toc_covered_chapters.append(c)
            elif _flat(c.get("title", "")):
                other_chapters.append(c)

        m.pop("title_translated", None)

        llm_chapters = []
        for c in other_chapters:
            chapter = store.load_chapter(c["index"])
            segs = chapter.segments
            heading_target = ""
            if (
                segs
                and segs[0].kind == KIND_HEADING
                and _flat(segs[0].source) == _flat(c.get("title", ""))
            ):
                heading_target = _flat(segs[0].target)
            if heading_target:
                c["title_translated"] = heading_target
            elif not c.get("title_translated"):
                llm_chapters.append(c)

        for c in toc_covered_chapters:
            entry_translated = entries_by_id[c["toc_entry_id"]].get("title_translated")
            if entry_translated:
                c["title_translated"] = entry_translated

        store.save_manifest(m)  # 先落盘复用/同步结果，即便后续 LLM 调用失败也不丢失

        if not llm_chapters and not toc_entries_pending:
            store.log_event("titles_skipped", reason="already_translated")
            fp = self._fingerprint(store)
            return NodeOutcome(fingerprint=fp)

        pending_items: list[dict] = [*llm_chapters, *toc_entries_pending]
        unique_titles: list[str] = []
        title_slot: dict[str, int] = {}
        for item in pending_items:
            key = _flat(item.get("title", ""))
            if key not in title_slot:
                title_slot[key] = len(unique_titles)
                unique_titles.append(key)

        if request.progress:
            request.progress(0, 0, "翻译章节标题…")
        analysis = store.load_analysis() or {}
        book_synopsis = analysis.get("book_synopsis") or "（无）"
        system = prompts.render(
            "title_translator_system",
            src=src,
            tgt=tgt,
            n=len(unique_titles),
        )
        # 标题裁剪始终生效，不受 glossary_scope 配置影响（标题文本很短，全量术语表是噪声）。
        title_terms = terms_matching_text(glossary.all_terms(), "\n".join(unique_titles))
        user = prompts.render(
            "title_translator_user",
            src=src,
            tgt=tgt,
            book_synopsis=book_synopsis,
            glossary=prompts.render_glossary(title_terms),
            n=len(unique_titles),
            numbered_titles=prompts.numbered(unique_titles),
        )
        # provider 失败必须冒泡（必需节点落失败态、计划中止），不伪装成功空结果。
        data = self.client.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            stage="title_translate",
            agent="translator",
            operation="title.translate",
        )
        out = data.get("titles") if isinstance(data, dict) else data
        if not isinstance(out, list) or len(out) != len(unique_titles):
            # 数量不符是协议错误：节点落失败态保持可重试（幂等），下次 run 重试，
            # 而不是记录 succeeded 后永久跳过（标题会一直缺失）。
            store.log_event(
                "titles_translation_rejected",
                reason="count_mismatch",
                expected=len(unique_titles),
                actual=len(out) if isinstance(out, list) else None,
            )
            raise WorkflowProtocolError("title_count_mismatch")
        out = [str(t).strip() for t in out]
        if not all(out):
            # 空/缺失的标题项：协议错误（保持可重试），不得回退占位导致标题缺失。
            store.log_event(
                "titles_translation_rejected",
                reason="empty_item",
                expected=len(unique_titles),
            )
            raise WorkflowProtocolError("title_empty_item")
        for item in pending_items:
            key = _flat(item.get("title", ""))
            item["title_translated"] = out[title_slot[key]] or item.get("title")

        for c in toc_covered_chapters:
            entry_translated = entries_by_id[c["toc_entry_id"]].get("title_translated")
            if entry_translated:
                c["title_translated"] = entry_translated

        store.save_manifest(m)
        store.log_event(
            "titles_translated",
            titles=[
                {"index": i, "source": source, "target": target}
                for i, (source, target) in enumerate(zip(unique_titles, out, strict=False))
            ],
        )
        fp = self._fingerprint(store)
        return NodeOutcome(fingerprint=fp)

    def _fingerprint(self, store) -> str:
        m = store.load_manifest()
        titles = [str(c.get("title", "")) for c in m.get("chapters", []) if c.get("title")]
        meta = m.get("meta")
        raw_toc = meta.get("toc_entries") if isinstance(meta, dict) else None
        if isinstance(raw_toc, list):
            titles.extend(
                str(e.get("title", "")) for e in raw_toc if isinstance(e, dict) and e.get("title")
            )
        book_synopsis = (store.load_analysis() or {}).get("book_synopsis") or ""
        return titles_input_fingerprint(
            titles, self.src, self.tgt, book_synopsis, primary_model_profile(self.config)
        )


class ConsistencyQANode:
    """跨章一致性扫描；成功空 issue 列表 = succeeded(findings_count=0)，绝非失败。"""

    node_id = NODE_CONSISTENCY_QA
    scope = SCOPE_BOOK

    def __init__(self, *, checker, glossary: GlossaryStore, config: Config):
        self.checker = checker
        self.glossary = glossary
        self.config = config

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if request.progress:
            request.progress(0, 0, "检查全书一致性…")
        issues = self.checker.check(store, self.glossary, strict=True)
        store.log_event(
            "consistency_qa_finished",
            issue_count=len(issues),
            issues_sha256=stable_digest(issues),
        )
        # 持久化到节点 output：崩溃在 QA 之后、report 之前，或跨 tools qa → tools
        # report 调用，report 仍能取到本轮发现的问题（runner 内 artifacts 只活一轮）。
        store.record_node_output(NODE_CONSISTENCY_QA, {"issues": issues})
        fp = consistency_input_fingerprint(
            self._targets_text(store),
            glossary_semantic_fingerprint_part(self.glossary.all_terms()),
            fast_model_profile(self.config),
        )
        return NodeOutcome(findings_count=len(issues), artifacts={"issues": issues}, fingerprint=fp)

    @staticmethod
    def _targets_text(store) -> str:
        from trans_novel.pipeline.runstore import STATUS_DONE

        parts: list[str] = []
        for c in store.load_state().chapters:
            if store.load_progress(c.index).status != STATUS_DONE:
                continue
            parts.append(
                "\n".join(s.target or "" for s in store.load_chapter(c.index).text_segments)
            )
        return "\n".join(parts)


class ReportNode:
    """QA 报告：汇总术语冲突/漏译/审校/回译疑点，合并本轮一致性扫描结果。"""

    node_id = NODE_REPORT
    scope = SCOPE_BOOK

    def __init__(self, *, glossary: GlossaryStore):
        self.glossary = glossary

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if request.progress:
            request.progress(0, 0, "生成报告…")
        report = build_report(store, self.glossary)
        # 本轮 artifacts 优先；跨调用/崩溃后续跑回退到持久化的 QA 产物（不丢问题）。
        consistency_issues = (request.artifacts.get("consistency_qa") or {}).get("issues")
        if consistency_issues is None:
            qa_node = store.load_state().nodes.get(NODE_CONSISTENCY_QA)
            persisted = (qa_node.output or {}).get("issues") if qa_node else None
            consistency_issues = persisted if persisted is not None else []
        report["consistency_issues"] = consistency_issues
        store.save_report(report)
        store.log_event("report_saved", path=store.report_path)
        fp = self._fingerprint(store, consistency_issues)
        return NodeOutcome(artifacts={"report": report}, fingerprint=fp)

    def _fingerprint(self, store, consistency_issues: list | None = None) -> str:
        state = store.load_state()
        review_issues: list[dict] = []
        bt_issues: list[dict] = []
        titles: list[str] = []
        for c in state.chapters:
            pg = store.load_progress(c.index)
            review_issues.extend(pg.review_issue_dicts())
            bt_issues.extend(pg.backtranslation_issue_dicts())
            if c.title:
                titles.append(c.title)
        if consistency_issues is None:
            qa_node = state.nodes.get(NODE_CONSISTENCY_QA)
            consistency_issues = (qa_node.output or {}).get("issues") if qa_node else None
        return report_input_fingerprint(
            review_issues,
            bt_issues,
            consistency_issues or [],
            [t.source for t in self.glossary.all_terms()],
            titles,
        )


class AssembleNode:
    """正式回填（mono/bilingual 按输出配置）；就绪门禁在 writer.assemble 内部。"""

    node_id = NODE_ASSEMBLE
    scope = SCOPE_BOOK

    def __init__(self, *, config: Config, out_format: str = "epub", out_path: str | None = None):
        self.config = config
        self.out_format = out_format
        self.out_path = out_path

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if request.progress:
            request.progress(0, 0, "生成译文文件…")
        out_cfg = self.config.output
        do_mono, do_bilingual = out_cfg.mono, out_cfg.bilingual
        if not do_mono and not do_bilingual:
            do_mono = True  # 兜底：mono/bilingual 都关时至少产一个单语产物
        outputs: list[str] = []
        if do_mono:
            outputs.append(
                assemble(
                    store,
                    request.input_path,
                    out_path=self.out_path,
                    out_format=self.out_format,
                    bilingual=False,
                )
            )
        if do_bilingual:
            bi_out_path = bilingual_out_path(self.out_path) if self.out_path else None
            outputs.append(
                assemble(
                    store,
                    request.input_path,
                    out_path=bi_out_path,
                    out_format=self.out_format,
                    bilingual=True,
                    order=out_cfg.bilingual_order,
                )
            )
        store.log_event("assembled", outputs=outputs, out_format=self.out_format)
        fp = assemble_input_fingerprint(
            self._targets_text(store),
            mono=do_mono,
            bilingual=do_bilingual,
            out_format=self.out_format,
            bilingual_order=self.config.output.bilingual_order,
        )
        return NodeOutcome(artifacts={"outputs": outputs}, fingerprint=fp)

    @staticmethod
    def _targets_text(store) -> str:
        from trans_novel.pipeline.runstore import STATUS_DONE

        parts: list[str] = []
        for c in store.load_state().chapters:
            if store.load_progress(c.index).status != STATUS_DONE:
                continue
            parts.append(
                "\n".join(s.target or "" for s in store.load_chapter(c.index).text_segments)
            )
        return "\n".join(parts)


__all__ = ["AssembleNode", "ConsistencyQANode", "ReportNode", "TitlesNode"]
