"""书级收尾节点：标题、确定性 QA、报告与装配。

- titles：章标题与目录项翻译，幂等且严格校验模型输出；
- deterministic_qa：对所有已完成章节执行确定性 lint，不调用模型；
- report：QA 报告；
- assemble：正式回填，就绪门禁拒绝不完整状态。
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from trans_novel.agents import prompts
from trans_novel.agents.base import WorkflowProtocolError, retry_protocol
from trans_novel.assemble import assemble_outputs, bilingual_out_path
from trans_novel.assemble.report import build_report
from trans_novel.config import Config, OutputConfig
from trans_novel.epub.slots import distribute_slot_translation
from trans_novel.glossary.store import GlossaryStore, terms_matching_text
from trans_novel.ingest.models import (
    CANONICAL_TITLE_ID_META,
    KIND_HEADING,
    sanitize_generated_text,
)
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.planning import (
    analyst_model_profile,
    assemble_input_fingerprint,
    build_title_catalog,
    deterministic_qa_input_fingerprint,
    glossary_semantic_fingerprint_part,
    report_input_fingerprint,
    titles_input_fingerprint,
)
from trans_novel.pipeline.quality import lint_targets
from trans_novel.pipeline.state import (
    NODE_ASSEMBLE,
    NODE_DETERMINISTIC_QA,
    NODE_REPORT,
    NODE_TITLES,
    SCOPE_BOOK,
)
from trans_novel.postprocess.punct import normalize_heading_numbering


class _TranslatedTitle(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    target: str

    @field_validator("id", "target")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title response values must be nonempty")
        return value


class _TitleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    titles: list[_TranslatedTitle]


class TitlesNode:
    """一次性翻译并持久化完整的全书层级标题体系。"""

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
            return NodeOutcome()
        manifest = store.load_manifest()
        catalog = build_title_catalog(manifest)
        if request.progress:
            request.progress(0, 0, "翻译全书标题…")
        targets = retry_protocol(
            lambda: self._translate_titles(catalog.items, store),
            retries=self.config.pipeline.protocol_retry_limit,
        )
        manifest["title_translated"] = targets["book"]
        meta = manifest.get("meta")
        entries = meta.get("toc_entries", []) if isinstance(meta, dict) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("entry_id")
            requested_id = (
                catalog.entry_aliases.get(entry_id) if isinstance(entry_id, str) else None
            )
            if requested_id is not None:
                entry["title_translated"] = targets[requested_id]
        for chapter in manifest.get("chapters", []):
            if not isinstance(chapter, dict) or type(chapter.get("index")) is not int:
                continue
            title_id = catalog.chapter_title_ids.get(chapter["index"])
            if title_id is not None:
                chapter["title_translated"] = targets[title_id]
        store.save_manifest(manifest)
        skipped_heading_sync = self._overwrite_canonical_title_segments(
            store,
            manifest,
            targets,
            catalog.chapter_title_ids,
            catalog.entry_aliases,
        )
        store.log_event(
            "titles_translated",
            titles=[
                {"id": item.id, "source": item.source, "target": targets[item.id]}
                for item in catalog.items
            ],
            skipped_chapter_heading_sync=sorted(set(skipped_heading_sync)),
        )
        return NodeOutcome(fingerprint=self._fingerprint(store))

    def _translate_titles(self, items, store) -> dict[str, str]:
        records = [item.request_record() for item in items]
        combined_source = "\n".join(item.source for item in items)
        title_terms = terms_matching_text(self.glossary.all_terms(), combined_source)
        system = prompts.render("title_translator_system", src=self.src, tgt=self.tgt)
        user = prompts.render(
            "title_translator_user",
            src=self.src,
            tgt=self.tgt,
            glossary=prompts.render_glossary(title_terms),
            request_json=json.dumps({"titles": records}, ensure_ascii=False, separators=(",", ":")),
        )
        data = self.client.complete_json(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            stage="title_translate",
            agent="analyst",
            operation="title.translate",
        )
        try:
            response = _TitleResponse.model_validate(data)
        except ValidationError as error:
            self._reject(store, "invalid_schema")
            raise WorkflowProtocolError("title_response_invalid") from error
        expected = [item.id for item in items]
        actual = [item.id for item in response.titles]
        if len(actual) != len(set(actual)):
            self._reject(store, "duplicate_id")
            raise WorkflowProtocolError("title_duplicate_id")
        if set(actual) != set(expected):
            self._reject(store, "id_set_mismatch")
            raise WorkflowProtocolError("title_id_set_mismatch")
        translated: dict[str, str] = {}
        for item in response.titles:
            target = normalize_heading_numbering(
                sanitize_generated_text(item.target).strip()
            ).strip()
            if not target:
                self._reject(store, "empty_target")
                raise WorkflowProtocolError("title_empty_target")
            translated[item.id] = target
        return translated

    @staticmethod
    def _reject(store, reason: str) -> None:
        store.log_event("titles_translation_rejected", reason=reason)

    @staticmethod
    def _overwrite_canonical_title_segments(
        store,
        manifest,
        targets: dict[str, str],
        chapter_title_ids: dict[int, str],
        aliases: dict[str, str],
    ) -> list[int]:
        skipped: list[int] = []
        for chapter_meta in manifest.get("chapters", []):
            if not isinstance(chapter_meta, dict) or type(chapter_meta.get("index")) is not int:
                continue
            chapter_index = chapter_meta["index"]
            title_id = chapter_title_ids.get(chapter_index)
            source_title = " ".join(str(chapter_meta.get("title") or "").split()).casefold()
            chapter = store.load_chapter(chapter_index)
            matches = [
                segment
                for segment in chapter.segments
                if segment.kind == KIND_HEADING
                and " ".join(segment.source.split()).casefold() == source_title
            ]
            changed = False
            if title_id is None or not source_title or len(matches) != 1:
                skipped.append(chapter_index)
            else:
                segment = matches[0]
                target = targets[title_id]
                translation = (
                    distribute_slot_translation(segment.epub_state, target)
                    if segment.epub_state is not None
                    else target
                )
                segment.assign_translation(translation)
                segment.meta[CANONICAL_TITLE_ID_META] = title_id
                changed = True
            for segment in chapter.segments:
                entry_id = segment.meta.get("mirrored_toc_entry_id")
                canonical_id = aliases.get(entry_id) if isinstance(entry_id, str) else None
                if canonical_id is None:
                    continue
                target = targets[canonical_id]
                translation = (
                    distribute_slot_translation(segment.epub_state, target)
                    if segment.epub_state is not None
                    else target
                )
                segment.assign_translation(translation)
                segment.meta[CANONICAL_TITLE_ID_META] = canonical_id
                changed = True
            if changed:
                store.save_chapter(chapter)
        return skipped

    def _fingerprint(self, store) -> str:
        manifest = store.load_manifest()
        catalog = build_title_catalog(manifest)
        topology = [
            json.dumps(item.request_record(), ensure_ascii=False, sort_keys=True)
            for item in catalog.items
        ]
        identity = manifest.get("identity") if isinstance(manifest.get("identity"), dict) else {}
        src = identity.get("source_lang") or self.config.source_lang
        tgt = identity.get("target_lang") or self.config.target_lang
        return titles_input_fingerprint(topology, src, tgt, analyst_model_profile(self.config))


class DeterministicQANode:
    node_id = NODE_DETERMINISTIC_QA
    scope = SCOPE_BOOK

    def __init__(self, *, glossary: GlossaryStore):
        self.glossary = glossary

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        terms = [term for term in self.glossary.all_terms() if getattr(term, "locked", 0)]
        issues: list[dict] = []
        target_texts: list[str] = []
        state = store.load_state()
        for chapter_meta in state.chapters:
            if chapter_meta.processing is not None and chapter_meta.processing.action == "preserve":
                continue
            if store.load_progress(chapter_meta.index).status != "done":
                continue
            chapter = store.load_chapter(chapter_meta.index)
            target_texts.append(
                "\n".join(segment.target or "" for segment in chapter.text_segments)
            )
            segments = chapter.text_segments
            found = lint_targets(
                [segment.source for segment in segments],
                [segment.target or "" for segment in segments],
                locked_terms=terms,
                src_lang=state.identity.source_lang or "en",
            )
            for item in found:
                segment = segments[item.index]
                issues.append(
                    {
                        "chapter": chapter_meta.index,
                        "index": segment.index,
                        "type": item.type,
                        "detail": item.detail,
                    }
                )
        store.record_node_output(NODE_DETERMINISTIC_QA, {"issues": issues})
        fp = deterministic_qa_input_fingerprint(
            "\n".join(target_texts),
            glossary_semantic_fingerprint_part(terms),
        )
        return NodeOutcome(findings_count=len(issues), artifacts={"issues": issues}, fingerprint=fp)


class ReportNode:
    node_id = NODE_REPORT
    scope = SCOPE_BOOK

    def __init__(self, *, glossary: GlossaryStore):
        self.glossary = glossary

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        report = build_report(store, self.glossary)
        store.save_report(report)
        store.log_event("report_saved", path=store.report_path)
        return NodeOutcome(
            artifacts={"report": report},
            fingerprint=self._fingerprint(store, report["deterministic_issues"]),
        )

    def _fingerprint(self, store, deterministic: list | None = None) -> str:
        state = store.load_state()
        lint_issues = [issue for ci in state.progress.values() for issue in ci.lint_issues]
        titles = [c.title for c in state.chapters if c.title]
        if deterministic is None:
            deterministic = lint_issues
        return report_input_fingerprint(
            lint_issues, deterministic or [], [t.source for t in self.glossary.all_terms()], titles
        )


class AssembleNode:
    """正式回填（mono/bilingual 按输出配置）；就绪门禁在 writer.assemble 内部。"""

    node_id = NODE_ASSEMBLE
    scope = SCOPE_BOOK

    def __init__(
        self,
        *,
        output: OutputConfig,
        out_format: str = "epub",
        out_path: str | None = None,
        theme=None,
        output_digest: str | None = None,
    ):
        self.output = output
        self.out_format = out_format
        self.out_path = out_path
        self.theme = theme
        self.output_digest = output_digest

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store

        if request.progress:
            request.progress(0, 0, "生成译文文件…")
        do_mono = self.output.mono
        do_bilingual = self.output.bilingual.enabled
        requests: list[tuple[str | None, bool]] = []
        if do_mono:
            requests.append((self.out_path, False))
        if do_bilingual:
            requests.append((bilingual_out_path(self.out_path) if self.out_path else None, True))
        outputs = assemble_outputs(
            store,
            request.input_path,
            requests,
            self.out_format,
            order=self.output.bilingual.order,
            theme=self.theme,
            output_digest=self.output_digest,
        )
        store.log_event("assembled", outputs=outputs, out_format=self.out_format)
        fp = assemble_input_fingerprint(
            self._targets_text(store),
            mono=do_mono,
            bilingual=do_bilingual,
            out_format=self.out_format,
            bilingual_order=self.output.bilingual.order,
            output_digest=self.output_digest,
        )
        return NodeOutcome(
            artifacts={"outputs": outputs, "output_digest": self.output_digest},
            fingerprint=fp,
        )

    @staticmethod
    def _targets_text(store) -> str:
        from trans_novel.pipeline.state import STATUS_DONE

        parts: list[str] = []
        for c in store.load_state().chapters:
            if store.load_progress(c.index).status != STATUS_DONE:
                continue
            parts.append(
                "\n".join(s.target or "" for s in store.load_chapter(c.index).text_segments)
            )
        return "\n".join(parts)


__all__ = ["AssembleNode", "DeterministicQANode", "ReportNode", "TitlesNode"]
