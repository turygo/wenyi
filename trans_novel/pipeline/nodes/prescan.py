"""全书预扫节点：digest / mine_terms / name_terms / book_synopsis。

保留迁移前 _build_understanding 的语义：
- digest 逐章独立、可续跑（source_digest + 输入指纹双检查点）；digest 与
  mine_terms 由 runner 并行层真正并发执行；
- mine_terms（尽力而为）失败不落 term_mining_done，续跑重试；
- name_terms 与 mine_terms 共享 term_mining_done 检查点：任一失败都不落完成标记；
- book_synopsis 只覆盖稳定输入（digests + 风格 + 语言），不含术语表（翻译期
  术语库持续增长不得失效已生成的全书概览）。
"""

from __future__ import annotations

from trans_novel.agents import prompts
from trans_novel.agents.base import WorkflowProtocolError
from trans_novel.config import Config
from trans_novel.glossary.miner import mine_candidates
from trans_novel.glossary.store import TYPE_PERSON, GlossaryStore
from trans_novel.pipeline.backmatter import is_back_matter
from trans_novel.pipeline.contracts import NodeOutcome, NodeRequest
from trans_novel.pipeline.fingerprints import (
    fast_model_profile,
    frozen_input_fingerprint,
    primary_model_profile,
)
from trans_novel.pipeline.state import (
    NODE_BOOK_SYNOPSIS,
    NODE_DIGEST,
    NODE_MINE_TERMS,
    NODE_NAME_TERMS,
    SCOPE_BOOK,
    SCOPE_CHAPTER,
    input_fingerprint,
    normalize_lang_code,
)


# ── 输入指纹（planner 需要-run 判定与节点记录共用同一公式）─────────────────
def digest_input_fingerprint(source_text: str, src_lang: str, model: str = "") -> str:
    return input_fingerprint(source_text, normalize_lang_code(src_lang), model)


def mine_terms_input_fingerprint(
    chapter_texts: list[str], src_lang: str, concurrency: int, model: str = ""
) -> str:
    return input_fingerprint(chapter_texts, normalize_lang_code(src_lang), concurrency, model)


def name_terms_input_fingerprint(
    candidate_surfaces: list[str],
    style_brief: str,
    digests: list[str],
    concurrency: int,
    model: str = "",
) -> str:
    return input_fingerprint(candidate_surfaces, style_brief, digests, concurrency, model)


def book_synopsis_input_fingerprint(
    digests: list[str],
    style_brief: str,
    src_lang: str,
    tgt_lang: str,
    model: str = "",
) -> str:
    return input_fingerprint(
        digests,
        style_brief,
        normalize_lang_code(src_lang),
        normalize_lang_code(tgt_lang),
        model,
    )


def _import_frozen_glossary(glossary: GlossaryStore, frozen_book) -> None:
    for term in frozen_book.glossary:
        existing = glossary.get_term(term.source)
        if existing is not None:
            if existing != term:
                raise ValueError(f"frozen glossary conflict: {term.source}")
            continue
        glossary.upsert_term(term)


def _body_chapter_count(store) -> int:
    state = store.load_state()
    n = len(state.chapters)
    return sum(1 for c in state.chapters if not is_back_matter(c.title, index=c.index, total=n))


class DigestNode:
    """逐章梗概：存入 ChapterProgress.source_digest。"""

    node_id = NODE_DIGEST
    scope = SCOPE_CHAPTER

    def __init__(self, *, synopsizer, config: Config, frozen_book=None):
        self.synopsizer = synopsizer
        self.config = config
        self.frozen_book = frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        ci = request.ci
        store = request.store
        chapter = store.load_chapter(ci)
        src = "\n".join(s.source for s in chapter.text_segments)
        progress = store.load_progress(ci)
        if self.frozen_book is not None:
            original_index = request.shared.frozen_chapter_index(ci)
            digest = self.frozen_book.chapter_digest(ci, original_index=original_index)
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                (self.frozen_book.book_id, original_index),
                digest,
            )

            def _commit(request) -> None:
                progress = store.load_progress(ci)
                progress.source_digest = digest
                store.save_progress(ci, progress)

            return NodeOutcome(fingerprint=fp, commit=_commit)
        fp = digest_input_fingerprint(src, self.synopsizer.src, fast_model_profile(self.config))
        if progress.source_digest:
            node = store.load_state().nodes.get(request.key)
            if node is not None and node.input_fingerprint == fp:
                return NodeOutcome(fingerprint=fp)
        total = _body_chapter_count(store)
        if request.progress:
            request.progress(0, total, "通读全书章节…")
        digest = self.synopsizer.digest_chapter_strict(src)

        # 并行层 worker 只做 LLM 计算：产物经 commit 回调由 runner 在 join 后
        # 于主线程持久化（save_progress 是 manifest 读改写，worker 并发写会竞争
        # 固定 tmp 文件名——RunStore 写回主线程的仓库约定）。
        def _commit(request) -> None:
            pg = store.load_progress(ci)
            pg.source_digest = digest
            store.save_progress(ci, pg)
            store.log_event("book_understanding_chapter_digest_saved", chapter=ci, digest=digest)

        if request.progress:
            request.progress(1, total, "通读全书章节…")
        return NodeOutcome(fingerprint=fp, commit=_commit)


class MineTermsNode:
    """源文侧候选挖掘（尽力而为）：产物经 artifacts 交给 name_terms。"""

    node_id = NODE_MINE_TERMS
    scope = SCOPE_BOOK

    def __init__(self, *, namer, config: Config, frozen_book=None):
        self.namer = namer
        self.config = config
        self.frozen_book = frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if self.frozen_book is not None:
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                self.frozen_book.book_id,
                [t.source for t in self.frozen_book.glossary],
            )

            def _commit(request) -> None:
                # Commit runs on the runner's main thread. Keep the SQLite
                # connection owned by that thread and do not reuse a
                # connection opened while this parallel node was planned.
                glossary = GlossaryStore(store.glossary_path)
                try:
                    _import_frozen_glossary(glossary, self.frozen_book)
                finally:
                    glossary.close()
                state = store.load_state()
                state.analysis_flags.term_mining_done = True
                store.save_state(state)

            return NodeOutcome(fingerprint=fp, artifacts={"candidates": []}, commit=_commit)
        state = store.load_state()
        chapters = list(state.chapters)
        n = len(chapters)
        # 挖掘输入必须用 is_back_matter 排除附属章（不受 back_matter=full 影响）。
        mine_chapters = [
            c.index for c in chapters if not is_back_matter(c.title, index=c.index, total=n)
        ]
        src_chapters = [
            (
                ci,
                "\n".join(s.source for s in store.load_chapter(ci).text_segments),
            )
            for ci in mine_chapters
        ]
        on_progress = (
            (lambda i, total: request.progress(i, total, "查找专有名词…"))
            if request.progress
            else None
        )
        try:
            candidates = mine_candidates(
                self.namer.src,
                src_chapters,
                self.namer,
                concurrency=max(1, self.config.pipeline.prescan_concurrency),
                on_progress=on_progress,
            )
        except Exception as exc:
            # 事件日志是单次 O_APPEND 追加（原子、无 tmp 文件竞争），worker 可直接记；
            # 其余 RunStore 写全部留给 commit（主线程）。
            store.log_event("cast_naming_failed", error=str(exc))
            raise

        def _commit(request) -> None:
            store.log_event("term_candidates_mined", count=len(candidates))

        fp = mine_terms_input_fingerprint(
            [text for _, text in src_chapters],
            self.namer.src,
            self.config.pipeline.prescan_concurrency,
            fast_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp, artifacts={"candidates": candidates}, commit=_commit)


class NameTermsNode:
    """一次性定名 + 术语入库；term_mining_done 唯一落盘点（与 mine_terms 共享）。"""

    node_id = NODE_NAME_TERMS
    scope = SCOPE_BOOK

    def __init__(
        self,
        *,
        namer,
        analyzer,
        glossary: GlossaryStore,
        config: Config,
        frozen_book=None,
    ):
        self.namer = namer
        self.analyzer = analyzer
        self.glossary = glossary
        self.config = config
        self.frozen_book = frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        if self.frozen_book is not None:
            _import_frozen_glossary(self.glossary, self.frozen_book)
            state = store.load_state()
            state.analysis_flags.term_mining_done = True
            store.save_state(state)
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                self.frozen_book.book_id,
                [t.source for t in self.frozen_book.glossary],
            )
            return NodeOutcome(fingerprint=fp, artifacts={"named_terms": []})
        candidates = (request.artifacts.get("mine_terms") or {}).get("candidates")
        if candidates is None:
            # mine（尽力而为）失败未产出候选：不得把 name_terms 标成功——
            # 抛协议错误保持可重试；term_mining_done 未落盘，续跑重跑挖掘+定名。
            raise WorkflowProtocolError("missing_mine_candidates")
        state = store.load_state()
        digests = [store.load_progress(c.index).source_digest for c in state.chapters]
        analysis = store.load_analysis() or {}
        style_brief = self.analyzer.style_brief(analysis)
        glossary = self.glossary
        on_progress = (
            (lambda i, total: request.progress(i, total, "统一译名…")) if request.progress else None
        )
        existing = glossary.all_terms()
        try:
            named = self.namer.name_terms(
                candidates,
                style_brief,
                digests,
                existing=existing,
                concurrency=max(1, self.config.pipeline.prescan_concurrency),
                on_progress=on_progress,
            )
        except Exception as exc:
            store.log_event("cast_naming_failed", error=str(exc))
            raise
        inserted = 0
        for t in named:
            result = glossary.upsert_term(t, chapter=0)
            if result in ("inserted", "updated"):
                inserted += 1
            if t.type == TYPE_PERSON:
                # namer 确认沿用已有译法时升级 locked/confidence（防锁错译法）。
                glossary.confirm_locked(t.source, t.target)
        state = store.load_state()
        state.analysis_flags.term_mining_done = True
        store.save_state(state)
        store.log_event("cast_named", count=inserted)
        fp = name_terms_input_fingerprint(
            [c.surface for c in candidates],
            style_brief,
            digests,
            self.config.pipeline.prescan_concurrency,
            primary_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp, artifacts={"named_terms": named})


class BookSynopsisNode:
    """全书概览：digests + 风格 + 定名人物表 → analysis.json["book_synopsis"]。"""

    node_id = NODE_BOOK_SYNOPSIS
    scope = SCOPE_BOOK

    def __init__(
        self,
        *,
        synopsizer,
        analyzer,
        glossary: GlossaryStore,
        config: Config,
        frozen_book=None,
    ):
        self.synopsizer = synopsizer
        self.analyzer = analyzer
        self.glossary = glossary
        self.config = config
        self.frozen_book = frozen_book

    def execute(self, request: NodeRequest) -> NodeOutcome:
        store = request.store
        state = store.load_state()
        digests = [store.load_progress(c.index).source_digest for c in state.chapters]
        analysis = store.load_analysis() or {}
        style_brief = self.analyzer.style_brief(analysis)
        if self.frozen_book is not None:
            analysis["book_synopsis"] = self.frozen_book.book_synopsis
            store.save_analysis(analysis)
            fp = frozen_input_fingerprint(
                request.shared.frozen_preparation.preparation_sha256,
                self.node_id,
                self.frozen_book.book_id,
                (self.frozen_book.book_synopsis, tuple(digests)),
            )
            return NodeOutcome(fingerprint=fp)
        if not any(d.strip() for d in digests):
            return NodeOutcome()
        if request.progress:
            request.progress(0, 0, "生成全书概览…")
        glossary = self.glossary
        cast_text = prompts.render_glossary(
            [t for t in glossary.all_terms() if t.type == TYPE_PERSON]
        )
        synopsis = self.synopsizer.book_synopsis_strict(digests, style_brief, cast=cast_text)
        analysis["book_synopsis"] = synopsis
        store.save_analysis(analysis)
        store.log_event("book_synopsis_saved", synopsis=synopsis)
        fp = book_synopsis_input_fingerprint(
            digests,
            style_brief,
            self.synopsizer.src,
            self.synopsizer.tgt,
            fast_model_profile(self.config),
        )
        return NodeOutcome(fingerprint=fp)


__all__ = [
    "BookSynopsisNode",
    "DigestNode",
    "MineTermsNode",
    "NameTermsNode",
    "book_synopsis_input_fingerprint",
    "digest_input_fingerprint",
    "mine_terms_input_fingerprint",
    "name_terms_input_fingerprint",
]
