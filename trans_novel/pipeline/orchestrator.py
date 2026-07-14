"""编排器：驱动全流程，章级状态机 + 断点续跑。

单章流水线（章内批次**串行**翻译，逐批刷新滚动上下文与术语快照；跨章亦串行传递梗概）：
  每批：渲染上下文（含前一批刚译出的译文）→ 翻译（强档，唯一阻塞关键路径的调用，段数对齐保证）→
        立即落盘（crash-safe）→ 术语抽取（fast 档，落盘后主线程同步抽取，供下一批参照）→
        润色（可选，提交共享线程池后台跑，批间无依赖，不阻塞下一批翻译）。
  章末（串行）：排干本章全部润色 future（含续跑遗留的 pending_polish，写回前按需标点规范化）→
        全章术语兜底抽取（在润色后的最终文本上，仍在审校前）→ 整章分块审校
        （review=true 且 autofix_severe=false 时提交线程池异步跑，不阻塞下一章；
          autofix_severe=true 时因重译要写回正文而保持同步，含严重项定向重译）→
        回译抽检（从最终文本抽样）→ 写 TM → 落盘标记 done。
  run() 起一个 4-worker 共享线程池（润色 + 各章异步审校复用同一个池；术语抽取因主线程要立即
  用新词，改为主线程同步、不入池，避免被在飞 future 占满时拖回翻译关键路径）。SQLite 术语库与
  RunStore 的读写全部留在主线程，worker 线程只做 LLM 调用；异步审校在 manifest 记 review_pending
  持久标记，run() 收尾（发 translate_run_finished 前）排干所有在飞 future 并清标记，崩溃续跑据
  残留标记补跑，避免异步审校结果丢失。
翻译前先预扫源文建立全书理解（逐章梗概+全书概览，fast 档并行），作恒定前缀注入每章翻译。

run_all：在翻译全书后接 术语 AI 审计统一 → 一致性 QA → 写报告 → 回填出 EPUB，一气呵成。
进度回调 progress(done_segments, total_segments, label) 与 UI 无关，每批完成即触发。
"""

from __future__ import annotations

import os
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from ..agents import prompts
from ..agents.analyzer import Analyzer
from ..agents.namer import CastNamer
from ..agents.naturalizer import Naturalizer, naturalize_chapter
from ..agents.polisher import Polisher
from ..agents.reviewer import BackTranslator, Reviewer
from ..agents.synopsis import Synopsizer
from ..agents.translator import Translator
from ..config import Config
from ..glossary.extractor import GlossaryExtractor
from ..glossary.miner import mine_candidates
from ..glossary.store import TYPE_PERSON, GlossaryStore
from ..ingest.models import KIND_HEADING
from ..ingest.segmenter import batch_segments, load_document
from ..llm.base import LLMClient, build_client, merge_usage_summaries, usage_delta
from ..postprocess.punct import normalize_zh
from . import checks, lint
from .backmatter import is_back_matter
from .context import RollingContext
from .runstore import STATUS_DONE, STATUS_PENDING, RunStore, slugify

ProgressFn = Callable[[int, int, str], None]


# 语言名/代码 → ISO 639-1 两字母代码（模型检测结果归一化）
_LANG_ALIASES = {
    "japanese": "ja",
    "日语": "ja",
    "日文": "ja",
    "jp": "ja",
    "jpn": "ja",
    "english": "en",
    "英语": "en",
    "英文": "en",
    "eng": "en",
    "russian": "ru",
    "俄语": "ru",
    "俄文": "ru",
    "rus": "ru",
    "chinese": "zh",
    "中文": "zh",
    "汉语": "zh",
    "zh-cn": "zh",
    "zho": "zh",
    "korean": "ko",
    "韩语": "ko",
    "韩文": "ko",
    "kor": "ko",
    "french": "fr",
    "法语": "fr",
    "法文": "fr",
    "german": "de",
    "德语": "de",
    "德文": "de",
    "spanish": "es",
    "西班牙语": "es",
    "西班牙文": "es",
    "italian": "it",
    "意大利语": "it",
    "意大利文": "it",
    "portuguese": "pt",
    "葡萄牙语": "pt",
    "葡萄牙文": "pt",
}


def _normalize_lang(code: str) -> str:
    c = (code or "").strip().lower()
    if not c or c in {"auto", "unknown", "und", "uncertain", "mixed", "多语言", "未知"}:
        return ""
    if c in _LANG_ALIASES:
        return _LANG_ALIASES[c]
    return c[:2] if c[:2].isalpha() else ""


# ── 锁定人物的部分称呼匹配（章级术语快照用）────────────────────────────────
# 任意文字系统的「词」（字母序列，不含数字/下划线）；CJK 无空格文本会成为整段长 run，
# 由 _person_mentioned 的汉字分支单独处理。
_WORD_RE = re.compile(r"[^\W\d_]+")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


def _person_mentioned(term, text: str, words: set[str]) -> bool:
    """锁定人物是否以「部分形式」出现在本章源文里（全名/别名的整体匹配由 terms_in 负责）。

    - 多词姓名（"Greg McKeown"、"田中 太郎"）：任一 ≥2 字符的组成词命中即算——
      含汉字的词按子串匹配，其余文字按整词（words 集合）匹配且要求首字母大写
      （跳过 "the"/"van"/"de" 等小写虚词，否则 "Catherine the Great" 会命中一切英文章节）；
      覆盖后文只呼姓/名的段落。
    - 无空格纯汉字姓名（"田中太郎"）：取 2/3 字前缀做子串匹配（日文姓氏典型长度），
      覆盖只呼姓的段落；前缀等于全名时跳过（terms_in 已管）。
    误报代价 = 多注入一条词条（无害）；漏报代价 = 该章译名靠模型自拟，故宁松勿紧。
    """
    for name in (term.source, *(term.aliases or [])):
        parts = [p for p in _WORD_RE.findall(name) if len(p) >= 2]
        if len(parts) >= 2:
            for part in parts:
                if _HAN_RE.search(part):
                    if part in text:
                        return True
                elif part[0].isupper() and part in words:
                    return True
        elif _HAN_RE.search(name):
            for plen in (2, 3):
                if plen < len(name) and name[:plen] in text:
                    return True
    return False


def _resume_batches(segments, max_chars: int) -> list[list]:
    """按字符预算分批后，再沿“已完成/待翻译”边界切开。

    用户调整批次预算时，新的批次可能同时包含已有译文和空译文。若直接重跑
    该混合批次会覆盖已确认内容；按完成状态分组可只补译缺失段。
    """
    batches: list[list] = []
    for raw_batch in batch_segments(segments, max_chars):
        current: list = []
        current_done: bool | None = None
        for segment in raw_batch:
            done = bool(segment.target and segment.target.strip())
            if current and done != current_done:
                batches.append(current)
                current = []
            current.append(segment)
            current_done = done
        if current:
            batches.append(current)
    return batches


class Orchestrator:
    def __init__(self, config: Config, client: LLMClient | None = None):
        self.config = config
        self.client = client or build_client(config)
        # client 的统计是进程内累计；checkpoint 用于每次落盘时只提取新增部分。
        self._usage_checkpoint = self.client.usage_summary()
        self.analyzer = Analyzer(self.client, config)
        self.synopsizer = Synopsizer(self.client, config)
        self.translator = Translator(self.client, config)
        self.reviewer = Reviewer(self.client, config)
        self.backtrans = BackTranslator(self.client, config)
        self.polisher = Polisher(self.client, config)
        self.extractor = GlossaryExtractor(self.client, config)
        self.namer = CastNamer(self.client, config)
        self.naturalizer = Naturalizer(self.client, config)

    def _flush_usage(self, store: RunStore, *, scope: str) -> dict[str, Any]:
        """把当前 client 尚未落盘的用量增量合并到本书 usage.json。

        持久化门控不能只看 totals.calls（=by_tier 里成功响应计数）：一次完全失败的
        逻辑调用（如 Agent._ask_json 捕获异常回退 default）不会走 usage.record()，
        by_tier/by_stage/totals 全零，但 by_operation 的 attempts/failed_attempts/
        logical_calls 仍真实增长——这类 operation-only 增量同样必须落盘，否则续跑
        后这次失败尝试的底层统计永久丢失。usage_delta 内部已用 _nonneg_delta 过滤
        掉全零槽位，故 increment["by_operation"] 非空即代表确有变化。
        """
        current = self.client.usage_summary()
        increment = usage_delta(current, self._usage_checkpoint)
        self._usage_checkpoint = current
        accumulated = store.load_usage() or {"totals": {}, "by_tier": {}, "by_stage": {}}
        has_activity = bool(increment["totals"]["calls"]) or bool(increment.get("by_operation"))
        if not has_activity:
            return merge_usage_summaries(accumulated, increment)
        cumulative = merge_usage_summaries(accumulated, increment)
        store.save_usage(cumulative)
        store.log_event(
            "usage_summary",
            scope=scope,
            increment=increment,
            cumulative=cumulative,
        )
        return cumulative

    # ── 语言解析 ────────────────────────────────────────────────────────────
    def _apply_language(self, lang: str) -> None:
        """把解析出的源语言应用到 config 与各 agent（auto 检测后调用）。"""
        resolved = lang or self.config.source_lang
        self.config.source_lang = resolved
        for ag in (
            self.analyzer,
            self.synopsizer,
            self.translator,
            self.reviewer,
            self.backtrans,
            self.polisher,
            self.extractor,
            self.namer,
            self.naturalizer,
        ):
            ag.src = resolved

    # ── 准备 / 续跑入口 ──────────────────────────────────────────────────
    def prepare(self, input_path: str, *, progress: Optional[ProgressFn] = None) -> RunStore:
        if progress:
            progress(0, 0, "读取原书…")
        # 超长段按句拆分（max_chars_per_segment），续段标 cont 供回填并回
        doc = load_document(
            input_path,
            self.config.source_lang,
            self.config.target_lang,
            split_segments=self.config.segment.max_chars_per_segment,
        )
        run_dir = os.path.join(self.config.state_dir, slugify(doc.title))
        store = RunStore(run_dir)
        with store.lock():
            return self._prepare_locked(doc, store, input_path, progress)

    def _prepare_locked(
        self,
        doc,
        store: RunStore,
        input_path: str,
        progress: Optional[ProgressFn],
    ) -> RunStore:
        if store.exists():
            store.log_event("run_resumed", input_path=input_path, run_dir=store.run_dir)
            return store  # 已有进度 → 直接续跑，不重置（语言在 run() 里按 manifest 应用）

        # 新建：auto 时只使用模型检测主要语言；失败则要求用户显式指定。
        if self.config.source_lang in ("auto", "", None):
            if progress:
                progress(0, 0, "识别语言…")
            detected = self._detect_language_ai(doc)
            if not detected:
                store.log_event("language_detection_failed", source_lang=doc.source_lang)
                raise RuntimeError(
                    "自动识别源语言失败：请检查模型配置，或在 config.yaml 的 "
                    "language.source 指定 ISO 639-1 语言代码（如 ja/en/ko/ru/fr/de/es）。"
                )
            doc.source_lang = detected
            store.log_event("language_detected", source_lang=doc.source_lang)
        self._apply_language(doc.source_lang)

        manifest = store.stage_document(doc)
        glossary = GlossaryStore(store.glossary_path)
        try:
            if progress:
                progress(0, 0, "分析全书风格…")
            sample = self._sample_text(doc)
            analysis = self.analyzer.analyze(sample) if sample else {}
            if analysis:
                self.analyzer.seed_glossary(glossary, analysis)
            store.save_analysis(analysis)
            store.log_event("analysis_saved", has_analysis=bool(analysis))
            store.save_context(
                RollingContext(
                    max_recent_keep=max(40, self.config.pipeline.rolling_context_segments)
                ).to_dict()
            )

            # manifest 是初始化完成标志，必须最后原子落盘。
            manifest["initialized"] = True
            store.save_manifest(manifest)
            store.log_event(
                "run_initialized",
                input_path=input_path,
                run_dir=store.run_dir,
                title=doc.title,
                fmt=doc.fmt,
                source_lang=doc.source_lang,
                target_lang=doc.target_lang,
                chapters=len(doc.chapters),
                config={
                    "review": self.config.pipeline.review,
                    "autofix_severe": self.config.pipeline.autofix_severe,
                    "polish": self.config.pipeline.polish,
                    "backtranslate_sample": self.config.pipeline.backtranslate_sample,
                    "consistency_qa": self.config.pipeline.consistency_qa,
                    "book_understanding": self.config.pipeline.book_understanding,
                },
            )
        finally:
            glossary.close()
        return store

    def _detect_language_ai(self, doc) -> str:
        """用模型检测正文主要语言，返回 ISO 代码（如 ja/en/ru）。失败返回空串。"""
        # labeled=False：纯源文样本，防多点采样的中文标签污染语言检测
        sample = self._sample_text(doc, labeled=False)[:1500]
        if not sample.strip():
            return ""
        system = (
            "你是语言识别器。判断给定文本的主要自然语言，"
            '仅输出 JSON：{"language":"<ISO 639-1 两字母代码，如 ja/en/ru/ko/fr/de/zh>"}。'
            "无法判断时 language 置为空字符串。"
        )
        try:
            data = self.client.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": sample}],
                tier="cheap",
                stage="language_detect",
                operation="language.detect",
            )
            code = (data.get("language") if isinstance(data, dict) else "") or ""
            return _normalize_lang(str(code))
        except Exception:
            return ""

    @staticmethod
    def _sample_text(doc, *, labeled: bool = True) -> str:
        """取风格分析样章。labeled=True 时多点采样（开头/中部/结尾各一段，带中文标注），
        让分析覆盖全书风格全貌；labeled=False 返回单段纯源文（语言检测用，不能混入中文标签）。"""
        texts = ["\n".join(s.source for s in ch.text_segments) for ch in doc.chapters]
        texts = [t for t in texts if len(t) > 200]
        if not texts:  # 兜底：全书都是短章
            joined = "\n".join(s.source for ch in doc.chapters[:2] for s in ch.text_segments)
            return joined[:6000]
        if not labeled:
            return texts[0][:6000]
        picks = [(0, "开头样章"), (len(texts) // 2, "中部样章"), (len(texts) - 1, "结尾样章")]
        parts: list[str] = []
        seen: set[int] = set()
        for idx, tag in picks:
            if idx in seen:  # 短书（1-2 章）去重，不重复取同一章
                continue
            seen.add(idx)
            t = texts[idx]
            chunk = t[-2800:] if tag == "结尾样章" else t[:2800]
            parts.append(f"【{tag}】\n{chunk}")
        return "\n\n".join(parts)

    def run(
        self,
        input_path: str,
        *,
        only_chapter: int | None = None,
        progress: Optional[ProgressFn] = None,
    ) -> RunStore:
        store = self.prepare(input_path, progress=progress)
        with store.lock():
            return self._run_locked(store, only_chapter=only_chapter, progress=progress)

    def _run_locked(
        self,
        store: RunStore,
        *,
        only_chapter: int | None,
        progress: Optional[ProgressFn],
    ) -> RunStore:
        manifest = store.load_manifest()
        self._apply_language(manifest.get("source_lang") or self.config.source_lang)
        chapter_indices = {chapter.get("index") for chapter in manifest.get("chapters", [])}
        if only_chapter is not None and only_chapter not in chapter_indices:
            available = sorted(index for index in chapter_indices if isinstance(index, int))
            valid_range = f"0–{available[-1]}" if available else "无可翻译章节"
            raise ValueError(f"章节编号 {only_chapter} 不存在；可用范围：{valid_range}")
        glossary = GlossaryStore(store.glossary_path)
        context = RollingContext.from_dict(
            store.load_context() or {},
            min_recent_keep=max(40, self.config.pipeline.rolling_context_segments),
        )
        style = self.analyzer.style_brief(store.load_analysis() or {})
        # 翻译前预扫源文，建立全书理解（幂等、可续跑）；全书概览注入每章翻译
        book_synopsis = self._build_understanding(store, glossary, progress=progress)
        # 附属章档位升档（skip→light/full、light→full）时重开已完成的附属章重译；
        # 旁路产物（原文副本/fast 粗翻）否则会被批级续跑当成已译整批复用。降档不回退。
        self._reopen_upgraded_back_matter(store)

        if only_chapter is not None:
            targets = [only_chapter]
        else:
            targets = store.pending_chapters()

        total = self._count_segments(store, targets)
        done = 0
        store.log_event(
            "translate_run_started",
            only_chapter=only_chapter,
            chapters=targets,
            total_segments=total,
        )
        # 共享线程池：章内批次翻译仍严格串行（关键路径唯一阻塞点）；worker 线程只做 LLM
        # 调用——1 润色在飞 + 1 术语抽取 + 各章审校任务复用同一个池；硬编码 4，YAGNI 不加配置。
        # SQLite（GlossaryStore）与 RunStore 的读写全部留在主线程（仓库既有约定）。
        # review_executor：专用于 review chunk 调用的独立池，全书共享、生命周期 = 本次
        # run()。章级审校任务（同步/异步）本身跑在 executor 的 worker 线程上，内部再把
        # 各 chunk 提交到 review_executor 并等待结果——两个池分开，避免"审校任务"和它
        # 自己要等的"chunk 调用"抢同一个 executor 的 worker 造成嵌套死锁；四个 chunk
        # 并发上限book-wide 生效（跨章节共享，不是每章各起 4 个）。
        pending_reviews: list[tuple[int, Future]] = []
        executor = ThreadPoolExecutor(max_workers=4)
        review_executor = ThreadPoolExecutor(max_workers=4)
        try:
            self._resume_pending_reviews(
                store, glossary, executor, review_executor, pending_reviews, skip=set(targets)
            )
            for ci in targets:
                done = self._translate_chapter(
                    ci,
                    store,
                    glossary,
                    context,
                    style,
                    book_synopsis,
                    executor=executor,
                    review_executor=review_executor,
                    pending_reviews=pending_reviews,
                    progress=progress,
                    done=done,
                    total=total,
                )
                store.save_context(context.to_dict())
                # 每章落一次累计用量快照（含 by_stage/by_operation），中途即可做成本归因，不必等 report
                store.log_event("usage_snapshot", chapter=ci, **self.client.usage_summary())
                # 机会性排干：只处理已完成的审校 future，不阻塞下一章翻译
                self._drain_ready_reviews(pending_reviews, store, blocking=False)
            # 全书译完后翻译各章标题和目录项（书名保持原文，借术语表保持专名一致）
            if not store.pending_chapters():
                self._translate_titles(store, glossary, progress=progress)
        finally:
            # 先排干所有在飞 future（含审校）——此时 review_executor 仍在跑，_review_chapter
            # 内部对它的 chunk future 才能真正等到结果——再依次关闭两个池；review_executor
            # 后关，保证所有 chunk 结果已写回、无孤儿任务。
            self._drain_ready_reviews(pending_reviews, store, blocking=True)
            executor.shutdown(wait=True)
            review_executor.shutdown(wait=True)
            glossary.close()
            self._flush_usage(store, scope="translate")
        if progress and total:
            progress(total, total, "翻译完成")
        store.log_event("translate_run_finished", total_segments=total)
        return store

    def _drain_ready_reviews(
        self, pending_reviews: list[tuple[int, Future]], store: RunStore, *, blocking: bool
    ) -> None:
        """写回已完成的章末审校 future（review=true 且 autofix_severe=false 的异步路径）。

        blocking=False：只处理已完成的（每章结束时机会性排干，不阻塞下一章翻译）；
        blocking=True：阻塞等全部剩余完成（run() 收尾必须调用，防止异步审校结果丢失）。
        审校本身只做 LLM 调用（worker 线程跑），落盘、发事件统一回主线程做。
        异常：记一条失败事件后跳过该章，不中断整个 run。
        """
        remaining: list[tuple[int, Future]] = []
        for ci, fut in pending_reviews:
            if not blocking and not fut.done():
                remaining.append((ci, fut))
                continue
            try:
                new_issues = fut.result()
            except Exception:
                store.log_event("chapter_review_failed", chapter=ci)
                continue
            for it in new_issues:
                it["chapter"] = ci
                it.setdefault("fixed", False)
                it["stage"] = "review"
            chapter = store.load_chapter(ci)
            lint_kept = [
                i for i in chapter.meta.get("review_issues", []) if i.get("stage") == "lint"
            ]
            chapter.meta["review_issues"] = lint_kept + new_issues
            store.save_chapter(chapter)
            store.set_review_pending(ci, False)
            store.log_event(
                "chapter_reviewed",
                chapter=ci,
                issue_count=len(new_issues),
                issues=new_issues,
            )
        pending_reviews[:] = remaining

    def _resume_pending_reviews(
        self,
        store: RunStore,
        glossary: GlossaryStore,
        executor: ThreadPoolExecutor,
        review_executor: ThreadPoolExecutor,
        pending_reviews: list[tuple[int, Future]],
        *,
        skip: set[int],
    ) -> None:
        """续跑补跑：扫描已标 done 但 review_pending 未清的章，重新提交异步审校。

        覆盖崩溃窗口——章标 done 后、异步审校结果写回前进程被杀，manifest 里
        review_pending 标记残留在磁盘，据此重跑，保证异步审校不静默丢失。
        skip：本轮 targets（会在翻译流程里自行重跑审校），避免重复提交。
        """
        for ci in store.review_pending_chapters():
            if ci in skip:
                continue
            chapter = store.load_chapter(ci)
            pairs = [(s.source, s.target or "") for s in chapter.text_segments]
            term_snapshot = self._chapter_term_snapshot(glossary, chapter.text_segments)
            fut = executor.submit(self._review_chapter, pairs, list(term_snapshot), review_executor)
            pending_reviews.append((ci, fut))

    @staticmethod
    def _count_segments(store: RunStore, chapter_indices: list[int]) -> int:
        total = 0
        for ci in chapter_indices:
            total += len(store.load_chapter(ci).text_segments)
        return total

    # ── 全书理解预扫（源文逐章梗概 + 全书概览）────────────────────────────────
    def _build_understanding(
        self, store: RunStore, glossary: GlossaryStore, progress: Optional[ProgressFn] = None
    ) -> str:
        """翻译前预扫源文：逐章梗概存入 chapter.meta，归并出全书概览存入 analysis。

        幂等、可续跑：已有梗概/概览则跳过。返回全书概览（注入各章翻译 prompt）。
        关闭 book_understanding 时直接返回空串（含一次性定名阶段，一并跳过）。
        """
        if not self.config.pipeline.book_understanding:
            store.log_event("book_understanding_skipped", reason="disabled")
            return ""
        manifest = store.load_manifest()
        chapters = manifest.get("chapters", [])
        analysis = store.load_analysis() or {}

        # 各章梗概相互独立 → 并行调用（LLM 调用进线程池；落盘全部在主线程，
        # 保持原子写不竞争，且逐章增量落盘、续跑粒度不变）。已有梗概的章跳过（幂等）。
        loaded = {
            c.get("index", i): store.load_chapter(c.get("index", i)) for i, c in enumerate(chapters)
        }
        body_chapters = [
            ci
            for ci, ch in loaded.items()
            if not self._back_matter_mode(ch.title, ci, len(chapters))
        ]
        todo = [
            (ci, "\n".join(s.source for s in loaded[ci].text_segments))
            for ci in body_chapters
            if not loaded[ci].meta.get("source_digest")
        ]

        # 一次性全书定名分支：与逐章梗概真正重叠执行——本轮不再"digest 全跑完才挖掘"。
        # mine_candidates 本身是同步阻塞调用（内部按 concurrency 自建线程池并发各章），
        # 提交进独立的单线程后台池后立即返回 future；主线程随即进入下面 digest 的
        # as_completed 落盘循环，二者的 LLM 调用天然同时在跑。SQLite/RunStore 写入
        # 全程只留在主线程（mining_pool 里的调用只做 LLM 请求，不碰 store/glossary）。
        # 幂等标记 term_mining_done：已跑过（含续跑）则跳过，不重复定名、不起后台池。
        need_mining = not analysis.get("term_mining_done")
        mining_pool: ThreadPoolExecutor | None = None
        mining_future: Future | None = None
        if need_mining:
            # 挖掘输入必须用 is_back_matter（而非 _back_matter_mode）排除附属章：
            # back_matter=full 时 _back_matter_mode 恒返回 None（不旁路），但附属章
            # 仍是附属章——引文人名/书目标题混进候选正是本次重构要消灭的污染源。
            mine_chapters = [
                ci
                for ci, ch in loaded.items()
                if not is_back_matter(ch.title, index=ci, total=len(chapters))
            ]
            src_chapters = [
                (ci, "\n".join(s.source for s in loaded[ci].text_segments)) for ci in mine_chapters
            ]
            mining_pool = ThreadPoolExecutor(max_workers=1)
            mining_future = mining_pool.submit(
                mine_candidates,
                self.config.source_lang,
                src_chapters,
                self.namer,
                concurrency=max(1, self.config.pipeline.prescan_concurrency),
                on_progress=(lambda i, n: progress(i, n, "查找专有名词…")) if progress else None,
            )

        # digest 分支：与上面的挖掘后台线程并发跑；本分支自身的异常（含线程池收尾时
        # 冒出的意外异常）保留"整体冒泡"的旧同步语义——记录后待挖掘分支排干完再抛出，
        # 不静默吞掉，也不因为并发化就丢弃尚未处理完的挖掘结果。
        digest_exc: Exception | None = None
        if todo:
            store.log_event(
                "book_understanding_chapter_digest_started",
                chapters=[ci for ci, _ in todo],
                workers=max(1, self.config.pipeline.prescan_concurrency),
            )
            workers = max(1, self.config.pipeline.prescan_concurrency)
            if progress:
                progress(0, len(todo), "通读全书章节…")
            try:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = {ex.submit(self.synopsizer.digest_chapter, src): ci for ci, src in todo}
                    for n_done, fut in enumerate(as_completed(futs), 1):
                        ci = futs[fut]
                        loaded[ci].meta["source_digest"] = (
                            fut.result()
                        )  # 失败时 _ask_text 已回退 ""
                        store.save_chapter(loaded[ci])
                        store.log_event(
                            "book_understanding_chapter_digest_saved",
                            chapter=ci,
                            digest=loaded[ci].meta["source_digest"],
                        )
                        if progress:
                            progress(n_done, len(todo), "通读全书章节…")
            except Exception as e:
                digest_exc = e

        # 按 manifest 章序组装（与并发完成顺序无关）；已落盘的 digest（即便挖掘/命名
        # 随后异常或 digest 分支本身异常）保留在 chapter.meta 里，续跑不重复调用。
        digests = [
            loaded[c.get("index", i)].meta.get("source_digest", "") or ""
            for i, c in enumerate(chapters)
        ]

        if need_mining:
            assert mining_pool is not None and mining_future is not None
            try:
                candidates = mining_future.result()
                mining_exc: Exception | None = None
            except Exception as e:
                candidates = None
                mining_exc = e
            finally:
                # 两条分支都必须排干：即便 digest 分支已经异常，也要等挖掘的后台线程
                # 池收尾完，不留孤儿任务。
                mining_pool.shutdown(wait=True)

            if digest_exc is not None:
                # digest 异常优先级更高，与旧版"digest 先跑、失败直接冒泡"的同步行为
                # 一致；挖掘分支的结果/异常本身不再单独处理（term_mining_done 也不落盘）。
                raise digest_exc

            if mining_exc is not None:
                store.log_event("cast_naming_failed", error=str(mining_exc))
                named = None
            else:
                store.log_event("term_candidates_mined", count=len(candidates))
                existing = glossary.all_terms()
                try:
                    named = self.namer.name_terms(
                        candidates,
                        self.analyzer.style_brief(analysis),
                        digests,
                        existing=existing,
                        concurrency=max(1, self.config.pipeline.prescan_concurrency),
                        on_progress=(lambda i, n: progress(i, n, "统一译名…"))
                        if progress
                        else None,
                    )
                except Exception as e:
                    store.log_event("cast_naming_failed", error=str(e))
                    named = None
            if named is not None:
                inserted = 0
                for t in named:
                    result = glossary.upsert_term(t, chapter=0)
                    if result in ("inserted", "updated"):
                        inserted += 1
                    if t.type == TYPE_PERSON:
                        # namer 确认沿用已有译法时（seed_glossary 先种入的 medium/未锁）
                        # upsert_term 的同译法分支不会升级 locked/confidence，这里显式
                        # 补一次；仅当当前 target 与确认值一致才生效，防止锁错译法。
                        glossary.confirm_locked(t.source, t.target)
                analysis["term_mining_done"] = True
                store.save_analysis(analysis)
                store.log_event("cast_named", count=inserted)
        elif digest_exc is not None:
            raise digest_exc

        synopsis = analysis.get("book_synopsis", "")
        if not synopsis and any(d.strip() for d in digests):
            if progress:
                progress(0, 0, "生成全书概览…")
            cast_text = prompts.render_glossary(
                [t for t in glossary.all_terms() if t.type == TYPE_PERSON]
            )
            synopsis = self.synopsizer.book_synopsis(
                digests, self.analyzer.style_brief(analysis), cast=cast_text
            )
            analysis["book_synopsis"] = synopsis
            store.save_analysis(analysis)
            store.log_event("book_synopsis_saved", synopsis=synopsis)
        return synopsis

    # ── 章节标题 / 目录项翻译（书名保持原文）──────────────────────────────
    def _translate_titles(
        self, store: RunStore, glossary: GlossaryStore, progress: Optional[ProgressFn] = None
    ) -> None:
        """把各章标题和额外目录项翻成中文，写回 manifest（幂等：已全部译过则跳过）。

        书名保持原文，不写 title_translated；借术语表保证章节标题里的专名一致。
        """
        from ..agents import prompts

        m = store.load_manifest()
        chapters = m.get("chapters", [])

        # 标题压成单行，避免内嵌换行破坏 numbered 对齐
        def _flat(s: object) -> str:
            return " ".join(str(s or "").split())

        raw_meta = m.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        chapter_hrefs = {c.get("href") for c in chapters if c.get("href")}
        raw_toc_entries = meta.get("toc_entries", [])
        toc_entry_items = raw_toc_entries if isinstance(raw_toc_entries, list) else []
        toc_entries = [
            e
            for e in toc_entry_items
            if isinstance(e, dict)
            and e.get("href") not in chapter_hrefs
            and _flat(e.get("title", ""))
        ]

        titled_chapters = [c for c in chapters if _flat(c.get("title", ""))]
        m.pop("title_translated", None)

        # 正文标题复用：首段是已译 heading 的章，标题直接取该段译文（与正文用词一致，
        # 避免独立标题 agent 无上下文另起译法），不进 LLM 列表。
        llm_chapters = []
        for c in titled_chapters:
            chapter = store.load_chapter(c["index"])
            segs = chapter.segments
            heading_target = _flat(segs[0].target) if segs and segs[0].kind == KIND_HEADING else ""
            if heading_target:
                c["title_translated"] = heading_target
            else:
                llm_chapters.append(c)
        store.save_manifest(m)  # 先落盘复用结果，即便后续 LLM 调用失败也不丢失

        if all(c.get("title_translated") for c in llm_chapters) and all(
            e.get("title_translated") for e in toc_entries
        ):
            store.log_event("titles_skipped", reason="already_translated")
            return  # 已译（含复用），断点续跑不重复调用

        titles = [_flat(c.get("title", "")) for c in llm_chapters] + [
            _flat(e.get("title", "")) for e in toc_entries
        ]
        if not any(t.strip() for t in titles):
            return  # 全部复用/已译，LLM 列表为空，不发请求
        if progress:
            progress(0, 0, "翻译章节标题…")
        analysis = store.load_analysis() or {}
        book_synopsis = analysis.get("book_synopsis") or "（无）"
        system = prompts.render(
            "title_translator_system",
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            n=len(titles),
        )
        user = prompts.render(
            "title_translator_user",
            src=self.config.source_lang,
            tgt=self.config.target_lang,
            book_synopsis=book_synopsis,
            glossary=prompts.render_glossary(glossary.all_terms()),
            n=len(titles),
            numbered_titles=prompts.numbered(titles),
        )
        try:
            data = self.client.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tier="strong",
                stage="title_translate",
                operation="title.translate",
            )
        except Exception:
            return
        out = data.get("titles") if isinstance(data, dict) else data
        if not isinstance(out, list) or len(out) != len(titles):
            store.log_event(
                "titles_translation_rejected",
                reason="count_mismatch",
                expected=len(titles),
                actual=len(out) if isinstance(out, list) else None,
            )
            return
        out = [str(t).strip() for t in out]
        chapter_out = out[: len(llm_chapters)]
        toc_out = out[len(llm_chapters) :]
        for c, t in zip(llm_chapters, chapter_out):
            c["title_translated"] = t or c.get("title")
        for e, t in zip(toc_entries, toc_out):
            e["title_translated"] = t or e.get("title")
        store.save_manifest(m)
        store.log_event(
            "titles_translated",
            titles=[
                {"index": i - 1, "source": src, "target": tgt}
                for i, (src, tgt) in enumerate(zip(titles, out))
            ],
        )

    # ── 单章 ──────────────────────────────────────────────────────────────

    def _back_matter_mode(self, title: str, index: int, total: int) -> str | None:
        """附属章旁路档位：skip/light 且标题+位置命中时返回该档；否则 None（完整流水线）。

        full 档不旁路（但仍不抽术语，见 _translate_chapter）。识别含位置门控：
        正文区标题撞词（如 "The Index Case"）不旁路，防整章静默降质。
        """
        mode = self.config.pipeline.back_matter
        if mode in ("skip", "light") and is_back_matter(title, index=index, total=total):
            return mode
        return None

    _BM_RANK = {"skip": 0, "light": 1, "full": 2}

    def _reopen_upgraded_back_matter(self, store: RunStore) -> None:
        """附属章档位升档时重开已完成的附属章（skip→light/full、light→full）。

        旁路档的 target（skip=原文副本、light=fast 粗翻）非空，会被批级续跑当成
        已译整批复用——不清掉就升档形同虚设。降档不回退：更高质量译文保留。
        位置门控收紧后不再命中的章同样按升档到 full 处理（此前属误伤）。
        """
        m = store.load_manifest()
        chapters = m.get("chapters", [])
        n = len(chapters)
        for c in chapters:
            if c.get("status") != STATUS_DONE:
                continue
            ch = store.load_chapter(c["index"])
            prev = ch.meta.get("back_matter_mode")
            if prev not in self._BM_RANK:
                continue
            cur = self._back_matter_mode(c.get("title", ""), c["index"], n) or "full"
            if self._BM_RANK[cur] <= self._BM_RANK[prev]:
                continue
            for s in ch.segments:
                s.target = None
            ch.meta.pop("back_matter_mode", None)
            ch.meta.pop("pending_polish", None)
            ch.meta.pop("review_issues", None)
            ch.meta.pop("backtranslation_issues", None)
            store.save_chapter(ch)
            store.set_chapter_status(c["index"], STATUS_PENDING)
            store.log_event(
                "back_matter_reopened",
                chapter=c["index"],
                title=c.get("title", ""),
                prev_mode=prev,
                mode=cur,
            )

    def _translate_back_matter(
        self, mode, ci, chapter, text_segs, store, *, progress=None, done=0, total=0
    ) -> int:
        """附属章旁路：skip=原文直通；light=fast 档粗翻。不碰 glossary/context/style/executor。"""
        label = f"第{ci}章 {chapter.title}"
        store.log_event("chapter_back_matter", chapter=ci, title=chapter.title, mode=mode)

        if mode == "skip":
            for s in text_segs:
                s.target = s.source
            store.save_chapter(chapter)
            done += len(text_segs)
            if progress:
                progress(done, total, label)
        elif mode == "light":
            batches = batch_segments(text_segs, self.config.segment.max_chars_per_batch)
            seg_base = 0
            for b in batches:
                existing = [s.target for s in b if s.target and s.target.strip()]
                if len(existing) == len(b):
                    done += len(b)
                    seg_base += len(b)
                    if progress:
                        progress(done, total, label)
                    continue
                raw = self.translator.translate_batch(
                    [s.source for s in b],
                    glossary_terms=[],
                    style="",
                    context="",
                    book_synopsis="",
                    chapter_digest="",
                    tier="fast",
                )
                if self.config.punctuation_normalize:
                    raw = [normalize_zh(t) if t else t for t in raw]
                for s, t in zip(b, raw):
                    s.target = t
                # 先落盘再记事件（与正文路径相反）：崩溃窗口只漏一条事件，
                # 续跑按已落盘 target 整批复用，不重发 fast 调用。
                store.save_chapter(chapter)
                store.log_event(
                    "batch_translated",
                    chapter=ci,
                    start_index=seg_base,
                    count=len(b),
                    polished=False,
                    punctuation_normalized=self.config.punctuation_normalize,
                    back_matter=True,
                    tier="fast",
                    segments=[
                        {"index": seg_base + i, "source": s.source, "target": t}
                        for i, (s, t) in enumerate(zip(b, raw))
                    ],
                )
                done += len(b)
                seg_base += len(b)
                if progress:
                    progress(done, total, label)

        # 记录旁路档位：report 上报给人工复核；升档续跑据此重开本章。
        # 顺带清掉旧版完整流水线半跑遗留的润色标记（旁路档永不消费它）。
        chapter.meta["back_matter_mode"] = mode
        chapter.meta.pop("pending_polish", None)
        chapter.meta["review_issues"] = []
        chapter.meta["backtranslation_issues"] = []
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_DONE)
        store.log_event(
            "chapter_done",
            chapter=ci,
            title=chapter.title,
            segment_count=len(text_segs),
            review_issue_count=0,
            backtranslation_issue_count=0,
            back_matter=True,
            mode=mode,
        )
        return done

    def _translate_chapter(
        self,
        ci: int,
        store: RunStore,
        glossary: GlossaryStore,
        context: RollingContext,
        style: str,
        book_synopsis: str = "",
        *,
        executor: ThreadPoolExecutor,
        review_executor: ThreadPoolExecutor,
        pending_reviews: list[tuple[int, Future]],
        progress: Optional[ProgressFn] = None,
        done: int = 0,
        total: int = 0,
    ) -> int:
        chapter = store.load_chapter(ci)
        text_segs = chapter.text_segments
        if not text_segs:
            store.set_chapter_status(ci, STATUS_DONE)
            store.log_event("chapter_skipped", chapter=ci, reason="empty")
            return done
        n_ch = len(store.load_manifest()["chapters"])
        bm_mode = self._back_matter_mode(chapter.title, ci, n_ch)
        if bm_mode:
            return self._translate_back_matter(
                bm_mode, ci, chapter, text_segs, store, progress=progress, done=done, total=total
            )
        # full/正文路径：附属章照常翻译，但不抽术语（skip/light 已在上方旁路返回）。
        bm = is_back_matter(chapter.title, index=ci, total=n_ch)
        # 走到完整流水线就清掉旁路痕迹（换档/门控收紧后 report 不再误报）。
        chapter.meta.pop("back_matter_mode", None)
        chapter_digest = chapter.meta.get("source_digest", "")

        batches = _resume_batches(text_segs, self.config.segment.max_chars_per_batch)
        label = f"第{ci}章 {chapter.title}"
        # prepare() 的最后一个标签通常是“解析文档…”。续跑首批可能先恢复术语，
        # 若不在章首刷新，整个模型请求期间都会错误地显示成仍在解析源文件。
        if progress:
            progress(done, total, label)
        # 章内术语快照会在每个批次术语抽取后刷新，让新确认的称呼/口癖/固定表达
        # 立即影响后续批次。glossary_scope=chapter 时仍按本章源文裁剪，避免全量表过大。
        term_snapshot = self._chapter_term_snapshot(glossary, text_segs)

        # 逐批串行：每批渲染最新上下文 → 翻译（强档，本函数唯一阻塞关键路径的调用）→
        # 立即落盘 → 把 raw 译文并入滚动上下文供下一批参照。术语抽取因下一批要用新词，
        # 落盘后在主线程同步做（fast 档，耗时远小于下一批翻译，不入池避免被在飞 future 阻塞）；
        # 润色批间无依赖，提交后不等，全部挪到章末统一排干（不阻塞下一批翻译）。
        # 断点续跑（段/批级）：上次中断前已译完并落盘的批次，整批跳过、不重翻，只重建上下文。
        review_issues: list[dict] = [
            i
            for i in chapter.meta.get("review_issues", [])
            if i.get("stage") not in ("length", "lint")
        ]
        # lint 层（确定性、零 LLM）产出的未解决 issue 单独累积，最终并入 review_issues：
        # review 分支会重置/异步覆盖 review_issues 本身，须避免被那条通道悄悄冲掉。
        lint_review_issues: list[dict] = []
        polish_on = self.config.pipeline.polish
        # start_index → 本轮新提交的润色 future；章末排干时与 chapter.meta["pending_polish"]
        # （含续跑遗留、本轮未重译的批次）合并处理，保证续跑不丢润色。
        polish_futures: dict[int, Future] = {}
        seg_base = 0  # 当前批首段的章内段号（issue 批内下标 → 章内段号）
        for b in batches:
            existing_targets = [s.target for s in b if s.target and s.target.strip()]
            if len(existing_targets) == len(b):
                # 该批上次已在原位、原上下文中译完 → 复用，重建滚动上下文后跳过；
                # 抽取保持同步现状，遗留的 pending_polish 标记留到章末统一排干。
                context.add_targets(existing_targets)
                summary = None
                if not bm and self.config.pipeline.inflight_glossary:
                    summary, changed = self._extract_batch_glossary(
                        glossary, store, ci, seg_base, b
                    )
                    # 与新译批一致的条件刷新：新词命中本章剩余源文才重建快照，保前缀缓存。
                    remaining_src = "\n".join(s.source for s in text_segs[seg_base + len(b) :])
                    if changed and GlossaryStore.terms_in(changed, remaining_src):
                        term_snapshot = self._chapter_term_snapshot(glossary, text_segs)
                # 崩溃续跑：跳过批次不重译（保持跳过语义），但确定性 lint 零成本，
                # 仍跑一遍记录未修复项，防止崩溃窗口下确定性问题永久漏检。
                locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
                for it in lint.lint_targets(
                    [s.source for s in b],
                    existing_targets,
                    locked_terms=locked,
                    src_lang=self.config.source_lang,
                ):
                    lint_review_issues.append(
                        {
                            "chapter": ci,
                            "index": seg_base + it.index,
                            "type": it.type,
                            "detail": it.detail,
                            "stage": "lint",
                            "fixed": False,
                        }
                    )
                store.log_event(
                    "batch_skipped",
                    chapter=ci,
                    start_index=seg_base,
                    count=len(b),
                    reason="already_translated",
                    glossary_extraction=summary,
                    segments=[
                        {"index": seg_base + i, "source": s.source, "target": s.target}
                        for i, s in enumerate(b)
                    ],
                )
                done += len(b)
                seg_base += len(b)
                if progress:
                    progress(done, total, label)
                continue

            ctx_text = context.render(self.config.pipeline.rolling_context_segments)
            raw_targets = self._process_batch(
                b, term_snapshot, ctx_text, style, book_synopsis, chapter_digest
            )

            # 确定性 lint（零 LLM，宁漏勿误报）：引号丢失/数字失配/锁定专名漂移/
            # 未译残留 + 复用的长度异常。flag 段带审校意见定向重译，每段最多一轮，
            # 采纳条件=重译后 issue 数严格减少（防越修越糟）。
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            lint_issues = lint.lint_targets(
                [s.source for s in b],
                raw_targets,
                locked_terms=locked,
                src_lang=self.config.source_lang,
            )
            if lint_issues:
                store.log_event(
                    "batch_linted",
                    chapter=ci,
                    start_index=seg_base,
                    issues=[
                        {"index": seg_base + it.index, "type": it.type, "detail": it.detail}
                        for it in lint_issues
                    ],
                )
                by_idx: dict[int, list] = {}
                for it in lint_issues:
                    by_idx.setdefault(it.index, []).append(it)
                for idx, seg_issues in sorted(by_idx.items()):
                    if not any(it.type in lint.ACTIONABLE_TYPES for it in seg_issues):
                        # too_short/too_long 等非定向重译类型：只记录，不重译
                        # （en→zh 合法压缩比波动实测极大，交由人工/审校 agent 判断）。
                        for it in seg_issues:
                            lint_review_issues.append(
                                {
                                    "chapter": ci,
                                    "index": seg_base + idx,
                                    "type": it.type,
                                    "detail": it.detail,
                                    "stage": "lint",
                                    "fixed": False,
                                }
                            )
                        continue
                    seg = b[idx]
                    feedback = "；".join(it.detail for it in seg_issues)
                    before = "\n".join(raw_targets[j] for j in range(max(0, idx - 2), idx))
                    after = "\n".join(raw_targets[j] for j in range(idx + 1, min(len(b), idx + 3)))
                    new_t = self.translator.retranslate_with_feedback(
                        seg.source,
                        feedback=feedback,
                        operation="translate.lint_fix",
                        glossary_terms=term_snapshot,
                        style=style,
                        context_before=before,
                        context_after=after,
                        book_synopsis=book_synopsis,
                        chapter_digest=chapter_digest,
                    )
                    new_issues = (
                        lint.lint_targets(
                            [seg.source],
                            [new_t],
                            locked_terms=locked,
                            src_lang=self.config.source_lang,
                        )
                        if new_t
                        else []
                    )
                    if new_t and len(new_issues) < len(seg_issues):
                        self.client.usage.record_outcome("translate.lint_fix", accepted=True)
                        store.log_event(
                            "lint_refixed",
                            chapter=ci,
                            index=seg_base + idx,
                            before=raw_targets[idx],
                            after=new_t,
                            issues=[{"type": it.type, "detail": it.detail} for it in seg_issues],
                        )
                        raw_targets[idx] = new_t
                        remaining = new_issues
                    else:
                        self.client.usage.record_outcome("translate.lint_fix", accepted=False)
                        remaining = seg_issues
                    for it in remaining:
                        lint_review_issues.append(
                            {
                                "chapter": ci,
                                "index": seg_base + idx,
                                "type": it.type,
                                "detail": it.detail,
                                "stage": "lint",
                                "fixed": False,
                            }
                        )
            for s, t in zip(b, raw_targets):
                s.target = t
            batch_start = seg_base
            done += len(b)
            seg_base += len(b)

            if polish_on:
                # 滚动上下文改喂未润色译文：polisher 批间无依赖，容忍上下文带原始腔调。
                context.add_targets(raw_targets)
                chapter.meta.setdefault("pending_polish", []).append(
                    {"start": batch_start, "count": len(b)}
                )
                event_targets = raw_targets
                punctuation_normalized = False
            else:
                # polish 关闭：保持现行为，翻译后立即标点规范化，无 pending 标记。
                final_targets = raw_targets
                if self.config.punctuation_normalize:
                    final_targets = [normalize_zh(t) if t else t for t in final_targets]
                    for s, t in zip(b, final_targets):
                        s.target = t
                context.add_targets(final_targets)
                event_targets = final_targets
                punctuation_normalized = self.config.punctuation_normalize

            store.log_event(
                "batch_translated",
                chapter=ci,
                start_index=batch_start,
                count=len(b),
                polished=False,  # 润色异步/关闭时此刻均未完成；结果见 batch_polished 事件
                punctuation_normalized=punctuation_normalized,
                segments=[
                    {"index": batch_start + i, "source": s.source, "target": t}
                    for i, (s, t) in enumerate(zip(b, event_targets))
                ],
            )
            # 增量持久化：本批译文（+ pending_polish 标记）立即落盘，下次中断从此批之后
            # 续跑；crash-safe 保住强档翻译成果——不领先落盘的只有术语库入库（见下）。
            chapter.meta["review_issues"] = review_issues
            store.save_chapter(chapter)

            if polish_on:
                # 润色批间无依赖：提交后不等待，章末统一排干，不阻塞下一批翻译。
                polish_futures[batch_start] = executor.submit(
                    self.polisher.polish,
                    list(raw_targets),
                    [s.source for s in b],
                    glossary_terms=list(term_snapshot),
                    style=style,
                )

            # 术语抽取：existing 按本批源文裁剪；落盘后在主线程同步抽取
            # （fast 档，耗时远小于下一批强档翻译），不进共享池——否则会被在飞的润色/审校
            # future 占满 4 worker 时反向阻塞主线程，把后台工作拖回翻译关键路径。保住
            # 不变量 (a)（落库不领先落盘）与 (d)（下一批可见上一批新词）。
            # 附属章 full 档跳过抽取；新词仅当命中剩余源文时才刷新快照以保前缀缓存。
            # inflight_glossary=False（新默认）：术语只来自翻译前一次性定名，本处不抽取。
            if not bm and self.config.pipeline.inflight_glossary:
                batch_src = "\n".join(s.source for s in b)
                terms = self.extractor.extract(
                    batch_src,
                    "\n".join(raw_targets),
                    GlossaryStore.terms_in(term_snapshot, batch_src),
                )
                summary, changed = self.extractor.store_terms(glossary, terms, ci)
                remaining_src = "\n".join(s.source for s in text_segs[seg_base:])
                if changed and GlossaryStore.terms_in(changed, remaining_src):
                    term_snapshot = self._chapter_term_snapshot(glossary, text_segs)
                store.log_event(
                    "batch_glossary_extracted",
                    chapter=ci,
                    start_index=batch_start,
                    count=len(b),
                    summary=summary,
                )
            if progress:
                progress(done, total, label)

        # 章末：排干本章全部润色 future（本轮新提交的 + 续跑遗留在 pending_polish 里的），
        # 写回最终译文、清掉标记；每批清完即落盘，保证续跑不丢润色（不变量 b）。
        self._drain_chapter_polish(
            chapter, text_segs, polish_futures, executor, style, term_snapshot, store, ci
        )

        # 全章术语抽取入库：保留为兜底，捕捉跨段才能确认的称呼/口癖/固定表达；
        # 在润色后的最终文本上跑，放在 review 前让本章审校也能用上兜底抽出的术语。
        # 附属章 full 档不抽词；inflight_glossary=False（新默认）时整段跳过（不刷新快照）。
        if not bm and self.config.pipeline.inflight_glossary:
            src_text = "\n".join(s.source for s in text_segs)
            tgt_text = "\n".join(s.target or "" for s in text_segs)
            self.extractor.extract_and_store(glossary, src_text, tgt_text, ci)
            term_snapshot = self._chapter_term_snapshot(glossary, text_segs)
            store.log_event("chapter_glossary_extracted", chapter=ci)

        # 去翻译腔（三道关卡，见 naturalizer.py）：插入点在润色写回完成之后、章末审校
        # 之前，让审校能看到去腔后的最终文本。幂等靠 chapter.meta["naturalized"]——
        # 续跑时已处理过的章节不重复跑。back_matter 旁路由 naturalize_chapter 内部
        # 按 is_back_matter 过滤，此处不重复判断。标记与改写在 naturalize_chapter
        # 内部同一次 save_chapter 中一并落盘（避免两次保存之间崩溃导致的续跑重复审读）。
        if self.config.pipeline.naturalize and not chapter.meta.get("naturalized"):
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            naturalize_chapter(
                self.naturalizer,
                chapter,
                ci,
                n_ch,
                locked,
                self.config,
                store,
                dry_run=False,
                remaining=None,
            )

        # ── 章末整章审校（块内 index 映射回章内段号）──
        # 幂等：续跑重入章末时清掉旧审校项，防重复累积。
        if self.config.pipeline.review:
            review_issues = []
            pairs = [(s.source, s.target or "") for s in text_segs]
            if self.config.pipeline.autofix_severe:
                # 严重项定向重译要写回正文，必须留在关键路径上，完全同步（现状不变）。
                new_issues = self._review_chapter(pairs, term_snapshot, review_executor)
                store.log_event(
                    "chapter_reviewed",
                    chapter=ci,
                    issue_count=len(new_issues),
                    issues=new_issues,
                )
                self._autofix_severe(
                    text_segs,
                    new_issues,
                    term_snapshot,
                    style,
                    book_synopsis,
                    chapter_digest,
                    store=store,
                    chapter_index=ci,
                )
                for it in new_issues:
                    it["chapter"] = ci
                    it.setdefault("fixed", False)
                    it["stage"] = "review"
                review_issues.extend(new_issues)
            else:
                # 审校异步：提交线程池，本章不等待；future 与 ci 一起挂到 run 级列表，
                # 由 run() 机会性/收尾统一排干、写回 review_issues、发 chapter_reviewed 事件。
                # set_review_pending 落持久标记（写 manifest，随后 set STATUS_DONE 会保留它）：
                # 崩溃发生在标 done 后、审校写回前时，续跑据此补跑，异步审校结果不静默丢失。
                review_future = executor.submit(
                    self._review_chapter, pairs, list(term_snapshot), review_executor
                )
                pending_reviews.append((ci, review_future))
                store.set_review_pending(ci, True)

        # 回译抽检：从最终（润色/规范化后）文本抽样，默认关闭（rate=0），不值得异步化。
        bt_samples: list[tuple[str, str]] = []
        rate = self.config.pipeline.backtranslate_sample
        if rate > 0:
            for seg in text_segs:
                if random.random() < rate:
                    bt_samples.append((seg.source, seg.target or ""))
        bt_issues: list[dict] = []
        if bt_samples:
            srcs = [a for a, _ in bt_samples]
            tgts = [b for _, b in bt_samples]
            for it in self.backtrans.check(srcs, tgts):
                it["chapter"] = ci
                bt_issues.append(it)
            store.log_event(
                "chapter_backtranslation_checked",
                chapter=ci,
                sample_count=len(bt_samples),
                issue_count=len(bt_issues),
                issues=bt_issues,
            )

        # 翻译记忆库（仅作记录/参考，不用于跨位置复用译文）
        for s in text_segs:
            if s.target:
                glossary.add_tm(s.source, s.target, ci)

        chapter.meta["review_issues"] = review_issues + lint_review_issues
        chapter.meta["backtranslation_issues"] = bt_issues
        store.save_chapter(chapter)
        store.set_chapter_status(ci, STATUS_DONE)
        store.log_event(
            "chapter_done",
            chapter=ci,
            title=chapter.title,
            segment_count=len(text_segs),
            review_issue_count=len(review_issues),
            backtranslation_issue_count=len(bt_issues),
        )
        return done

    def _drain_chapter_polish(
        self,
        chapter,
        text_segs,
        futures_by_start: dict[int, Future],
        executor: ThreadPoolExecutor,
        style: str,
        term_snapshot,
        store: RunStore,
        ci: int,
    ) -> None:
        """章末排干本章全部润色 future：本轮新提交的 + 上轮中断遗留在 pending_polish 里的。

        遗留批次（本轮走了批跳过路径、未重译）没有对应 future，从已落盘的 raw 译文
        重新提交润色。写回按批序进行；每批清完立即落盘，保证续跑不丢润色（不变量 b）。
        异常（超出 Polisher 自身的失败兜底）时结果回退 raw，不阻断整章。
        """
        pending = list(chapter.meta.get("pending_polish", []))
        if not pending:
            return
        for entry in pending:
            start = entry["start"]
            if start not in futures_by_start:
                count = entry["count"]
                raw = [text_segs[start + i].target or "" for i in range(count)]
                futures_by_start[start] = executor.submit(
                    self.polisher.polish,
                    raw,
                    [text_segs[start + i].source for i in range(count)],
                    glossary_terms=list(term_snapshot),
                    style=style,
                )
        for entry in sorted(pending, key=lambda e: e["start"]):
            start, count = entry["start"], entry["count"]
            fut = futures_by_start.pop(start, None)
            raw = [text_segs[start + i].target or "" for i in range(count)]
            try:
                final = fut.result() if fut is not None else raw
            except Exception:
                final = raw  # Polisher 本身失败已回退原文；这里再兜一层意外异常
            if self.config.punctuation_normalize:
                final = [normalize_zh(t) if t else t for t in final]
            # 润色回退：final 若比 raw（同样先 normalize_zh，公平比较）多引入新的 lint
            # issue 类型（如剥掉引号/丢了锁定专名），该段保留 raw，发 polish_rejected。
            srcs = [text_segs[start + i].source for i in range(count)]
            raw_normalized = (
                [normalize_zh(t) if t else t for t in raw]
                if self.config.punctuation_normalize
                else raw
            )
            locked = [t for t in term_snapshot if getattr(t, "locked", 0)]
            raw_types: dict[int, set[str]] = {}
            for it in lint.lint_targets(
                srcs, raw_normalized, locked_terms=locked, src_lang=self.config.source_lang
            ):
                raw_types.setdefault(it.index, set()).add(it.type)
            final_types: dict[int, set[str]] = {}
            for it in lint.lint_targets(
                srcs, final, locked_terms=locked, src_lang=self.config.source_lang
            ):
                final_types.setdefault(it.index, set()).add(it.type)
            for i in range(count):
                introduced = final_types.get(i, set()) - raw_types.get(i, set())
                if not introduced:
                    self.client.usage.record_outcome("polish.batch", accepted=True)
                    continue
                self.client.usage.record_outcome("polish.batch", accepted=False)
                rejected_text = final[i]
                final[i] = raw_normalized[i]
                store.log_event(
                    "polish_rejected",
                    chapter=ci,
                    index=start + i,
                    reason=sorted(introduced),
                    polished=rejected_text,
                )
            for i, t in enumerate(final):
                text_segs[start + i].target = t
            chapter.meta["pending_polish"] = [
                e for e in chapter.meta.get("pending_polish", []) if e.get("start") != start
            ]
            store.log_event(
                "batch_polished",
                chapter=ci,
                start_index=start,
                count=count,
                segments=[
                    {"index": start + i, "source": text_segs[start + i].source, "target": t}
                    for i, t in enumerate(final)
                ],
            )
            store.save_chapter(chapter)

    def _chapter_term_snapshot(self, glossary: GlossaryStore, text_segs) -> list:
        """返回当前章节要注入的术语快照；实时入库后可重新调用刷新。

        chapter 范围 = 本章源文命中的词条（terms_in 按 source/alias 全文匹配）
        + 以「部分形式」出现的锁定人物（只呼姓/名，见 _person_mentioned）。
        锁定人物不做无条件全量兜底：非虚构书定名动辄数百个一次性轶事人名
        （实测 357 条 ≈ 1.1 万字符），全量注入会占翻译/润色/审校 prompt 的七成以上，
        且对本章翻译是纯噪声。
        """
        terms = glossary.all_terms()
        if self.config.pipeline.glossary_scope != "chapter":
            return terms
        src_text = "\n".join(s.source for s in text_segs)
        hit = {t.source for t in GlossaryStore.terms_in(terms, src_text)}
        words = set(_WORD_RE.findall(src_text))
        return [
            t
            for t in terms
            if t.source in hit
            or (t.type == TYPE_PERSON and t.locked and _person_mentioned(t, src_text, words))
        ]

    def _extract_batch_glossary(
        self,
        glossary: GlossaryStore,
        store: RunStore,
        chapter: int,
        start_index: int,
        batch,
    ) -> tuple[dict[str, int], list]:
        """续跑批跳过时同步抽取术语入库（新译批次的抽取已挪到批循环内联的异步流程），
        供同章后续批次使用。返回 (入库汇总, inserted/updated 词条) 供条件刷新。"""
        src_text = "\n".join(s.source for s in batch)
        tgt_text = "\n".join(s.target or "" for s in batch)
        summary, changed = self.extractor.extract_and_store(glossary, src_text, tgt_text, chapter)
        store.log_event(
            "batch_glossary_extracted",
            chapter=chapter,
            start_index=start_index,
            count=len(batch),
            summary=summary,
        )
        return summary, changed

    # ── 章末审校 + 严重项定向重译 ────────────────────────────────────────────
    _SEVERE_TYPES = ("missing", "mistranslation")

    def _review_chapter(
        self, pairs: list[tuple[str, str]], terms, review_executor: ThreadPoolExecutor
    ) -> list[dict]:
        """整章分块审校（章末统一做）。review=true 且 autofix_severe=false 时在
        worker 线程里跑（只做 LLM 调用，不碰 store）；autofix_severe=true 时同步跑。

        pairs：章内段的 (source, target) 纯数据副本，与 Segment 对象解耦，可安全跨线程传递。
        块 = 连续段序列（约 3 倍翻译批大小，减少调用次数与重复注入的输入 token）；
        块内 reviewer 返回的 index 是块内下标，加块首段偏移映射回章内段号；
        越界 index 直接丢弃（模型幻觉防御）。

        各 chunk 的 review 调用提交进 review_executor（全书共享、有界 4-worker，
        与本方法运行所在的线程池分开——本方法自身可能就跑在另一个 executor 的
        worker 线程上，若两者共用同一个池，chunk 调用会在等自己的父任务腾位置，
        造成嵌套死锁）。按提交顺序（=chunk 原始顺序）依次取结果，而非按完成顺序，
        保证合并结果严格保持原 chunk 顺序；chunk 内 issue 顺序由单次 reviewer.review
        调用本身决定，不受并发影响。
        """
        budget = self.config.segment.max_chars_per_batch * 3
        chunks = self._pack_contiguous(pairs, budget)
        futures = [
            review_executor.submit(
                self.reviewer.review, [s for s, _ in chunk], [t for _, t in chunk], terms
            )
            for chunk in chunks
        ]
        issues: list[dict] = []
        base = 0
        for chunk, fut in zip(chunks, futures):
            for it in fut.result():
                idx = it.get("index")
                if isinstance(idx, int) and 0 <= idx < len(chunk):
                    it["index"] = base + idx
                    issues.append(it)
            base += len(chunk)
        return issues

    @staticmethod
    def _pack_contiguous(pairs: list[tuple[str, str]], budget: int) -> list[list]:
        """按源文字符预算把 (source, target) 对保序打包成若干连续块。"""
        chunks: list[list] = []
        cur: list = []
        size = 0
        for p in pairs:
            src = p[0]
            if cur and size + len(src) > budget:
                chunks.append(cur)
                cur, size = [], 0
            cur.append(p)
            size += len(src)
        if cur:
            chunks.append(cur)
        return chunks

    def _autofix_severe(
        self,
        text_segs,
        issues,
        terms,
        style,
        book_synopsis: str = "",
        chapter_digest: str = "",
        *,
        store: RunStore | None = None,
        chapter_index: int | None = None,
    ) -> None:
        """对审校严重项（漏译/误译）带审校意见定向重译，每段最多一次。

        采纳条件 = 重译非空且过长度校验：采纳则标点规范化后更新 seg.target 并标 fixed=True；
        不采纳保持 fixed=False 留人工。章末重译时原滚动上下文已失效，用该段前后各 2 段译文做局部上下文。
        """
        by_seg: dict[int, list[dict]] = {}
        for it in issues:
            if it.get("type") in self._SEVERE_TYPES:
                by_seg.setdefault(it["index"], []).append(it)
        for idx, seg_issues in sorted(by_seg.items()):
            seg = text_segs[idx]
            before = "\n".join(text_segs[j].target or "" for j in range(max(0, idx - 2), idx))
            after = "\n".join(
                text_segs[j].target or "" for j in range(idx + 1, min(len(text_segs), idx + 3))
            )
            feedback = "；".join(
                f"{it.get('detail', '')}（建议：{it.get('suggestion', '')}）" for it in seg_issues
            )
            new_t = self.translator.retranslate_with_feedback(
                seg.source,
                feedback=feedback,
                operation="translate.review_fix",
                glossary_terms=terms,
                style=style,
                context_before=before,
                context_after=after,
                book_synopsis=book_synopsis,
                chapter_digest=chapter_digest,
            )
            accepted = bool(new_t) and not checks.length_flags([seg.source], [new_t])
            self.client.usage.record_outcome("translate.review_fix", accepted=accepted)
            if accepted:
                if self.config.punctuation_normalize:
                    new_t = normalize_zh(new_t)
                old_t = seg.target
                seg.target = new_t
                for it in seg_issues:
                    it["fixed"] = True
                if store is not None:
                    store.log_event(
                        "autofix_applied",
                        chapter=chapter_index,
                        index=idx,
                        source=seg.source,
                        before=old_t,
                        after=new_t,
                        issues=seg_issues,
                    )
            elif store is not None:
                store.log_event(
                    "autofix_rejected",
                    chapter=chapter_index,
                    index=idx,
                    source=seg.source,
                    before=seg.target,
                    proposed=new_t,
                    issues=seg_issues,
                )

    def _process_batch(
        self,
        batch,
        terms,
        ctx_text: str,
        style: str,
        book_synopsis: str = "",
        chapter_digest: str = "",
    ) -> list[str]:
        """单个批次：整批翻译（段数对齐由 Translator 内部重试/兜底保证）。

        每段都在自身上下文里翻译，不跨位置复用译文（避免丢失语境信息）。
        全书概览/本章梗概作为恒定前缀注入，让译者把握全局。
        润色、标点规范化、术语抽取、审校均移出批内关键路径（见 _translate_chapter
        批循环内联逻辑与章末排干），这里只保留必须阻塞的翻译调用本身。
        """
        sources = [s.source for s in batch]
        return self.translator.translate_batch(
            sources,
            glossary_terms=terms,
            style=style,
            context=ctx_text,
            book_synopsis=book_synopsis,
            chapter_digest=chapter_digest,
        )

    # ── 可选步骤 / 连续全流程 ────────────────────────────────────────────────
    ALL_STEPS = ("translate", "qa", "report", "assemble")

    def run_steps(
        self,
        input_path: str,
        steps,
        *,
        progress: Optional[ProgressFn] = None,
        out_format: str = "epub",
        out_path: str | None = None,
    ) -> dict[str, Any]:
        """按需执行步骤子集（可单选可全选）。steps ⊆ ALL_STEPS。"""

        steps = set(steps)
        run_steps_input = sorted(steps)

        if "translate" in steps:
            store = self.run(input_path, progress=progress)
        else:
            store = self.prepare(input_path, progress=progress)
            m = store.load_manifest()
            self._apply_language(m.get("source_lang") or self.config.source_lang)
        with store.lock():
            return self._finish_steps_locked(
                store,
                input_path=input_path,
                steps=steps,
                run_steps_input=run_steps_input,
                progress=progress,
                out_format=out_format,
                out_path=out_path,
            )

    def _finish_steps_locked(
        self,
        store: RunStore,
        *,
        input_path: str,
        steps: set[str],
        run_steps_input: list[str],
        progress: Optional[ProgressFn],
        out_format: str,
        out_path: str | None,
    ) -> dict[str, Any]:
        from ..agents.consistency import ConsistencyChecker
        from ..assemble.report import build_report
        from ..assemble.writer import assemble, bilingual_out_path

        store.log_event("run_steps_started", steps=run_steps_input, input_path=input_path)

        glossary = GlossaryStore(store.glossary_path)
        qa_issues: list[dict] = []
        report: dict[str, Any] | None = None
        try:
            if "qa" in steps:
                if progress:
                    progress(0, 0, "检查全书一致性…")
                qa_issues = ConsistencyChecker(self.client, self.config).check(store, glossary)
                store.log_event(
                    "consistency_qa_finished",
                    issue_count=len(qa_issues),
                    issues=qa_issues,
                )

            self._flush_usage(store, scope="pipeline")
            if "report" in steps:
                if progress:
                    progress(0, 0, "生成报告…")
                report = build_report(store, glossary)
                report["consistency_issues"] = qa_issues
                store.save_report(report)
                store.log_event("report_saved", path=store.report_path)
        finally:
            glossary.close()
            self._flush_usage(store, scope="pipeline")

        outputs: list[str] = []
        if "assemble" in steps:
            if progress:
                progress(0, 0, "生成译文文件…")
            out_cfg = self.config.output
            do_mono, do_bilingual = out_cfg.mono, out_cfg.bilingual
            if not do_mono and not do_bilingual:
                do_mono = True  # 兜底：mono/bilingual 都关时至少产一个单语产物
            if do_mono:
                outputs.append(
                    assemble(
                        store,
                        input_path,
                        out_path=out_path,
                        out_format=out_format,
                        bilingual=False,
                    )
                )
            if do_bilingual:
                bi_out_path = bilingual_out_path(out_path) if out_path else None
                outputs.append(
                    assemble(
                        store,
                        input_path,
                        out_path=bi_out_path,
                        out_format=out_format,
                        bilingual=True,
                        order=out_cfg.bilingual_order,
                    )
                )
            store.log_event("assembled", outputs=outputs, out_format=out_format)

        store.log_event(
            "run_steps_finished",
            steps=run_steps_input,
            outputs=outputs,
            qa_issue_count=len(qa_issues),
        )
        return {
            "store": store,
            "output": outputs[0] if outputs else None,
            "outputs": outputs,
            "report": report,
            "qa_issues": qa_issues,
        }

    def run_all(
        self,
        input_path: str,
        *,
        progress: Optional[ProgressFn] = None,
        out_format: str = "epub",
        out_path: str | None = None,
        do_qa: bool | None = None,
    ) -> dict[str, Any]:
        """翻译 → 一致性 QA → 报告 → 回填 EPUB，返回结果汇总。"""
        steps = {"translate", "report", "assemble"}
        if do_qa if do_qa is not None else self.config.pipeline.consistency_qa:
            steps.add("qa")
        return self.run_steps(
            input_path, steps, progress=progress, out_format=out_format, out_path=out_path
        )
