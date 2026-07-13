"""源文侧术语候选挖掘（翻译前，零 LLM 优先）。

现架构在翻译后从 (源文,译文) 对里抽术语回灌术语库，把首译直译固化成全书铁律，
还会把普通名词/一次性修辞误判为术语。改为：先在源文侧挖出候选表面形式（不给
译名），交给 agents/namer.CastNamer 一次性定名，翻译期术语表只读（见 orchestrator
._build_understanding）。

英文源双通道：确定性大写正则统计（mine_candidates_en，零成本、可复现，只抓大写词/
缩写）∪ fast 档逐章 LLM 挖掘（mine_candidates_llm，补大写通道的结构性盲区——领域
术语与主题词多以小写反复出现，如 lithography/foundry/yield，大写正则天生抓不到；
此类词译法不统一是实伤：Chip War 里 yield 曾译裂成"良率/产率"）。en 每章因此多一次
fast 档调用，与逐章梗概同量级，成本可接受。其它语言没有可靠的大小写信号，只走
fast 档逐章 LLM 挖掘（mine_candidates_llm）。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

# 大写词序列中允许夹在中间的小写连接词（如 "University of Tokyo"）。
_CONNECTORS = {
    "of",
    "the",
    "and",
    "de",
    "von",
    "van",
    "der",
    "du",
    "la",
    "le",
    "for",
    "in",
    "at",
    "&",
}
# 句子边界：句末标点（含省略号）后跟任意闭引号再跟空白，或换行，才是句子边界——
# 对话体 `…?" Do you…` 里 Do 紧跟闭引号，若不认闭引号会被误判成句中大写，
# 让"句首孤词过滤"对对话体完全失效。
_SENT_SPLIT = re.compile(r'(?<=[.!?\u2026])[”"\'\u2019]*\s+|\n+')
# Unicode 字母类（排除数字/下划线），不限 ASCII——"Charlotte Brontë"/"García Márquez"
# 这类带附加符号的姓名不能被 [A-Za-z] 拦腰截断；大小写判据（_is_titlecase/_is_allcap）
# 用 str.isupper/islower 本身即 Unicode 语义，无需额外改动。
_TOKEN = re.compile(r"[^\W\d_](?:[^\W\d_]|['\-])*", re.UNICODE)
_ROMAN = re.compile(r"^[IVXLCDM]+$")

# 英文封闭词类停用表：代词/限定词/连词/介词/助动词/否定词/常见句副词/常见感叹——
# 这些词永远不可能是专名本身，只要在（哪怕正确识别的）句中位置大写出现就会污染候选
# （对话体尤甚：She/Do/But 等紧跟闭引号，句首过滤对其失效）。
# 拿不准的开放类词（look/wait/okay/hey 等动词感叹）刻意不收录——可能撞人名，
# 交给下游 CastNamer 定名层去丢弃，不在这里一刀切。
_EN_STOP = frozenset(
    {
        # 冠词/指示词
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        # 人称代词
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        # 物主代词/形容词
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        # 反身代词
        "myself",
        "yourself",
        "himself",
        "herself",
        "itself",
        "ourselves",
        "yourselves",
        "themselves",
        # 并列/从属连词
        "and",
        "but",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "because",
        "although",
        "though",
        "while",
        "if",
        "unless",
        "since",
        "until",
        "before",
        "after",
        "when",
        "whenever",
        "whereas",
        "wherever",
        # 疑问词（对话体高频："What(193)/How/Why/Who/Where" 类，Wedding People 全书
        # 复测暴露的漏项——疑问句首/引号后极易被误判成句中大写）
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "where",
        # 常用介词
        "in",
        "on",
        "at",
        "by",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "without",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "over",
        "under",
        "again",
        "further",
        "once",
        "of",
        "off",
        "out",
        # be/do/have/情态助动词
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "having",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        # 否定词
        "not",
        "no",
        "never",
        "none",
        "nothing",
        "nobody",
        "nowhere",
        "neither",
        # 缩写分词碎片：源文用弯引号（’）时 _TOKEN 不认其为续接字符，"Don't"/"Doesn't"
        # 会被切成 "Don"+"t" 两个独立 token，前半截首字母大写、高频出现，若不拦会当作
        # 人名候选反复入表（Wedding People 复测里 Don(26) 即此）。作为独立人名的 "Don"
        # 极罕见，且真出现在多词序列（如 "Don Draper"）时不受这里的单词停用规则影响
        # （多词序列首尾剥离只剥停用词本身，"Don Draper" 剥不掉中间/唯一的专名部分）。
        "don",
        "didn",
        "doesn",
        "isn",
        "aren",
        "wasn",
        "weren",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
        "ain",
        "hasn",
        "haven",
        "hadn",
        "mustn",
        "needn",
        "ll",
        "ve",
        "re",
        "em",
        # 常见句副词
        "there",
        "here",
        "now",
        "then",
        "just",
        "even",
        "also",
        "too",
        "very",
        "still",
        "only",
        "already",
        "soon",
        "always",
        "often",
        "sometimes",
        "usually",
        "perhaps",
        "maybe",
        "actually",
        "really",
        "quite",
        "rather",
        "almost",
        "indeed",
        # 常见感叹
        "oh",
        "ah",
        "well",
        "yes",
        "hey",
        "wow",
        "huh",
        "hmm",
        "um",
        "uh",
    }
)


def _is_titlecase(tok: str) -> bool:
    """首字母大写、其余小写，且长度≥2（排除 "I" 等单字母代词误判）。"""
    return len(tok) >= 2 and tok[0].isupper() and tok[1:].islower()


def _is_allcap(tok: str) -> bool:
    """全大写缩写（TSMC/DARPA），排除罗马数字（III/IV 等易误判）。"""
    return len(tok) >= 2 and tok.isupper() and not _ROMAN.match(tok)


def _context(sentence: str) -> str:
    return sentence.strip()[:80]


@dataclass
class Candidate:
    surface: str
    count: int = 0
    chapters: list[int] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)


def mine_candidates_en(chapters: list[tuple[int, str]]) -> list[Candidate]:
    """英文源确定性挖掘：纯正则/字符串统计，零 LLM。

    抓：连续 Capitalized 词序列（允许 of/the/and 等小词夹中间）、全大写缩写。
    滤：封闭词类停用表（代词/虚词永不可能是专名，见 _EN_STOP）、只在句首出现过的单个
    常见大写词（该词从未在句中位置以大写出现过则丢弃）、纯数字/罗马数字、出现次数 <2
    的单词候选（多词序列出现 1 次也保留）。按 count 降序排列。
    """
    # 第一遍：记录每个 Title-case token 是否在非句首位置也出现过，以及总出现次数——
    # 判据独立于它后续是否被并入多词序列，避免漏判。
    ever_non_initial: dict[str, bool] = {}
    token_total: dict[str, int] = {}
    per_chapter_sentences: list[tuple[int, list[str]]] = []
    for ci, text in chapters:
        sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        per_chapter_sentences.append((ci, sentences))
        for sent in sentences:
            toks = _TOKEN.findall(sent)
            for i, tok in enumerate(toks):
                if _is_titlecase(tok):
                    token_total[tok] = token_total.get(tok, 0) + 1
                    if i != 0:
                        ever_non_initial[tok] = True

    candidates: dict[str, Candidate] = {}

    def _add(surface: str, ci: int, ctx: str) -> None:
        c = candidates.setdefault(surface, Candidate(surface=surface))
        c.count += 1
        if ci not in c.chapters:
            c.chapters.append(ci)
        if len(c.contexts) < 2:
            snippet = _context(ctx)
            if snippet not in c.contexts:
                c.contexts.append(snippet)

    def _consider_unigram(word: str, ci: int, sent: str) -> None:
        if word.lower() in _EN_STOP:
            return
        if ever_non_initial.get(word) and token_total.get(word, 0) >= 2:
            _add(word, ci, sent)

    # 第二遍：贪心拼接大写词序列（Title-case 或 Title-case+连接词+Title-case），
    # 再从首尾剥离停用词——"But Phoebe"（But 因紧邻标题词被并入同一序列）剥成
    # "Phoebe"，"The Cornwall Inn" 剥成 "Cornwall Inn"。规则取简单版：只剥首尾、
    # 不管内部（内部停用词属于专名一部分的场景，如 "Bank of England"，交给 namer
    # 定名层判断，这里不深究）。剥离后长度归 1 的按单词候选规则重新判定。
    for ci, sentences in per_chapter_sentences:
        for sent in sentences:
            toks = _TOKEN.findall(sent)
            i, n = 0, len(toks)
            while i < n:
                tok = toks[i]
                if _is_titlecase(tok):
                    run = [tok]
                    j = i + 1
                    while j < n:
                        nxt = toks[j]
                        if _is_titlecase(nxt):
                            run.append(nxt)
                            j += 1
                        elif (
                            nxt.lower() in _CONNECTORS and j + 1 < n and _is_titlecase(toks[j + 1])
                        ):
                            run.append(nxt)
                            j += 1
                        else:
                            break
                    stripped = list(run)
                    while stripped and stripped[0].lower() in _EN_STOP:
                        stripped.pop(0)
                    while stripped and stripped[-1].lower() in _EN_STOP:
                        stripped.pop()
                    if len(stripped) > 1:
                        _add(" ".join(stripped), ci, sent)
                    elif len(stripped) == 1:
                        _consider_unigram(stripped[0], ci, sent)
                    i = j
                elif _is_allcap(tok):
                    # 全大写章首排版伪影（THE/AND 等停用词全大写化）不得入候选。
                    if tok.lower() not in _EN_STOP:
                        _add(tok, ci, sent)
                    i += 1
                else:
                    i += 1

    # 全大写缩写沿用同一条"出现次数 <2 丢弃"规则；多词序列（含空格）豁免。
    out = [c for c in candidates.values() if c.count >= 2 or " " in c.surface]
    out.sort(key=lambda c: c.count, reverse=True)
    return out


def mine_candidates_llm(
    chapters: list[tuple[int, str]],
    agent: Any,
    *,
    concurrency: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Candidate]:
    """非英文源退路：fast 档逐章挖掘（只看源文，不给译名），本函数负责跨章合并计数。

    各章调用相互独立 → 按 concurrency 并行（LLM 调用进线程池；合并计数在主线程，
    且按输入章序合并，输出与串行完全一致）。on_progress(done, total) 按完成数回调
    （主线程触发，供进度条使用）。
    """
    from ..agents import prompts

    todo = [(ci, text) for ci, text in chapters if text.strip()]

    def _mine_one(ci: int, text: str) -> list[str]:
        system = prompts.render("term_miner_system", src=agent.src, tgt=agent.tgt)
        user = prompts.render(
            "term_miner_user", src=agent.src, tgt=agent.tgt, chapter=ci, source=text[:8000]
        )
        # 不设 default：某章挖掘失败若被兜成空列表，会让 term_mining_done 静默永久
        # 落盘——异常整体冒泡，交由调用方（orchestrator）捕获并放弃本次落标记、下次续跑重试。
        raw = agent._ask_json(
            system, user, tier="fast", key="candidates", operation="prescan.term_mine"
        )
        return [s.strip() for s in raw or [] if isinstance(s, str) and s.strip()]

    results: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(_mine_one, ci, text): ci for ci, text in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            results[futs[fut]] = fut.result()  # 异常冒泡；池随 with 收尾
            if on_progress:
                on_progress(i, len(todo))

    # 按输入章序合并（与并发完成顺序无关，保证输出确定性）
    candidates: dict[str, Candidate] = {}
    for ci, _ in todo:
        for surface in results.get(ci, []):
            c = candidates.setdefault(surface, Candidate(surface=surface))
            c.count += 1
            if ci not in c.chapters:
                c.chapters.append(ci)

    out = list(candidates.values())
    out.sort(key=lambda c: c.count, reverse=True)
    return out


def mine_candidates(
    src_lang: str,
    chapters: list[tuple[int, str]],
    agent: Any,
    *,
    concurrency: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Candidate]:
    """入口：en 走"确定性大写通道 ∪ fast 档 LLM 通道"双通道合并；其它语言只走 LLM 通道。"""
    if not (src_lang or "").strip().lower().startswith("en"):
        return mine_candidates_llm(
            chapters, agent, concurrency=concurrency, on_progress=on_progress
        )

    det = mine_candidates_en(chapters)
    llm = mine_candidates_llm(chapters, agent, concurrency=concurrency, on_progress=on_progress)
    return _merge_candidates(det, llm)


def _merge_candidates(*channels: list[Candidate]) -> list[Candidate]:
    """按 surface 大小写不敏感合并多通道候选：同 surface 两通道都命中时 count 取大者，
    chapters/contexts 取并集（contexts 仍受 ≤2 条上限）；先出现的通道决定保留的原样
    surface（en 双通道调用时传参顺序是"大写通道在前"，故大写通道产物保留原样 surface）。
    """
    merged: dict[str, Candidate] = {}
    for channel in channels:
        for c in channel:
            key = c.surface.lower()
            existing = merged.get(key)
            if existing is None:
                merged[key] = Candidate(
                    surface=c.surface,
                    count=c.count,
                    chapters=list(c.chapters),
                    contexts=list(c.contexts),
                )
                continue
            existing.count = max(existing.count, c.count)
            existing.chapters = sorted(set(existing.chapters) | set(c.chapters))
            for ctx in c.contexts:
                if len(existing.contexts) >= 2:
                    break
                if ctx not in existing.contexts:
                    existing.contexts.append(ctx)
    out = list(merged.values())
    out.sort(key=lambda c: c.count, reverse=True)
    return out
