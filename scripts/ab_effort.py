"""reasoning_effort 降档 A/B 实验：polish（strong 档）与 review（cheap 档）。

为什么不能直接改 config 跑两本书对比：
- effort 是按档配置的，strong 同时管翻译——降档后润色的输入（直译稿）本身就变了，实验被污染；
- 跨书之间体裁/难度不可比。
正确做法 = 段级同素材对照：
- polish：从 events.jsonl 的 batch_translated 事件取「原文 + 润色前直译」（翻译时已落盘），
  同一批素材用 high / 低档各跑一遍 Polisher，然后三道盲评：
  ① 确定性 lint（零成本，引入新问题直接判负）
  ② 忠实度关卡（fidelity_check，对照原文）
  ③ 成对盲评（judge_pair 正反两序，两序一致才算胜，否则 tie）
- review：Reviewer 是检测器，不能成对盲评，要测查全率/误报率。
  对已译完的干净章节程序化注入已知错误（删句/改数字/换人称）造 ground truth，
  两个 effort 各跑一遍看各自抓到多少（查全率）；再对未注错章节跑两遍数误报。

用法（先 export DEEPSEEK_API_KEY）：
  uv run python scripts/ab_effort.py polish state/Atomic_Habits --sample 150 --effort medium
  uv run python scripts/ab_effort.py review state/Atomic_Habits --chapters 6 --effort medium
  加 --dry-run 用 FakeClient 空跑验证流程（不发网络请求、不花钱）。

判定规则（预先定死，跑完按数字执行，不看着结果找理由）：
- polish 降档可接受 =（M 忠实度失败数 ≤ H）且（无新增 lint 问题）
  且（H 净胜率 = H胜 − M胜 ≤ 样本的 10%）；
- review 降档可接受 = M 对 severe 注错（missing/mistranslation）的查全率下降 ≤ 5pp
  且干净章节误报数不升。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trans_novel.agents.analyzer import Analyzer
from trans_novel.agents.naturalizer import Naturalizer
from trans_novel.agents.polisher import Polisher
from trans_novel.agents.reviewer import Reviewer
from trans_novel.config import Config
from trans_novel.glossary.store import GlossaryStore
from trans_novel.llm.base import FakeClient, build_client
from trans_novel.pipeline import lint

SEVERE = ("missing", "mistranslation")


# ── 素材装载 ─────────────────────────────────────────────────────────────
def load_events(run_dir: Path) -> list[dict]:
    with open(run_dir / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_batches(run_dir: Path) -> list[dict]:
    """润色前的翻译批次（正文章、非 fast 旁路）；续跑重复时取最后一次。"""
    dedup: dict[tuple[int, int], dict] = {}
    for e in load_events(run_dir):
        if e.get("event") != "batch_translated" or e.get("back_matter"):
            continue
        segs = [s for s in e.get("segments", []) if s.get("target") and s.get("source")]
        if segs:
            dedup[(e["chapter"], e["start_index"])] = {"chapter": e["chapter"], "segments": segs}
    return [dedup[k] for k in sorted(dedup)]


def load_run_context(run_dir: Path, cfg: Config):
    """manifest 语言 + 风格简报 + 术语表（两臂共用，保证唯一变量是 effort）。"""
    manifest = json.load(open(run_dir / "manifest.json", encoding="utf-8"))
    cfg.source_lang = manifest.get("source_lang") or cfg.source_lang
    analysis = {}
    ana_path = run_dir / "analysis.json"
    if ana_path.exists():
        analysis = json.load(open(ana_path, encoding="utf-8"))
    style = Analyzer(FakeClient(), cfg).style_brief(analysis)
    glossary = GlossaryStore(str(run_dir / "glossary.db"))
    terms = glossary.all_terms()
    glossary.close()
    return style, terms


def make_client(cfg_path: str, tier: str, effort: str | None, dry_run: bool):
    """按指定档位 effort 构造独立 client（用量互不串账）。

    effort=None 保持 config 原值；"off" 关掉该档 thinking（真正砍 reasoning token）；
    其余值写入 reasoning_effort。
    """
    cfg = Config.load(cfg_path)
    if dry_run:
        cfg.llm.provider = "fake"
    if effort == "off":
        cfg.llm.tiers[tier].thinking = False
    elif effort is not None:
        cfg.llm.tiers[tier].reasoning_effort = effort
    return cfg, build_client(cfg)


# ── polish A/B ───────────────────────────────────────────────────────────
def run_polish(args) -> None:
    run_dir = Path(args.run_dir)
    cfg_h, client_h = make_client(args.config, "strong", None, args.dry_run)
    cfg_m, client_m = make_client(args.config, "strong", args.effort, args.dry_run)
    cfg_j, client_j = make_client(args.config, "strong", None, args.dry_run)  # 评审固定用原档
    style, terms = load_run_context(run_dir, cfg_h)
    for c in (cfg_m, cfg_j):
        c.source_lang = cfg_h.source_lang

    batches = raw_batches(run_dir)
    rng = random.Random(args.seed)
    rng.shuffle(batches)
    picked, n_segs = [], 0
    for b in batches:
        picked.append(b)
        n_segs += len(b["segments"])
        if n_segs >= args.sample:
            break
    print(f"素材：{len(picked)} 个批次 / {n_segs} 段（seed={args.seed}）")

    pol_h, pol_m = Polisher(client_h, cfg_h), Polisher(client_m, cfg_m)
    judge = Naturalizer(client_j, cfg_j)
    locked = [t for t in terms if t.locked]

    def polish_batch(b):
        srcs = [s["source"] for s in b["segments"]]
        raws = [s["target"] for s in b["segments"]]
        batch_terms = GlossaryStore.terms_in(terms, "\n".join(srcs))
        out_h = pol_h.polish(raws, srcs, glossary_terms=batch_terms, style=style)
        out_m = pol_m.polish(raws, srcs, glossary_terms=batch_terms, style=style)
        return srcs, raws, out_h, out_m

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(polish_batch, picked))

    stats = {"identical": 0, "h_win": 0, "m_win": 0, "tie": 0}
    fid_fail = {"H": 0, "M": 0}
    lint_fail = {"H": 0, "M": 0}

    def judge_seg(item):
        src, raw, h, m = item
        if h.strip() == m.strip():
            return ("identical", None)
        raw_clean = not lint.lint_targets(
            [src], [raw], locked_terms=locked, src_lang=cfg_h.source_lang
        )
        bad = [
            arm
            for arm, out in (("H", h), ("M", m))
            if raw_clean
            and lint.lint_targets([src], [out], locked_terms=locked, src_lang=cfg_h.source_lang)
        ]
        if bad:
            return ("lint", bad)  # 这些臂引入了原译没有的确定性问题
        fid = []
        for arm, out in (("H", h), ("M", m)):
            if not judge.fidelity_check(src, raw, out):
                fid.append(arm)
        if fid:
            return ("fidelity", fid)
        r1 = judge.judge_pair(h, m)  # A=H B=M
        r2 = judge.judge_pair(m, h)  # A=M B=H
        if r1 == "A" and r2 == "B":
            return ("h_win", None)
        if r1 == "B" and r2 == "A":
            return ("m_win", None)
        return ("tie", None)

    items = [
        (src, raw, h, m)
        for srcs, raws, hs, ms in results
        for src, raw, h, m in zip(srcs, raws, hs, ms)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verdicts = list(pool.map(judge_seg, items))
    for kind, detail in verdicts:
        if kind == "lint":
            for arm in detail:
                lint_fail[arm] += 1
        elif kind == "fidelity":
            for arm in detail:
                fid_fail[arm] += 1
        else:
            stats[kind] += 1

    judged = stats["h_win"] + stats["m_win"] + stats["tie"]
    print("\n── polish A/B 结果（H=high，M=%s）──" % (args.effort))
    print(f"总段数 {len(items)}；两臂输出逐字相同 {stats['identical']} 段")
    print(f"lint 引入新问题：H={lint_fail['H']}  M={lint_fail['M']}（有即判该臂负）")
    print(f"忠实度失败：H={fid_fail['H']}  M={fid_fail['M']}")
    print(
        f"成对盲评（两序一致才计胜）：H胜 {stats['h_win']} / M胜 {stats['m_win']} / 平 {stats['tie']}"
    )
    if judged:
        margin = (stats["h_win"] - stats["m_win"]) / len(items)
        print(f"H 净胜率（占全部段）：{margin:+.1%}（判定阈值 ≤ +10%）")
    for arm, client in (("H", client_h), ("M", client_m), ("judge", client_j)):
        t = client.usage_summary()["totals"]
        print(
            f"用量[{arm}]：completion {t['completion_tokens']:,} tok，"
            f"prompt {t['prompt_tokens']:,} tok，{t['calls']} 次"
        )
    ok = (
        fid_fail["M"] <= fid_fail["H"]
        and lint_fail["M"] == 0
        and judged
        and (stats["h_win"] - stats["m_win"]) / len(items) <= 0.10
    )
    print("结论建议：", "可降档（按预定规则通过）" if ok else "保持 high（未通过预定规则）")


# ── review A/B（注错法）──────────────────────────────────────────────────
def _mutate(pairs: list[tuple[str, str]], k: int, rng: random.Random, locked) -> list[dict]:
    """向 (source, target) 对注入 k 处已知错误，返回 [{index,type,before,after}]。"""
    muts: list[dict] = []
    idxs = [i for i, (_, t) in enumerate(pairs) if len(t) >= 20]
    rng.shuffle(idxs)
    for i in idxs:
        if len(muts) >= k:
            break
        src, tgt = pairs[i]
        kind = ["missing", "mistranslation", "pronoun"][len(muts) % 3]
        new = None
        if kind == "missing":
            sents = re.split(r"(?<=[。！？])", tgt)
            sents = [s for s in sents if s.strip()]
            if len(sents) >= 2:
                drop = rng.randrange(len(sents))
                new = "".join(s for j, s in enumerate(sents) if j != drop)
        elif kind == "mistranslation":
            m = re.search(r"\d+", tgt)
            if m:
                new = tgt[: m.start()] + str(int(m.group()) + 3) + tgt[m.end() :]
        elif kind == "pronoun":
            if "他" in tgt:
                new = tgt.replace("他", "她", 1)
            elif "她" in tgt:
                new = tgt.replace("她", "他", 1)
        if new and new != tgt:
            pairs[i] = (src, new)
            muts.append({"index": i, "type": kind})
    return muts


def _pack(pairs: list[tuple[str, str]], budget: int) -> list[tuple[int, list]]:
    """按源文预算保序打包，返回 (块首段偏移, 块) 列表（复刻生产逻辑）。"""
    chunks, cur, size, base = [], [], 0, 0
    for p in pairs:
        if cur and size + len(p[0]) > budget:
            chunks.append((base, cur))
            base += len(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p[0])
    if cur:
        chunks.append((base, cur))
    return chunks


def run_review(args) -> None:
    run_dir = Path(args.run_dir)
    cfg_h, client_h = make_client(args.config, "cheap", None, args.dry_run)
    cfg_m, client_m = make_client(args.config, "cheap", args.effort, args.dry_run)
    _, terms = load_run_context(run_dir, cfg_h)
    cfg_m.source_lang = cfg_h.source_lang
    locked = [t for t in terms if t.locked]
    budget = cfg_h.segment.max_chars_per_batch * 3

    chapters = []
    for p in sorted((run_dir / "chapters").glob("*.json")):
        c = json.load(open(p, encoding="utf-8"))
        pairs = [
            (s["source"], s["target"])
            for s in c.get("segments", [])
            if s.get("kind", "text") == "text" and s.get("target") and s.get("source")
        ]
        if c.get("meta", {}).get("back_matter_mode"):
            continue
        if len(pairs) >= 10:
            chapters.append((c["index"], pairs))
    rng = random.Random(args.seed)
    rng.shuffle(chapters)
    mutated = chapters[: args.chapters]
    clean = chapters[args.chapters : args.chapters * 2]
    print(f"注错章：{[i for i, _ in mutated]}；干净对照章：{[i for i, _ in clean]}")

    rev_h, rev_m = Reviewer(client_h, cfg_h), Reviewer(client_m, cfg_m)
    detected = {"H": 0, "M": 0}
    detected_severe = {"H": 0, "M": 0}
    total_muts, total_severe = 0, 0
    false_pos = {"H": 0, "M": 0}

    def review_chunk(job):
        ci, arm, rev, base, chunk = job
        srcs = [s for s, _ in chunk]
        tgts = [t for _, t in chunk]
        chunk_terms = GlossaryStore.terms_in(terms, "\n".join(srcs))
        issues = rev.review(srcs, tgts, chunk_terms)
        kept = [
            it
            for it in issues
            if isinstance(it.get("index"), int) and 0 <= it["index"] < len(chunk)
        ]
        return ci, arm, base, kept

    jobs, mut_index = [], {}
    for ci, pairs in mutated:
        work = list(pairs)
        muts = _mutate(work, args.errors_per_chapter, rng, locked)
        total_muts += len(muts)
        total_severe += sum(1 for m in muts if m["type"] in SEVERE)
        for base, chunk in _pack(work, budget):
            for arm, rev in (("H", rev_h), ("M", rev_m)):
                jobs.append((ci, arm, rev, base, chunk))
        mut_index[ci] = {m["index"]: m["type"] for m in muts}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(review_chunk, jobs))
    for ci, arm, base, issues in results:
        muts = mut_index[ci]
        for it in issues:
            gi = base + it["index"]
            if gi in muts:
                detected[arm] += 1
                if muts[gi] in SEVERE:
                    detected_severe[arm] += 1
            else:
                false_pos[arm] += 1

    clean_fp = {"H": 0, "M": 0}
    jobs2 = [
        (ci, arm, rev, base, chunk)
        for ci, pairs in clean
        for base, chunk in _pack(pairs, budget)
        for arm, rev in (("H", rev_h), ("M", rev_m))
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _, arm, _, issues in pool.map(review_chunk, jobs2):
            clean_fp[arm] += len(issues)

    print("\n── review A/B 结果（H=high，M=%s）──" % (args.effort))
    print(f"注入错误 {total_muts} 处（severe {total_severe} 处）")
    for arm in ("H", "M"):
        r = detected[arm] / total_muts if total_muts else 0
        rs = detected_severe[arm] / total_severe if total_severe else 0
        print(
            f"  {arm}：查全 {detected[arm]}/{total_muts}（{r:.0%}），"
            f"severe {detected_severe[arm]}/{total_severe}（{rs:.0%}），"
            f"注错章误报 {false_pos[arm]}，干净章误报 {clean_fp[arm]}"
        )
    for arm, client in (("H", client_h), ("M", client_m)):
        t = client.usage_summary()["totals"]
        print(f"用量[{arm}]：completion {t['completion_tokens']:,} tok，{t['calls']} 次")
    if total_severe:
        drop = (detected_severe["H"] - detected_severe["M"]) / total_severe
        ok = drop <= 0.05 and clean_fp["M"] <= clean_fp["H"]
        print(f"severe 查全率下降：{drop:+.0%}（判定阈值 ≤ +5pp）")
        print("结论建议：", "可降档（按预定规则通过）" if ok else "保持 high（未通过预定规则）")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["polish", "review"])
    ap.add_argument("run_dir", help="state 运行目录，如 state/Atomic_Habits")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--effort", default="medium", help="低档臂的 reasoning_effort（medium/low）")
    ap.add_argument("--sample", type=int, default=150, help="polish：抽样段数下限")
    ap.add_argument("--chapters", type=int, default=6, help="review：注错章数（另取等量干净章）")
    ap.add_argument("--errors-per-chapter", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="FakeClient 空跑验证流程，不发请求")
    args = ap.parse_args()
    if args.mode == "polish":
        run_polish(args)
    else:
        run_review(args)


if __name__ == "__main__":
    main()
