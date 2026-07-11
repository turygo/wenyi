"""命令行入口（typer + rich）。

日常只需 `translate` 一个命令：连续全流程（分析→翻译→审校→一致性 QA→报告→回填 EPUB），
中断后再次运行自动续跑。其余 `resume` / `status` 为常用辅助；
细粒度/调试工具收敛到 `tools`：glossary / assemble / qa / report。
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import Config
from .ingest.segmenter import load_document
from .pipeline.runstore import STATUS_DONE, RunStore, slugify


def _configure_windows_console(
    streams: tuple[object, ...] | None = None,
    *,
    is_windows: bool | None = None,
) -> None:
    """让 Windows 控制台能输出中文；PyInstaller 单文件启动时尤其需要。"""
    if is_windows is None:
        is_windows = os.name == "nt"
    if not is_windows:
        return
    for stream in streams or (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_configure_windows_console()

app = typer.Typer(add_completion=False, help="多 Agent 小说翻译系统（多语言 → 中文）")
tools_app = typer.Typer(
    add_completion=False,
    help="高级/调试工具：glossary（术语表）/ assemble（回填）/ qa / report",
)
console = Console()

_CONFIG = {"path": "config.yaml"}


@app.callback()
def _root(
    config: str = typer.Option("config.yaml", "--config", "-c", help="配置文件路径"),
):
    _CONFIG["path"] = config


def _load_config() -> Config:
    return Config.load(_CONFIG["path"])


def _require_input_file(input_path: str) -> None:
    if not os.path.isfile(input_path):
        console.print(f"[red]输入文件不存在：{input_path}[/]")
        raise typer.Exit(1)


def _runstore_for(config: Config, input_path: str) -> RunStore:
    _require_input_file(input_path)
    doc = load_document(input_path, config.source_lang, config.target_lang)
    run_dir = os.path.join(config.state_dir, slugify(doc.title))
    return RunStore(run_dir, create=False)


def _translate_impl(
    input_path: str,
    *,
    chapter: Optional[int] = None,
    fmt: str = "epub",
    out: Optional[str] = None,
    polish: Optional[bool] = None,
    qa: Optional[bool] = None,
    mono: Optional[bool] = None,
    bilingual: Optional[bool] = None,
) -> None:
    """translate/resume 共享实现，避免 CLI 参数转发漂移。"""
    from .pipeline.orchestrator import Orchestrator

    _require_input_file(input_path)
    config = _load_config()
    if polish is not None:
        config.pipeline.polish = polish
    if mono is not None:
        config.output.mono = mono
    if bilingual is not None:
        config.output.bilingual = bilingual
    orch = Orchestrator(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("准备中…", total=None)

        def cb(done: int, total: int, label: str) -> None:
            prog.update(task, completed=done, total=total or None, description=label)

        if chapter is not None:
            store = orch.run(input_path, only_chapter=chapter, progress=cb)
            console.print(f"[green]已翻第 {chapter} 章[/]，状态目录：{store.run_dir}")
            _print_usage({"usage": orch.client.usage_summary()})
            return

        result = orch.run_all(
            input_path,
            progress=cb,
            out_format=fmt,
            out_path=out,
            do_qa=qa,
        )

    s = result["report"]["summary"]
    console.print(
        f"[bold green]完成[/]：{s['chapters_done']}/{s['chapters_total']} 章，"
        f"术语 {s['terms']}，一致性问题 {len(result['qa_issues'])} 项。"
    )
    _print_usage(result["report"])
    _print_back_matter(result["report"])
    for path in result.get("outputs") or [result["output"]]:
        console.print(f"译文：[bold]{path}[/]")


def _print_back_matter(report: dict) -> None:
    """列出被简化处理的附属章供人工复核——误伤正文章会静默降质，必须可见。"""
    bm = report.get("back_matter_chapters") or []
    if not bm:
        return
    mode_desc = {"skip": "保留原文，未翻译", "light": "快速粗翻，未精校润色"}
    console.print(
        "[yellow]以下章节被识别为附属内容（致谢、作者简介、注释、索引、版权页等），"
        "为节省成本只做了简化处理：[/]"
    )
    for b in bm:
        console.print(f"  第{b['chapter']}章 {b['title']} —— {mode_desc.get(b['mode'], b['mode'])}")
    console.print(
        "如果这里混进了需要完整翻译的正文章节，请打开 config.yaml，"
        "把 pipeline.back_matter 一行改成 full，再重新运行一次，程序会自动重译这些章节。"
    )


def _print_usage(report: dict) -> None:
    """打印本次运行的 token 用量与分档缓存命中率（无用量数据时静默跳过）。"""
    usage = report.get("usage") or {}
    totals = usage.get("totals") or {}
    if not totals.get("total_tokens"):
        return
    console.print(
        f"用量：{totals['total_tokens']:,} tok"
        f"（提示 {totals['prompt_tokens']:,} / 生成 {totals['completion_tokens']:,}），"
        f"缓存命中率 {totals.get('cache_hit_rate', 0.0):.1%}"
        f"（命中 {totals['cache_hit_tokens']:,} / 未命中 {totals['cache_miss_tokens']:,} tok）"
    )
    for tier, v in sorted(usage.get("by_tier", {}).items()):
        console.print(
            f"  · {tier}：{v['total_tokens']:,} tok，{v['calls']} 次调用，"
            f"缓存命中率 {v['cache_hit_rate']:.1%}"
        )
    stages = usage.get("by_stage") or {}
    for stage, v in sorted(stages.items(), key=lambda kv: -kv[1]["total_tokens"]):
        console.print(
            f"  · 阶段 {stage}：{v['total_tokens']:,} tok"
            f"（提示 {v['prompt_tokens']:,} / 生成 {v['completion_tokens']:,}），"
            f"{v['calls']} 次调用，缓存命中率 {v['cache_hit_rate']:.1%}"
        )


# ── translate / resume：连续全流程 ──────────────────────────────────────────
@app.command()
def translate(
    input: str = typer.Argument(..., help="输入文件（.epub / .txt / .md）"),
    chapter: Optional[int] = typer.Option(None, "--chapter", help="只翻指定章（调试用，不做收尾）"),
    fmt: str = typer.Option("epub", "--format", help="输出格式：epub | txt"),
    out: Optional[str] = typer.Option(
        None, "--out", help="输出路径（默认 <源文件名>.zh.<ext>，落在源文件目录）"
    ),
    polish: Optional[bool] = typer.Option(
        None,
        "--polish/--no-polish",
        help="覆盖配置文件中的润色开关",
    ),
    qa: Optional[bool] = typer.Option(
        None,
        "--qa/--no-qa",
        help="覆盖配置文件中的一致性 QA 开关",
    ),
    mono: Optional[bool] = typer.Option(
        None,
        "--mono/--no-mono",
        help="覆盖配置文件中的单语版产出开关",
    ),
    bilingual: Optional[bool] = typer.Option(
        None,
        "--bilingual/--no-bilingual",
        help="覆盖配置文件中的双语版产出开关",
    ),
):
    """翻译（连续全流程；可断点续跑）。"""
    _translate_impl(
        input,
        chapter=chapter,
        fmt=fmt,
        out=out,
        polish=polish,
        qa=qa,
        mono=mono,
        bilingual=bilingual,
    )


@app.command()
def resume(
    input: str = typer.Argument(..., help="输入文件"),
    fmt: str = typer.Option("epub", "--format", help="输出格式：epub | txt"),
):
    """断点续跑（等价于再次 translate）。"""
    _translate_impl(input, fmt=fmt)


# ── 查询 / 细粒度命令 ──────────────────────────────────────────────────────
@app.command()
def status(input: str = typer.Argument(..., help="输入文件")):
    """查看各章进度与术语库统计。"""
    from .glossary.store import GlossaryStore

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    m = store.load_manifest()
    console.print(f"《{m['title']}》（{m['fmt']}）  {m['source_lang']}→{m['target_lang']}")
    table = Table("", "#", "章节", "状态")
    for c in m["chapters"]:
        mark = "✓" if c["status"] == STATUS_DONE else "·"
        table.add_row(mark, str(c["index"]), c["title"], c["status"])
    console.print(table)
    g = GlossaryStore(store.glossary_path)
    console.print("术语库：", g.stats())
    g.close()


@tools_app.command()
def glossary(
    input: str = typer.Argument(..., help="输入文件"),
    action: str = typer.Argument("list", help="list | conflicts | audit | lock | resolve"),
    arg1: Optional[str] = typer.Argument(None),
    arg2: Optional[str] = typer.Argument(None),
):
    """术语库管理。audit 自动统一译法并改写正文。"""
    from .glossary import resolver
    from .glossary.store import GlossaryStore

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    g = GlossaryStore(store.glossary_path)
    try:
        if action == "list":
            table = Table("原文", "译文", "类型", "置信/状态", "锁")
            for t in g.all_terms():
                table.add_row(
                    t.source,
                    t.target,
                    f"{t.type}{'/' + t.gender if t.gender else ''}",
                    f"{t.confidence}{'/' + t.status if t.status != 'ok' else ''}",
                    "🔒" if t.locked else "",
                )
            console.print(table)
        elif action == "conflicts":
            for c in g.open_conflicts():
                console.print(
                    f"  {c['source']}: 现有「{c['existing_target']}」 vs "
                    f"提议「{c['proposed_target']}」（第{c['chapter']}章）"
                )
        elif action == "audit":
            from .agents.glossary_auditor import GlossaryAuditor
            from .llm.base import build_client

            applied = GlossaryAuditor(build_client(config), config).audit(store, g)
            console.print(f"已统一 {len(applied)} 组术语：")
            for u in applied:
                console.print(
                    f"  {u['source']} → [bold]{u['canonical']}[/]"
                    f"（替换 {', '.join(u['variants']) or '—'}）"
                )
        elif action == "lock":
            if arg1 is None:
                console.print("[red]lock 需要提供原文术语。[/]")
                raise typer.Exit(1)
            resolver.lock(g, arg1)
            term = g.get_term(arg1)
            if term is None:
                console.print(f"[red]术语不存在：{arg1}[/]")
                raise typer.Exit(1)
            console.print(f"已锁定 {arg1} → {term.target}")
        elif action == "resolve":
            if arg1 is None or arg2 is None:
                console.print("[red]resolve 需要提供原文术语和目标译名。[/]")
                raise typer.Exit(1)
            resolver.resolve(g, arg1, arg2)
            console.print(f"已裁定并锁定 {arg1} → {arg2}")
        else:
            console.print(f"[red]未知 glossary 子命令：{action}[/]")
            raise typer.Exit(1)
    finally:
        g.close()


@tools_app.command()
def assemble(
    input: str = typer.Argument(..., help="输入文件"),
    out: Optional[str] = typer.Option(None, "--out"),
    fmt: str = typer.Option("epub", "--format", help="epub | txt"),
    mono: Optional[bool] = typer.Option(
        None,
        "--mono/--no-mono",
        help="覆盖配置文件中的单语版产出开关",
    ),
    bilingual: Optional[bool] = typer.Option(
        None,
        "--bilingual/--no-bilingual",
        help="覆盖配置文件中的双语版产出开关",
    ),
):
    """回填生成译文文件（默认 EPUB）。"""
    from .assemble.writer import assemble as do_assemble
    from .assemble.writer import bilingual_out_path

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    do_mono = config.output.mono if mono is None else mono
    do_bilingual = config.output.bilingual if bilingual is None else bilingual
    if not do_mono and not do_bilingual:
        do_mono = True  # 兜底：至少产一个单语产物
    paths: list[str] = []
    if do_mono:
        paths.append(do_assemble(store, input, out_path=out, out_format=fmt, bilingual=False))
    if do_bilingual:
        bi_out = bilingual_out_path(out) if out else None
        paths.append(
            do_assemble(
                store,
                input,
                out_path=bi_out,
                out_format=fmt,
                bilingual=True,
                order=config.output.bilingual_order,
            )
        )
    for path in paths:
        console.print(f"已生成译文：[bold]{path}[/]")


@tools_app.command()
def qa(input: str = typer.Argument(..., help="输入文件")):
    """全书跨章一致性扫描。"""
    from .agents.consistency import ConsistencyChecker
    from .glossary.store import GlossaryStore
    from .llm.base import build_client

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    g = GlossaryStore(store.glossary_path)
    issues = ConsistencyChecker(build_client(config), config).check(store, g)
    g.close()
    console.print(f"一致性问题 {len(issues)} 项：")
    for it in issues:
        console.print(f"  [{it.get('type')}] {it.get('detail')}  ({it.get('where', '')})")


@tools_app.command()
def report(input: str = typer.Argument(..., help="输入文件")):
    """生成 QA 报告（漏译/冲突/低置信度汇总）。"""
    from .assemble.report import build_report
    from .glossary.store import GlossaryStore

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    g = GlossaryStore(store.glossary_path)
    rep = build_report(store, g)
    g.close()
    store.save_report(rep)
    s = rep["summary"]
    console.print(f"QA 报告已写入 {store.report_path}")
    console.print(
        f"  章节 {s['chapters_done']}/{s['chapters_total']}  术语 {s['terms']}  "
        f"待裁决冲突 {s['open_conflicts']}  审校问题 {s['review_issues']}  "
        f"回译疑点 {s['backtranslation_issues']}  附属章旁路 {s.get('back_matter_chapters', 0)}"
    )
    _print_back_matter(rep)


@tools_app.command()
def naturalize(
    input: str = typer.Argument(..., help="输入文件"),
    chapters: Optional[str] = typer.Option(
        None, "--chapters", help="逗号分隔章 index，缺省=全部正文章"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="只跑审读+改写+三道关卡，打印结果但不落盘、不写事件"
    ),
    limit: Optional[int] = typer.Option(None, "--limit", help="最多采纳写回 N 段（缺省无限）"),
):
    """去翻译腔：单语审读 → 单语改写 → 三道关卡（lint/忠实度/成对）→ 写回。

    主流水线已内置同名环节（pipeline.naturalize）；本命令用于对存量已译书手动补跑。
    """
    from .agents.naturalizer import Naturalizer, run_naturalize
    from .glossary.store import GlossaryStore
    from .llm.base import build_client

    config = _load_config()
    store = _runstore_for(config, input)
    if not store.exists():
        console.print("[yellow]尚无进度。先运行 translate。[/]")
        raise typer.Exit(1)
    chapter_indices: Optional[list[int]] = None
    if chapters:
        try:
            chapter_indices = [int(x) for x in chapters.split(",") if x.strip()]
        except ValueError as e:
            raise typer.BadParameter(
                f"--chapters 含非法片段：{chapters!r}（须为逗号分隔整数）"
            ) from e
    g = GlossaryStore(store.glossary_path)
    agent = Naturalizer(build_client(config), config)
    stats = run_naturalize(
        agent, store, g, config, chapters=chapter_indices, dry_run=dry_run, limit=limit
    )
    g.close()
    console.print(
        f"审读 {stats['screened']} 段  嫌疑 {stats['suspects']}  改写 {stats['rewritten']}  "
        f"lint拒 {stats['lint_rejected']}  忠实拒 {stats['fidelity_rejected']}  "
        f"成对拒 {stats['pairwise_rejected']}  采纳 {stats['applied']}"
        + ("（dry-run，未落盘）" if dry_run else "")
    )
    if dry_run:
        for e in stats["applied_entries"]:
            console.print(f"[dim]第{e['chapter']}章 #{e['index']}[/]")
            console.print(f"  before: {e['before']}")
            console.print(f"  after:  {e['after']}")


app.add_typer(tools_app, name="tools")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
