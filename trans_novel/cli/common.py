"""Shared state and rendering helpers for the command-line applications."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import typer
import yaml
from rich.console import Console

from trans_novel.config import Config
from trans_novel.pipeline.state import RunStore
from trans_novel.pipeline.state import runstore_for as resolve_runstore

console = Console()
_CONFIG = {"path": "config.yaml"}


def configure_windows_console(
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
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


configure_windows_console()


def set_config_path(path: str) -> None:
    _CONFIG["path"] = path


def config_path() -> Path:
    return Path(_CONFIG["path"]).expanduser()


def load_config() -> Config:
    path = Path(_CONFIG["path"]).expanduser()
    try:
        return Config.load(str(path))
    except (OSError, UnicodeError, TypeError, ValueError, yaml.YAMLError) as error:
        console.print(f"[red]配置文件无效：{path}[/]")
        console.print(str(error))
        raise typer.Exit(2) from None


def require_input_file(input_path: str) -> None:
    if not os.path.isfile(input_path):
        console.print(f"[red]输入文件不存在：{input_path}[/]")
        raise typer.Exit(1)


def validate_output_format(fmt: str) -> str:
    normalized = fmt.strip().lower()
    if normalized not in {"epub", "txt"}:
        console.print(f"[red]不支持的输出格式：{fmt}（可选 epub / txt）[/]")
        raise typer.Exit(2)
    return normalized


def runstore_for(config: Config, input_path: str) -> RunStore:
    require_input_file(input_path)
    return resolve_runstore(config, input_path)


def print_back_matter(report: dict) -> None:
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
        "如果这里混进了需要完整翻译的正文章节，请用 "
        "`--back-matter full` 重新运行，程序会自动重译这些章节。"
    )


def print_usage(report: dict) -> None:
    """打印本书累计 token 用量与分 Agent 缓存命中率（无数据时静默跳过）。"""
    usage = report.get("usage") or {}
    totals = usage.get("totals") or {}
    if not totals.get("total_tokens"):
        return
    console.print(
        f"用量（本书累计）：{totals['total_tokens']:,} tok"
        f"（提示 {totals['prompt_tokens']:,} / 生成 {totals['completion_tokens']:,}），"
        f"缓存命中率 {totals.get('cache_hit_rate', 0.0):.1%}"
        f"（命中 {totals['cache_hit_tokens']:,} / 未命中 {totals['cache_miss_tokens']:,} tok）"
    )
    for agent, value in sorted(
        usage.get("by_agent", {}).items(), key=lambda item: -item[1]["total_tokens"]
    ):
        console.print(
            f"  · {agent}：{value['total_tokens']:,} tok，{value['calls']} 次调用，"
            f"缓存命中率 {value['cache_hit_rate']:.1%}"
        )
    stages = usage.get("by_stage") or {}
    for stage, value in sorted(stages.items(), key=lambda item: -item[1]["total_tokens"]):
        console.print(
            f"  · 阶段 {stage}：{value['total_tokens']:,} tok"
            f"（提示 {value['prompt_tokens']:,} / 生成 {value['completion_tokens']:,}），"
            f"{value['calls']} 次调用，缓存命中率 {value['cache_hit_rate']:.1%}"
        )


__all__ = [
    "config_path",
    "configure_windows_console",
    "console",
    "load_config",
    "print_back_matter",
    "print_usage",
    "require_input_file",
    "runstore_for",
    "set_config_path",
    "validate_output_format",
]
