"""Typer command registration for offline benchmark workflows."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
benchmark_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="离线基准测试工具。",
)
corpus_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="扫描、冻结和校验基准语料库。",
)
benchmark_run_app = typer.Typer(add_completion=False, no_args_is_help=True)
integration_app = typer.Typer(add_completion=False, no_args_is_help=True)
evaluate_app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="准备并汇总自动译文评审。"
)
report_app = typer.Typer(add_completion=False, no_args_is_help=True, help="构建自动评审基准报告。")


def _corpus_error(error: Exception) -> typer.NoReturn:
    console.print(f"[red]{error}[/]")
    raise typer.Exit(2) from error


@corpus_app.command("scan")
def benchmark_corpus_scan(
    book_spec: str = typer.Argument(..., metavar="BOOK_SPEC.yaml"),
    out: str = typer.Option(..., "--out", metavar="INVENTORY_DIR"),
) -> None:
    """Parse each source book once and write deterministic selection inventory."""
    from trans_novel.benchmark.corpus import CorpusError, scan

    try:
        scan(book_spec, out)
    except CorpusError as error:
        _corpus_error(error)
    console.print(f"[bold green]Inventory written:[/] {Path(out).expanduser().resolve()}")


@corpus_app.command("build")
def benchmark_corpus_build(
    book_spec: str = typer.Argument(..., metavar="BOOK_SPEC.yaml"),
    selection: str = typer.Argument(..., metavar="SELECTION.yaml"),
    out: str = typer.Option(..., "--out", metavar="CORPUS_DIR"),
) -> None:
    """Reparse sources and freeze runner/evaluator corpus artifacts."""
    from trans_novel.benchmark.corpus import CorpusError, build

    try:
        build(book_spec, selection, out)
    except CorpusError as error:
        _corpus_error(error)
    console.print(f"[bold green]Corpus written:[/] {Path(out).expanduser().resolve()}")


@corpus_app.command("validate")
def benchmark_corpus_validate(
    corpus_dir: str = typer.Argument(..., metavar="CORPUS_DIR"),
) -> None:
    """Validate a frozen corpus without opening original source books."""
    from trans_novel.benchmark.corpus import CorpusError, validate_corpus

    try:
        result = validate_corpus(corpus_dir)
    except CorpusError as error:
        _corpus_error(error)
    table = Table(title="Benchmark corpus")
    table.add_column("Scope")
    table.add_column("Name")
    table.add_column("Count")
    for split, count in result["split_counts"].items():
        table.add_row("split", split, str(count))
    for bucket, count in result["bucket_counts"].items():
        table.add_row("bucket", bucket, str(count))
    for book_id, count in result["book_counts"].items():
        table.add_row("book", book_id, str(count))
    table.add_row("total", "passages", str(result["runner_count"]))
    table.add_row("total", "context keys", str(result["challenge_count"]))
    console.print(table)
    console.print(result["corpus_sha256"])


@benchmark_run_app.command("canary")
def benchmark_run_canary(
    book_spec: str = typer.Argument(..., metavar="BOOK_SPEC.yaml"),
    candidates: str = typer.Argument(..., metavar="CANDIDATES.yaml"),
    out: str = typer.Option(..., "--out", metavar="RUN_DIR"),
    book_id: str | None = typer.Option(None, "--book-id"),
) -> None:
    from trans_novel.benchmark.run import BenchmarkError, CanaryRunner

    try:
        CanaryRunner().run(book_spec, candidates, out, book_id=book_id)
    except BenchmarkError as error:
        _corpus_error(error)
    console.print(f"[bold green]Canary passed:[/] {Path(out).expanduser().resolve()}")


@benchmark_run_app.command("full")
def benchmark_run_full(
    book_spec: str = typer.Argument(..., metavar="BOOK_SPEC.yaml"),
    candidates: str = typer.Argument(..., metavar="CANDIDATES.yaml"),
    out: str = typer.Option(..., "--out", metavar="RUN_DIR"),
) -> None:
    from trans_novel.benchmark.run import BenchmarkError, FullRunner

    try:
        result = FullRunner().run(book_spec, candidates, out)
    except BenchmarkError as error:
        _corpus_error(error)
    console.print(f"[bold green]Full run complete:[/] {result['branch_count']} branches")


@integration_app.command("run")
def benchmark_integration_run(
    corpus_dir: str = typer.Argument(..., metavar="CORPUS_DIR"),
    book_spec: str = typer.Argument(..., metavar="BOOK_SPEC.yaml"),
    candidates: str = typer.Argument(..., metavar="CANDIDATES.yaml"),
    integration_spec: str = typer.Argument(..., metavar="INTEGRATION_SPEC.yaml"),
    out: str = typer.Option(..., "--out", metavar="INTEGRATION_DIR"),
) -> None:
    from trans_novel.benchmark.integration import IntegrationError, IntegrationRunner

    try:
        result = IntegrationRunner().run(corpus_dir, book_spec, candidates, integration_spec, out)
    except (IntegrationError, OSError, ValueError) as error:
        _corpus_error(error)
    failed = list(result.get("failed_candidates", []))
    status = "no-op" if result.get("no_op") else "resumed" if result.get("resumed") else "fresh"
    console.print(f"[bold yellow]Integration {status}:[/] {Path(out).expanduser().resolve()}")
    if failed:
        console.print(f"[bold red]Failed candidates:[/] {', '.join(sorted(failed))}")
        for cid in sorted(failed):
            result_path = Path(out).expanduser().resolve() / "candidates" / cid / "result.json"
            with contextlib.suppress(OSError, ValueError):
                evidence = json.loads(result_path.read_text(encoding="utf-8"))
                reasons = evidence.get("failure_reasons", ["failed predicate"])
                console.print(f"  {cid}: {', '.join(str(reason) for reason in reasons)}")
        raise typer.Exit(1)


@evaluate_app.command("prepare")
def benchmark_review_prepare(
    run_dir: str = typer.Argument(..., metavar="RUN_DIR"),
    review_spec: str = typer.Argument(..., metavar="REVIEW_SPEC.yaml"),
    out: str = typer.Option(..., "--out", metavar="REVIEW_DIR"),
) -> None:
    from trans_novel.benchmark.review import ReviewArtifactError, prepare_review

    try:
        prepare_review(run_dir, review_spec, out)
    except ReviewArtifactError as error:
        _corpus_error(error)
    console.print(f"[bold green]Review shards written:[/] {Path(out).expanduser().resolve()}")


@evaluate_app.command("finalize")
def benchmark_review_finalize(
    review_dir: str = typer.Argument(..., metavar="REVIEW_DIR"),
    results_dir: str = typer.Argument(..., metavar="RESULTS_DIR"),
) -> None:
    from trans_novel.benchmark.review import ReviewArtifactError, finalize_review

    try:
        finalize_review(review_dir, results_dir)
    except ReviewArtifactError as error:
        _corpus_error(error)
    console.print(f"[bold green]Review finalized:[/] {Path(review_dir).expanduser().resolve()}")


@evaluate_app.command("validate")
def benchmark_review_validate(
    review_dir: str = typer.Argument(..., metavar="REVIEW_DIR"),
) -> None:
    from trans_novel.benchmark.review import ReviewArtifactError, validate_review

    try:
        result = validate_review(review_dir)
    except ReviewArtifactError as error:
        _corpus_error(error)
    console.print(
        f"[bold green]Review valid:[/] {result['review_sha256']} status={result['status']}"
    )


@report_app.command("build")
def benchmark_report_build(
    run_dir: str = typer.Argument(..., metavar="RUN_DIR"),
    review_dir: str = typer.Argument(..., metavar="REVIEW_DIR"),
    price_path: str = typer.Argument(..., metavar="PRICE.yaml"),
    out: str = typer.Option(..., "--out", metavar="REPORT_DIR"),
    integration: str | None = typer.Option(None, "--integration", metavar="INTEGRATION.json"),
) -> None:
    from trans_novel.benchmark.report import ReportError, build_report

    try:
        result = build_report(
            run_dir,
            review_dir,
            price_path,
            out,
            integration_path=integration,
        )
        report_hash = result["report_sha256"]
        status = result["status"]
    except (ReportError, OSError, ValueError) as error:
        _corpus_error(error)
    console.print(
        f"[bold green]Report written:[/] {result['out_dir']} "
        f"status={status} report_sha256={report_hash}"
    )


@report_app.command("validate")
def benchmark_report_validate(report_dir: str = typer.Argument(..., metavar="REPORT_DIR")) -> None:
    from trans_novel.benchmark.report import ReportError, validate_report

    try:
        value = validate_report(Path(report_dir))
    except (ReportError, OSError, ValueError) as error:
        _corpus_error(error)
    console.print(f"[bold green]Report valid:[/] {value['report_sha256']}")


benchmark_app.add_typer(report_app, name="report")
benchmark_app.add_typer(benchmark_run_app, name="run")
benchmark_app.add_typer(corpus_app, name="corpus")
benchmark_app.add_typer(evaluate_app, name="evaluate")
benchmark_app.add_typer(integration_app, name="integration")


__all__ = [
    "benchmark_app",
    "benchmark_corpus_build",
    "benchmark_corpus_scan",
    "benchmark_corpus_validate",
    "benchmark_integration_run",
    "benchmark_report_build",
    "benchmark_report_validate",
    "benchmark_review_finalize",
    "benchmark_review_prepare",
    "benchmark_review_validate",
    "benchmark_run_canary",
    "benchmark_run_full",
]
