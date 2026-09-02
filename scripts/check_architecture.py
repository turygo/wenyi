#!/usr/bin/env python3
"""Deterministic, dependency-free architecture and size gate."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

LIMITS = {"file": 800, "function": 120, "class": 400}
WARNINGS = {"file": 500, "function": 80, "class": 250}
DEFAULT_ROOTS = ("trans_novel", "tests", "scripts")

# Values are package names, deliberately kept as data so the rule is reviewable.
ALLOWED_DEPENDENCIES: dict[str, frozenset[str]] = {
    "cli": frozenset(("config", "pipeline", "benchmark")),
    "pipeline": frozenset(
        (
            "config",
            "ingest",
            "epub",
            "agents",
            "glossary",
            "llm",
            "assemble",
            "postprocess",
            "model_profiles",
        )
    ),
    "ingest": frozenset(("epub",)),
    "assemble": frozenset(("ingest", "epub", "postprocess")),
    "epub": frozenset(("postprocess",)),
    "agents": frozenset(("llm", "config", "glossary", "ingest")),
    "glossary": frozenset(),
    "llm": frozenset(("config", "model_profiles")),
    "benchmark": frozenset(
        (
            "config",
            "pipeline",
            "ingest",
            "assemble",
            "epub",
            "agents",
            "glossary",
            "llm",
            "postprocess",
            "model_profiles",
        )
    ),
    "postprocess": frozenset(),
    "config": frozenset(("model_profiles",)),
    "model_profiles": frozenset(),
}
# Capability names are relative to ``trans_novel`` and match the capability
# root itself or one of its descendants.  Keep this map explicit so ownership
# directions remain easy to review.
CAPABILITY_DEPENDENCIES: dict[str, frozenset[str]] = {
    "pipeline.application": frozenset(
        (
            "pipeline.composition",
            "pipeline.contracts",
            "pipeline.execution",
            "pipeline.nodes",
            "pipeline.planning",
            "pipeline.quality",
            "pipeline.state",
        )
    ),
    "pipeline.composition": frozenset(("pipeline.contracts", "pipeline.nodes", "pipeline.state")),
    "pipeline.execution": frozenset(("pipeline.contracts", "pipeline.planning", "pipeline.state")),
    "pipeline.nodes": frozenset(
        ("pipeline.contracts", "pipeline.planning", "pipeline.quality", "pipeline.state")
    ),
    "pipeline.planning": frozenset(("pipeline.contracts", "pipeline.state")),
    "pipeline.contracts": frozenset(("pipeline.state",)),
    "pipeline.quality": frozenset(),
    "pipeline.state": frozenset(),
    "assemble.epub.publication": frozenset(("assemble.epub.verification",)),
    "assemble.epub.verification": frozenset(("assemble.epub.rendering",)),
    "assemble.epub.rendering": frozenset(),
    "benchmark.corpus": frozenset(),
    "benchmark.run": frozenset(("benchmark.corpus",)),
    "benchmark.integration": frozenset(("benchmark.corpus", "benchmark.run")),
    "benchmark.review": frozenset(("benchmark.corpus",)),
    "benchmark.report": frozenset(("benchmark.review",)),
}


# The agents -> ingest edge is limited to the data-contract module.
ALLOWED_DEPENDENCY_MODULES: dict[tuple[str, str], frozenset[str]] = {
    ("agents", "ingest"): frozenset(("trans_novel.ingest.models",)),
}


@dataclass(frozen=True)
class SymbolRecord:
    path: str
    symbol: str
    kind: str
    value: int


@dataclass(frozen=True)
class ImportRecord:
    source: str
    target: str
    path: str
    line: int
    kind: str
    imported: str = ""


@dataclass(frozen=True)
class ScanResult:
    files: tuple[tuple[str, int], ...]
    symbols: tuple[SymbolRecord, ...]
    imports: tuple[ImportRecord, ...]
    cycles: tuple[tuple[str, tuple[str, ...]], ...]
    private_imports: tuple[ImportRecord, ...]
    forbidden_edges: tuple[tuple[str, str], ...]
    forbidden_capability_edges: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class Diagnostic:
    rule: str
    path: str
    symbol: str
    current: int | str
    limit: int | str
    remediation: str
    warning: bool = False

    def render(self) -> str:
        level = "WARNING" if self.warning else "ERROR"
        subject = f"{self.path}:{self.symbol}" if self.symbol else self.path
        return f"{level} {self.rule} {subject} current={self.current} limit={self.limit}; {self.remediation}"


def _python_files(root: Path, roots: Iterable[str] = DEFAULT_ROOTS) -> list[Path]:
    files: list[Path] = []
    for relative in roots:
        directory = root / relative
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*.py") if path.is_file())
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _module_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _span(node: ast.AST) -> int:
    start = getattr(node, "lineno", 1)
    decorators = getattr(node, "decorator_list", ())
    if decorators:
        start = min(start, *(decorator.lineno for decorator in decorators))
    end = getattr(node, "end_lineno", start)
    return end - start + 1


def _symbols(tree: ast.AST, path: str) -> Iterator[SymbolRecord]:
    def visit(node: ast.AST, parents: tuple[str, ...]) -> Iterator[SymbolRecord]:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            name = ".".join((*parents, node.name))
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            yield SymbolRecord(path, name, kind, _span(node))
            parents = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            yield from visit(child, parents)

    yield from visit(tree, ())


def _is_type_checking(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_candidates(name: str, modules: set[str]) -> str | None:
    if name in modules:
        return name
    pieces = name.split(".")
    for index in range(len(pieces) - 1, 0, -1):
        candidate = ".".join(pieces[:index])
        if candidate in modules:
            return candidate
    return None


def _resolve_import(
    source: str,
    node: ast.ImportFrom,
    imported: str,
    modules: set[str],
    *,
    is_package: bool = False,
) -> str | None:
    source_parts = source.split(".")
    package = source_parts if is_package else source_parts[:-1]
    if node.level:
        if node.level > len(package) + 1:
            return None
        base = package[: len(package) - node.level + 1]
    else:
        base = []
    prefix = ".".join(base)
    requested = node.module or ("" if imported == "*" else imported)
    full = ".".join(part for part in (prefix, requested) if part)
    if node.module and imported != "*":
        submodule = _module_candidates(f"{full}.{imported}", modules)
        if submodule:
            return submodule
    target = _module_candidates(full, modules)
    if target:
        return target
    if node.module is None:
        return _module_candidates(".".join(part for part in (prefix, imported) if part), modules)


def _package_for(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def _capability_for(module: str) -> str | None:
    relative = module.removeprefix("trans_novel.")
    matches = [
        capability
        for capability in CAPABILITY_DEPENDENCIES
        if relative == capability or relative.startswith(f"{capability}.")
    ]
    return max(matches, key=len, default=None)


def _imports(tree: ast.AST, source: str, path: str, modules: set[str]) -> list[ImportRecord]:
    records: list[ImportRecord] = []

    def visit(node: ast.AST, type_only: bool = False) -> None:
        if isinstance(node, ast.If):
            child_type = type_only or _is_type_checking(node.test)
            for child in node.body:
                visit(child, child_type)
            for child in node.orelse:
                visit(child, type_only)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _module_candidates(alias.name, modules)
                if target:
                    records.append(
                        ImportRecord(
                            source,
                            target,
                            path,
                            node.lineno,
                            "type" if type_only else "runtime",
                            alias.name.rsplit(".", 1)[-1],
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                target = _resolve_import(
                    source,
                    node,
                    alias.name,
                    modules,
                    is_package=Path(path).name == "__init__.py",
                )
                if target:
                    records.append(
                        ImportRecord(
                            source,
                            target,
                            path,
                            node.lineno,
                            "type" if type_only else "runtime",
                            alias.name,
                        )
                    )
        for child in ast.iter_child_nodes(node):
            visit(child, type_only)

    visit(tree)
    return records


def _scc(edges: Iterable[ImportRecord], kind: str | None) -> list[tuple[str, ...]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if kind is None or edge.kind == kind:
            graph[edge.source].add(edge.target)
    nodes = sorted(set(graph) | {target for values in graph.values() for target in values})
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    result: list[tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph.get(node, ())):
            if target not in indices:
                strongconnect(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component: list[str] = []
            while True:
                target = stack.pop()
                active.remove(target)
                component.append(target)
                if target == node:
                    break
            if len(component) > 1 or node in graph.get(node, set()):
                result.append(tuple(sorted(component)))

    for node in nodes:
        if node not in indices:
            strongconnect(node)
    return sorted(result)


def scan_repository(root: str | Path, roots: Iterable[str] = DEFAULT_ROOTS) -> ScanResult:
    root = Path(root).resolve()
    files = _python_files(root, roots)
    modules = {
        _module_for(path, root) for path in files if path.is_relative_to(root / "trans_novel")
    }
    symbols: list[SymbolRecord] = []
    imports: list[ImportRecord] = []
    file_sizes: list[tuple[str, int]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        file_sizes.append((relative, len(text.splitlines())))
        tree = ast.parse(text, filename=relative)
        symbols.extend(_symbols(tree, relative))
        imports.extend(_imports(tree, _module_for(path, root), relative, modules))
    private = [
        edge
        for edge in imports
        if edge.source in modules and edge.source != edge.target and edge.imported.startswith("_")
    ]
    forbidden: set[tuple[str, str]] = set()
    forbidden_capability: set[tuple[str, str, str, str]] = set()
    production = set(modules)
    for edge in imports:
        if edge.source not in production or edge.target not in production:
            continue
        source_package = _package_for(edge.source)
        target_package = _package_for(edge.target)
        if source_package == "__main__":
            source_package = "cli"
        allowed_packages = ALLOWED_DEPENDENCIES.get(source_package, frozenset())
        allowed_modules = ALLOWED_DEPENDENCY_MODULES.get((source_package, target_package))
        if source_package != target_package and (
            target_package not in allowed_packages
            or (allowed_modules is not None and edge.target not in allowed_modules)
        ):
            forbidden.add((edge.source, edge.target))

        source_capability = _capability_for(edge.source)
        target_capability = _capability_for(edge.target)
        if (
            source_capability
            and target_capability
            and source_capability.split(".", 1)[0] == target_capability.split(".", 1)[0]
            and source_capability != target_capability
            and target_capability not in CAPABILITY_DEPENDENCIES[source_capability]
        ):
            forbidden_capability.add(
                (edge.source, edge.target, source_capability, target_capability)
            )
    production_imports = [
        edge for edge in imports if edge.source in production and edge.target in production
    ]
    cycles = [("runtime", component) for component in _scc(production_imports, "runtime")]
    for component in _scc(production_imports, None):
        members = set(component)
        kinds = {
            edge.kind
            for edge in production_imports
            if edge.source in members and edge.target in members
        }
        if "type" in kinds:
            cycles.append(("type", component))
    cycles = sorted(set(cycles))
    return ScanResult(
        tuple(file_sizes),
        tuple(sorted(symbols, key=lambda x: (x.path, x.symbol, x.kind))),
        tuple(sorted(imports, key=lambda x: (x.path, x.line, x.target, x.imported))),
        tuple(cycles),
        tuple(sorted(private, key=lambda x: (x.path, x.line, x.target, x.imported))),
        tuple(sorted(forbidden)),
        tuple(sorted(forbidden_capability)),
    )


def _violation_records(scan: ScanResult) -> dict[str, list[dict[str, object]]]:
    return {
        "files": [
            {"path": path, "lines": value} for path, value in scan.files if value > LIMITS["file"]
        ],
        "symbols": [
            {"path": x.path, "symbol": x.symbol, "kind": x.kind, "lines": x.value}
            for x in scan.symbols
            if x.value > LIMITS[x.kind]
        ],
        "cycles": [{"kind": kind, "modules": list(component)} for kind, component in scan.cycles],
        "private_imports": [
            {"path": x.path, "line": x.line, "module": x.target, "name": x.imported}
            for x in scan.private_imports
        ],
        "forbidden_edges": [
            {"source": source, "target": target} for source, target in scan.forbidden_edges
        ],
        "capability_edges": [
            {
                "source": source,
                "target": target,
                "source_capability": source_capability,
                "target_capability": target_capability,
            }
            for source, target, source_capability, target_capability in scan.forbidden_capability_edges
        ],
    }


def baseline_data(scan: ScanResult) -> dict[str, object]:
    return {"version": 1, "limits": LIMITS, "violations": _violation_records(scan)}


def _compare_baselines(candidate: dict[str, object], base: dict[str, object]) -> list[Diagnostic]:
    candidate_violations = candidate.get("violations", {})
    base_violations = base.get("violations", {})
    if not isinstance(candidate_violations, dict) or not isinstance(base_violations, dict):
        return [_error("baseline violations must be objects", "architecture-baseline.json")]
    diagnostics: list[Diagnostic] = []
    for category in (
        "files",
        "symbols",
        "cycles",
        "private_imports",
        "forbidden_edges",
        "capability_edges",
    ):
        candidate_records = [
            item for item in candidate_violations.get(category, []) if isinstance(item, dict)
        ]
        base_records = [
            item for item in base_violations.get(category, []) if isinstance(item, dict)
        ]
        base_by_key = {_key(category, item): item for item in base_records}
        for item in candidate_records:
            key = _key(category, item)
            old = base_by_key.get(key)
            value = item.get("lines")
            old_value = old.get("lines") if old else None
            if old is None or (
                category in {"files", "symbols"}
                and isinstance(value, int)
                and isinstance(old_value, int)
                and value > old_value
            ):
                subject = str(item.get("path", item.get("source", "")))
                diagnostics.append(
                    Diagnostic(
                        "baseline-ratchet",
                        subject,
                        str(item.get("symbol", "")),
                        value if value is not None else "present",
                        old_value if old_value is not None else "absent",
                        "remove the new baseline entry or lower its recorded value",
                    )
                )
    return diagnostics


def _key(category: str, record: dict[str, object]) -> tuple[object, ...]:
    if category == "files":
        return (record.get("path"),)
    if category == "symbols":
        return (record.get("path"), record.get("symbol"), record.get("kind"))
    if category == "cycles":
        return (record.get("kind"), tuple(record.get("modules", ())))
    if category == "private_imports":
        return (record.get("path"), record.get("line"), record.get("module"), record.get("name"))
    if category == "capability_edges":
        return (
            record.get("source"),
            record.get("target"),
            record.get("source_capability"),
            record.get("target_capability"),
        )
    return (record.get("source"), record.get("target"))


def _diagnostics(scan: ScanResult, baseline: dict[str, object]) -> list[Diagnostic]:
    current = _violation_records(scan)
    raw = baseline.get("violations", {})
    if not isinstance(raw, dict):
        raw = {}
    diagnostics: list[Diagnostic] = []
    for category in current:
        old_records = [item for item in raw.get(category, []) if isinstance(item, dict)]
        old_by_key = {_key(category, item): item for item in old_records}
        current_by_key = {_key(category, item): item for item in current[category]}
        for key, item in current_by_key.items():
            old = old_by_key.get(key)
            value = item.get("lines", 1)
            old_value = old.get("lines", 0) if old else None
            if old and category not in {"files", "symbols"}:
                continue
            if (
                old
                and category in {"files", "symbols"}
                and isinstance(value, int)
                and isinstance(old_value, int)
                and value <= old_value
            ):
                continue
            subject = str(item.get("path", item.get("source", "")))
            symbol = str(item.get("symbol", item.get("name", "")))
            if category == "cycles":
                subject = ",".join(item.get("modules", []))
            if category in {"forbidden_edges", "capability_edges"}:
                subject = f"{item.get('source')} -> {item.get('target')}"
            rule = "capability_dependencies" if category == "capability_edges" else category
            diagnostics.append(
                Diagnostic(
                    rule,
                    subject,
                    symbol,
                    value,
                    old_value if old_value is not None else "baseline",
                    "remove the violation or update the code before changing the baseline",
                )
            )
        for key, item in old_by_key.items():
            if key not in current_by_key:
                subject = str(
                    item.get("path", item.get("source", ",".join(item.get("modules", []))))
                )
                diagnostics.append(
                    Diagnostic(
                        "stale-baseline",
                        subject,
                        str(item.get("symbol", "")),
                        "absent",
                        "present",
                        "remove the resolved entry from architecture-baseline.json",
                    )
                )
    return diagnostics


def _diff_additions(root: Path, base: str, head: str) -> tuple[dict[str, int], str | None]:
    if head == "WORKTREE":
        command = ["git", "-C", str(root), "diff", "--numstat", "-M", base, "--", *DEFAULT_ROOTS]
    else:
        command = [
            "git",
            "-C",
            str(root),
            "diff",
            "--numstat",
            "-M",
            base,
            head,
            "--",
            *DEFAULT_ROOTS,
        ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return {}, str(error)
    additions: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        path = fields[2]
        if " => " in path:
            if "{" in path:
                prefix, renamed = path.split("{", 1)
                path = prefix + renamed.rstrip("}").split(" => ", 1)[1]
            else:
                path = path.rsplit(" => ", 1)[1]
        if path.endswith(".py"):
            net = int(fields[0]) - int(fields[1])
            if net > 0:
                additions[path] = net
    return additions, None


def _load_json(path: Path) -> tuple[dict[str, object] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, str(error)
    if not isinstance(value, dict):
        return None, "baseline must be a JSON object"
    return value, None


def _baseline_at_revision(
    root: Path, revision: str, baseline_file: Path
) -> tuple[dict[str, object] | None, str | None]:
    try:
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", f"{revision}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return None, f"invalid base revision: {error}"
    try:
        relative = baseline_file.relative_to(root).as_posix()
    except ValueError:
        relative = baseline_file.name
    object_name = f"{revision}:{relative}"
    exists = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", object_name],
        capture_output=True,
        text=True,
    )
    if exists.returncode != 0:
        return None, None
    try:
        shown = subprocess.run(
            ["git", "-C", str(root), "show", object_name],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(shown.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return None, f"invalid baseline at {revision}: {error}"
    if not isinstance(value, dict):
        return None, f"baseline at {revision} must be a JSON object"
    return value, None


def _error(message: str, path: str) -> Diagnostic:
    return Diagnostic(
        "baseline", path, "", message, "valid baseline", "repair the baseline or git range"
    )


def check(
    root: str | Path,
    baseline_path: str | Path = "architecture-baseline.json",
    *,
    base: str | None = None,
    head: str | None = None,
) -> list[Diagnostic]:
    root = Path(root).resolve()
    scan = scan_repository(root)
    baseline_file = Path(baseline_path)
    if not baseline_file.is_absolute():
        baseline_file = root / baseline_file
    candidate, candidate_error = _load_json(baseline_file)
    if candidate_error:
        return [_error(candidate_error, str(baseline_file))]
    assert candidate is not None
    diagnostics: list[Diagnostic] = []
    if base:
        historical, historical_error = _baseline_at_revision(root, base, baseline_file)
        if historical_error:
            return [_error(historical_error, str(baseline_file))]
        if historical is not None:
            diagnostics.extend(_compare_baselines(candidate, historical))
    diagnostics.extend(_diagnostics(scan, candidate))
    for path, value in scan.files:
        if WARNINGS["file"] < value <= LIMITS["file"]:
            diagnostics.append(
                Diagnostic(
                    "file-warning",
                    path,
                    "",
                    value,
                    LIMITS["file"],
                    "consider splitting responsibilities",
                    True,
                )
            )
    for item in scan.symbols:
        if WARNINGS[item.kind] < item.value <= LIMITS[item.kind]:
            diagnostics.append(
                Diagnostic(
                    f"{item.kind}-warning",
                    item.path,
                    item.symbol,
                    item.value,
                    LIMITS[item.kind],
                    "consider splitting responsibilities",
                    True,
                )
            )
    if base and head:
        diff_result = _diff_additions(root, base, head)
        if isinstance(diff_result, tuple):
            net_added_by_path, diff_error = diff_result
        else:
            net_added_by_path, diff_error = diff_result, None
        if diff_error:
            diagnostics.append(
                Diagnostic(
                    "diff-range",
                    f"{base}..{head}",
                    "",
                    diff_error,
                    "valid git range",
                    "provide valid base/head revisions",
                )
            )
        else:
            for path, net_added in sorted(net_added_by_path.items()):
                if net_added > 300:
                    diagnostics.append(
                        Diagnostic(
                            "architecture-delta",
                            path,
                            "",
                            net_added,
                            300,
                            "provide Architecture Delta review for this concentrated change",
                            True,
                        )
                    )
    return sorted(diagnostics, key=lambda x: (x.path, x.symbol, x.rule, str(x.current)))


def write_baseline(
    root: str | Path, path: str | Path = "architecture-baseline.json"
) -> list[Diagnostic]:
    root = Path(root).resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    scan = scan_repository(root)
    candidate = baseline_data(scan)
    if target.exists():
        baseline, error = _load_json(target)
        if error:
            return [_error(error, str(target))]
        assert baseline is not None
        diagnostics = _diagnostics(scan, baseline)
        hard = [item for item in diagnostics if item.rule != "stale-baseline" and not item.warning]
        if hard:
            return diagnostics
    target.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline", type=Path, default=Path("architecture-baseline.json"))
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--update-baseline", "--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.base) != bool(args.head):
        parser.error("--base and --head must be supplied together")
    if args.update_baseline:
        diagnostics = write_baseline(args.root, args.baseline)
        for diagnostic in diagnostics:
            print(diagnostic.render())
        return 1 if any(not diagnostic.warning for diagnostic in diagnostics) else 0
    diagnostics = check(args.root, args.baseline, base=args.base, head=args.head)
    for diagnostic in diagnostics:
        print(diagnostic.render())
    return 1 if any(not diagnostic.warning for diagnostic in diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
