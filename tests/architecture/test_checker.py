from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_architecture


class ArchitectureFixtures(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "trans_novel").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "trans_novel" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "tests" / "__init__.py").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def add(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def baseline(self) -> dict[str, object]:
        return check_architecture.baseline_data(check_architecture.scan_repository(self.root))

    def save_baseline(self, data: dict[str, object] | None = None) -> None:
        payload = self.baseline() if data is None else data
        (self.root / "architecture-baseline.json").write_text(json.dumps(payload), encoding="utf-8")


class ArchitectureCheckerTests(ArchitectureFixtures):
    def test_line_limits_and_decorated_nested_symbols(self) -> None:
        body = "\n".join("    x = 1" for _ in range(121))
        self.add(
            "trans_novel/large.py",
            "def deco(fn):\n    return fn\n\n@deco\ndef outer():\n    class Inner:\n        pass\n\n"
            + body
            + "\n",
        )
        scan = check_architecture.scan_repository(self.root)
        symbols = {(item.symbol, item.kind): item.value for item in scan.symbols}
        self.assertEqual(symbols[("outer", "function")], 126)
        self.assertEqual(symbols[("outer.Inner", "class")], 2)
        self.assertEqual(dict(scan.files)["trans_novel/large.py"], 129)

    def test_baseline_growth_deletion_and_reintroduction(self) -> None:
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(801)))
        self.save_baseline()
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(802)))
        diagnostics = check_architecture.check(self.root)
        self.assertTrue(any(item.rule == "files" and not item.warning for item in diagnostics))
        self.add("trans_novel/large.py", "small = 1\n")
        diagnostics = check_architecture.check(self.root)
        self.assertTrue(any(item.rule == "stale-baseline" for item in diagnostics))
        self.save_baseline()
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(801)))
        diagnostics = check_architecture.check(self.root)
        self.assertTrue(any(item.rule == "files" and not item.warning for item in diagnostics))

    def test_private_imports_local_and_type_only_edges(self) -> None:
        self.add("trans_novel/a.py", "from .b import _private\nfrom . import c\n")
        self.add("trans_novel/b.py", "_private = 1\n")
        self.add(
            "trans_novel/c.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .b import _private\n",
        )
        scan = check_architecture.scan_repository(self.root)
        self.assertEqual(len(scan.private_imports), 2)
        kinds = {(item.source, item.target, item.kind) for item in scan.imports}
        self.assertIn(("trans_novel.a", "trans_novel.b", "runtime"), kinds)
        self.assertIn(("trans_novel.c", "trans_novel.b", "type"), kinds)

    def test_runtime_and_type_cycles_are_separate(self) -> None:
        self.add("trans_novel/a.py", "from .b import value\n")
        self.add("trans_novel/b.py", "from .a import value\n")
        self.add(
            "trans_novel/c.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .d import value\n",
        )
        self.add(
            "trans_novel/d.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .c import value\n",
        )
        scan = check_architecture.scan_repository(self.root)
        self.assertIn(("runtime", ("trans_novel.a", "trans_novel.b")), scan.cycles)
        self.assertIn(("type", ("trans_novel.c", "trans_novel.d")), scan.cycles)

    def test_forbidden_edges_are_exact_and_deterministic(self) -> None:
        self.add("trans_novel/ingest/read.py", "from trans_novel.pipeline import value\n")
        self.add("trans_novel/pipeline/value.py", "value = 1\n")
        self.add("trans_novel/model_profiles.py", "value = 1\n")
        self.add(
            "trans_novel/pipeline/profile_use.py",
            "from trans_novel.model_profiles import value\n",
        )
        first = check_architecture.scan_repository(self.root)
        second = check_architecture.scan_repository(self.root)
        self.assertEqual(first.forbidden_edges, second.forbidden_edges)
        self.assertEqual(
            first.forbidden_edges, (("trans_novel.ingest.read", "trans_novel.pipeline.value"),)
        )

    def test_agents_ingest_edge_allows_only_models_contract(self) -> None:
        self.add(
            "trans_novel/agents/translator.py",
            "from trans_novel.ingest.models import KIND_HEADING\n",
        )
        self.add(
            "trans_novel/agents/parser.py",
            "from trans_novel.ingest.epub_reader import read_epub\n",
        )
        self.add("trans_novel/ingest/models.py", "KIND_HEADING = 'heading'\n")
        self.add(
            "trans_novel/ingest/epub_reader.py",
            "def read_epub():\n    return None\n",
        )

        scan = check_architecture.scan_repository(self.root)

        self.assertEqual(
            scan.forbidden_edges,
            (("trans_novel.agents.parser", "trans_novel.ingest.epub_reader"),),
        )

    def test_baseline_generation_and_deterministic_output(self) -> None:
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(501)))
        baseline = self.root / "architecture-baseline.json"
        self.assertEqual(
            check_architecture.main(
                ["--root", str(self.root), "--baseline", str(baseline), "--update-baseline"]
            ),
            0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = check_architecture.main(
                ["--root", str(self.root), "--baseline", str(baseline)]
            )
        self.assertEqual(status, 0)
        first_output = output.getvalue()
        self.assertIn("WARNING", first_output)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check_architecture.main(["--root", str(self.root), "--baseline", str(baseline)])
        self.assertEqual(output.getvalue(), first_output)

    def test_exact_non_numeric_debt_ratchets_and_stales(self) -> None:
        self.add("trans_novel/a.py", "from .b import value\n")
        self.add("trans_novel/b.py", "from .a import value\n")
        self.add("trans_novel/c.py", "from .d import _private\n")
        self.add("trans_novel/d.py", "_private = 1\n")
        self.add("trans_novel/ingest/read.py", "from trans_novel.pipeline import value\n")
        self.add("trans_novel/pipeline/value.py", "value = 1\n")
        self.save_baseline()
        self.assertFalse(any(not item.warning for item in check_architecture.check(self.root)))

        self.add("trans_novel/a.py", "from .b import value\n")
        self.add("trans_novel/b.py", "from .a import value\n")
        self.add("trans_novel/e.py", "from .f import value\n")
        self.add("trans_novel/f.py", "from .e import value\n")
        diagnostics = check_architecture.check(self.root)
        self.assertTrue(any(item.rule == "cycles" and not item.warning for item in diagnostics))

        self.add("trans_novel/a.py", "value = 1\n")
        self.add("trans_novel/b.py", "value = 1\n")
        diagnostics = check_architecture.check(self.root)
        self.assertTrue(
            any(item.rule == "stale-baseline" and not item.warning for item in diagnostics)
        )

    def test_base_baseline_bootstrap_and_revision_are_authoritative(self) -> None:
        baseline = self.baseline()
        (self.root / "architecture-baseline.json").write_text("{", encoding="utf-8")
        with (
            mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(baseline, None),
            ),
            mock.patch.object(
                check_architecture,
                "_diff_additions",
                return_value=({}, None),
            ),
        ):
            diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertTrue(any(item.rule == "baseline" for item in diagnostics))
        (self.root / "architecture-baseline.json").unlink()
        diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertTrue(any(item.rule == "baseline" for item in diagnostics))
        self.save_baseline()
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(801)))
        self.save_baseline()

        with (
            mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(None, None),
            ),
            mock.patch.object(
                check_architecture,
                "_diff_additions",
                return_value=({}, None),
            ),
        ):
            diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertFalse(any(not item.warning for item in diagnostics))

    def test_base_ratchet_rejects_candidate_growth_and_allows_deletion(self) -> None:
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(801)))
        historical = self.baseline()
        candidate = json.loads(json.dumps(historical))
        candidate["violations"]["files"][0]["lines"] = 802
        (self.root / "architecture-baseline.json").write_text(
            json.dumps(candidate), encoding="utf-8"
        )
        with (
            mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(historical, None),
            ),
            mock.patch.object(
                check_architecture,
                "_diff_additions",
                return_value=({}, None),
            ),
        ):
            diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertTrue(any(item.rule == "baseline-ratchet" for item in diagnostics))

        self.add("trans_novel/large.py", "small = 1\n")
        candidate["violations"]["files"] = []
        (self.root / "architecture-baseline.json").write_text(
            json.dumps(candidate), encoding="utf-8"
        )
        with (
            mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(historical, None),
            ),
            mock.patch.object(
                check_architecture,
                "_diff_additions",
                return_value=({}, None),
            ),
        ):
            diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertFalse(any(not item.warning for item in diagnostics))

    def test_update_baseline_existing_file_is_ratchet_only(self) -> None:
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(801)))
        self.assertEqual(check_architecture.write_baseline(self.root), [])
        before = (self.root / "architecture-baseline.json").read_text(encoding="utf-8")
        self.add("trans_novel/large.py", "\n".join("x = 1" for _ in range(802)))
        diagnostics = check_architecture.write_baseline(self.root)
        self.assertTrue(any(item.rule == "files" for item in diagnostics))
        self.assertEqual(
            (self.root / "architecture-baseline.json").read_text(encoding="utf-8"),
            before,
        )

        self.add("trans_novel/large.py", "small = 1\n")
        self.assertEqual(check_architecture.write_baseline(self.root), [])
        self.assertFalse(
            check_architecture.baseline_data(check_architecture.scan_repository(self.root))[
                "violations"
            ]["files"]
        )

    def test_wildcard_import_records_module_without_private_symbol(self) -> None:
        self.add("trans_novel/b.py", "value = 1\n")
        self.add("trans_novel/a.py", "from .b import *\n")
        scan = check_architecture.scan_repository(self.root)
        self.assertIn(
            ("trans_novel.a", "trans_novel.b", "runtime"),
            {(item.source, item.target, item.kind) for item in scan.imports},
        )
        self.assertFalse(scan.private_imports)

    def test_diff_net_additions_renames_and_failures(self) -> None:
        completed = mock.Mock(
            stdout="10\t4\ttrans_novel/a.py\n"
            "5\t2\ttrans_novel/{old.py => new.py}\n"
            "-\t-\ttrans_novel/binary.py\n"
        )
        with mock.patch.object(check_architecture.subprocess, "run", return_value=completed) as run:
            additions, error = check_architecture._diff_additions(self.root, "base", "WORKTREE")
        self.assertIsNone(error)
        self.assertEqual(additions, {"trans_novel/a.py": 6, "trans_novel/new.py": 3})
        self.assertNotIn("WORKTREE", run.call_args.args[0])

        self.save_baseline()
        with (
            mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(self.baseline(), None),
            ),
            mock.patch.object(
                check_architecture,
                "_diff_additions",
                return_value=({}, "invalid range"),
            ),
        ):
            diagnostics = check_architecture.check(self.root, base="base", head="WORKTREE")
        self.assertTrue(any(item.rule == "diff-range" for item in diagnostics))

        self.add("trans_novel/new.py", "value = 1\n")
        self.save_baseline()
        original = check_architecture._diff_additions
        check_architecture._diff_additions = lambda *_args: {"trans_novel/new.py": 301}
        try:
            with mock.patch.object(
                check_architecture,
                "_baseline_at_revision",
                return_value=(self.baseline(), None),
            ):
                diagnostics = check_architecture.check(self.root, base="base", head="head")
        finally:
            check_architecture._diff_additions = original
        self.assertEqual(
            [item.rule for item in diagnostics if item.rule == "architecture-delta"],
            ["architecture-delta"],
        )
        self.assertTrue(
            all(item.warning for item in diagnostics if item.rule == "architecture-delta")
        )


class CapabilityArchitectureCheckerTests(ArchitectureFixtures):
    def test_capability_edges_allow_same_and_declared_directions(self) -> None:
        self.add(
            "trans_novel/assemble/epub/verification/dom.py",
            "from .source import value\n",
        )
        self.add("trans_novel/assemble/epub/verification/source.py", "value = 1\n")
        self.add(
            "trans_novel/assemble/epub/publication.py",
            "from trans_novel.assemble.epub.verification import value\n",
        )
        scan = check_architecture.scan_repository(self.root)

        self.assertFalse(scan.forbidden_capability_edges)

    def test_capability_edges_reject_reverse_direction(self) -> None:
        self.add(
            "trans_novel/pipeline/planning/planner.py",
            "from trans_novel.pipeline.nodes import value\n",
        )
        self.add("trans_novel/pipeline/nodes/__init__.py", "value = 1\n")

        scan = check_architecture.scan_repository(self.root)
        self.assertEqual(
            scan.forbidden_capability_edges,
            (
                (
                    "trans_novel.pipeline.planning.planner",
                    "trans_novel.pipeline.nodes",
                    "pipeline.planning",
                    "pipeline.nodes",
                ),
            ),
        )

    def test_capability_edges_reject_epub_and_benchmark_reverse_directions(self) -> None:
        self.add(
            "trans_novel/assemble/epub/rendering/generated.py",
            "from trans_novel.assemble.epub.verification import value\n",
        )
        self.add("trans_novel/assemble/epub/verification/__init__.py", "value = 1\n")
        self.add(
            "trans_novel/benchmark/corpus/validation.py",
            "from trans_novel.benchmark.run import value\n",
        )
        self.add("trans_novel/benchmark/run/__init__.py", "value = 1\n")
        scan = check_architecture.scan_repository(self.root)
        self.assertEqual(
            scan.forbidden_capability_edges,
            (
                (
                    "trans_novel.assemble.epub.rendering.generated",
                    "trans_novel.assemble.epub.verification",
                    "assemble.epub.rendering",
                    "assemble.epub.verification",
                ),
                (
                    "trans_novel.benchmark.corpus.validation",
                    "trans_novel.benchmark.run",
                    "benchmark.corpus",
                    "benchmark.run",
                ),
            ),
        )

    def test_capability_edges_skip_cross_domain_tracked_capabilities(self) -> None:
        self.add(
            "trans_novel/benchmark/corpus/identity.py",
            "from trans_novel.pipeline.state import value\n",
        )
        self.add("trans_novel/pipeline/state/__init__.py", "value = 1\n")
        scan = check_architecture.scan_repository(self.root)
        self.assertFalse(scan.forbidden_capability_edges)

    def test_capability_match_requires_boundary_and_uses_longest_prefix(self) -> None:
        self.assertEqual(
            check_architecture._capability_for("trans_novel.assemble.epub.verification.dom"),
            "assemble.epub.verification",
        )
        self.assertIsNone(check_architecture._capability_for("trans_novel.pipeline.planning_extra"))
        self.assertIsNone(check_architecture._capability_for("trans_novel.assemble.epub.metadata"))

    def test_cli_module_overrides_are_removed(self) -> None:
        self.assertFalse(hasattr(check_architecture, "PACKAGE_OVERRIDES"))

    def test_capability_baseline_records_and_diagnostics_are_deterministic(self) -> None:
        self.add(
            "trans_novel/benchmark/corpus/identity.py",
            "from trans_novel.benchmark.run import value\n",
        )
        self.add("trans_novel/benchmark/run/__init__.py", "value = 1\n")
        first = check_architecture.scan_repository(self.root)
        second = check_architecture.scan_repository(self.root)
        self.assertEqual(first.forbidden_capability_edges, second.forbidden_capability_edges)
        baseline = check_architecture.baseline_data(first)
        self.assertEqual(
            baseline["violations"]["capability_edges"],
            [
                {
                    "source": "trans_novel.benchmark.corpus.identity",
                    "target": "trans_novel.benchmark.run",
                    "source_capability": "benchmark.corpus",
                    "target_capability": "benchmark.run",
                }
            ],
        )
        clean_baseline = json.loads(json.dumps(baseline))
        clean_baseline["violations"]["capability_edges"] = []
        self.save_baseline(clean_baseline)
        diagnostics = check_architecture.check(self.root)
        self.assertEqual(
            [(item.rule, item.path) for item in diagnostics if not item.warning],
            [
                (
                    "capability_dependencies",
                    "trans_novel.benchmark.corpus.identity -> trans_novel.benchmark.run",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
