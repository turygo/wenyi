from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import yaml

from tests.fake_llm import routing_handler
from tests.sample_data import write_inline_sample_epub, write_phase9_epub, write_sample_epub
from trans_novel.benchmark.runner import (
    CanaryRunner,
    FullRunner,
    load_candidate_spec,
    validate_candidate_capabilities,
)
from trans_novel.config import PipelineConfig
from trans_novel.llm.providers.fake import FakeClient
from trans_novel.pipeline.bootstrap import Application
from trans_novel.pipeline.runstore import RunStore


class _SinkFakeClient(FakeClient):
    def __init__(self) -> None:
        super().__init__(handler=routing_handler)
        self.telemetry_sink = None

    def set_telemetry_sink(self, sink) -> None:
        self.telemetry_sink = sink


def _write_yaml(path: Path, value: object) -> None:
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _candidate_spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "benchmark_id": "test-production-path",
        "provider": "bailian",
        "fast_model": "qwen3.7-plus:off",
        "temperature": 0.1,
        "seed": None,
        "replicates": 1,
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "primary_model": "qwen3.7-plus:off",
                "editor_model": "deepseek-v4-flash:off",
            }
        ],
    }


def _complete_book_entries(root: Path) -> list[dict[str, str]]:
    books: list[dict[str, str]] = []
    for split, count in (("screen", 3), ("formal", 6), ("hidden", 1)):
        for index in range(1, count + 1):
            source = root / f"{split}-{index:02d}.epub"
            (write_inline_sample_epub if split == "formal" else write_sample_epub)(str(source))
            with zipfile.ZipFile(source, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("META-INF/benchmark-id.txt", f"{split}-{index:02d}")
            books.append({"book_id": f"{split}-{index:02d}", "path": source.name, "split": split})
    return books


def _request_fingerprints(calls: list[dict[str, object]]) -> list[tuple[str, str, str]]:
    rows = []
    for call in calls:
        messages = json.dumps(
            call["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        rows.append(
            (
                str(call["agent"]),
                str(call["operation"]),
                hashlib.sha256(messages.encode("utf-8")).hexdigest(),
            )
        )
    return sorted(rows)


def _only_store(state_root: Path) -> RunStore:
    paths = [path for path in state_root.iterdir() if (path / "manifest.json").is_file()]
    if len(paths) != 1:
        raise AssertionError(f"expected one run store, got {paths}")
    return RunStore(str(paths[0]))


def _targets(store: RunStore) -> list[str]:
    values: list[str] = []
    for chapter in store.load_state().chapters:
        values.extend(
            segment.target or "" for segment in store.load_chapter(chapter.index).text_segments
        )
    return values


class TestBenchmarkProductionRunner(unittest.TestCase):
    def test_quality_presets_always_polish_except_economy(self) -> None:
        self.assertFalse(PipelineConfig.for_quality("economy").polish)
        self.assertTrue(PipelineConfig.for_quality("balanced").polish)
        self.assertTrue(PipelineConfig.for_quality("quality").polish)

    def test_canary_requests_and_targets_match_direct_production_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books = _complete_book_entries(root)
            source = root / "screen-01.epub"
            book_spec = root / "books.yaml"
            _write_yaml(
                book_spec,
                {
                    "schema_version": 1,
                    "source_language": "en",
                    "target_language": "zh",
                    "books": books,
                },
            )
            candidates = root / "candidates.yaml"
            _write_yaml(candidates, _candidate_spec())

            direct_client = _SinkFakeClient()
            direct_state = root / "direct-state"
            direct_output = root / "direct.epub"
            config = FullRunner._config(
                load_candidate_spec(candidates),
                "qwen3.7-plus:off",
                "deepseek-v4-flash:off",
                quality=True,
                state_dir=str(direct_state),
            )
            Application(config, client=direct_client).run_all(
                str(source), out_format="epub", out_path=str(direct_output)
            )

            runner_client = _SinkFakeClient()
            result = CanaryRunner(client=runner_client).run(book_spec, candidates, root / "canary")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["branch_count"], 1)
            self.assertEqual(
                _request_fingerprints(runner_client.calls),
                _request_fingerprints(direct_client.calls),
            )

            rows = [
                json.loads(line)
                for line in next((root / "canary").rglob("segments.jsonl"))
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["target"] for row in rows], _targets(_only_store(direct_state)))

    def test_full_runner_processes_exactly_six_formal_epubs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books = _complete_book_entries(root)
            formal_books = [book for book in books if book["split"] == "formal"]
            book_spec = root / "books.yaml"
            _write_yaml(
                book_spec,
                {
                    "schema_version": 1,
                    "source_language": "en",
                    "target_language": "zh",
                    "books": books,
                },
            )
            candidates = root / "candidates.yaml"
            _write_yaml(candidates, _candidate_spec())
            clients: list[_SinkFakeClient] = []

            def factory(**_kwargs):
                client = _SinkFakeClient()
                clients.append(client)
                return client

            result = FullRunner(client_factory=factory).run(book_spec, candidates, root / "full")
            self.assertEqual(result["branch_count"], 6)
            artifacts = json.loads((root / "full" / "candidates.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {row["book_id"] for row in artifacts},
                {row["book_id"] for row in formal_books},
            )
            self.assertEqual(len(clients), 6)
            self.assertTrue(all(client.calls for client in clients))

    def test_full_runner_rejects_multi_chapter_formal_epub_before_clients(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books = _complete_book_entries(root)
            write_phase9_epub(str(root / "formal-01.epub"))
            book_spec = root / "books.yaml"
            _write_yaml(
                book_spec,
                {
                    "schema_version": 1,
                    "source_language": "en",
                    "target_language": "zh",
                    "books": books,
                },
            )
            candidates = root / "candidates.yaml"
            _write_yaml(candidates, _candidate_spec())
            factory = mock.Mock()

            with self.assertRaisesRegex(
                ValueError, "formal-01 must contain exactly one chapter; found 2"
            ):
                FullRunner(client_factory=factory).run(book_spec, candidates, root / "full")

            factory.assert_not_called()

    def test_candidate_editor_is_required(self) -> None:
        spec = _candidate_spec()
        del spec["candidates"][0]["editor_model"]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.yaml"
            _write_yaml(path, spec)
            with self.assertRaisesRegex(ValueError, "editor_model"):
                load_candidate_spec(path)

    def test_repository_candidates_are_catalogued_opencode_go_models(self) -> None:
        path = Path(__file__).parents[1] / "CANDIDATES.yaml"
        spec = load_candidate_spec(path)
        validate_candidate_capabilities(spec)
        self.assertEqual(spec.provider, "opencode-go")
        self.assertEqual(len(spec.candidates), 3)


if __name__ == "__main__":
    unittest.main()
