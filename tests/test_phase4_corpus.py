"""Focused offline tests for deterministic corpus primitives and scan output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from trans_novel.benchmark.corpus import (
    CorpusError,
    build,
    canonical_json,
    count_words,
    passage_id,
    scan,
    segment_id,
    sha256_bytes,
    source_digest,
    validate_corpus,
)
from trans_novel.benchmark.schema import BookEntry, BookSpec, ContextChallenge, PassageSelection
from trans_novel.cli import app
from trans_novel.ingest.models import Chapter, Document, Segment


def _artifact_hash(corpus: dict, runner: list[dict], challenge_keys: list[dict]) -> str:
    semantics = {
        "corpus": {key: value for key, value in corpus.items() if key != "corpus_sha256"},
        "runner_segments": runner,
        "challenge_keys": challenge_keys,
    }
    encoded = json.dumps(semantics, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_minimal_artifact(
    root: Path,
    runner: list[dict],
    *,
    manifest_books: list[dict] | None = None,
    challenge_keys: list[dict] | None = None,
) -> None:
    challenge_keys = challenge_keys or []
    manifest_books = manifest_books or [
        {
            "book_id": "book",
            "source_sha256": "a" * 64,
            "basename": "book.txt",
            "split": "screen",
            "format": "text",
            "title": "book",
            "chapter_count": 1,
            "parser_schema": 1,
        }
    ]
    corpus = {
        "schema_version": 1,
        "benchmark_name": "fixture",
        "word_counter": "en-v1",
        "parser_schema": 1,
        "run_input_schema_version": 1,
        "books": manifest_books,
        "passages": [
            {
                key: row[key]
                for key in (
                    "passage_id",
                    "subset",
                    "book_id",
                    "chapter_index",
                    "start",
                    "end",
                    "word_count",
                    "strata",
                )
            }
            for row in runner
        ],
        "quotas": {
            "targets": {
                "screen": 10_000,
                "continuous": 30_000,
                "stratified": 15_000,
                "context": 5_000,
            },
            "actual": {
                "screen": 0,
                "continuous": 0,
                "stratified": 0,
                "context": 0,
                "hidden": 0,
                "formal": 0,
            },
            "tolerance": 0.2,
        },
    }
    corpus["corpus_sha256"] = _artifact_hash(corpus, runner, challenge_keys)
    (root / "corpus.json").write_text(
        json.dumps(corpus, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (root / "source_manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "run_input_schema_version": 1, "books": manifest_books},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "runner_segments.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in runner
        ),
        encoding="utf-8",
    )
    (root / "challenge_keys.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in challenge_keys
        ),
        encoding="utf-8",
    )


def _simple_runner(indices: list[int]) -> dict:
    book_sha = "a" * 64
    sources = {index: f"segment {index}" for index in indices}
    segments = [
        {
            "segment_id": (
                f"{book_sha}:c0000:s{index:04d}:"
                f"{hashlib.sha256(sources[index].encode()).hexdigest()[:8]}"
            ),
            "index": index,
            "source": sources[index],
            "kind": "text",
            "cont": False,
            "anchor": None,
            "resource_href": None,
            "meta": {},
        }
        for index in indices
    ]
    joined = "\n".join(sources[index] for index in indices)
    digest = hashlib.sha256(joined.encode()).hexdigest()[:12]
    return {
        "passage_id": f"book:c0000:s0000-0002:{digest}",
        "subset": "screen",
        "book_id": "book",
        "chapter_index": 0,
        "start": 0,
        "end": 2,
        "word_count": len(joined.split()),
        "strata": [],
        "segments": segments,
        "context": None,
    }


class CorpusSchemaTests(unittest.TestCase):
    def test_strict_context_and_extra_fields(self) -> None:
        with self.assertRaises(ValueError):
            BookSpec.model_validate(
                {
                    "schema_version": 1,
                    "source_language": "en",
                    "target_language": "zh",
                    "books": [
                        {
                            "book_id": "x",
                            "path": "x.txt",
                            "split": "screen",
                            "unexpected": True,
                        }
                    ],
                }
            )
        with self.assertRaises(ValueError):
            BookEntry.model_validate(
                {
                    "book_id": "x",
                    "path": "x.txt",
                    "split": "screen",
                    "license_note": "owned",
                }
            )
        with self.assertRaises(ValueError):
            PassageSelection.model_validate(
                {
                    "subset": "screen",
                    "book_id": "x",
                    "chapter_index": 0,
                    "start_segment_index": 0,
                    "end_segment_index": 0,
                    "context": {
                        "challenge_type": "polysemy",
                        "source_before": [{"chapter_index": 0, "segment_index": 0}],
                        "frozen_target_before": [
                            {"chapter_index": 0, "segment_index": 0, "target": "前文"}
                        ],
                        "answer_key": "答案",
                        "rationale": "理由",
                    },
                }
            )

    def test_context_frozen_targets_match_coordinates(self) -> None:
        challenge = ContextChallenge.model_validate(
            {
                "challenge_type": "polysemy",
                "source_before": [{"chapter_index": 0, "segment_index": 1}],
                "source_after": [],
                "frozen_target_before": [
                    {"chapter_index": 0, "segment_index": 1, "target": "前文"}
                ],
                "answer_key": "答案",
                "rationale": "理由",
            }
        )
        self.assertEqual(challenge.frozen_target_before[0].target, "前文")


class CorpusPrimitiveTests(unittest.TestCase):
    def test_en_v1_count_and_exact_hashes(self) -> None:
        text = "Well-known O’Neil paid 1,200.50 — twice."
        self.assertEqual(count_words(text), 5)
        self.assertEqual(source_digest("a"), hashlib.sha256(b"a").hexdigest())
        self.assertEqual(
            segment_id("a" * 64, 2, 4, "x"),
            f"{'a' * 64}:c0002:s0004:{source_digest('x')[:8]}",
        )
        selected_sources = ["a", "b"]
        expected_passage_digest = hashlib.sha256(
            "\n".join(selected_sources).encode("utf-8")
        ).hexdigest()[:12]
        self.assertEqual(expected_passage_digest, "7e18f737311b")
        self.assertEqual(
            passage_id("book", 0, 1, 2, selected_sources),
            f"book:c0000:s0001-0002:{expected_passage_digest}",
        )

    def test_validate_rejects_recursive_runner_leakage_after_hash_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus_dir = Path(directory)
            runner = [{"passage_id": "tampered", "nested": [{"model_id": "secret"}]}]
            challenge_keys: list[dict[str, object]] = []
            corpus = {
                "schema_version": 1,
                "word_counter": "en-v1",
                "run_input_schema_version": 1,
                "passages": [],
                "quotas": {},
            }
            manifest = {"run_input_schema_version": 1, "books": []}
            semantics = {
                "corpus": corpus,
                "runner_segments": runner,
                "challenge_keys": challenge_keys,
            }
            corpus["corpus_sha256"] = sha256_bytes(canonical_json(semantics).encode("utf-8"))
            (corpus_dir / "corpus.json").write_text(canonical_json(corpus) + "\n", encoding="utf-8")
            (corpus_dir / "source_manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8"
            )
            (corpus_dir / "runner_segments.jsonl").write_text(
                canonical_json(runner[0]) + "\n",
                encoding="utf-8",
            )
            (corpus_dir / "challenge_keys.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(CorpusError, r"forbidden key 'model_id'"):
                validate_corpus(corpus_dir)

    def test_validate_rejects_missing_or_reordered_target_indexes(self) -> None:
        for indexes in ([0, 2], [1, 0, 2]):
            with self.subTest(indexes=indexes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_minimal_artifact(root, [_simple_runner(indexes)])
                with self.assertRaisesRegex(CorpusError, "not contiguous"):
                    validate_corpus(root)

    def test_validate_rejects_strict_context_tampering(self) -> None:
        runner = _simple_runner([0, 1, 2])
        runner["subset"] = "context"
        runner["context"] = {
            "challenge_type": "polysemy",
            "source_before": [],
            "source_after": [],
            "frozen_target_before": [],
        }
        runner["passage_id"] = "book:c0000:s0000-0002:tampered"
        challenge = {
            "passage_id": runner["passage_id"],
            "challenge_type": "polysemy",
            "answer_key": " ",
            "rationale": "rationale",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_minimal_artifact(root, [runner], challenge_keys=[challenge])
            with self.assertRaises(CorpusError):
                validate_corpus(root)

    def test_validate_rejects_path_bearing_or_malformed_manifest(self) -> None:
        for updates in (
            {"basename": "/private/book.txt"},
            {"basename": "nested/book.txt"},
            {"source_sha256": "bad"},
            {"license_note": "fixture"},
            {"parser_schema": "1"},
        ):
            with self.subTest(updates=updates), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = {
                    "book_id": "book",
                    "source_sha256": "a" * 64,
                    "basename": "book.txt",
                    "split": "screen",
                    "format": "text",
                    "title": "book",
                    "chapter_count": 1,
                    "parser_schema": 1,
                }
                manifest.update(updates)
                _write_minimal_artifact(root, [], manifest_books=[manifest])
                with self.assertRaises(CorpusError):
                    validate_corpus(root)

    def test_build_validate_round_trip_and_cli_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            books: list[dict[str, str]] = []
            documents: dict[str, Document] = {}
            passages: list[dict] = []
            strata = [
                "narrative",
                "dialogue",
                "literary",
                "long_sentence",
                "idiom_metaphor_wordplay",
                "terminology",
                "numbers_entities",
                "special_format",
            ]
            for index in range(10):
                book_id = f"book{index}"
                source_path = root / f"{book_id}.txt"
                source_path.write_text(f"source fixture {index}", encoding="utf-8")
                split = "screen" if index < 3 else "formal" if index < 9 else "hidden"
                segments: list[Segment] = []
                if split == "screen":
                    segments = [Segment(index=0, source=("screenword " * 3333).strip())]
                    passages.append(
                        {
                            "subset": "screen",
                            "book_id": book_id,
                            "chapter_index": 0,
                            "start_segment_index": 0,
                            "end_segment_index": 0,
                        }
                    )
                elif index < 6:
                    segments.append(Segment(index=0, source=("continuousword " * 10000).strip()))
                    passages.append(
                        {
                            "subset": "continuous",
                            "book_id": book_id,
                            "chapter_index": 0,
                            "start_segment_index": 0,
                            "end_segment_index": 0,
                        }
                    )
                    for segment_index in range(1, 21):
                        segments.append(
                            Segment(index=segment_index, source=("stratifiedword " * 250).strip())
                        )
                        passages.append(
                            {
                                "subset": "stratified",
                                "book_id": book_id,
                                "chapter_index": 0,
                                "start_segment_index": segment_index,
                                "end_segment_index": segment_index,
                                "strata": strata,
                            }
                        )
                elif index < 9:
                    context_count = 7
                    for context_index in range(context_count):
                        before_index = context_index * 2
                        target_index = before_index + 1
                        segments.extend(
                            [
                                Segment(index=before_index, source="beforeword"),
                                Segment(index=target_index, source=("contextword " * 250).strip()),
                            ]
                        )
                        passages.append(
                            {
                                "subset": "context",
                                "book_id": book_id,
                                "chapter_index": 0,
                                "start_segment_index": target_index,
                                "end_segment_index": target_index,
                                "context": {
                                    "challenge_type": "polysemy",
                                    "source_before": [
                                        {"chapter_index": 0, "segment_index": before_index}
                                    ],
                                    "frozen_target_before": [
                                        {
                                            "chapter_index": 0,
                                            "segment_index": before_index,
                                            "target": "前文",
                                        }
                                    ],
                                    "answer_key": "答案",
                                    "rationale": "理由",
                                },
                            }
                        )
                else:
                    segments = [Segment(index=0, source="hiddenword")]
                    passages.append(
                        {
                            "subset": "hidden",
                            "book_id": book_id,
                            "chapter_index": 0,
                            "start_segment_index": 0,
                            "end_segment_index": 0,
                        }
                    )
                documents[str(source_path.resolve())] = Document(
                    title=book_id,
                    source_lang="en",
                    target_lang="zh",
                    fmt="text",
                    source_path=str(source_path.resolve()),
                    chapters=[Chapter(index=0, segments=segments)],
                )
                books.append(
                    {
                        "book_id": book_id,
                        "path": source_path.name,
                        "split": split,
                    }
                )
            spec = root / "BOOK_SPEC.yaml"
            selection = root / "SELECTION.yaml"
            spec.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "source_language": "en",
                        "target_language": "zh",
                        "books": books,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            selection.write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "benchmark_name": "round-trip",
                        "quota_tolerance": 0.2,
                        "passages": passages,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            out = root / "corpus"
            with patch(
                "trans_novel.benchmark.corpus.load_document",
                side_effect=lambda path, *_: documents[path],
            ):
                build(spec, selection, out)
            result = validate_corpus(out)
            self.assertEqual(result["runner_count"], len(passages))
            self.assertEqual(result["split_counts"]["screen"], 3)
            self.assertEqual(result["bucket_counts"]["context"], 21)
            cli_result = CliRunner().invoke(
                app,
                ["tools", "benchmark", "corpus", "validate", str(out)],
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.output)
            self.assertIn("split", cli_result.output)
            self.assertIn("bucket", cli_result.output)
            self.assertIn("book0", cli_result.output)
            self.assertIn(result["corpus_sha256"], cli_result.output)

    def test_scan_is_canonical_and_has_deterministic_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = []
            for index in range(10):
                source = root / f"book{index}.txt"
                source.write_text(
                    f"# Chapter {index}\n\n— Hello {index}.\n\nA Capital Name has 123.\n",
                    encoding="utf-8",
                )
                sources.append(source)
            rows = []
            for index, source in enumerate(sources):
                split = "screen" if index < 3 else "formal" if index < 9 else "hidden"
                rows.append(
                    {
                        "book_id": f"book{index}",
                        "path": "~/book0.txt" if index == 0 else source.name,
                        "split": split,
                    }
                )
            spec = root / "BOOK_SPEC.yaml"
            spec.write_text(
                "schema_version: 1\nsource_language: en\ntarget_language: zh\nbooks:\n"
                + "\n".join(f"  - {json.dumps(row)}" for row in rows)
                + "\n",
                encoding="utf-8",
            )
            out = root / "inventory"
            with patch.dict(os.environ, {"HOME": str(root)}):
                scan(spec, out)
                first = (out / "segments.jsonl").read_bytes()
                scan2 = root / "inventory2"
                scan(spec, scan2)
            self.assertEqual(first, (scan2 / "segments.jsonl").read_bytes())
            records = [json.loads(line) for line in first.splitlines()]
            self.assertIn("dialogue", records[1]["suggestion_tags"])
            self.assertIn("numbers_entities", records[2]["suggestion_tags"])
            self.assertNotIn(str(root), (out / "inventory.json").read_text(encoding="utf-8"))
            with self.assertRaises(CorpusError):
                scan(spec, out)


if __name__ == "__main__":
    unittest.main()
