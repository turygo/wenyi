from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import yaml

from trans_novel.benchmark.corpus import (
    build,
    canonical_json,
    count_words,
    sha256_bytes,
    validate_corpus,
)
from trans_novel.benchmark.pricing import load_price_snapshot, quote_usage
from trans_novel.benchmark.report_cost import (
    CostAnalysisError,
    _aggregate,
    _metrics,
    _million_estimates,
    _serial_aggregate,
    analyze_cost_system,
    reprice_cost_system,
)
from trans_novel.benchmark.report_schema import ReportSpec
from trans_novel.benchmark.runner import (
    FullRunner,
    _preparation_hash,
    _safe_book_id,
    _safe_id,
    load_preparation_bundle,
    validate_preparation,
)
from trans_novel.benchmark.schema import BookPreparation, PreparationBundle, PreparationSpec
from trans_novel.ingest.models import Chapter, Document, Segment
from trans_novel.llm.telemetry import CallAttemptTelemetry
from trans_novel.llm.usage import merge_usage_summaries


class Phase8CostFixtureTests(unittest.TestCase):
    def _write(self, path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, dict | list):
            path.write_text(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(value, encoding="utf-8")

    def _usage(self, prompt: int = 100, completion: int = 10, hit: int = 20):
        totals = {
            "calls": 1,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cache_hit_tokens": hit,
            "cache_miss_tokens": prompt - hit,
        }
        return merge_usage_summaries({}, {"schema_version": 2, "totals": totals})

    def _telemetry(
        self,
        *,
        run_id: str = "run",
        candidate: str = "candidate",
        book: str | None = "book0",
        prompt: int = 100,
        completion: int = 10,
        hit: int = 20,
        status: str = "success",
        logical: str = "call",
    ):
        return {
            "schema_version": 1,
            "logical_call_id": logical,
            "attempt_index": 1,
            "started_at": "2025-01-01T00:00:00Z",
            "elapsed_ms": 100,
            "stage": "translate",
            "agent": "translator",
            "operation": "translate.batch",
            "provider": "fixture",
            "requested_model": "m",
            "resolved_model": "m",
            "reasoning_enabled": False,
            "reasoning_effort": None,
            "temperature": 0.1,
            "seed": None,
            "json_mode": True,
            "max_tokens": 100,
            "status": status,
            "retry_class": None,
            "http_status": 200 if status == "success" else 500,
            "finish_reason": "stop",
            "response_id": None,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cache_hit_tokens": hit,
            "cache_miss_tokens": prompt - hit,
            "reasoning_tokens": 0,
            "billed_usage_unknown": False,
            "request_sha256": "1" * 64,
            "response_sha256": "2" * 64,
            "benchmark_id": "bench",
            "candidate_id": candidate,
            "run_id": run_id,
            "book_id": book,
        }

    def _corpus(self, root: Path) -> tuple[Path, str, dict]:
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
                for context_index in range(7):
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
                    "license_note": "generated fixture",
                }
            )
        spec_path = root / "BOOK_SPEC.yaml"
        selection_path = root / "SELECTION.yaml"
        spec_path.write_text(
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
        selection_path.write_text(
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
        corpus = root / "corpus"
        with patch(
            "trans_novel.benchmark.corpus.load_document",
            side_effect=lambda path, *_: documents[path],
        ):
            build(spec_path, selection_path, corpus)
        facts = validate_corpus(corpus)
        runner_rows = [
            json.loads(line)
            for line in (corpus / "runner_segments.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        screen = next(row for row in runner_rows if row["subset"] == "screen")
        return corpus, facts["corpus_sha256"], screen

    def _price(self, root: Path, rate: str = "1", *, bands: bool = False) -> tuple[Path, str]:
        rules = [
            {
                "min_prompt_tokens": 0,
                "max_prompt_tokens": None,
                "time_band": "all",
                "input_uncached_per_million": rate,
                "input_cached_per_million": "0.5",
                "output_per_million": "2",
            }
        ]
        if bands:
            rules = [
                {
                    "min_prompt_tokens": 0,
                    "max_prompt_tokens": None,
                    "time_band": "day",
                    "input_uncached_per_million": rate,
                    "input_cached_per_million": "0.5",
                    "output_per_million": "2",
                },
                {
                    "min_prompt_tokens": 0,
                    "max_prompt_tokens": None,
                    "time_band": "night",
                    "input_uncached_per_million": rate,
                    "input_cached_per_million": "0.5",
                    "output_per_million": "2",
                },
            ]
        path = root / "price.yaml"
        self._write(
            path,
            {
                "schema_version": 1,
                "provider": "fixture",
                "region": "cn",
                "currency": "CNY",
                "retrieved_at": "2025-01-01T00:00:00Z",
                "source_urls": ["https://prices.example.test/snapshot"],
                "models": {"m": {"model_id": "m", "rules": rules}},
            },
        )
        snapshot = load_price_snapshot(path)
        return path, sha256_bytes(canonical_json(snapshot.model_dump(mode="json")).encode())

    def _preparation(self, root: Path, corpus: str = "0" * 64, usage=None) -> str:
        spec = PreparationSpec(
            schema_version=1,
            provider="fake",
            primary_model="m:off",
            editor_model="m:off",
            fast_model="m:off",
            temperature=0.1,
            seed=None,
        )
        book_usage = usage if usage is not None else merge_usage_summaries({}, {})
        book = BookPreparation(
            book_id="book0",
            source_sha256="1" * 64,
            analysis={},
            style="fixture style",
            style_brief="brief",
            book_synopsis="fixture synopsis",
            chapter_digests={"0": ""},
            telemetry_path="telemetry/book0.jsonl",
            telemetry_sha256=sha256_bytes(b""),
            usage=book_usage,
        )
        bundle = PreparationBundle(
            schema_version=1,
            corpus_sha256=corpus,
            preparation_spec=spec,
            preparation_spec_sha256=sha256_bytes(
                canonical_json(spec.model_dump(mode="python")).encode()
            ),
            preparation_sha256="0" * 64,
            books={"book0": book},
        )
        bundle = bundle.model_copy(update={"preparation_sha256": _preparation_hash(bundle)})
        self._write(root / "preparation.json", bundle.model_dump(mode="python"))
        disk_bundle, _ = load_preparation_bundle(root / "preparation.json")
        disk_book = disk_bundle.books["book0"]
        safe_book = _safe_book_id("book0")
        self._write(root / "telemetry/book0.jsonl", "")
        self._write(root / f"books/{safe_book}.json", disk_book.model_dump(mode="python"))
        book_usage = disk_book.usage
        self._write(root / f"usage/{safe_book}.json", book_usage)
        self._write(root / "usage.json", {"book0": book_usage})
        completion = {
            "schema_version": 1,
            "preparation_sha256": bundle.preparation_sha256,
            "preparation_path": "preparation.json",
            "preparation_file_sha256": sha256_bytes((root / "preparation.json").read_bytes()),
            "usage_path": "usage.json",
            "usage_file_sha256": sha256_bytes((root / "usage.json").read_bytes()),
            "books": {
                "book0": {
                    "export_path": f"books/{safe_book}.json",
                    "export_sha256": sha256_bytes((root / f"books/{safe_book}.json").read_bytes()),
                    "usage_path": f"usage/{safe_book}.json",
                    "usage_sha256": sha256_bytes((root / f"usage/{safe_book}.json").read_bytes()),
                    "telemetry_path": "telemetry/book0.jsonl",
                    "telemetry_sha256": sha256_bytes(b""),
                }
            },
        }
        completion["completion_sha256"] = sha256_bytes(canonical_json(completion).encode())
        self._write(root / "preparation_complete.json", completion)
        freeze = {
            "schema_version": 1,
            "corpus_sha256": corpus,
            "preparation_sha256": bundle.preparation_sha256,
        }
        freeze["immutable_sha256"] = sha256_bytes(canonical_json(freeze).encode())
        self._write(root / "freeze.json", freeze)
        self._write(
            root / "freeze_state.json",
            {
                "status": "completed",
                "completion_sha256": completion["completion_sha256"],
                "books": {"book0": {"status": "completed"}},
            },
        )
        validate_preparation(root)
        return bundle.preparation_sha256

    def _spec(
        self, prep_hash: str, run_hash: str, price_hash: str, corpus: str = "0" * 64
    ) -> ReportSpec:
        return ReportSpec(
            schema_version=1,
            benchmark_id="bench",
            corpus_sha256=corpus,
            run_hash=run_hash,
            preparation_sha256=prep_hash,
            pack_sha256="3" * 64,
            evaluation_sha256="4" * 64,
            price_snapshot_sha256=price_hash,
            bootstrap_seed=7,
            bootstrap_replicates=1000,
            editor_hourly_rates=[Decimal("50")],
        )

    def _attribution(
        self, root: Path, prep_hash: str, corpus_hash: str, screen: dict, *, unknown: bool = False
    ) -> tuple[Path, str]:
        run = root / "run"
        artifact = run / "translation" / "t"
        usage = self._usage()
        passage_id_value = screen["passage_id"]
        screen_segment = screen["segments"][0]
        segment_id_value = screen_segment["segment_id"]
        source_value = screen_segment["source"]
        book_id = screen["book_id"]
        safe_passage = _safe_id(passage_id_value)
        (artifact / "passages").mkdir(parents=True, exist_ok=True)
        self._write(
            artifact / "manifest.json",
            {
                "schema_version": 1,
                "kind": "translation",
                "artifact_key": "t",
                "corpus_sha256": corpus_hash,
                "preparation_sha256": prep_hash,
                "scope": {
                    "subset": "screen",
                    "context_strategy": "c2",
                    "passage_ids": [passage_id_value],
                },
            },
        )
        self._write(artifact / "usage.json", usage)
        record = self._telemetry(status="error" if unknown else "success", book=book_id)
        if unknown:
            record["billed_usage_unknown"] = True
        final_value = "一二"
        segment = {
            "segment_id": segment_id_value,
            "source": source_value,
            "translation_raw": final_value,
            "translation_after_lint": final_value,
            "translation_lint_issues": [{"type": "alignment"}],
            "polish_proposal": None,
            "polish_accepted": None,
            "polish_rejection_reasons": [],
            "final": final_value,
            "translation_raw_sha256": sha256_bytes(final_value.encode()),
            "translation_after_lint_sha256": sha256_bytes(final_value.encode()),
            "final_sha256": sha256_bytes(final_value.encode()),
            "output_sha256": sha256_bytes(final_value.encode()),
        }
        source_hash = sha256_bytes(
            canonical_json([{"segment_id": segment_id_value, "source": source_value}]).encode()
        )
        self._write(artifact / "telemetry.jsonl", json.dumps(record, sort_keys=True) + "\n")
        self._write(
            artifact / f"passages/{safe_passage}.json",
            {
                "status": "complete",
                "artifact_key": "t",
                "passage_id": passage_id_value,
                "book_id": book_id,
                "source_hash": source_hash,
                "segments": [segment],
            },
        )
        row = {
            "candidate_id": "candidate",
            "replicate": 1,
            "translation_artifacts": {"screen:c2": "translation/t"},
            "editor_artifacts": {},
        }
        self._write(run / "candidates.json", [row])
        self._write(run / "actual_usage.json", usage)
        immutable = {
            "schema_version": 1,
            "run_mode": "attribution",
            "prompt_version": "benchmark-context-v1",
            "benchmark_id": "bench",
            "spec_sha256": "3" * 64,
            "corpus_sha256": corpus_hash,
            "preparation_sha256": prep_hash,
            "canary_sample_id": None,
        }
        self._write(run / "run.json", immutable)
        self._write(run / "run_state.json", {"status": "completed"})
        run_hash = sha256_bytes(canonical_json(immutable).encode())
        return run, run_hash

    def test_attribution_analysis_and_reprice_after_source_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_hash = self._preparation(prep, corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            result = analyze_cost_system(
                corpus,
                run,
                prep,
                price,
                {
                    "input_hashes": {},
                    "postedit": {"candidate": {"surface": {"macro": {"minutes_per_10k": "10"}}}},
                },
                spec,
            )
            self.assertEqual(
                result["candidate_costs"]["candidate"]["validated_source_words"],
                count_words(screen["segments"][0]["source"]),
            )
            expected_quote = quote_usage(
                load_price_snapshot(price),
                "m",
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                    "cache_hit_tokens": 20,
                    "cache_miss_tokens": 80,
                    "reasoning_tokens": 0,
                },
            )
            self.assertEqual(
                result["candidate_costs"]["candidate"]["api_cost"],
                format(expected_quote.total_cost, "f"),
            )
            self.assertEqual(result["system_metrics"]["candidate"]["lint_findings"]["alignment"], 1)
            self.assertEqual(result["system_metrics"]["candidate"]["lint_rejected"], 0)
            self.assertIn(
                "translation:t", result["normalized_pricing_facts"]["physical_artifact_ids"]
            )
            self.assertIn(
                {"scope": "candidate=candidate", "reason": "missing_system_evidence"},
                result["insufficient_data"],
            )
            new_price, _ = self._price(root, "2")
            repriced = reprice_cost_system(
                result["normalized_pricing_facts"],
                new_price,
                {},
                [Decimal("50")],
                bootstrap_seed=7,
                bootstrap_replicates=1000,
            )
            moved = root / "moved"
            moved.mkdir()
            prep.rename(moved / "prep")
            run.rename(moved / "run")
            self.assertEqual(repriced["system_metrics"], result["system_metrics"])

    def test_base_runner_telemetry_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_hash = self._preparation(prep, corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            telemetry = json.loads(
                (run / "translation/t/telemetry.jsonl").read_text().splitlines()[0]
            )
            for key in ("benchmark_id", "candidate_id", "run_id", "book_id"):
                telemetry.pop(key)
            self._write(
                run / "translation/t/telemetry.jsonl", json.dumps(telemetry, sort_keys=True) + "\n"
            )
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            result = analyze_cost_system(corpus, run, prep, price, {}, spec)
            self.assertEqual(result["system_metrics"]["candidate"]["attempts"], 1)

    def test_logical_and_resume_duplicate_scope_is_physical(self):
        def attempt(logical):
            value = self._telemetry(logical=logical)
            return {
                "telemetry": CallAttemptTelemetry.model_validate(
                    {
                        key: value[key]
                        for key in value
                        if key not in {"benchmark_id", "candidate_id", "run_id", "book_id"}
                    }
                )
            }

        first = attempt("one")
        second = attempt("two")
        artifact_a = {"artifact_id": "translation:a", "attempts": [first, second]}
        artifact_b = {"artifact_id": "translation:b", "attempts": [attempt("one")]}
        metrics = _metrics([artifact_a, artifact_b], {})
        self.assertEqual(metrics["logical_calls"], 3)
        self.assertEqual(metrics["resume_duplicate_operations"], 1)

    def test_lineage_tamper_and_unknown_million_withhold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_hash = self._preparation(prep, corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen, unknown=True)
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            result = analyze_cost_system(corpus, run, prep, price, {}, spec)
            passage_path = run / f"translation/t/passages/{_safe_id(screen['passage_id'])}.json"
            original_passage = json.loads(passage_path.read_text())
            deleted_segment = {
                **original_passage,
                "segments": [],
                "source_hash": sha256_bytes(canonical_json([]).encode()),
            }
            self._write(passage_path, deleted_segment)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, prep, price, {}, spec)
            self._write(passage_path, original_passage)
            tampered_run = json.loads((run / "run.json").read_text())
            tampered_run["run_hash"] = run_hash
            self._write(run / "run.json", tampered_run)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, prep, price, {}, spec)
            self._write(
                run / "run.json",
                {key: value for key, value in tampered_run.items() if key != "run_hash"},
            )
            result = analyze_cost_system(corpus, run, prep, price, {}, spec)
            estimate = result["million_word_estimate"]["candidate"]
            self.assertFalse(estimate["complete"])
            self.assertIsNone(estimate["value"])
            (run / f"translation/t/passages/{_safe_id(screen['passage_id'])}.json").unlink()
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, prep, price, {}, spec)

    def test_full_branch_increment_and_unknown_cost_lower_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_usage = self._usage(20, 2, 0)
            prep_hash = self._preparation(prep, corpus_hash, usage=prep_usage)
            price, price_hash = self._price(root)
            run = root / "run"
            full_source = screen["segments"][0]["source"]
            full_segment_id = screen["segments"][0]["segment_id"]
            raw = self._usage(100, 10, 20)
            increment = self._usage(50, 5, 0)
            branch_total = merge_usage_summaries(raw, increment)
            for kind, artifact_id, usage in (
                ("raw", "raw-id", raw),
                ("branch", "branch-id", branch_total),
            ):
                base = run / ("raw" if kind == "raw" else "branches") / artifact_id
                self._write(
                    base / "manifest.json",
                    {
                        "schema_version": 1,
                        "kind": kind,
                        "artifact_key": artifact_id,
                        "corpus_sha256": corpus_hash,
                        "preparation_sha256": prep_hash,
                    },
                )
                self._write(base / "usage.json", usage)
                self._write(base / "book0/usage.json", usage)
                self._write(
                    base / "telemetry.jsonl",
                    json.dumps(
                        self._telemetry(
                            prompt=100 if kind == "raw" else 50,
                            completion=10 if kind == "raw" else 5,
                            hit=20 if kind == "raw" else 0,
                        ),
                        sort_keys=True,
                    )
                    + "\n",
                )
                store = base / "book0"
                target = "raw" if kind == "raw" else "一二"
                self._write(
                    store / "manifest.json",
                    {
                        "run_state_schema": 2,
                        "chapters": [{"index": 0}],
                        "progress": {"0": {"status": "done"}},
                        "nodes": {},
                    },
                )
                chapter = Chapter(
                    index=0,
                    segments=[
                        Segment(
                            index=0,
                            source=full_source,
                            target=target,
                            kind="text",
                            cont=False,
                            anchor=None,
                            resource_href=None,
                            meta={"original_segment_id": full_segment_id},
                        )
                    ],
                )
                self._write(store / "chapters_v2/ch0.json", chapter.model_dump(mode="python"))
            stage = {
                "segment_id": full_segment_id,
                "source": full_source,
                "raw_after_translation_lint": "raw",
                "final_after_full_pipeline": "一二",
                "final_sha256": sha256_bytes("一二".encode()),
                "review_findings": [],
                "lint_findings": [],
                "backtranslation_findings": [],
            }
            allocation = FullRunner._preparation_allocation_id("candidate", "book0", prep_hash)
            row = {
                "candidate_id": "candidate",
                "book_id": "book0",
                "replicate": 1,
                "raw_artifact_id": "raw-id",
                "branch_artifact_id": "branch-id",
                "preparation_allocation_id": allocation,
                "raw_telemetry_sha256": sha256_bytes(
                    (run / "raw/raw-id/telemetry.jsonl").read_bytes()
                ),
                "branch_telemetry_sha256": sha256_bytes(
                    (run / "branches/branch-id/telemetry.jsonl").read_bytes()
                ),
                "allocated_usage": {
                    "preparation": prep_usage,
                    "raw": raw,
                    "branch_increment": increment,
                },
                "stage": [stage],
            }
            row_two = {
                **row,
                "candidate_id": "candidate-two",
                "preparation_allocation_id": FullRunner._preparation_allocation_id(
                    "candidate-two", "book0", prep_hash
                ),
            }
            self._write(run / "candidates.json", [row, row_two])
            actual = {}
            for value in (prep_usage, prep_usage, raw, increment):
                actual = merge_usage_summaries(actual, value)
            self._write(run / "actual_usage.json", actual)
            immutable = {
                "schema_version": 1,
                "run_mode": "full",
                "benchmark_id": "bench",
                "spec_sha256": "3" * 64,
                "corpus_sha256": corpus_hash,
                "preparation_sha256": prep_hash,
                "replicates": 1,
            }
            self._write(run / "run.json", immutable)
            self._write(run / "run_state.json", {"status": "completed"})
            spec = self._spec(
                prep_hash, sha256_bytes(canonical_json(immutable).encode()), price_hash, corpus_hash
            )
            result = analyze_cost_system(
                corpus, run, prep, price, {"input_hashes": {}, "postedit": {}}, spec
            )
            self.assertIn(
                "branch:branch-id",
                result["normalized_pricing_facts"]["candidate_artifact_ids"]["candidate"],
            )
            self.assertIn("candidate-two", result["candidate_costs"])
            tampered = json.loads((run / "candidates.json").read_text())
            tampered[0]["stage"][0]["source"] = "tampered"
            self._write(run / "candidates.json", tampered)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(
                    corpus, run, prep, price, {"input_hashes": {}, "postedit": {}}, spec
                )

    def test_system_findings_use_explicit_evidence_and_missing_protocol_is_insufficient(self):
        value = {
            "book_id": "book",
            "source_words": 2,
            "completed": True,
            "lint_findings": [],
            "review_findings": [{"type": "review", "fixed": False}],
            "backtranslation_findings": [{"type": "back"}],
            "consistency_findings": [{"type": "consistency"}],
            "polish_accepted": None,
            "polish_rejection_reasons": [],
            "required_node_failures": [],
        }
        metrics = _metrics([], {("book", "segment"): value})
        self.assertEqual(metrics["review_unresolved"], 1)
        self.assertEqual(metrics["backtranslation_unresolved"], 1)
        self.assertEqual(metrics["consistency_unresolved"], 1)
        self.assertIsNone(metrics["protocol_errors"])
        self.assertIsNone(metrics["json_errors"])
        self.assertIsNone(metrics["alignment_errors"])

    def test_attribution_hash_and_polish_tamper_are_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep_hash = self._preparation(root / "prep", corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            passage = run / f"translation/t/passages/{_safe_id(screen['passage_id'])}.json"
            value = json.loads(passage.read_text())
            value["segments"][0]["final_sha256"] = "0" * 64
            self._write(passage, value)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, root / "prep", price, {}, spec)
            value["segments"][0]["final_sha256"] = sha256_bytes(
                value["segments"][0]["final"].encode()
            )
            value["segments"][0]["polish_proposal"] = value["segments"][0]["final"]
            value["segments"][0]["polish_accepted"] = False
            value["segments"][0]["polish_rejection_reasons"] = ["tampered"]
            self._write(passage, value)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, root / "prep", price, {}, spec)

    def test_editor_artifact_baseline_and_polish_gate_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep_hash = self._preparation(root / "prep", corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            source_path = run / f"translation/t/passages/{_safe_id(screen['passage_id'])}.json"
            source = json.loads(source_path.read_text())
            editor_dir = run / "candidates/_shared/e"
            editor_segment = dict(source["segments"][0])
            editor_segment.update(
                {
                    "polish_proposal": editor_segment["translation_raw"],
                    "polish_accepted": True,
                    "polish_rejection_reasons": [],
                    "final": editor_segment["translation_raw"],
                }
            )
            editor_passage = {
                **source,
                "kind": "editor",
                "artifact_key": "e",
                "translation_artifact_key": "t",
                "segments": [editor_segment],
            }
            self._write(
                editor_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "kind": "editor",
                    "artifact_key": "e",
                    "translation_artifact_key": "t",
                    "corpus_sha256": corpus_hash,
                    "preparation_sha256": prep_hash,
                },
            )
            self._write(editor_dir / "usage.json", self._usage())
            self._write(
                editor_dir / "telemetry.jsonl", (run / "translation/t/telemetry.jsonl").read_text()
            )
            self._write(
                editor_dir / f"passages/{_safe_id(screen['passage_id'])}.json", editor_passage
            )
            row = json.loads((run / "candidates.json").read_text())[0]
            row["editor_artifacts"] = {"screen:c2": "candidates/_shared/e"}
            self._write(run / "candidates.json", [row])
            self._write(
                run / "actual_usage.json", merge_usage_summaries(self._usage(), self._usage())
            )
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            result = analyze_cost_system(corpus, run, root / "prep", price, {}, spec)
            self.assertIn("editor:e", result["normalized_pricing_facts"]["physical_artifact_ids"])
            self.assertEqual(result["system_metrics"]["candidate"]["lint_findings"]["alignment"], 1)

    def test_metrics_deduplicate_backtranslation_and_separate_polish_reasons(self):
        finding = {"type": "meaning", "detail": "same"}
        shared = {
            "book_id": "book",
            "source_words": 1,
            "completed": True,
            "backtranslation_findings": [finding],
            "lint_findings": [],
            "review_findings": [],
            "consistency_findings": [],
            "polish_accepted": None,
            "polish_rejection_reasons": [],
        }
        artifacts = [
            {
                "artifact_id": "branch:one",
                "segments": [
                    {
                        "book_id": "book",
                        "segment_id": "a",
                        "backtranslation_findings": [finding],
                        "review_findings": [{"type": "review", "fixed": False}],
                    },
                    {
                        "book_id": "book",
                        "segment_id": "b",
                        "backtranslation_findings": [finding],
                        "review_findings": [{"type": "review", "fixed": False}],
                    },
                ],
            },
            {
                "artifact_id": "branch:two",
                "segments": [
                    {
                        "book_id": "book",
                        "segment_id": "a",
                        "backtranslation_findings": [],
                        "review_findings": [],
                    },
                ],
            },
        ]
        first = {("book", "a"): shared, ("book", "b"): {**shared}}
        metrics = _metrics(artifacts, first)
        self.assertEqual(metrics["backtranslation_findings"]["meaning"], 1)
        self.assertEqual(metrics["review_findings"]["review"], 1)
        self.assertEqual(metrics["review_unresolved"], 1)
        a_metric = _metrics(
            [
                {
                    "artifact_id": "branch:multi",
                    "segments": [
                        {
                            "book_id": "book-a",
                            "segment_id": "a",
                            "backtranslation_findings": [{"type": "bt"}],
                        }
                    ],
                }
            ],
            {
                ("book-a", "a"): {
                    "book_id": "book-a",
                    "source_words": 1,
                    "completed": True,
                    "backtranslation_findings": [{"type": "bt"}],
                }
            },
        )
        b_metric = _metrics(
            [
                {
                    "artifact_id": "branch:multi",
                    "segments": [
                        {"book_id": "book-b", "segment_id": "b", "backtranslation_findings": []}
                    ],
                }
            ],
            {
                ("book-b", "b"): {
                    "book_id": "book-b",
                    "source_words": 1,
                    "completed": True,
                    "backtranslation_findings": [],
                }
            },
        )
        self.assertEqual(a_metric["backtranslation_findings"]["bt"], 1)
        self.assertEqual(b_metric["backtranslation_findings"], {})
        rejected = {
            **shared,
            "backtranslation_findings": [],
            "lint_findings": [{"type": "x", "fixed": False}],
            "polish_accepted": False,
            "polish_rejection_reasons": ["x"],
        }
        accepted = {
            **shared,
            "backtranslation_findings": [],
            "lint_findings": [],
            "polish_accepted": True,
            "polish_rejection_reasons": [],
        }
        fixes = _metrics([], {("book", "a"): rejected, ("book", "b"): accepted})
        self.assertEqual(fixes["lint_fixes"], 1)
        self.assertEqual(fixes["polish_rejection_reasons"], {"x": 1})

    def test_reprice_rejects_dangling_normalized_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_hash = self._preparation(prep, corpus_hash)
            price, price_hash = self._price(root)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            result = analyze_cost_system(corpus, run, prep, price, {}, spec)
            facts = json.loads(json.dumps(result["normalized_pricing_facts"]))
            facts["candidate_artifact_ids"]["candidate"].append("translation:missing")
            with self.assertRaises(CostAnalysisError):
                reprice_cost_system(
                    facts, price, {}, [Decimal("50")], bootstrap_seed=7, bootstrap_replicates=1000
                )
            tampered_words = json.loads(json.dumps(result["normalized_pricing_facts"]))
            tampered_words["artifacts"][0]["source_words"] = 1
            with self.assertRaises(CostAnalysisError):
                reprice_cost_system(
                    tampered_words,
                    price,
                    {},
                    [Decimal("50")],
                    bootstrap_seed=7,
                    bootstrap_replicates=1000,
                )

    def test_multibook_enriched_split_and_base_withholds_book_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            price, _ = self._price(root)
            snapshot = load_price_snapshot(price)

            def parsed(value):
                return CallAttemptTelemetry.model_validate(
                    {
                        key: value[key]
                        for key in value
                        if key not in {"benchmark_id", "candidate_id", "run_id", "book_id"}
                    }
                )

            enriched = {
                "artifact_id": "translation:multi",
                "book_id": None,
                "source_words": 5,
                "source_words_by_book": {"book-a": 2, "book-b": 3},
                "attempts": [
                    {
                        "telemetry": parsed(self._telemetry(book="book-a", logical="a")),
                        "book_id": "book-a",
                    },
                    {
                        "telemetry": parsed(self._telemetry(book="book-b", logical="b")),
                        "book_id": "book-b",
                    },
                ],
            }
            split = _serial_aggregate(_aggregate([enriched], snapshot), words=5)
            self.assertTrue(split["book_attribution_complete"])
            self.assertEqual(
                sum(Decimal(item["api_cost_lower_bound"]) for item in split["by_book"].values()),
                Decimal(split["api_cost_lower_bound"]),
            )
            base = {
                "artifact_id": "translation:multi-base",
                "book_id": None,
                "source_words": 5,
                "source_words_by_book": {"book-a": 2, "book-b": 3},
                "attempts": [{"telemetry": parsed(self._telemetry(book=None)), "book_id": None}],
            }
            bad = {
                **base,
                "artifact_id": "translation:multi-bad",
                "attempts": [
                    {
                        "telemetry": parsed(self._telemetry(book="book-a", logical="ba")),
                        "book_id": "book-a",
                    },
                    {
                        "telemetry": parsed(self._telemetry(book="book-b", logical="bb")),
                        "book_id": "book-b",
                    },
                    {
                        "telemetry": parsed(self._telemetry(book=None, logical="none")),
                        "book_id": None,
                    },
                    {
                        "telemetry": parsed(self._telemetry(book="bogus", logical="bogus")),
                        "book_id": "bogus",
                    },
                ],
            }
            bad_serial = _serial_aggregate(_aggregate([bad], snapshot), words=5)
            self.assertFalse(bad_serial["book_attribution_complete"])
            self.assertNotIn("", bad_serial["by_book"])
            bad_million = _million_estimates(
                {"candidate": ["translation:multi-bad"]},
                {"translation:multi-bad": bad},
                {"candidate": {"book-a": 2, "book-b": 3}},
                snapshot,
                seed=7,
                replicates=10,
            )
            self.assertFalse(bad_million["candidate"]["complete"])
            self.assertIsNone(bad_million["candidate"]["value"])
            base_serial = _serial_aggregate(_aggregate([base], snapshot), words=5)
            self.assertFalse(base_serial["book_attribution_complete"])
            million = _million_estimates(
                {"candidate": ["translation:multi-base"]},
                {"translation:multi-base": base},
                {"candidate": {"book-a": 2, "book-b": 3}},
                snapshot,
                seed=7,
                replicates=10,
            )
            self.assertFalse(million["candidate"]["complete"])
            self.assertIsNone(million["candidate"]["value"])
            self.assertTrue(
                all(
                    item["estimate_lower_bound"] is None
                    for item in million["candidate"]["by_book"].values()
                )
            )

    def test_price_time_band_ambiguity_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, corpus_hash, screen = self._corpus(root)
            prep = root / "prep"
            prep_hash = self._preparation(prep, corpus_hash)
            price, price_hash = self._price(root, bands=True)
            run, run_hash = self._attribution(root, prep_hash, corpus_hash, screen)
            spec = self._spec(prep_hash, run_hash, price_hash, corpus_hash)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(corpus, run, prep, price, {}, spec)
            tampered = json.loads((run / "actual_usage.json").read_text())
            tampered["totals"]["prompt_tokens"] = 999
            self._write(run / "actual_usage.json", tampered)
            price, price_hash = self._price(root)
            with self.assertRaises(CostAnalysisError):
                analyze_cost_system(
                    corpus,
                    run,
                    prep,
                    price,
                    {},
                    self._spec(prep_hash, run_hash, price_hash, corpus_hash),
                )


if __name__ == "__main__":
    unittest.main()
