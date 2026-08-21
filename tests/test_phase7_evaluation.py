from __future__ import annotations

import json
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from trans_novel.benchmark.corpus import canonical_json, sha256_bytes
from trans_novel.benchmark.evaluation import (
    EvaluationError,
    _leak,
    _pair_adjudication,
    _seed,
    build_units,
    generate_pack,
    import_responses,
    validate_pack,
)
from trans_novel.benchmark.runner import PROMPT_VERSION
from trans_novel.benchmark.schema import EvaluationSpec


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _empty_usage() -> dict:
    zeros = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_hit_tokens": 0,
        "cache_miss_tokens": 0,
        "cache_hit_rate": 0.0,
    }
    return {
        "schema_version": 2,
        "totals": dict(zeros),
        "by_agent": {},
        "by_operation": {},
        "by_provider": {},
        "by_model": {},
        "by_stage": {},
    }


def _synthetic_inputs(root: Path, *, replicas: bool = False) -> tuple[Path, Path, dict]:
    """Create a compact but complete phase-five/six-shaped evaluation fixture."""
    root.mkdir(parents=True, exist_ok=True)
    corpus, run = root / "corpus", root / "run"
    corpus.mkdir()
    run.mkdir()
    rows = []
    source_tokens = (
        "sourcezero",
        "sourceone",
        "sourcetwo",
        "sourcethree",
        "sourcefour",
        "sourcefive",
        "sourcesix",
        "sourceseven",
    )
    for index in range(8):
        subset = ("context", "continuous", "screen", "stratified")[index % 4]
        passage_id = f"p{index}"
        rows.append(
            {
                "passage_id": passage_id,
                "book_id": f"book{index % 2}",
                "subset": subset,
                "chapter_index": index,
                "strata": ["narrative"],
                "segments": [
                    {
                        "segment_id": f"s{index}",
                        "source": f"{source_tokens[index]} " * 180,
                        "word_count": 180,
                    }
                ],
            }
        )
    (corpus / "runner_segments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (corpus / "challenge_keys.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "passage_id": row["passage_id"],
                    "answer_key": "answer",
                    "rationale": "rationale",
                }
            )
            + "\n"
            for row in rows
            if row["subset"] == "context"
        ),
        encoding="utf-8",
    )
    corpus_sha = "b" * 64
    preparation_sha = "d" * 64
    generation = {
        "temperature": 0.1,
        "seed": None,
        "require_catalogued_model": True,
        "require_thinking_disabled": True,
    }
    empty_context = {
        "source_before": "",
        "target_before": "",
        "source_after": "",
    }
    context_hash = sha256_bytes(
        canonical_json(
            {
                "preparation": {
                    "style": "",
                    "book_synopsis": "",
                    "chapter_digest": "",
                    "glossary": "",
                },
                "batches": [empty_context],
            }
        ).encode()
    )
    batch_context_hash = sha256_bytes(canonical_json(empty_context).encode())
    candidates = []
    for candidate in ("c1", "c2", "c3"):
        for replicate in (1, 2) if replicas else (1,):
            primary_model = f"primary-{candidate}:off"
            editor_model = "editor-model:off" if candidate == "c2" else None
            translations, editors, stage = {}, {}, []
            for row in rows:
                pid, sid = row["passage_id"], row["segments"][0]["segment_id"]
                strategies = ("c0", "c1", "c2") if row["subset"] == "context" else ("c2",)
                for strategy in strategies:
                    scope = f"{row['subset']}:{strategy}"
                    key_material = {
                        "schema_version": 1,
                        "prompt_version": PROMPT_VERSION,
                        "provider": "fake",
                        "primary_model": primary_model,
                        "generation": generation,
                        "corpus_sha256": corpus_sha,
                        "preparation_sha256": preparation_sha,
                        "replicate": replicate,
                        "scope": {
                            "subset": row["subset"],
                            "context_strategy": strategy,
                            "passage_ids": [
                                item["passage_id"]
                                for item in rows
                                if item["subset"] == row["subset"]
                            ],
                        },
                    }
                    translation_key = sha256_bytes(canonical_json(key_material).encode())
                    reference = f"translation/{translation_key}"
                    translations[scope] = reference
                    artifact = run / reference
                    artifact.mkdir(parents=True, exist_ok=True)
                    _write_json(
                        artifact / "manifest.json",
                        {
                            "schema_version": 1,
                            "artifact_key": translation_key,
                            "kind": "translation",
                            "prompt_version": PROMPT_VERSION,
                            "provider": "fake",
                            "model": primary_model,
                            "corpus_sha256": corpus_sha,
                            "preparation_sha256": preparation_sha,
                            "replicate": replicate,
                            "scope": key_material["scope"],
                        },
                    )
                    _write_json(artifact / "usage.json", _empty_usage())
                    source = row["segments"][0]["source"]
                    source_hash = sha256_bytes(
                        canonical_json(
                            [
                                {
                                    "segment_id": sid,
                                    "source": source,
                                }
                            ]
                        ).encode()
                    )
                    translation = f"translation-{candidate}-{replicate}-{pid}"
                    segment = {
                        "segment_id": sid,
                        "source": source,
                        "translation_raw": translation,
                        "translation_after_lint": translation,
                        "translation_lint_issues": [],
                        "polish_proposal": None,
                        "polish_accepted": None,
                        "polish_rejection_reasons": [],
                        "final": translation,
                    }
                    segment["translation_raw_sha256"] = sha256_bytes(
                        segment["translation_raw"].encode()
                    )
                    segment["translation_after_lint_sha256"] = sha256_bytes(
                        segment["translation_after_lint"].encode()
                    )
                    segment["final_sha256"] = sha256_bytes(segment["final"].encode())
                    segment["output_sha256"] = segment["final_sha256"]
                    passage = {
                        "status": "complete",
                        "kind": "translation",
                        "artifact_key": translation_key,
                        "passage_id": pid,
                        "subset": row["subset"],
                        "book_id": row["book_id"],
                        "chapter_index": row["chapter_index"],
                        "replicate": replicate,
                        "context_strategy": strategy,
                        "provider": "fake",
                        "primary_model": primary_model,
                        "source_hash": source_hash,
                        "context_hash": context_hash,
                        "preparation_sha256": preparation_sha,
                        "batch_context_hashes": [batch_context_hash],
                        "segments": [segment],
                        "usage_delta": _empty_usage(),
                    }
                    _write_json(artifact / "passages" / f"{pid}.json", passage)
                    if editor_model is not None and strategy == "c2":
                        editor_key = sha256_bytes(
                            canonical_json(
                                {
                                    "schema_version": 1,
                                    "editor_model": editor_model,
                                    "translation_artifact_key": translation_key,
                                    "generation": generation,
                                }
                            ).encode()
                        )
                        editor_reference = f"candidates/_shared/{editor_key}"
                        editors[scope] = editor_reference
                        editor_artifact = run / editor_reference
                        editor_artifact.mkdir(parents=True, exist_ok=True)
                        _write_json(
                            editor_artifact / "manifest.json",
                            {
                                "schema_version": 1,
                                "kind": "editor",
                                "artifact_key": editor_key,
                                "translation_artifact_key": translation_key,
                                "provider": "fake",
                                "editor_model": editor_model,
                                "corpus_sha256": corpus_sha,
                                "preparation_sha256": preparation_sha,
                            },
                        )
                        _write_json(editor_artifact / "usage.json", _empty_usage())
                        edited = dict(segment)
                        edited["polish_proposal"] = f"proposal-{candidate}-{replicate}-{pid}"
                        edited["polish_accepted"] = True
                        edited["polish_rejection_reasons"] = []
                        edited["final"] = f"edited-{candidate}-{replicate}-{pid}"
                        edited["final_sha256"] = sha256_bytes(edited["final"].encode())
                        edited["output_sha256"] = edited["final_sha256"]
                        edited_passage = dict(passage)
                        edited_passage.update(
                            {
                                "kind": "editor",
                                "artifact_key": editor_key,
                                "translation_artifact_key": translation_key,
                                "editor_model": editor_model,
                                "segments": [edited],
                                "usage_delta": _empty_usage(),
                            }
                        )
                        _write_json(
                            editor_artifact / "passages" / f"{pid}.json",
                            edited_passage,
                        )
                stage.append(
                    {
                        "passage_id": pid,
                        "segment_id": sid,
                        "source": source,
                        "raw_after_translation_lint": (
                            f"translation-{candidate}-{replicate}-{pid}"
                        ),
                        "final_after_full_pipeline": (f"full-{candidate}-{replicate}-{pid}"),
                        "review_findings": [],
                        "lint_findings": [],
                        "backtranslation_findings": [],
                        "raw_artifact_id": f"raw-{candidate}-{replicate}",
                        "branch_artifact_id": f"branch-{candidate}-{replicate}",
                        "preparation_sha256": preparation_sha,
                        "primary_model": primary_model,
                        "editor_model": editor_model,
                        "fast_model": "fast-model:off",
                        "final_sha256": sha256_bytes(
                            f"full-{candidate}-{replicate}-{pid}".encode()
                        ),
                        "polish_proposal": (
                            f"proposal-{candidate}-{replicate}-{pid}"
                            if editor_model is not None
                            else None
                        ),
                    }
                )
            candidates.append(
                {
                    "candidate_id": candidate,
                    "replicate": replicate,
                    "provider": "fake",
                    "primary_model": primary_model,
                    "editor_model": editor_model,
                    "raw_artifact_id": f"raw-{candidate}-{replicate}",
                    "branch_artifact_id": f"branch-{candidate}-{replicate}",
                    "translation_artifacts": translations,
                    "editor_artifacts": editors,
                    "editor_artifact_id": next(iter(editors.values()), None),
                    "allocated_usage": _empty_usage(),
                    "stage": stage,
                }
            )
    _write_json(run / "candidates.json", candidates)
    run_manifest = {
        "schema_version": 1,
        "run_mode": "attribution",
        "prompt_version": PROMPT_VERSION,
        "benchmark_id": "synthetic",
        "spec_sha256": "e" * 64,
        "corpus_sha256": corpus_sha,
        "preparation_sha256": preparation_sha,
        "canary_sample_id": None,
    }
    run_hash = sha256_bytes(canonical_json(run_manifest).encode())
    _write_json(run / "run.json", run_manifest)
    _write_json(run / "run_state.json", {"status": "completed"})
    evaluation_spec = {
        "schema_version": 1,
        "benchmark_id": "synthetic",
        "run_hash": run_hash,
        "corpus_sha256": corpus_sha,
        "seed": 7,
        "raters": ["r1", "r2", "r3"],
        "candidate_ids": ["c1", "c2", "c3"],
        "hidden_duplicate_fraction": 0.1,
        "calibration_units": 30,
        "absolute": {"target_source_words": 1, "ratings_per_output": 3},
        "pairwise": {"target_source_words": 720, "ratings_per_comparison": 2},
        "polish": {"target_source_words": 1, "ratings_per_pair": 3},
        "mqm": {"target_source_words": 1, "annotators_per_output": 2},
        "postedit": {"target_source_words": 1, "editors_per_output": 1},
        "context": {"target_source_words": 1, "ratings_per_output": 3},
        "enabled_surfaces": ["attribution_final", "full_final"],
        "study_protocol": {
            "eligibility_text": "eligible < adults",
            "consent_text": "consent < required",
            "compensation_text": "compensation",
            "retention_text": "retention",
        },
    }
    return corpus, run, evaluation_spec


def _make_pack(root: Path, *, replicas: bool = False) -> tuple[Path, dict]:
    corpus, run, evaluation_spec = _synthetic_inputs(root, replicas=replicas)
    pack = root / "pack"
    with patch(
        "trans_novel.benchmark.evaluation.validate_corpus",
        return_value={"corpus_sha256": evaluation_spec["corpus_sha256"]},
    ):
        generate_pack(corpus, run, evaluation_spec, pack)
    return pack, evaluation_spec


def _response_row(item: dict, rater: str, pack_hash: str) -> dict:
    row = {
        "schema_version": 1,
        "assignment_id": item["assignment_id"],
        "rater_id": rater,
        "pack_hash": pack_hash,
        "started_at": "2026-01-01T00:00:00Z",
        "submitted_at": "2026-01-01T00:00:01Z",
        "active_ms": 500,
        "kind": item["kind"],
    }
    if item["kind"] == "absolute":
        row.update(dict.fromkeys(item["dimensions"], 3))
    elif item["kind"] == "pairwise":
        row["preference"] = "a_much_better"
    elif item["kind"] == "polish":
        row["outcome"] = "clearly_improved"
    elif item["kind"] == "mqm":
        row["errors"] = []
    elif item["kind"] == "context":
        row["judgment"] = "correct"
    else:
        row["edited_target"] = "edited answer"
    return row


def _write_responses(pack: Path, root: Path) -> None:
    manifest = json.loads((pack / "pack.json").read_text())
    pack_hash = manifest["pack_semantic_sha256"]
    root.mkdir()
    for rater in manifest["raters"]:
        envelope = json.loads((pack / "raters" / rater / "assignments.json").read_text())
        _write_json(
            root / f"responses-{rater}.json",
            {
                "schema_version": 1,
                "pack_sha256": pack_hash,
                "rater_id": rater,
                "consented_at": "2026-01-01T00:00:00Z",
                "responses": [
                    _response_row(item, rater, pack_hash) for item in envelope["assignments"]
                ],
            },
        )


class Phase7EvaluationTests(unittest.TestCase):
    def test_unitization_keeps_segments_whole_and_is_deterministic(self):
        rows = [
            {
                "passage_id": "p",
                "book_id": "book",
                "subset": "continuous",
                "chapter_index": 0,
                "strata": ["narrative"],
                "segments": [
                    {"segment_id": "s1", "source": "one two", "word_count": 2},
                    {"segment_id": "s2", "source": "three four", "word_count": 2},
                ],
            }
        ]
        first = build_units(rows, "a" * 64)
        second = build_units(rows, "a" * 64)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["segment_ids"], ["s1", "s2"])
        self.assertTrue(first[0]["boundary"])

    def test_evaluation_spec_rejects_unsafe_or_duplicate_raters(self):
        base = {
            "schema_version": 1,
            "benchmark_id": "demo",
            "run_hash": "a" * 64,
            "corpus_sha256": "b" * 64,
            "seed": 7,
            "candidate_ids": ["a", "b"],
            "study_protocol": {
                "eligibility_text": "eligible",
                "consent_text": "consent",
                "compensation_text": "compensation",
                "retention_text": "retention",
            },
        }
        with self.assertRaises(ValidationError):
            EvaluationSpec.model_validate({**base, "raters": ["r", "r", "r"]})
        with self.assertRaises(ValidationError):
            EvaluationSpec.model_validate({**base, "raters": ["r", "r2", "../r3"]})

    def test_response_models_are_strict_and_context_explains_negative_judgment(self):
        common = {
            "assignment_id": "a",
            "rater_id": "r1",
            "pack_hash": "c" * 64,
            "started_at": "2026-01-01T00:00:00Z",
            "submitted_at": "2026-01-01T00:00:01Z",
            "active_ms": 100,
            "kind": "absolute",
        }
        valid = {
            **common,
            "fidelity": 3,
            "naturalness": 3,
            "style_voice": 3,
            "consistency": 3,
            "context_handling": 3,
            "readability": 3,
            "format_integrity": 3,
        }
        from trans_novel.benchmark.schema import AbsoluteResponse, ContextResponse

        self.assertEqual(AbsoluteResponse.model_validate(valid).fidelity, 3)
        with self.assertRaises(ValidationError):
            AbsoluteResponse.model_validate({**valid, "unexpected": True})
        with self.assertRaises(ValidationError):
            ContextResponse.model_validate({**common, "kind": "context", "judgment": "incorrect"})

    def test_unit_id_changes_when_source_or_segment_order_changes(self):
        first = build_units(
            [
                {
                    "passage_id": "p",
                    "book_id": "b",
                    "segments": [
                        {"segment_id": "a", "source": "one", "word_count": 1},
                        {"segment_id": "b", "source": "two", "word_count": 1},
                    ],
                }
            ],
            "d" * 64,
        )
        second = build_units(
            [
                {
                    "passage_id": "p",
                    "book_id": "b",
                    "segments": [
                        {"segment_id": "b", "source": "two", "word_count": 1},
                        {"segment_id": "a", "source": "one", "word_count": 1},
                    ],
                }
            ],
            "d" * 64,
        )
        self.assertNotEqual(first[0]["unit_id"], second[0]["unit_id"])

    def test_unitization_flushes_short_group_before_oversize_segment(self):
        units = build_units(
            [
                {
                    "passage_id": "p",
                    "book_id": "b",
                    "subset": "continuous",
                    "segments": [
                        {"segment_id": "a", "source": "x " * 100, "word_count": 100},
                        {"segment_id": "b", "source": "y " * 300, "word_count": 300},
                    ],
                }
            ],
            "e" * 64,
        )
        self.assertEqual([unit["segment_ids"] for unit in units], [["a"], ["b"]])
        self.assertTrue(all(unit["word_count"] <= 350 for unit in units))

    def test_public_generate_validate_import_all_kinds_and_incremental_state(self):
        corpus_hash = "b" * 64
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, run, fixture_spec = _synthetic_inputs(root)
            spec = {
                **fixture_spec,
                "candidate_ids": ["c1", "c2"],
                "absolute": {"target_source_words": 1, "ratings_per_output": 3},
                "pairwise": {"target_source_words": 360, "ratings_per_comparison": 2},
                "polish": {"target_source_words": 1, "ratings_per_pair": 3},
                "mqm": {"target_source_words": 1, "annotators_per_output": 2},
                "postedit": {"target_source_words": 1, "editors_per_output": 1},
                "context": {"target_source_words": 1, "ratings_per_output": 3},
                "study_protocol": {
                    "eligibility_text": "eligible",
                    "consent_text": "consent",
                    "compensation_text": "compensation",
                    "retention_text": "retention",
                },
            }
            pack = root / "pack"
            with patch(
                "trans_novel.benchmark.evaluation.validate_corpus",
                return_value={"corpus_sha256": corpus_hash},
            ):
                generate_pack(corpus, run, spec, pack)
            self.assertEqual(validate_pack(pack)["rater_count"], 3)
            from typer.testing import CliRunner

            from trans_novel.cli import app

            cli_result = CliRunner().invoke(
                app, ["tools", "benchmark", "evaluate", "validate", str(pack)]
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.stdout)
            responses = root / "responses"
            responses.mkdir()
            pack_hash = json.loads((pack / "pack.json").read_text())["pack_semantic_sha256"]
            for rater in ("r1", "r2", "r3"):
                envelope = json.loads((pack / "raters" / rater / "assignments.json").read_text())
                response_rows = []
                for item in envelope["assignments"]:
                    row = {
                        "schema_version": 1,
                        "assignment_id": item["assignment_id"],
                        "rater_id": rater,
                        "pack_hash": pack_hash,
                        "started_at": "2026-01-01T00:00:00Z",
                        "submitted_at": "2026-01-01T00:00:00Z",
                        "active_ms": 0,
                        "kind": item["kind"],
                    }
                    if item["kind"] == "absolute":
                        row.update(dict.fromkeys(item["dimensions"], 3))
                    elif item["kind"] == "pairwise":
                        row["preference"] = "tie"
                    elif item["kind"] == "polish":
                        row["outcome"] = "no_material_change"
                    elif item["kind"] == "mqm":
                        row["errors"] = []
                    elif item["kind"] == "context":
                        row["judgment"] = "correct"
                    else:
                        row["edited_target"] = item["target"]
                    response_rows.append(row)
                (responses / f"responses-{rater}.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "pack_sha256": pack_hash,
                            "rater_id": rater,
                            "consented_at": "2026-01-01T00:00:00Z",
                            "responses": response_rows,
                        }
                    ),
                    encoding="utf-8",
                )
            evaluation = root / "evaluation"
            import_responses(pack, responses, evaluation)
            self.assertEqual(
                json.loads((evaluation / "import_state.json").read_text())["status"],
                "complete",
            )

    def test_missing_candidate_surface_remains_a_hard_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, run, spec = _synthetic_inputs(root / "inputs")
            candidates = json.loads((run / "candidates.json").read_text())
            candidates[0]["stage"] = []
            _write_json(run / "candidates.json", candidates)
            with (
                patch(
                    "trans_novel.benchmark.evaluation.validate_corpus",
                    return_value={"corpus_sha256": spec["corpus_sha256"]},
                ),
                self.assertRaises(EvaluationError),
            ):
                generate_pack(corpus, run, spec, root / "pack")

    def test_pack_hashes_are_deterministic_and_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, run, spec = _synthetic_inputs(root / "inputs")
            with patch(
                "trans_novel.benchmark.evaluation.validate_corpus",
                return_value={"corpus_sha256": spec["corpus_sha256"]},
            ):
                first = generate_pack(corpus, run, spec, root / "pack1")
                second = generate_pack(corpus, run, spec, root / "pack2")

            def files(path):
                return {
                    str(item.relative_to(path)): item.read_bytes()
                    for item in path.rglob("*")
                    if item.is_file()
                }

            self.assertEqual(files(first), files(second))
            manifest = json.loads((first / "pack.json").read_text())
            self.assertEqual(
                manifest["pack_semantic_sha256"],
                json.loads((second / "pack.json").read_text())["pack_semantic_sha256"],
            )
            sentinel = root / "pack1" / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                generate_pack(corpus, run, spec, first)
            self.assertEqual(sentinel.read_text(), "keep")

    def test_completed_run_lineage_requires_exact_manifest_and_state(self):
        for mutation in ("pending", "unknown_key", "tampered"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                corpus, run, spec = _synthetic_inputs(root / "inputs")
                if mutation == "pending":
                    _write_json(run / "run_state.json", {"status": "pending"})
                else:
                    manifest = json.loads((run / "run.json").read_text())
                    if mutation == "unknown_key":
                        manifest["run_hash"] = spec["run_hash"]
                    else:
                        manifest["prompt_version"] = "tampered"
                    _write_json(run / "run.json", manifest)
                with (
                    patch(
                        "trans_novel.benchmark.evaluation.validate_corpus",
                        return_value={"corpus_sha256": spec["corpus_sha256"]},
                    ),
                    self.assertRaises(EvaluationError),
                ):
                    generate_pack(corpus, run, spec, root / "pack")

    def test_six_kind_quotas_word_targets_book_relaxation_and_balanced_raters(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            manifest = json.loads((pack / "pack.json").read_text())
            self.assertEqual(
                set(manifest["tasks"]),
                {
                    "absolute",
                    "pairwise",
                    "polish",
                    "mqm",
                    "context",
                    "postedit",
                },
            )
            self.assertTrue(all(manifest["tasks"][kind] > 0 for kind in manifest["tasks"]))
            self.assertTrue(
                all(manifest["task_source_words"][kind] > 0 for kind in manifest["tasks"])
            )
            self.assertTrue(
                all(value["book_cap_relaxed"] for value in manifest["book_balance"].values())
            )
            self.assertEqual(manifest["allocation"]["calibration_units_per_rater"], 30)
            for kind in manifest["tasks"]:
                counts = [
                    manifest["allocation"]["base_counts"][rater][kind]
                    for rater in manifest["raters"]
                ]
                self.assertLessEqual(max(counts) - min(counts), 1, kind)

    def test_pair_graph_orientation_and_presentation_ids_are_blind_and_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            manifest = json.loads((pack / "pack.json").read_text())
            secret = json.loads((pack / "secret_mapping.json").read_text())["assignments"]
            ids, edge_counts, orientations = set(), {}, {}
            for rater in manifest["raters"]:
                envelope = json.loads((pack / "raters" / rater / "assignments.json").read_text())
                for item in envelope["assignments"]:
                    aid = item["assignment_id"]
                    self.assertNotIn(aid, ids)
                    ids.add(aid)
                    metadata = secret[aid]
                    if (
                        item["calibration"]
                        or item["kind"] != "pairwise"
                        or metadata.get("duplicate_of") is not None
                    ):
                        continue
                    candidates = tuple(sorted(p["candidate_id"] for p in metadata["positions"]))
                    key = (metadata["surface"], item["unit_id"], candidates)
                    edge_counts[key] = edge_counts.get(key, 0) + 1
                    orientations.setdefault(key, set()).add(
                        tuple(p["candidate_id"] for p in metadata["positions"])
                    )
            expected_edges = {("c1", "c2"), ("c1", "c3")}
            self.assertEqual(
                {
                    surface: {key[2] for key in edge_counts if key[0] == surface}
                    for surface in ("attribution_final", "full_final")
                },
                {
                    "attribution_final": expected_edges,
                    "full_final": expected_edges,
                },
            )
            self.assertTrue(all(count == 2 for count in edge_counts.values()))
            self.assertTrue(all(len(values) == 2 for values in orientations.values()))
            self.assertEqual(len(ids), manifest["assignment_count"])

    def test_polish_reverse_swaps_position_objects_without_relabeling(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            secret = json.loads((pack / "secret_mapping.json").read_text())["assignments"]
            orientations = []
            for rater_dir in (pack / "raters").iterdir():
                items = json.loads((rater_dir / "assignments.json").read_text())["assignments"]
                for item in items:
                    metadata = secret[item["assignment_id"]]
                    if (
                        item["kind"] != "polish"
                        or item["calibration"]
                        or metadata["duplicate_of"] is not None
                    ):
                        continue
                    labels = tuple(
                        position["polish_position"] for position in metadata["positions"]
                    )
                    orientations.append(labels)
                    self.assertIn(labels, {("raw", "proposal"), ("proposal", "raw")})
            self.assertIn(("raw", "proposal"), orientations)
            self.assertIn(("proposal", "raw"), orientations)

    def test_calibration_is_thirty_rows_five_per_kind_for_every_rater(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            kinds = {"absolute", "pairwise", "polish", "mqm", "context", "postedit"}
            for rater_dir in (pack / "raters").iterdir():
                items = json.loads((rater_dir / "assignments.json").read_text())["assignments"]
                calibration = [item for item in items if item["calibration"]]
                self.assertEqual(len(calibration), 30)
                self.assertEqual(
                    {kind: sum(item["kind"] == kind for item in calibration) for kind in kinds},
                    dict.fromkeys(kinds, 5),
                )

    def test_hidden_duplicates_use_per_rater_and_kind_denominator_and_reverse_pair_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            secret = json.loads((pack / "secret_mapping.json").read_text())["assignments"]
            kinds = {"absolute", "pairwise", "polish", "mqm", "context", "postedit"}
            for rater_dir in (pack / "raters").iterdir():
                items = json.loads((rater_dir / "assignments.json").read_text())["assignments"]
                self.assertEqual({item["kind"] for item in items}, kinds)
                for kind in kinds:
                    base = [
                        item
                        for item in items
                        if item["kind"] == kind
                        and not item["calibration"]
                        and secret[item["assignment_id"]]["duplicate_of"] is None
                    ]
                    duplicates = [
                        item
                        for item in items
                        if item["kind"] == kind
                        and secret[item["assignment_id"]]["duplicate_of"] is not None
                    ]
                    self.assertEqual(len(duplicates), int(len(base) * 0.1 + 0.5))
                    for item in duplicates:
                        self.assertIn(
                            secret[item["assignment_id"]]["duplicate_of"],
                            {other["assignment_id"] for other in base},
                        )

    def test_replicate_split_and_postedit_diversity_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp), replicas=True)
            manifest = json.loads((pack / "pack.json").read_text())
            self.assertTrue(
                any(
                    position.get("replicate") == 2
                    for metadata in json.loads((pack / "secret_mapping.json").read_text())[
                        "assignments"
                    ].values()
                    for position in metadata.get("positions", [])
                )
            )
            diversity = manifest["allocation"]["postedit_diversity"]
            self.assertTrue(diversity)
            self.assertTrue(
                all(
                    value["unavoidable_one_output"] or len(value["raters"]) >= 2
                    for value in diversity.values()
                )
            )
            secret = json.loads((pack / "secret_mapping.json").read_text())["assignments"]
            owners: dict[tuple[str, str, str, int], set[str]] = {}
            for rater in manifest["raters"]:
                items = json.loads((pack / "raters" / rater / "assignments.json").read_text())[
                    "assignments"
                ]
                for item in items:
                    metadata = secret[item["assignment_id"]]
                    if (
                        item["kind"] != "postedit"
                        or item["calibration"]
                        or metadata["duplicate_of"] is not None
                    ):
                        continue
                    position = metadata["positions"][0]
                    key = (
                        position["candidate_id"],
                        metadata["surface"],
                        item["unit_id"],
                        position["replicate"],
                    )
                    owners.setdefault(key, set()).add(rater)
            self.assertTrue(owners and all(len(value) == 1 for value in owners.values()))
            first_key = min(owners)
            preferred = _seed(
                7,
                "base_rater_allocation",
                [
                    "postedit",
                    first_key[0],
                    first_key[1],
                ],
            ) % len(manifest["raters"])
            self.assertEqual(owners[first_key], {manifest["raters"][preferred]})

    def test_secret_provenance_permissions_and_rater_asset_leakage_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            pack, _ = _make_pack(Path(temp))
            secret_path = pack / "secret_mapping.json"
            self.assertEqual(stat.S_IMODE(secret_path.stat().st_mode), 0o600)
            secret = json.loads(secret_path.read_text())
            required = {
                "artifact_id",
                "artifact_key",
                "stage",
                "stage_name",
                "provider",
                "requested_model_id",
                "resolved_model_id",
                "replicate",
                "surface",
                "context_strategy",
                "raw_artifact_id",
                "editor_artifact_id",
            }
            self.assertTrue(
                all(
                    required <= set(position)
                    for metadata in secret["assignments"].values()
                    for position in metadata.get("positions", [])
                )
            )
            self.assertTrue(
                all(
                    metadata.get("positions") and metadata["positions"][0].get("artifact_id")
                    for metadata in secret["assignments"].values()
                )
            )
            for path in (pack / "raters").rglob("*"):
                self.assertFalse(
                    any(token in str(path.relative_to(pack)) for token in ("c1", "c2", "c3"))
                )
                if path.is_file() and path.suffix == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertFalse(_leak(payload, {"c1", "c2", "c3"}))
            app = (pack / "raters" / "r1" / "app.js").read_text()
            self.assertIn('fetch("./assignments.json")', app)
            self.assertNotIn("secret_mapping", app)
            self.assertNotIn("XMLHttpRequest", app)
            self.assertNotIn("import(", app)
            self.assertNotRegex(app, r"https?://|cdn|script\.src")
            self.assertIn("textContent", app)
            self.assertIn("visibilitychange", app)
            self.assertIn("localStorage", app)
            self.assertIn("responses-", app)
            html = (pack / "raters" / "r1" / "index.html").read_text()
            self.assertIn("参与资格：", html)
            self.assertIn("知情同意：", html)
            self.assertIn("&lt;", html)
            self.assertIn("disabled", html)

    def test_validate_rejects_pack_asset_mapping_and_hash_tampering(self):
        for tamper in ("pack", "asset", "asset_rehashed", "mapping", "hash"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                pack, _ = _make_pack(Path(temp))
                if tamper == "pack":
                    path = pack / "pack.json"
                    value = json.loads(path.read_text())
                    value["tasks"]["absolute"] += 1
                    _write_json(path, value)
                elif tamper in {"asset", "asset_rehashed"}:
                    path = pack / "raters" / "r1" / "app.js"
                    path.write_text(path.read_text() + "x", encoding="utf-8")
                    if tamper == "asset_rehashed":
                        manifest_path = pack / "pack.json"
                        manifest = json.loads(manifest_path.read_text())
                        relative = str(path.relative_to(pack))
                        manifest["rater_files"][relative] = sha256_bytes(path.read_bytes())
                        manifest["pack_sha256"] = sha256_bytes(
                            canonical_json(
                                {
                                    key: value
                                    for key, value in manifest.items()
                                    if key != "pack_sha256"
                                }
                            ).encode()
                        )
                        _write_json(manifest_path, manifest)
                elif tamper == "mapping":
                    path = pack / "secret_mapping.json"
                    value = json.loads(path.read_text())
                    value["assignments"]["missing"] = {}
                    _write_json(path, value)
                else:
                    path = pack / "pack.json"
                    value = json.loads(path.read_text())
                    value["pack_sha256"] = "0" * 64
                    _write_json(path, value)
                with self.assertRaises(EvaluationError):
                    validate_pack(pack)

    def test_response_contract_rejects_rfc3339_active_mqm_assignment_kind_and_consent_errors(self):
        mutations = {
            "timestamp": lambda row, envelope: row.update({"started_at": "not-a-time"}),
            "active": lambda row, envelope: row.update({"active_ms": 2000}),
            "kind": lambda row, envelope: row.update({"kind": "mqm"}),
            "assignment": lambda row, envelope: row.update({"assignment_id": "unknown"}),
            "pack": lambda row, envelope: row.update({"pack_hash": "0" * 64}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pack, _ = _make_pack(root / "pack-input")
                responses = root / "responses"
                _write_responses(pack, responses)
                path = responses / "responses-r1.json"
                envelope = json.loads(path.read_text())
                mutate(envelope["responses"][0], envelope)
                _write_json(path, envelope)
                with self.assertRaises(EvaluationError):
                    import_responses(pack, responses, root / "evaluation")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, _ = _make_pack(root / "consent-input")
            responses = root / "responses"
            _write_responses(pack, responses)
            path = responses / "responses-r1.json"
            envelope = json.loads(path.read_text())
            del envelope["consented_at"]
            _write_json(path, envelope)
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, root / "evaluation")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, _ = _make_pack(root / "mqm-input")
            responses = root / "responses"
            _write_responses(pack, responses)
            path = responses / "responses-r1.json"
            envelope = json.loads(path.read_text())
            mqm_row = next(row for row in envelope["responses"] if row["kind"] == "mqm")
            mqm_row["errors"] = [
                {
                    "segment_id": "unknown",
                    "severity": "major",
                    "type": "fluency",
                    "note": "bad segment",
                }
            ]
            _write_json(path, envelope)
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, root / "evaluation")

    def test_offline_ui_state_and_all_kind_submit_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, _ = _make_pack(root / "input")
            app = (pack / "raters" / "r1" / "app.js").read_text()
            for token in (
                "const draftValue=",
                "current_started_at",
                "current_started_ms",
                "visibleStart",
                "active_ms",
                "pairValues",
                "polishValues",
                "contextValues",
                "segmentIds",
                "seenErrors",
                "Number(values[key])",
                "flush(endMs)",
                "submitted_at:new Date(endMs).toISOString()",
            ):
                self.assertIn(token, app)
            assignments = json.loads((pack / "raters" / "r1" / "assignments.json").read_text())[
                "assignments"
            ]
            absolute = next(item for item in assignments if item["kind"] == "absolute")
            representative = _response_row(
                absolute,
                "r1",
                json.loads((pack / "pack.json").read_text())["pack_semantic_sha256"],
            )
            self.assertNotIn("note", representative)
            invalid = {
                "absolute": lambda row: row.update({"fidelity": 6}),
                "pairwise": lambda row: row.update({"preference": "invalid"}),
                "polish": lambda row: row.update({"outcome": "invalid"}),
                "context": lambda row: row.update({"judgment": "incorrect", "note": " "}),
                "postedit": lambda row: row.update({"edited_target": " "}),
                "mqm": lambda row: row.update(
                    {
                        "errors": [
                            {
                                "segment_id": "unknown",
                                "severity": "major",
                                "type": "fluency",
                                "note": "bad",
                            }
                        ]
                    }
                ),
            }
            for kind, mutate in invalid.items():
                with self.subTest(kind=kind):
                    case_root = root / kind
                    case_root.mkdir()
                    responses = case_root / "responses"
                    _write_responses(pack, responses)
                    path = responses / "responses-r1.json"
                    envelope = json.loads(path.read_text())
                    mutate(next(row for row in envelope["responses"] if row["kind"] == kind))
                    _write_json(path, envelope)
                    with self.assertRaises(EvaluationError):
                        import_responses(pack, responses, case_root / "evaluation")

    def test_import_requires_exact_download_response_filenames(self):
        for filename in ("r1.json", "responses-.json", "extra-responses-r1.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                pack, _ = _make_pack(root / "input")
                responses = root / "responses"
                _write_responses(pack, responses)
                source = responses / "responses-r1.json"
                source.rename(responses / filename)
                for rater in ("r2", "r3"):
                    (responses / f"responses-{rater}.json").unlink()
                with self.assertRaises(EvaluationError):
                    import_responses(pack, responses, root / "evaluation")

    def test_pair_relation_requires_two_primary_presentations(self):
        mapping = {
            "only": {
                "kind": "pairwise",
                "unit_id": "u",
                "surface": "s",
                "positions": [
                    {
                        "candidate_id": "a",
                        "replicate": 1,
                        "segment_provenance": [],
                    },
                    {
                        "candidate_id": "b",
                        "replicate": 1,
                        "segment_provenance": [],
                    },
                ],
                "calibration": False,
                "duplicate_of": None,
            },
        }
        with self.assertRaises(EvaluationError):
            _pair_adjudication([], mapping)

    def test_import_is_atomic_incremental_idempotent_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, _ = _make_pack(root / "input")
            responses = root / "responses"
            _write_responses(pack, responses)
            r2_path = responses / "responses-r2.json"
            r2_envelope = json.loads(r2_path.read_text())
            pair = next(
                row
                for row in r2_envelope["responses"]
                if row["kind"] == "pairwise" and not row.get("calibration")
            )
            pair["preference"] = "b_much_better"
            _write_json(r2_path, r2_envelope)
            for rater in ("r2", "r3"):
                (responses / f"responses-{rater}.json").unlink()
            evaluation = root / "evaluation"
            import_responses(pack, responses, evaluation)
            state = json.loads((evaluation / "import_state.json").read_text())
            self.assertEqual(state["status"], "incomplete")
            _write_responses(pack, root / "all-responses")
            all_r2 = root / "all-responses" / "responses-r2.json"
            all_r2_envelope = json.loads(all_r2.read_text())
            all_pair = next(
                row
                for row in all_r2_envelope["responses"]
                if row["kind"] == "pairwise" and not row.get("calibration")
            )
            all_pair["preference"] = "b_much_better"
            _write_json(all_r2, all_r2_envelope)
            for rater in ("r2", "r3"):
                shutil.copy(
                    root / "all-responses" / f"responses-{rater}.json",
                    responses / f"responses-{rater}.json",
                )
            import_responses(pack, responses, evaluation)
            self.assertEqual(
                json.loads((evaluation / "import_state.json").read_text())["status"], "complete"
            )
            adjudication = json.loads((evaluation / "adjudication_needed.json").read_text())
            self.assertEqual(
                adjudication, sorted(adjudication, key=lambda row: row["relation_key"])
            )
            self.assertTrue(
                adjudication
                and all(
                    {"relation_key", "surface", "unit_id", "positions", "responses"} <= set(row)
                    for row in adjudication
                )
            )
            before = {
                str(path.relative_to(evaluation)): path.read_bytes()
                for path in evaluation.rglob("*")
                if path.is_file()
            }
            import_responses(pack, responses, evaluation)
            after = {
                str(path.relative_to(evaluation)): path.read_bytes()
                for path in evaluation.rglob("*")
                if path.is_file()
            }
            self.assertEqual(before, after)
            conflicting = json.loads((responses / "responses-r1.json").read_text())
            absolute = next(row for row in conflicting["responses"] if row["kind"] == "absolute")
            absolute["fidelity"] = 4
            _write_json(responses / "responses-r1.json", conflicting)
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, evaluation)
            shutil.copy(
                root / "all-responses" / "responses-r1.json", responses / "responses-r1.json"
            )
            source = evaluation / "source_responses" / "r1.json"
            source.write_text(source.read_text() + "tampered", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, evaluation)
            shutil.copy(root / "all-responses" / "responses-r1.json", source)
            derived = json.loads((evaluation / "evaluation_complete.json").read_text())[
                "derived_files"
            ]
            self.assertIn("responses.jsonl", derived)
            (evaluation / "responses.jsonl").write_text("tampered", encoding="utf-8")
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, evaluation)

    def test_import_mixed_valid_invalid_batch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, _ = _make_pack(root / "input")
            responses = root / "responses"
            _write_responses(pack, responses)
            path = responses / "responses-r2.json"
            envelope = json.loads(path.read_text())
            envelope["responses"][0]["active_ms"] = 5000
            _write_json(path, envelope)
            with self.assertRaises(EvaluationError):
                import_responses(pack, responses, root / "evaluation")
            self.assertFalse((root / "evaluation").exists())

    def test_cli_pack_validate_import_wrappers_and_nonzero_errors(self):
        from typer.testing import CliRunner

        from trans_novel.cli import app

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, run, spec = _synthetic_inputs(root / "inputs")
            spec_path = root / "spec.json"
            _write_json(spec_path, spec)
            with patch(
                "trans_novel.benchmark.evaluation.validate_corpus",
                return_value={"corpus_sha256": spec["corpus_sha256"]},
            ):
                packed = CliRunner().invoke(
                    app,
                    [
                        "tools",
                        "benchmark",
                        "evaluate",
                        "pack",
                        str(corpus),
                        str(run),
                        str(spec_path),
                        "--out",
                        str(root / "pack"),
                    ],
                )
            self.assertEqual(packed.exit_code, 0, packed.stdout)
            checked = CliRunner().invoke(
                app,
                [
                    "tools",
                    "benchmark",
                    "evaluate",
                    "validate",
                    str(root / "pack"),
                ],
            )
            self.assertEqual(checked.exit_code, 0, checked.stdout)
            responses = root / "responses"
            _write_responses(root / "pack", responses)
            imported = CliRunner().invoke(
                app,
                [
                    "tools",
                    "benchmark",
                    "evaluate",
                    "import",
                    str(root / "pack"),
                    str(responses),
                    "--out",
                    str(root / "evaluation"),
                ],
            )
            self.assertEqual(imported.exit_code, 0, imported.stdout)
            failed = CliRunner().invoke(
                app,
                [
                    "tools",
                    "benchmark",
                    "evaluate",
                    "validate",
                    str(root / "missing"),
                ],
            )
            self.assertNotEqual(failed.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
