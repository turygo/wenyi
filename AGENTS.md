# Repository Guidelines

## Project Overview

`trans-novel` is a synchronous Python CLI for translating EPUB, FB2, TXT, and Markdown books into Chinese EPUB/TXT while preserving source structure and terminology consistency. The normal command is `translate`; `resume`, `status`, and `tools` support persisted runs and maintenance. Source lives in `trans_novel/`; both `python -m trans_novel` and the installed `trans-novel` command delegate to `trans_novel.cli:main`.

## Architecture & Data Flow

1. `trans_novel/cli.py` loads the strict configuration, applies per-run CLI overrides, and calls `Application` in `trans_novel/pipeline/bootstrap.py`. `Application` is the production composition root: it constructs and injects clients, agents, nodes, the planner, and the runner, then drives prepare, translate, and finish phases.
2. `trans_novel/ingest/` parses and segments source books. Stage behavior belongs in `trans_novel/pipeline/nodes/`; translation nodes handle batching, alignment, lint/fix, context, glossary checkpoints, and optional polish, while finish nodes handle titles, deterministic QA, reports, and assembly.
3. `Planner` in `pipeline/planner.py` owns dependency planning, scopes, optional nodes, chapter targets, backmatter policy, fingerprints, and invalidation. `WorkflowRunner` in `pipeline/runner.py` owns execution lifecycle, locking, status transitions, failure policy, chapter concurrency, persistence coordination, and usage flushing. Keep the runner business-agnostic.
4. Version-3 Pydantic state in `pipeline/state.py` and `RunStore` in `pipeline/runstore.py` own persistence and resume. Source/language identity and node fingerprints protect resume safety; migrations, interrupted-run recovery, checkpoint journals, and descendant invalidation must remain deterministic.
5. Runtime calls remain synchronous. Chapter work uses a bounded `ThreadPoolExecutor` with at most four workers; ordered commits and `RunStore` reads/writes are serialized through the runner's lock and main-thread persistence path.
6. `trans_novel/assemble/writer.py` reopens EPUB sources to preserve navigation/resources or emits TXT, supporting mono/bilingual output and source fallback.

`trans_novel/benchmark/` is a separate offline evaluation subsystem exposed under `tools benchmark`. It reuses the production `Application` with isolated state and records corpus, run, review, telemetry, and report artifacts; detailed operating rules live in `DOCS/benchmark-guide.md`.

## Key Directories

- `trans_novel/pipeline/`: composition, planning, lifecycle, resumable state, and stage nodes.
- `trans_novel/ingest/` and `trans_novel/assemble/`: source parsing/segmentation and structural EPUB/TXT reconstruction.
- `trans_novel/llm/` and `trans_novel/agents/`: provider routing/retry/usage boundaries and prompt-driven behavior.
- `trans_novel/glossary/`: terminology persistence and resolution.
- `trans_novel/benchmark/`: isolated offline evaluation tooling, not production workflow ownership.
- `tests/`: offline `unittest` suites plus fake LLM and generated-book fixtures.
- `scripts/` and `.github/workflows/`: changelog/release gates, test matrix, and binary builds.

`state/`, `benchmark_data/`, `benchmark_runs/`, `build/`, `dist/`, local `config.yaml`, databases, source books, and generated translations are ignored/generated artifacts. Do not edit or commit them as repository source.

## Development Commands

```bash
uv sync --locked --group dev
uv run trans-novel --help
uv run trans-novel translate book.epub
uv run python -m unittest tests.test_pipeline
uv run python -m unittest tests.test_routing
uv run --no-sync python -m unittest discover -s tests
uv run ruff check --fix .
uv run ruff format .
uv run pre-commit install
uv build
```

The package-installed source command is `trans-novel`; `python -m trans_novel` is an equivalent source entry point. Release workflows build the single-file executable as `wenyi` (`wenyi.exe` on Windows), so do not use that binary name in source-development commands.

## Code Conventions & Common Patterns

- Support Python `>=3.10`; keep public boundaries typed and use absolute imports. Ruff targets Python 3.10, a 100-column line length, import sorting, and the configured `E/W/F/I/B/C4/PIE/RUF/SIM/TID/UP` rules.
- Keep configuration narrow and strict. YAML exposes model lists for the `translator`, `analyst`, `editor`, and `fast` roles plus `quality`; unknown keys fail validation. CLI values override the loaded quality preset, and missing config uses code defaults.
- Preserve constructor injection. `Application` is the only production construction site; agents and nodes receive precise dependencies rather than creating hidden clients, stores, or configuration.
- Route agents to ordered model-role candidates through `llm/router.py` and construct transports through `llm/registry.py`. Keep retry/fallback classification centralized in `llm/retrying.py`; do not add a second provider-routing or retry path.
- Keep state writes behind `RunStore` and runner coordination. Preserve identity hashes, fingerprints, node/chapter statuses, advisory locking, ordered checkpoint commits, and idempotent recovery instead of special-casing resume in individual nodes.
- Put stage-specific decisions in `pipeline/nodes/`, deterministic parsing/formatting in their existing subsystems, and external validation in Pydantic models. Reuse existing patterns rather than adding parallel orchestration or configuration layers.
- Tests use `Test...` classes, `test_...` methods, `assertRaises`/`assertRaisesRegex`, and `subTest` for matrices.

## Important Files

- `trans_novel/pipeline/bootstrap.py`: `Application` composition root and workflow facade.
- `trans_novel/pipeline/planner.py`, `runner.py`, `state.py`, `runstore.py`: planning, lifecycle, V3 state, persistence, migration, and recovery contracts.
- `trans_novel/pipeline/nodes/translate.py`, `finish.py`: translation and terminal stage behavior.
- `trans_novel/config.py`, `config.example.yaml`: strict schema, model roles, defaults, and quality presets.
- `trans_novel/llm/router.py`, `registry.py`, `retrying.py`: role routing, transport construction, and centralized retry/fallback.
- `trans_novel/ingest/segmenter.py`, `trans_novel/assemble/writer.py`: batching and output reconstruction boundaries.
- `tests/fake_llm.py`, `tests/sample_data.py`: canonical offline client seam and generated book fixtures.
- `tests/test_pipeline.py`, `tests/test_checkpoint.py`, `tests/test_routing.py`, `tests/test_ingest.py`, `tests/test_assemble.py`: representative workflow, recovery, provider, and structural contracts.
- `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`: package/tool configuration and local gates.
- `README.md`, `DOCS/benchmark-guide.md`, `CHANGELOG.md`: user behavior, benchmark operations, and release history.
- `scripts/check_changelog.py`, `scripts/prepare_release.py`: changed-path changelog policy and tag/version/release-note validation.

## Runtime/Tooling Preferences

- Use `uv` with the committed `uv.lock`. CI installs locked dev dependencies and tests Python 3.10 and 3.12.
- Hatchling builds the wheel; Typer/Rich provide the CLI; release CI uses PyInstaller for five platform/architecture targets and smoke-checks the resulting `wenyi` binaries.
- Keep runtime behavior synchronous. Use the existing bounded executor and main-thread persistence protocol rather than introducing an isolated `asyncio` call graph.
- Provider credentials belong in environment variables such as `OPENCODE_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, and `OPENROUTER_API_KEY`, never plaintext repository configuration.
- Routing, retries, timeouts, and concurrency are internal policy, not general YAML knobs. Unsupported configuration or model capabilities should fail explicitly.
- The changelog gate runs locally and in CI. Changes under `trans_novel/`, `tests/`, `scripts/`, `.github/workflows/`, or to `.pre-commit-config.yaml`, `pyproject.toml`, or `uv.lock` must update `CHANGELOG.md` under `[Unreleased]` in the same change.

## Testing & QA

- Use standard-library `unittest`; the canonical full CI command is `uv run --no-sync python -m unittest discover -s tests`. `pytest` is a dev dependency but is not the repository's canonical runner.
- Keep the suite offline. Inject `FakeClient` from `tests/fake_llm.py` or stub transports, inspect call metadata when relevant, and never send provider requests from tests.
- Isolate filesystem state with `tempfile.TemporaryDirectory`. Build TXT/FB2/EPUB inputs with `tests/sample_data.py` or focused inline fixtures, then reopen outputs to assert serialized state and ZIP/XML structure.
- Use Typer's `CliRunner` for CLI behavior, normalize ANSI output for stable assertions, and patch only external boundaries.
- Pipeline changes should cover observable planning, fingerprints, lifecycle statuses, resume/idempotency, checkpoint recovery, ordered persistence, and unexpected LLM calls. Routing changes should cover candidate order, retry classification, error identity, and usage attribution. Ingest/assembly changes should cover structural round trips and unsafe/malformed input boundaries.
- There is no numeric coverage threshold. Every behavior change still requires a focused behavioral regression in an existing appropriate module and the full offline suite before submission.
