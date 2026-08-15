# Repository Guidelines

## Project Overview

`trans-novel` is a Python CLI for translating long-form EPUB, FB2, and text-family inputs into Chinese while preserving book structure and terminology consistency. The system combines deterministic parsing/linting with specialized LLM agents for analysis, naming, translation, review, polishing, and consistency work. Source lives in `trans_novel/`; the supported source entry point is `trans_novel.cli:main`.

## Architecture & Data Flow

1. `trans_novel/cli.py` loads `Config`, applies CLI overrides, and dispatches `translate`, `resume`, `status`, or `tools` commands.
2. `ingest/` parses the input into `Book`, `Chapter`, and `Segment` models. EPUB navigation and resources are retained for later reconstruction.
3. `llm/registry.py` builds the provider transport; `AgentRouter` maps production agents to the configured `primary` or `fast` model. Agents receive `LLMClient` and `Config` through constructors.
4. `pipeline/orchestrator.py` detects language, analyzes a sample, builds the style guide, pre-scans chapters, mines terminology, resolves names once, and generates book context.
5. Translation runs chapter-by-chapter and batch-by-batch. Deterministic lint checks quotes, numbers, locked names, untranslated text, and segment counts; failures trigger targeted retries or safe fallback. Optional polishing, punctuation normalization, chapter review, severe-issue autofix, and consistency QA follow the quality preset.
6. `RunStore` owns resumable state under `state/<book-slug>/`. It atomically writes manifests, chapter JSON, context, reports, usage, and events; `glossary.db` becomes read-only during translation. Completed segments must not be translated again on resume.
7. `assemble/` writes EPUB/TXT outputs and QA reports, reusing original EPUB structure where possible.

Key patterns: a synchronous call graph with bounded `ThreadPoolExecutor` concurrency, constructor-injected LLM clients/configuration, strict Pydantic configuration boundaries, agent-local business fallbacks, centralized provider retries, and persistent state owned by `RunStore`. Do not introduce a second routing, retry, configuration, or state-writing path.

## Key Directories

- `trans_novel/ingest/`: EPUB/FB2/text parsing, TOC handling, domain models, segmentation.
- `trans_novel/llm/`: stable `LLMClient` interface, provider registry/transports, routing, retries, JSON parsing, usage accounting.
- `trans_novel/agents/`: prompt-driven analysis, translation, naming, review, polish, synopsis, naturalization, and consistency behavior.
- `trans_novel/glossary/`: SQLite terminology store, candidate mining, extraction, resolution, auditing.
- `trans_novel/pipeline/`: orchestration, resumable state, rolling context, deterministic lint, back-matter policy.
- `trans_novel/postprocess/`: deterministic Chinese punctuation normalization.
- `trans_novel/assemble/`: EPUB/TXT reconstruction and report generation.
- `tests/`: offline `unittest` coverage and reusable fake LLM/sample-book helpers.
- `scripts/`: changelog gate and release metadata validation.
- `.github/workflows/`: Python 3.10/3.12 tests and multi-platform PyInstaller release builds.

`state/`, `build/`, `dist/`, local `config.yaml`, databases, source books, and translated outputs are generated/ignored artifacts. Never edit or commit them as source.

## Development Commands

```bash
uv sync                                           # install locked runtime and dev dependencies
uv build                                          # build the Hatchling source/wheel packages
uv tool install .                                 # install the source CLI as trans-novel
uv run trans-novel --help                         # run from the managed environment
uv run trans-novel translate book.epub            # real provider; requires its API key
uv run ruff format .                              # format
uv run ruff check --fix .                         # lint and import-sort
uv run python -m unittest discover -s tests        # full offline suite
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
uv run pre-commit install                         # enable changelog + Ruff commit hooks
```

Run a focused module while iterating, for example:

```bash
uv run python -m unittest tests.test_orchestrator
uv run python -m unittest tests.test_llm
```

Release binaries are built by `.github/workflows/build.yml` as `wenyi`; do not confuse that packaged executable name with the source-installed `trans-novel` command.

## Code Conventions & Common Patterns

- Target Python 3.10; use `from __future__ import annotations` where needed and keep public boundaries typed.
- Ruff is authoritative: 100-column target, `E/W/F/I` checks, and `E501` ignored for long prompt literals.
- Use `snake_case` for modules/functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants, and leading underscores for module-private helpers. Tests use `Test...` classes and `test_...` methods.
- Prefer dataclasses for runtime/domain values and Pydantic for external configuration validation. `Config` rejects unknown and deprecated keys; do not silently accept legacy schemas.
- Configuration precedence is CLI override, then `config.yaml` (`llm` and `quality` only), then code defaults. Change `trans_novel/config.example.yaml` when changing generated defaults.
- Keep the runtime synchronous. Existing parallel work uses `ThreadPoolExecutor`; do not add isolated `asyncio` APIs to this call graph.
- Pass `LLMClient`/`Config` explicitly. Tests substitute `FakeClient`; production code must not instantiate hidden clients inside agents.
- Every LLM request needs stable `agent` and `operation` identifiers for routing, retry behavior, and usage attribution. Unknown production agents fail explicitly.
- Keep transport/retry failures in `llm/errors.py`, `llm/retrying.py`, and provider transports. Business-specific default responses belong in the agent layer; do not swallow provider failures globally.
- Preserve resume invariants: atomic JSON replacement, `.run.lock` mutual exclusion, `STATUS_DONE`, batch-level reuse, and persisted `review_pending` work.
- During translation the terminology store is read-only. Read `README.md` sections “工作流程” and “一致性机制” before changing naming, glossary injection, lint, or resume behavior.
- Prefer quality/CLI configuration for translation behavior before changing pipeline code.

## Important Files

- `pyproject.toml`: package metadata, Python/dependency constraints, CLI entry point, Hatchling and Ruff settings.
- `trans_novel/cli.py`, `trans_novel/__main__.py`: command tree and executable entry points.
- `trans_novel/config.py`, `trans_novel/config.example.yaml`: strict config schema, defaults, quality presets.
- `trans_novel/pipeline/orchestrator.py`: end-to-end state machine and resume logic.
- `trans_novel/pipeline/runstore.py`: state layout, locking, atomic writes, events, usage persistence.
- `trans_novel/pipeline/lint.py`: deterministic translation invariants.
- `trans_novel/llm/base.py`, `router.py`, `registry.py`, `retrying.py`: provider abstraction and model routing contract.
- `trans_novel/glossary/store.py`: SQLite terminology ownership and locking semantics.
- `trans_novel/assemble/writer.py`: final EPUB/TXT reconstruction.
- `tests/fake_llm.py`: canonical offline routing handler/config helper.
- `README.md`: user behavior, workflow, consistency, configuration, and release process.
- `CHANGELOG.md`, `scripts/check_changelog.py`, `scripts/prepare_release.py`: change and release gates.

## Runtime/Tooling Preferences

- Python `>=3.10`; CI covers 3.10 and 3.12.
- Use `uv` and the committed `uv.lock`; do not introduce Poetry, pip requirement files, or another environment manager.
- Build backend: Hatchling. CLI: Typer + Rich. Distribution binaries: PyInstaller through CI.
- Default provider is OpenCode Go. Provider-specific credentials include `OPENCODE_API_KEY`, `DEEPSEEK_API_KEY`, `BAILIAN_API_KEY`, `OPENAI_API_KEY`, and `OPENROUTER_API_KEY`.
- Offline development and tests must use `llm.provider: fake`/`FakeClient`; never send real network requests from tests or debugging.
- Standard providers own their endpoint and API-key variable. Only `openai-compatible` accepts custom `base_url`/`api_key_env` configuration.
- Code, tests, dependencies, scripts, build/workflow changes, or pre-commit changes must update `CHANGELOG.md` under `[Unreleased]` in the same change. Commit messages are English.

## Testing & QA

- Canonical framework and runner: standard-library `unittest`, despite `pytest` being a dev dependency.
- Use `tempfile.TemporaryDirectory` for books, SQLite files, and `state_dir`; never depend on the repository’s local `state/` contents.
- Use `FakeClient` and `tests/fake_llm.py` handlers for LLM behavior. CLI tests use Typer’s `CliRunner`; use `unittest.mock` for boundaries and strip ANSI output where assertions require stable text.
- Representative suites: `test_ingest.py`, `test_llm.py`, `test_routing.py`, `test_orchestrator.py`, `test_backmatter.py`, `test_usage.py`, `test_cli.py`, and `test_release_tooling.py`.
- For pipeline changes, test observable resume/idempotency behavior: completed segments are reused, pending reviews recover, atomic state remains readable, and no unexpected LLM calls occur.
- For routing/provider changes, test agent/model selection, request dialect, retry classification, error propagation, and usage schema attribution.
- For ingest/assemble changes, test generated temporary EPUB/FB2/TXT fixtures and structural round trips rather than repository books.
- No numeric coverage threshold is configured. Every behavior change still needs a focused regression test plus the full offline suite before submission.
