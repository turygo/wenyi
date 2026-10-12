# Repository Instructions

## Scope and stopping rules

- Implement only the user's requested behavior. Every additional change must address a named acceptance criterion, a demonstrated supported-use failure, or a concrete safety requirement. "More complete", "more compatible", and "future-proof" are not justification.
- A difference between libraries is not automatically a bug. Before writing an adapter, prove that the difference breaks a supported valid input, an existing user artifact, or an explicit requirement. Do not reproduce a legacy library's accidental tolerance of malformed inputs unless the user requests it.
- Prefer existing libraries and repository patterns. Do not turn a feature into a parser, compatibility layer, framework, or replacement subsystem without demonstrated necessity and explicit scope approval.
- Prioritize the complete production path. Exercise the real requested command or API before polishing unrelated component edges. Component tests, code volume, and subagent reports are not end-to-end delivery.
- Investigate only the concrete question blocking the next required implementation step. Once evidence permits a decision, implement it and stop researching adjacent cases.
- Before delegation, verify actual shared interfaces and fix file ownership, inputs, results, errors, and acceptance. Do not describe nonexistent interfaces as existing or delegate unresolved shared design. The main agent owns integration and acceptance.
- Explain a necessary scope expansion before implementing it. Ask the user about material product or risk tradeoffs; resolve repository-answerable questions from evidence.
- When the user says stop, immediately cancel active writers and stop editing. Task reminders, pending acceptance, and earlier authorization do not permit restarting. Resume only on a new explicit instruction.
- Preserve unrelated user changes and data. A rollback removes only the identified task's changes. Do not change global agent instructions when the request targets this repository.

## Verification and reporting

- Verify local behavior with repository tests and complete workflows through production entry points. Reuse existing private test support; do not create standalone experiment programs, temporary projects, servers, or driver scripts unless explicitly requested.
- Keep tests offline and isolate filesystem mutations in temporary test directories. Never use user books, configuration, state, or results as test data.
- Report only observed results. Name incomplete behavior and failed or unexercised acceptance explicitly; never present a scaffold, test count, queued CI run, or source inspection as completion.
- Do not add progress documents, speculative modules, compatibility shims, or unrelated cleanup.

## Repository essentials

`trans-novel` is a synchronous Python CLI. Source lives in `trans_novel/`, tests in `tests/`, and shared test fixtures in `tests/fixtures/`. `trans_novel.pipeline.application.Application` is the production composition root. Preserve existing dependency injection, centralized model routing/retry, and coordinated state persistence.

Use the committed `uv.lock`, standard-library `unittest`, and existing Ruff configuration. For Python production, test, or architecture changes, follow `docs/architecture-governance.md`; do not weaken its executable gate or add a baseline bypass. Relevant code/test/tooling changes must update `[Unreleased]` in `CHANGELOG.md` as required by the existing gate.

```sh
uv run trans-novel --help
uv run python -m unittest tests.pipeline.test_application
uv run --no-sync python -m unittest discover -s tests
uv run python scripts/check_architecture.py
```

Use `trans-novel` or `python -m trans_novel` for Python source execution; `wenyi` is the release executable name. Do not access provider credentials or make provider requests for ordinary implementation verification.
