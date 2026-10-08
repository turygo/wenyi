# R0 — Reference Baseline and Minimum Harness Specification

## Status and authority

**Status: R0 implemented and exercised; user acceptance pending. R1 and later modules are not authorized.**

This specification defines R0 under [Rust Migration Design](rust-migration-design.md). The user confirmed two decisions through Asktool:

- R0 delivers native help/version and `--config`/`-c` argument handling only. Strict YAML loading, model-role validation and output-presence semantics remain in R1.
- The first admitted snapshot uses one representative structured EPUB, not the simplest text-only sample. Its reference outputs cover mono, bilingual target-first and bilingual source-first.

On 2026-10-07, the user reviewed this specification, approved it and authorized the next session to complete R0 development for user acceptance. This authorizes R0 only, not R1 or later modules. After implementation, R0 acceptance and the next-module approval remain separate checkpoints.

This document is the approved R0 implementation and acceptance contract. The migration design retains authority over cross-module ownership, compatibility, safety and cutover. A conflict must be resolved explicitly before affected implementation; neither document permits silently narrowing acceptance. Updating a fixture or expected result to accommodate a Rust mismatch requires review of the underlying Python evidence, not automatic re-recording.

## Outcome and non-goals

R0 delivers three independently reviewable capabilities:

| Deliverable | Required result | What it does not establish |
|---|---|---|
| Reference inventory | Complete command/options and supported-state ledger bound to the Python baseline, with evidence classified | Rust command, configuration or historical-state implementation |
| Admitted EPUB snapshot | An unstamped, completed, current-policy Python run, reopened and bound to exact source/content/accounting evidence | Live-provider availability, translation quality, general themes or Rust assembly |
| Minimum Rust Harness | A real `wenyi` help/version binary, actual-process tests and private static-fixture admission/cleanup support | EPUB parsing, state admission, init/status, translation or publication |

R0 makes no Python production behavior changes. Do not create empty core, book, store, LLM, engine, app or benchmark crates. Do not add placeholder subcommands, a test server, a general workflow framework, a universal fixture abstraction or a Python-backed Rust command. Do not read, copy, mutate or commit user books, configuration, state or results.

The snapshot is an offline model-accounted reference, not a claim that money was spent. “Paid state” means the exact accepted target and accounting artifacts that downstream assembly must preserve.

## Fixed baseline

The reference source commit is:

```text
61835587606b5cb81cdcd81eec8427fd4f638df4
```

This commit must not move merely because R0 documentation, tests or fixtures are committed later. Record both the reference commit and exporter/support hashes. The exporter is a test overlay over the baseline implementation, not evidence that its own code existed at that commit.

Before generation, verify the reference production Python tree, packaged assets, `pyproject.toml` and `uv.lock` against that commit. A changed production baseline blocks generation until the discrepancy is resolved. Do not reset the checkout or overwrite user changes. If historical source must be used, use an isolated baseline checkout with the explicit test overlay; do not reconstruct an approximate baseline from the current implementation.

Record actual Python/package versions, OS/architecture, Python Unicode database version, SQLite version, lxml/libxml2 versions, working-directory convention and relevant deterministic environment settings. Record installed dependency versions and the lockfile hash; a matching lockfile alone is not proof that the installed environment matches it. Credentials and unrelated environment variables are excluded.

## Reference inventory

Create two static JSON ledgers under `fixtures/migration/`. Their private metadata schema is version 1; it is not a production-state schema.

### Command and option ledger

`surface-inventory.json` covers every row in the migration design's complete command/option scope, including all benchmark leaves, positional arguments, aliases, allowed values, defaults, conflicts, precedence and error/exit categories. Include strict configuration fields and their omission/false/null/unknown-field distinctions. Do not replace an omitted field with its effective default in the inventory.

Each case records a stable case ID, command/config input, expected semantic outcome, owning migration module, baseline source location and one of these evidence classes:

- `source`: inspected baseline implementation; no runtime claim.
- `reference_test`: an identified existing regression exercised during R0.
- `reference_process`: an actual Python command exercised during R0, with arguments and observed exit status.

Unexercised exit statuses remain explicitly unobserved. Terminal colors, wrapping and incidental wording are not equality oracles. Command spellings, argument rules, error categories and exit statuses are compatibility requirements. Run help for each command group/leaf as needed to qualify the inventory, not just root help.

Existing Python status/recovery side effects must be recorded as baseline behavior alongside the approved Rust read-only exception. Do not accidentally require Rust to reproduce status-triggered migration, recovery, SQLite writes or directory creation.

### State and admission ledger

`state-inventory.json` records wire owners, source locations, defaults, unknown-field policy, admitted versions, migration paths, rejection conditions and owning migration module. Required families are:

- Run-state schema 4, run-input schema 2 and translation policy 5.
- EPUB source-slot schema 4, RichSource v2 and admitted RichTarget v2/v3.
- Source identity, node fingerprints, chapter/progress statuses and annotation provenance.
- Usage schema 2, accounting dimensions, WAL admission and unknown-accounting semantics.
- SQLite `glossary` and `term_conflicts`, column types, nullable/default/locked fields and ancillary SQLite bookkeeping.
- Output-selection v2 and source-validated disposal of obsolete valid v1 selections.
- Supported historical bookkeeping, checkpoint/note journals, recovery and explicit reannotation eligibility.

Historical migration tests do not authorize arbitrary old EPUB sources or stale annotations. A test helper's manually constructed historical state is inventory evidence, not the completed R0 snapshot. Inventory the supported history now; generate historical Rust fixtures only in their owning modules.

## Structured source and model boundary

### Source fixture

Use one small synthetic English EPUB with two logical chapters and two spine XHTML resources. Reuse `write_phase9_epub` and existing synthetic-book patterns where appropriate; bounded source extensions belong in private test support, not ingest production code.

The admitted book contains NAV and NCX labels, canonical headings, a cross-chapter link, a footnote reference/body/backlink, one valid image, local CSS, a list, a blockquote, a table, semantic bold/italic scopes and an inline linebreak. Include a supplementary-plane scalar and a combining sequence in controlled prose/targets so downstream code cannot equate scalar coordinates with UTF-8 bytes or normalize accepted text unnoticed. All resources and references are synthetic and local.

Keep prose bounded; multi-batch interruption and archive-safety matrices are not R0 criteria. Exact generated source bytes are frozen after qualification. ZIP timestamps/order may be controlled by the source generator, but whole-output ZIP equality is not an assembly oracle.

### Configuration and responses

Use `quality: balanced`, explicit English source language, Chinese target language, mono enabled, bilingual enabled and explicit `override_theme: null`. Keep the built-in bilingual stylesheet. The initial full flow uses `target_first`; the additional reference assembly uses `source_first`.

Use the supported `deepseek/deepseek-chat` reference for all four roles, each as a one-element ordered list. This keeps reference routing simple without inventing a fake production provider. Record the exact validated configuration, explicit-field presence and expanded internal policy.

Inject an `AgentRouter` with a real `OpenAICompatibleTransport` and a private SDK-client stub, then pass that client through `Application(config, client=...)`. A private recording adapter validates the `complete` call's agent/operation/model metadata before forwarding to the real transport; the SDK stub selects responses by the actual model/messages, not a shared arrival-order queue. The registry and transport must share the router's usage tracker. Reuse the injection patterns in `tests/llm/test_routing.py` and `tests/llm/test_usage.py`. Routing, response extraction, usage tracking, agents, planning, persistence, rendering and verification remain production code. Only the external model boundary is stubbed; no credentials or network are needed.

The response support validates agent/operation, model identity and source-bound request contents before selecting deterministic responses. It must validate requested source/target identities for rich annotation. Match concurrent requests by semantic request identity, not thread arrival order. Exhausted, duplicated or unexpected requests fail the fixture run; do not return generic `{}`, empty required results or a catch-all successful answer.

Responses contain valid target text/annotations and explicit known token accounting. Accounting constants describe synthetic responses, not live-provider billing. Record response definitions and their hashes. Wall-clock timing and generated call IDs remain recorded raw evidence but are not deterministic replay expectations. R0 does not introduce an HTTP endpoint; production-transport process coverage starts in R3.

## Snapshot lifecycle and admission

The exporter lives in repository unittest/private support and follows this sequence:

1. Validate baseline inputs and create an isolated workspace containing synthetic source, configuration, state and outputs only.
2. Use a stable relative source path such as `source/book.epub`, a relative state root and relative output paths. Confine any temporary working-directory change to the test, restore it on all exits and wait for all application workers to finish first.
3. Invoke real `Application.run_all(..., out_format="epub")`. Do not use `Application.run` plus manually stamped finish prerequisites. Do not replace nodes, planner, runner, persistence, renderer or verifier with successful mocks.
4. Close application-owned resources. Reopen the JSON files, SQLite database, original source ZIP and every produced EPUB using fresh handles. Qualification reads the actual persisted state, not just `run_all`'s returned dictionary.
5. Prove readiness from the real selected workflow: required nodes completed, chapters committed, source-bound annotations admitted, report persisted, outputs independently verified and publication receipts present. Record legitimate policy-driven skips separately; do not require optional balanced polish or theme analysis to run.
6. Confirm no pending checkpoint/accounting/note-migration journal, no running node or unfinished required repair. Inspect SQLite integrity and persisted rows after close; a database dependent on an uncollected WAL is not admitted.
7. Freeze the initial completed run. Copy it into a separate isolated workspace and invoke production `Application.assemble` with source-first presentation and a model boundary that rejects every call. Reopen that output and its proof/receipt. The frozen original must remain byte-unchanged.
8. Compare exact ordered targets, rich identities/annotations, upstream content-node fingerprints and raw usage before/after the additional assembly. Output-selection, assembly fingerprints, reports, events and publication receipts may change on the copy only. Do not demand whole-manifest equality across requested presentation changes.
9. Write reference metadata only after all checks pass. A failed run produces no admitted fixture and never replaces an existing reference bundle. Publish a newly qualified bundle only through an explicit reviewed generation action.

The normal qualification test generates/reopens a candidate in a temporary directory and checks it against the frozen semantic contract; it does not update checked-in files. First recording or later reviewed replacement uses an explicit, test-support-only generation mode on that same qualified path, not a standalone script or product flag. Reject destinations outside `fixtures/migration/` and reject overwrite unless separately requested.

Do not rewrite persisted statuses, source digests, fingerprints, targets, usage, paths or timestamps to make the snapshot portable or deterministic. Reconstruct the recorded workspace-relative layout when consuming it. Any absolute path emitted by the baseline is preserved and declared as an origin-bound field; a real downstream consumer needing relocation must receive an explicit contract in its owning module, not silently patched JSON.

## Static bundle contract

The following fixture roots are materialized by the qualified R0 recording action:

```text
fixtures/migration/
  surface-inventory.json
  state-inventory.json
  epub/structured-v1.epub
  protocol/structured-v1-responses.json
  state/structured-epub-balanced-v1/
    reference.json
    expected.json
    run/state/<actual-book-slug>/...
    outputs/<actual-reference-output-names>.epub
    assembly/source-first/...
```

`<actual-book-slug>` and output filenames come from the production run and are recorded, not fabricated names in persisted state. Keep the complete durable run artifacts; omit only transient handles/lock files and verified-empty SQLite sidecars. Record omissions explicitly. The source-first subtree contains the additional output, actual persisted verification/receipt evidence and comparison results, not a second independently translated book.

### `reference.json`: private provenance envelope

Required fields:

| Field | Contract |
|---|---|
| `schema_version`, `fixture_id` | Exactly 1 and `structured-epub-balanced-v1` |
| `baseline_commit` | Exact fixed reference commit |
| `reference_inputs` | Lockfile, production-tree/asset and exporter/support hashes; exact configuration and explicit-field presence |
| `environment` | Actual reference versions/platform/Unicode database and reproducible workspace convention |
| `wire_versions` | Observed state/input/source/rich/policy/usage versions, not assumed constants |
| `files` | Sorted unique records: fixture-root-relative path, workspace-relative destination where applicable, byte length and lowercase SHA-256 |
| `outputs` | Role `mono`, `bilingual_target_first` or `bilingual_source_first`; actual file, proof and receipt locations |
| `semantic_projection` | Path and hash of `expected.json`, plus its projection-format version |
| `generation` | Qualified test identifier, generation mode, executed checks and explicit omissions/origin-bound fields |

All paths are relative, bounded and contained within their declared root. Reject absolute paths, traversal, aliases/duplicates and symlink escapes in the fixture envelope. Metadata may describe an origin-bound persisted value as data; that value is never used as an unchecked host path.

`reference.json` does not list its own hash: avoid a circular digest. Its bytes, inventories and fixture bundle are bound by the accepted fixture commit; content hashes do not constitute a cryptographic signature. The inventories are included in the reference input hash set.

The implemented file record is `{path, size_bytes, sha256, workspace_path}`; `workspace_path` is either a contained relative destination or null for provenance-only data. Each output record is `{role, file, proof, receipt}`. The projection record is `{path, sha256, format_version}`. All artifact paths are relative to `fixtures/migration/`; the envelope is `state/structured-epub-balanced-v1/reference.json`.

The baseline emits an absolute `manifest.source_path`. It is retained verbatim in both run copies and `expected.json`, with three `{artifact, pointer, value}` records under `generation.origin_bound_fields`. Admission recomputes these declarations from the hashed raw JSON/JSONL files. The independent observer reads `source/book.epub` from the reconstructed workspace and checks the declared origin as literal data; it never opens that origin path. Replay compares the bound relative layout and source digest while excluding only this declared origin from the semantic comparison. R0 does not make the saved manifest directly relocatable for a production consumer; that contract remains with its consuming module.


### `expected.json`: projection format 1

Freeze these semantic observations without importing rendering's expected-tree builders:

- Ordered chapter/segment identities, source/rich-source digests, exact target strings, target scalar lengths, rich runs/scopes/atoms and canonical heading/navigation labels.
- Original persisted node statuses/fingerprints and effective configuration/policy; raw values remain unchanged.
- Usage schema/dimensions and accounting counters, with explicit synthetic response provenance. Store raw usage unchanged; identify elapsed-time fields separately for replay comparison.
- SQLite DDL and rows in stable key order, preserving NULL versus empty string, locked values and all other raw fields. Identify time fields separately rather than replacing them with zero.
- Source resource hashes, outer structure/attributes, navigation/reference graph, source-copy pairing and target/source order for each output variant.
- Actual reopened EPUB target text, scopes/objects/resources and verification/publication results.

Raw artifact hashes prove fixture integrity. Semantic projections prove the compared behavior. Replay may differ in ZIP bytes, timestamps, durations and call IDs only where explicitly declared nondeterministic. Target text, Unicode form, source identities, fingerprints and deterministic accounting counters have no such exemption. New nondeterministic exclusions require review.

The independent observer parses actual ZIP/XML/JSON/SQLite bytes. It must not call rendering/projection builders to construct the expected answer. Production verification is necessary but is not the only snapshot-admission assertion. Recompute raw hashes and semantic observations from reopened artifacts; metadata assertions alone do not qualify the exporter.

## Minimum native binary and private interfaces

Create one workspace member, `wenyi-cli`, exposing the real binary `wenyi`. Use edition 2024, pinned Rust 1.96.0, committed `Cargo.lock` and `#![forbid(unsafe_code)]`. Obtain dependencies through normal Cargo resolution; no custom downloader, native libraries or `build.rs` are needed in R0.

Use `clap` for native argument handling. Set the initial package version to the baseline package version 1.3.3 and derive displayed version from Cargo package metadata. Production dependencies are limited to argument handling. Private fixture admission may use `serde`, `serde_json`, `sha2` and `tempfile` as dev dependencies; exact resolved versions belong in `Cargo.lock`. No production-to-test or workspace-to-Python dependency is allowed.

Private parser contract: `config: PathBuf` with default `config.yaml`; `--config PATH` and `-c PATH`; help/version actions. Config values are opaque native paths in R0. No YAML reading or filesystem probing occurs. Do not invent an application facade or a configuration-validation command to give the option a consumer.

| Actual binary invocation | Required observable result |
|---|---|
| `wenyi --help` | Exit 0; identifies `wenyi`, help/version and both config spellings; no side effects |
| `wenyi --version` | Exit 0; identifies binary and package version; no side effects |
| `wenyi` | Root help with exit 2, matching the observed Python no-arguments category |
| `wenyi --config PATH` or `wenyi -c PATH`, without an implemented action | Root help, exit 2; never a silent successful no-op |
| `wenyi --config <missing-or-malformed-file> --help` | Exit 0; no attempt to open/validate the file |
| `wenyi -c <path-with-spaces> --version` | Exit 0; native path accepted; no file access |
| `wenyi --config` or `wenyi -c` | Missing-value argument error, exit 2 |
| Unknown option or unavailable subcommand | Argument error, exit 2; never a successful placeholder |

Only implemented actions appear as commands. R0 help may identify the module's limited scope but must not advertise functional `init`, `status`, `translate` or `tools`. Their eventual implementation remains required by the full inventory.

Private test-support contracts are concrete and local to CLI tests:

- `admit_fixture(root: &Path) -> Result<AdmittedFixture, FixtureError>` validates envelope version, fixed baseline, contained unique paths, file sizes/hashes and required roles. `AdmittedFixture` owns validated metadata and paths; it is not a production `RunStore` or a rich-text parser.
- `FixtureError` distinguishes invalid metadata, baseline mismatch, unsafe path, missing artifact and hash mismatch. No silent repair, migration, default-empty fixture or overwrite.
- `FixtureWorkspace` owns a `TempDir`, materializes the admitted workspace layout and exposes its root. Checked-in fixtures are never execution working directories for mutating commands.
- `run_wenyi(args: &[OsString], cwd: &Path, deadline: Duration) -> Result<ProcessResult, ProcessError>` launches Cargo's actual production binary path, captures status/stdout/stderr and always kills/reaps a timed-out child before releasing its workspace. No shell command strings, fixed ports, detached children or test-product flags.

Use condition-based, deadline-bounded process completion and bounded output capture. R0 does not need a socket, HTTP protocol helper or durable-state waiter. Add those only with their real consuming module. Test success, argument failure, timeout cleanup and fixture-isolation behavior through repository tests; do not add another helper executable.

## File ownership and implementation slices

| Slice | Files owned | Responsibility and boundary |
|---|---|---|
| Reference inventory and exporter | `tests/fixtures/migration_{epub,reference,provenance,boundary,responses,accounting,artifacts,run_observer,epub_observer,observation,admission,bundle,errors}.py`, `tests/pipeline/test_reference_snapshot.py`, `fixtures/migration/**` | Synthetic inputs, validated responses/accounting, real full-flow export, independent reopened state and EPUB observations, bounded admission/recording and frozen evidence; no Python production edits |
| Native CLI and Harness | Root `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`; `crates/wenyi-cli/{Cargo.toml,src/main.rs,tests/cli_surface.rs,tests/fixture_admission.rs,tests/support/{mod,fixtures,fixture_paths,fixture_workspace,process}.rs}` | Minimal real binary; envelope orchestration, filesystem/hash mechanism, isolated workspace materialization and process lifecycle have distinct private owners; no book/store/config implementation |
| Main integration | This specification, migration-design link/boundaries, `.gitignore` if build-output exclusion is needed, `CHANGELOG.md`, accepted ledgers/bundle | Shared fixture schema, errors, dependency/size review, final verification and acceptance evidence |

The shared interface is the version-1 fixture envelope and workspace layout above. CLI process work is independent of full-flow generation; fixture admission is integrated against the qualified bundle. At most two implementation writers may work concurrently, each with fixed ownership and a bounded assignment. Neither writer may invent shared contracts or run competing full-workspace checks.

Reuse existing private fixtures without adding a second production convention. Keep source generation, model response selection and reopened-artifact observation separate when their size/responsibilities warrant it; do not turn the exporter into a general migration framework. Apply Python architecture governance and the migration design's Rust/test size and dependency budgets. R0 needs no empty `xtask`; nonexistent future gates are not reported passed.

## Acceptance and verification

R0 passes only when all rows below have observed evidence. A test count alone is not acceptance.

| Criterion | Required evidence |
|---|---|
| Inventory complete | Every design command/config/state family mapped to source and an explicit evidence class; no unobserved process claim |
| Baseline bound | Exact source commit/input hashes and actual reference environment recorded |
| Real completed snapshot | Unmodified production `run_all`; actual required lifecycle statuses, source-bound content, closed/reopened durable artifacts and no pending recovery |
| Three output roles | Reopened mono and both bilingual orders; independent target/scope/object/resource/reference observations |
| No paid-content regeneration | Additional assembly rejects all model calls; frozen original plus copied targets, upstream content fingerprints and usage preserved |
| Exporter qualified | Behavioral regressions reject incomplete, mismatched-source, altered-target/accounting and invalid publication evidence; no failed-candidate publication |
| Static fixture integrity | Rust private admission accepts the qualified bundle and rejects unsafe paths, missing/corrupt artifacts, incompatible metadata and wrong baseline |
| Actual native process | All CLI rows above exercised against the built `wenyi`; isolated files unchanged and timeout child cleanup proved |
| Applicable gates | Focused/full Python checks, Rust tests/format/lint and ownership/dependency review completed; changelog updated |

Exporter regressions manipulate candidates in temporary directories only. They must exercise real admission/observation behavior, not assert source text, copied constants or mocked validator results. At least one exact target change that leaves a syntactically valid JSON file must be detected semantically, and absent required publication evidence must fail despite an existing EPUB file. A failed recording must leave an existing accepted bundle unchanged.

Selected existing reference regressions are usage persistence, checkpoint recovery, state migration/output-selection and independent EPUB verification/publication. Preserve their existing behavioral expectations; do not re-pin incidental wording or wiring tests to manufacture parity.

Implementation verification commands (all targets now exist):

```sh
uv run --no-sync python -m unittest tests.pipeline.test_reference_snapshot
uv run --no-sync python -m unittest tests.llm.test_usage_persistence tests.pipeline.state.test_checkpoint tests.pipeline.state.test_migration tests.pipeline.state.test_output_selection tests.assemble.epub.verification.test_api
uv run --no-sync python -m unittest tests.assemble.epub.test_publication.TestEpubStage2.test_exhausted_polish_protocol_still_publishes_verified_epub
uv run --no-sync python -m unittest discover -s tests
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_architecture.py
git diff --name-only -z 61835587606b5cb81cdcd81eec8427fd4f638df4 -- | uv run python scripts/check_changelog.py --null-from-stdin
cargo test --locked -p wenyi-cli --test cli_surface --test fixture_admission
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo metadata --locked --format-version 1
```

Actual-process E2E supplies the native smoke evidence; no separate temporary binary, validation script or HTTP server is required. The complete Python suite remains the baseline regression gate while Python is retained. Static Rust fixture tests must not launch Python, and no permanent Python-oracle CI is introduced.

The changelog script requires changed filenames; running it with no input is not a gate. Its existing code-path classification covers Python/test/tooling paths, not new Cargo sources. Main must also review that the R0 changelog covers native changes; extending Rust governance tooling is not implied by a successful Python gate.

## Failure, acceptance and next-module transition

Any incomplete lifecycle, source/content/accounting mismatch, invalid proof, unexpected model call or origin/portability ambiguity prevents snapshot admission. Preserve the failed diagnostic and original fixture; do not stamp readiness, normalize targets, relax the schema or change production code merely to make the reference pass.

A snapshot blocker blocks R2 assembly, not independent R1 mechanism work. It does not turn a partial R0 snapshot into acceptance: before any R1 work, report the exact failed criterion and obtain explicit user approval for that limited transition, leaving snapshot admission outstanding. The normal route is complete R0 acceptance, scoped commits, then explicit R1 approval.

Acceptance reporting names actually executed commands, admitted fixture ID/hash and commit, reopened artifacts, request/accounting observations, preserved regressions and unaccepted scope. R0 certifies only the environment actually exercised; it does not certify five-platform native or release portability. Do not start R1 or delete Python after delivering R0.

## Exercised R0 implementation evidence

Recording used `WENYI_R0_RECORD=fixtures/migration uv run --no-sync python -m unittest tests.pipeline.test_reference_snapshot` on the qualified production path. Before any implementation commit, the repository-pinned Ruff 0.12.7 hook found one tuple-to-union `isinstance` correction and one blank-line removal that the installed Ruff 0.15.21 had accepted. The user explicitly authorized replacement of the uncommitted initial bundle rather than weakening its source binding or bypassing the hook. The original envelope (`84eab5f775728e7abc1018cfe3fd7b4ec739ae52b4c69416099a57b74fb6d856`) and complete bundle were retained for comparison; the reviewed replacement was regenerated through that same qualification command, never patched in place.

Independent old/new comparison passed the complete stable semantic projection, exact ordered targets and rich annotations, content fingerprints, deterministic accounting and byte-identical source EPUB and SDK protocol. Environment and all reference input fields except the two corrected support hashes and their aggregate exporter hash remained identical. The explicitly declared origin/time/ZIP observations may differ. Normal qualification subsequently reopened and matched the replacement without recording mode or skipped tests.

| Observation | Exercised result |
|---|---|
| Reference inventory | 170 command/config cases; help for all 28 registered groups/leaves and 36 actual Python processes; 19 state families, 22 wire models and 45 selected existing reference regressions |
| Fixed inputs/environment | Production Python, packaged assets, `pyproject.toml` and `uv.lock` match `61835587606b5cb81cdcd81eec8427fd4f638df4`; Python 3.12.7, package 1.3.3, macOS arm64, Unicode 15.0.0, SQLite 3.46.0, lxml 6.1.1/libxml2 2.14.6; actual installed distributions recorded, including the unrelated `quickjs-ng` installation |
| Completed source/state | Two chapters, two spine XHTML resources and 17 accepted ordered targets; 11 succeeded nodes plus legitimate layout/two-polish skips; fresh JSON/SQLite/ZIP/XML handles, SQLite integrity `ok`, no pending recovery journals |
| Accounting | 13 validated synthetic SDK responses through the real router/transport; 481 prompt + 169 completion = 650 tokens; zero failed attempts and fallbacks; no live-provider or monetary-spend claim |
| Publication/preservation | Mono, bilingual target-first and source-first independently reopened with passed proofs/physical receipts; additional production assembly made zero model calls and preserved raw usage, chapter files, all upstream content-node fingerprints and the frozen original |
| Exporter regressions | 12 focused tests passed without skips, including a second independent full-flow semantic/protocol replay; altered valid targets/accounting/origins, incomplete lifecycle, pending journals, missing publication, copied fingerprint changes and failed staged recording are rejected |
| Native process/admission | 11 Rust tests passed against the actual `wenyi` binary and qualified static bundle; help/version/argument categories, no filesystem effects, owned timeout kill/reap, output bounds, unsafe/aliased/missing/corrupt metadata and materialization revalidation exercised |
| Applicable gates | Full offline Python suite: 1,107 tests passed after replacement. Whole-repository Ruff check/format, repository-pinned pre-commit hooks, Python architecture gate, locked Rust focused/workspace tests, rustfmt, Clippy with `-D warnings`, Cargo metadata and the changelog filename gate passed |

The admitted envelope binds 38 artifacts and retains three declared absolute-origin fields. Its SHA-256 is `1a7571fe2984a764c13f8b27e4efad074c3c5e3b320c673a346ca02a0895789d`; `expected.json` is `0cc5fc8d4d07dc911625114ea51fd22221a29e66da013dc31ac972fae495898c`; the frozen source EPUB is `c4b572020d268eb64fd02dd99eeed7408bd4904ffb8fd1e95b2c27a37fac2a10`. The envelope does not hash itself. The implementation commits bind these bytes; user acceptance remains separate.

Ownership review found no production Python edits, new production Python dependency edges, baseline bypass, extra workspace members, owned `build.rs` or owned unsafe code. Native production dependencies are only `clap 4.6.7 -> clap_builder 4.6.7 -> anstyle 1.0.14 / clap_lex 1.1.1`; serialization/hashing/tempfile dependencies are dev-only. Largest new Python file/function/class: 483/99/314 lines; largest owned Rust file: 414 lines. Existing and sub-hard-limit architecture warnings remain warnings, not exemptions or failed gates.

The two read-only risk reviews identified a missing copied-content fingerprint comparison and an incorrect stdout-only no-argument help assertion. The fingerprint defect was reproduced with a real copied manifest before correction and a passing regression afterward; native process E2E observes semantic help across both output streams.

No Rust EPUB parsing, YAML/configuration loading, init/status, translation, assembly, general-theme acceptance, live-provider quality/cost, reader appearance or five-platform release portability is certified by R0. Do not start R1 or remove Python while awaiting user acceptance.


## Evidence available during specification drafting

The baseline commit was checked and matches the fixed reference above. Existing full-flow and support source was inspected, including `Application.run_all`/`assemble`, `RunStore`, state/config/usage/SQLite contracts and the unstamped publication regression. The publication regression currently uses `fake/...` roles and checks file existence plus production verification; it is not an admitted structured snapshot.

Two production entry-point cases were exercised during this specification round:

- `uv run --no-sync trans-novel`: root help, exit 2.
- `uv run --no-sync trans-novel --config`: missing option argument, exit 2.

No full suite, exporter, static bundle, Cargo target or Rust behavior was exercised in this round. Earlier checks recorded in the migration design remain earlier evidence, not new R0 acceptance.
