# Rust Migration Design

**Status: R0 accepted; R1 production-path and five-target verification completed on 2026-10-13, awaiting user module acceptance.** Earlier library-parity requirements remain superseded. On 2026-10-12 the user authorized read-only status on the designated EPUB, classified local commits and actual five-platform CI on a new remote acceptance branch, including fixes and reruns. [Recorded execution and artifact evidence](rust-r1-harness-design.md#recorded-execution-evidence) binds the verified code commit. Remote main changes, releases and R2 remain unauthorized.

## Purpose and fixed decisions

The reference Python baseline is fixed at `61835587606b5cb81cdcd81eec8427fd4f638df4`, which was `HEAD` during design. R0 implementation commits do not move that reference. Historical Rust transcripts and native remnants are not implementation or equivalence evidence. Preserve user books, state, results, and any local native remnants. Do not open or copy user state during migration design or fixture creation.

Foundation libraries are not migration compatibility targets. Use maintained Rust libraries and standard behavior; do not reproduce Python ZIP/YAML/CSS/URI/Unicode/JSON internals, incidental spelling or malformed-input tolerance. An adapter requires a concrete current business requirement, demonstrated existing-artifact failure or specific safety defect. An old design paragraph or a new synthetic parity test is not that evidence. This rule applies to every module, including earlier work; remove unsupported compatibility rather than preserving it behind aliases.

The migration is EPUB-first, but the priority order does not reduce final parity: FB2, TXT, Markdown, all aliases, generated outputs, benchmark, governance, release, and cutover remain required. The first usable capability is `tools assemble` over a completed, current-policy Python paid EPUB state. It must create independently verified mono and bilingual EPUB variants, in either bilingual order, preserve exact Chinese and usage, and make no new model calls. General theme is not certified in this module; its acceptance belongs to R7.

Each module requires independent complete acceptance and explicit approval before the next. Rust becomes sole production only after R13 and user cutover approval; Python remains runnable meanwhile. No runtime Python wrapper, vendored upstream Git tree or repository secrets. Preserve business workflows and existing paid artifacts, not a foundation library's behavior. Resumed R1 follows the new bounded producer/consumer and Harness contracts.

Rust status is strictly read-only: it may inspect state and report that recovery or migration is needed, but MUST NOT perform status-triggered directory, migration, or SQLite side effects. This is an explicit exception to prior status behavior.

Preserve run-state schema 4, run-input schema 2, EPUB source-slot schema 4, translation policy 5, RichSource v2, admitted RichTarget v2/v3, usage schema 2 and existing SQLite tables. Preserve each format's absence/default/unknown-field rules and admitted historical migrations, not arbitrary obsolete state. Do not bump schema or silently accept stale annotations to make migration easier. Preserve command error categories and exit statuses; incidental terminal wording/layout is not the acceptance oracle.

## Ownership and dependency direction

Materialize owners only with live consumers; empty crates are not delivered. Keep dependencies directed as follows:

| Owner | Owns / allowed dependencies |
|---|---|
| `wenyi-core` | Validated semantic types, Unicode and digest rules, rich-text contracts, pure policy; no workspace dependency or I/O. |
| `wenyi-book` | Bounded archive/source/XML/CSS, ingestion, rendering, independent verification and publication; depends on core. |
| `wenyi-llm` | Blocking physical attempts, centralized routing/retry, usage and telemetry; depends on core. |
| `wenyi-store` | Wire admission, locking, journal/WAL/SQLite and bookkeeping; depends on core. |
| `wenyi-engine` | Typed stages, planning and coordinator; depends on core, book, llm, store. |
| `wenyi-app` | Strict config, sole service construction and typed facade; depends on core, book, llm, store, engine. |
| `wenyi-benchmark` | Production-facade consumer; depends on core, book, llm, app. No private store writes or second translation pipeline. |
| `wenyi-cli` | Native arguments and terminal UX; depends on core, app, benchmark. |
| `xtask` | Development tasks only; not a runtime owner. |

Short owner names below refer to these packages; paths use `crates/wenyi-<owner>/`. Materialize crates only when real consumers exist. The workspace uses edition 2024, a pinned Rust 1.96.0 toolchain and committed `Cargo.lock`; this proposal adds no dependency.

Required internal edges: `book.source` is neutral; CSS uses source, never the reverse. Ingest, render and verify use source/CSS; **verify never imports render or its expected-tree/projection builders**. Publish uses render/verify. Engine planning uses semantic snapshots; agents use core/llm, stages use core/book/agents, and execution alone accesses store. Store codecs/recovery do not call engine/LLM. LLM transport performs one physical attempt; routing alone drives retry/fallback. Config validates but does not construct services. Benchmark calls typed app operations, not private engine/store APIs.

| Interface boundary | Required producer, consumer and failure contract |
|---|---|
| Source and paid target | Book produces source-bound semantic identities; core validates scalar coordinates and exact targets; engine/render consume them. No parser nodes in core, invented IDs, accepted-text normalization or source-length formatting redistribution. |
| Model result | LLM preserves provider/protocol/accounting errors; stage outputs are concrete types, not dynamic artifact dictionaries. No default-empty result for required work; transport hidden retries disabled. |
| Accounting | Workers emit typed usage events. Coordinator durably records them and acknowledges before completion/retry/fallback. Local storage failure rejects waiting acknowledgements and stops scheduling, never retries a provider. |
| Content commit | Coordinator owns mutable book, terminology/context and non-Send/non-Sync session. At most four workers receive immutable snapshots. Content commits follow submission order with bounded pending work. Scoped revisions exclude unrelated accounting/lifecycle writes. |
| Publication | Independently verify actual bytes of every variant, persist proof, then sequentially replace and record physical receipts. A later failure cannot claim rollback or multi-file atomicity. |

All model calls run on workers, including preparation, annotation and maintenance; the coordinator must service accounting acknowledgements instead of joining a waiting worker. Keep typed identity/readiness, protocol/accounting, persistence, verification/publication and interruption errors. Persisted business hash projections and scalar coordinates remain; generic Python float/Unicode/digest-profile emulation does not. No async cutover, generic workflow framework, universal shared context or unused public API.

Rust signatures are deliberately not invented in this rhythm document. Before each implementation assignment, main must fix concrete producer/consumer types, errors, transitions, file ownership and tests for that bounded slice. A missing shared contract blocks the affected slice; it is not delegated as “use your judgment.”

## Delivery modules and acceptance

All criteria below are **planned**, not existing Rust results. Every module retains earlier accepted regressions. A module is a user confirmation checkpoint, not a single giant worker assignment: split its implementation into bounded slices with fixed interfaces, and commit independently completed capabilities separately.

UT means owner-local tests, including private `#[cfg(test)]` contracts. Integration means real application/component behavior with real files. E2E means an actual production binary launched by repository tests. Private support is infrastructure reused by these tests, not a fourth standalone executable.

### R0 — Reference baseline and minimum Harness

R0's approved implementation and acceptance contract is [rust-r0-spec.md](rust-r0-spec.md). The user accepted R0 on 2026-10-08. It has the qualified `structured-epub-balanced-v1` snapshot, reference inventories and real native help/version/process Harness. R1 specification/design review and development authorization were completed through Asktool on 2026-10-10.

**Scope/exit:** Capture the command/options and supported-state inventory, then admit one synthetic completed structured EPUB snapshot. Run actual Python `Application.run_all` with supported production model-role references and injected client/transport; close/reopen JSON, SQLite, source and outputs; bind hashes, inputs/fingerprints and reference environment to the baseline commit. Freeze mono and both bilingual orders with explicit `override_theme: null`, preserving exact targets and usage without new model calls for the additional assembly. Do not stamp succeeded nodes or use user state. Snapshot admission failure blocks R2, not unrelated R1 work; any incomplete-R0 transition still requires explicit user approval.

**UT/integration:** Replay the selected reference cases; qualify the actual full-flow snapshot exporter inside repository unittest/private support. Existing `TestEpubStage2.test_exhausted_polish_protocol_still_publishes_verified_epub` shows an unstamped `run_all` path, not an already-frozen snapshot.

**E2E/support:** Establish only the minimal Rust binary help/version, `--config`/`-c` argument parsing and argument-error process tests, fixture admission and cleanup support needed by R1. “Config” in R0 means native argument handling only; strict YAML loading, model/output validation and real init/status remain R1. This is infrastructure acceptance, not translated-book delivery. Do not advertise unimplemented commands or create empty production scaffolding, a full fixture port or a universal AST/YAML study; grow Harness with each real capability.

### R1 — Native EPUB admission, init and read-only status

The R1 product decisions retain read-only recovery/migration diagnosis (exit1), NAV HTML admission and no-readable-spine rejection (exit2). After compatibility cleanup the user resumed R1 with Rust ecosystem libraries, approved Servo standard matching and independent style semantic projection for nested cancellation. Actual production-path verification and five-target execution completed on 2026-10-13; user module acceptance remains a separate checkpoint. Remote CI runs only under explicit user authorization.

**Depends on:** Approved R0 checkpoint. **Scope/exit:** Bounded ZIP/path/resource admission, selected XML/HTML/encoding, source/CSS/Unicode identities, strict YAML/model/output presence and current manifest/SQLite inspection. Real `init/status` work; status performs no writes, model calls, migration or recovery.

**UT:** Book parser/serialization/encoding/safety fixtures, core coordinates/digests and store state admission. Selected native family is libxml2 + ICU and Expat: fixed source/hash/patch/license declarations, verified external cache, no build-script downloading or system XML requirement. Remnants are not equivalence proof. Qualified Cargo-bundled mechanisms are preferred where they meet the contract.

**Integration/E2E:** Real config/status facades and binary invocations against isolated files, including malformed input, absent/corrupt state and recovery-needed outcomes; compare paid files before/after.

**Private support/exit:** Real archive/state fixtures and read-only observation; independently reviewed native lifetimes/source/build inputs, no raw handles outside the bounded FFI module. Parser/library differences alone never trigger adapters or parity regressions. Use repository workflows to prove actual product and safety behavior before downstream assignment.

Expose native portability risk here, not for the first time at R13: each of the five supported targets must build/link the selected native boundary and exercise a bounded parser/encoding fixture through repository tests plus EPUB status. This is a mechanism gate, not full five-platform feature/release acceptance. Record actually exercised targets; a macOS-only result cannot certify this gate. R13 later reruns the complete packaged workflows.

### R2 — First usable EPUB: offline paid-state assembly

**Depends on:** R1 and the admitted R0 snapshot. **Scope/exit:** Actual `tools assemble` from completed current-policy Python state, with explicit `override_theme: null`. Generate mono/bilingual EPUB in both orders, preserving exact paid Chinese/usage, rich scopes/objects, source copies, navigation, resources and metadata. No model call or readiness stamping. Source CSS semantics remain necessary; general layout/theme parity is R7.

**UT:** Rich rendering, independent tamper detection, source/output aliases and receipt admission.

**Integration:** Real assembly/publication reopens snapshots and outputs. Inject deterministic second replacement/fsync failure at a private publication I/O boundary, then reopen both finals, proofs and receipts. This proves partial publication, not process-level fault injection.

**E2E/support:** Actual binary offline assembly, unchanged paid chapter/usage hashes and verified outputs. Verification failure preserves prior finals. A failure rejected during preflight is not partial-publication proof; count a process partial-failure scenario only if it demonstrably occurs after all-candidate verification and the first replacement. Reuse admitted snapshots and private I/O fault support.

### R3 — Durable EPUB preparation

**Depends on:** R1–R2 contracts and approval. **Scope/exit:** Actual `translate --prepare`: language/classification/analysis, terminology mining/naming and SQLite, production provider protocols/routing, durable accounting and manifest-last initialization. Preserve best-effort versus required source behavior; no body/finish/publication work.

**UT/integration:** Engine preparation and LLM dialect/retry/accounting; real app flow through a local HTTP fixture, reopening manifest/analysis/SQLite/usage. Storage failure must not cause provider retries or acknowledgement deadlock.

**E2E/support:** Prepare, interrupt and restart via the binary; completed preparation is reused without new requests. Add the single bounded endpoint/request recorder and durable-state wait to existing private process support. Provider conformance begins here and completes in R4/R5/R10; no fake production provider.

### R4 — Fresh balanced EPUB translation and resume

**Depends on:** R3. **Scope/exit:** Full-block body/heading translation, committed rolling context, source-bound annotations, canonical titles/navigation, deterministic QA, **real bounded Repair**, report and verified EPUB publication. Required stages may not be skipped, mocked or returned empty. No-general-theme is the selected supported scenario, not full theme acceptance.

**UT/integration:** Rich identity contracts, actual stage graph/protocols, durably consumed issue budgets, ordered commits and accounting failure. Real app/transport/store/book flow reopens all resulting artifacts.

**E2E/support:** Fresh binary translate; terminate after an observed durable batch; resume with exact paid-target/usage preservation; endpoint-off completed resume makes zero new requests. Private durable predicates and child-process cleanup are reused. Provider response or request arrival alone is not commit proof.

### R5 — Quality presets and scoped execution

**Depends on:** R4. **Scope/exit:** Economy/balanced/quality, explicit polish flags, accepted/rejected Editor journal/audit, exhausted-protocol preservation, Repair budget recovery, `--chapter`, prepare/chapter conflict, output precedence and model/honorific/language fingerprint changes. Retain unrelated paid chapters and completed work; Repair was already real in R4.

**UT/integration:** Config belongs to app, fingerprints/quality policy to core/engine, persistence journals to store. Real-facade state matrices cover last-batch interruption, rejection/exhaustion and scoped invalidation.

**E2E/support:** Binary option/config cases, selected-chapter execution, no-call cached resume and output-only rebuild. Extend existing endpoint responses and admitted state/config fixture variants, not another server.

### R6 — EPUB maintenance

**Depends on:** R4–R5 state contracts. **Scope/exit:** Glossary list/conflicts/audit/lock/resolve, QA/report and repeated assembly. Preserve locked/nullable/legacy SQL fields, exact paid title/body and rich identities; interrupted audit recovers.

**UT/integration:** Store/glossary rules plus actual engine/app maintenance on reopened SQLite and paid chapters; writes happen only for requested mutations.

**E2E/support:** All five glossary actions, QA/report and repeat assemble via binary; observe persisted changes and absence of unexplained requests. Reuse SQL snapshots, endpoint and interruption support.

### R7 — EPUB layout and themes

**Depends on:** R1 source CSS and R2 publication contracts; earlier modules remain accepted. **Scope/exit:** General theme analysis, bounded sampling/conflicts/unknown, missing/stale/cached profile, `--reanalyze-layout`, built-in/custom CSS and bilingual styles. Output/CSS changes do not retranslate paid content; failed refresh retains the old artifact but fails the requested refresh.

**UT/integration:** CSS cascade/reset/inheritance/import safety, source-bound profiles/sampling and independent theme/source-copy tamper proof. Real app flow checks cached zero-layout-call assembly.

**E2E/support:** Themed binary translation/reassembly/refresh, reopened variants, unchanged paid content and usage attribution. Reuse analyst responses and synthetic layout fixtures. Separate actual reader views from archive proof; no private uploads or reader-appearance claim from XML assertions.

### R8 — Supported historical bookkeeping

**Depends on:** R1/R3/R4. **Scope/exit:** Admitted historical migrations, unknown fields/counters/digest profiles, WAL/journal/recovery prefix, obsolete valid output-selection v1 handling and strict v2/source errors. Current-state durability was delivered earlier; this completes historical admission, not automatic trust of stale annotations.

**UT/integration:** Store migrations and crash windows reopen real historical snapshots; compare accepted content/accounting and explicit unsupported-policy/schema failures.

**E2E/support:** Binary resume/status/assemble on each admitted static historical case. Status remains read-only. Reuse immutable source-bound fixtures and private fault support; no runtime Python oracle or invented schema acceptance.

### R9 — Independent reannotation

**Depends on:** R4/R8. **Scope/exit:** `tools reannotate --state-copy` for admitted completed source-bound policy 3/4/5 states. Original state and exact ordered Chinese digest remain unchanged; accepted annotations/provenance/fingerprints precede policy advancement.

**UT/integration:** Admission, semantic/source coverage, rebasing and interruption on a real independent copy. Changed source/text/provenance fails before calls or mutations.

**E2E/support:** Binary reannotation, interruption/restart without repeating accepted annotations, reopening both original and copy. Reuse source-bound snapshots and endpoint recorder; no in-place rewrite or production fault API.

### R10 — Remaining formats and provider conformance

**Depends on:** Stabilized EPUB pipeline. **Scope/exit:** FB2, TXT/Markdown and aliases; generated EPUB/TXT and EPUB-to-TXT; full language/quality/output and provider capability/dialect/candidate/retry behavior. Lower priority, still required for cutover; no shadow text engine.

**UT/integration:** Format safety plus provider conformance cases through the same production app and transport.

**E2E/support:** Binary format/output/resume matrix with independent reopened artifacts. Extend canonical source/protocol fixtures; every supported provider path needs conformance evidence, not a name in a catalog.

### R11 — Benchmark corpus and production runs

**Depends on:** Typed app facade and prior acceptance. **Scope/exit:** Corpus scan/build/validate, canary/full, canonical hashes/provenance and isolated candidate/chapter/replicate state; minimal/polish reuse via app, not a second translation pipeline or raw state edits.

**UT/integration:** Corpus/run identities, price snapshots and physical-usage attribution; real app generation/copy/polish and durable observation.

**E2E/support:** Binary corpus/canary/full on a synthetic corpus satisfying actual source predicates; validate real artifacts. Reuse private corpus/spec/endpoint fixtures. Unknown usage/cost stays unknown, never zero.

### R12 — Benchmark integration, evaluation and reports

**Depends on:** R11. **Scope/exit:** Integration run; evaluate prepare/finalize/validate; report build/validate. Bind literal source/target review evidence, hashes and lineage; reject tampered/stale artifacts. Missing price/usage retains source-compatible unknown costs and a provisional report with no winner, not blanket report rejection.

**UT/integration:** Validators, closed-run observation after durable receipt/accounting/synced event, quote evidence, price provenance and provisional-report behavior. Build and validate actual artifacts; do not mock their validator.

**E2E/support:** Binary integration/evaluation/report and interruption/reuse, followed by artifact validation. Extend corpus/review fixtures and private API/process support; no benchmark-to-private-store dependency.

### R13 — Five-platform release and clean cutover

**Depends on:** Every prior acceptance row. **Scope/exit:** Windows x64, Linux x64/ARM64, macOS x64/ARM64 packaged `wenyi`, embedded assets/notices and proven native provenance. Verify the executable without Python, source checkout or native source/cache. Document ordinary OS runtime dependencies rather than claiming every library is static.

**UT/integration:** Packaging/provenance/asset checks and all applicable workspace gates. Remove remaining Python fixture generators, governance/release tooling and runtime dependencies only at accepted cutover.

**E2E/support:** The same repository process tests target each actual release binary for EPUB translation/resume/theme/reannotation and representative benchmark. Private release support selects the packaged executable; no new smoke program. Full command/config/history/failure ledger and user cutover approval precede removal of superseded Python code/tests/package/lock/hooks/CI/docs. Never delete user data.

## Harness, fixtures, and fault evidence

Harness is repository `cargo test` UT, behavioral integration, actual-binary E2E, and private test support—not a new service or framework. Crate-local `#[cfg(test)]` tests may inspect internals; crate integration tests exercise real facades, ZIP/XML/SQLite/JSON outputs, and provider-only injected transport; CLI tests launch the actual binary. Private fixtures, protocol endpoint, child process and bounded durable-state waits live under CLI test support, not production crates. Other crates keep local private support. Share static fixture data, not global generic helpers or production-to-dev edges. Reuse one bounded local HTTP fixture implementation for success, protocol/provider failure and delayed completion.

Private store/LLM tests inject exact WAL/journal/fsync/replace failure boundaries and reopen real files. Coordinator integration checks acknowledgement failure, deadlock/boundedness and ordering. Process E2E terminates only at an observed durable predicate (committed manifest, cleared required journal and persisted accounting); a later dependent request is usable only when that same predicate is established. Provider completion/request arrival alone is not commit proof. Low-level crash windows and deterministic partial publication use private tests of real production code; ordinary publication and verification rejection use actual-process E2E. Do not introduce product test flags, environment fault backdoors, fake successful nodes or manifest edits to manufacture full-flow acceptance.

**Planned Harness layout** (not implemented):

```text
fixtures/migration/{epub,state,protocol,benchmark}/
crates/wenyi-<owner>/src/...            owner-local #[cfg(test)] contracts
crates/wenyi-<owner>/tests/...          real-facade/component integration
crates/wenyi-cli/tests/
  cli_surface.rs                      R0/R1: config, admission, read-only status
  epub_assembly.rs                     R2/R6: paid-state assembly/publication
  preparation.rs                      R3: durable preparation
  epub_translation.rs                 R4: fresh translation and process recovery
  quality_scope.rs                    R5: quality, fingerprints, chapter scope
  maintenance.rs                      R6: glossary, QA and report
  layout.rs                           R7: themes/profile/output reuse
  history.rs                          R8: historical admission
  reannotation.rs                     R9: exact independent copy
  formats.rs                          R10: remaining format/provider workflows
  benchmark.rs                        R11/R12: real benchmark artifacts
  release.rs                          R13: packaged binary selection
  support/{mod,fixtures,protocol,process}.rs
```

Private support owns checked-in response selection, request validation, child processes, OS-assigned ports and bounded durable-state waits. Fail on unexpected requests; verify source-bound operation/IDs and reported usage, not a canned response echo. Give every wait/process/socket a deadline and cleanup path. Use isolated state and explicit conditions, not fixed sleeps/ports. Release tests select an actual packaged executable through test-runner configuration; the product receives no test-only interface.

**Forbidden:** standalone Harness projects, temporary Cargo projects/source/binaries/HTTP servers/driver scripts, a separate smoke program after equivalent E2E, production imports of test helpers, and permanent Python-oracle CI. Temporary directories contain test inputs, state and outputs only. The shared endpoint is part of repository tests, not another server executable. Transport injection proves component behavior; only production-transport process E2E proves CLI/HTTP/accounting integration. Neither proves live-provider availability or translation quality.

Fixture admission records baseline commit, reference package/environment versions, source/state/protocol hashes and expected semantic projections. Compare output text, scope, identities, resources and references independently; whole ZIP-byte equality is not the EPUB oracle. Export initial snapshots through repository Python unittest/private support. After cutover use static provenance-bound snapshots and Rust tests/`xtask` generation only. Add fixtures with their owning module rather than porting the entire corpus upfront; no private books are committed or uploaded.

Early safety matrix: ZIP identity/traversal/encryption/types/allocation/expanded bounds/CRC; external XML denial/entity limits and real encoding/source representation; source CSS only as needed by actual rich consumers; scalar versus UTF-8 offsets and exact Chinese without normalization; real slot/rich/hash/source/schema/policy bindings. No frozen runtime-property database, serializer spelling oracle or claim of exactly-once remote billing. Structural checks do not prove model semantics or reader appearance.

## Complete command and option scope

The existing root command is `trans-novel [--config/-c PATH] [--version]` (no-args help). Final Rust command UX is native and binary is `wenyi`; table records existing exposed behavior to inventory and map, not an assertion that Rust signatures exist. Root, `translate`, and `tools` help were exercised in the design round; other spellings below are static-source inventory only.

| Existing command/options | Owner and delivery coverage |
|---|---|
| `init [--force]` | `cli`/`app`; R1. Creates default config; refuses overwrite without force. |
| `translate INPUT [--chapter N] [--format epub|txt] [--out PATH] [--quality economy|balanced|quality] [--source-language CODE] [--honorifics keep_style|normalize|drop] [--polish/--no-polish] [--mono/--no-mono] [--bilingual/--no-bilingual] [--prepare]` | R3–R5, R10. `N` minimum 0; `--prepare` conflicts with `--chapter`. Omitted output values resolve CLI/config/saved/default per field. |
| `resume INPUT [--format epub|txt]` | R4, R8, R10. Shared translation implementation; no per-invocation chapter/quality/language/polish/output flags. |
| `status INPUT` | R1, R6, R8. Strict read-only in Rust, including no migration or recovery side effects. |
| `tools glossary INPUT [ACTION=list|conflicts|audit|lock|resolve] [ARG1] [ARG2]` | R6. Preserve five actions, arguments and explicit mutation semantics. |
| `tools assemble INPUT [--out PATH] [--format epub|txt] [--mono/--no-mono] [--bilingual/--no-bilingual] [--reanalyze-layout]` | R2, R7, R10. First usable capability is EPUB no-theme assembly from paid state; theme refresh is R7. |
| `tools reannotate INPUT --state-copy DIR` | R9. Independent copy only. |
| `tools qa INPUT`; `tools report INPUT` | R6. Existing run required; report writes/refreshes report. |
| `tools benchmark corpus scan BOOK_SPEC --out DIR`; `corpus build BOOK_SPEC SELECTION --out DIR`; `corpus validate CORPUS_DIR` | R11. |
| `tools benchmark run canary BOOK_SPEC CANDIDATES --out DIR [--book-id ID]`; `run full BOOK_SPEC CANDIDATES --out DIR` | R11. |
| `tools benchmark integration run CORPUS_DIR BOOK_SPEC CANDIDATES INTEGRATION_SPEC --out DIR` | R12. |
| `tools benchmark evaluate prepare RUN_DIR REVIEW_SPEC --out DIR`; `finalize REVIEW_DIR RESULTS_DIR`; `validate REVIEW_DIR` | R12. |
| `tools benchmark report build RUN_DIR REVIEW_DIR PRICE.yaml --out DIR [--integration INTEGRATION.json]`; `report validate REPORT_DIR` | R12. |

Strict config inventory: `llm.models` roles `translator`, `analyst`, `editor`, `fast`, each a nonempty unique ordered list; optional per-model OpenRouter routing; optional custom OpenAI-compatible `base_url`/`api_key_env`; `quality`; `output` with mono default true, bilingual enabled/`target_first`, optional `override_theme.styles`, and `bilingual_styles` default builtin. Unknown fields fail. Credentials are environment-only. Internal defaults (not public knobs): batch 1800, segment 1200 characters, protocol retry 2, context window 6, prescan concurrency 4. Quality expansion makes economy/balanced non-polish and quality polish; explicit polish flags override. Existing provider registry names are DeepSeek, OpenCode Go, Bailian, OpenAI, OpenRouter, generic OpenAI-compatible, Ollama and vLLM. Exact model catalog and all per-operation physical routing must be captured from baseline sources during R0; do not infer valid model names or claim transport conformance from names alone.

Output admission distinguishes omission from explicit false/null and retains asset origins. Resolve each field CLI -> explicit config -> saved selection -> default; disabling both variants keeps the source's mono-only normalization. Explicit `override_theme: null` clears a saved general theme without translation/layout calls. Relative assets resolve from their config origin; missing unused theme assets must not break TXT export. These are behavior matrices, not tests of incidental serialized wording.

## Execution, approval, and evidence

### One module's execution loop

1. Main fixes the approved module's scope, real command/scenarios, owner, producer/consumer types, error/state/I/O contract, affected files and Harness criteria. Reuse existing evidence; ask only about material product/risk choices or a changed shared design.
2. Write one bounded implementation assignment per independent slice. Parallelize at most two writers inside the same approved module, with distinct files and fixed interfaces. Executors read global/delegation instructions, skip competing integration checks, and never invent or redelegate shared design.
3. Implement a real path and its behavior tests together. Run focused owner tests while iterating; after writers settle, main integrates and runs module process E2E and applicable workspace gates once. Do not run the whole workspace for every tiny edit.
4. Review changed high-risk contracts: native lifetimes, paid-state/accounting, worker ordering, output verification and publication. Fix actual findings and re-review those corrections, not the entire subsystem repeatedly.
5. Report actual commands/scenarios/results, reopened artifacts, request/usage observations, preserved regressions, remaining unaccepted scope and independently scoped commit hashes. **Stop and obtain user confirmation before starting the next module.**

No implementation starts from this proposal. Design confirmation authorizes only the next agreed module, not unrestricted autonomous completion of R0–R13. A module cannot pass on component tests alone where its criterion is a production workflow. Partial delivery, a library catalog or test count is not acceptance.

### Research and blocker exits

- A factual investigation names the exact missing fact and blocked acceptance criterion. Once source evidence permits implementation, stop researching and implement the adapter plus regression. Do not enumerate unrelated YAML/AST/parser corners.
- A selected native mechanism must pass the owned coordinates/encoding/serialization/safety tests; if it cannot, report the failing case and bounded decision through asktool. A missing native fact blocks only actual dependents, not every store/LLM activity within the approved module.
- Resolve interface/schema/dependency or acceptance conflicts before affected work proceeds. Do not hide them with defaults, retry additions, relaxed tests, policy bumps or a stronger executor.
- If 30 minutes of work produces no movement on a named criterion, report the actual change/result, blocker and next bounded action. This is a communication check, not a new timer/advisor/progress-file framework.
- While awaiting the user's next-module confirmation, do not open the next module's implementation. Keep progress in the conversation/task list; no persistent agent progress files.

### Verification commands and gates

These are **planned** commands, used only after their targets exist. A module's process tests select the real `wenyi` binary through private test support; Cargo test binaries are not replacement production applications.

```sh
cargo test --locked -p wenyi-book
cargo test --locked -p wenyi-cli --test epub_assembly
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
cargo xtask architecture
cargo xtask changelog
cargo run --locked -p wenyi-cli --bin wenyi -- --help
cargo build --release --locked -p wenyi-cli --bin wenyi
```

Early modules run applicable actual targets; nonexistent future targets are not claimed green. Native all-feature gates require preprovisioned verified inputs. Build/dependency and private-module gates grow with the live workspace, using Cargo metadata and parsed Rust imports/re-exports including cfg/target/dev/build edges, not a speculative general-purpose compiler project. Reject cycles, hidden generated application logic, wildcard exports and capability escapes.

Owned Rust/test/tooling uses warning/hard line budgets: files 500/800, functions 80/120, impls 250/400, with 400-line hard aggregate per type/capability. No new baseline bypass; splitting files cannot replace ownership review. API/dependency/state/concurrency/publication changes, three-owner changes or >300-line additions need main's owner/invariant/interface/error/I/O/edge/file/verification review. Necessary XML FFI is confined to `crates/wenyi-book/src/source/xml_ffi.rs`; its safe API and pinned native build/license inputs receive independent review. The sole native build exception is `crates/wenyi-book/build.rs`, consuming verified pinned external sources/owned glue, never downloading or hiding generated application logic.

Every module preserves earlier accepted behavior. Final cutover requires the entire command/config/history/failure ledger, all applicable gates and five-platform packaged E2E. Code/tests/tooling changes update CHANGELOG; independently completed capabilities are committed separately by main. No Python deletion or user-data mutation before explicit accepted cutover.

The existing Python baseline checks actually exercised for this design are:

- `uv run --no-sync python -m unittest tests.llm.test_usage_persistence tests.pipeline.state.test_checkpoint tests.assemble.epub.verification.test_api` — 36 tests passed.
- `uv run --no-sync python -m unittest tests.assemble.epub.test_publication.TestEpubStage2.test_exhausted_polish_protocol_still_publishes_verified_epub` — 1 test passed. It exercises real `Application.run_all` publication and is a reference for planned R0 snapshot creation, not proof that the qualifying snapshot exists.
- Root, `translate`, `tools`, and `tools benchmark` help invocations passed. The inventory beyond exercised help is static-source evidence.

No full Python suite was run. These checks do not establish Rust behavior, complete Python acceptance, live-provider quality/cost, reader appearance or platform packaging.

## Durable baseline references

| Baseline evidence | What it establishes / limit |
|---|---|
| `trans_novel/cli/{app,tools}.py`, `trans_novel/benchmark/cli.py` | Source command/options inventory; exercised help is recorded above. |
| `trans_novel/pipeline/application.py`, `planning/fingerprints.py`, `state/{models,store}.py` | Stage order, fingerprint/state admission and existing status/migration side effects. |
| `tests/fixtures/{books,fake_llm}.py`, `tests/pipeline/test_application.py` | Synthetic fixtures and real in-process Application behavior, not actual binary/HTTP transport proof. |
| `tests/llm/test_usage_persistence.py`, `tests/pipeline/state/test_checkpoint.py` | Existing paid-accounting and checkpoint crash-window reference cases. |
| `tests/assemble/epub/test_publication.py`, `tests/assemble/epub/verification/` | Unstamped full-flow reference, independent tamper checks and partial-publication failure cases. Direct-writer prerequisite stamping is not full-flow evidence. |
| `tests/pipeline/nodes/test_reannotate.py`, `tests/pipeline/test_reannotation_fingerprints.py` | Exact-target independent-copy and recovery/rebase reference behavior. |
| `tests/benchmark/integration/`, `tests/benchmark/report/test_report.py` | Isolated runs/recovery/artifact lineage and provisional missing-price behavior; mocked validators are not E2E proof. |
| `scripts/smoke_theme_assembly.py`, `.github/workflows/{tests,build}.yml` | Existing Python binary/HTTP and five-platform release patterns; port their behavior into Rust repository tests, not a new smoke executable. |

Semantic requirements remain in [richtext-contract.md](richtext-contract.md), [epub-theme-design.md](epub-theme-design.md), [benchmark-guide.md](benchmark-guide.md) and [architecture-governance.md](architecture-governance.md). These are not inherited Rust acceptance. Where old prose conflicts with current source/tests—for example benchmark policy 2 versus current policy 5—resolve the factual baseline from source and regressions. Product changes require user decisions; never silently narrow compatibility to whichever paragraph is convenient.
