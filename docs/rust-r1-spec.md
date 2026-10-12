# R1 — Native EPUB Admission, Init and Read-Only Status

## Authority and current status

The user resumed R1 implementation on 2026-10-10 after removing unnecessary compatibility from all prior migration work. The new design uses Rust ecosystem libraries and actual business consumers, not the withdrawn parity-driven plan. R1 remains unaccepted until production-path verification and the five actual target executions are reported; this does not authorize R2.

Use maintained Rust libraries and their supported standards. A difference from a Python library is not a bug. Add an adapter only for a concrete current business requirement, demonstrated failure of an existing artifact, or specific safety defect. Do not reproduce parser internals, incidental formatting, malformed-input tolerance or a runtime's Unicode database. Do not use newly authored parity tests or the old specification to justify those layers.

The fixed R0 fixture remains immutable. Its exact accepted targets/accounting, source identities, persisted versions and original process harness remain business evidence. Original Python production remains runnable and unchanged. Implementation verification uses isolated repository fixtures. On 2026-10-12 the user explicitly authorized read-only status on the designated Thinking, Fast and Slow EPUB and classified local commits under the current R1 boundary. This does not authorize writes to user books/config/state/results, credential access, provider requests, pushes or remote CI.

## R1 product boundary

R1's production boundary is EPUB source admission, configuration/init and read-only status, not translation/publication. Only init and status become new CLI commands. Existing root help/version/config-path behavior remains. Do not advertise translate/resume/tools, add testing commands/flags or wrap Python at runtime.

The user's prior product decisions remain:

- NAV may use the selected HTML parser; container/OPF/NCX use the selected XML parser.
- No readable spine is an input-validation error, exit 2.
- Status diagnoses required recovery/migration without executing it, exit 1.
- Five actual targets remain an execution gate, not a claim inferred from a workflow or host-only run. Remote CI is user-triggered.

## Owners

| Owner | Responsibilities | Workspace dependencies |
|---|---|---|
| core | Scalar coordinates, source/slot/rich business formats and model policy | None; no I/O/native nodes |
| book | Archive/package/source parsing, encoding and source evidence | core |
| store | Actual state wire admission and non-creating read-only observation | core |
| app | Public config/template, output presence/origins, sole composition | core/book/store |
| CLI | Arguments, terminal projection and exit mapping | core/app |
| xtask | Development provisioning/governance | No runtime consumers |

Only owned book source/xml_ffi.rs may contain unsafe Rust; other owned production/tests forbid it. Native C glue and build/source provenance require independent review. Keep ownership and file/function/impl budgets from the migration design; no baseline bypass or generated application logic.

## Source admission and library boundaries

Use zip for ordinary ZIP parsing/names/extras/decompression/CRC, fluent-uri for URI/IRI syntax, selected libxml2/Expat/ICU for XML/HTML/encoding, and native Rust character/string operations. Do not add CP437 tables, UnicodePath byte masks, ZIP64 metadata virtualization, LZMA overrides or urllib character blacklists merely to reproduce Python behavior.

Retain concrete untrusted-input policies: at most 10,000 members, 512 MiB expanded member, 4 GiB expanded aggregate and 64 MiB construction metadata. Validate bounds before allocations and count actual returned bytes while reading; verify all admitted members, including unused resources. Reject unsafe decoded names, canonical file/directory collisions, encrypted/special entries and corrupt/budget-exceeding data. Names are case-sensitive and never host-extracted or Unicode-normalized. Hash and parse the same source version; detect replacement/changes.

Required package semantics are actual EPUB resources/title/spine order and safe local reference mapping. Missing optional declarations remain data; required broken package/spine fails. External references never authorize fetching. Preserve owned source trees, namespaces/attributes/text/tails/opaque nodes and scalar/element locators required by real artifacts. Native parser defaults own syntax, encoding aliases, recovery and serialization details; no hand-written replacement formatter. External resources/entities/DTDs/XInclude must not bypass the loader boundary or safety limits.

CSS syntax and typed property/shorthand parsing use `lightningcss`; standard selector matching uses Servo `selectors` over the existing source tree. The user chose this bridge after the bundled matcher was found to panic on standard `:has()`/nth-of selectors. Use the library printer and standard parsing, never a handwritten selector engine or Python profile. Native source slots, raw rich IDs and bytes remain geometric authority; a separate validated styled projection supplies emphasis, structural roles and decoration requirements. The user authorized derived semantic IDs where nested style cancellation has different text footprints. Conditional alternatives must agree on actual source semantic footprints; unknown relevant states/values, matched animation and unsupported cascade/scope semantics are explicit errors. No invented viewport, browser layout, external fetch, nonstandard selectors, language/direction tables, charset alias database or token-spelling emulation.

`admit_epub` owns one archive admission, package/spine order, NAV/NCX inventory, fragments, notes, source blocks/slots/raw rich evidence, logical chapters and styled evidence before `archive.finish`. Resources and logical chapters reference one segment inventory; original bytes and owned parsed trees are retained. `Application::status` loads configuration first, admits the actual input, observes the named run, verifies claimed stored locators against fresh source evidence and brackets the source through observation. Configuration-relative assets resolve from the configuration origin; saved assets retain absolute/builtin references and recorded directory provenance without opening CSS. The facade exposes typed source/state/config/output data; the CLI projects terminal status and exit categories.

## Configuration and init

Use serde-saphyr's normal deserialization, then validate the public schema. No custom YAML event arena, PyYAML scalar/tag/alias/merge profile, sexagesimal/Unicode-digit constructor, binary/string/numeric coercion or duplicate-key emulation.

Keep real product rules: missing/non-file config returns defaults without writes; empty/null root loads defaults; malformed/nonmapping/unknown fields fail. Ordered nonempty unique model lists validate provider/model/thinking policy. Custom endpoint constraints remain distinct from standard providers; no credentials are read. Quality is economy/balanced/quality with existing policy values.

Output fields preserve omission independently from false and explicit theme null. Resolve CLI, explicit config, saved selection and defaults per field; normalize both variants disabled only at resolution. Preserve selected asset origin/home/relative paths without eagerly opening unused assets.

Saved selections use the existing production artifact format: custom asset references are absolute and builtin references remain identifiers. A saved relative reference is invalid state rather than an invented working-directory origin. This does not restrict normal relative/home paths in configuration and does not open unused assets.

Init uses the selected config path, creates parent directories and uses exclusive creation without force. Existing refusal and I/O failures are exit 1. Force replaces malformed existing config without parsing it first. No book/store is constructed on this path. The embedded template must load through the actual native config parser.

## Existing artifacts and status

The actual command input is the sole source-file authority. Historical manifest.source_path is metadata, not a file to open or rewrite. Select the named run from its source title; do not scan other runs. Preserve current state schema4/input2/source-slot4/policy5, RichSource2, admitted RichTarget2/3 and usage2 business records, exact paid text/SQL values and source SHA bindings.

Stored fingerprint strings may be observed without reconstructing their former input-generation algorithm. Compact slot/rich projections and integrity hashes are concrete persisted formats; generic Python JSON/float repr and frozen Unicode classification are not.

Retain typed current/missing/recovery/migration/unavailable/invalid outcomes; precedence invalid, unavailable, migration, recovery, current. Current incomplete pending/skipped/failed work is not recovery. Active writer means unavailable, not crashed. Config errors precede source lookup. Config/arguments and malformed/unsafe input exit2; missing/unreadable source exit1; noncurrent observations exit1.

Inspection never creates directories/locks/SQLite sidecars, rewrites defaults/status/targets, performs DDL/checkpoint, deletes journals, recovers or migrates. Open an existing observation lock without writing bytes; otherwise bracket membership/identity/content/size/mtime, including missing/new locks and actual source. Changes invalidate observation; no retry or atomic-snapshot claim. Ignore unavoidable atime only.

Precedence applies only to a stable observation. Source or state changes invalidate that snapshot, discard observed content/counts/selections and report `ObservationUnavailable`; an earlier invalid-state diagnosis from the discarded view is not retained as a definitive outcome.

Use immutable read-only SQLite after sidecar/stability checks. Nonempty WAL or recovery dependence yields diagnosis with no invented counts. Preserve actual nullable/locked values and explicit corruption/version/source errors. Keep recovery-record/source/hash validation for real state, not serializer/runtime emulation or transformations.

## Verification and acceptance

Use repository UT and real production workflows/private support over isolated data. Reopen the immutable R0 fixture and verify content/identity/no writes. Retain original CLI process tests and child cleanup. Delete parity-only test programs, spelling/table/tolerance assertions and placeholder capabilities; never re-pin them to another library.

Run affected owner tests and actual CLI/app init/status paths, then locked workspace tests, format/lint/architecture and retained Python regression checks. Report only observed results. Helper/component counts do not establish completed admission/status. Local macOS arm64 results do not satisfy the five-target mechanism gate; the workflow is user-triggered. The Rust release executable was built and exercised on the user-designated EPUB: source SHA256 matched the saved identity and remained unchanged, while status exited 1 because the saved translation policy is 3 rather than current policy 5. The user confirmed this diagnosis is within the current R1 boundary and authorized classified local commits. Push, remote CI execution and R2 remain unauthorized.
