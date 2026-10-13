# R1 Harness and Library Boundaries

## Authority

The user resumed R1 on 2026-10-10 after compatibility cleanup, using a new Rust-ecosystem design. Earlier Python foundation-library emulation requirements remain withdrawn. Source admission, init and read-only status must pass real production-path evidence; the five actual target runs remain separate acceptance.

A library difference is not a defect. Use maintained Rust libraries and their supported standard behavior. An owned adapter requires a concrete current business requirement, a demonstrated failure of an existing artifact, or a specific safety defect. Synthetic tests authored to imitate a Python library do not establish any of those grounds.

The immutable R0 fixture and original Python production are unchanged. Their accepted content, accounting, source bindings and persisted business formats are evidence; Python parser internals and incidental spelling are not targets.

## Retained business contracts

- Exact accepted source/target text, scalar coordinates, slot/rich identities and actual stored hashes.
- Current state/input/source/policy versions and source-bound read-only observation.
- Real configuration fields, ordered model choices, quality policy, output presence/null clearing and asset origins.
- Exclusive init creation/refusal/force and existing root CLI argument/help/version behavior.
- Bounded untrusted input, no external resource resolution, no source/state writes during observation and no provider requests.

Do not reconstruct old model-input fingerprints merely because their saved values are displayed. Do not generate Unicode tables, YAML constructors or numeric serializers to imitate a runtime version.

## Library ownership

| Operation | Mechanism | Owned work |
|---|---|---|
| YAML | `serde-saphyr`, deserialize-only | Public configuration validation and presence/origins; no PyYAML event graph, scalar/merge/tag/coercion profile |
| ZIP | `zip` | Explicit resource/allocation budgets, decoded safe identities, actual member accounting and source stability; no CP437 table or metadata byte overlays |
| URI/IRI | `fluent-uri` | EPUB-local member mapping and confinement; no urllib normalization/error emulation |
| JSON | `serde_json` | Concrete persisted projections and required stable ordering; no Python float repr engine |
| Unicode | Rust string/character operations and normal library normalization where business labels require it | No frozen Python classification/decimal database or version pin solely for language parity |
| XML/HTML/encoding | Selected libxml2/Expat/ICU boundary | Owned safe values, lifecycle, external-resource denial and real source envelopes; no replacement formatter |
| CSS syntax/properties | `lightningcss`, standard typed AST and printer | Source-only semantic projection; no Python grammar or spelling target |
| CSS selectors | Servo `selectors`, neutral source-tree adapter | Standard matching; the user approved the library bridge instead of patching a matcher's unimplemented branches |
| SQLite | `rusqlite`, immutable read-only open | Actual wire/schema/integrity/count observation; no write, recovery or invented zero counts |

ZIP construction metadata has an explicit 64 MiB safety budget, separate from the 10,000-member, 512 MiB-member and 4 GiB-expanded-data budgets. Bounds must precede allocations. These are resource policies, not promises to reproduce Python ZIP acceptance. Unsupported/malformed inputs need no tolerance adapter.

Source CSS is interpreted only for actual rich-text requirements, not browser layout or output themes. The user authorized a separate styled projection for subtree cancellation: original slot/raw rich identities and bytes remain unchanged; derived semantic IDs express different actual text footprints. Conditional alternatives compare active source emphasis scopes, not incidental numeric/token spelling. Relevant ambiguity is an explicit error, never a guessed style or no-op.

## Harness

Reuse the original private `admit_fixture`, `FixtureWorkspace` and bounded/reaping `run_wenyi` support. No new product testing flags, oracle program, server, wrapper or JSON-output command.

Retain tests of real observable behavior:

- Reopen the actual immutable R0 source/state/SQLite and verify exact content/identity and no file changes.
- Run actual production CLI help/version/argument behavior and completed app init/config operations in temporary directories.
- Exercise archive bounds/corruption/source replacement, parser resource denial/lifetimes and real package loading.
- Exercise current state errors, locks/sidecars/recovery indications without mutation.
- Invoke actual `init/status` through the bounded process helper. Check config-error precedence, exclusive creation/force, current and pending work, missing runs, active locks, sidecars, recovery/migration and malformed/unsafe source exit mapping.
- In read-only workflows snapshot directory membership, type/symlink target, permissions, file size, mtime and content; ignore atime. Existing paths are never used to authorize writes.
- Rebind stored block paths/fingerprints, exact slot text/coordinates and rich object/link geometry to the actual command source, not just saved hashes or historical `source_path`.
- Exercise styled semantics through real admission, including inherited/explicit resets, tail ownership, split emphasis footprints and sibling restoration while raw source evidence is unchanged.

The user confirmed native Windows filesystem semantics for observation on 2026-10-13. Keep production identity handles: Windows rejected whole-run directory renames while child artifact handles were open in actual CI. Retain populated-run rename/disappearance scenarios on Unix, and exercise empty-run directory replacement and disappearance on every target to verify that changed directory identity or membership discards even an invalid snapshot as `ObservationUnavailable`. File mutation, lock contention and full file-conservation checks remain required on Windows. Do not emulate Unix rename behavior or remove production identity evidence to make a test operation possible.

Delete tests whose only acceptance criterion is the spelling, tolerance, table version or internal behavior of a Python foundation library. Do not re-pin those tests to a different library's incidental behavior.

### Local execution and portability

Use a verified `WENYI_NATIVE_CACHE` and CMake 4.1.3. Build scripts do not provision or download sources. `cargo xtask native provision --cache DIR` is the explicit development operation. If the CMake executable belongs to an ephemeral tool environment, keep that environment alive for the complete Cargo invocation rather than recording a stale cache path.

```sh
cargo xtask architecture
cargo xtask changelog
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-targets --all-features --locked
```

The Rust architecture gate uses Cargo metadata for normal/dev/build/target edges and parsed Rust syntax for owned budgets, forbidden generated code/wildcard exports, neutral-source dependency direction and the exact unsafe boundary. Type declarations and impls are aggregated across files; there is no baseline bypass. Changelog checks inspect the actual working change or an explicit `--base REF --head REF` range. Retained Python architecture, Ruff and offline unittest gates still apply independently.

`.github/workflows/rust.yml` supports `workflow_dispatch` and push on the single user-authorized `r1-acceptance-20261012` branch. The narrow push trigger is necessary because GitHub returned 404 for dispatch while this workflow was absent from the default branch; it does not enable other branch pushes or change remote main. Every row specifies the actual Rust target; the native builder rejects HOST/TARGET mismatch. Repository process tests exercise the production binary before release building. Uploaded `native-build.json` and notices are build evidence, not release/cross-platform acceptance without successful runs.

## Reporting and remaining work

Report actual commands/scenarios and exercised targets only. Native source/build review and host-only checks are not five-platform proof. The repository workflow provisions verified pinned sources and runs the same native source/CSS/status tests on Windows x64, Linux x64/arm64 and macOS x64/arm64. The user-authorized local Rust release status check read the designated EPUB without changing its bytes and diagnosed unsupported saved policy 3 instead of silently accepting it as policy 5. The user authorized classified local commits and subsequently authorized the independent remote acceptance branch, actual five-platform CI, fixes and reruns. Remote main changes, releases and R2 remain unauthorized; queued or partial CI does not establish R1 acceptance.

### Recorded execution evidence

On 2026-10-13, [GitHub Actions run 38419090024](https://github.com/turygo/wenyi/actions/runs/38419090024) completed successfully for code commit `fed11c0bce27a34870e00790e0191012b0e6a2ff` on the independent `r1-acceptance-20261012` branch. Windows x64/MSVC, Linux x64/ARM64 and macOS x64/ARM64 each provisioned verified pinned sources, passed architecture/changelog/format/Clippy, ran locked workspace tests including real EPUB init/status processes, built the release executable and uploaded native provenance and notices.

All five uploaded bundles were downloaded and inspected. Executable formats and architectures matched their target triples; each bundle contained schema-1 `native-build.json`, five static-library hash/size records and all five required notice files. Corresponding notice bytes matched across targets. The Windows source-lock digest records the actual CRLF checkout bytes; it matches Git's CRLF checkout projection, while the other four match the LF source-lock digest. No hash normalization or dependency adapter was added.

The downloaded macOS ARM64 release binary executed `--version`, `--help` and `init` successfully with an owned configuration path under `target/r1-acceptance/fed11c0/`. Other targets' production CLI scenarios were exercised by their repository process tests, not by running those foreign binaries on the local host. The designated user EPUB was not uploaded or reused as CI test data. This completes the R1 five-target mechanism gate, not R13 packaged translation/release acceptance or user authorization for R2.
