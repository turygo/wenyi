# Independent EPUB translation and rendering contract

## Decision

Source character lengths and source XML text slots do not determine Chinese formatting
boundaries. The previous weighted redistribution could put a footnote inside a word or move
bold emphasis onto unrelated Chinese. Clearing Chinese italic slots redistributed text again.
Rendering and verification shared that assumption, so structural success concealed semantic errors.

A complete source block is the translation unit. Inline markup and BR elements are evidence
and objects within that block, not translation boundaries. Original coordinates remain schema 4;
`RichSource` and `RichTarget` have independent version 1 contracts. Translation policy is 4.
The run bookkeeping schema remains 4 because its lifecycle fields did not change.

The analyst annotates the exact accepted Chinese target using trusted mark and object IDs.
Chinese runs can change order and split emphasis into multiple ranges. Chinese italics remain
on their corresponding phrases. Source CSS evidence distinguishes semantic emphasis from an
initial letter decoration, which is omitted in Chinese. Page anchors retain their IDs and move
to the nearest Chinese sentence end from the analyst's semantic position, choosing the later
end on a tie. Quotes and brackets following terminal punctuation belong to that end. Code
and URL literals are protected; Chinese runs, mark scopes and other objects do not move.
The final assembly applies this deterministic rule again after local text edits. Note references
remain indivisible objects.

Text edits use explicit target offsets. Local punctuation decisions inspect complete Chinese
context, with code and URL literals protected. Terminology replacements retain their existing
mark context; replacements crossing different contexts or an object are skipped and recorded.
No production operation assigns formatted Chinese by source character weight.

## Ownership and boundaries

- `ingest.epub` extracts trusted source coordinates, inline identities, and atomic objects.
- `epub.richtext` validates independent data contracts; `epub.richtext_edits` owns local edits.
- `assemble.epub.richtext_sources` verifies complete source coverage for explicit migration;
  `richtext_styles` enriches source evidence through the existing stylesheet reader.
- `agents.richtext_annotator` uses the existing analyst router and protocol retry boundary.
- Pipeline nodes determine when accepted text requires annotation. `Application` constructs
  agents, binds usage persistence, and coordinates the independent migration copy.
- Rendering rebuilds translated inline contents only. Verification reads actual output prose,
  scopes, object signatures and identities independently before checking the unchanged outer DOM.

No dependency is added, and no package or capability dependency direction changes.

## Persistence and recovery

Old slot values are never automatically accepted as semantic annotations. `tools reannotate
INPUT --state-copy DIRECTORY` clones a closed original run and establishes provenance before
work. Only completed source-bound policy-3 or policy-4 EPUB runs are admitted. Existing Chinese
characters, including whitespace and punctuation, are frozen by an ordered target digest.

At most four annotation batches are in flight. Only the main thread writes checkpoints, in
submission order. On failure it saves valid results from already issued independent requests,
stops scheduling, and raises the first failure. A later invocation validates provenance, source
inventories and the frozen target digest and reuses accepted annotations. Policy advances only
after all required targets and their persisted digest validate. Original state remains intact.

Decoration classification accepts standalone first-letter words and opening quotes when the
source has explicit added enlargement and bold evidence. Target runs deterministically omit
these decoration scopes while retaining original anchor identities as empty nodes. Explicit
migration can reuse paid annotation when the only verified source-evidence change is the
reclassification from style to decoration; any other change still requires annotation.

Explicit migration rebases succeeded content-node fingerprints to the accepted current
contract and records both original and updated fingerprints with an append-only audit.
It does not change node statuses or paid content. Normal resume then retains accepted work;
later configuration changes still invalidate the appropriate dependencies normally.

Output-only assembly retains paid upstream work and uses the saved presentation and layout
profile. Source-preserving or unchanged machine-readable text uses the original DOM.

## Rendering exceptions and proof

Source mark attributes, hrefs and atom contents come from the original archive. Repeated
marks retain `id` and `name` only once. Source bilingual copies remain original text and keep
an independent proof; blocks containing BR use complete block pairing where appropriate.

An empty identity record is bookkeeping when all of its marks also belong to actual text
or an atom. Rendering skips that redundant wrapper and gives the original identity to the
actual node, preserving internal-reference multiplicity. Standalone empty source identities
remain nodes. Reference-graph equality is still mandatory; no graph projection exemption
is introduced for repeated nonempty internal-link wrappers.

XML comments and processing instructions are opaque objects. A negative final locator step
encodes their index in all children; positive steps continue to count element children only.
Their original tails remain immutable. A necessary following unformatted target run uses an
XHTML span with exactly `data-tn-richtext="text"`; the independent verifier recognizes only
that exact generated wrapper and still proves its text and scope.

Structural validation proves exact text, trusted references, scopes, object conservation and
archive preservation. It does not prove the model's semantic judgment. The two reported
paragraphs have explicit regression fixtures for correct Chinese bold ranges and note placement.

## Rejected approaches and acceptance

Proportional cuts, forced nonempty target slots, clearing Chinese italic text, and transferring
old source offsets through a whole-text diff cannot establish semantic ownership. Rendering
raw model-supplied HTML would permit fabricated IDs and links. Neither approach is retained.

Acceptance requires exact old/new target digest equality, unknown/missing/duplicate identity
rejection, source coverage proof, local edit regressions, bilingual and theme preservation,
annotation failure/recovery tests, full offline tests, Ruff and the architecture gate.
