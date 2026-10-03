# Independent EPUB translation and rendering contract

## Decision

Source character lengths and source XML text slots do not determine Chinese formatting
boundaries. The previous weighted redistribution could put a footnote inside a word or move
bold emphasis onto unrelated Chinese. Clearing Chinese italic slots redistributed text again.
Rendering and verification shared that assumption, so structural success concealed semantic errors.

A complete source block is the translation unit. Inline markup and BR elements are evidence
and objects within that block, not translation boundaries. Original coordinates remain schema 4;
`RichSource` keeps version 2; ordinary targets remain version 2 and explicit zero-range
declarations or restored note objects require target version 3. Translation policy is 5.
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
work. Only completed source-bound policy-3, policy-4 or policy-5 EPUB runs are admitted. Existing Chinese
characters, including whitespace and punctuation, are frozen by an ordered target digest.

At most four annotation batches are in flight. Only the main thread writes checkpoints, in
submission order. On failure it saves valid results from already issued independent requests,
stops scheduling, and raises the first failure. A later invocation validates provenance, source
inventories and the frozen target digest and reuses accepted annotations. Policy advances only
after all required targets and their persisted digest validate. Original state remains intact.
Preservation is resolved per segment through the shared preservation contract. A source-preserved
reference chapter still requires rich annotation for its translated canonical headings; chapter-level
preservation cannot bypass those headings. Plain headings use deterministic annotation, while
headings with inline semantics or objects use the existing analyst path.

Decoration classification accepts standalone first-letter words and opening quotes when the
source has proven added relative enlargement, float or initial-letter evidence. Semantic tags,
bold-only styling and unresolved absolute font sizes do not establish decoration. Target runs deterministically omit
these decoration scopes while retaining original anchor identities as empty nodes. Explicit
migration can reuse paid annotation when the only verified source-evidence change is the
reclassification from style to decoration. A v1 upgrade also ignores deprecated CSS-evidence
serialization differences only after archive identity and complete source geometry proof;
fresh v2 semantics and required wrappers remain mandatory. V2 evidence changes still require annotation.

Explicit migration rebases succeeded content-node fingerprints to the accepted current
contract and records both original and updated fingerprints with an append-only audit.
It does not change node statuses or paid content. Normal resume then retains accepted work;
later configuration changes still invalidate the appropriate dependencies normally.

Output-only assembly retains paid upstream work and uses the saved presentation and layout
profile. Source-preserving or unchanged machine-readable text uses the original DOM.

## Rendering exceptions and proof

Exact whole-subtree bilingual proofs read the actual EPUB resource bytes with
the safe resource parser. The normalized BeautifulSoup view is used only for
text and pairing checks; its folding of whitespace-only nodes must never feed
an exact source-shape comparison. Changed source whitespace still fails proof.

Inline identity preservation is independent of optional formatting: any mark
with `id` or `name` must be represented, even with empty semantics and
`required=false`. Untranslated source punctuation may retain its identity in
an empty run at the corresponding target semantic boundary. Protocol retries
keep the source and target payload fixed and add rejection diagnostics;
interruption events identify the failed chapter and segment batch.

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

## Version 2: semantic and identity conservation

Marks carry multiple typed `semantics` roles (bold, italic, link, superscript,
subscript, literal). Immutable tag semantics remain required even when an optional
style field is empty. `required` independently preserves wrappers that reset an
inherited style; a normal-font span inside italic text must not disappear.
Chinese emphasis may reorder or split; links cannot nest or reopen one source link
entity. Rendering requires source v2 and target v2 or v3; v1 remains readable for
explicit migration.
Fresh source proof and v2 target validation decide whether a paid v1 annotation
can be reused; an incompatible annotation needs a new analyst annotation, never
a rewritten target. Bookkeeping schema 4, input schema 2 and source-slot schema 4
remain unchanged.

Independent output proof includes identity events at their approved target
positions and complete scope chains. Equal-style wrappers are not interchangeable
when their IDs belong to different Chinese text or link contexts. Empty identity
records survive text deletion; neither edits nor rendering may invent identities.

CSS selector lists are parsed into ordinary and pseudo-element branches. Ordinary
semantic evidence does not include pseudo-element declarations. A bounded cascade
computes necessary typography and resets, including inherited and inline styles;
unresolved semantic conditions fail explicitly instead of guessing.
Dynamic selectors are skipped only after a conservative static match upper bound
proves that no source node can match; possible or unknown matches still fail.
CSS matching serializes a detached DOM copy to avoid DTD-induced synthetic nodes,
and retains exact node-count and order proof against the original tree. Production
serialization uses the same detached-document boundary, preserving encoding,
declaration policy, DOCTYPE value and root siblings without automatic XHTML meta
insertion. Source first-letter decorations have an exact source-derived Chinese-language override
independent of optional themes. Original stylesheet assets and English bilingual
presentation remain unchanged; both resource and DOM verification prove the
generated override before excluding it from immutable-resource comparison.

Animation validation uses a read-only CSS projection: animation or transition declarations
are excluded from validation only when every selector branch has a proven empty match
upper bound in the actual DOM. Actual source stylesheets, selectors, other declarations
and keyframes remain intact. Possible matches and inline animations still fail explicitly.
Theme validation additionally requires a missing positive original ID or class selector,
excluding reserved `tn-` identities, so future generated nodes cannot invalidate the proof.
Root-free specificity checks retain the previous strict animation policy; no animation
execution or timeline inference is introduced.

Layout inventory evidence preserves original CSS rule text when every selector branch
is an admitted ordinary selector. The matching set is the union of those branches;
mixed pseudo-element or unsupported branches still retain only admitted ordinary
branches. Original declaration order, conditions and inline evidence remain intact.
This prevents formatting-only evidence serialization from invalidating accepted layout
profiles. Rich-text evidence retains its v2 serialization independently. Existing exact
inventory digest and complete assignment-ledger checks remain mandatory; no profile
digest is rewritten and genuine evidence changes still require layout analysis.

Punctuation normalization is idempotent and uses original-coordinate edits.
Qualified ellipsis/dash groups merge after whitespace cleanup without swallowing
an isolated sentence stop. Email and mailto labels join code and URL protection;
format-context and atom barriers still conservatively reject ambiguous edits.

Migration binds raw JSON targets before model loading and proves the same digest
after loading and persistence. Legacy slot targets and rich run targets are
validated against their respective raw authoritative text. Only explicit policy-3
migration may restore discarded XML-whitespace BR groups: source hashes, original
slot values, complete retained old groups and the exact omission must all prove
the old slicing behavior. NBSP, missing non-whitespace or unexplained fields fail.
When fresh validation rejects an existing annotation, migration preserves the authoritative
target, clears stale slot targets, and persists a source/target-bound pending record. Resume
admits incomplete legacy slot projections only when that record and the frozen migration
digest prove the exact unchanged text. Successful annotation removes the pending record.
Recovered source slots are audited in the independent copy; complete source
coverage remains mandatory in both hydration and independent verification.

Acceptance includes structure-independent role preservation, same-style identity
mutations, nested/reopened link rejection, reset inheritance, punctuation
idempotence and fragmentation equivalence, protected literals, legacy whitespace
negative cases and paid-annotation interruption/recovery. Structural proof does
not claim to establish semantic Chinese alignment; that remains the analyst's
responsibility and is assessed separately. No book-specific selector or path is
part of a production compatibility rule.


## Zero-range alignment and recovered references

A source emphasis range need not have a separate Chinese spelling. Multiple source marks
may share a meaningful Chinese range. When the meaning is implicit or merged without
a separate range, the analyst may declare an empty target run with `omission` equal to
`implicit` or `merged` and a nonempty `explanation`. Its ordered position is the Chinese
boundary, and the referenced source mark supplies the original wording and identities.
Only emphasis and required style resets are eligible; links, superscripts, subscripts,
literals, decoration and empty source marks cannot use this declaration. A mark cannot
be both omitted and applied to meaningful text. Identities remain mandatory and their
approved boundaries are independently checked. The explanation is a model judgment,
not a deterministic semantic proof.

Missing note labels are different: a verified reference can be restored as an independent
zero-text note object. A separate read-only detector requires safe unique identifiers,
matching short labels, reciprocal unambiguous links, a note-start backlink and note prose.
A backlink may target the reference itself, its preceding empty identifier or a transparent
single-reference span/sup ancestor. A terminal period is ignored only for matching numeric
labels; original labels remain intact. Without explicit note semantics, the body reference
must not also be at its block start. Class names and book-specific paths are not evidence.

Recovery proofs are attached to existing source link marks after full-resource source
inspection. Original slots, source runs, legacy note metadata and protected extraction
paths remain unchanged. A note object points to this proven reference and retains its
source mark chain; its original label is rendered separately from the immutable Chinese
text. Existing Chinese numeric labels are aligned to the link instead of restored again.
The target cannot both restore and spell the same reference. Assembly and independent
verification recheck the reciprocal relation against the original archive and verify the
actual rendered label, link, formatting and identity boundary. Unprovable references fail.

The new target contract accepts prior v2 annotations without mass invalidation. Added
source proof fields and run fields are absent from historical serialization when unused,
preserving existing source digests and pending migration proofs. Fresh source changes
invalidate only their affected annotations. All migration target-string digests remain
exactly equal, including existing superscript glyphs and punctuation.

Protocol rejection feedback reports the first changed target position, expected and actual
Unicode values, lengths and exact surrounding text. This is diagnostic evidence for the
existing bounded retry policy; no whitespace repair, acceptance relaxation or routing
fallback is introduced. The original request records remain frozen on retry.
