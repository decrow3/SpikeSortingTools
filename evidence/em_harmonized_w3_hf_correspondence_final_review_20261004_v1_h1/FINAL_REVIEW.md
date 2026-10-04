# Independent H/F correspondence accounting review

## Verdict

**SCOPED GO WITH ADDITIVE NUMERIC CORRECTION; NO AUTOMATIC PROGRESSION.**

The executed implementation matches the frozen mutual-unique event-accounting
method.  The saved tables close exhaustively and support one narrow statement:
within the radius-2 mutual-unique, both-assigned coincidence subset, H and F
partition substantial event mass differently.  This is limited by low overall
coverage and substantial ambiguity and is not evidence of biological identity,
purity, or correct/incorrect merges or splits.

The immutable H5 packet, its preserved incomplete first output, and its final
tables remain unchanged.  `SUMMARY.json:18` contains a negligible transcription
error: exact-sample F assigned coverage is
`223114 / 489575 = 0.4557299698718276`, as serialized in
`results/ARM_EVENT_ACCOUNTING.csv`, not `0.4557299698532772`.  This correction
does not change any classification or interpretation.

Per-label `mutual_unique_assigned_events` and `coverage_fraction` must be read as
**both-assigned** pair coverage.  The global per-label sums are 236,800, whereas
the arm table's source-assigned mutual totals also include 175 F-assigned/H-negative
pairs and 107 F-negative/H-assigned pairs.  Negative-label events are preserved
and closed in the arm and pair tables; they are intentionally absent from the
assigned-label sparse partition table.

## Method and provenance review

- Packet integrity: every member matches `MANIFEST.sha256`
  (`02c6ccebfa74057b798e3486fe18654f8f4de7d9ea85c408c5d97e52b64848ef`),
  and `COMPLETE.json` is strictly newer than all other packet members.
- Frozen contract: `SPEC.json:22-35` fixes the 29,999.7591667 Hz clock, global
  `[239998073,250197991)`, local `[0,10199918)`, inclusive radius 2, radius-0
  sensitivity, row identity, duplicates, negative labels, mutual uniqueness,
  and prohibited greedy/filtering operations.  The runner refuses a non-frozen
  status and verifies every input hash before opening arrays
  (`source/hf_correspondence.py:228-238`).
- Clock, columns, and labels: both arms require `times_samples`, `labels`,
  `sampling_frequency`, `channels`, and `geom`; sampling rate, aligned shapes,
  and half-open local support are checked at lines 243-260.  Stable sorting
  carries labels and original rows together at lines 33-49.  Domain assignment
  uses `(global_origin + local_sample) / fs` at lines 263-272.
- Matching: inclusive `searchsorted(..., side="left/right")` counts every
  opposite-arm row at lines 53-58.  Mutual pairs require exactly one neighbour
  on both sides and assert reciprocal symmetry at lines 61-79.  No greedy
  choice, label-aware pairing, reciprocal-best filter, or double counting is
  present.
- Fixtures: the six preserved fixtures contain explicit known answers for
  equality, inclusive +/-2, radius-3 exclusion, duplicate timestamps,
  many-to-one ambiguity, crop edges, stable label/row reordering, negative
  labels, and exhaustive status closure (`tests/test_hf_correspondence.py:6-63`).
  Direct source review confirms the expectations are independent of labels and
  that duplicated or many-to-one events cannot become mutual-unique.
- Domains and boundaries: matching occurs before domain accounting
  (`source/hf_correspondence.py:276-286`).  The full 3x3 transition table retains
  two radius-2 `negative_excursion` F to `catalogue_outside_remainder` H pairs;
  both are assigned and remain visible in sparse cells `(219,88)` and `(885,421)`.
  Common-domain mass is 45,537 negative, 154,833 flat, and 36,428 remainder,
  plus two cross-domain pairs, totaling 236,800.
- Full sparse and per-label outputs: the table has 14,115 unique, positive,
  lexically ordered F-label x H-label cells.  No top-N or threshold appears in
  construction (`source/hf_correspondence.py:110-129`); every assigned-label
  combination with nonzero radius-2 mutual mass is serialized.  Per-label tables
  contain 972 x 4 F rows and 461 x 4 H rows with unique label/domain keys.
- Independent serialized closure: all 16 arm rows close population, status, and
  assigned/negative partitions.  Radius-2 and radius-0 mutual counts are
  symmetric (237,092 and 223,402).  Assignment categories and all nine domain
  transitions independently sum to each radius total.  Sparse global/common/
  cross-domain masses close; per-label denominators sum to 489,575 F and 535,579
  H assigned events; both per-label mutual sums equal 236,800.
- Directional dispersion: direct regrouping of all sparse cells reproduces
  F-to-H outside-largest mass 97,272/236,800 = 0.410777027027027 and H-to-F
  124,592/236,800 = 0.5261486486486486.  All common-domain totals and directional
  largest/outside partitions reproduce exactly.  These are source-label-
  directional partition summaries, not reciprocal identity assignments.
- Exact-sample sensitivity: radius 0 yields 223,402 mutual pairs, with total
  coverage 0.41513732485970195 H and 0.455045789336462 F.  The lower exact-sample
  coverage does not remove the radius-2 coverage/ambiguity limitation.
- Preserved failure: the first completed output lacking explicit arm assigned-
  coverage columns is retained.  All common arm columns and every other result
  table are byte-identical; the final arm table only adds assigned/negative
  mutual counts and assigned coverage.

## Interpretation

At radius 2, mutual-unique coverage is 44.1% of all H events and 48.3% of all F
events.  H additionally has 85,086 zero-neighbour, 63,800 one-sided-unique
nonmutual, and 152,162 multiple-neighbour events; F has 53,296, 42,126, and
158,430 respectively.  Thus the 41.1% F-to-H and 52.6% H-to-F outside-largest
cell masses support different coincidence-defined label partitioning only in
the covered both-assigned subset.  They cannot determine which partition is
better, whether labels represent the same cells, or whether any merge/split is
correct.

## At most one next test

Only if cross-window stability becomes decision-relevant: apply this unchanged,
hash-frozen accounting once to one independent representative medium window
with already-existing H/F saved outputs.  Precommit the same radii, closure
rules, and directional summaries.  It could test replication of the covered-
subset partition pattern; it still could not establish biological identity or
merge correctness.  This review does not authorize or start that test.

## Implementation checks

- Done: what ran -> source hash `a144ebd...e29870`, spec hash
  `ce3407dc...82007`, exact input bindings, run receipt, preserved failures, and
  sealed outputs inspected.
- Done: clocks/frames/columns -> exact W3 origin, half-open bounds, sampling
  frequency, required columns, stable ordering, and domain time conversion
  traced to consuming code.
- Done: matching/counting -> inclusive row-level +/-2 and radius 0, reciprocal
  one-neighbour rule, duplicate/negative preservation, and no greedy or
  filtering path verified in source and fixtures.
- Done: exhaustive outputs -> independent serialized closure over both arms,
  both radii, all assignment states, full 14,115-cell sparse table, label/domain
  rows, boundary transitions, and both directional dispersion masses.
- Done: correction -> exact-sample F assigned coverage recomputed from the
  authoritative arm table; `SUMMARY.json:18` is the only numeric discrepancy
  found and does not affect the result.
- Not done: direct H1 reopening of producer-local H/F NPZ inputs -> those
  `/home/huklaban5/...` paths are not present on H1.  Their hashes, counts,
  ordering facts, and clock are cross-bound by the run receipt and the earlier
  sealed score result, but this is provenance verification rather than an H1
  regeneration from source NPZs.
- Not done: external proof that the spec timestamp preceded every producer-side
  outcome read -> the runner enforces the frozen status and hashes, but the spec
  and result were published together.  No tuning surface or tolerance sweep was
  found.
- Can establish: internally consistent exhaustive coincidence-defined
  correspondence and material partition dispersion within the mutual-unique,
  both-assigned covered subset on this W3 run.
- Cannot establish: full-population correspondence, causal training-context
  effect, cross-window replication, biological identity/purity, or merge/split
  correctness.
