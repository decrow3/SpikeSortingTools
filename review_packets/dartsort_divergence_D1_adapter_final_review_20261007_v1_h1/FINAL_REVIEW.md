# H1 focused independent review — D1 saved-output adapter readiness

## Verdict

**REPAIR_REQUIRED; PRODUCTION EXECUTION REMAINS DISABLED.**

The packet is sealed correctly, the loader checks are fail-closed for the
relationships they implement, and the accepted inclusive event counter is
sound. The four frozen Kilosort neighborhood pairs, half-open clock,
three separate pair names, non-transitive interpretation, DARTsort
matching-versus-final label distinction, and execution-disable boundary are
carried forward correctly.

The packet is not yet a complete adapter for its promised earliest-divergence
diagnostic. Three load-bearing repairs are required before a bounded production
execution contract can be released.

## Required repairs

### 1. Bind Kilosort execution provenance per arm

`BINDINGS.json` gives A and REF separate output identities but then places one
patched `io.py` hash, one source commit, and one set of downstream/curation
hashes under a shared `executed_kilosort` object (lines 23--47). The source
qualification only states that **A** binds the patched `io.py` hash (line 5).
REF's reviewed result instead binds release commit `32d04e7d...`, and H1 can
verify that the three project files at that commit have the claimed
downstream/curation/loader hashes; D1 does not bind REF's actually installed
Kilosort `io.py` bytes. The A bounded-pread validation distinguishes the
unpatched input hash `27acaef1...` from patched hash `c6093e14...`, so the
single shared hash cannot silently stand for both runs.

Repair by adding independent `kilosort_a.executed_source` and
`kilosort_ref.executed_source` receipts: exact Kilosort package/version and
`io.py` hash, SpikeInterface wrapper/environment lock, project producer source,
effective settings, and curator source. If REF's installed bytes cannot be
recovered, label that binding unavailable and limit REF intermediate semantics
to empirically validated internal-array relationships rather than claiming an
exact executable-source binding.

### 2. Implement the ambiguity-preserving curated-to-original relation

The contract correctly says there is no saved curated-parent index and that
duplicate `(time, detection-template)` keys remain ambiguous (CONTRACT lines
51--56). However, the adapter implements only internal `full_st/full_clu/kept`
validation (adapter lines 54--111). It contains no curated-to-original join,
and its tests contain no repeated-key parent-map fixture. The source
qualification explicitly admits that the join was not done (lines 34--39).

Repair with a function and independent fixtures that emit, for every curated
row, `unique_parent`, `ambiguous_parent`, or `unmatched_parent`, retaining all
candidate original parent indices for ambiguous keys. Include duplicated keys,
deleted duplicates, out-of-order attempts, one-to-many and many-to-one cases,
and wrong-template negative controls. Do not assign a unique parent by greedy
occurrence rank unless that rule is separately justified and frozen.

### 3. Emit event-level relations and freeze DARTsort candidate selection

`pairwise_metrics` returns aggregate counts only (adapter lines 179--212). It
does not return the matched index pairs or the event rows needed to populate
the promised `AMBIGUOUS_AND_UNMATCHED.csv` and `STAGE_ACCOUNTING.csv`
(CONTRACT lines 58--63). Likewise, the four groups specify only A and REF units;
the statement that DARTsort candidates will be “reported from temporal
correspondence” (line 42) does not freeze the DARTsort candidate universe,
ranking, tie handling, minimum support, or output of multiple candidates. That
leaves an outcome-dependent selection path.

Repair by freezing and testing an event-level schema with explicit
`parent_row_id`, `matching_label`, and `final_label`; pair-local matched row
indices; all-neighbor candidate sets/degrees; unmatched rows; and no transitive
identifier. Freeze the DARTsort candidate universe and deterministic reporting
rule before production arrays are inspected. The accepted counter may still
provide the summary count, but the event-level relation must reproduce it.

## Positive findings

- All nine packet members match the manifest; MANIFEST SHA-256 is
  `3db81680caa5810206b853c44a615a32b8a8fee942fc0199cc2bfcbe0eb790a2`
  and COMPLETE SHA-256 is
  `83e92b2e691f1d92215647116d214796c47b43a62695c74ef575445bac6059bf`.
- H1 reran the published suite unchanged: 10 tests passed. H1 additionally
  compared the frozen greedy counter with an independently written dynamic
  program over 176,400 exhaustive sorted multiset cases; all passed.
- `bind_kilosort_export` correctly distinguishes detection template
  (`full_st[kept,1]`) from post-clustering `spike_templates`, validates Boolean
  and strictly increasing integer kept selectors, checks half-open bounds, and
  rejects ancestry drift.
- `bind_dartsort_rows` requires exact final/H5 time-row equality while allowing
  matching-stage labels to differ from final labels. This incorporates D0's
  central semantic correction, though the production output schema still needs
  the three explicit fields described above.
- Common support includes its start and excludes its stop; the event tolerance
  is inclusive; pair names are restricted to REF--A, REF--DARTsort, and
  A--DARTsort; combined transitive pairing is rejected.
- No production arrays, voltage, GPU, replay, sort, RF, or holdout were used by
  the packet or this review. No executable production runner is included, and
  the contract explicitly stops before production execution.

## Implementation checks

- Done: packet integrity -> all nine member hashes/sizes, MANIFEST, and
  COMPLETE independently verified.
- Done: executed source semantics -> inspected the frozen adapter, accepted
  matcher snapshot, project exporter/curator/loader sources, A bounded-pread
  source receipt, and REF release-source hashes.
- Done: clock/support/count semantics -> traced half-open support, inclusive
  tolerance, kept-row ancestry, template meanings, pair separation, ambiguity
  counts, and non-transitive interpretation to code.
- Done: fixture independence -> reran 10 tests and independently exhaustively
  checked 176,400 matcher cases against a separate dynamic program.
- Done: execution gates -> packet is preparation-only, has no production
  runner, and explicitly disables production traversal pending review and a
  later frozen contract.
- Not done: real production-array traversal -> prohibited and unnecessary for
  this implementation review.
- Not done: exact REF installed-package binding -> absent from D1; must be
  supplied per arm or marked unavailable.
- Not done: curated-to-original parent relation and event-level stage output ->
  not implemented in the reviewed source.
- Can establish: the implemented internal-array validators and aggregate
  pairwise counter behave as specified on synthetic arrays.
- Cannot establish: readiness to localize the earliest production divergence,
  exact REF executable semantics, unique curated ancestry for duplicate keys,
  or any real event/scientific result.

