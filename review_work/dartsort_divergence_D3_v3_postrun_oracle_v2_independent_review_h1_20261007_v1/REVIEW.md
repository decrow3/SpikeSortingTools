# Focused independent delta review: D3 v3 post-run oracle v2

## Verdict

`BLOCKED_REPAIR_INCOMPLETE`

V2 correctly rejects all three original v1 counterexamples, its 11-test suite
passes independently, and its packet seal and source bindings verify. Two
false-accept paths remain inside the same repaired boundaries, so v2 must not
review or authorize a future production outcome.

No production namespace, outcome, H5 output, raw voltage, waveform, RF/holdout,
or sorting data was accessed.

## Original repairs that now work

- A resealed reciprocal pair separated by 1000 samples is rejected by the
  inclusive ±15-sample neighbour reconciliation.
- A single ranking row with `eligible_final_units = 2` is rejected.
- A count-inconsistent retention value is rejected.
- Ranking count-derived retention/F1 and emitted passing-pair totals are now
  checked at `postrun_oracle.py:137-143` and `259-262`.

## Remaining blockers

### Candidate-label uniqueness and canonical identity are not established

V2 compares ranking row counts to `eligible_final_units` and compares label
sets across anchors (`postrun_oracle.py:193-196`), but never requires each
ranking label to be unique or requires the set size to equal the eligible
count. A fixture with `eligible_final_units = 2` and two identical rows for
label 7 in every anchor ranking still returns GO.

Required repair: require exactly one row per label, require
`len(label_set) == eligible_final_units`, and bind the canonical eligible label
set in `DARTSORT_MATCHER_ORDERING.json` so equality is checked against the
producer's candidate-index receipt rather than only across mutually dependent
ranking copies.

### Matched local indices are not range checked

V2 directly indexes `b_rows[row["matched_other_pair_local_row_index"]]`
(`postrun_oracle.py:249-253`) without requiring an integer index in
`[0, len(b_rows))`. Python interprets `-1` as the last row. Changing the final
matched A row's index from 24 to -1 therefore leaves time/source reciprocity
apparently valid and returns GO, even though the serialized relation violates
the producer schema.

Required repair: for both sides, require matched rows to have non-null integer
local/source indices in range, unmatched rows to have null matched fields, and
then verify reciprocal index, source ID, and ±15-sample time. Add a negative-
index fixture.

## Seal and regression checks

- All eight v2 members match MANIFEST size/hash.
- MANIFEST SHA-256:
  `d1bc141c59385f21c8642fa72901d49ac8f067d2791a5b4d0e384d3bdd551d34`.
- COMPLETE SHA-256:
  `0e657bb4eac9c2a30a731fc055201a78e1e9824a3f632b4d0840ecc6e1b42501`.
- Member → MANIFEST → COMPLETE filesystem chronology is monotonic.
- Both predecessor packets and declared members verify.
- Baseline: 11 passed in 0.22 seconds; compile and predecessor checks passed.

Implementation checks
- Done: v2 immutable seal/source bindings -> eight members match; COMPLETE binds MANIFEST; predecessor verification passes.
- Done: original repair probes -> all three v1 counterexamples now reject (`TEST_RECEIPT.json`).
- Done: candidate-universe boundary -> duplicated labels satisfy row count and return GO because lines 193-196 compare count and cross-anchor sets but not per-anchor uniqueness or an independent canonical set.
- Done: matched-index boundary -> `-1` returns GO because lines 249-253 use Python indexing before validating range (`independent_delta_review.py`, SHA-256 `293840b4...`).
- Not done: future production output or scientific-result validation -> prohibited by this pre-outcome delta review.
- Can establish: the three exact original examples are repaired, but v2 retains two material false-accept paths in those same boundaries.
- Cannot establish: readiness to interpret a production outcome, any correspondence value, intermediate matching-to-final attribution, transitive/biological identity, purity, or causality.
