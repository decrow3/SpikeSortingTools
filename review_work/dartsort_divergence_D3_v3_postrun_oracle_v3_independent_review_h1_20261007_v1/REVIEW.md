# Independent review: D3 v3 post-run oracle v3

## Verdict

`BLOCKED_CANONICAL_UNIVERSE_NOT_SOURCE_BOUND`

V3 correctly repairs the two literal v2 false accepts: duplicate candidate
rows and negative matched indices are rejected. All 13 baseline tests pass,
including every earlier counterexample and the positive control. The candidate
universe is still only internally self-consistent, not independently bound to
the producer's canonical eligible labels, leaving one material false accept.

No production namespace, outcome, H5 output, raw voltage, waveform, RF/holdout,
or sorting data was accessed.

## Repairs verified

- Duplicate final-label rows are rejected before ranking validation
  (`postrun_oracle.py:131-133`).
- Candidate label-set size must equal `eligible_final_units`
  (`postrun_oracle.py:196-200`).
- Matched indices must be exact integers, nonnegative, and in range; both sides
  are checked for reciprocity, source identity, ±15-sample tolerance, and
  neighbour membership (`postrun_oracle.py:252-270`).
- The v2 duplicate-row and negative-index fixtures independently reject.
- The positive baseline returns the limited final-correspondence GO verdict.

## Remaining blocker

`DARTSORT_MATCHER_ORDERING.json` contains only the eligible-unit count. The
consumer derives the supposed canonical label set from the eight ranking files
and checks those copies against each other. It does not compare them with a
separately emitted candidate-index label list.

An independent fixture replaced label 7 with label 999 consistently in all
eight rankings, every DARTsort pair summary and event relation, and the stage
rows. The count, uniqueness, cross-anchor equality, metrics, event relations,
and seal all remained valid; v3 returned
`GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY`. The consumer therefore cannot
distinguish the actual eligible label set from a consistently substituted one.

Required repair: change the producer to emit the sorted canonical
`eligible_final_labels` from `candidate_index` in
`DARTSORT_MATCHER_ORDERING.json`; require its length to equal
`eligible_final_units`; then require every anchor ranking's unique label set to
equal that exact list. Freeze and review the producer/schema delta before using
the oracle on an outcome.

## Seal and regression checks

- All eight v3 members match MANIFEST size/hash.
- MANIFEST SHA-256:
  `d367aa95fac7623bff1e14607bdebf80707ae10ef72cba0db515ec36131e84ed`.
- COMPLETE SHA-256:
  `0a2433a25513b71a36fe63c34deb59e36e5f3c212a290fe7ca87da784b517742`.
- Member → MANIFEST → COMPLETE filesystem chronology is monotonic.
- Both predecessor packets and declared members verify.
- Baseline: 13 passed in 0.27 seconds; compile and predecessor checks passed.

Implementation checks
- Done: immutable seal/source bindings -> eight members match; COMPLETE binds MANIFEST; predecessor verification passes.
- Done: complete regression set -> positive accepted and all five earlier counterexamples rejected in the 13-test baseline (`TEST_RECEIPT.json`).
- Done: v2 blocker fixtures -> duplicate rows and negative indices independently reject (`independent_v3_review.py`).
- Done: canonical-universe substitution -> consistent label 7→999 substitution returns GO because ordering exposes only a count, while the consumer compares ranking-derived sets only to each other (`postrun_oracle.py:196-200`; script SHA-256 `e6a9672b...`).
- Not done: production outcome or source-array validation -> prohibited by this pre-outcome review.
- Can establish: v3 closes the two literal v2 examples but does not source-bind the identity of the eligible-label universe.
- Cannot establish: readiness to interpret a future production outcome, any correspondence value, intermediate matching-to-final attribution, transitive/biological identity, purity, or causality.
