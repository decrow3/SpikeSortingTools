# Independent review of the D3 v3 post-run oracle

## Verdict

`BLOCKED_FALSE_ACCEPTS`

The frozen packet is immutable and correctly sealed, its bound predecessor
members verify, and all six required fixture classes behave as expected.
However, the consumer is not safe to use on a future production outcome. Three
independent resealed adversarial fixtures that violate load-bearing contract
semantics still return
`GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY`.

No production namespace or outcome was inspected. This verdict applies only to
the frozen pre-outcome consumer and does not interpret scientific results.

## Blocking findings

### 1. Pair timing is not checked against the inclusive 15-sample tolerance

The consumer checks that each pair-event time lies in global support and that
rows are sorted, but it never checks the time difference between reciprocal
matched events (`postrun_oracle.py:200-225`). An independently resealed fixture
shifted every side-B event by 1000 samples while retaining reciprocal source IDs
and summaries; the consumer still returned GO. The bound runner actually forms
matches using the frozen tolerance (`production_saved_output_runner.py:497-540`),
but a post-run oracle must verify the serialized evidence rather than trust the
producer that it is intended to audit.

Required repair: for every matched row, validate a non-null, in-range reciprocal
index and source ID on both sides and require
`abs(a.sample_time - b.sample_time) <= 15`. Validate all-neighbour membership
and degrees against the same inclusive window, or narrow the claim if those
lists are not independently checkable.

### 2. The complete uncapped candidate universe is not established

`eligible_final_units` is only checked as a nonnegative integer
(`postrun_oracle.py:178-182`). The ranking validator checks per-anchor ordering
but never requires each ranking to contain that many distinct eligible labels
(`postrun_oracle.py:121-148`). A resealed fixture reporting two eligible final
units with only one ranked label per anchor still returned GO.

Required repair: the output schema must expose the canonical eligible label set,
not only its count, and the consumer must require each of the eight rankings to
contain that exact set once. A count-only receipt cannot distinguish an omitted
label from a complete universe.

### 3. Ranking denominators and derived metrics are not reconciled

The consumer does not validate nonnegative integer event totals, recompute
retention/F1, or reconcile candidate totals with the corresponding pair-event
side. A resealed fixture changed each row to 1000 candidate events and 2000
anchor events while leaving 25 matches, 0.25 retention, and 0.25 F1; it still
returned GO. The correct values for those totals are 0.025 retention and 1/60
F1. The producer computes these formulas at runner lines 452-493, but the oracle
does not independently enforce them.

Required repair: validate types/ranges; recompute
`smaller_train_retention = matches / min(anchor_events, candidate_events)` and
`exclusive_f1 = 2*matches / (anchor_events + candidate_events)`; reconcile both
event totals to emitted relations for every reported context.

## Checks that passed

- Packet MANIFEST SHA-256 is
  `3162ed5c3b5145abedecb30a43a891a6e61015ca30260d7352420cdbfc6ef807`;
  all seven members match size/hash and COMPLETE binds it.
- COMPLETE was written after MANIFEST, which was written after every member.
- Both predecessor packets and their declared members verify against the frozen
  hashes.
- Baseline suite: 8 passed; compile and predecessor checks passed.
- Independent positive, missing-field, stale-hash, reordered-row,
  false-parent, and failure-closure fixtures all received the expected
  classification.
- Final clock name, half-open global support, stable serialization, null
  DARTsort parent/matching attribution, success resource ceilings, and
  success prohibited-work flags are checked by the current consumer.

## Claim boundary

The intended claim remains appropriately narrow: direct final REF/A-to-DARTsort
correspondence for the frozen anchors and support only, with intermediate
matching-to-final attribution unavailable. The false accepts mean the current
consumer cannot authorize even that limited claim. No transitive identity,
biological identity/purity, or causal stage localization is supported.

Implementation checks
- Done: immutable seal and predecessor bindings -> all declared sizes/hashes and both COMPLETE bindings verified (reviewed MANIFEST `3162ed5c...`; derivative MANIFEST `10cdbcd2...`; technical-review MANIFEST `7874adc4...`).
- Done: required fixture exercise -> positive accepted; missing field, stale hash, reordered row, and false parent rejected; preserved failure accepted only as no-science failure closure (`TEST_RECEIPT.json`).
- Done: independent false-pass probes -> 1000-sample pair separation, incomplete candidate universe, and inconsistent ranking metrics each returned GO (`independent_adversarial_review.py`, SHA-256 `ab588619...`).
- Done: executed-source trace -> matching tolerance is applied by the producer at runner lines 497-540, rankings are constructed at 452-493, while consumer lines 121-148 and 178-225 omit the corresponding independent checks.
- Not done: future production output inspection or scientific outcome validation -> prohibited by the pre-outcome scope.
- Can establish: the current frozen oracle has three material false-accept paths and must be repaired/re-frozen before use.
- Cannot establish: any production correspondence value, intermediate matching-to-final attribution, transitive or biological identity, purity, or causal stage.
