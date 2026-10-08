# Independent H1 review: corrected common-support correspondence v2

## Verdict

`GO_DESCRIPTIVE_COMMON_SUPPORT_WITH_SCOPE_LIMITS`.

The cache-isolated H5 v2 run correctly applies the frozen common-opportunity boundary and produces internally closed descriptive correspondence results for the 30 selected Arm-A units. The reviewed headline is 121/124 evaluable imec0 unit-state groups and 110/115 imec1 groups with strictly positive A-coverage excess and candidate-edge excess against each of the four fixed temporal nulls.

This establishes candidate correspondence relative to these four null shifts only. It does not establish biological identity or purity, recovered spikes, true extras or bad merges, whole-sort prevalence, a causal mechanism, an intervention benefit, sorting superiority, statistical significance, RF, or holdout performance.

## Immutable identities

- Reviewed publication: `/mnt/NPX/Luke/DARTsort_motion_experiments/motion_diagnosis_saved_correspondence_common_support_repair_h5_20261007_v2`.
- Publication MANIFEST: `1772634b8d5b90d278efc7c6a12ee9ac585e1fe83412a73b0a9d9f9882ac82c0`.
- Publication COMPLETE: `692a9b9637bd3e1d06a0e20ed771ae55399c3edebdabe1be578b39652e9e7f91`.
- Result MANIFEST: `5972d47773ab668e67b8b32194f5b5d027b3eec669ab15da15ae248e042a698c`.
- Result COMPLETE: `d9f7250efc97ca0baf7f1c165bb0446bc2e68b34aada6d56564bbd93ff5dcf1e`.
- Contract: `350b8ec1a695f757d49c19789551a0ed356d2014f399f770c2d56c4204480463`.
- Executed source: `3e40224aaed11429725c1c081ed68b5cbd3debf9c7c518128345b2e860ff0728`.

All 22 publication members and all 9 result-manifest members passed independent size and SHA256 verification.

## Changed-boundary review

The implementation makes observed and null comparisons use the same A events. For each shift `s`, an event is eligible only when both inclusive windows `A +/- 15` and `A - s +/- 15` remain in the event's original field-state segment (source lines 121-130). Observed matching uses the unshifted REF clock; null matching uses the rounded shifted clock; both use the same comparison shift for eligibility (lines 462-470). Candidate REF events retain their original event time, segment/state, x position, and depth corrected as `saved_y - q(original_REF_time)` (lines 146-160 and 179-195). The REF denominator is the full finite-position original-state population and is identical between modes (lines 258-267 and 483-508).

The candidate statistic remains inclusive `+/-15` samples and inclusive 40-um Euclidean x/y distance. It retains every admissible neighbour; it does not perform greedy or unique assignment or cap partners (lines 133-198 and 270-312). Negative Arm-A cluster labels cannot index the selected-unit lookup (lines 228-255).

The prior and corrected runs share 105 input paths with identical hashes. The only provenance-path difference is the frozen contract/config. Selected units, actual sampling rates, rounded shifts, field audits, state sets/exposures, and full finite-position REF denominators are exactly equal across the two results. Thus no cohort, saved array, field, clock, radius, or null shift changed alongside the support repair.

## Independent checks and saved-table closure

- H1 reran the five published production-path tests in a fresh cache: 5 passed in 2.85 seconds.
- H1 compared the production matcher with a separately written brute-force implementation on 500 randomized multi-segment/state cases spanning shifts -100, -50, 0, 50, and 100 samples. Degrees and exact candidate-index pairs agreed in every case.
- `PER_UNIT_STATE_MODE.csv` has 2,280 unique mode keys: 1,140 observed and 1,140 null rows, covering 15 units on each probe, 9 imec0 states, 10 imec1 states, and four shifts.
- `NULL_EXCESS.csv` and `OPPORTUNITY_AUDIT.csv` each have 1,140 unique keys. Every observed/null A denominator and every REF denominator is equal; all saved equality flags are true.
- All A coverage, multi-candidate fraction, REF coverage, coverage-excess, REF-excess, and edge-excess arithmetic recomputes from the published counts with zero failures.
- All bounds hold: covered A events do not exceed eligible A events, multi-candidate events do not exceed covered events, REF covered events do not exceed the REF denominator, and edge counts are at least both covered-event counts.
- `PARTNER_EDGES.csv` has 18,586 rows and no duplicate mode/key/partner rows or negative partner IDs. Partner edge sums, distinct-partner counts, top-partner counts, and per-row edge fractions close exactly to every nonempty mode row.
- The saved headline recomputes exactly. There are 124 evaluable imec0 unit-state groups, of which 121 exceed all four nulls on both primary conditions; imec1 is 110/115. Groups without finite coverage because common support is empty are excluded, not counted as failures.
- Weighted A candidate coverage ranges by shift from 0.4275 to 0.4531 observed versus 0.0712 to 0.0723 null on imec0, and 0.8282 to 0.8405 observed versus 0.1686 to 0.1698 null on imec1. These are descriptive selected-cohort rates, not population estimates or significance tests.

## Execution and cache isolation

The fixture cache contained 10 files after the exact H5 tests. The distinct execution cache was empty after the exact interpreter/source-path preflight and contained 12 files after managed execution. The service used the frozen interpreter, source path, contract hash, selection, output path, `PYTHONPATH`, and execution cache. It ran once under the transient user service from 18:44:44 to 18:45:25 PDT, consumed 33.764 CPU seconds, sealed the result, and left no live PID. No recording, voltage, GPU, sorter, RF, or holdout work occurred.

## Correction to prior H1 review

The earlier H1 packet `motion_diagnosis_saved_correspondence_repair_final_review_20261007_v1_h1` accepted the historical result as using a valid same-support null. That statement is superseded. The broad implementation audit subsequently established that the historical observation/null opportunity construction was not comparable. The historical packet remains useful as preserved arithmetic but must not be cited as chance-controlled correspondence evidence. This corrected common-support v2 result is the first reviewed result for that claim boundary.

## Remaining limitations

- The four shifts are fixed descriptive controls, not a sampled null distribution; no p-value or calibrated uncertainty follows from exceeding all four.
- The panel is deliberately selected and state groups are not an independently sampled population. Group fractions are not whole-sort prevalence.
- H1 did not recompute the full result from H5-local event arrays. The managed run hash-verified those inputs, and compact rows close independently.
- The exact field-loader dependency is hash-bound at execution but is not copied into this publication. Its outputs and 105 shared input hashes match the prior reviewed lineage; a future packet should include exact dependency snapshots for self-contained provenance.

## Implementation checks

- Done: publication/result seals -> all 31 manifest members independently rehashed with no failures.
- Done: actual execution -> contract, source, argv, environment, cache isolation, managed-service state, completion, and provenance receipts inspected.
- Done: intended-only delta -> 105 common input hashes, cohort, clocks, fields, state exposures, shifts, radii, and REF denominators match the historical run; only support/denominator/null construction and defensive boundary handling changed.
- Done: axes/frames/clocks -> actual per-probe sample rates, original REF state/time/depth correction, inclusive temporal/spatial gates, and no crop offset traced in source and saved summary.
- Done: matching/counting -> all-neighbour semantics, collision preservation, negative-label guard, mode pairing, denominators, partner tables, and headline recomputed.
- Done: independent known-answer work -> published fixtures rerun plus 500 randomized brute-force cases.
- Not done: full saved-array recomputation on H1 -> arrays are H5-local and were not transferred.
- Not done: exact dependency-source reread -> dependency hashes are execution-bound but exact files are absent from this publication.
- Can establish: corrected descriptive candidate correspondence above each of four fixed comparable-opportunity temporal controls for nearly all evaluable selected unit-state groups.
- Cannot establish: identity, purity, recovery, extras, merge safety, whole-sort prevalence, causal mechanism, intervention benefit, sorting superiority, statistical significance, RF, or holdout performance.
