# Final delta-only acceptance: refractory impact map v2

## Verdict

**GO** for the three repaired deltas only.

Reviewed snapshot:

- `/mnt/NPX/Luke/DARTsort_motion_experiments/refractory_interval_defect_impact_map_h1_20261007_v2`
- MANIFEST SHA-256 `3bfb285271ac7f5991c48a48c995383710612648f44786bfe954f1f7bc009555`
- COMPLETE SHA-256 `15e23b28e8e01b80c82b6ddd6a7baf93cbddbd4c16f888ae03af0b32b7ca9c6e`
- Predecessor independent review MANIFEST `da12dea81713994b39ab4f33567b75c1dea4b9abc8684b1a419fcd8a333c64ce`

This review does not reopen the accepted defect mechanism, sufficient-statistic finding, or spike-time-only correction choice. It does not inspect or duplicate the H5 child's current D3 contract review.

## Delta findings

### 1. Mechanics versus membership, including D3 routing: accepted

The repaired map now separates:

- unaffected D0 comparison/inventory design (`IMPACT_MAP.csv:11`);
- unaffected D1 adapter/matcher mechanics (`:12`);
- affected-upstream D1/D2 exact neighborhood membership (`:13`);
- unaffected D2 managed synthetic lifecycle proof (`:14`);
- affected-upstream, unlaunched D3 v1 anchors (`:15`); and
- later D3 revision/readiness as owner-routed unknown under the existing H5-child review (`:16`).

The action is scientifically adequate: revise the anchors, independently show corrected membership is unchanged, or explicitly scope a retained run to the historical panel. `CORRECTION.md:15` and `RECALCULATION_PLAN.md:12` preserve accepted engineering work without allowing it to validate the affected selection. This closes the prior failure mode in which correct correspondence mechanics could be interpreted as answering a corrected-category question while running obsolete anchors.

The D3 v1 hold is recorded without claiming an outcome from the later D3 revision or its active review. No production launch is authorized by this acceptance.

### 2. Exact historical ACG boundary: accepted

`CORRECTION.md:16` and `RECALCULATION_PLAN.md:6-7` freeze the historical observed-count behavior as:

- within unit, sorted samples, `j>i`;
- integer `d=t[j]-t[i]` with `0 <= d < 280`;
- bin `d//7`;
- same 300-s block;
- equal Boolean `state_nonzero` class for q0/displaced;
- no exact field-segment gate; and
- retained same-sample `d=0` pairs.

Segment-gated or zero-lag-excluding variants are explicitly labeled redesigns rather than the one-factor clock correction. Corrected upper edges, centers, and eligibility are fixed in the exact probe clock (`RECALCULATION_PLAN.md:7`). This closes the prior risk of changing observed pair membership at the same time as the modeled clock.

### 3. Exact-source-or-reconstructed qualification: accepted

The map labels both R1c results `affected_provenance_qualified` (`IMPACT_MAP.csv:3-4`). `CORRECTION.md:17` and `RECALCULATION_PLAN.md:5` require either exact evaluator bytes with SHA-256 `415f29d7...` or an explicitly named `RECONSTRUCTED_R1C_CLOCK_REPAIR` that independently reproduces reproducible unchanged historical fields/point values before interpreting the edge-only delta. Ambiguous reconstruction is a stop condition (`RECALCULATION_PLAN.md:14`).

This appropriately retains the source gap rather than converting the census contract/audit into an executed-source claim. It is a contract prerequisite, not a corrected scientific outcome.

## Scope and residual gates

No blocking defect remains in these three deltas. Remaining work is execution-contract and outcome work: freeze resources/input hashes, implement the fixtures, satisfy exact-source or reconstructed reproduction, emit sufficient statistics, and independently review the actual correction before interpreting results. The current H5 D3 review remains with its assigned owner.

## Implementation checks

- Done: repaired packet integrity -> MANIFEST/COMPLETE binding and every member size/hash verified.
- Done: mechanics/membership delta -> exact v2 classifications and actions read from `IMPACT_MAP.csv:11-16`, `CORRECTION.md:15`, and `RECALCULATION_PLAN.md:12`.
- Done: historical count-boundary delta -> same-block/Boolean-state, no-segment, and retained `d=0` rules are explicit (`CORRECTION.md:16`; `RECALCULATION_PLAN.md:6-7`).
- Done: provenance delta -> exact-source or explicitly reconstructed path, reproduction prerequisite, and ambiguous-reconstruction stop are explicit (`CORRECTION.md:17`; `RECALCULATION_PLAN.md:5,14`).
- Not done: accepted-core re-review -> excluded by the delta-only assignment.
- Not done: current H5-child D3 contract review, production launch, spike traversal, or corrected outcomes -> explicitly excluded and separately owned.
- Can establish: all three blocking findings from the v1 review are repaired in the bound v2 routing packet.
- Cannot establish: corrected unit/anchor/verdict transitions, D3 successor readiness, exact R1c source behavior, or biological identity/purity.

