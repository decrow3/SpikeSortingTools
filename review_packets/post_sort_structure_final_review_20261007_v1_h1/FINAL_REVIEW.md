# Independent H1 review: post-sort structure census and panel

## Verdict

**ACCEPT_SCOPED_WITH_NONBLOCKING_CORRECTIONS.** The saved-table census, deterministic panel selection, and bounded pair measurements are arithmetically coherent and support the packet's stated descriptive use. They do not establish biological identity, duplicate frequency, merge safety, a causal pipeline stage, or arm superiority. State-specific and state-preserving-null claims remain qualified until the exact executed field-loader source is published for independent inspection.

## Reviewed immutable inputs

- Census: `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_census_h5_20261007_v1`, MANIFEST `022ad0e1c558a283a97cabe7f27aaa318701bc228197ea7819ae75bcae2f1028`, COMPLETE `db6c9ea4f4f0c9f3841cc3bb351947b1aaf04979f7c0ef8257f01bd634ad3d43`, SOURCE `dc33e5d4d536998cfcd74a8d317b979577ab982c20c339e158b32925f74388ac`.
- Selection: `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_panel_selection_h5_20261007_v1`, MANIFEST `93d5fc82430ec526c9f50224471f4fa4a8375082736ee1c00ea4324dc8eea877`, COMPLETE `1d1eb658f95061cbd48bbfff59d8504e48eddc4871bf864acc17b9068cc9fb16`, SOURCE `f919f14cdfb960a0bfbddf47a7364d4e0bf35d0772a1aeb96ef7a2b46c7b3319`.
- Panel: `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_panel_analysis_h5_20261007_v1`, MANIFEST `db6eed1e5ca4c3a6df222c1b5676e99819bcfbdd689dfea6ee23e2ee327ac9c4`, COMPLETE `eec515883374088c0e44055a3b436c4c5e6d90d9f17a79406a016d53878b9bee`, SOURCE `90a24e67a85538958675e42cbc785655ea2c42599208ad5e0ae5da56e85a34ed`.
- Frozen plan: `e3ee6fbd7b0bb290bc1e5e9891ee16f19ed8cbf256cc7146427cec07192dd36c`.

All three manifest hashes, COMPLETE bindings, sizes, and member hashes were independently checked from the shared packets.

## Findings

1. **Census denominators and missingness close.** `CENSUS_UNITS.csv` has 2,766 unique probe/arm/unit rows and 142,965,975 spikes. `GROUP_COUNTS.csv` all-row denominators reproduce both totals. For the six requested continuous metrics, direct nonfinite counts exactly reproduce `MISSINGNESS.csv`: waveform 0, sliding-RP 2,190, noise cutoff 62, ISI ratio 1, amplitude cutoff 1,074, and presence ratio 0. The availability ledger separately marks three absent arrays (confidence matrix, observed lag counts, amplitude distribution), four probe/arm rows each. Source distinguishes a missing column from a saved nonfinite value at census SOURCE lines 344-363.

2. **Prevalence uses explicit populations.** Group counts use unit and spike denominators for all rows, in-domain rows, and excluded-domain rows (census SOURCE lines 287-341). The selection prevalence table uses the in-domain population before category selection (selection SOURCE lines 89-124). The panel report sums the four probe/arm denominators to 2,243 units and 118,406,735 spikes; it labels the panel deterministic and enriched.

3. **SRP timing qualification is explicit and numerically correct for the audited reconstruction.** The saved evaluator historically models forty 0.25-ms bins (10 ms), while forty 7-sample bins cover 9.333384 ms on imec0 and 9.333408 ms on imec1. `SRP_AUDIT.csv` reports both clocks and separates a 90%-confidence-at-10%-contamination target from the weaker any-finite-tested-bound question. The reconstructed minimum spike counts and zero-pass population counts reproduce the source formulas (census SOURCE lines 131-166 and 366-405). This validates the addendum arithmetic, not historical biological purity or the original numerator construction.

4. **The refractory comparators are distinct and properly named.** Global adjacent differences use `< round(0.0015*fs)`, hence samples 0-44 at both probe clocks (maximum included positive lag about 1.4667 ms). The production short-interval count uses positive differences 9-29 samples inclusive (about 0.3000-0.9667 ms), restricted to the same 300-s block and exact field segment (panel SOURCE lines 234-280). Across the 123 measured units, same-segment/block global counts never exceed global counts; 55 units lose at least one boundary-crossing event, as expected.

5. **Selection is deterministic and caps close.** There are exactly 16 unique anchors: one per four categories in each of four probe/arm groups. The union contains 117 unique pairs, 4-10 per anchor. Saved inclusion reasons total 57 spatial/state, 64 waveform-similarity, and 9 legacy cross-arm correspondence contributions, exactly matching source-audit post-cap totals. All cap and truncation flags recompute. Legacy correspondence enters only as a bounded candidate reason (selection SOURCE lines 162-176); neither selection nor report treats it as identity.

6. **Pair, matching, and null tables close.** The 117 pair rows each have three state rows, twenty null rows, and one hundred 1-ms correlogram bins. Exclusive matches obey both event-count bounds and never exceed all-neighbour counts. All empirical upper-tail probabilities recompute from `NULL_DRAWS.csv`; 75 rows at `p <= 0.05` are all exactly `1/21`, the minimum possible with twenty draws. This is coarse descriptive separation, not independently calibrated significance. The 55-case fixture compares production chronological matching with an independent maximum-cardinality dynamic program and covers inclusive boundaries, duplicates, collisions, unmatched cases, and randomized examples (panel SOURCE lines 116-153).

7. **Clocks and support are coherent in the inspected implementation.** Fixed intervals use exact final exposure; interval spike counts close to all 123 unit totals. Candidate circular shifts stay within each fixed-block/exact-segment intersection and avoid offsets within the match tolerance (panel SOURCE lines 285-301); no event was immovable. Observed state event counts partition exactly into applied-q0 and displaced rows. However, the exact executed `full_session_r1_evaluation.py` dependency named only by SHA256 `415f29d7...` is not a member of the published panel packet or otherwise present on the shared tree. The panel checks that hash at runtime (panel SOURCE lines 418-421), but H1 could not independently inspect the exact implementation that constructs field states and segments. Therefore exact state-frame and state-preserving-null semantics remain a provenance-qualified claim.

8. **Input-member verification is incomplete in producer source but output closure mitigates it here.** Selection and panel `verify_packet` helpers check MANIFEST/COMPLETE identity but do not iterate and hash all input members before reading them. H1 independently verified the published packets and confirmed that all 117 panel pair keys/reasons exactly equal the published selection and all 123 panel unit memberships, spike counts, state counts, and saved census labels match the published census. Future runs should verify every consumed member or bind each consumed file hash explicitly.

9. **Figures are readable with one labeling defect.** All seven figures were visually inspected. Anchor plots label 300-s bins, recording-origin minutes, sorter amplitudes, and the applied-q0 limitation. Pair overview distinguishes cross-arm saved frames and disclaims identity. In `census_overview.png`, the lower-left title says “both probes pooled,” but source groups by probe and immediately breaks after the first group (census SOURCE lines 438-446); that panel is imec0-only. Correct the title or truly pool probes in a new, non-overwriting figure revision. The SRP plot is valid but crowded.

## Decision scope

The packets are suitable for choosing a small bounded follow-up based on recurring saved-output patterns, provided the decision remains exploratory and the exact field-loader snapshot is supplied before relying on state-specific/null semantics. The simplest next step is to repair provenance and the mislabeled figure; no recomputation of voltage or new sort is needed. If a targeted intervention is selected, preserve strict-pass anchors as negative controls and use one fixed bounded window before medium/full escalation.

## Implementation checks

- Done: frozen packet integrity -> all three manifests, COMPLETE bindings, member sizes and hashes verified (manifest hashes above).
- Done: executed top-level source/config review -> census, selection, and panel `SOURCE.py` and contracts inspected; parameters traced to consumers (source hashes and line citations above).
- Done: denominators/missingness/selection/table arithmetic -> all stated closures reproduced from saved CSVs.
- Done: axes, clocks, boundaries, match/count semantics -> 7/fs versus 0.25-ms SRP clocks, strict global refractory threshold, inclusive 9-29 sample counter, exclusive versus all-neighbour counts, fixed intervals, state partitions, null and empirical-p arithmetic checked.
- Done: independent known-answer fixture -> published 55-case production-versus-DP result inspected; exact production counter source hash `0f6468a2...` independently located and read.
- Done: figures -> all seven visually inspected; one incorrect pooling label found.
- Not done: exact executed field-loader dependency review -> SHA256 is recorded and runtime-checked, but the exact source snapshot is absent from the shared packet/tree.
- Not done: raw voltage or wholesale recomputation -> outside this saved-output review and unnecessary for the scoped descriptive verdict.
- Can establish: internally coherent census prevalence, deterministic enriched selection, bounded descriptive unit/pair structure, and candidates for a targeted follow-up.
- Cannot establish: biological identity/purity, duplicate frequency, merge safety, causal mechanism, arm superiority, population prevalence from the enriched panel, or exact state/null semantics without the missing dependency snapshot.
