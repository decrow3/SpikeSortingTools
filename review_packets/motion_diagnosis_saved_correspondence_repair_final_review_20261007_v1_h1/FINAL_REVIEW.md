# Independent review: saved-output correspondence repair v2

Verdict: `GO_OUTCOME_SCOPED_WITH_EVIDENCE_CORRECTIONS`.

The load-bearing primary result is reproducible from the published compact rows. Observed candidate coverage exceeds every one of the four same-support temporal nulls in 120/122 evaluable imec0 unit-state groups and 106/113 imec1 groups. Pooled observed coverage is 0.452531-0.520942 on imec0 versus 0.069886-0.071000 under the nulls, and 0.839910-0.863531 on imec1 versus 0.167026-0.167457 under the nulls.

The executed source applies the intended common frame: Arm A positions are already corrected; REF depth is `saved_y - q(original_REF_time)`. It uses inclusive +/-15-sample timing and inclusive 40-um Euclidean x/y distance. Each +/-50/100-ms null retains the REF event's original position, displacement, state and segment, while both arms use the same shift-specific boundary trim. All candidate collisions are retained, unique assignment is omitted, and full partner ambiguity is serialized.

The scientific scope remains descriptive candidate correspondence for the frozen selected cohort. It is not evidence of biological identity, recovered spikes, true extras, purity, a bad merge, or sorting superiority.

Three evidence corrections must travel with the GO:

1. The published test file contains five test functions rather than the contract's claimed six, and they are in-memory synthetic fixtures rather than a file-backed loader fixture. Production file loading was inspected and the compact tables close, so this does not alter the primary arithmetic, but file-backed fixture coverage cannot be claimed.
2. The secondary median distinct-partner claim `[10, 11]` depends on an unstated support-row filter; it is not the unqualified median over all observed compact rows. Do not cite those medians without naming the filter.
3. The producer's `REVIEW_STATUS.json` labels itself as an H1 independent review. It should be treated as H5's scoped review summary; this packet is the independent H1 review.

Implementation checks

- Done: verified all 15 publication-manifest members and all seven result-manifest members; no size or hash failures.
- Done: traced the coordinate sign/frame, full-session clock, support trimming, inclusive temporal/spatial gates, null construction, collision semantics and output sealing in executed source SHA256 `1d5eb39118eeb7bd9a0b035726a688ec78e6d834ae5b716e2706c52ae04f2842`.
- Done: independently closed 2,280 mode rows, 1,140 excess rows and 16,806 partner rows with zero duplicate keys, partner-closure failures or excess-arithmetic failures.
- Done: independently recomputed the primary counts and pooled observed/null coverage from the compact CSVs.
- Not done: independent full recomputation from H5-local saved arrays; those arrays are not mounted on H1.
- Not done: actual file-backed fixture; none is present in the published packet.
- Can establish: scoped descriptive all-neighbor candidate correspondence above the frozen temporal-null baseline for the selected cohort.
- Cannot establish: identity, purity, recovered/extra spikes, merge mechanism, whole-sort prevalence, causality, intervention benefit, sorting superiority, or RF/holdout performance.
