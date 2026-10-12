# Focused independent review: joint-model K7 frame-score fixture v1

Proposed verdict: **GO_SYNTHETIC_SCORE_ORACLE_ONLY**.

Review the frozen contract, implementation and tests. Independently verify:

- stable union includes matched-filter-only events and rejects conflicting duplicate metadata;
- training requires two states separated by at least 80 µm;
- only held-out states with `abs(q) >= 80 µm` contribute;
- at least 50 events are required per contributing state;
- score is median of per-state medians of `log(residual_probe/residual_tissue)`;
- external thresholds have the correct direction and no implicit default;
- single-state training and insufficient held-out support abstain;
- undetermined is reported as undetermined while operationally defaulting to tissue;
- linked-group evidence stays member-local and never triggers an automatic group merge;
- integer/lattice clocks/states, residuals, thresholds and support parameters fail closed.

Use independent known-answer/adversarial fixtures. No reviewer production
actions. Do not implement or inspect template learning, matched-filter detection,
whitening, threshold calibration, real units/arms, recordings/voltage, DARTsort,
GPU, sorts, RF/holdout or C4. Publish a distinct immutable review.

Implementation checks
- Done: scoring semantics frozen before implementation; 26 author fixtures pass.
- Not done: equal-flexibility template training, whitening-after-placement, block bootstrap or real calibration.
- Can establish: synthetic candidate-union, score, abstention and member-local group-rule oracle.
- Cannot establish: full K7 validity, real labels, N3/N4 or N0/N1 readiness.
