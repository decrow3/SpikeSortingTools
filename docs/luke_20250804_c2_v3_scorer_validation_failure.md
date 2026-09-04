# C2 v3 first run — a scorer validation failure, not a drift result

**Date:** 2026-09-03
**Status:** the run is void as a C2 result; it is kept as the test case that
found the bug fixed in [decision 0014](decisions/0014-injected-truth-scoring-is-per-cluster.md)
**Run:** `testing/luke_rescue_c2_drift_challenge.py` (C2 v3 prespec, schema
`luke-rescue-c2-drift-challenge-v3`), output
`testing/outputs/luke_rescue_c2_drift_challenge_v3/`

## What was run

All 14 compact D2b-2 donors (73–295 µV, 7 neg / 7 pos), injected into the quiet
imec1 depth strip with geometry-preserving placement, static arm plus three
moving trajectories (`rigid_15um`, `rigid_40um`, `osc_20um_40s`), scored under
`RESCUE` and `LEGACY_STYLE`. 112 sort conditions, ~2 h wall.

## Result: no drift penalty could be computed

Every donor failed the static-qualification gate (static accuracy ≥ 0.8 under
both sorter configs), so `qualified_templates = []` and no moving-minus-static
comparison was produced.

The static-arm accuracies are the tell:

| donor | peak µV | pol | rescue static acc | tp | fp | fn |
|---|---:|---|---:|---:|---:|---:|
| D03 | 74 | neg | 0.792 | 623 | 79 | 85 |
| D08 | 73 | neg | 0.780 | 623 | 91 | 85 |
| D04 | 122 | neg | 0.791 | 622 | 78 | 86 |
| D07 | 182 | pos | 0.794 | 624 | 78 | 84 |
| D12 | 295 | pos | 0.783 | 619 | 83 | 89 |
| D14 | 255 | neg | 0.785 | 619 | 81 | 89 |

Twelve of fourteen donors sit at **0.77–0.79** with a near-constant
**~80 FP + ~87 FN**, independent of amplitude (74 vs 295 µV) and polarity. A
detection benchmark cannot fail identically for a 74 µV and a 295 µV neuron —
this is an artifact of the scorer, not the pipeline.

(Two donors fail differently and for real reasons: D10/rescue collapses to 0.40
with 327 FP, D01/legacy_style to 0.45 with 366 FN — genuine low-SNR / config
failures, not the systematic floor.)

## Cause

`ground_truth_scores` (v2) matched the injected train against the **pooled spike
river** — every spike in the ~10⁷-spike imec1 sort — with one global 1:1
matcher, then credited each match to its cluster. Within ±0.5 ms of any injected
event there are dozens of unrelated spikes, so ~10 % of injected events were
handed to a background cluster that happened to come first in interval order.
Each stolen event became one FN and one FP for the injected unit's real cluster.
The floor is the background coincidence rate, ~constant across donors.

Full write-up and the fix: [decision 0014](decisions/0014-injected-truth-scoring-is-per-cluster.md).
`ground_truth_scores` now scores each candidate cluster independently
(exclusive match against that cluster's spikes only) and picks the best.

## What happens next (not yet done)

1. ✅ `ground_truth_scores` fixed, `SCORE_SCHEMA` → v3, three regression tests.
2. **Re-score the 14 static arms only**, both configs, and confirm the ~0.78
   floor is gone and recovery is sensible across 73–295 µV before spending
   compute on moving arms.
3. Freeze a new **C2 v4** prespec/output namespace with the Luke-calibrated rigid family
   (~4–5 / 10–12 / 20–25 µm) per
   [decision 0013](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md);
   the `15 / 40 / osc-20` set is stale.
4. Run the C2 v4 moving arms and compute the drift penalty.

This is not gated by the Luke/Yates comparison; decision 0013 and the frozen
within-Luke dose-response windows provide the current calibration path.
