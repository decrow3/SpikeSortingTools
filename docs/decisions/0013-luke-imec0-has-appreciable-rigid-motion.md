# 0013 — Luke imec0 has appreciable rigid motion; the 1.28 µm sidecar is withdrawn

**Status:** Adopted 2026-09-03
**Corrects a premise used in:** the Phase A2 "tension" paragraph of
[`../pipeline_improvement_plan.md`](../pipeline_improvement_plan.md), decision
[0009](0009-cross-sort-comparisons-must-be-unit-matched.md)'s point 5, and the
rescue-status memory
**Evidence:** [`../luke_yates_stable_window_overlap_result.md`](../luke_yates_stable_window_overlap_result.md)
(run `testing/luke_yates_stable_window_overlap.py`, commit `5808bc4`)

## Problem

Several analyses have cited *"DREDGE rigid motion on Luke0804 imec0 is ~1.28 µm
total over the session (6.4 % of one site pitch)"* and used it to treat imec0 as
effectively stationary — e.g. to argue that lack of motion correction cannot
explain the amplitude-truncation differences, and that any motion-driven
re-clustering on imec0 would have to be non-rigid.

That 1.28 µm figure came from a rigid-only, QC-unqualified sidecar with
`weights_thresh` entirely non-finite. It is **not reproduced by any accepted
motion estimate on disk.** The four estimator arrays under
`.../dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion/` give, for
full-session rigid P95−P5:

| Estimator | Luke imec0 full-session rigid range | median per 120 s window |
|---|---:|---:|
| `ks-motion` | 6.6 µm | 4.0 µm |
| `dredge-motion` | 15.3 µm | 9.5 µm |
| `medicine` (MEDiCINe) | 21.9 µm | 11.0 µm |
| `decentralized-motion` | 29.5 µm | 20.5 µm |

Every accepted estimator puts imec0's rigid motion at least ~5× above 1.28 µm and
several-fold above the Yates comparison recording (Yates full-session rigid range
1.5–4 µm across the same estimators). The motion-overlap feasibility gate found
**zero** of 87 Luke imec0 120 s windows quiet enough to reach the Yates 75th
percentile.

## Decision

1. **The 1.28 µm value is withdrawn from all inferential use.** Replace every
   occurrence that supports a "motion is negligible on imec0" claim with:

   > Earlier analyses cited a 1.28 µm rigid range from a degenerate,
   > QC-unqualified sidecar. That value is not reproduced by the accepted on-disk
   > motion estimates and is withdrawn. Multiple independent estimators agree
   > that Luke imec0 has appreciable rigid motion (~7–30 µm full-session,
   > ~4–20 µm per 120 s window), though they disagree substantially on its
   > magnitude.

2. **The estimator disagreement (~6.6 vs ~29.5 µm) is a calibration/estimation
   uncertainty, not a premise error.** It is carried forward as an open
   quantification problem (which estimate to trust is unresolved), but it does
   **not** reinstate "imec0 is stationary" as a working assumption.

3. **At 120 s scale the dominant Luke-vs-Yates motion difference on imec0 is
   rigid translation magnitude and rate, not non-rigid deformation.** Luke's
   depth-span-normalised non-rigid gradient (~1.7 µm/mm median) is at or below
   Yates. Descriptions of Luke's central problem as "non-rigid motion from
   headpost twisting" should be qualified: that mechanism may still dominate at
   faster timescales, on imec1, or in other sessions, but the present analysis
   says rigid displacement is what most separates imec0 from the known-good
   recording.

## Consequences

- The Phase A2 "tension" (imec0 re-clustering must be non-rigid *or* the estimate
  is wrong) resolves toward **the estimate was wrong**. Rigid motion on imec0 is
  a live candidate mechanism for the re-clustered losses, not an excluded one.
- This makes the conventional rigid motion-correction route (C2 v4 / D2) more
  promising, not less: rigid displacement is the regime standard correction
  handles best.
- C2 v4's first motion family is set to **rigid, Luke-calibrated** (~4–5 →
  ~10–12 → ~20–25 µm rigid excursion, with representative speed profiles) rather
  than elaborate non-rigid trajectories. See `../pipeline_improvement_plan.md`
  §C2.

## What would reopen this

A documented, QC-passing motion estimate showing imec0 rigid motion materially
below the `ks-motion` figure, with support diagnostics — or evidence that all
four on-disk estimators share a common inflation artifact (e.g. a stimulus-locked
or artifact-locked component being read as tissue motion).
