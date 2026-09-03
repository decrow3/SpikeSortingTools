# Increment 1 result: the motion-overlap feasibility gate FAILS

**Run:** 2026-09-03, `testing/luke_yates_stable_window_overlap.py` (committed
`5808bc4`), prespec
[`luke_yates_stable_period_comparison_plan.md`](luke_yates_stable_period_comparison_plan.md)
§3. Outputs in `testing/outputs/luke_yates_stable_window_overlap/`
(`window_signatures.csv`, `overlap_gate.json`, `overlap_scatter.png`).

## Verdict

**FAIL, decisively, under every common estimator.** Zero of 87 Luke imec0
120 s windows fall inside the Yates-Q75 overlap box.

| Estimator | Yates-Q75 box (rigid excursion µm) | Luke imec0 windows in box | Luke imec0 windows passing *rigid-excursion axis alone* | Yates unique quiet intervals |
|---|---:|---:|---:|---:|
| `medicine` (primary) | 3.64 | **0 / 87** | 0 / 87 (Luke min 4.22) | 9 |
| `ks-motion` | 1.01 | **0 / 87** | 0 / 87 (Luke min 1.67) | 15 |
| `decentralized-motion` | 1.36 | **0 / 87** | 0 / 87 (Luke min 5.23) | 16 |

The Yates side is fine (9–16 unique quiet intervals, > the 6 required). The
failure is entirely on the Luke side, and it is not marginal.

## What fails, and what does not

- **Rigid excursion is the wall.** Per 120 s window, Luke imec0 rigid P95−P5:
  median 11.0 µm, **minimum 4.22 µm** (primary estimator). Yates: median ~3.0 µm,
  Q75 3.64 µm. **Luke's single quietest 2-minute window still drifts more than
  the Yates 75th percentile.** This holds under 3 of 4 estimators; under
  `ks-motion` (most favourable to Luke) Luke's quietest window (1.67 µm) is only
  comparable to a typical Yates window, and Luke's median window (4.0 µm) is
  still ~4× Yates.
- **Rigid speed also excludes most Luke windows** under the primary estimator
  (Luke median 0.58 µm/s, min 0.23; Yates Q75 0.35; 11 % of Luke passes).
- **Non-rigid gradient is *not* a Luke problem.** Depth-span-normalised, Luke
  imec0 sits at 1.7 µm/mm median — at or below Yates (2.3–3.7). 98 % of Luke
  windows pass this axis. Luke's motion is dominantly rigid translation, not
  differential/non-rigid deformation, at this timescale.

## Estimator disagreement — real, but the conclusion survives it

Full-session rigid P95−P5 for the *same* Luke imec0 recording:

| Estimator | Luke imec0 full-session (µm) | Luke imec0 median per-120 s window (µm) |
|---|---:|---:|
| `ks-motion` | 6.6 | 4.0 |
| `dredge-motion` | 15.3 | 9.5 |
| `medicine` | 21.9 | 11.0 |
| `decentralized-motion` | 29.5 | 20.5 |

A ~4.5× spread on the absolute magnitude. So *how much* Luke moves is uncertain.
But every estimator agrees Luke moves several-fold more than Yates
(Yates full-session: 1.5–4 µm across estimators), and none yields a usable set of
Yates-quiet-equivalent Luke windows.

### Discrepancy to resolve — the "1.28 µm" figure

`docs/decisions/` and the rescue status memory repeatedly state *"DREDGE rigid
motion on Luke0804 imec0 is ~1.28 µm total over the session (6.4 % of one site
pitch)"* and use it to argue motion is negligible on imec0 (e.g. that lack of
motion correction cannot explain the amplitude-truncation differences). **No
motion array on disk reproduces 1.28 µm.** The `dredge-motion` array under
`.../motion/dredge-motion/` shows 15.3 µm full-session rigid excursion and 9.5 µm
median per-120 s-window. The "1.28 µm" sidecar was flagged in the plan as
rigid-only and QC-unqualified with `weights_thresh` entirely non-finite; this
result suggests it was a degenerate estimate and the imec0 "motion is negligible"
caveat should be re-examined.

## Consequence for the stable-period comparison

The prespec stop rule (§2, increment 1): *"If it fails, the motion-matched design
is not viable as written — report and stop; do not silently widen the box."*

**There is no genuinely motion-quiet subset of Luke_20250804 imec0.** The user's
premise for un-deferring the depth-resolved biological comparison — select
stable windows so motion handling is off the critical path — does not hold:
Luke's calmest epochs are still in the motion regime that this design intended
only as the *control* arm. The `dataset × motion-regime` interaction cannot be
run within Luke, because Luke has only the one regime.

### What this does and does not tell us

- It **does not** answer "is there healthy neural signal in Luke when the
  recording behaves itself" — the recording never behaves itself at the
  Yates-quiet level, so quiet-window selection cannot isolate that question.
- It **does** strengthen the case that motion handling is on the critical path
  for essentially all of Luke's data (C2 v3 / D2), not a secondary lever that
  window selection could sidestep.
- Non-rigid deformation is not the issue at 120 s scale; **rigid translation
  magnitude and rate** are. That is squarely in scope for a rigid or
  low-nonrigid-order correction — consistent with the plan's position that
  `nblocks=1` alone is insufficient but a better rigid *field* might help.

## Options (need a decision; not started)

1. **Accept the FAIL and stop the motion-matched arm.** Fold the finding into the
   plan: Luke imec0 is a persistently-moving recording; the "viable when quiet"
   question is not separable from motion correction. Resolve the 1.28 µm
   discrepancy in the decision log.
2. **Weaker salvage — "best-case Luke".** Compare Luke's quietest decile
   (rigid excursion ~4–6 µm) against Yates as a *generous upper bound*, with its
   own prespec and explicitly weaker inference. If Luke is deficient even there,
   informative; if not, inconclusive (the residual motion confound remains).
3. **Invert to a within-Luke motion-gradient analysis.** Luke does span
   4–23 µm/120 s. Ask whether unit quality / units-per-mm degrade monotonically
   with window motion *within Luke imec0* — no Yates needed, no matched-depth
   requirement, uses the high-motion controls already enumerated.
