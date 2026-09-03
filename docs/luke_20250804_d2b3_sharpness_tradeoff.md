# D2b-3: interpolation cost on compact real donors — Candidate 2's ceiling was inflated

> **RETRACTED PENDING RERUN — 2026-09-03.** The claimed residual “intrinsic to
> resampling” is unsupported because forward injection and inverse correction
> used different spatial operators on the four-column probe. The donor cohort
> remains usable, but all interpolation-cost, kernel, SNR-crossover, and oracle
> recovery conclusions below are withdrawn until geometry-aware forward motion
> and exclusive scoring are rerun from content-invalidated caches.

**Date:** 2026-09-03
**Advances:** Phase D / D2b-3 of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Modules:** `testing/ladder_synthetic_donors.py`, `testing/luke_d2b3_sharpness_tradeoff.py`,
`testing/luke_d2b3_interp_kernel.py`

## The question

Candidate 2's oracle arm (exact field → rescue sort) recovered severe drift to
accuracy ≈ 0.99 — but on the flat plateau pilot donors. D2b-1 warned those
donors are not compact. D2b-3 re-runs the same experiment on **compact real
donors** (D2b-2) plus a synthetic sharpness grid, and asks: with a *perfectly
known* 40 µm field, how much does kriging interpolation still cost, by waveform
class?

`interpolation_cost = static_accuracy(rescue) − moving_accuracy(oracle)`: what a
perfect correction cannot recover.

## The headline

**On compact real donors, a perfectly-known correction leaves a large residual
penalty — the plateau donors overstated how well voltage interpolation works.**

| donor | µV | polarity | static | moving (no corr.) | moving (oracle) | oracle recovers | interp. cost |
|---|---:|---|---:|---:|---:|---:|---:|
| D08 | 73 | neg | 0.99 | 0.37 | **0.09** | **−0.28** | 0.91 |
| D04 | 122 | neg | 1.00 | 0.41 | 0.48 | +0.07 | 0.52 |
| D06 | 149 | pos | 0.99 | 0.26 | 0.49 | +0.23 | 0.50 |
| D01 | 159 | neg | 0.92 | 0.48 | 0.50 | +0.02 | 0.42 |
| D14 | 255 | neg | 0.99 | 0.62 | **0.42** | **−0.20** | 0.57 |
| D02 | 274 | pos | 0.99 | 0.43 | **0.99** | **+0.56** | 0.01 |

- **Only D02 recovers to its static accuracy.** Every other compact donor keeps a
  0.4–0.6 accuracy penalty *even with the exact motion*.
- **The two extremes match Candidate 2 / T01:** the highest-amplitude cleanest
  donor (D02) is fully recovered; the lowest-amplitude one (D08, 73 µV) is
  *destroyed* by the correction — oracle interpolation turns 0 false positives
  into 1079.
- **The middle is muddy.** 120–255 µV donors show `oracle_recovers` from −0.20 to
  +0.23, not ordered by amplitude, compactness, or (cleanly) polarity. At
  11 donors on a 120 s snippet this does **not** resolve a simple selective-
  correction threshold.

## Why the plateau donors lied

A flat template — every channel carrying the same waveform — is trivially
interpolated: shifting it by a fractional channel and re-sampling barely changes
it. A compact single-channel-peak footprint has a steep spatial gradient exactly
where the kriging kernel has to guess, so fractional-channel interpolation
distorts the peak. Candidate 2's "0.40 → 0.99" recovery on T04/T06 was measuring
the easy case.

## Synthetic grid — a build note

The synthetic templates sort cleanly *stationary* only for trough width
≤ ~0.30 ms (0.45–0.55 ms troughs read as LFP to KS4's universal templates and
fail the static bar). Within the sortable range the synthetic donors behave like
the mid-amplitude real ones — `interpolation_cost` 0.3–0.4, `oracle_recovers`
near zero — and add no separating power beyond the real cohort. The
`trough_width_ms` sharpness knob did **not** predict interpolation cost in this
pass; amplitude/SNR and per-unit sorting variance dominate.

## What this means for D2

1. **Candidate 2's ceiling is lower than reported.** A non-rigid motion
   representation applied by voltage interpolation recovers the drift penalty
   *fully* only for the highest-SNR, cleanest units; most compact Luke units keep
   a substantial residual even with a perfect field.
2. **This raises the priority of the architecture gate.** A sorter that handles
   motion by tracking templates rather than resampling voltage (motion-aware
   template matching) would not pay this interpolation cost. D2a is still worth
   running — the real question is whether an *estimated* field on the full
   session helps the units it can help — but the expected upside is now "recover
   the D02-like units and don't harm the rest", not "close the penalty". And the
   kernel check confirms there is no better interpolation to reach for.
3. **A selective-correction rule is still indicated** (correct high-SNR, leave
   low-SNR alone — D08 and T01 are both harmed), but D2b-3 has not pinned the
   threshold.

## Retracted historical kernel comparison

`luke_d2b3_interp_kernel.py` re-ran the oracle-corrected 40 µm arm for D02, D14,
D04, D06 under kriging σ = 20, kriging σ = 40, and inverse-distance σ = 30:

| donor | µV | kriging σ20 | kriging σ40 | IDW σ30 |
|---|---:|---:|---:|---:|
| D02 | 274 | 0.985 | 0.985 | 0.626 |
| D14 | 255 | 0.419 | 0.407 | 0.412 |
| D04 | 122 | 0.477 | 0.503 | 0.490 |
| D06 | 149 | 0.487 | 0.474 | 0.467 |

The original run interpreted kriging σ as inert and IDW as equal or worse.
That interpretation, the claim that the residual is intrinsic, and closure of
the D2c interpolation-kernel axis are all withdrawn pending corrected rerun.

## Limits

- 11 donors, one background window, one drift magnitude (40 µm rigid), 120 s.
  Per-unit KS4 variance is large relative to the effects in the muddy middle.
- Injected donors are units the rescue pipeline already recovers; this measures
  motion/interpolation handling, not detection.
- The synthetic spatial model (`1/(1+(Δy/λ)²)`, λ = 25 µm) may be more compact
  than real footprints — if so it slightly overstates synthetic interpolation
  cost, which is consistent with the mechanism above.
