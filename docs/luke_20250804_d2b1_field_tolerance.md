# RETRACTED: D2b-1 motion-field tolerance evaluation

> **RETRACTED PENDING RERUN — 2026-09-03.** This tolerance envelope perturbed an
> oracle field that was not the inverse of the injected motion on Luke's
> four-column geometry. Its quantitative tolerances and provisional pre-sort
> gate are withdrawn. The tables remain for provenance and must not qualify a
> real field until the corrected geometry-aware experiment is rerun. Field
> qualification now fails closed without independently measured support,
> split-half reproducibility, and estimation-error evidence.

**Date:** 2026-09-02
**Advances:** Phase D / D2b-1 of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Modules:** `testing/ladder_motion.py` (field perturbations + `waveform_preservation`),
`testing/luke_rescue_d2b1_field_tolerance.py`
**Evaluation:** cached C2 injected recordings, frozen rescue sort, no new injection.

## Historical question and method — results withdrawn

Candidate 2's oracle arm showed the *exact* field closes the severe rigid-drift
penalty (T04, T06 at 40 µm: 0.40 → 0.99). D2b-1 corrupts that exact field —
amplitude gain, temporal lag, temporal smoothing, constant bias, spurious depth
gradient — and asks **how far off an estimated field can be before corrected
voltage performs no better than uncorrected rescue.** That envelope is the
pre-sort gate (D2 sequence step 5): a candidate DREDGE/external field outside it
does not earn a full-session sorting run.

`recovery_fraction = (accuracy − rescue_accuracy) / (oracle_accuracy −
rescue_accuracy)`: 1.0 = the perturbed field is as good as the exact one, 0 =
no better than no correction, < 0 = worse than no correction.

## Retracted historical envelope

### Severe rigid drift (T04, T06 at 40 µm) — forgiving

**Every perturbation tested still beat no-correction.** ±50 % amplitude error, a
20 µm constant bias, a 6 s temporal lag, a spurious 0.7 depth gradient — corrected
voltage stayed ahead of uncorrected rescue in all of them. But the *fraction of
the benefit retained* varies:

| perturbation | level | T04 recovery | T06 recovery | note |
|---|---|---:|---:|---|
| amplitude gain | ×0.75–1.25 | 0.52–0.86 | 0.15–0.40 | over-estimation (×1.25–1.5) makes false positives — T06 ×1.25: 795 FP |
| temporal lag | 2–6 s | 0.99 | 0.31–0.41 | a slow ramp barely moves in 6 s |
| temporal smoothing | σ 3–10 s | 0.94–0.99 | 0.31–0.38 | ramp is already smooth |
| constant bias | 8–20 µm | **0.20–0.43** | **1.0** | donor-split: a residual offset mis-registers a slow target for T04, is harmless for T06 |
| depth gradient | 0.3 / 0.7 | 1.0 / 0.36 | 0.39 / 0.65 | mild spatial error fine; a strong spurious non-rigid structure costs 40–65 % |

Take: for severe rigid drift an estimated field within **±25 % amplitude** and
roughly the right timing keeps most of the benefit; nothing at these levels
flips the win to a loss.

### Oscillation (T04 at 20 µm, 40 s period) — fragile

The oscillatory regime — the one A2's "rapid template flicker" actually
resembles — breaks under errors the rigid regime shrugs off:

| perturbation | level | recovery | verdict |
|---|---|---:|---|
| temporal smoothing | σ 3 s | 1.02 | fine |
| **temporal smoothing** | **σ 10 s** | **0.00** | **correction = no correction** — σ = ¼ period removes the signal |
| constant bias | 8 µm | 1.17 | fine |
| **constant bias** | **20 µm** | **−0.20** | **worse than no correction** |
| amplitude gain | ×0.5–1.5 | 1.1–2.0 | tolerated (and the "exact" field is not even optimal here — see below) |
| temporal lag | 2–6 s | 1.1–1.5 | tolerated |

An oscillatory / fast-flicker field must **preserve temporal components down to
~⅓ of the motion period** and keep bias near zero.

### T01 — the wrong side of the tradeoff, at any field quality

T01's exact-oracle accuracy (0.29) is *below* its uncorrected rescue accuracy
(0.40). No perturbation level makes correction help — a couple cross rescue by
chance but `recovery_fraction` is undefined because the ceiling is below the
floor. **This is a waveform-class exclusion, not a field-quality problem** —
D2b-2's job to establish which class (T01 is the lowest-amplitude donor at
−68 µV; the interpolation spatial blur costs more than the motion for it).

## An aside worth chasing in D2b-3

For T04 oscillation the *exact* field is not optimal: several perturbed fields
(gain ×1.25, ×1.5) sort **better** than the exact trajectory (0.99 vs 0.78),
because the exact correction still leaves 161 FP while a slightly stronger field
suppresses them. The best operating point for oscillatory motion may be a
*deliberately* over-scaled correction. Do not assume "exact = best".

## The waveform-preservation metric — and a donor-cohort problem it exposed

`waveform_preservation` (cosine, peak-amplitude ratio, peak-channel shift of the
spike-triggered average vs the injected template) ran on every condition. Two
things came out:

- Within a donor the cosine is **stable across every perturbation** (~0.85–0.89
  for T04/T06) — interpolation shape-damage is small and roughly constant in
  that regime, so the accuracy differences above are residual *mis-registration*,
  not waveform blur. That is the useful relative signal.
- The absolute numbers are weak because **the C2 pilot donor templates are not
  spatially compact.** Inspected directly (raw `donor_templates.npz`, before any
  cropping): T04's per-channel peak is 242–270 µV *flat* across the whole ±16
  channel window (±160 µm); T06's is 151–170 µV flat; T01's is 16–68 µV, i.e. at
  noise level and flat. None of them decays like a single-neuron footprint — a
  real spike is < 10 % of peak within ~100 µm. `peak_channel_shift` and absolute
  cosine are near-meaningless on a plateau, and T01's corrected STA barely
  correlates with its template (cosine 0.17) because T01 is essentially noise.

**This is a discovery-cohort defect, not a measurement bug.** `observed_waveform`
is correct; the donors are common-mode-contaminated or were never spatially
localised during pilot extraction. Consequences:

- The **waveform-preservation guardrail (criterion 4) cannot be frozen** on these
  donors.
- The drift-penalty *direction* still holds — a broad high-amplitude source
  fragments under motion and correction still helps — but the **magnitudes may
  not transfer to compact real neurons**, which could be either more sensitive
  (sharper spatial gradient → interpolation blur costs more) or less (higher
  local SNR).
- **D2b-2 is now a prerequisite, not a follow-up:** rebuild the donor cohort from
  verified spatially-compact reviewed waveforms (real amplitude decay, ≥ 8 units,
  both polarities) before any correction gate or the waveform guardrail is
  defined.

## Consequence for D2a

The pre-sort field-qualification gate (step 5):

- **Severe rigid drift** needs only a coarse field — ±25 % amplitude, approximate
  timing. If Luke's real motion were a slow ramp, almost any DREDGE estimate
  would do.
- **But Luke's real regime is fast flicker** (A2: ~18–22 ownership flips/hr, no
  slow trajectory). The oscillation result is the relevant one: the field must
  retain fast temporal structure and near-zero bias. That is exactly what
  snippet-scale estimation could not deliver (C2: KS4 `nblocks=6` worse than
  nothing) — so D2a's full-session external estimate has to be judged on its
  **temporal bandwidth and bias**, not just its total displacement.
- Reject a candidate field whose amplitude error is > ~30 %, whose temporal
  bandwidth is below ~⅓ the dominant motion period, or whose support/confidence
  is low across the strip — before it reaches a sorter.

## Limits

- Coarse levels, 2–4 per perturbation; enough to locate a crossover region, not
  to fit a boundary.
- Four (donor, trajectory) pairs. D2b-2 must widen the donor cohort (≥ 8,
  T01–T10 are available incl. T08 positive) and D2b-3 must add the tradeoff axes
  before any selective-correction rule is fitted.
- One background window, negative-compact polarity (T01/T04/T06), 120 s.
- Perturbations applied singly; a real field's errors are correlated.
