# Within-Luke rigid-motion dose–response: result

**Run:** 2026-09-04. Code `testing/luke_within_rigid_motion_dose_response.py`
(committed `9b2e431`, Codex v1 + v2 reviews applied; 23 unit tests).
Prespec [`luke_within_rigid_motion_dose_response_plan.md`](luke_within_rigid_motion_dose_response_plan.md)
(frozen `4ee5dc4` / `c3e2250` / `9b2e431`). Frozen window list
[`luke_within_rigid_motion_windows.frozen.json`](luke_within_rigid_motion_windows.frozen.json)
(`8930a06`). Outputs in
`testing/outputs/luke_within_rigid_motion_dose_response/`
(`window_endpoints.csv`, `dose_response.json`, `run_index.json`).

## The question

Within one recording — Luke 20250804 imec0, same probe, same anatomy, same
acquisition — do sorting metrics **covary** with estimated rigid motion across
the recording's own ~2–25 µm / 120 s range? Motion here is an **observed
exposure, not a manipulation**: within-Luke sampling removes fixed animal /
probe / anatomy differences but not time-varying neural state, stimulus regime,
electrode settling, or session-time-locked drift accumulation. There is **no
normal reference** in this design (the failed increment-1 overlap gate showed
Luke's quietest window is still above the Yates-quiet regime). The
interventional counterpart is C2 v4.

## Verdict

**The primary test is null.** Over Luke imec0's observed rigid-motion range,
the QC-qualified unit yield per mm does **not** covary with the consensus
rigid-excursion dose. Several within-unit quality endpoints (waveform
stability, qualified firing rate, train fragmentation) **do** degrade with
motion, consistently across the three concordant estimators — but these are
descriptive-only by prespec and cannot carry an inferential motion claim on
their own.

## Cohort

24 windows of 120 s, selected at even ranks of the consensus rigid-excursion
dose (mean percentile rank of `rigid_excursion_um` across
`{ks-motion, dredge-motion, decentralized-motion}`; MEDiCINe demoted to a
sensitivity arm per prespec §2). Consensus dose rank spans 0.027 → 1.000.
Per-window rigid excursion: ks-motion 1.9–16.1 µm, dredge-motion 4.1–24.6 µm.
Every window sorted on the frozen unwarped RESCUE graph with no motion
correction. `n_qualified` per window 46–118 (median 68); E3 12.0–30.7
qualified units/mm. No window has `n_qualified < 8` → **not low power**.

The three concordant estimators rank-agree on the 24-window dose
(pairwise Spearman 0.73–0.88); MEDiCINe agrees only 0.36–0.39, as expected
from §2.

## Primary endpoint — E3 qualified units/mm vs consensus rigid-excursion dose

| Quantity | Value |
|---|---|
| Spearman ρ | **−0.069**, 2000-sample bootstrap CI **[−0.52, +0.41]** |
| Adjacent-window block-bootstrap CI | [−0.48, +0.45] |
| Mann–Kendall τ_b (tie-corrected) | −0.026, permutation p = 0.88, S = −7 |
| ρ vs consensus **speed** dose | −0.14, CI [−0.60, +0.37] |
| Session-time partial (linear detrend) | −0.41 |
| Session-time partial (quadratic detrend) | −0.45 |
| `exposure_validity` | **`no_association`** ( \|ρ\| < 0.10 ) |
| Per-estimator ρ (medicine / ks / dredge / decentralized) | +0.07 / −0.05 / −0.01 / −0.33 |

### `primary_supported = false`

Frozen `primary_decision_reasons` (all six must hold):

| Reason | Holds? |
|---|---|
| `rho_negative` (ρ < 0) | ✅ (−0.069) |
| `ci_excludes_zero` (bootstrap CI entirely below 0) | ❌ CI [−0.52, +0.41] |
| `linear_partial_survives` ( same sign, \|partial\| ≥ 0.10 ) | ✅ (−0.41) |
| `quadratic_partial_survives` | ✅ (−0.45) |
| `exposure_resolved` | ❌ `no_association` |
| `not_low_power` | ✅ |

Two reasons fail: the raw rank correlation is indistinguishable from zero, and
the exposure is not resolved (the estimators do not even agree on the sign of a
near-zero effect). Per prespec §6, a `no_association` primary result **draws no
motion conclusion**.

### Reading (from the pre-committed table, §1)

> *No covariation across the full dose range* → **Not** evidence that motion is
> unimportant. Compatible with: every window already past a failure threshold;
> exposure measurement too noisy (the estimators disagree — §2); 120 s giving
> too few units for power; the effect being non-monotonic.

The session-time partials (−0.41 linear, −0.45 quadratic) are notable: once
`window_start_recording_s` is removed, a modest negative dose slope appears that
the raw correlation does not show. But `session_time_partial_given_dose` for E3
is +0.62 — session time carries at least as much of the structure as the dose
does. Under the prespec reading-table row 4, that pattern points to
**drift-accumulation / settling / a session-time-locked state change** rather
than rigid motion per se, and in any case the exposure is unresolved, so this is
noted, not concluded. (The dose and session time are themselves correlated at
ρ ≈ 0.48 across the 24 windows, which limits how cleanly the two can be
separated with N = 24.)

## Supportive / diagnostic endpoints — descriptive only

Spearman ρ vs the consensus rigid-excursion dose, with tie-corrected
Mann–Kendall and per-estimator sign. p-values unadjusted and descriptive per
prespec §6.

| Endpoint | Prereg dir | ρ vs excursion | CI | MK p | Cross-estimator | Matches? |
|---|:---:|---:|---|---:|---|:---:|
| **E6 waveform stability median** | ↓ | **−0.66** | [−0.88, −0.31] | 0.0004 | resolved (4/4 negative) | ✅ |
| **E7 qualified firing rate median** | (none) | **−0.59** | [−0.81, −0.20] | 0.006 | resolved (4/4 negative) | — |
| **E8 fragmentation index** | ↑ | **+0.42** | [+0.02, +0.69] | 0.048 | resolved | ✅ |
| C1 detected-event density | ambiguous | +0.49 | [+0.07, +0.81] | 0.019 | **unresolved** (MEDiCINe flips) | — |
| E4 refractory burden median | ↑ | +0.30 | [−0.12, +0.65] | 0.20 | resolved | ✅ |
| C2 events near a qualified unit | ↓ | −0.24 | [−0.63, +0.28] | 0.43 | resolved | ✅ |
| E5 similar pairs / qualified unit | ↑ | −0.11 | [−0.45, +0.24] | 0.61 | unresolved | ❌ |

**Concordance summary (the prespec deliverable): 4 of 5 endpoints with a
pre-registered direction move as predicted.** E5 is the exception (near-zero,
exposure-unresolved).

### What the supportive endpoints suggest (not a conclusion)

A coherent within-unit-degradation pattern that survives the session-time
partial for the two strongest endpoints:

- **E6 (waveform stability, ρ = −0.66):** `session_time_partial_given_dose`
  = 0.08 — almost none of the association is session time. Partial-linear
  detrend −0.63. This is the cleanest motion association in the set.
- **E7 (qualified firing rate, ρ = −0.59):** median qualified-unit rate falls
  from ~7 Hz in the low-dose windows to ~3.6 Hz in the high-dose windows;
  `session_time_partial_given_dose` = 0.23, partial-linear −0.61 — mostly
  survives session-time control. (Firing rate is explicitly **not** a covariate
  in this design — it can be an effect of motion.)
- **E8 (fragmentation, ρ = +0.42):** dose and session time roughly co-equal
  (partial 0.27 vs `session_time_partial_given_dose` 0.29).
- **C1 (+0.49)** rises with motion but is exposure-unresolved (MEDiCINe flips
  sign) and is the endpoint most plausibly inflated by drift artefact adding
  spurious threshold crossings — consistent with C2 (fraction of detected
  events near a qualified unit) trending *down* with motion (−0.24).

Taken together the supportive endpoints are consistent with motion at this
scale **not removing units wholesale** (E3 flat) but **degrading spike
assignment for the units that remain** — smeared waveforms, lower effective
rates, more fragmented trains. This is a hypothesis for C2 v4 to test
interventionally, not a finding of this design.

## What this does and does not establish

- It **does not** establish that rigid motion limits sorting quality on Luke
  imec0. The primary, most conservative endpoint (unit yield/mm) is flat, and
  the exposure axis is unresolved.
- It **does not** establish that motion is unimportant — the null is fully
  compatible with every 120 s window already being past a drift-failure
  threshold (which the increment-1 overlap-gate failure independently
  suggests), with the exposure being too noisily measured, or with N = 24
  windows of 120 s being underpowered for the yield endpoint.
- It **does** show that, descriptively, the within-unit quality metrics
  (E6/E7/E8) covary with the consensus rigid-motion dose in the pre-registered
  direction and consistently across the concordant estimators, with E6 and E7
  largely surviving a session-time partial.
- The interventional question — does correcting the motion recover quality, and
  are the quiet windows actually healthy — remains with **C2 v4**.

## Feed-forward

- **C2 v4 rigid family calibration** is unchanged by this result: the first C2
  motion family stays rigid and Luke-calibrated at the measured per-window
  excursions (~4–5 / ~10–12 / ~20–25 µm; ks-motion / dredge-motion bracket).
- The E6/E7 degradation slopes give C2 v4 a **pre-registered comparison target**:
  if a rigid correction is doing its job, waveform stability and qualified
  firing rate in the corrected high-motion windows should move back toward the
  low-motion-window values.
- No change to decision 0013 (Luke imec0 has appreciable rigid motion; the
  1.28 µm figure is withdrawn) — this analysis neither confirms nor overturns
  it, it operates entirely within the observed motion range.
