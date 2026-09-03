# Prespec: within-Luke rigid-motion dose–response

**Status:** frozen 2026-09-03; **revised 2026-09-03 after Codex review** (findings
1–12), **not yet run**.
**Motivated by:** [`luke_yates_stable_window_overlap_result.md`](luke_yates_stable_window_overlap_result.md)
— the Luke↔Yates matched design failed because Luke imec0 has no motion-quiet
subset. This replaces it as the primary analysis.
**Runs parallel to** C2 v3; gates nothing, gated by nothing.

## 1. The question — and what this design can and cannot answer

> Within one recording — same probe, same anatomy, same acquisition — do sorting
> metrics **covary** with estimated rigid motion across Luke imec0's own
> ~2–25 µm / 120 s range?

Motion here is an **observed exposure, not a manipulation.** Within-Luke sampling
removes fixed animal / probe / anatomy differences. It does **not** remove
time-varying neural state, stimulus regime, behaviour, noise/artifact structure,
or electrode settling, and there is **no normal reference** in this design (Luke's
quietest window is still above the intended quiet regime — see the failed
overlap gate). The interventional counterpart is **C2 v3**.

### Pre-committed readings (revised — no causal or baseline claims)

| Observation | Reading |
|---|---|
| The **primary** metric (E3, QC-qualified units/mm) declines monotonically with the consensus rigid-excursion dose, the decline **survives the session-time partial and the nonlinear session-time sensitivity**, and the sign **agrees across the three concordant estimators** | Sorting quality covaries with estimated rigid motion within this session. **Consistent with** motion being a live limiting factor; does **not** establish it as the main one, and does **not** establish that quiet windows are healthy — C2 v3 is the test for both. |
| No covariation across the full dose range | **Not** evidence that motion is unimportant. Compatible with: every window already past a failure threshold; exposure measurement too noisy (the estimators disagree — §2); 120 s giving too few units for power; the effect being non-monotonic. |
| The consensus dose and the MEDiCINe sensitivity dose **disagree in sign** on the primary metric | **Exposure unresolved.** Report as such; draw no motion conclusion. |
| Degradation tracks `window_start_recording_s` more than the dose in the partials | The association is drift-accumulation / settling / a session-time-locked state change, not rigid motion per se. |

**Not endpoints** (carried from `pipeline_improvement_plan.md` §5): `KSLabel`
counts, total spikes, stable-bin occupancy, population medians across sorts.

## 2. Motion exposure — the dose axis (revised, finding 1)

The four motion estimators **do not agree on which windows are quiet.**
Cross-estimator Spearman of the per-window rigid-excursion rank, Luke imec0, 87
windows joined on `time_interval_id`:

| | MEDiCINe | ks-motion | dredge-motion | decentralized |
|---|---:|---:|---:|---:|
| MEDiCINe | — | 0.11 | 0.07 | 0.14 |
| ks-motion | | — | **0.85** | **0.69** |
| dredge-motion | | | — | **0.78** |

`{ks-motion, dredge-motion, decentralized-motion}` are mutually concordant
(0.69–0.85); **MEDiCINe is the outlier** and is demoted from primary.

- **Primary dose = the mean percentile rank of `rigid_excursion_um` across the
  three concordant estimators** (`ks-motion`, `dredge-motion`,
  `decentralized-motion`), joined on `time_interval_id`. A secondary speed dose
  is the same construction on `rigid_speed_um_s`.
- **MEDiCINe is a sensitivity arm only.** Its per-window excursion is also
  reported; a primary result whose sign flips under the MEDiCINe dose is
  **"exposure unresolved"**, not a finding.
- **Exposure-validity rule (frozen):** a motion-covariation claim for any
  endpoint requires the Spearman sign vs the consensus dose to **agree with the
  sign vs each of the three concordant estimators' individual excursion**. The
  cross-estimator sign table is reported for every endpoint.

## 3. Window selection — frozen

- Source: `testing/outputs/luke_yates_stable_window_overlap/window_signatures.csv`,
  `dataset == Luke`, `probe == imec0`. Increment 1 QC applied
  (`finite_fraction ≥ 0.9`, `max_time_gap_s ≤ 3·dt_median_s`) per estimator
  before the join; a `time_interval_id` is kept only if all three concordant
  estimators pass.
- **Selection rule:** rank the surviving intervals by the consensus
  `rigid_excursion_um` percentile rank; take the intervals at
  `round(linspace(0, n-1, 24))`. Deterministic. **N = 24.** No speed top-up
  (finding 9); speed-dose coverage is reported descriptively, and if the 24
  windows do not span the 10th–90th percentile of the consensus speed rank the
  primary speed analysis is marked underpowered rather than the sample changed.
- **Snippet start = `window_start_recording_s`** (0, 120, 240, …) — the plain
  recording-relative boundary, *not* the `−dt/2` motion-trace convention
  (finding 10). An integration check asserts the first and last included
  motion-bin centres fall inside `[start, start+120)`.
- Written once to `docs/luke_within_rigid_motion_windows.frozen.json` with the
  source CSV path, its SHA-256, the code+prespec git commit, and the selected
  `time_interval_id`s. `load` revalidates all of it (finding 12).

## 4. Sorting — the production RESCUE graph, unchanged

Each window: `ladder_snippets.build_snippet` (full probe, 384 ch) from the
**accepted imec0 RESCUE recording**, then `ladder_l1.l1_run` (sorter = `None` =
frozen RESCUE, `do_correction`/`nblocks = 0`) + the standard curation stage.

- **Storage:** a full-probe int16 120 s snippet ≈ **2.76 GB**; 24 windows ≈
  **66 GB** before sorter caches. Snippets on `/media/huklab/Data`, never `/mnt`.
- **Outcome-blind pilot first (finding, Phase 2 rec):** build + sort **one**
  window (the median-dose rank), record wall-clock and disk, **before** the
  batch. If per-window cost is off the ~2–3 min / ~3 GB estimate, amend this
  section before continuing.
- **`probe_mm`** for the per-mm endpoints = span of `channel_positions.npy` y
  plus one full site pitch (half a pitch beyond each end contact). Frozen here.
- Depth is held constant across windows (same 384 ch every time). A
  depth-stratified (shallow / mid / deep thirds) summary of E3 is reported as a
  sensitivity — not a separate primary test.

## 5. Endpoints — per window, frozen (revised: findings 3, 5, 6)

All operate over **every curated cluster**; `KSLabel` is never read.

### E3 — QC-qualified units (the anchor; E2/E4/E5/E6/E7/E8 all condition on it)

A cluster qualifies iff **all** of:

| Criterion | Frozen value | Source / rationale |
|---|---|---|
| spike count | ≥ **150** in the 120 s window | below this the RV fraction is not estimable (<150 ISIs); also the floor for E6 half-splits (≥ 60 per half) |
| bandpass amplitude | ≥ **15 µV** peak\|·\| of the KS template row mapped to µV via `whitening_mat_inv` and the unit's bandpass spike-triggered-average peak (the D2b-2 `ladder_donors` scaling), computed once per cluster | `ladder_donors` scaling is the repo's only validated template→µV path; 15 µV ≈ the rescue detection floor |
| refractory-violation fraction | ≤ **0.01** (ISI < 1.5 ms) | a flat 1 % contamination ceiling — not "2× an unfrozen reference"; the Phase A v2 matched-unit cohort sits at ~0.1–0.5 % |
| presence | spikes in ≥ **9 of 12** 10 s sub-bins | 10 s bins (not 20 s) so the rule means "roughly continuous", not "present in 4 of 6" |

`E3_qualified_units_per_mm = n_qualified / probe_mm`. **`n_qualified` is reported
per window**; windows with `n_qualified < 8` flag the primary test as
low-power.

### The rest

| # | Endpoint | Definition (frozen) |
|---|---|---|
| E4 | refractory burden | median RV fraction across E3-qualified units |
| E5 | similar-unit burden | pairs of **E3-qualified** units with template cosine ≥ 0.8 and peak-channel depth within 100 µm, ÷ n_qualified. Own implementation — **not** `ladder_score.nearby_similar_good_pairs` (which uses `sort["good"]`) |
| E6 | within-window waveform stability | per E3-qualified unit with ≥ 60 spikes per half: cosine of the mean multi-channel waveform (±1.3 ms, top 8 channels by amplitude) first-half vs second-half; report the median. Traces from the built snippet |
| E7 | qualified-unit firing rate | median + IQR of per-unit rate among E3-qualified units. **Reported, never a headline; never used as a covariate** (rate can be an effect of motion) |
| E8 | fragmentation index | pairs of E3-qualified units that are (peak-depth ≤ 40 µm apart) **and** (≤ 5 % of the smaller train coincident within ±0.5 ms) **and** (refractory-clean union, RV ≤ 0.01); n_flagged_units ÷ n_qualified. Pure pair metric, frozen thresholds — **no** `stitch_families` (loads `cluster_KSLabel.tsv`) |

### E1 / E2 — demoted to input/context integrity (finding 6)

| # | Metric | Definition |
|---|---|---|
| C1 | detected-event density /mm | `detect_peaks` (both polarities, `locally_exclusive`, thr 4) on the bandpassed common-referenced snippet, ÷ probe_mm. **Raw count — no template-derived compactness gate** (the 0.45 gate was calibrated on averaged templates, not single events; applying it here is unvalidated) |
| C2 | fraction of detected events near a qualified unit | share of C1 events within ±0.5 ms and 40 µm of an **E3-qualified** unit's spike |

C1/C2 are **context**, reported alongside the endpoints and used to interpret
them (e.g. a C1 rise with motion signals more detections, not better sorting),
**not** dose-response endpoints in their own right. A separate future task may
validate a single-event compactness gate and promote a compact-event metric.

## 6. Analysis — frozen (revised: findings 4, 7, 11)

### Primary test — one

> **Spearman ρ of E3 (`E3_qualified_units_per_mm`) vs the consensus
> rigid-excursion dose**, across the 24 windows, with a 2000-sample bootstrap CI.
> Pre-registered expected sign: **negative** (more motion → fewer qualified
> units/mm). Supported only if: ρ < 0, the bootstrap CI excludes 0, the sign
> survives the **partial** given `window_start_recording_s` **and** a
> LOESS/quadratic session-time detrend, **and** the cross-estimator sign table
> (§2) agrees.

### Supportive / diagnostic — everything else

E4–E8 and C1/C2 vs {consensus excursion rank, consensus speed rank}: Spearman +
CI, tie-corrected Mann–Kendall (report τ_b and the permutation p), session-time
partial. **Descriptive only** — p-values are reported unadjusted and labelled
descriptive; a global concordance statement ("k of 7 supportive endpoints move in
the pre-registered direction") is the summary, not per-endpoint significance.

Pre-registered supportive directions: E4 ↑, E5 ↑, E6 ↓, E8 ↑ with motion; E7 no
prediction; C1 ambiguous; C2 ↓.

### Confound handling

- Session time: linear-rank partial **and** a nonlinear (quadratic) detrend
  sensitivity. Both reported.
- Independent state covariates (stimulus block, running/pupil) folded in **only
  if** an independent source exists for this session; sorted firing rate is
  **not** used as a covariate.
- MK and bootstrap CIs assume window-level independence; a block-bootstrap
  (adjacent-window blocks) robustness check is reported.

### Estimator sensitivity

Repeat the primary and every supportive Spearman with the dose taken from each of
the 4 estimators individually (joined on `time_interval_id`). Emit the
sign-agreement / rank-correlation table. This is the §2 exposure-validity gate.

## 7. Deliberately excluded

`KSLabel` counts / total spikes / stable-bin occupancy as endpoints; any tuning
of the RESCUE graph or curation against these endpoints; cross-animal comparison
(that is the demoted §B arm of
[`luke_yates_stable_period_comparison_plan.md`](luke_yates_stable_period_comparison_plan.md)).

## 8. Reproducibility

- `testing/luke_within_rigid_motion_dose_response.py` — phase 1 (consensus-dose
  selection) + phase 4 (statistics) implemented + tested; phases 2–3
  (build/sort, endpoint extraction) implemented against the frozen §5 defs, with
  the trace-level parts (E1/C1, E6) exercised only on synthetic data until the
  pilot.
- `testing/test_luke_within_rigid_motion_dose_response.py`
- `docs/luke_within_rigid_motion_windows.frozen.json` (written once by `--select`)
- Outputs: `testing/outputs/luke_within_rigid_motion_dose_response/`
