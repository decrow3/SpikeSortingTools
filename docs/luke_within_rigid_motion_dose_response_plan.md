# Prespec: within-Luke rigid-motion dose–response

**Status:** frozen 2026-09-03 (parameters below), **not yet run**.
**Motivated by:** [`luke_yates_stable_window_overlap_result.md`](luke_yates_stable_window_overlap_result.md)
— the Luke↔Yates matched design failed because Luke imec0 has no motion-quiet
subset. This replaces it as the primary analysis.
**Runs parallel to** C2 v3; gates nothing, gated by nothing.

## 1. The question

> Within one recording — same probe, same anatomy, same acquisition — does
> sorting quality deteriorate systematically as **rigid** motion increases across
> Luke imec0's observed ~4–23 µm / 120 s range?

Removing the cross-animal confounds (depth, probe, preprocessing history,
acquisition) leaves motion as the manipulated variable. It is observational, not
interventional — C2 v3 is the interventional counterpart — but the dose axis is
wide (~5×) and internal.

### Pre-committed readings

| Observation | Reading |
|---|---|
| QC-qualified units/mm and per-unit quality are near-normal in the quietest windows and degrade **monotonically** as rigid excursion (or speed) rises | The neural signal is present; motion handling is the main limiting factor on imec0. Directly supports the C2 v3 / D2 line. |
| **No** relationship across the full ~5× motion range | Motion is less central to the current sorting deficit than assumed; priority shifts to detection / clustering / denoising. |
| Flat then collapse above a threshold | Report the threshold; it calibrates where correction has to succeed. |
| Degradation tracks **session time** more than motion (partial correlation) | The apparent motion effect is drift-accumulation / electrode settling / state; motion per se is not demonstrated. |

## 2. Window selection — frozen, motion-signature only

- Source: `testing/outputs/luke_yates_stable_window_overlap/window_signatures.csv`,
  rows `dataset == Luke`, `probe == imec0`, `estimator == medicine` (primary).
  87 non-overlapping 120 s windows, native start `t.min() + k·120`.
- **Dose axes:** `rigid_excursion_um` (primary), `rigid_speed_um_s` (secondary).
  Both recorded per window; both used as independent univariate doses.
- **Selection rule:** sort the 87 windows by `rigid_excursion_um`; take the
  windows at rank indices `round(linspace(0, 86, 24))`. This is deterministic,
  spans both tails, and is fixed before any sort. → **N = 24**.
- **Speed coverage check (pre-sort, allowed):** if the 24 selected windows do not
  cover the 10th–90th percentile of the full `rigid_speed_um_s` distribution,
  add the windows nearest each uncovered speed decile, up to **N = 28**. Record
  whether this fired.
- Any window failing increment 1's QC (`finite_fraction < 0.9` or
  `max_time_gap_s > 3·dt_median_s`) is dropped and the next rank substituted.
- The resulting native start-time list is written to
  `docs/luke_within_rigid_motion_windows.frozen.json` by the selection step and
  **not changed afterwards**.

## 3. Sorting — the production RESCUE graph, unchanged

Each window is cut with `ladder_snippets.build_snippet` (full probe, all 384
channels; the RESCUE graph handles ch-191 and blanking) from the **accepted
imec0 recording**, then `ladder_l1.l1_run` with `ladder_sorter.RESCUE`
(`do_correction = 0`, `nblocks = 0`) and the standard curation stage.

- This is the current production pipeline, so the result speaks directly to it.
- No external and no internal motion correction — that is the point: we are
  measuring what uncorrected KS4 does as a function of motion.
- 3-layer content cache; snippets on `/media/huklab/Data`, never `/mnt`.
- Runtime budget: ~24–28 × (~26 s build + ~90–120 s KS4 + curation + score)
  ≈ 1 h. Get user confirmation on cost before launch.

## 4. Endpoints — per window, frozen definitions

Computed by `score_sort` (`testing/ladder_score.py`) plus a small extension.
None depends on `KSLabel`.

| # | Endpoint | Definition |
|---|---|---|
| E1 | compact-event density /mm | `detect_peaks` (both polarities, `locally_exclusive`) on the bandpassed voltage, kept if energy within ±3 channels ≥ 0.45 of the event's total (the D2b-2 compactness gate), ÷ sampled probe mm |
| E2 | fraction of events assigned | share of E1 events with a sorted spike of a non-noise unit within ±0.5 ms and 40 µm |
| E3 | QC-qualified units /mm | units with: refractory-violation fraction ≤ 2× the matched-unit reference **and** bandpass amplitude ≥ 15 µV **and** spikes present in ≥ 60 % of 20 s sub-bins, ÷ sampled probe mm |
| E4 | refractory burden | median RV fraction across E3-qualified units |
| E5 | similar-unit burden | similar good–good pairs (similarity ≥ 0.8 within 100 µm) per qualified unit — `ladder_score` guardrail |
| E6 | within-window waveform stability | per qualified unit, cosine of the first-half vs second-half mean template; report the median |
| E7 | qualified-unit firing rate | median and IQR of per-unit rate among E3-qualified units — reported, **not** a headline (state-confounded) |
| E8 | KSLabel-free fragmentation | count of qualified units whose train has a spatially-close (≤ 40 µm), non-coincident, refractory-complementary partner unit (the `ladder_stitch` family primitive), ÷ qualified units |

## 5. Analysis — frozen

For each endpoint E1–E8, across the N windows:

1. **Primary:** Spearman ρ vs `rigid_excursion_um` and vs `rigid_speed_um_s`,
   with 2000-sample bootstrap CI. Mann–Kendall trend test.
2. **Confound control:** partial Spearman of each endpoint vs motion **given**
   `window_start_recording_s` (session time), and vs session time given motion.
   Report both partials. A motion effect claim requires the motion partial to
   survive.
3. **Estimator sensitivity:** repeat step 1 with the window-level dose taken from
   `ks-motion` and `dredge-motion` instead of `medicine`. The **rank** ordering
   of windows is expected to be more stable than the magnitudes; report the
   cross-estimator rank correlation of the dose and whether the endpoint
   correlations keep their sign.
4. **Report** a single figure per endpoint (endpoint vs excursion, coloured by
   session time) and one summary table (ρ, CI, MK p, partials, per-estimator
   sign).

No p-value threshold gates a conclusion; the reading table in §1 is applied to
the pattern across E1–E8 together, not to any single test.

## 6. What is deliberately excluded

- `KSLabel == good` counts, total spikes, stable-bin occupancy as endpoints.
- Any tuning of the RESCUE graph or curation against these endpoints — this
  measures the fixed pipeline, it does not optimise it.
- Cross-animal comparison — that is the demoted §B arm of
  [`luke_yates_stable_period_comparison_plan.md`](luke_yates_stable_period_comparison_plan.md).

## 7. Reproducibility

- Selection + analysis: `testing/luke_within_rigid_motion_dose_response.py` (to build)
- Tests: `testing/test_luke_within_rigid_motion_dose_response.py` (to build)
- Frozen window list: `docs/luke_within_rigid_motion_windows.frozen.json`
- Outputs: `testing/outputs/luke_within_rigid_motion_dose_response/`
