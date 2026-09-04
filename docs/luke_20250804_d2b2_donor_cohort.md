# D2b-2: a compact injected-donor cohort — and why the old one was flat

**Date:** 2026-09-03
**Advances:** Phase D / D2b-2 of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Modules:** `testing/ladder_donors.py`, `testing/luke_d2b2_donor_cohort.py`,
`testing/luke_d2b2_donor_sanity.py`
**Output:** `testing/outputs/luke_d2b2_donor_cohort/donor_templates.npz` — 14
compact imec0 donors, 73–295 µV, both polarities

> **QUALIFICATION UPDATE — 2026-09-03.** The cohort construction and waveform
> measurements stand. Only six donors were included in the original static
> sanity sample, and that scoring predates decision 0014's per-cluster scorer.
> Therefore neither “all 14 sanity-passing” nor the six historical accuracies is
> a current qualification result. All 14 must pass static qualification under
> both sorter configurations with the corrected scorer before entering C2.

## TL;DR

The pilot injected-truth cohort was 10 flat plateaus from 3 imec1 units. The
replacement is 14 spatially-compact real neurons from the imec0 sort, built as
de-whitened Kilosort template shapes scaled to their true bandpass-STA µV. Six
sampled donors appeared to recover at accuracy ≥ 0.92 under the historical
scorer; corrected qualification is pending. A sanity check on the way
caught that `cluster_Amplitude` is not µV (≈ 4–7× small). The sharp very-high-SNR
stratum still needs synthetic templates (D2b-3).

## Why the pilot donors were flat

The C2 / D2b-1 pilot donors (`luke_injected_ground_truth_pilot`) are:

- **10 templates from 3 imec1 units** (265, 294, 338) — not 10 neurons. One of
  the three (294 → T04/T05) has a per-event `common_mode_ratio` of 2.2 and
  `local_energy_fraction` 0.08: it is common-mode, not a localised neuron.
- **single reviewed events, not spike-triggered averages** — one snippet of an
  **unfiltered** (`is_filtered: false`) recording. A single unfiltered snippet
  is dominated by LFP and shared broadband voltage, both spatially flat, so
  every "template" is a ~200 µV plateau across the whole ±16-channel crop
  (energy-within-±3-channels ≈ 0.22, half-energy width = the full 33 channels).

That is why D2b-1's waveform-cosine metric was uninformative and why the
drift-penalty magnitudes cannot be assumed to transfer to real compact neurons.

## The replacement: de-whitened Kilosort templates

`ladder_donors.build_donor_cohort` takes an existing imec0 sort and, for each
well-isolated good unit (≥ 800 spikes, ContamPct ≤ 8 %):

1. de-whitens the KS `templates.npy` row with `whitening_mat_inv.npy` — this is
   the spike-triggered average KS already built after its internal
   high-pass + CAR + whitening, mapped back to sensor space — used for **shape**
   (cleaner than a raw STA, which keeps residual imec0 common-mode);
2. crops ±16 channels around the peak, unit-peak-normalises;
3. gates on a real amplitude-decay footprint (≥ 45 % of per-channel energy
   within ±3 channels, half-energy width ≤ 16 channels);
4. **scales to µV by the unit's bandpass (300 Hz–6 kHz) spike-triggered-average
   peak** — *not* `cluster_Amplitude`, which is not µV and runs ~4–7× small (the
   first cohort was injected at 22–70 µV instead of the real 150–275 µV and 5 of
   6 sanity donors failed to sort);
5. selects a spread across true-µV amplitude bands (< 120 / 120–200 / ≥ 200 µV)
   × polarity.

One bandpass STA per candidate is read from the conditioned recording; no new
manual review.

## The new cohort

| | pilot (imec1, single event) | new (imec0, de-whitened) |
|---|---:|---:|
| independent neurons | **3** | **14** |
| energy within ±3 ch (median) | 0.22 | **0.73** |
| half-energy width, ch (median) | **33** (whole window) | **1** (max 5) |
| bandpass-STA peak µV | 36–270 (plateau-inflated, meaningless) | **73–295** (median 135) |
| polarity | 9 neg / 1 pos | **7 neg / 7 pos** |

Compact, correctly scaled, both polarities, on the probe the promotion question
is about.

## What the historical sanity check found

`luke_d2b2_donor_sanity.py` injects each donor static into the C2 quiet
background and checks it recovers at accuracy ≥ 0.8 (the plan-C bar). The **first
cohort failed 5/6** — and that is how the `cluster_Amplitude` scaling bug was
caught. With the bandpass-STA scaling, **6/6 pass**:

| donor | µV | polarity | accuracy |
|---|---:|---|---:|
| D02 | 274 | pos | 0.99 |
| D03 | 74 | neg | 0.98 |
| D08 | 73 | neg | 0.99 |
| D01 | 159 | neg | 0.92 |
| D12 | 295 | pos | 0.99 |
| D14 | 255 | neg | 0.99 |

Median 0.99, spanning the full 73–295 µV range and both polarities. This
demonstrated that the amplitude-scaling fix was directionally sensible, but it
does not qualify the full cohort under the corrected scorer.

## The injectable dynamic range on Luke imec0

The compact clean good units span **73–295 µV** (bandpass-STA peak). The larger
"good" units in the sort are broad-footprint or contaminated, so the cohort
naturally thins above ~250 µV. A genuinely sharp, very-high-SNR stratum for
D2b-3 / criterion 4 is best rounded out with **synthetic** parametric templates
(controlled sharpness × SNR × polarity) on top of these 14 real anchors.

## Next

- **D2b-3** — synthetic compact-template extension for the sharpness axis, then
  re-run D2b-1 (field-error tolerance) and the C2 drift penalty on the full
  cohort, per amplitude/sharpness stratum. Only then freeze the
  waveform-preservation guardrail (criterion 4).
- **D2a** — the full-session external non-rigid field, judged against the
  provisional pre-sort gate and the D2b-1 envelope.

## Limits

- µV scale is the bandpass-STA peak over ~400 spikes on ±4 channels — a real
  measurement, but a raw STA still carries a little residual imec0 common-mode.
- Donors are from the *rescue* imec0 sort — the pipeline under test. A donor is
  a unit that pipeline already recovers; injecting it back tests
  motion/interpolation handling, not de-novo detection.
- Single background window (C2 imec1 quiet); a second window is still owed.
