# Fast MEDiCINe: Bacon, Luke and Allen

18/18 motion fits are sealed. Noise is available from 18 windows.

## Reviewed findings (2026-09-13)

All 18 extraction/fit stages completed; the independent manager returned exit 0 at 18:50:35 UTC (11:50 PDT), after about 3 h 19 min. Actual service state is active/exited with MainPID=0 and Result=success. All 18 compact field hashes match their sealed receipts.

Luke and Bacon separate strongly in the sampled motion estimates. On both Luke probes the early and late windows have median-depth P95−P5 excursions of 186–196 µm, versus 6.7–10.7 µm across all Bacon windows. Luke's middle window is quieter (13.8/31.1 µm on imec0/1). This supports a large, time-dependent Luke–Bacon difference; it does not establish session-wide prevalence.

**Do not interpret Allen's headline speed/depth-spread values as verified motion.** Direct inspection of shank1 p10 and shank2 p90 shows approximately stationary dense depth bands alongside highly variable fields at sparsely populated depths. Per-depth checks across all six Allen fits place the largest excursions (56–195 µm) at the top model depth, while shank1's two interior depths have excursions 6–14 µm. Shank2's approximately 316 µm model depth has 10–11 µm excursions; its approximately 640 µm depth is also variable (50–81 µm). These are model-support limitations requiring review, not evidence that Allen truly has Luke-like fast movement. Full per-depth values are retained in `per_depth_excursion_audit.json`; no depth was removed or refitted to improve the comparison.

Referenced 300–6000 Hz voltage variability is much closer across datasets than the Luke–Bacon motion difference: median shank-referenced sigma is Luke 8.8–9.6 µV, Bacon 10.0–11.4 µV, Allen 9.8–9.9 µV (probe-level medians across three windows). Unreferenced values are higher on Luke/Bacon than Allen. These summaries use the new matched conventions, not the historical high-pass-only noise measure.

The cheapest next check is to inspect cached Allen peak-depth bands and quantify local support around the unstable depths. That can distinguish supported trajectories from sparsely constrained model regions without another extraction. A trustworthy all-depth Allen motion ranking may still require a support-appropriate model or independent trajectory validation.

The question is whether these sessions differ in motion magnitude, speed, depth dependence and noise under a consistent fast estimator. Luke having the largest/fastest motion is a hypothesis, not an input to selection.

Each recording has three prespecified 120 s windows centered at 10%, 50% and 90% of its duration. The six probe/shank streams belong to three sessions; they are not six independent preparations. Relative session position does not match behavior.

| Dataset/probe | Completed windows | Median depth excursion (µm) | P99 local speed (µm/s) | P95 dynamic depth spread (µm) |
|---|---:|---:|---:|---:|
| Bacon probeA | 3 | 9.51 | 34.35 | 9.96 |
| Bacon probeB | 3 | 6.76 | 28.37 | 8.42 |
| Allen shank1 | 3 | 17.96 | 242.13 | 85.33 |
| Allen shank2 | 3 | 32.94 | 293.79 | 97.28 |
| Luke imec0 | 3 | 191.28 | 317.30 | 179.78 |
| Luke imec1 | 3 | 185.55 | 300.54 | 103.81 |

Entries are medians of the available window-level measurements. Inspect individual points, peak support and retained channels before using these summaries. No session-level significance or biological causality is inferred.

| Dataset/probe | No reference (µV) | Shank median (µV) | Local ≤100 µm, excluding self (µV) |
|---|---:|---:|---:|
| Bacon probeA | 40.02 | 11.36 | 10.07 |
| Bacon probeB | 35.24 | 10.00 | 9.01 |
| Allen shank1 | 17.27 | 9.86 | 9.45 |
| Allen shank2 | 17.26 | 9.78 | 9.44 |
| Luke imec0 | 40.47 | 9.59 | 8.17 |
| Luke imec1 | 32.57 | 8.83 | 7.40 |

Noise entries summarize per-channel MAD/0.67449 across twelve 1 s samples in each window, then channels and available windows. Spikes remain in this voltage-variability measure. Local neighborhoods contain different channel counts on NP and Nandy probes. Allen shanks are referenced separately.

## Reading the evidence

- MEDiCINe: 0.25 s bins, 1 s triangular kernel, four depth bins, bound 500 µm, 10000 Adam steps and seed 0; DARTsort native initial detection only, with per-window denoiser fitting.
- Frontend recorded for this run: DARTsort ibllikecmr with coherence gates retained and absolute HF-PSD bad-channel cutoff disabled identically for all datasets; native initial_detection only; full coverage; per-window denoiser.
- Speed is the finite difference of a smoothed model field at 250 ms spacing. Four samples per second is not 4 Hz physical-motion bandwidth.
- Excursion is P95−P5 through time. Dynamic depth spread is the across-depth P95−P5 after removing constant offsets independently at each depth. Opposing movement can be large even if the rigid median is small.
- The NP and Nandy depth spans differ. Their whole-shank differential spread does not measure deformation over equal physical distances.
- Greater estimated motion alone does not prove greater accuracy. Peak-depth rasters, per-depth field traces, localization support, channel exclusions and loss traces remain the direct checks. These fits have no new independent cell-identity validation.
- Windows do not measure full-session drift and can miss rare large events. All windows were selected before inspecting the new estimates.
- Raw voltage is not motion-corrected and no spike sorting is performed. Original full-session estimates remain separate historical evidence.

## Files

[Motion comparison](comparison.png) · [Noise comparison](noise_comparison.png) · [Motion rows](review_motion_windows.csv) · [Noise rows](review_noise_windows.csv) · [Settings](config.json)
