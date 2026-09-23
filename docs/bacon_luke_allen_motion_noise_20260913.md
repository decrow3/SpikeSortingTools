# Bacon, Luke and Allen motion/noise comparison

Status: v2 completed all 18 fits successfully on 2026-09-13 at 18:50:35 UTC (11:50 PDT). Verified service Result=success, ExecMainStatus=0, MainPID=0, all seals and compact field hashes. Luke–Bacon separation is substantial; Allen edge-depth estimates remain poorly supported.

The primary question is how Bacon (halo), Luke (headpost), and Allen (Yates) differ in motion magnitude, speed, depth dependence, and voltage noise under a consistent fast estimator. The user hypothesizes that Luke has the largest/fastest motion and that the original estimators obscured the difference. Quantifying missed motion relative to old estimates is secondary. Older estimates are historical controls, not ground truth.

| Dataset/probe | AP variability after common-median reference (µV) | Rigid P95−P5 (µm) | Median across-depth P95−P5 (µm) |
|---|---:|---:|---:|
| Luke 2025-08-04 / imec0 | 11.68 | 5.50 | 5.15 |
| Luke 2025-08-04 / imec1 | 11.13 | 13.50 | 7.45 |
| Bacon halo 2025-10-16 / probeA | 29.89 | 7.50 | 2.10 |
| Bacon halo 2025-10-16 / probeB | 21.40 | 3.00 | 1.90 |
| Yates Allen 2022-02-16 / Nandy64 | 11.64 | 2.00 / 1.50 | 0.00 / 0.85 |

Allen motion entries are shank 1 / shank 2; its noise entry combines the 64-channel probe. Motion is full-session Kilosort-style registration from differing pipelines/depth grids, not a matched estimator rerun. Noise uses 12 sampled one-second windows, third-order 300 Hz high-pass, common-median subtraction and robust sigma. It includes spikes and is not the later 300–6000 Hz quiet-window metric.

## Configuration audit

The referenced recent field uses medicine-neuro 1.5, 0.25 s bins (4 Hz), 1 s triangular kernel, four depth bins, bound 500 µm, network (256,256), Adam 0.0005, batch 4096, 10000 training steps, seed 0. Its input is a DARTsort denoised-PTP population, distinct from the screened AP workflow in this repo. Do not silently substitute one population for the other.

The user confirmed that 4 Hz bins and a 1 s kernel are the intended Fast MEDiCINe configuration. Original Yates MEDiCINe uses 2 s bins / 30 s kernel and cannot serve as the new fast comparison.

Original Luke DREDge time_bins.npy has median spacing 1 s. The historical 20 s CSV is a summary export. The quoted Luke/Bacon 9.50/5.25 µm excursion and 6.30/2.00 µm spread numbers are Kilosort-style summaries.

## Bounded rerun design

Start with cached populations where provenance permits and duration-matched short windows across both Luke probes, both Bacon probes, and each Allen shank. This is cheaper than full sessions and can test whether fast estimates change the ordering; session-wide magnitude still needs broader time coverage. Fix windows before inspecting new fits, and keep targeted motion-event windows separate from representative windows.

Use the exact intended estimator and a consistent population recipe, with per-recording geometry, gains and sample rates. Keep raw/unreferenced, within-shank median and local-reference noise separate. Reuse historical noise only for its original definition/windows. Compare old/new motion on shared temporal support, report native rapid excursions as well as matched-lag steps, and retain depth coverage/peak support. Spatial span and model-grid differences must remain visible. Greater estimated displacement alone does not establish accuracy or a head-fixation cause.

No sorting or voltage resampling is needed. Persist resolved settings, input provenance, fields, diagnostics and final status. Long estimation jobs should use the independent launcher; interrupted optimizer fits restart because there is no within-fit checkpoint.

Source tables: /home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/outputs/ephys_quality_pilot/luke_20250804
Machine-readable audit: /home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/outputs/bacon_luke_allen_motion_noise_20260913/baseline_audit.json

## Authorized pilot and reproducibility

- Windows: 120 seconds centered at 10%, 50%, and 90% of each session, fixed before inspecting new fits. Both Luke probes, both Bacon probes, and Allen shanks 1/2 give 18 fits. Relative position is not behavioral matching. Short windows cannot establish full-session drift or capture every transient.
- Frontend: the same DARTsort `ibllikecmr` preprocessing and native initial detection/localization recipe used by the recent local-raw Luke run, with complete temporal detection coverage and a fresh per-window denoiser. Runtime uses one GPU worker. No clustering, matching, spike sorting, or motion-corrected voltage is produced. Allen is processed separately by shank.
- Native reader metadata supplies geometry, sample rate, gains and acquisition phase shifts; Luke reads the local Expansion copy, Bacon the Open Ephys source, and Allen the collated raw binary. All six raw read/scale checks passed. Allen's binary reader explicitly sets the zero voltage offset required for scaling. Actual channel geometry is saved, not inferred from historical prose.
- MEDiCINe uses the confirmed 4 Hz/1 s configuration, 10000 steps, seed 0. All valid denoised PTP peaks are used without an additional amplitude cutoff. Fields retain native coordinates and session-relative time separately.
- Motion endpoints: P95−P5 displacement at each model depth, P99 local/rigid 250 ms finite-difference speed, P99 1 s local displacement increments, and dynamic differential spread after removing each depth's constant temporal median. The derivative is estimator-resolved speed, not unrestricted physical velocity. Save full displacement ranges and peak support as well.
- Noise: 12 uniformly spaced one-second samples in each window, third-order 300–6000 Hz bandpass, robust sigma with no reference, shank-wide median, and local median ≤100 µm excluding the measured channel. All references use the same source samples. Neighborhood sizes differ by probe geometry and are saved. These trace statistics include spikes.
- Evidence: `testing/outputs/cross_dataset_fast_motion_v1/config.json` records all raw paths, sizes/mtimes, geometry, gains, sample-exact windows, settings and source hashes. Per-stage input populations, denoisers, HDF5 detection evidence, fields, losses, noise tables, logs and final exits are retained. Disposable preprocessed binaries are removed only after successful extraction.
- Checks: three tests cover known rigid speed, opposing depth movements invisible in a rigid average, and interrupted/corrupt stage refusal. Six real raw-read checks passed. The dummy service was observed active/running after its launcher returned, then completed with exit 0; user lingering is enabled.
- Manager: `cross-motion-pilot-20260913-v1.service`; launch/receipts are in `testing/outputs/cross_motion_pilot_20260913_v1_job`. Source scripts are frozen there. The manager persists after chat closure. Completed hash-sealed stages can be reused explicitly, but interrupted extraction/fits refuse overwrite and have no advertised within-stage resume.
- Completion: require all 18 sealed fits, `summary.json`, a successful manager receipt, and actual service exit state. Plot/CSV snapshots may be partial while the service runs.

## v1 failure and v2 restart

V1 completed Luke imec0/imec1 p10 windows, then exited 1 at 2026-09-13 10:32:31 UTC on Bacon probe A. The automatic IBL coherence+PSD detector rejected all 384 channels; saving a zero-channel recording raised `OSError: Invalid argument`. No OOM occurred. Original logs, source snapshot, populations and two completed fields remain unchanged.

A six-chunk direct audit on both Bacon probes found 383 noisy / one dead channel before reference. Simply moving the classifier after median reference still rejected most channels and is not the fix. Disabling only the absolute HF-PSD threshold while preserving the coherence-based dead/noisy/outside checks retained 383 good / one dead (AP191) on each Luke/Bacon probe and 32 good channels on each Allen shank. Audit JSON files are in the v1 output root. The detector's absolute noise threshold excluded the recording-level noise property that this study is trying to compare.

V2 keeps MEDiCINe, windows, detection/localization, filtering, phase correction, common-median references, standardization and noise measurements unchanged. All six streams now use the same coherence-only channel gates (100 chunks, seed 0, absolute HF-PSD cutoff disabled); complete per-channel labels are saved. A zero-channel result now fails explicitly before materialization. This is a documented frontend deviation from unmodified `ibllikecmr`, not a change to Fast MEDiCINe.

The new output is `testing/outputs/cross_dataset_fast_motion_v2`, with independent service `cross-motion-pilot-20260913-v2.service` and frozen launch evidence at `testing/outputs/cross_motion_pilot_20260913_v2_job`. Existing dummy/disconnection proof applies to the unchanged launcher. The new order processes Bacon, Allen, then Luke at each session fraction. All 18 fits will be rerun under the common channel rule; v1 Luke fits are not silently counted as v2 results.

V1 preliminary Luke p10 estimates: median across-depth P95−P5 excursion 190.38 µm (imec0), 181.64 µm (imec1); P99 local model speed 372.65 and 371.98 µm/s. The imec1 peak raster shows repeated large shifts. These are one-window model estimates, not a cross-dataset ranking or independent cell-identity validation.

## Completed v2 review

See `testing/outputs/cross_dataset_fast_motion_v2/REVIEW.md` for the reviewed comparison, Allen support caveat, motion/noise tables and figure links. The v2 batch is complete; no replacement run is active.

## Bacon historical-method cross-check (2026-09-14)

Saved KS4 fields were compared with MEDiCINe on identical 2 s batches and nine shared depth centers. Both give small pilot-window motion, with MED higher (rigid medians A/B 2.33/1.85 µm versus KS 1.0/1.0 µm). Full-session KS has occasional larger steps outside the pilot windows (32.5 µm near 102 s on A; 37.5 µm near 6438 s on B). See `testing/outputs/bacon_saved_motion_crosscheck_20260914/README.md` for definitions, all metrics, caveats and provenance. No new fits were launched.
