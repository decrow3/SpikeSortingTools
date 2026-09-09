# Screened MEDiCINe: Luke snippet workflow

Use **5σ detection with relaxed waveform screening** as the primary candidate configuration for larger Luke snippets. Retain **6σ/full screening** as a sensitivity comparison. This is a documented experimental shortlist, not recording-wide validation or authorization to alter sorting preprocessing.

## Evidence for the shortlist

The completed 930–1030 s comparison used 40 combinations and 17 frozen waveform-only lighthouse candidates. On strict observations after940 s, median across candidate median absolute differences was11.4 µm overall /24.4 µm for excursions≥120 µm for5σ/relaxed MEDiCINe;6σ/full gave10.0 /27.3 µm. Best DREDGE8σ/full gave12.3 /58.3 µm. These summaries include14 candidates overall and6 with enough large-excursion evidence, not17 independent verified neurons. Every shortlisted method missed unit161's large excursion by approximately450 µm. Offsets remain unchanged; do not select an offset to fit later traces.

Sources: `testing/outputs/luke_ap_methods_completed40_v1/{descriptive_agreement,per_candidate_differences}.csv`. Unit identities, strict/lower-score/ambiguous evidence, and seed references remain frozen. Lighthouse observations currently cover only930–1030 s. Later peak-raster agreement is useful inspection, not independent cell validation.

## Frozen configuration

- Same300–6000 Hz third-order forward/backward filtering,50 ms padding, frozen31-tap shared-response compensation model, and frozen original noise vector as the completed sweep.
- Negative locally-exclusive peaks,50 µm radius,5σ primary or6σ comparison; exclude first/last50 samples per20 s chunk.
- Primary relaxed screen: SNR≥6, central energy fraction≥0.5, qualified-neighbor cosine≥0.6. Comparison full screen: SNR≥8, center≥0.65, neighbor≥0.8.
- Both also require waveform width0.067–0.8 ms, broad-channel fraction<0.2, and reject shared-response explained fraction≥0.5 combined with cosine≥0.8. Screening uses the frozen residual-noise vector. Exact definitions remain in `testing/luke_screen_sweep.py`.
- Monopolar peak localization radius75 µm. Four workers. Absolute localized depths and absolute detector amplitudes enter MEDiCINe; no additional amplitude-quantile cutoff.
- Documented MEDiCINe v6:0.25 s time bins,1 s triangular kernel,four depth bins,motion_bound500,network(256,256),Adam lr0.0005,batch4096,seed0,initial motion noise0.1 annealed over2000 steps. CUDA,float32, four CPU threads. Default10000 steps.

The internal bound is pre-centering±250 µm and is not DREDGE's pairwise bound. Extending duration does not expand that bound. Temporal support follows actual first/last peaks; no terminal padding is appended.

## Scaling without changing the estimator

The runner computes only the two shortlisted populations. It reads one20 s voltage chunk at a time, detects both thresholds separately, computes features once for their union in bounded2048-event batches, screens first, and localizes the union of retained peaks once. Localization is per-event; removing discarded events must not change retained localizations. Tests verify batching preserves feature values/masks and localization is invariant to removing unrelated events. Full voltage is never retained across chunks. Peak arrays still grow with duration; the fit is one continuous field over the requested snippet, not stitched independently centered windows.

Completed original930–1030 s chunk evidence is reused after hash verification. New chunks preserve detected union peaks, feature values, retained indices, final peak/location arrays and resource timings. Report localized versus detected counts to quantify work avoided. Do not claim a measured speedup from the smaller amount of work alone.

The100 s primary input is fitted through the new wrapper before extending the snippet. Time/depth grids must match exactly and maximum field discrepancy must be<0.1 µm; otherwise the run stops for investigation. This tolerance checks reproduction, not biological accuracy.

For the first930–1230 s benchmark, compare10000 versus30000 steps on5σ/relaxed, retaining6σ/full at10000. Tripling duration increases the number of time parameters and reduces samples per time interval at fixed training budget. The extra fit tests budget sensitivity; longer training is not assumed to improve correctness. Export loss, peak counts, runtime, maximum RSS and peak CUDA allocation, plus full-span peak histograms/fields and the unchanged lighthouse comparison on the original100 s only.

## Run and inspect

Use the verified independent systemd launcher:

```bash
environments/rescue-production/.venv/bin/python -m testing.launch_luke_motion_overnight \
  --unit luke-screened-medicine-300s-v1 \
  --module testing.luke_screened_medicine -- \
  --start 930 --stop 1230 \
  --output /home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/outputs/luke_screened_medicine_300s_v1
```

The benchmark currently requires start930 s and duration divisible by20 s to retain the existing seed reference and chunk conventions. `testing/luke_screened_medicine_fit.py` is the reusable fit interface for saved inputs: structured `peaks.npy` with snippet-relative sample indices, matching `locations.npy`, and `input.json` defining sampling frequency/start/stop. It does not assume a100 s duration.

Use a fresh job unit for any relaunch. Inspect `systemctl --user show luke-screened-medicine-300s-v1.service`, the job receipt and actual processes; systemd's inactive state alone does not prove success. The launcher mechanism was verified previously with a dummy job surviving disconnection. Persisted job logs and receipts are under the sibling `_job` folder. A final summary is written only after all fits, source checks and reports succeed.

Completed sealed chunks/fits are hash-validated on explicit relaunch. An unsealed stage refuses reuse; preserve its evidence and investigate before deciding whether to restart it. There is **no optimizer checkpoint/resume inside a fit**. Configuration changes require a new output directory. No full-session sort or motion correction is launched by this workflow.

## First benchmark launch and validation

The930–1230 s benchmark was launched as `luke-screened-medicine-300s-v1.service` on2026-09-09 UTC (September8 Pacific). Both optimization unit tests passed. The100 s fit reproduction then passed with **zero maximum and RMS difference** from the saved5σ/relaxed field. The first five cached chunks were sealed in the new run and extraction of the additional200 s began. This records launch-time progress, not completion; use the actual service state and final `summary.json` for status. The fit source/configuration hashes remain fixed during the run.
