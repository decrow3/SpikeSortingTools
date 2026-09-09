# AP motion estimation: peak conditioning and DREDGE/MEDiCINe changes

This note documents the Luke0804_V2V1 imec0 diagnostic on recording-relative 930–1030 s. The current comparison uses 0.25 s temporal bins in both methods, DREDGE histogram Gaussian sigma 0.5 s, and MEDiCINe motion-kernel width 1 s. These are diagnostic configurations, not recording-wide validated production settings. No sorting preprocessing or production motion correction was changed.

## What changed in the peaks

The observations are detector peaks and their monopolar localizations, not Kilosort unit positions or lighthouse-selected spikes. All recent estimator comparisons reuse the exact same cached arrays in `testing/outputs/luke_long_context_validation_v1/`. They reproduce the counts in the earlier 0.25 s × 10 µm raster. That visualization binning is independent of each estimator's input representation.

1. Read AP voltage from the rescue recording, convert counts to µV, and apply a third-order Butterworth 300–6000 Hz forward/backward bandpass with 50 ms padding around 20 s processing chunks.
2. Preserve an original-input control: subtract the instantaneous global channel median from the bandpassed voltage.
3. Construct compensated input instead by subtracting the predicted channel-specific response to the pre-reference global median. The frozen 31-tap response model was fitted at 4180–4190 s and reused here without refitting. In notation, original channel voltage is `x_c(t) - m(t)`; compensated voltage is `x_c(t) - sum_l h_c(l) m(t+l)`. Compensation is applied before detection/localization; it is not a filter on already localized peaks. The model is `testing/outputs/luke_common_event_screen_v1/shared_response_model.npz`.
4. Detect negative locally exclusive peaks, radius 50 µm, at 5 sigma using the same frozen original-input channelwise noise vector in both arms. Absolute thresholds therefore do not decrease after compensation. Exclude events within 50 samples of each chunk boundary. Localize separately in each arm with monopolar triangulation, radius 75 µm. The cached 970–990 s chunk is reused exactly.
5. Concatenate the five chunks: 211,879 original-input peaks; 188,514 compensated-input peaks. Both estimators receive the same arrays within each arm, with event times, absolute depth and amplitudes retained.

No additional 3 kHz low-pass, whitening, center-energy screen, or other waveform screening is used in this head-to-head. The earlier 3-sigma experiments at 4160–4260 s are a separate branch; this early-interval population remains 5 sigma. MEDiCINe receives absolute peak amplitudes and `amplitude_threshold_quantile=0`, retaining all supplied peaks; the old wrapper requested a 0.2 cutoff. DREDGE uses amplitude-weighted histogram sums, not count-only or mean-amplitude histograms.

Compensation changes event membership, amplitudes and localization together. The original/compensated comparison demonstrates sensitivity to that combined intervention, not a decomposition of its mechanism. Prior selected-waveform preservation checks and transfer checks supported investigating compensation, but do not establish preservation for every cell or every session interval. See [compensation evidence](luke_shared_response_compensation_20260907.md).

## Current estimator settings

| Setting | DREDGE v8 | MEDiCINe v6, reused unchanged |
|---|---|---|
| Temporal bins | 0.25 s; 400 times | 0.25 s; 400 times |
| Temporal smoothing | Gaussian histogram sigma 0.5 s | Triangular motion kernel width 1 s |
| Spatial model | 20 overlapping nonrigid Gaussian windows; step 200 µm, scale 300 µm, margin 50 µm | Four depth bins; interpolated displacement across depth |
| Depth histogram | 1 µm bins, Gaussian sigma 1 µm; amplitude-weighted sum | Learned joint activity distribution of amplitude/depth |
| Motion bound | Strict pairwise shifts within ±250 µm | `motion_bound=500`: internal pre-centering ±250 µm range |
| Other settings | Pairwise time horizon 60 s; min correlation 0.1 | 10,000 training steps; batch 4096; seed 0; network (256,256); Adam learning rate 0.0005; noise 0.1 annealed over 2,000 steps |
| Compute | CPU, four numerical threads | CUDA, four CPU threads |

The temporal kernels are not equivalent: DREDGE smooths the input histogram, while MEDiCINe smooths its motion parameters. A Gaussian sigma is not total kernel support. The spatial bases are also different; 20 overlapping windows are not directly equivalent to four MEDiCINe bins. DREDGE's bound constrains pairwise shifts, not the final field's absolute magnitude or peak-to-peak span. MEDiCINe centers its bounded internal field. No plotted output was clipped to force a shared absolute limit.

DREDGE's historical diagnostic helper was hard-coded to ±80 µm. The separately versioned `testing/luke_dredge_bounded_250_v1.py` explicitly evaluates and selects only ±250 µm pairwise lags before solving, and asserts the saved pairwise bound. It preserves the uncorrected library result as diagnostic evidence. Original files and earlier fields remain intact.

## Sequence of the recent comparisons

| Output version | MEDiCINe | DREDGE |
|---|---|---|
| v1 | 1 s bins, 50 s kernel, 2 depth bins, bound 800 | Reused 1 s bins / 1 s histogram sigma / ±80 µm field |
| v2 | 1 s bins, 1 s kernel, 2 depth bins | Fresh replay of the same DREDGE settings |
| v3 | 1 s bins, 5 s kernel, 2 depth bins | Reused v2 |
| v4 | 1 s bins, 1 s kernel, 4 depth bins | Reused v2 |
| v5 | 0.25 s bins, 1 s kernel, 4 depth bins | Reused v2 |
| v6 | Same as v5, bound reduced from 800 to 500 | 0.25 s bins, 1 s histogram sigma, strict ±250 µm |
| v7 | Reused v6 | Only histogram sigma changed to 0.25 s |
| v8 | Reused v6 | Only histogram sigma changed to 0.5 s |

At 1 s bins, MEDiCINe's 1 s kernel discretizes to a single tap; at 0.25 s bins, that same nominal kernel spans neighboring bins. The 5 s kernel attenuated the short drops. More depth/time bins permitted richer trajectories. Expanding DREDGE's bound while refining time bins in v6 produced much larger central movement, with more visible agreement with MEDiCINe on compensated peaks. That combined DREDGE change did not isolate temporal resolution from bound effects. Reducing DREDGE sigma to 0.25 s in v7 sharpened drops but also introduced brief positive excursions, especially deeper; v8 tests the intermediate sigma without changing other settings.

## What differs from the earlier full-recording work

The saved legacy full-session source is `/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0/motion/`.

- Saved DREDGE: 10,474 temporal samples at 1 s spacing, 40 depth windows at 100 µm spacing. Pipeline code specifies 150 µm window scale and inherits 1 s histogram smoothing; these are code/default provenance, not a saved resolved configuration. Its exact search-bound override is not preserved.
- Saved MEDiCINe parameters: 1 s bins, 50 s kernel, two depth bins, bound 800, 10,000 training steps. The old wrapper also appended five terminal constant rows. The saved parameter file omits amplitude-cutoff provenance, although the wrapper requests 0.2.
- Exact legacy peak preprocessing/detection overrides were not saved in the inventory. The current `pipelineold/preprocess.py` has separate 300–3000 Hz motion filtering and local median reference; it must not be retroactively treated as proof of what the historical run executed. The newly fitted shared-response model was not part of the legacy peak inputs.
- Legacy saved motion timestamps have a different origin (~3057.677 s); current figures use recording-relative seconds. Direct historical overlays require aligning origins.

The later `motion_scale_sweep` imec0 manifests refer to 120 s at 8160–8280 s, not full-session reruns. Their `split='full'` means all peaks within that snippet. They tested rigid DREDGE 0.25 s bins with 0.25 s histogram smoothing and requested max displacement 80 µm; nonrigid spatial sweeps mostly retained 1 s bins. MEDiCINe tested two depth bins/50 s kernel and eight depth bins/20 s kernel, both at 1 s temporal bins and requested amplitude cutoff 0.2. Those trials did not test the present compensated, nonrigid, subsecond, wide-bound configuration across the recording.

## Evidence, reproducibility and scope

Output folders are `testing/outputs/luke_early_medicine_peak_comparison_v1` through `v8`. For v8, `01_peak_motion_overlay` shows thick cyan DREDGE and white MEDiCINe on the legacy raster; zooms retain historical limits and can clip large excursions. `02_full_displacement_comparison` shows full-range blue current DREDGE, gray v6 DREDGE, orange unchanged MEDiCINe. Each trace is referenced to its own 970–975 s median; anchor depths illustrate the motion field, not unit identities.

Independent systemd services persist commands, logs and per-stage exit receipts in sibling `_job` folders. Completed fields are reusable; interrupted fitting requires restarting that arm, not resuming an internal optimizer checkpoint. No full-session fit or sort is launched here, and the full-sort hold remains. Agreement between estimators using the same peaks is corroboration, not independent biological ground truth.

## Result of the 0.5 s DREDGE check

Both v8 DREDGE fits completed successfully in about 82 s each. Fields are finite 400-by-20 arrays and the saved pairwise shifts satisfy ±250 µm. Compared with sigma 0.25 s, sigma 0.5 s retains the major compensated drops while reducing many brief positive excursions. Several remain near 3381 µm (around 947, 1010 and 1020 s). This is an intermediate smoothing result, not a selected winner.

Compensated peak-to-peak spans at 2250 / 2919 / 3381 µm are 239.4 / 254.4 / 310.1 µm for DREDGE sigma 0.5 s, versus 277.5 / 304.3 / 359.4 µm for sigma 0.25 s and 216.3 / 198.4 / 193.0 µm for sigma 1 s. MEDiCINe remains 205.7 / 216.5 / 231.3 µm. These spans include positive excursions and cannot rank accuracy by themselves.

[Current peak overlay](../testing/outputs/luke_early_medicine_peak_comparison_v8/01_peak_motion_overlay.pdf) · [Current displacement comparison](../testing/outputs/luke_early_medicine_peak_comparison_v8/02_full_displacement_comparison.pdf).
