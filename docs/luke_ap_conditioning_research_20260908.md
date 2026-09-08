# AP-only DREDGE conditioning: bounded research and cached reference inventory

This review supports the authorized conditioning comparison; it launches no experiments. Shared-response compensation remains the most promising local intervention. The next useful question is whether mild temporal filtering preserves spatially informative neural events while improving the AP raster. Neither literature nor the existing lighthouse evidence establishes that the current configuration works recording-wide.

## What primary sources support

- DREDGE accepts an AP representation constructed by detecting and localizing unsorted spikes, then binning their amplitudes in time and depth. The paper describes modular AP preprocessing, with an approximately 300–6000 Hz band and dataset-specific filtering/detection/localization. Its IBL example includes ADC-offset correction, bad-channel handling and spatial high-pass filtering. This supports controlled input experiments, not a universal best band. [DREDGE author manuscript, AP preprocessing section](https://pmc.ncbi.nlm.nih.gov/articles/PMC10634799/); [published paper](https://doi.org/10.1038/s41592-025-02614-5).
- SpikeInterface explicitly recommends filtering/denoising and avoiding whitening before motion estimation. Its motion pipeline separates detection, selection, localization and registration, so each step can be investigated separately. [Maintainer source: motion preprocessing](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/preprocessing/motion.py).
- SpikeInterface exposes multiple localization methods; its documentation identifies center of mass as fast but biased near probe borders. Localization is therefore an additional measurement stage, not simply a label attached to a detector peak. [Maintainer documentation: sorting components](https://spikeinterface.readthedocs.io/en/stable/modules/sortingcomponents.html).
- The maintained DREDGE implementation exposes raster smoothing and amplitude aggregation controls. These are potentially cheap experiments using saved peaks, but API names alone do not establish which array is transformed. [Maintainer source: DREDGE](https://github.com/SpikeInterface/spikeinterface/blob/main/src/spikeinterface/sortingcomponents/motion/dredge.py).

The published-paper full text was blocked by the web fetch during this review; the accessible author-manuscript AP methods and maintained source were used for detailed claims. No LFP experiment is proposed. No claim is made that DREDGE literature validates a 3000 Hz cutoff or a learned denoiser for this recording.

## Installed-code check matters for raster experiments

Read the installed `environments/rescue-production/.venv/lib/python3.12/site-packages/spikeinterface/sortingcomponents/motion/dredge.py`, `motion_utils.py` and `testing/luke_dredge_bounded.py` on 2026-09-08. The installed DREDGE builds an amplitude-weighted histogram, transposes it, and correlates that raster. `amp_scale_fn` is unused. `post_transform` is passed into weighting code; it does not transform the correlation raster at construction. The bounded replay reconstructs an amplitude-sum histogram with the configured Gaussian depth/time smoothing and explicitly enforces ±80 µm search.

A future log/square-root/count-raster comparison must transform the actual correlation input consistently with the bounded replay and persist that input. Setting `amp_scale_fn` or assuming `post_transform=np.log1p` already compresses the registration image would be an implementation error. Existing helper assumptions also hard-code 1 µm bins and 80 µm displacement bounds.

## Next hypotheses, ordered by cost and evidential value

1. **Complete the bounded 3σ temporal-filter/center-energy comparison.** Additional zero-phase low-pass filtering near 3000 Hz could reduce high-frequency disturbance while retaining useful spike structure. This is a hypothesis, not a literature-established optimum. Compare frozen absolute thresholds with filter-adjusted noise thresholds; otherwise a changed detection count may be mistaken for improved waveform quality. Localize fresh filtered detections on the intended filtered voltage and preserve branch-specific amplitudes/noise units. Inspect event-triggered multichannel waveforms and depth/time rasters, not only aggregate lighthouse disagreement.
2. **Check transfer before expanding tuning.** Use existing early/late cached candidate events for preservation and descriptive footprint checks. They cannot supply a definitive motion-error score. Repeat representative waveform preservation for the new filter rather than inheriting the compensation-only result.
3. **If needed, separate detection conditioning from localization conditioning.** Detect on noise-standardized or locally whitened voltage, but remeasure amplitudes and localize at those frames on fixed unwhitened compensated voltage. This is an engineering inference from the modular pipeline and whitening caution; it has not been validated here. Channel mixing changes spatial footprints, so do not apply a point-source spatial model directly to whitened amplitudes without calibration. Compare event membership and location changes separately. Begin with noise standardization before full spatial whitening.
4. **Only then test one targeted raster alternative from cached detections.** Compare amplitude sum with a compressed raster or count representation if dominant amplitude families remain visible. Persist the raw and transformed raster and pairwise lag spectra. Smoothing can remove noise and also erase short movement; higher correlation alone is not acceptance. Prior spatial-profile filtering and aggressive screening failures argue against an unrestricted raster/denoiser sweep.

Keep lighthouse event selection fixed across conditioning arms; evaluate motion at actual accepted event frames. Scores remain descriptive unless identity and displacement sensitivity are qualified. Nonzero motion is not itself proof, and quiet bins must not dominate a score that is meant to assess difficult transitions.

## Cached early/late evidence available now

| Cache | Contents and suitable reuse | Limitation |
|---|---|---|
| `testing/outputs/luke_candidate_footprint_audit_v1/` | `full_probe_waveforms.npz`, summary and full-probe sheets for 12 direct-detection cohorts at 960–970 and 9510–9520 s | Whole-probe squared-energy compactness was biased by finite-count background. Cross-half coherent local energy supplies a better diagnostic, not a calibrated acceptance probability. |
| `testing/outputs/luke_transfer_template_holdout_v1/` | Fresh `s970_peaks.npy` and `s9520_peaks.npy`, matched event frames, `tracks.csv`, `waveforms_events.npz` | Four frozen candidates yielded 1,036 matches and 13/16 five-second bins with ≥10 events; specificity remains limited by sparse competitors and fixed support. |
| `testing/outputs/luke_translated_holdout_review_v1/` | 2,341 matched events, offset counts, settings and figures | Expanded spatial search increased matches but exposed mixed offset populations; these are not motion ground truth. |

Fixed-template holdout counts were early 2920 µm: 180; early 3380 µm: 377; late 2900 µm: 381; late 3300 µm: 98. Low-count bins remain gaps. These observations are separate temporal holdouts from their training templates, but are already inspected development evidence for future work.

The translated matcher recovered all 20 noiseless geometric injections, yet only 16 passed competitor margins. Six of nine shape-specificity stress cases fit the wrong target; some stress shifts exceeded the local search domain, while late 3300-versus3340 µm confusion occurred within it. Real events often occupied multiple offsets in one bin. Never turn their mean/modal shift into physical displacement or tune DREDGE to match it. See [translated matcher audit](luke_translated_matcher_validation_20260907.md) and [footprint audit](luke_candidate_footprint_audit_20260907.md).

This review preserves the full-sort hold and does not restart the previously cancelled run. Any new run requires its own preserved settings, independent managed-job launch and exit evidence.
