# Motion evidence figure index for overnight continuation

The working baseline remains frozen shared-response compensation, broadband 300–6000 Hz, 3σ detection, and amplitude-sum DREDGE registration. Neither low-pass filtering, center-energy screening, nor global mean-amplitude aggregation earned promotion. Recording-wide validity and the shallow reference problem remain open.

## One-page checkpoint

[Six-panel contact sheet](../testing/outputs/luke_motion_overnight_figure_index_v1/01_prior_round_contact_sheet.png) · [PDF](../testing/outputs/luke_motion_overnight_figure_index_v1/01_prior_round_contact_sheet.pdf)

| Panel / question | Supported conclusion | Detailed figure |
|---|---|---|
| A — Does filtering or screening improve the current comparison? | Low-pass worsens provisional-reference agreement; screening's small aggregate gain is inconsistent across metrics. | [Six-arm lighthouse overlay](../testing/outputs/luke_3sigma_lowpass_screen_v2/02_lighthouse_overlay.png) |
| B — Is event removal sufficient to explain low-pass behavior? | At identical event membership, amplitude/localization changes nearly reproduce the filtered result. This is not an additive decomposition of physical error. | [Common-event factorial](../testing/outputs/luke_lowpass_common_events_v2/01_sweep_summary.png) |
| C — Are neural waveform amplitudes preserved uniformly? | Low-pass/broadband ratios vary strongly across cells; noise rescaling cannot preserve their original weighting. | [All-unit waveform contact sheet](../testing/outputs/luke_lowpass_waveform_preservation_v2/02_all_unit_waveform_contact_sheet.png) |
| D — Does the lighthouse statistic measure injected movement faithfully? | Median individual-event changes are compressed; median-waveform centroids are more sensitive but still asymmetric under the approximate interpolation model. | [Template sensitivity](../testing/outputs/luke_shallow_template_centroid_audit_v4/01_template_sensitivity.png) |
| E — Does global mean-amplitude aggregation fix the problem? | Shallow gains accompany deeper regressions; global mean is rejected, without fitting a depth hybrid. | [All-unit mean/sum overlay](../testing/outputs/luke_mean_raster_diagnostic_v3/02_all_unit_overlay.png) · [Actual input rasters](../testing/outputs/luke_mean_raster_diagnostic_v3/03_actual_rasters.png) |
| F — Is the short transition now fully validated? | Unit154 supports two five-second bins only. DREDGE and descriptive waveform changes differ; later sparse bins and the 220/410 µm region remain unvalidated. | [Reference report](luke_shallow_transition_audit_20260908.md) · [Measurement report](luke_shallow_fractional_calibration_20260908.md) |

The overview is rendered by `testing/luke_motion_overnight_figure_index_v1.py` from saved CSV metrics only. Panels A/B/E use provisional lighthouse disagreement rather than measured physical error. Panel C pools four audited waveform bins per unit. Panel D uses gain1 paired accepted injections within ±10 µm and shows no extrapolation beyond that support. Panel F samples the two DREDGE fields at the newly accepted unit154 frames, with the same first-bin offset; the median-waveform value is a different statistic from median per-event centroids. The PDF and PNG were visually inspected. No voltage reads or motion fitting were performed.

## Overnight direction: independent audit requirements

Prioritize reference qualification and a small frozen-baseline transfer check. Both answer current uncertainty more directly than another same-window global filtering or screening sweep.

- Treat 4160–4260 s and any previously inspected calibration windows as development evidence. Record which training, qualification, and transfer windows were previously viewed. New method comparisons on an old holdout are not independent confirmation of the full method-selection process.
- Freeze candidate definitions, acceptance gates, and the retained motion configuration before opening transfer results. Report failed qualification, identity ambiguity, support loss, and boundary hits without changing thresholds to rescue the observed trajectory.
- Early/late baseline transfer is descriptive robustness evidence unless an independent reference is qualified there. Candidate rediscovery is not proof of identity continuity with the development interval.
- Keep event selection independent of DREDGE. Report measurement sensitivity separately from biological uncertainty. Conditional bootstrap intervals do not cover identity mistakes or injection-model error.
- Commit resumable stages atomically with a receipt linking source/configuration/input hashes and every required output checksum. Refuse changed inputs or corrupt completed outputs; preserve interrupted partial evidence and restart only incomplete stages. Prove this behavior with a cheap kill/restart test before entrusting unattended work.
- Maintain independently managed jobs and actual manager-state checks. Reuse of completed stages is not within-stage checkpointing; keep the full-sort hold intact.

Both overnight implementations passed independent source review before launch. Transfer review required and verified an exclusive output lock, explicit empty-reference handling, version/source/input hashes, validated chunk reuse, and preserved epoch attempts. Reference review required and verified bounded raw-content fingerprints, independent-family injection controls, reused-development labels, boundary gaps, and source consistency checks before final report commit. Cheap tests exercised actual SIGKILL/restart behavior, preserved partial evidence, changed inputs/settings, and corrupt or missing completed outputs. Scientific outcome and liveness must still be read from the actual managed jobs.

The separate cached-data postprocessor `testing/luke_shallow_reference_overnight_report_v1.py` validates completed receipts and compares the frozen DREDGE field at the exact dominant-shift events used in each median waveform. Synthetic zero-pass and supported-plus-gap cases passed. It produces explicit failed-qualification/gap figures when no cell passes, and labels bootstrap intervals as conditional on a fixed reference offset and accepted identity. No ground-truth accuracy score is computed.


## Accessible Panel A rendering

[Panel A grouped bars](../testing/outputs/luke_lowpass_accessible_figures_v1/01_panel_a_colorblind.png) · [PDF](../testing/outputs/luke_lowpass_accessible_figures_v1/01_panel_a_colorblind.pdf)

[Six-arm overlay with color and line-style encoding](../testing/outputs/luke_lowpass_accessible_figures_v1/02_lighthouse_overlay_colorblind.png) · [PDF](../testing/outputs/luke_lowpass_accessible_figures_v1/02_lighthouse_overlay_colorblind.pdf)

These renderings retain the original numerical data and reference error bars. “Low-pass failed” overstates the evidence: aggregate disagreement increased, substantially driven by unit80, but the references' identity and displacement sensitivity are unresolved. This comparison alone does not establish that filtered motion is physically less accurate. Original images are preserved.


## Complete screening-sweep lighthouse overlays

Primary screening review: [21-page PDF](../testing/outputs/luke_screen_all_lighthouse_overlays_v1/all_21_screening_lighthouse_overlays.pdf) and [individual variant index](../testing/outputs/luke_screen_all_lighthouse_overlays_v1/README.md). Every sweep arm is shown against its compensated5σ baseline and all nine provisional lighthouse tracks, with identical per-unit limits across pages. Includes baseline, all screen variants and three thinning controls. Saved predictions were reused; no new motion estimation. The later3σ gentle-screen comparison remains separate. These displacement overlays should precede waveform-amplitude summaries when assessing motion behavior.


## Peak depth/time plots for every screening variant

[All21pages PDF](../testing/outputs/luke_screen_depth_time_atlas_v1/all_21_depth_time.pdf) · [Per-variant index paired with lighthouse overlays](../testing/outputs/luke_screen_depth_time_atlas_v1/README.md). Baseline/retained/removed counts and amplitude mass use common color limits,1s×10µm bins and cached masks/locations. See [screening shortlist](luke_screening_shortlist_20260908.md) for local gains and regressions; no arm is promoted from these development scores.
