# Luke motion audit: conversation and pause state

User explicitly paused work on2026-09-08 and requested a summary. Newly launched service `luke-3sigma-lowpass-screen-v1.service` was stopped; actual final state inactive/dead,MainPID0. No low-pass comparison results are available. Its wrapper receipt remains stale at running because termination prevented finalization; `testing/outputs/luke_3sigma_lowpass_screen_job_v1/cancellation.json` records the user stop and verified state. Preserve partial outputs and do not automatically restart.

## Objective

Determine whether motion estimates and correction are reliable enough to justify another full-session sort. Initial concerns were that earlier comparisons might miss the known loss of amplitude completeness without motion correction. Short anchors are unsuitable for truncation/completeness assessment. The present work addresses motion-input/estimation validity, not final sorting quality or completeness.

## Main sequence and findings

1. Reviewed prior runs/shared work before launching additional tests. Built a top20catalog of large, repeatable, neural-looking lighthouse candidates and tracked nine cells across220–3100µm over4160–4260s. User approved their waveforms. They corroborate several slow movements and depth-dependent displacement. Fixed-template matching can select stationary-looking events when cells move; these centroids are provisional, not calibrated motion truth. Sparse bins and identity ambiguity remain visible.
2. Identified a broad shared disturbance with unequal channel responses. Global median referencing leaves channel-dependent residuals, including strong stationary peak families near2380–2520µm. Acquisition data already contain the mismatch; hardware origin is unproven. Raising detection thresholds alone did not solve this original artifact.
3. Learned a frozen31tap per-channel response to the prereference common signal. Subtraction on the motion-input voltage suppresses those residuals; source/sorting voltage remains preserved. Compensated peaks let DREDGE detect central movement where the original estimate was nearly flat. Selected neural waveform preservation and sampled temporal model transfer checks were favorable, but not full-session validation.
4. Found and corrected an installed DREDGE pairwise search-bound issue in the diagnostic helper: clipped windows could search outside requested±80µm. Subsequent matched comparisons explicitly enforce that bound. This was not the sole cause of failure.
5. Examined peak-depth/time histograms and included/excluded waveforms. More aggressive shared-event rejection removed previously rescued artifact-explained peaks. Broader neural-morphology screening was then tested. A fixed-family requirement was far too restrictive and set aside. An event-level8σ amplitude/spatial/shape screen retained24% of compensated peaks but worsened DREDGE, producing large unsupported central excursions.
6. Extended the comparison to100s with all nine lighthouses, including motion sampled at actual accepted spike times. Compensation is more encouraging than original input or the aggressive screen, but disagreements remain. Brief large shallow excursions around4245–4250s are not resolved by the lighthouse summaries.
7. Ran21screen variants and random-thinning controls. Amplitude and neighboring-waveform gates are the main contributors to the screen's deterioration. Random retention at the same count performs better; matching depth/time counts is intermediate. Screening promotes an alternative55–65µm pairwise alignment in the problematic central window: large-shift constraint weight rises from4.9% to38.5% while median correlation is nearly unchanged. Individual waveform families responsible have not yet been traced. No tested screen meaningfully beats compensation alone; center-energy-only exclusion removes15% with essentially unchanged agreement.
8. Freshly detected3σ,4σ,5σ,6σ on the same100s compensated voltage, keeping the noise vector/localization/DREDGE fixed.5σ baseline verified on common chunk interiors.3σ gives the best aggregate lighthouse agreement in this interval;4σ gives a smaller overall improvement;6σ worsens agreement and instability.

## Latest completed results

| Detection | Peaks | Overall lighthouse difference,µm | Drop/recovery difference,µm |
|---|---:|---:|---:|
|3σ|1299549|1.073|1.270|
|4σ|437358|1.214|1.444|
|5σ|216715|1.353|1.433|
|6σ|122885|1.666|1.480|

3σ improves these two scores by approximately21% and11% versus5σ, at about6times the peak count. These are descriptive differences from provisional centroids, not ground-truth errors. Improvement is not uniform; large shallow excursions remain. Added low-amplitude detections include ambiguous waveforms and are not all certified neural. No new full-session configuration is validated by these results.

## Paused next experiment

Starting from3σ:

- Center-energy screening only: require≥65% of detector-waveform energy within±0.5ms, relative to±1ms.
- Additional zero-phase third-order low-pass at3000Hz after existing broadband compensation.
- Both together.
- Narrowband detection with either the original absolute noise thresholds or channelwise noise-ratio adjustment, to distinguish filtering from a threshold-scale change.

Six total arms including baseline/controls were planned on the same100s, with narrowband events freshly localized on narrowband voltage. Script `testing/luke_3sigma_lowpass_screen.py`; output/settings under `testing/outputs/luke_3sigma_lowpass_screen_v1/`. The user stopped the job before results. A later restart requires respecting this pause and preserving the existing run evidence; the script has no automatic or within-stage resume.

## Key completed artifacts

- `docs/luke_detection_threshold_sweep_20260908.md`; `testing/outputs/luke_detection_threshold_sweep_v1/`: threshold summary, nine lighthouse overlays, continuous trajectories, histograms, added waveform examples and audits.
- `docs/luke_screen_sweep_20260908.md`; `testing/outputs/luke_screen_sweep_v1/`:21variant scores, controls, pairwise mechanism and trajectories.
- `docs/luke_long_lighthouse_motion_20260908.md`; `testing/outputs/luke_long_lighthouse_motion_v1/`:100s original/compensated/screened comparison.
- `docs/luke_motion_goal_evidence.md`: wider goal evidence ledger. The original full-run decision remains unresolved.

No new motion correction was applied to production voltage and no full-session sort was launched as part of these latest experiments. Full-run hold remains in force. User preference: concise dot points, inspect input waveforms/histograms, export figures at significant progress, prioritize mechanistic diagnosis and corroboration over broad estimator sweeps. Custom continuous lighthouse motion estimation remains deferred.
