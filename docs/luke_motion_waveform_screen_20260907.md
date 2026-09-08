# Motion-only waveform input screen

A bounded event-level inclusion screen is implemented and audited for 4180–4200 s. It retains 13,424 of 55,912 compensated peaks (24.0%). No motion estimator, sort, or source-voltage modification was performed.

## Current screen (v2)

Input: existing fresh negative locally-exclusive 5σ detections and their compensated-voltage locations. The screen reads matching compensated multichannel waveforms. Noise normalization uses the frozen shared-model residual MAD. The 8σ gate is maximum absolute amplitude in the detector waveform, not a rerun of negative 8σ detection.

Rules:

- Exclude detector waveforms substantially explained by the frozen shared-response model (original referenced explained energy ≥0.5 and predicted cosine ≥0.8).
- Require maximum absolute detector waveform ≥8 residual-noise sigma and ≥65% of its energy within ±0.5 ms of detection.
- Require a neighboring channel within ±60 µm with amplitude ≥25% of detector maximum, ≥4 noise sigma, and waveform cosine ≥0.8 to the detector waveform.
- Require the dominant lobe's contiguous half-height width to be 0.067–0.8 ms.
- Exclude simultaneous residual excursions ≥4 noise sigma on ≥20% of the probe, measured within ±3 samples of the detector waveform's dominant extremum.

These are deliberately selective engineering criteria, not a validated neural/artifact classifier. No channel blacklist, lighthouse displacement, or DREDGE field enters selection. Repeatable waveform-family matches are recorded as supporting metadata but do not control v2 inclusion.

## Results

- Amplitude-only gate retains 24,118 events. The additional rules exclude 10,694 of those, leaving 13,424.
- First half retains 5,877; second half 7,547. Within each probe-quarter/half combination retention ranges from 17.4% to 34.1%; there is no whole-quarter collapse.
- Examples show exclusion of off-center large excursions, weak spatial support, incoherent neighboring waveforms and overly narrow/broad dominant lobes. Some excluded events are plausible neurons. The removal fraction must not be interpreted as artifact prevalence.
- Existing central lighthouse coincidences survive well for units317 (141/145),445 (134/134),463 (149/150), while losses are substantial for some other cells, especially510 (6/82 input coincidences). These are time/depth coincidences, not identity-confirmed recovery, and were not used to tune or veto the screen.
- Histograms still contain stationary structures. This pass does not establish complete removal of channel-locked artifacts or sufficient motion information throughout the recording.

## Rejected first design

The preserved v1 output required close matches to fixed per-channel multichannel families trained on the first 10 s. It retained only328 peaks from5families and left most depths empty. Its geometry/identity requirement was too restrictive for a motion input, so that mask is not adopted. V2 uses per-event spatial waveform evidence, allowing events without a frozen family match.

## Verification and provenance

Scripts: `testing/luke_motion_waveform_screen.py` (v1), `testing/luke_motion_waveform_screen_v2.py`, and `testing/luke_motion_waveform_screen_audit.py`.

Current outputs: `testing/outputs/luke_motion_waveform_screen_v2/`:

- `keep_mask.npy`, retained peaks/locations and excluded peaks.
- `peak_decisions.csv` with measured features and first-failed-rule labels; six boundary events are excluded without waveform classification.
- `settings.json`, `summary.json`, `audit.json`, `depth_time_retention.csv`, `lighthouse_coincidences.csv`.
- `01_histograms.png/.pdf`: input, retained, excluded.
- `03_amplitude_control.png/.pdf`: input, amplitude-only, full screen.
- `02_decisions_depth1..4.png/.pdf`: included/excluded individual multichannel waveform examples in each probe quarter.

Audit passes exact source-index/array matching, partition checks and independent reconstruction of the explicit v2 mask from saved event features. SHA256 hashes of input peaks, locations, frozen model and v2 script are saved in audit.json. Both independent systemd services exited0; actual service state was checked after launch and completion. Commands, logs and receipts reside in `testing/outputs/luke_motion_waveform_screen_job_v1/` and `..._job_v2/`. These short diagnostics have no within-stage checkpoint; interruption would require a new output version.

## Next input-review work

Review remaining stationary waveform families and transfer the same event criteria to a separate cached interval, inspecting depth/time coverage and waveform examples. Investigate residual channel locking against independently corroborated *local* biological movement. Do not infer that the current retained mask is clean simply because it is smaller. No estimator comparison or full sort is authorized by this diagnostic result.
