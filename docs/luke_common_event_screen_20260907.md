# Probe-wide shared-event screen: first bounded result

An event-level screen, with no channel blacklist or motion-agreement criterion, excludes 3,362 of 60,719 fresh 5σ peaks (5.54%) in 4180–4200 s. All 1,940 existing peak coincidences with the nine independently tracked lighthouse trains remain available. Central summed peak amplitude falls 28.92%, but the registration-input diagnostic still prefers zero shift. This is a useful conservative exclusion pilot, not a complete motion fix or production-ready neuronal classifier.

## Rule and model

Use the previously reviewed pre-reference voltage (accepted cached conditioning, then 300–6000 Hz filtering). The per-sample median across all 384 channels supplies a shared waveform reference. Fit each channel's response to that reference with a 31-tap FIR spanning ±0.5 ms, using every third sample of 4180–4190 s, ridge penalty 0.001 times mean covariance diagonal. Freeze coefficients before the 4190–4200 s evaluation half. All channels receive the same model and rule; no selected source-channel IDs, lighthouse events or displacement fields enter fitting or rejection.

Reject a detected peak only if all four conditions hold:

1. At least 80% of pre-reference channel waveforms have cosine >0.8 with the simultaneous global-reference waveform.
2. The predicted shared contribution explains at least 80% of the referenced detection-channel waveform energy.
3. Observed and predicted referenced waveform cosine is >=0.95.
4. After accounting for that contribution, there is no local extremum >=4 residual-noise sigma within ±40 µm and ±0.3 ms.

Residual noise is per-channel MAD estimated from the first-half model remainder. Unmodeled or ambiguous signals remain eligible. The fourth condition protects potentially distinct local spikes, including coincident ones, but does not prove that every retained remainder is neuronal. The model is only used for classification in this pilot: stored production voltage and existing peak localizations are unchanged. Retained peak amplitudes/localizations remain the original fresh5σ values.

## Results and limits

- First-half screen: 1,726 rejected; second-half screen: 1,636 rejected.
- Nine independent lighthouse tracks supply 2,501 events in this pair. The original negative-only 5σ peak set has 1,940 spatial/temporal coincidences; all 1,940 survive. First half 849/849; second half 1091/1091. This does not claim 100% recovery of all lighthouse spikes or establish peak identity at each coincidence.
- Rejections concentrate on channels 252 (1507), 238 (1060), 251 (582), and 253 (213), despite no channel-specific exclusion. That concentration follows the waveform explanation rule, rather than being prescribed.
- In 2000–2900 µm, the screen removes 3,362 of 20,443 peaks and 28.92% of summed absolute peak amplitude. Amplitude mass is descriptive, not the exact weighting of a full estimator.
- At 1960–2560 µm, zero-shift profile correlation decreases from 0.9942 to 0.9184; at 2260–2860 µm, from 0.9962 to 0.9515. Zero remains the maximum over ±20 µm. This is the same summed-amplitude, 1 µm smoothed pairwise diagnostic, not a full DREDGE/decentralized rerun.
- Reviewed examples show a shared disturbance explaining most of the spike-shaped waveform. An event with a residual local SNR of 4.2 is retained despite matching the shared component strongly. This conservative protection contributes to retaining ambiguous detections; it should not be relaxed merely to obtain the expected motion.

The entire interval has already been used for diagnostic development. The response fit is first-half-only, but this is not untouched prospective validation. A learned common component can include biological population signals, and a linear response model may leave artifact residuals. We have not measured a ground-truth false-rejection rate or assessed full-session stability. The next discriminating check is the nature of the retained local remainders: distinguish repeatable neural waveforms from model error using their multichannel shapes. Do not simply discard all peaks occurring during the common disturbance.

## Artifacts and execution

Output directory testing/outputs/luke_common_event_screen_v1 contains settings, FIR coefficients and residual noise, per-peak decision metrics, keep mask, retained/rejected peak arrays, unchanged retained localizations, lighthouse-preservation counts, alignment-objective scores, and two PNG/PDF figure pairs. Peak indices/times are relative to the existing 4180–4200 s recording slice, as in the fresh detection cache. The examples in 02_event_examples are first qualifying category/depth examples and occur in the training half, not prospective examples or prevalence samples.

Managed job luke-common-event-screen-v1 completed with MainPID=0, ExecMainStatus=0, active/exited. Launch command/log/receipt persisted. Both figures inspected. No production preprocessing, sort, correction or motion estimator rerun; full-sort hold preserved.
