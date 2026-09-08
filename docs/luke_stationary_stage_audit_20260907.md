# Stationary disturbance traced through acquisition and conditioning

The dominant common event and mismatched channel responses are already present in acquisition AP voltage. Replaying recorded conditioning does not introduce the main residual. Global referencing leaves a large fixed-channel waveform because the affected channel response differs from the probe-wide signal.

## Results

Same 300 event times from the 4180–4200 s source251/252 cohort. Each stage is subjected to the same diagnostic 300–6000 Hz third-order forward/backward filter and global median reference; therefore the table compares what referencing would leave at each stage, not the production stage's native output.

| Input stage | Global-reference event peak µV | Ch252 residual peak µV | Ch238 residual peak µV |
|---|---:|---:|---:|
| Acquisition AP | 206.03 | 180.52 | 116.66 |
| Phase corrected | 207.55 | 184.50 | 128.44 |
| Saturation blanked | 207.46 | 184.50 | 128.44 |
| Interpolated | 207.55 | 184.50 | 128.45 |
| Cached reference | 207.54 | 184.50 | 128.44 |

All five stages have 381/384 event-averaged channel waveforms with cosine >0.9 against the global reference waveform. Phase correction modestly changes residual amplitude, especially ch238, but does not create the disturbance or most of the residual. Blanking leaves the suspect waveform essentially unchanged. Only 0.000264% of phase-corrected samples exceed the 500 µV blanking threshold in the padded interval; the fraction within selected event windows is 0.000417%.

The recorded interpolation graph modifies only AP191, not the implicated channels. Full reconstructed interpolation output and materialized cache differ in 0.2988% of entries, by at most one ADC count (2.34375 µV); 99th percentile absolute difference is zero. Their event waveforms overlap. This is close numerical agreement, not bit-identical replay; differing processing windows/rounding have not been separately assigned as the cause of the one-count differences.

Saved IMRO configurations are identical after channel index for all 384 channels: (0,0,500,250,1). This excludes a recorded channel-specific settings difference of those fields, not actual channel/electrode transfer-function differences.

## What this establishes

The plausible failure mechanism is now specific: a large acquired common disturbance encounters channel-dependent responses; median subtraction removes the typical response but leaves strong residuals on atypical channels. Their fixed positions can dominate a registration fingerprint. This matches the earlier stationary voltage and zero-shift profile evidence. Merely raising peak thresholds or treating every large residual as a neural spike is not adequate.

It does not establish the physical source of the common disturbance, whether the electrode/hardware responses are faulty, or how much of the full recording is affected. Acquisition AP voltage already includes acquisition-side filtering and reference behavior. This audit does not yet reproduce every historical motion-input conditioning branch or explain native KS large jumps.

## Next bounded comparison

For motion estimation input only, test removal or compensation of detections explained by these shared-event/channel-response residuals. Preserve useful neural spikes, including coincident spikes on affected channels. Compare disappearance of stationary registration peaks and recovery of the existing lighthouse movement before any full-session sort. Do not drop all synchronous events or assume a blanket channel deletion is harmless. The implicated channel group extends beyond the previously excluded 238/251/252 (notably 253 and other weaker residual channels).

## Provenance and execution

Resolved the saved recording/provenance.json, including fill_value=9 ADC counts, blank threshold 213.333 counts, phase margin 40 ms and stored interpolation weights. Acquisition folder and stream match the recording manifest; shape and sampling frequency asserted. Evaluation padding is 100 ms.

Initial managed run luke-stationary-stage-audit-v1 completed acquisition, phase and blanking, then failed with exit1 because the top-level interpolation provenance required an explicit base_folder. Failure log/receipt preserved. Corrected v2 supplies the recording directory, reuses the completed first-three-stage metrics from the v1 log and completes interpolation/cache without repeating those stages. V2 completed with MainPID=0, ExecMainStatus=0, active/exited. No sort, motion rerun, production channel change or hold removal.

Main figure: testing/outputs/luke_stationary_stage_audit_v2/02_origin_summary.png and .pdf. Supporting replay/cache waveform figure and arrays, combined stage_metrics.csv, resolved provenance, launch commands and receipts retained. Both figure sets visually inspected. Note that 01_stage_comparison contains the recovered interpolation/cache traces only; first-three-stage waveform arrays were not persisted before v1 failed, while their quantitative metrics were retained in its log.
