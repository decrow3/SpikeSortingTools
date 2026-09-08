# Stationary sources: voltage audit and common-signal mechanism

The central stationary peaks have a concrete voltage-level explanation: a probe-wide event waveform is largely removed by global median referencing, but large fixed-channel residuals remain where channel responses differ. This strongly supports a common disturbance with channel-dependent residuals, rather than localization alone pinning otherwise moving sources. The upstream physical origin is not established.

## Direct paired-voltage evidence

Compared 4180–4190 and 4190–4200 s on the accepted conditioned imec0 voltage, with the same 300–6000 Hz third-order forward/backward filter and global median reference as the threshold audit. Baseline source templates were built from the upper amplitude quartile of source-channel detections; all fresh 5σ detections within ±100 µm were candidates, with no target localized-depth gate. Fixed-channel weighted waveform matching (cosine >=0.9, gain 0.5–2, ±3 samples) selected repeated events; local template support ±140 µm.

- Channel-238 source: 993/959 matched events before/after, multichannel waveform cosine 0.9944; voltage centroid change −0.557 µm, localized-depth change −0.796 µm.
- Channel-251/252 source: 1401/1392 matched events, cosine 0.9954; voltage centroid change −0.085 µm, localized-depth change −0.067 µm.
- The channel-238 cohort's largest voltage energy actually lies near channels 251–253 within the expanded footprint. Source channel does not identify an isolated biological unit.
- Channel-cohort medians without amplitude or waveform-score selection also show little centroid change (+0.620 and −0.219 µm). They remain detection-channel conditioned, so are supplementary rather than independent identity validation.

Centroid changes are descriptive, not calibrated physical displacement. The important observation is overlapping waveform and channel-amplitude patterns, with strong peaks recurring on separated channels.

## Full-probe follow-up

The two apparent source groups are overwhelmingly synchronous: 99.9488% of 1952 matched channel-238 source events have a channel-251/252-group event within ±0.2 ms. The latter group has 2793 events. These are directed coincidence counts, not a one-to-one mapping.

For 300 uniformly selected matched 251/252-group events, the event-triggered global median reference waveform has a 207.5 µV peak. Before subtracting it, 381 of 384 channels have event-averaged waveform cosine >0.9 with that reference waveform, and all 384 have event-averaged absolute peak >50 µV.

Global median referencing removes most of this probe-wide component, leaving particularly strong residuals on channels with a different response:

| Channel | Before-reference peak µV | Waveform cosine to global reference | After-reference peak µV |
|---|---:|---:|---:|
| 252 | 118.3 | 0.802 | 184.5 |
| 251 | 127.0 | 0.854 | 171.1 |
| 253 | 129.1 | 0.866 | 161.3 |
| 238 | 133.2 | 0.930 | 128.4 |
| 248 | 142.1 | 0.962 | 99.4 |

The subtraction can leave a residual larger than that channel's pre-reference event waveform because its time course differs from the population median. This explains why referenced artifacts can look spatially localized and spike-shaped. A morphology screen limited to already-referenced voltage can therefore mistake them for neural events.

These are conditional event averages, not per-event coherence claims or full-recording prevalence estimates. They establish a repeatable shared component in this epoch; they do not identify its hardware or biological origin. The cached voltage already includes phase correction, saturation blanking and bad-channel interpolation, so upstream acquisition and conditioning remain to be separated.

## Implication for the motion audit

Together with the earlier zero-shift peak-profile objective, this is a plausible mechanism for the central registration estimate being anchored by stationary residuals. The direct evidence applies to the current lighthouse/threshold conditioning branch. Historical pipelines used different conditioning; the same mechanism must be verified in their exact inputs before claiming it explains every historical field. Native KS large jumps may have an additional cause.

Next focused work should inspect these channel responses in acquisition voltage and prior conditioning stages, then test targeted common-signal/response correction or event rejection while protecting coincident lighthouse spikes. Do not simply delete every synchronous event or relabel stationary waveforms as artifacts. No production channel exclusions, reference changes, sorting or motion reruns were performed here.

## Figures and execution

- testing/outputs/luke_stationary_voltage_audit_v1/01_stationary_voltage.png and .pdf: paired source waveforms, energy footprints and channel amplitudes. Full waveform evidence and event lists saved.
- testing/outputs/luke_stationary_common_signal_v1/01_common_signal.png and .pdf: full-probe before/after reference and subtraction waveform. Full event-averaged arrays and channel-coherence table saved.

Both managed jobs completed with MainPID=0, ExecMainStatus=0, active/exited; launch commands, logs and receipts persisted under testing/outputs. Figures visually inspected. First audit asserted raw file size/mtime unchanged. Full-sort hold preserved.
