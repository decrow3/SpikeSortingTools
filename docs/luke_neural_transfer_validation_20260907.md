# Neural transfer validation: preservation demonstrated at a distant middle epoch; coverage incomplete

Screened provisional existing-sort candidates on original voltage in 960–970, 5710–5720 and 9510–9520 s before assessing compensation. The shared-response model remained frozen from 4180–4190 s. Candidate thresholds were unchanged: at least 15 events in each five-second half, peak ≥150 µV and ≥10 local MAD noise units, half-template cosine ≥0.9, ≥70% energy within ±60 µm, and ≥80% second-half events with peak-channel cosine ≥0.8. A 120 µm probe-edge margin was applied. No motion/lighthouse agreement entered screening.

474 cluster/epoch candidates had sufficient events. Four passed all criteria, all at 5710 s:

| Provisional unit | Depth (µm) | Original peak (µV) |
|---|---:|---:|
| 341 | 1820 | 158 |
| 445 | 2240 | 202 |
| 632 | 3240 | 256 |
| 666 | 3540 | 269 |

All eight half-interval preservation checks passed cosine ≥0.9 and amplitude ratio 0.8–1.2. Minimum preservation cosine was 0.998360; amplitude ratios were 0.993542–1.005053. These are fixed-event median waveform comparisons, not independent detection precision or identity validation.

The complete four-candidate contact sheet was visually inspected. Units 341 and 445 have clear trough/recovery waveforms. Unit 666 has a positive-dominant waveform, which is not automatically non-neural but requires independent identity qualification. Unit 632 has a double-peaked positive component and should remain provisional pending exclusion of superposition or mixed identity. Numerical screening is not enough to certify every candidate as a lighthouse.

## Early and late coverage gaps

No candidate passed every fixed criterion at 960 or 9510 s. At 960 s, unit 628 has a large 358 µV, 42-noise-unit waveform, good half cosine (0.991) and compact footprint (0.927), but second-half event shape fraction is 0.790 versus the fixed 0.800 gate. No threshold was relaxed. Other large candidates fail more substantially. At 9510 s, unit 445 reaches 185 µV and 16.3 noise units but half cosine is 0.886 and local energy fraction 0.637, both below gates.

This does not show compensation failure in those epochs: qualification occurs before compensation. It shows that the current short-window, existing-sort candidate route provides insufficient early/late coverage. Existing sorted identities may mix events, and short windows may have inadequate stable candidate evidence. Missing qualification must not be reported as preservation success or absence of neurons.

## Next step

Address the early/late coverage using independently detected large-waveform cohorts with explicit morphology and repeatability checks, rather than weakening quality gates or assuming existing cluster identity. Unit 628 near the early epoch is a useful review lead, not an approved lighthouse. The middle-epoch candidates can support a subsequent independently validated tracking check after identity qualification.

The full goal remains incomplete: motion reliability at shallow depths and difficult epochs, recording-wide neural preservation, and downstream sorting/amplitude completeness are not yet established. No full sort is justified by this result alone.

## Artifacts and execution

Main script: `testing/luke_neural_transfer_validation.py`; contact sheet: `testing/luke_neural_transfer_figures.py`.
Outputs: `testing/outputs/luke_neural_transfer_validation_v1/`, including candidate and preservation CSVs, model hash/settings, all passing median waveforms, coverage/preservation figures and `03_waveforms.png` / `.pdf`.

Independent service `luke-neural-transfer-validation-v1` was verified live after launcher exit and later completed with zero exit status. Saved launch, log and job receipt are adjacent to the output directory. Contact-sheet generation also completed successfully. No production data, library or sorting settings were modified.
