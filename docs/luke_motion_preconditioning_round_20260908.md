# AP motion preconditioning: first implementation round

**Keep compensated broadband 3σ as the development baseline. Neither additional 3 kHz low-pass filtering nor gentle center-energy screening earns promotion.** The completed comparisons locate the main new failure in shallow registration and show that changing amplitudes and localizations can recreate it even with identical events. Recording-wide motion is not yet validated.

## Completed controlled comparisons

The six-arm experiment reused all 1,299,549 baseline peaks/locations exactly, kept the frozen shared-response model and DREDGE settings, and enforced the same ±80 µm correlation bounds. Noise adjustment was calibrated once on 4180–4200 s before comparisons. All references and scores cover the already-inspected 4160–4260 s interval.

| Input | Peaks | Overall lighthouse disagreement (µm) | Drop/recovery disagreement (µm) |
|---|---:|---:|---:|
| Compensated broadband 3σ | 1,299,549 | 1.073 | 1.270 |
| Broadband + gentle screen | 804,882 | 1.044 | 1.331 |
| Low-pass, fixed absolute thresholds | 412,662 | 2.197 | 1.701 |
| Fixed-threshold low-pass + screen | 330,169 | 2.390 | 1.800 |
| Low-pass, noise-adjusted thresholds | 867,571 | 1.943 | 1.826 |
| Adjusted-threshold low-pass + screen | 627,689 | 2.155 | 1.787 |

These are differences from provisional lighthouse centroids, not ground-truth errors. Approximately 98% of the adjusted-filter excess overall disagreement is contributed by unit80 at 220 µm. Excluding it is useful to locate the problem, not to excuse it. The central region does not reproduce the earlier dominant 55–65 µm screening failure.

Exact shallow correlation replay shows increased weight on competing large shifts and cycle inconsistency increasing from 9 to 60 µm (95th percentile). A four-thread replay changed one near-tied lag; matching the original single-thread backend reproduces every saved lag and correlation. Subsequent cached controls use this matching numerical setting.

## Why this is more informative than another score sweep

The median channel noise falls to 81% after filtering, but individual lighthouse peak amplitudes fall to different fractions of broadband: approximately 44% for unit510, 56% for unit80, and 96% for unit587. Noise adjustment therefore does not preserve the original neural population weighting.

A separate 316,089-exact-event factorial holds frame/channel membership fixed. Overall disagreement is 1.243 µm with broadband amplitudes/locations, 1.481 with filtered amplitudes alone, 1.532 with filtered locations alone, and 1.934 with both. The latter nearly reproduces the full filtered population's 1.943 µm. Event depletion alone is not sufficient to explain this failure. The intersection is biased by timing/channel changes and the interactions are nonlinear; this is not an additive decomposition of physical error.

Paired localization changes are distributed and bidirectional rather than a shared displacement: 40% exceed 5 µm in magnitude, while the median signed change is near zero. Small changes in median lighthouse energy centroids do not imply that all detected-peak localizations are preserved.

## Reference check and remaining decision

Existing ten-second centroids cannot adjudicate brief shallow motion. A bounded quiet-data audit replaced the old rival-template pre-pruning with nearby labeled cohorts plus independent waveform families. It uses no DREDGE input and distinguishes identity from shift ambiguity.

- Unit80: 43 labeled cohorts, 12 independent families, 198 spatial hypotheses. It fails the frozen sensitivity gate (65.3% injected target/shift recovery), despite no tested rival false acceptance. This is a measurement limitation, not evidence that the cell is non-neural or necessarily misidentified.
- Unit154: 25 labeled cohorts, 12 independent families, 158 hypotheses. It passes this preliminary gate (92.7% recovery, 60% quiet independent-detection recall, no tested rival false acceptance). Only this cell proceeds to a separately evaluated bounded transition audit.

No session-wide lighthouse census, new sort, or production interpolation was performed. The remaining shallow uncertainty should be resolved locally before tuning the estimator to these references or expanding to recording-wide validation.

### Completed bounded transition and measurement checks

The frozen unit154 matcher accepted 83 events over 4240–4260 s. Only the first two five-second bins have sufficient support (49 and 26 events); the later bins have 3 and 5 events and remain gaps. One late event reaches the −80 µm bank boundary. No transition-time DREDGE estimate entered event selection.

The median of individual-event energy centroids initially suggested only −1.225 µm movement between the supported bins, versus −10.312 µm for broadband DREDGE at the same events. Paired real-background injections exposed substantial compression in that event statistic: a nominal −10 µm injection appears as −5.47 µm at gain1.

Using the **centroid of the median waveform** instead gives **−5.770 µm** on the real supported bins. The same revised statistic recovers −9.09 µm for a nominal −10 µm injection and +6.92 µm for +10 µm at gain1. Much of that directional asymmetry is already present in the linearly interpolated injection template. These controls validate limited measurement sensitivity, not physical ground truth; no single slope correction or physical DREDGE-error estimate is justified. No confidence interval for the real transition was computed.

This is a concrete reason not to optimize motion toward stationary-looking lighthouse summaries. See [transition audit](luke_shallow_transition_audit_20260908.md) and [measurement calibration](luke_shallow_fractional_calibration_20260908.md).

### One actual raster-representation test

With all 1,299,549 broadband events and locations unchanged, a single mean-versus-sum amplitude-per-bin test used consistent representations in registration, strict replay, and solver weighting. The sum arm reproduced the saved baseline. Global mean aggregation worsened overall disagreement from 1.073 to **1.532 µm** and drop/recovery disagreement from 1.270 to **2.184 µm**.

At the new unit154 transition it changes the predicted displacement from −10.312 to −7.231 µm, closer to the descriptive −5.770 µm median-waveform value. That local improvement does not outweigh regressions in five deeper units. Global mean aggregation is not promoted. No depth-dependent hybrid was constructed from these same-window scores. [Raster comparison report](luke_mean_raster_diagnostic_20260908.md).

The retained configuration is therefore unchanged: compensated broadband3σ, amplitude-sum raster, original DREDGE settings. The next substantive requirement is independently supported shallow motion and transfer beyond this development interval, not additional global filtering/screening sweeps. Recording-wide validity, later sparse periods, and the 220/410 µm region remain unproven.

## Reproduction and verification

- Working motion-only configuration: `configs/luke_motion_diagnostic_v2.json`. Its status explicitly excludes production/recording-wide validation.
- Six-arm run: `testing.luke_3sigma_lowpass_screen_v2`; outputs `testing/outputs/luke_3sigma_lowpass_screen_v2/`.
- Independent field/curve audit: `testing.luke_lowpass_evidence_v2 --replay-curves`.
- Same-event waveform audit and cached factorial: `testing.luke_lowpass_waveform_preservation_v2`, `testing.luke_lowpass_common_events_v2`.
- Reference qualification: `testing.luke_shallow_identity_audit_v3`.
- Bounded reference follow-ups: `testing.luke_shallow_transition_audit_v3`, `testing.luke_shallow_fractional_calibration_v3`, and `testing.luke_shallow_template_centroid_audit_v4`.
- Actual raster representation comparison: `testing.luke_mean_raster_diagnostic_v3`.

All completed experiments ran under independent systemd user services with launch scripts, logs, receipts, and terminal manager-state checks preserved in their corresponding job directories. The launch method passed a separate disconnection-survival dummy test. The six-arm computation took approximately 17.5 minutes; the cached common-event experiment took 34 seconds. No automatic/within-stage resume is claimed. Cancelled v1 outputs and the full-sort hold remain intact.

Compilation, synthetic screening/matching checks, exact baseline reconstruction, independent score reproduction, and exact correlation replay passed. Waveform and motion figures were inspected. Detailed evidence: [six-arm audit](luke_lowpass_evidence_20260908.md), [waveform preservation](luke_lowpass_waveform_preservation_20260908.md), [common-event controls](luke_lowpass_common_events_20260908.md), and [literature/reference review](luke_ap_conditioning_research_20260908.md).
