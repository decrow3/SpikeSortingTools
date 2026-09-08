# Established alternative registration also fails the early compensated-input check

> Interpretation update: the fixed-position waveform matcher rejects exact ±40 µm translations of its own templates. Small centroids of accepted events therefore cannot decisively falsify the larger motion excursions. See `docs/luke_long_context_and_tracker_sensitivity_20260907.md`; earlier failure wording below must be read with this limitation.

Ran SpikeInterface's `iterative_template` registration on the exact cached original/compensated peak arrays and localizations at 970–990 and 9520–9540 s. This is the established histogram/template method implemented in SpikeInterface, not a new custom lighthouse estimator and not a native Kilosort sort.

Probe geometry and Gaussian window placement match the DREDGE comparison. Method-specific defaults were fixed and recorded before calculation, including 10 µm spatial bins and 2-second time bins. Search domains, histogram representation, binning and optimization differ from DREDGE. This is a method comparison, not an isolated solver ablation or a matched-resolution benchmark.

## Result

Five-second binned compensated-input displacement ranges:

| Candidate depth / interval | Iterative template (µm) | DREDGE (µm) | Provisional waveform range (µm) |
|---|---:|---:|---:|
| 2920 / early | 25.5 | 42.2 | 1.3 |
| 3380 / early | 20.1 | 40.4 | 1.7 |
| 2900 / late | 6.45 | 19.65 | 1.23 |
| 3300 / late | 4.55 | 21.55 | 0.73 |

Waveform ranges include only valid bins; the last candidate has only two valid bins and the early2920 candidate has one gap. These are descriptive ranges with different temporal smoothing, not calibrated error estimates. Smaller alternative-method excursions do not establish correctness.

The alternative method reproduces substantial early compensated-input excursions inconsistent with available provisional waveform observations. The problem is therefore not confined to the particular DREDGE solver. Shared-input population-profile changes, remaining contamination, short temporal context and identity limitations require further separation. Original-input near-zero estimates remain potentially artifact-pinned and are not validated by this comparison.

## Longer-context test launched

A separate independent job, `luke-long-context-validation-v1`, extends the early interval to 930–1030 s. It reuses the exact central970–990s peaks/localizations and adds fresh original/compensated inputs only in the surrounding80s. The model, thresholds and DREDGE settings remain fixed; strict±80um bounds apply. Both DREDGE and iterative-template estimates will be compared within the original central interval. Small50-sample exclusions at chunk edges are explicit. This is motion-context testing, not amplitude-completeness estimation.

At handoff, the100s job was verified live after launcher disconnection. Its outputs are pending; a live job is not evidence of a successful result. Completed chunk arrays, launch command, log and final receipt are persisted. No within-localization checkpoint or automatic restart is claimed.

## Artifacts

Alternative comparator script: `testing/luke_alternative_registration.py`; outputs: `testing/outputs/luke_alternative_registration_v1/`, including full resolved defaults, four fields, comparison CSV, summary and `01_method_comparison.png` / `.pdf`. Figure visually inspected; alternative-method service completed with zero exit status.

Long-context script: `testing/luke_long_context_validation.py`; outputs: `testing/outputs/luke_long_context_validation_v1/` and adjacent service log/job receipt. Actual process state must be checked before reporting subsequent progress. No production correction or spike sort was launched. Full goal remains incomplete.
