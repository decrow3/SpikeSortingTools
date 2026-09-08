# Longer context and a limitation in the waveform corroboration

The100-second context experiment completed, and the central970–990s peak/location arrays were verified exactly identical to the short-window inputs in both arms. Increasing context to930–1030s did not remove the central excursions:

| Depth | DREDGE20s /100s range (µm) | Iterative20s /100s range (µm) |
|---|---|---|
| 2920 µm | 42.2 /46.0 | 25.5 /15.55 |
| 3380 µm | 40.4 /30.17 | 20.1 /12.8 |

However, a subsequent sensitivity check exposed a critical limitation in interpreting disagreement with the provisional waveform centroids.

## Fixed-position matching rejects translated versions of the same template

Injected each of the four early/late local training waveforms into otherwise zero voltage, translated by exactly0,±40,±80µm using existing same-x contact geometry. This is a noiseless matcher diagnostic, not a biological motion simulation. It preserves the waveform shape and uses the existing frozen matcher with its0.9 cosine/gain gates.

All four templates passed at zero shift (cosine approximately1, gain1). Every±40µm and±80µm translation failed. At±40µm, cosines ranged approximately0.23–0.37, far below0.9, with gains also below0.4. The current fixed-position matcher therefore cannot provide representative position estimates during a40µm excursion.

A bin containing actual movement could lose most moving spikes and retain only events near the original footprint. Its conditional centroid could remain nearly stationary. Early matched-event counts drop strongly during the estimated excursion (for example107→21 events for the3380µm cohort), making this selection mechanism plausible. This observation does not prove actual movement, but it prevents using those stationary centroids as decisive evidence against the motion fields.

## Correction to earlier interpretation

The earlier statements that the distant comparison or alternative estimator had "failed" were too strong when based on these fixed-template centroids. The established result is disagreement with a position-selective subset of events, not a calibrated demonstration that the20–40µm estimates are false. Both the motion estimates and the provisional identity checks remain unvalidated for those excursions.

This does not reverse the demonstrated central stationary-input mechanism, waveform-preservation results, pairwise inconsistencies, or search-bound implementation discrepancy. It limits the claimed inference from the newer early/late centroid comparisons. More context changes estimates but does not settle their physical correctness.

## Next necessary step

Use translated-template comparisons with identity competition and synthetic recovery controls before interpreting these early/late cells through the estimated excursions. Keep motion estimates out of template selection and search calibration. A spatially flexible matcher can itself confuse similar cells at neighboring depths, so recovery of translated waveforms is necessary but not sufficient. This remains corroboration work; the user-deferred custom continuous motion estimator is not being revived.

## Artifacts and verification

Long-context scripts: `testing/luke_long_context_validation.py`, `testing/luke_long_context_analysis.py`. Outputs in `testing/outputs/luke_long_context_validation_v1/`, including exact-input verification, four motion fields, comparison tables and reviewed PNG/PDF figure. Independent service completed with zero exit status; actual terminal state was checked.

Sensitivity script: `testing/luke_tracker_shift_sensitivity.py`. Outputs in `testing/outputs/luke_tracker_shift_sensitivity_v1/`, including settings, all20 injected cases and reviewed `01_shift_sensitivity.png` / `.pdf`. Script completed successfully. No production correction or sort was launched; the full goal remains incomplete.
