# Frozen-model temporal validation: central benefit transfers, shallow discrepancy remains

The shared-response model fitted on 4180–4190 s was applied without refitting to 4240–4260 s. Central DREDGE estimates recovered upward movement while the original inputs again produced essentially zero. Both arms retained a large shallow displacement inconsistent with the selected lighthouse centroids. Compensation addresses one demonstrated failure mode; motion estimation is not yet validated across the probe.

## Predeclared comparison

The separate interval was selected from existing lighthouse tracks for upward recovery and multi-depth coverage before computing its compensated outputs. This is temporal transfer within a known recording, not blinded selection or a new recording. The model, original per-channel detection noise vector, and all DREDGE settings were frozen from the preceding development experiment. SHA256 hashes and resolved settings are saved in `testing/outputs/luke_compensation_validation_v1/settings.json`.

Both arms use fresh 5-sigma negative locally exclusive detections with 50 µm exclusion radius, followed by monopolar localization at 75 µm radius and identical nonrigid DREDGE. Compensation is predicted using padded voltage, avoiding the earlier diagnostic's zero-prediction edge samples. Both arms exclude the first/last 50 samples from detection comparisons. No channel blacklist or lighthouse information enters the model.

## Neural preservation and event recovery

- All 18 unit/half median-waveform checks passed the previously fixed cosine ≥0.9 and amplitude ratio 0.8–1.2 gates.
- Minimum cosine: 0.990636. Amplitude ratios: 0.993631–1.005886, changes below 0.7%.
- Total peaks: 46,329 original, 41,229 compensated (11.0% fewer).
- Among 1,891 independently matched reference events, both arms recovered 1,451 by the same timing/depth coincidence rule. Event-level comparison: 1,434 recovered by both; 17 original-only; 17 compensated-only; 423 by neither.
- All changes in reference coincidence occurred for units 510 and 549. The other seven units' event recovery was unchanged.
- Some second-half waveform medians use only 10–14 events (units 154, 317, 445, 463); their precision is limited. The selected waveform cohort does not establish safety for all neural activity. Coincidence is not detection precision.

## Motion result

Changes are second-half minus first-half, using temporal medians for DREDGE and the corresponding median-waveform centroid difference for lighthouses.

| Unit | Depth (µm) | Original DREDGE (µm) | Compensated DREDGE (µm) | Lighthouse centroid (µm) |
|---|---:|---:|---:|---:|
| 80 | 220 | 27.33 | 27.34 | 1.81 |
| 154 | 620 | 26.21 | 26.21 | 2.08 |
| 246 | 1360 | 0.88 | 9.51 | 5.61 |
| 317 | 1740 | approximately 0 | 4.33 | 3.68 |
| 445 | 2260 | approximately 0 | 1.43 | 3.40 |
| 463 | 2380 | approximately 0 | 1.15 | 3.82 |
| 510 | 2740 | approximately 0 | 1.02 | 1.15 |
| 549 | 2960 | approximately 0 | 1.03 | 0.74 |
| 587 | 3100 | approximately 0 | 1.11 | 0.28 |

The central result supports transfer of the stationary-contamination mechanism to a separate interval and the opposite movement direction. Agreement is incomplete: compensation underestimates centroid changes at 2260/2380 µm and overestimates at 1360 µm. Centroids remain descriptive, uncalibrated displacement proxies, so these differences are not formal physical error measurements.

## Remaining shallow failure

Both arms give approximately +26–27 µm at the shallow lighthouse depths while those cells' centroids change about +2 µm. The intervention barely changes that region. A complementary amplitude-weighted population-profile diagnostic prefers +6 µm over 0–900 µm and +4/+5 µm over 400–1000 µm. These broad alignment maxima do not favor the approximately +27 µm DREDGE result. The 400–1000 µm profile correlation is relatively weak (maximum approximately 0.35).

These profiles pool each 10-second half and are not DREDGE's individual pairwise constraints. They motivate inspecting shallow pairwise displacement/correlation matrices and temporal consistency before tuning any thresholds. Weak or inconsistent pairwise registration is a hypothesis, not an established cause. We should not attribute this discrepancy to the already-demonstrated central shared artifact or force the estimator toward lighthouse values.

## Artifacts, verification, and next bounded step

- Main figure: `testing/outputs/luke_compensation_validation_v1/01_motion_validation.png` and `.pdf`.
- Neural checks: `02_neural_preservation.png` and `.pdf`.
- Population diagnostic: `03_population_alignment.png` and `.pdf`.
- All peak arrays, localizations, fields, event-level recoveries, waveform checks, comparisons, and summaries are in that output directory.
- Reproduce with `testing/luke_compensation_validation.py`; supplementary analysis with `testing/luke_compensation_validation_profiles.py`.
- Independent service `luke-compensation-validation-v1` survived launcher disconnection and completed with zero exit status. Persisted launch command, service log and job receipt are alongside the output directory. Both motion fields are finite, 20 × 20, with bin centers 4240.5–4259.5 s. Model hash and exact estimator-settings equality were checked. All three exported figures were visually reviewed.

Next: inspect the shallow DREDGE pairwise constraints using these cached inputs, preserving all settings, to locate how the large offset arises. This is a small estimator diagnostic, not another preprocessing grid. Temporal model stability farther from training and broader neural preservation remain untested. No production correction or full sort was launched; the full-sort hold remains.
