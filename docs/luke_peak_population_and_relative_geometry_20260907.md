# Peak population and provisional relative lighthouse geometry

Priority: inspect and aggressively condition motion-estimation inputs before further estimator comparisons. Sorting source voltage is unchanged. These are cached-window diagnostics, not amplitude-completeness estimates or authorization for a full sort.

## Motion-only artifact mask

`testing/luke_peak_population_review.py` exports count and absolute-amplitude histograms (0.25 s × 10 µm), individual waveform decisions, and a stricter mask under `testing/outputs/luke_peak_population_review_v1/`.

On 4180–4200 s: 60,719 original peaks; previous mask rejects 3,362; new mask rejects 4,104 (6.76%), including 742 additional events. It rejects broad-coherent (≥0.8), shared-response-explained (energy ≥0.8), waveform-matching (cosine ≥0.95) events even when a local remainder exceeds the previous rescue threshold. Thresholds and model are existing diagnostic settings, not optimized against motion estimates. This is one artifact-family rule, not a neural-only classifier. Six individual examples are deterministic middle-index selections within decision category and depth stratum, not a representative prevalence estimate. Previously rescued examples remain predominantly explained by the common signal.

The separate 930–1030 s comparison uses original versus shared-response-compensated voltage with fresh detection, not this event-rejection mask. Strong horizontal bands near 2380–2520 µm are visibly reduced while moving population bands remain. Remaining structure and individual retained waveforms need further review; absence of this artifact signature is insufficient evidence of neural origin. Stationarity is suspicious when a waveform remains channel-locked despite independently corroborated *local* biological displacement; stationarity alone is not a universal rejection rule.

## Provisional relative geometry

`testing/luke_lighthouse_relative_review.py` reuses the nine original gentle-period tracks, retaining low-count measurements and marking bins with fewer than ten accepted events. Outputs under `testing/outputs/luke_lighthouse_relative_review_v1/` include cell changes and adjacent-pair separation changes. No DREDGE estimate enters the calculation or selection.

Across the 4185→4195 s step, all nine centroid changes have the same sign. Units 80–463 move approximately −4.6 to −9.6 µm; units 510/549/587 move −1.13/−0.48/−0.30 µm. Depth order is preserved. Adjacent pair separation ranges over the full 100 s are approximately 0.74–7.69 µm: relative geometry is not strictly rigid. This supports a coordinated event with depth-dependent apparent displacement, rather than proving a calibrated physical displacement field. Fixed-template acceptance can hide large movement and underestimate displacement.

Isolated departures should be flagged with local neighboring cells and waveform/acceptance evidence; coherent depth dependence should not be rejected merely for disagreeing with a single whole-probe median. This descriptive figure does not automatically reject cells or assign artifact identities.

## Next diagnostic step

Use the population histograms to select remaining stationary ridges and neighboring moving bands, then inspect included/excluded multi-channel waveform cohorts and their time dependence. Build an aggressive motion-only inclusion mask from those observations, accepting lost neural events where necessary, before another estimator comparison. Keep ambiguous lighthouse measurements available with flags and local corroboration. Preserve the original signal for subsequent standard sorting preprocessing and application of any validated motion field.
