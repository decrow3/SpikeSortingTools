# Remaining motion-input peak populations

Input cleanup is the current priority. No new motion estimator or sort was run.

## Completed review

Reviewed 288 individual compensated waveforms from 12 detector-channel populations in 930–1030 s. Selected three populations in each 960 µm probe quarter using 1-second occupancy times the fraction of events localized within 5 µm of detector depth. This ranks possible channel-locked structure for review, not artifact probability. Each population contributes 24 evenly indexed events. Source peaks and shared-response model are reused from existing diagnostics.

Scripts: `testing/luke_remaining_peak_cleanup.py` and `testing/luke_remaining_peak_cleanup_figures.py`. Outputs: `testing/outputs/luke_remaining_peak_cleanup_v1/`, including channel_population.csv, waveform_review.csv, all sampled full-probe waveforms and peak indices, settings.json, population/depth contact sheet, and three readable waveform sheets in PNG/PDF.

## Findings and implications

- Detector populations at 3140 and 3380 µm contain large repeatable positive-leading waveforms. Median maximum detector-channel amplitudes are 16.3 and 15.6 residual-noise sigma; local early/late median footprint cosines are 0.94 and 0.93. These are plausible neural signals despite high localization concentration near their detector depths. This is not proof of identity or local biological stationarity.
- Other populations show mixtures, collisions or large off-center excursions in individual traces. For example, ch173 has poor early/late local footprint agreement (0.30), and individual positive excursions at varying time offsets dominate some snippets. Morphology alone does not establish their origin. These individual events are poor candidates for an aggressively selective motion input until separated into waveform families.
- A detector-channel population is not a single cell. Rejecting all events on a channel because its average waveform or cohort consistency is poor would conflate mixtures with artifact identity. Event-level selection remains the appropriate next implementation.
- Negative-only detection includes trailing negative phases of positive-leading waveforms. Any morphology screen must evaluate the full local waveform and dominant lobe; a narrow negative-trough-only test would misdescribe these examples. This review does not establish that trailing-phase detection causes localization failure.

## Next cleanup implementation

Separate repeatable local waveform families from ambiguous individual events using multichannel shape, amplitude above noise, and shared-artifact explanation. Use an explicit uncertain/excluded category for motion estimation; preserving every neural event is unnecessary. Check retained/rejected waveforms and the resulting depth-time histograms before another motion-estimator trial. Channel-lock evidence should be assessed against local corroborated motion, not inferred from a flat trace alone. The existing stricter shared-artifact mask remains the implemented removal rule; this review introduces no new production mask or channel blacklist.

## Limits

These 288 targeted examples do not estimate contamination prevalence. Residual noise normalization comes from the frozen model's 4180–4190 s training window, not a new epoch-specific MAD estimate. Early/late groups contain 12 events each, not equal-duration halves. Shape scores use detector-time alignment without waveform realignment; mixtures and motion can lower agreement. No inference about amplitude completeness is made.
