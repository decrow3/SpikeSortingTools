# Short-window diagnostic: truncation interpretation withdrawn

The 120-second amplitude-truncation analysis below is retained for provenance only.
Its fit coverage mainly reflects the 1,000-spike requirement in windows that are
too short for most clusters. **Neither its 5.9% coverage result nor its conditional
missingness medians should be used to assess motion-correction efficacy.**
The event-count and waveform panels remain useful descriptive diagnostics.

Replacement: full-recording, spike-count-based fits with actual window durations,
explicit gaps and unfit residual spikes. See `luke_full_reference_truncation_20260907.md`.

## Population and clock

The verified source is the 384-channel, 12/9 motion-off reference with identity
`22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`.
All 710 curated clusters remain in denominators; KS-good (301) is a stratum,
not an eligibility filter. These are cluster counts, not established neuron counts.

Six previously selected input/motion discovery segments are frozen in
`testing/outputs/luke_full_reference_diagnostic_panel_v1/manifest.json`.
Each segment is 600 seconds; event counts use twenty half-open 30-second bins.
Each has one fixed 120-second anchor. Times are relative to recording frame zero,
converted using the actual 29,999.835983263598 Hz sampling frequency.
The labels quiet, motion and dropout describe prior input/estimator signatures,
not ground truth or randomized conditions. This is not an independent holdout.

## Results

| Fixed-anchor status | All clusters × 6 | KS-good clusters × 6 |
|---|---:|---:|
| Zero events | 986 | 437 |
| Fewer than 1,000 events | 2,486 | 1,253 |
| Fit failed | 0 | 0 |
| Fit at the 50% ceiling | 425 | 9 |
| Valid, uncensored fit | 363 | 107 |
| Total | 4,260 | 1,806 |

Only 107/1,806 (5.9%) KS-good cluster-anchor observations produce an uncensored
fit. The zero-event observations comprise 24.2%. Across all clusters, 29,637 of
85,200 30-second bins contain zero events. These are repeated observations of
the same clusters, not independent samples for statistical testing.

The 1,000-event threshold requires approximately 8.33 Hz within a 120-second
anchor, so limited coverage is partly built into the metric. Sparse firing,
physiology, recording artifacts, sorting loss, and identity fragmentation can
all contribute; the present analysis does not distinguish them.

For a concrete reference-only example, curated KS-good cluster 686 has 188
spikes in the quiet anchor, 1,857 in the large-motion anchor, and zero in the
support-dropout anchor. Cluster 602 has 401, 164, and zero respectively.
These are diagnostic examples, not evidence that either represents one stable
biological unit or that motion caused its disappearance.

## Fit interpretation

The fixed-anchor diagnostic applies `pipeline.truncation.fit_amp_cdf` to every
retained amplitude in an anchor with at least 1,000 events. It captures the
legacy fitter's printed fallback as an explicit failure and separates fits
at its 50% ceiling. No fallback occurred in this run.

This differs intentionally from adaptive production QC, which constructs
1,000-event blocks separated by gaps greater than 10 seconds. Here physical
windows stay fixed and maximum internal interspike gap is reported. Results
must not be described as a recomputation of the adaptive QC metric. Fits
remain conditional on detected events and can neither establish nor exclude
loss of events absent from the sorted train. A non-ceiling successful numerical
fit is not a guarantee that the assumed distribution is appropriate.

## Waveform inspection

Twelve KS-good reference clusters were selected across four depth bands using
current curated cached waveforms. Each band contributes a median-activity
example and two examples with high variation in anchor counts. Duplicates are
removed. Selection is illustrative and deliberately enriched for variation.

The extraction plan fixes the peak channel and its 12 nearest channels per
cluster, with up to 32 evenly spaced event ranks per anchor. Voltage comes from
the accepted cached recording with its manifest gain; channelwise median of
the first 10 samples is subtracted. No additional motion correction, temporal
filter or reference is applied. The cached recording already includes phase
correction, blanking and channel interpolation, so this is not untouched
acquisition voltage. Event-free anchors receive no trace, never a zero waveform.

Event-triggered plots show only retained events, with background and possible
overlapping spikes. They do not independently detect missing events and do not
establish cross-sort identity. The raw waveform package is a reference for a
future comparison, not a biological recovery result.

## Validation and reproducibility

Both analyses use the previously verified independent systemd manager, with
exact launch commands, logs and final receipts under `testing/outputs/`.
Inputs are read-only; neither job launches a sort. A stopped analysis must be
rerun into a new output directory; there is no within-job checkpoint.

The output tables have unique keys, all 710 clusters in each anchor, and 20
bins per cluster per segment. Counts reconcile with slice event totals.
Independent per-unit NumPy histogram checks passed for four cluster IDs across
all six segments. Amplitude/time lineage was checked against the retained
source rows. Empty, insufficient and failed-fit edge checks passed.

Scripts: `testing/luke_full_reference_panel_analysis.py` and
`testing/luke_full_reference_panel_waveforms.py`. Results and validation receipt:
`testing/outputs/luke_full_reference_diagnostic_panel_v1/analysis/`.

## Next decision

Review huklaban5's full-probe rigid registration field and its support before
choosing a corrected full sort. Any later comparison must retain these same
physical windows, report both coverage and conditional fits, preserve zero-event
bins, and validate proposed unit families with spatial and waveform evidence
and exclusive event assignments. The full-sort hold remains in force.

## Completed voltage output and review

The sparse extraction completed with 1,778 snippets across 12 clusters,
requesting 111,971,328 bytes. Both plots were visually inspected for labels,
scales and missing traces. The coverage plot preserves all denominators;
waveform panels share the same channel and voltage axis across windows within
each cluster. Different clusters have different voltage axes.

| Anchor | KS-good zero-event clusters / 301 | Valid fits / 301 | Median missingness among valid fits |
|---|---:|---:|---:|
| Noise control | 129 | 12 | 3.69% |
| Relative quiet | 108 | 26 | 2.18% |
| Moderate supported motion | 28 | 16 | 2.33% |
| Large supported motion | 38 | 26 | 1.78% |
| Support-dropout control | 114 | 16 | 3.11% |
| Motion with input anomaly | 20 | 11 | 2.21% |

These medians involve different eligible subsets, so their ordering must not
be interpreted as a paired motion effect. In particular, more zero-event
clusters in the quiet anchor than the large-motion anchor contradicts any
simple interpretation that the present zeros are necessarily motion loss.

The waveform montage shows both similar waveforms across anchors and marked
changes in some selected clusters (for example 676 in the support-dropout
anchor). These are detections from the uncorrected reference, not proof of
stable identity or correction benefit. Background, conditioning, overlapping
events, sample size, and biological changes remain possible explanations.

Plots: `testing/outputs/luke_full_reference_diagnostic_panel_v1/analysis/voltage/fit_coverage.png`
and `waveform_panel.png` in that same directory. Snippets, event-frame selection,
per-anchor sampling counts, and provenance are saved alongside them.
