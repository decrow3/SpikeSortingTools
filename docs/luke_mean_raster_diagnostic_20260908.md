# AP raster aggregation: amplitude mean versus sum

**Reject global mean-amplitude aggregation as the next working configuration.** It improves some shallow comparisons but introduces substantially worse deeper disagreement. Keep the compensated broadband 3σ amplitude-sum baseline; these results do not justify a depth-dependent hybrid selected from this same interval.

## Controlled comparison

Both arms use the identical 1,299,549 compensated 3σ peaks, amplitudes, and locations from 4160–4260 s. Only aggregation changes: sum of absolute amplitude versus mean absolute amplitude per occupied 1 µm × 1 s bin, before the same spatial and temporal smoothing. Empty-bin handling follows installed SpikeInterface. The selected aggregation is used consistently for initial registration, bounded correlation replay, and raster-dependent solver weighting. No voltage was read, no detections were changed, and no estimator parameters were tuned.

| Diagnostic | Amplitude sum | Amplitude mean |
|---|---:|---:|
| Equal-cell overall disagreement (µm) | 1.073 | 1.532 |
| Drop/recovery disagreement (µm) | 1.270 | 2.184 |
| Median cell p95 one-second field increment (µm) | 1.570 | 2.121 |

Overall disagreement increases approximately 43%; movement disagreement increases approximately 72%. These are provisional-reference comparisons, not ground-truth accuracy measurements. Greater or smaller field increments are diagnostic rather than an adoption criterion.

| Unit | Depth (µm) | Sum disagreement (µm) | Mean disagreement (µm) |
|---|---:|---:|---:|
| 80 | 220 | 2.272 | 2.132 |
| 154 | 620 | 2.107 | 0.935 |
| 246 | 1360 | 0.828 | 0.629 |
| 317 | 1740 | 1.246 | 1.049 |
| 445 | 2260 | 0.797 | 2.226 |
| 463 | 2380 | 1.474 | 3.549 |
| 510 | 2740 | 0.192 | 1.444 |
| 549 | 2960 | 0.500 | 1.063 |
| 587 | 3100 | 0.244 | 0.758 |

The overlay shows new large central/deeper excursions, including unit445's vicinity near 4235 s, alongside better shallow agreement. Actual saved input rasters show changes in the relative prominence of depth bands while the low-activity region around 4225–4240 s remains conspicuous. Raster color scales differ by aggregation because the quantities have different units; the images are for profile structure, not direct cross-row amplitude comparison.

## Qualified unit154 transition check

Sampling both fields at the newly accepted unit154 event times reproduces the independently calculated five-second changes:

- Reference interval 4240–4245 s: 49 accepted events, fixed zero offset.
- Transition interval 4245–4250 s: 26 accepted events; amplitude sum **−10.312 µm**, amplitude mean **−7.231 µm**.
- The independently measured **centroid of the median waveform** changes **−5.770 µm** across these intervals. This is a different estimator from the earlier median of per-event centroids; those quantities must not be conflated.
- The subsequent five-second intervals contain only 3 and 5 accepted events and remain unsupported. No recovery trajectory is inferred.

Mean aggregation is descriptively closer to the median-waveform observation for this cell and transition. Model-based fractional-shift calibration suggests useful sensitivity on the negative-shift side (−10 µm injected displacement gives approximately −9.09 µm measured change), but does not establish biological ground truth. That limited local result cannot validate the 220/410 µm region, rescue unit80's failed qualification, or outweigh the broader regressions.

## Interpretation and verification

This experiment demonstrates that AP raster representation materially changes motion estimates even with fixed event membership, amplitudes, and locations. It does not establish that population firing rate is the sole source of bias. Normalized correlation already cancels uniform profile scaling; the plausible mechanism concerns depth-dependent population composition and occupancy. Mean-per-occupied-bin aggregation can give sparse bins disproportionate influence and cannot remove composition changes. There is no basis here for a globally adopted mean raster or a fitted depth switch.

The amplitude-sum replay is **exactly identical** to the saved baseline for D, C, U, displacement, time bins, and depth bins (independent array-equality check). Both arms' bounded replay passed against their respective initial registration correlations. The generic independent audit's hardcoded amplitude-sum curve reconstruction was not used for the mean arm.

Saved-array renderer `testing/luke_mean_raster_figures_v3.py` produced and visually inspected:

- `testing/outputs/luke_mean_raster_diagnostic_v3/02_all_unit_overlay.png` and `.pdf`: nine-cell event-matched displacement comparisons with conditional bootstrap intervals; blue solid sum, magenta dashed mean, black provisional references.
- `testing/outputs/luke_mean_raster_diagnostic_v3/03_actual_rasters.png` and `.pdf`: full-depth and shallow views of the actual saved smoothed raster inputs, with separate row scales and explicit 99.5th-percentile clipping.
- `testing/outputs/luke_mean_raster_diagnostic_v3/qualified_unit154_sampling.csv`: reproducible accepted-frame sampling and unsupported subsequent bins.

The renderer used only cached CSV/NPZ arrays and performed no fitting or voltage reads. The experiment's settings, source hashes, score tables, and completion summary remain in the same output directory.
