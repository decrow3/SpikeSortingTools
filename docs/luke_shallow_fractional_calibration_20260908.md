# Unit154: fractional sensitivity changes the motion interpretation

**The original −1.225 µm individual-event centroid change understated the movement visible in the median waveform.** The same 49 and 26 accepted events give a median-waveform centroid change of **−5.770 µm** across 4240–4245 to 4245–4250 s. Paired quiet-background injections support using the median-waveform statistic for further diagnostic comparisons, but do not convert its difference from DREDGE into a measured physical motion error.

## What the calibration established

The frozen unit154 matcher scored independent-half templates interpolated along actual same-x probe columns at requested shifts −20, −10, −5, 0, +5, +10, +20 µm. Each amplitude gain (0.75, 1, 1.25) used 40 paired real quiet backgrounds, for 840 known-center injections. There was no redetection or DREDGE input. Exact −80, −40, 0, +40, +80 µm interpolation matched the saved geometric controls with zero waveform difference.

The first statistic—median of individual-event energy-centroid differences—compressed the response substantially and depended on amplitude. At gain 1, requested ±10 µm gave −5.468/+3.950 µm, with 75%/82.5% identity-and-unique-shift acceptance. Its descriptive response slope was 0.461 across supported tested shifts. Those results invalidate treating the real −1.225 µm statistic as a direct displacement reference. A global slope correction is also unjustified: response depends on direction, amplitude, background, and conditional acceptance.

The second audit froze every match and timing decision and instead measured the **centroid of the median waveform**. Shift and zero measurements used exactly the same accepted background pairs; both members had to retain unique zero-grid winners. Primary summaries require ≥10 pairs and requested shifts within ±10 µm. No rescore or tracking changes were made.

At gain 1:

| Requested interpolation shift (µm) | Clean interpolated template centroid change (µm) | Median individual-event change (µm) | Median-waveform change (µm) | Accepted paired events |
|---|---:|---:|---:|---:|
| −10 | −9.455 | −5.468 | −9.090 | 30/40 |
| −5 | −4.318 | −2.560 | −4.247 | 38/40 |
| 0 | 0 | 0 | 0 | 40/40 |
| +5 | +3.096 | +1.656 | +2.929 | 38/40 |
| +10 | +7.244 | +3.950 | +6.916 | 33/40 |

The median-waveform response is much closer to the clean injected waveform's energy-centroid change. **The injection model itself is directionally asymmetric:** a requested +10 µm interpolation moves its clean energy centroid only +7.244 µm, whereas −10 µm gives −9.455 µm. Thus the unequal recovered responses cannot all be attributed to matcher or measurement bias. Linear voltage interpolation is an approximation, not a validated model of physical tissue displacement.

At −10 µm, the median-waveform response ranges from −8.448 to −9.354 µm over the three gains; +10 µm ranges from +6.416 to +6.976 µm. Lower amplitude also loses more pairs. The real accepted-event median fitted gains were approximately 1.060 and 1.095 in the two supported bins, but this does not establish equality of real transition and quiet injected signal/noise conditions. Requested ±20 µm produced few accepted zero-grid pairs and remains outside the supported primary range. A coarse zero-grid winner clearly does not imply zero sub-grid displacement.

## Real transition interpretation

| Measure, second five-second bin relative to first | Change (µm) |
|---|---:|
| Compensated 3σ DREDGE, evaluated at accepted frames | −10.312 |
| Low-pass adjusted-threshold DREDGE, same frames | −11.784 |
| Median-waveform energy centroid, fixed ±120 µm support | −5.770 |
| Earlier median individual-event energy centroid | −1.225 |

The median-waveform centroids are 619.329 and 613.560 µm. This measurement better retains modeled fractional sensitivity than the earlier event-wise statistic. A residual numerical difference from DREDGE remains worth investigating, but **no uncertainty interval was computed for the real median-waveform change**, and the injected model does not establish biological displacement truth. Do not call the approximately 4.5 µm numerical gap a physical estimation error, rescale real movement by one global calibration gain, or tune the raster toward −1.225 µm.

The v4 bootstrap intervals resample the fixed accepted injected background pairs 200 times. They describe conditional sampling variability under that injection model, not real transition uncertainty, identity confidence, or physical accuracy. Selection and finite-sample median bias remain; some percentile intervals need not contain the observed plug-in estimate.

## Decision and boundaries

A single cached-data sum-versus-mean raster comparison can test whether amplitude-density representation contributes to the remaining discrepancy. It should remain a mechanism diagnostic with fixed peaks, locations, bounds, and estimator settings. Agreement with this one provisional reference is insufficient for adoption. No further filtering sweep or wider tracking is warranted by this calibration alone.

Only unit154 near 620 µm is represented. Unit80 failed sensitivity qualification; neither 220 µm nor 410 µm is validated here. The 4250–4260 s interval still has only three and five unique-shift events per five seconds and remains a gap. No result justifies extrapolation to the rest of the session.

Artifacts: `testing/outputs/luke_shallow_fractional_calibration_v3/` stores frozen injection decisions, exact-geometry checks and individual-event sensitivity; `testing/outputs/luke_shallow_template_centroid_audit_v4/` stores paired median templates, conditional bootstrap samples, real template measurements and the inspected sensitivity figure. Both managed jobs completed with exit 0. Scripts and source hashes are preserved independently; v3 evidence was not overwritten.
