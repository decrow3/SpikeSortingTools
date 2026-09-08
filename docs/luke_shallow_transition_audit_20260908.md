# Qualified unit154 transition audit: coverage narrows after 4250 s

**The frozen expanded matcher corroborates a compact unit154 population through 4240–4250 s, but does not recover enough support after 4250 s.** Completed fractional controls show that the original individual-event energy statistic compresses displacement; the centroid of the median waveform changes −5.770 µm instead. Neither statistic establishes that DREDGE overshoots physical motion.

The audit used only compensated broadband voltage, the previously qualified unit154 bank and weights, and independent 5σ both-sign detections within ±120 µm of 620 µm. Matching did not receive DREDGE fields. All detection decisions, full score matrices, accepted events, and a matching-complete receipt were saved before motion files were opened. Unit80 and other unqualified cells were excluded.

Of 3,008 detections, 83 events passed both identity and unique-shift margins. There were no <1 ms accepted-event intervals or duplicate aligned frames. Eighty-one accepted events overlapped the original fixed matcher within 0.5 ms; only two were new relative to that matcher, which had accepted 139 events in this interval. The new matches occurred late and did not restore adequate coverage; one won the −80 µm search boundary. Boundary matches remain diagnostic rather than a reason to extend the search automatically.

| Interval (s) | Unique-shift events | Dominant shift fraction | Location summary |
|---|---:|---:|---|
| 4240–4245 | 49 | 1.00 | Supported baseline |
| 4245–4250 | 26 | 1.00 | Supported descriptive comparison |
| 4250–4255 | 3 | 0.33 | Gap: sparse, mixed shifts |
| 4255–4260 | 5 | 1.00 | Gap: sparse |

The first two bins' winning coarse shifts are zero. A zero winner on a 40 µm grid **does not establish zero movement at the 5–20 µm scale**. Five-second summaries require at least ten unique-shift events and ≥80% at one discrete shift. Mixed/sparse bins remain gaps, while all event-level hypotheses remain available; this rule does not establish that mixed hypotheses are artifacts rather than genuine motion within a bin.

On the same accepted frames, relative to each measure's own first-five-second baseline, 4245–4250 s gives:

| Measure | Relative displacement (µm) |
|---|---:|
| Compensated 3σ DREDGE | −10.312 |
| Low-pass adjusted-threshold DREDGE | −11.784 |
| Median individual-event squared-energy centroid | −1.225 |

The centroid is measured on raw event waveforms within the footprint implied by the winning shift. It contains background energy and is not a calibrated monopolar position estimate. Existing qualification only tested exact 40 µm translations. Consequently, this difference cannot be called a 9–11 µm motion-estimation error. The completed frozen-matcher fractional controls demonstrated strong compression in this individual-event statistic. Replacing it with the centroid of the median waveform on the same real events gives 619.329→613.560 µm, a −5.770 µm change. Under the approximate injection model at gain 1, this statistic recovers −9.090/+6.916 µm for requested −10/+10 µm. The clean interpolated waveform itself has asymmetric centroid shifts (−9.455/+7.244 µm), so this is not a physical accuracy calibration. No confidence interval was computed for the real median-waveform change. The remaining difference from DREDGE is diagnostic, not a quantified physical error; no global gain correction is justified. See [completed fractional calibration](luke_shallow_fractional_calibration_20260908.md).

This observation is local to unit154 near 620 µm. It does not validate motion at 220 µm or 410 µm, establish physical motion for unmatched events, or support extrapolation to the full session. The post-4250 gap remains unresolved.

Artifacts: `testing/outputs/luke_shallow_transition_audit_v3/` contains the 1 s/5 s support tables, time/shift waveform contact sheet, event-level scores and centroids, and post-selection motion comparison. The managed run completed in 24.43 s with exit 0. Script: `testing/luke_shallow_transition_audit_v3.py`.
