# Common-event low-pass diagnostic: shallow registration is sensitive to both input changes

The filtered input's poorer aggregate lighthouse agreement persists when exact event membership is fixed. Both its amplitude weights and its localized depths contribute to the shallow discrepancy. The deterioration is concentrated in unit80, rather than a uniform failure across probe depth. This identifies an input-to-registration sensitivity; it does not identify physical motion truth or establish additive causal contributions.

## Controlled comparison

The completed cached-array run crossed broadband/low-pass amplitudes and localizations for 316,089 identical `(sample_index, channel_index)` detections over4160–4260s, with frozen DREDGE settings and enforced±80µm pairwise bounds. The exact intersection contains24.32% of broadband detections and36.43% of adjusted-threshold low-pass detections. No voltage was read and no detections or localizations were rerun. The run summary records33.64s elapsed and unchanged source hashes. A single compute thread preserves the replay backend used for the source fields.

| Same-event input | Overall difference,µm | Drop/recovery difference,µm | Unit80 overall,µm | Other eight overall,µm |
|---|---:|---:|---:|---:|
| Broadband amplitude / broadband localization |1.243|1.537|3.415|0.972|
| Low-pass amplitude / broadband localization |1.481|1.607|5.989|0.917|
| Broadband amplitude / low-pass localization |1.532|1.846|5.680|1.013|
| Low-pass amplitude / low-pass localization |1.934|1.832|9.392|1.002|
| Full broadband population, contextual |1.073|1.270|2.272|0.923|
| Full low-pass population, contextual |1.943|1.826|9.966|0.940|

The first four rows form the crossed comparison. Last two rows change membership and are contextual controls. All differences are descriptive deviations from provisional lighthouse centroids, averaged equally across cells. The two single-input changes worsen the aggregate overall measure, but their combination is nonlinear: the overall increment is0.691µm versus0.526µm for the sum of the two single-input increments. The movement metric has a different interaction. Do not interpret these as an additive decomposition of physiological error.

Unit80 explains most of the aggregate deterioration. Amplitude replacement actually improves overall agreement for six of nine units; localization replacement improves five. Removing unit80 from the summary is diagnostic, not grounds to hide its failure. Unit154 also worsens from2.490 to2.916µm with both replacements; several deeper units improve. At unit80, event-matched relative estimates at4235s are+6.46µm for broadband/broadband,+13.14µm for amplitude-only,+11.51µm for localization-only and+24.57µm for both. The full low-pass estimate is+26.03µm. At4255s the same-event estimates are+5.51,+13.97,+10.58 and+24.02µm, respectively. These are waveform-reference comparison predictions, not independently verified displacements.

## Paired event changes

Compute `dy = lowpass_y - broadband_y` and `ratio = abs(lowpass_amplitude / broadband_amplitude)` at the exact shared frames/channels. Depth groups below use detector contact depth, avoiding assignment to groups using the changed localization itself.

| Detector depth,µm | Events | dy5th / median /95th percentile,µm | Median amplitude ratio | Fraction abs(dy)>20µm |
|---|---:|---|---:|---:|
|0–480|56,796|−18.10 /−0.08 /17.38|0.732|8.25%|
|480–960|36,534|−16.26 /0.05 /18.74|0.766|8.23%|
|960–1440|20,967|−19.19 /−0.07 /18.45|0.783|8.86%|
|1440–1920|33,995|−18.73 /0.09 /17.83|0.766|8.66%|
|1920–2400|52,741|−17.47 /0.23 /18.22|0.719|8.23%|
|2400–2880|35,453|−17.87 /0.15 /18.33|0.764|8.57%|
|2880–3360|39,366|−14.19 /−0.03 /13.27|0.793|5.79%|
|3360–3840|40,237|−12.29 /0.14 /14.20|0.812|5.56%|

Across all events, median dy is+0.058µm; the interquartile interval is−3.49 to+3.73µm and the5–95% interval is−16.86 to+17.15µm. Forty percent of detections change localization by more than5µm;7.72% change by more than20µm. Only674events,0.213%, exceed80µm, contributing approximately0.15% of summed absolute amplitude in either branch. The largest paired difference is200.41µm. Large localization outliers exist, but the change is distributed across many events, with both signs; it is not simply a shared depth translation or only a handful of extreme values. Their exact contribution to registration is not proven by their frequency or amplitude mass.

Overall amplitude ratios have median0.761 and5–95% interval0.560–0.970. Total absolute amplitude mass falls to74.48% of broadband on these same events. Attenuation varies with depth and time, so it reweights population structure rather than merely multiplying the whole raster by one constant.

In0–480µm, every five-second paired-localization median lies between−0.328 and+0.209µm. During4225–4240s, when the later shallow estimates diverge, these medians remain approximately0.002,+0.209,+0.022µm. Their5–95% intervals narrow to roughly−9.4..11.2,−8.5..8.9 and−10.3..8.0µm. Simultaneously, median amplitude ratios are0.603,0.611 and0.667, compared with approximately0.73 in many earlier bins; exact-common counts drop to447,540 and619. This is a change in the selected population and its weighting, not evidence of a20µm common localization shift. Because matching is exact, count changes here cannot be equated with firing-rate changes.

## Interpretation and limits

The main supported mechanism is altered spatial/amplitude structure feeding a sensitive shallow registration, with amplitude reweighting and localization changes each sufficient to worsen the unit80 comparison on fixed exact events. A near-zero global or time-bin localization median does not preserve the detailed depth histogram or its competing correlations. The saved arrays do not isolate the specific waveform families or correlation constraints responsible, and no outlier-removal experiment was performed in this audit.

The near equality of the same-event both-low-pass score and the full-low-pass score shows the discrepancy survives substantial membership restriction; it does not establish membership is irrelevant. Low-pass timing or detector-channel shifts discard related biological events from exact matching. The retained subset is therefore biased, and full broadband versus subset broadband already differs. Fixed-template lighthouse centroids retain their known identity and displacement-sensitivity limitations. No production configuration or full-sort decision is validated.

## Artifacts

- Script: `testing/luke_lowpass_common_events_v2.py`; outputs under `testing/outputs/luke_lowpass_common_events_v2/`.
- Newly computed cached-only audit tables: `paired_event_distribution_audit.csv` (all/depth/five-second shallow distributions) and `paired_per_unit_audit.csv` (six-arm cell metrics). Quantiles are unweighted event quantiles; tail amplitude-mass fractions were computed from absolute cached amplitudes.
- Existing `common_event_ids.npz`, crossed peak/location arrays, fields, `scores.csv`, event-matched predictions, settings, hashes and timings preserve the controlled inputs and outputs.

The managed common-event experiment ran four DREDGE estimates from cached arrays. The subsequent independent distribution review only read those arrays and wrote diagnostic tables/report; it did not run additional estimates. Neither stage accessed voltage, retuned estimator parameters or launched a sort.
