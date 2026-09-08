# Shallow failure: strict search bounds and waveform evidence

## Controlled bound replay

Using identical cached peaks, localizations and solver settings, restricted correlation argmax candidates to the already-requested ±80 µm. Before restriction, reconstructed curves reproduced every saved D/C matrix across all 20 windows in both arms. All restricted constraints lie within ±80 µm and resulting fields are finite.

15/8000 original and 63/8000 compensated constraints changed. The shallow result persists: compensated displacement at 220 µm is 27.337 µm and at 620 µm changes from 26.213 to 26.043 µm. At 2380 µm the compensated change moves from 1.155 to 1.494 µm. Thus the search-domain discrepancy is real but is not the main cause of the shallow failure.

The independent `luke-dredge-strict-bounds-v1` service completed with zero exit status. Commands, settings, logs and receipt are saved. Script: `testing/luke_dredge_strict_bounds.py`; artifacts: `testing/outputs/luke_dredge_strict_bounds_v1/`.

## Waveform review

Reviewed top eight detection-channel cohorts by aggregate compensated amplitude mass at localized depths 0–900 µm in 4246–4247 and 4254–4255 s. These seconds correspond to the largest shallow pairwise mismatch. Selection does not use lighthouse agreement. Reconstructed voltage uses the frozen shared-response model, unchanged filter and physical scaling.

The median waveforms show plausible neural trough/recovery structure. This is insufficient for unit identity, but does not justify labeling the dominant cohorts non-neural. Channel cohorts can mix multiple cells. Median amplitudes are roughly 54–164 µV, with substantial variability in single-event waveform similarity.

Counts change unevenly across depth:

| Channel / depth | Earlier events | Later events |
|---|---:|---:|
| 9 / 80 µm | 93 | 14 |
| 10 / 100 µm | 45 | 10 |
| 18 / 180 µm | 15 | 39 |
| 17 / 160 µm | 24 | 25 |
| 15 / 140 µm | 27 | 27 |
| 3 / 20 µm | 26 | 0 |
| 13 / 120 µm | 38 | 4 |
| 62 / 620 µm | 42 | 4 |

Later cohorts with four events provide weak shape evidence. Probe-edge channel 3 cannot establish complete spatial support. The plot shows median and interquartile range at the detection channel; spatial median waveforms are also saved for subsequent identity review.

These findings support investigating population-activity changes as a source of broad erroneous registration, rather than expanding artifact rejection without evidence. Counts alone cannot distinguish firing-rate changes, changing unit mixture, detection sensitivity or actual movement. The stationary central artifact explanation remains separately supported.

## Next step and remaining scope

Preserve the successful compensation and strict requested bounds in diagnostic comparisons. Next examine whether broad population-rate structure dominates the shallow correlation, using the same peaks and a controlled profile representation comparison, with no lighthouse-driven tuning. Independently corroborate the waveform cohorts across time before claiming identity or excluding them.

Scripts: `testing/luke_shallow_waveform_review.py`. Outputs: `testing/outputs/luke_shallow_waveform_review_v1/`, including channel contribution counts, waveform metrics, spatial medians and `01_shallow_waveforms.png` / `.pdf`. Both this figure and the strict-bound comparison were visually reviewed. The waveform review completed with exit code zero. No classification or exclusion was applied, no production library was changed, and no sort was launched.

Full-recording transfer, difficult-motion reliability and downstream sorting/amplitude-completeness benefit remain unverified.
