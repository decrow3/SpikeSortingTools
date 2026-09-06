# Amplitude-fit window feasibility — retain 1,000 spikes

Date: 2026-09-06. Status: **calibration and baseline-only feasibility; no candidate reanalysis.**

## Recommendation

**Retain the existing 1,000-spike production endpoint.** None of the smaller
windows supports a defensible short-window candidate comparison at the proposed
5-percentage-point effect. The original Option A result remains unchanged and
inconclusive.

The smaller windows are useful as exploratory diagnostics, but they trade
availability for sampling error and origin sensitivity. They must not be used
to turn the closed Option A result into confirmation.

## Method

`pipeline.truncation.fit_amp_cdf` and its existing optimizer/bounds were reused
unchanged. Windows were explicit non-overlapping half-open slices consuming
exactly 250, 500, or 1,000 amplitudes. Historical production behavior remains
separate: its nominal 1,000 window fits 999 values through the legacy slice.

Synthetic calibration used the existing logistic/truncated distribution,
missingness 0.5–40%, fixed seed `20260906`, and 200 repetitions per truth and
window size. This is calibration under the fitter's assumptions, not validation
on real neurons. A fixed mixed-scale population was also tested as a model-
mismatch check.

## Synthetic calibration

| window | mean bias (pp) | mean absolute error (pp) | spread SD (pp) | fit failures | boundary-pinned |
|---:|---:|---:|---:|---:|---:|
| 250 | 0.654 | 3.043 | 3.774 | 0% | 3.2% |
| 500 | 0.276 | 1.926 | 2.397 | 0% | 1.6% |
| 1,000 | 0.120 | 1.436 | 1.761 | 0% | 0.2% |

For independent paired synthetic samples, the fraction exceeding the proposed
5-point improvement threshold was:

| window | unchanged data, false improvement | genuine 5-point improvement detected |
|---:|---:|---:|
| 250 | 10.0% | 47.0% |
| 500 | 4.0% | 50.5% |
| 1,000 | 1.5% | 55.5% |

The mixed-scale model-mismatch check produced approximately **40 pp bias for
all three sizes**. That is expected evidence that this fitter cannot validate
pooled incompatible amplitude populations; it is not evidence for a real-unit
window choice.

## Baseline-only feasibility

The fixed 124-unit baseline cohort from the completed feasibility screen was
used. No corrected-arm output was read. Each unit remained in the denominator;
too few spikes and insufficient finite fits were not discarded.

In the nominated [7200, 7320] s interval:

| window | measurable units | coverage | origin-shift paired units | median / P90 shift difference (pp) |
|---:|---:|---:|---:|---:|
| 250 | 19/124 | 15.3% | 15 | 0.55 / 7.31 |
| 500 | 11/124 | 8.9% | 10 | 1.54 / 5.40 |
| 1,000 | 4/124 | 3.2% | 1 | 1.76 / 1.76 |

Healthy intervals showed the same pattern: 250 measured 15–28 units, 500
measured 4–12, and 1,000 measured 1–2. These are availability changes, not
independent-neuron evidence or reliability gains. Non-overlapping windows from
one unit were not treated as independent neurons. Boundary-pinned fits were not
counted as measurements in this screen.

## Protocol consequence

No smaller-window evaluation protocol is authorized for candidate results. The
existing 1,000-spike protocol remains the primary endpoint and its Option A
result remains **inconclusive** at 2/53 measurable paired units. The 250- and
500-spike outputs are calibration and baseline planning evidence only; they are
not held-out confirmation and do not justify another sort, a threshold change,
or a candidate rerun.

Study outputs are in `docs/outputs/luke_truncation_window_feasibility_v1/`.
Production QC and the closed candidate branches were unchanged.