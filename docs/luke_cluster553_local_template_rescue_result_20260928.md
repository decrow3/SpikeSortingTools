# Cluster 553 local-template rescue result

Date: 2026-09-28. Status: **complete; candidate rejected.**

## Decision

Reject the frozen target-local 3-sigma peak proposal plus waveform-template
gate. It cannot preserve known healthy cluster-553 spikes and recovers none of
the independently anchored failing events. Do not tune this arm, remove one of
its gates after seeing the result, or promote it to a sort.

The negative result changes candidate ordering. A median missed-event waveform
can establish population-level recognizable structure, but it does not show
that individual events are separable from background. Future detector work must
pass a calibration-only single-event discrimination check before scanning a
continuous interval.

## Frozen operation and result

The operation used curated cluster 553 retained spikes from 5802.13--5954.24 s
to build a 16-channel median template. It proposed locally deduplicated negative
peaks above 3 per-channel noise sigma and required both cosine >=0.80 and a
matched-filter score above the larger of the target 10th percentile and shifted
background 99.9th percentile. Legacy cluster 425 was evaluation-only.

| measure | reference/calibration | failing evaluation |
|---|---:|---:|
| retained target spikes | 1,999 | 1,999 |
| 3-sigma proposals | 38,257 | 99,918 |
| proposal coverage of retained target | 21.16% | 21.01% |
| accepted proposals | 0 | 0 |
| accepted coverage of retained target | 0% | 0% |
| recovery of previously local-`full_st`-unmatched anchor | -- | 0% |

The separate calibration audit explains why removing only the proposal stage
would not rescue the arm. At the frozen score threshold, target and shifted
background acceptance are both 0.1001%. Their score distributions are nearly
identical: target/background medians are 12.18/8.70, 99th percentiles are
108.32/108.24, and 99.9th percentiles are 118.58/119.12. No individual waveform
passes cosine 0.80; the target median cosine is 0.487 and the background median
is 0.381. The population median template itself is coherent at 110.2 uV PTP,
so this is a single-event separability failure rather than an absent average
waveform.

## Integrity checks

The clocks and label lineage are aligned. In the reference window, 1,879 of
1,999 curated cluster-553 events match the 2,215 legacy-anchor events within
0.5 ms. Raw cluster 577 contributes 1,999 matched events to final cluster 553.
Adding or subtracting the recording manifest's `t_start` destroys alignment.

The result used config
[`luke_cluster553_local_template_rescue.v1.json`](../configs/luke_cluster553_local_template_rescue.v1.json),
SHA-256 `709c4a13cd04a5be78778c56f2218d5217faa385421d68ac757dded87a604aaa`.
The executable is
[`luke_cluster553_local_template_rescue.py`](../testing/luke_cluster553_local_template_rescue.py),
and the calibration-only audit is
[`luke_cluster553_template_calibration_audit.py`](../testing/luke_cluster553_template_calibration_audit.py).

No sort was launched. No existing spike, label, threshold, production output,
or raw recording was changed.
