# Full-recording reference amplitude truncation

This replaces the 120-second truncation analysis. The short anchors were too
brief for the 1,000-spike requirement in most clusters; their fit coverage and
conditional missingness medians are withdrawn as evidence for the motion
correction decision. Short-window counts and waveform plots retain their
separate descriptive uses.

## Method

Use every retained spike in the verified 384-channel, full-recording 12/9
motion-off reference. Preserve all 710 curated clusters, with the 301 KS-good
clusters reported as a stratum rather than an eligibility filter.

For each cluster, split its full train at inter-event gaps greater than 10
seconds, following production QC's continuity rule. Within each resulting
block, center as many disjoint 1,000-spike windows as fit. Fit all 1,000
amplitudes in each window. Store first/last sample boundaries, physical duration,
rate, numerical fit status and missingness. The window duration is determined
by the events, with no 120-second restriction.

Every event belongs either to one fit-input window or to an explicitly recorded
residual interval. No leftover spikes disappear from accounting. Report gaps
greater than 10 seconds, including leading/trailing silence, separately from
amplitude fits. Full-session 30-second counts retain zeros; those bins are not
amplitude-fit windows. Report per-cluster fit-time coverage and actual window
duration distributions alongside conditional fit results.

A successful fit at the 50% model boundary is recorded as ceiling-limited, not
as a reliable 50% measurement. Numerical failures are explicit rather than
accepted as the legacy fitter's fallback. Fit summaries give each eligible
cluster equal weight by first calculating its median across valid windows.
Counts of clusters without valid windows remain visible.

The production implementation stores inclusive last-event indices but slices
amplitudes with an exclusive upper bound: its nominal 1,000-event windows
actually fit 999 amplitudes. This replacement fits exactly 1,000; that difference
is intentional and source QC files are unchanged. Therefore numerical equality
with cached production missingness is not expected. The fit-window event spans
and count can still be checked against its saved window indices.

## Interpretation and subsequent comparison

This is a motion-off reference characterization, not evidence of correction
benefit. Amplitude fits remain conditional on detected spikes. A long window
may mix waveform states; actual durations and time-resolved results must be
inspected. An inter-event gap does not establish missing biological spikes.
Successful optimization does not validate the assumed amplitude distribution.

The exported reference window boundaries are frozen for a future paired
comparison. Once cross-sort unit families are validated, evaluate both arms
in the same physical intervals. Do not independently resize each arm's windows
and compare unmatched medians. If the corrected arm has insufficient events in
an interval, report that fact and the rate/gap evidence separately. Unit-family
matching requires spatial/raw-waveform support and exclusive event assignments.

## Provenance

Reference identity:
`22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`.
Runner: `testing/luke_full_reference_truncation.py`.
Outputs: `testing/outputs/luke_full_reference_truncation_v1/`.
Independent service: `luke-full-reference-truncation-v1.service`.
Exact launch, stdout/stderr, and final process receipt are saved in
`testing/outputs/luke_full_reference_truncation_v1.*`.
No sort or raw-recording scan is launched. There is no within-job checkpoint;
a justified restart would use a new output directory and preserve the failed one.

## Completed results

| Measure | All clusters | KS-good clusters |
|---|---:|---:|
| Clusters | 710 | 301 |
| Clusters with at least one valid fit | 287 | 106 |
| Clusters without a valid fit | 423 | 195 |
| Spikes entering fit windows | 26,227,000 | 4,075,000 |
| Explicit residual spikes | 3,000,829 | 1,296,595 |
| Median of per-cluster valid-fit medians | 7.49% | 2.23% |

All clusters: 26,227 windows; median duration 52.8 s, P5–P95 9.8–191.5 s. Status counts: `{'fit_at_50pct_ceiling': 14402, 'fit_valid': 11825}`. Median per-cluster fit-span coverage of the recording: 0.0%.

KS-good: 4,075 windows; median duration 74.1 s, P5–P95 8.0–294.9 s. Status counts: `{'fit_valid': 3737, 'fit_at_50pct_ceiling': 338}`. Median per-cluster fit-span coverage of the recording: 0.0%.

These are full-recording reference measurements. The successful-fit medians are conditional summaries, not an estimate of all lost spikes and not a motion-off versus corrected comparison.

Validation passed: Every saved fit has exactly 1000 input events; All 29,227,829 retained spikes accounted for as fit input plus residuals; Full-session count bins independently reconcile to retained total; Residual table reconciles per unit; All 710 per-unit adaptive window counts equal cached production QC; Selected four units have exact physical boundary agreement with cached production window indices. Partition edge tests also passed for empty trains, exact 1,000-event boundaries, residuals and gaps.

For KS-good clusters without valid fits, 191 never form a 1,000-event continuous-block window over the full recording; 4 form windows but have no uncensored successful fit. This distinction is recorded in `units.csv`; neither category is treated as zero missingness.
