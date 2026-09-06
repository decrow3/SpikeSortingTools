# Baseline recovery census: alternatives to cluster 37 exist

The available baseline artifacts contain other measurable deterioration cases.
The earlier failure to nominate a case beyond cluster 37 does not establish a
population-wide impasse. This census is exploratory case discovery, not evidence
that a replacement pipeline works.

## Scope and method

Searched all 1,273 inventoried clusters (710 rescue, 563 legacy), including explicit
no-window records; 529 have sorter label `good`. Only the saved permitted
development intervals were eligible. Rescue cluster 37 was excluded. No sorting,
candidate-arm analysis, threshold change, or fitter execution was performed.

Two searches are exported separately:

1. The original four-consecutive-window rule, with its nomination cap removed:
   two windows each at most 5% missing, then two each at least 15%, median increase
   at least 10 percentage points, total span at most 600 seconds. Existing
   validity, 1,000-spike nominal count and contiguity checks remain.
2. An exploratory search allowing intervening windows between those two valid
   pairs, within one permitted development interval. It selects the shortest
   qualifying span per cluster and explicitly reports intervening windows. It
   does not fit across gaps or pool amplitudes.

The original rule finds 19 runs in 17 rescue clusters beyond 37, including four
`good` clusters (21, 36, 452, 553), and nine runs in eight legacy clusters,
including one `good` cluster (424). The broader search finds 34 rescue and 30
legacy clusters, including seven and four labelled `good`, respectively.

These are opportunities for investigation. Sorter labels and contamination
estimates do not establish neuronal identity or validate a truncation model.

## Three concrete alternatives

All three below meet the original abrupt rule. Missing percentages are medians
of two stored QC fits per phase. Historical fits use 999 amplitudes from nominal
1,000-spike windows; these values have not yet been replayed with exact indexing.

| Rescue cluster | Recording interval (s) | Missing estimate, reference → failing | Reported ContamPct | Measured short-ISI fraction, reference → failing |
| --- | --- | --- | --- | --- |
| 553 | 5802.134–6360.328 | 2.76% → 34.34% | 0.7% | 0% → 0% |
| 452 | 3869.576–4198.765 | 2.15% → 21.19% | 1.3% | 0.050% → 0.150% |
| 36 | 4085.567–4365.883 | 3.03% → 16.43% | 1.6% | 0.050% → 0.050% |

The train checks read 2,000 retained spikes per phase and calculate the fraction
of adjacent intervals shorter than 1.5 ms. This is an empirical ISI fraction,
not an estimate of contamination. Amplitudes are from `full_st[kept_spikes]`,
with spike-time alignment checked against the curated train.

Median detected-spike depths change by −1.23, −0.39 and −3.02 µm, respectively.
Median sorter amplitudes fall by 8.1% and 15.2% in 553 and 452, but **rise by
11.4% in 36**. Thus 36 needs particular scrutiny for shape/model effects.
These depth summaries do not establish or rule out motion as a cause; they
summarize detected events, not independent motion ground truth.

Source voltage and template files exist for the rescue cases. Their existence
has been checked; raw waveforms have not been inspected in this task.

## Recommended next decision

Prioritize **553**, with **452** as the independent follow-up case. They combine
large estimated deterioration, low measured short-ISI fractions, four existing
fits in under ten minutes, and available source voltage. Keep 36 as a diagnostic
alternative because its amplitude direction is different. The overlapping
452/36 interval spans about 496 seconds, but proximity alone should not make
them a pooled endpoint or imply a common mechanism.

Give the next coding task one bounded question: **does the apparent loss in 553
survive exact replay and waveform inspection, and what intervention does that
evidence support?** Reuse existing replay and plotting functions; do not build
another general audit framework.

1. Save a new case-specific input record referencing these baseline source rows
   and recording coordinates. Preserve the previous frozen selection and results.
2. Replay its four windows with historical and exact indexing. Plot the individual
   amplitude distributions and fits, with temporal coverage visible. Reject a
   mechanistic interpretation if the estimate relies on incompatible populations
   or an unstable fit. Do not infer recovered spikes from a fitted tail alone.
3. Inspect bounded, deterministic raw waveform samples in both phases on common
   physical channels, together with the existing train/depth evidence. Record
   whether the observations support detection loss, assignment changes, mixture,
   or leave the cause unresolved. Cached whole-cluster templates alone cannot
   answer a time-local waveform question.
4. Produce one intervention recommendation with a concrete predicted observable
   change, or an explicit statement that this case does not support one. Only
   then write a new comparison contract. Do not automatically revive the closed
   motion or linker candidates. A second case is a bounded fallback, not a sweep.

There is a way forward at the **case-selection and measurement** stage. This
does not yet solve candidate-arm correspondence, prospective fit availability,
or population coverage. Two valid baseline fits per phase do not guarantee
that a newly sorted candidate will produce an evaluable endpoint. A successful
case repair would establish a mechanism worth testing more broadly, not a
step-change improvement of the full pipeline.

## Reproduction and artifacts

Run from the repository root using the rescue-production environment:

```sh
python -m testing.luke_baseline_recovery_census
python -m testing.luke_baseline_recovery_census --inspect-shortlist
python -m pytest -q testing/test_luke_baseline_recovery_census.py
```

Outputs are in `docs/outputs/luke_baseline_recovery_census_v1/`:
`cluster_census.csv` (one status per cluster), `uncapped_original_rule_cases.csv`,
`candidate_cases.csv`, `summary.json`, and `shortlist_evidence.json`.
The summary records the attested inventory hash, selection hash, frozen constants,
script hash and label-file hashes. Shortlist hashes cover the selected cluster
times, full-st rows and depths read; they are not full source-file attestations.
This is a descriptive baseline report, not a production qualification receipt.
