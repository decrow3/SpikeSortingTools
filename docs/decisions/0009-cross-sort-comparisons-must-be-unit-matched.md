# 0009 — Cross-sort quality comparisons must be unit-matched

**Status:** Adopted 2026-09-02
**Corrects the central empirical claim of:** [0008](0008-amplitude-completeness-gates-promotion.md)
**Evidence:** [`luke_20250804_truncation_fitter_audit.md`](../luke_20250804_truncation_fitter_audit.md)

## Decision

Any comparison of sorting-quality metrics **between configurations** must be
made on **matched units** — the same neurons identified in both sorts — or
explicitly stratified on the unit property the metric depends on. A comparison
of whole-population medians across sorts that admit different unit populations
is not evidence about detection quality.

Concretely, before any cross-sort quality claim:

1. Match units across the sorts (spike-time coincidence is sufficient and cheap;
   `testing/luke_truncation_matched_units.py` implements it).
2. Report the paired difference on matched units as the primary result.
3. Report the unmatched units separately, as a statement about *yield*, not
   about *quality*.

## What was wrong

[0008](0008-amplitude-completeness-gates-promotion.md) concluded that rescue
units are "typically less completely detected" than legacy and claim-mask units,
on the strength of population medians of estimated missing-spike fraction:
3.07% / 1.16% / 0.82% in the >1 Hz good-unit cohort.

Those numbers are arithmetically correct — they reproduce exactly. The
inference does not hold.

Matched on the same neurons, the configurations are indistinguishable:

| Matched pair | n | A | B | Paired Δ | p |
|---|---:|---:|---:|---:|---:|
| rescue vs legacy | 43 | 0.63% | 0.63% | −0.02 pp | 0.797 |
| rescue vs claim-mask | 47 | 0.75% | 0.73% | +0.03 pp | 0.193 |
| legacy vs claim-mask | 42 | 0.60% | 0.76% | +0.01 pp | 1.000 |

Robust across coincidence tolerances 0.25–1.0 ms and match thresholds 0.3–0.7.

The population statistic decomposes into shared and unique units:

- **rescue** = 43 shared @ 0.63% + **50 unique @ 9.45%** → 3.07%
- **legacy** = 43 shared @ 0.63% + 25 unique @ 3.01% → 1.16%

Estimated missingness depends strongly on unit amplitude (within-method
Spearman −0.44 rescue, −0.56 legacy). Rescue recovers more small-amplitude
units; small units sit closer to the detection floor in **every** pipeline —
legacy's own unique units show the same effect. Reweighted to a common amplitude
mix the ordering reverses, legacy 6.90 against rescue 5.72.

Stratifying by firing rate, as the evaluation did, does not control for this:
rate and amplitude are largely independent.

## What survives from 0008

- **Amplitude completeness is a real and missing acceptance dimension.** The
  frozen gate set does not measure per-unit detection completeness at all. That
  gap is genuine and item 1 of 0008's follow-up plan stands — but the gate must
  be specified on matched or amplitude-stratified comparisons. A
  population-median gate would penalize exactly the behaviour we want, namely
  recovering additional small units.
- **Yield alone must not gate promotion**, and contamination/refractory
  improvements do not establish completeness. Both hold.
- **`reject_universal_default` stands** — on its original grounds, the gates
  that actually failed: 27 similar good–good pairs against 8 and 11, and an
  edge-spike fraction of 2.004% pre-curation and ~2.066% post-curation against a
  2.000% threshold. It does **not** stand on a completeness deficit.

## What is now retracted

The claim that the rescue configuration detects units less completely than the
comparators. There is no evidence for it, and the matched comparison is
evidence against it.

The open question is different and narrower: **are rescue's 50 additional units
genuine neurons, fragments, or noise?** Truncation cannot answer that — high
missingness for a small unit is expected everywhere. It needs waveform,
refractory, CCG and spatial evidence on those specific units.

## The estimator itself is sound where it is used

Validated against synthetic ground truth
(`testing/test_truncation_fitter.py`): unbiased to under 0.5 pp for true missing
fractions from 0.5% to ~40%. All KS-good cohorts sit at 0.6–3%.

It is **hard-censored at 50%**: the bound `x0 >= x_min` makes the reported
statistic incapable of exceeding 50, so true 70% missing is reported as 50.0. A
window at exactly 50.0 is boundary-pinned and must be filtered
(`pipeline.truncation.is_saturated`) rather than averaged in. This affects
54.9% / 56.3% / 16.4% of *all* windows but ~0% of KS-good windows, so it is not
the source of the reported difference.

## Reopening conditions

This record is about method, not about one result. It applies to every future
cross-configuration comparison — sorter bake-offs included, where the same
confound would inflate or deflate any per-unit quality metric.

## Evidence pointers

- `docs/luke_20250804_truncation_fitter_audit.md`
- `testing/luke_truncation_fitter_audit.py`, `testing/luke_truncation_matched_units.py`
- `testing/test_truncation_fitter.py`
