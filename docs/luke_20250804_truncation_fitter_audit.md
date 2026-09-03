# Truncation estimator audit and unit-matched re-analysis (Luke0804 imec0)

> **PARTIAL RETRACTION — 2026-09-03.** The synthetic logistic-model fitter
> characterization and the population-composition confound remain valid. The
> 43/47/42 matched-unit counts, paired estimates, tolerance robustness, and
> reweighted empirical result are withdrawn because the matcher reused target
> events. Rerun `luke_truncation_matched_units.py` to create
> `matched_units_v2.csv` before citing a cross-sort numerical result.

**Date:** 2026-09-02
**Scope:** follow-up items 2 and 3 of
[`decisions/0008`](decisions/0008-amplitude-completeness-gates-promotion.md),
which gate item 1 (formalizing an amplitude-completeness acceptance gate).
**Headline:** the reported rescue completeness deficit is a **unit-composition
artifact**. On matched neurons the three configurations are indistinguishable.

Reproduce with:

```bash
python testing/luke_truncation_fitter_audit.py     # estimator + cohort reproduction
python testing/luke_truncation_matched_units.py    # cross-sort unit matching
pytest testing/test_truncation_fitter.py           # estimator validation
```

Outputs are written to `testing/outputs/luke_truncation_fitter_audit/`
(untracked, local disk). Nothing was written under `/mnt`.

## 1. The reported numbers are arithmetically correct

Every headline figure in
[`luke_20250804_imec0_postcuration_evaluation.md`](luke_20250804_imec0_postcuration_evaluation.md)
reproduces exactly from the stored `truncation_qc.npz` files and curated
outputs: 2.97 / 1.04 / 0.89 for all eligible good units, 3.07 / 1.16 / 0.82 for
good units >1 Hz, 68.8 / 77.9 / 91.8 percent below the 10% threshold, and
eligibility of 110/301, 78/228, 72/191.

The error is not in the computation. It is in what the comparison identifies.

## 2. Estimator validation

Synthetic ground truth: amplitudes drawn from a logistic distribution, hard
truncated so a known fraction lies below the cut, 1000 retained spikes per
window. Codified in `testing/test_truncation_fitter.py`.

| True missing | Reported (median) | Bias | Windows pinned at 50.0 |
|---:|---:|---:|---:|
| 0.5% | 0.59% | +0.09 | 0% |
| 2% | 2.13% | +0.13 | 0% |
| 10% | 10.32% | +0.32 | 0% |
| 30% | 30.01% | +0.01 | 0% |
| 40% | 40.46% | +0.46 | 6% |
| 60% | 50.00% | **−10.00** | 90% |
| 70% | 50.00% | **−20.00** | 99% |

**The estimator is sound where it is being used.** It is unbiased to well under
a percentage point from 0.5% to about 40%. Every KS-good cohort measured sits at
0.6–3%, comfortably inside that range.

### The 50% ceiling is a boundary artifact, not a measurement

`fit_truncated_sigmoid` bounds `x0` below by `x_min`, and
`truncated_sigmoid_missing_pct` returns `100 * sigmoid(x_min; x0, k)`. With
`x0 >= x_min` that expression cannot exceed 50. A window reported at exactly
50.0 is a fit whose location parameter was driven onto its lower bound — it
means "at least 50% missing", not "50% missing".

Incidence across **all** windows: rescue 54.9% (14,394/26,227), legacy 56.3%
(17,372/30,872), claim-mask 16.4% (2,513/15,336).

Incidence within the **KS-good cohorts** that the evaluation compared: median
fraction of censored windows per unit is **0** for all three configurations.

So the ceiling is real, must be filtered before any median is taken
(`pipeline.truncation.is_saturated`), and is **not** the explanation for the
reported deficit.

### Other estimator defects found

- **Silent fallback.** On a `curve_fit` exception the code substitutes
  `[mean_amp, 1, 1]`, which typically yields near-0% missing. It prints, but
  sets no flag in the returned arrays, so a failed fit reads as a perfectly
  complete unit. It did not fire in these three runs.
- **Discarded independent estimate.** For a truncated sigmoid CDF the
  renormalisation parameter satisfies `A = 1/(1-F(x_min))`, so `1 - 1/A` is a
  second estimate of the same quantity. It is fitted, stored, and never used.
  The two agree to about 2 percentage points at the median but differ by more
  than 5 points in roughly 20% of windows — a free goodness-of-fit signal.
  Exposed as `pipeline.truncation.missing_pct_from_normalisation`.
- **Off-by-one.** `analyze_amplitude_truncation` slices `[i0:i1]`, fitting 999
  of each window's 1000 spikes. Harmless at this size; pinned by test.
- **Rate-dependent eligibility.** A unit needs 1000 spikes inside a continuous
  block (gaps > `max_isi`, default 10 s) to yield any window. Roughly two thirds
  of KS-good units are therefore never estimated.
- **Scale invariance confirmed.** Missingness depends on `k*(x0 - x_min)`, so
  differing amplitude scales between sorts do not by themselves bias the
  statistic. The confound is composition, not units.

## 3. The confound: estimated missingness tracks unit amplitude

Within-method Spearman correlation of per-unit median missingness against unit
properties, good units >1 Hz:

| Method | vs KS amplitude | vs contamination | vs firing rate |
|---|---:|---:|---:|
| rescue | **−0.438** | +0.293 | −0.196 |
| legacy | **−0.559** | +0.171 | +0.036 |
| claim-mask | −0.152 | +0.087 | −0.153 |

Smaller units sit closer to the detection floor and are genuinely more
truncated. This is expected physics, not a pipeline defect.

The three configurations admit systematically different unit sizes (median KS
amplitude, good >1 Hz): claim-mask 23.2, legacy 18.1, rescue 17.2. The ordering
of reported missingness is the reverse of the ordering of unit amplitude.

Median missingness within matched absolute amplitude bins:

| KS amplitude | claim-mask | legacy | rescue |
|---|---:|---:|---:|
| <15 | 3.78 (n=7) | **18.42** (n=17) | 13.93 (n=33) |
| 15–20 | 0.73 (n=15) | 0.65 (n=21) | 1.24 (n=28) |
| 20–25 | 0.73 (n=15) | 0.42 (n=11) | 0.47 (n=15) |
| 25–30 | 0.35 (n=7) | 0.62 (n=12) | 0.86 (n=10) |
| >30 | 0.91 (n=17) | 0.40 (n=7) | 3.12 (n=7) |

Reweighting every configuration to the rescue amplitude mix moves the medians
from 0.82 / 1.16 / 3.07 (claim-mask / legacy / rescue) to **1.79 / 6.90 / 5.72**
— legacy becomes worse than rescue. The direction of the reported result is not
robust to composition adjustment.

## 4. Decisive test: the same neurons across sorts

Units matched by spike-time coincidence (mutual best match, coincident fraction
of the smaller unit ≥ 0.5), good and eligible and >1 Hz in both sorts.

| Matched pair | n neurons | A | B | Paired median Δ | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| rescue vs legacy | 43 | 0.63% | 0.63% | −0.02 pp | 0.797 |
| rescue vs claim-mask | 47 | 0.75% | 0.73% | +0.03 pp | 0.193 |
| legacy vs claim-mask | 42 | 0.60% | 0.76% | +0.01 pp | 1.000 |

Stable across coincidence tolerances of 0.25, 0.5 and 1.0 ms and match
thresholds of 0.3, 0.5 and 0.7 (n = 42–44, Δ = −0.02 to −0.03 pp, p = 0.59–0.80).

**On neurons that both pipelines find, the rescue pipeline detects them exactly
as completely as the comparators.**

### The population statistic decomposed

| | shared with the other sort | unique to this sort | reported population median |
|---|---|---|---|
| rescue | 43 units @ **0.63%**, amplitude 19.0 | **50** units @ 9.45%, amplitude 15.8 | 3.07% |
| legacy | 43 units @ **0.63%**, amplitude 17.8 | 25 units @ 3.01%, amplitude 18.4 | 1.16% |

The entire reported gap is the extra units rescue recovers. They are smaller and
therefore closer to the detection floor. Legacy's own unique units show the same
direction (3.01% against its shared 0.63%).

## 5. Motion is not the explanation for this recording

The rescue run's DREDGE rigid sidecar reports a total motion range of
**1.279 µm** over the full 10,473 s (std 0.111 µm, median per-bin change
0.0000 µm) — 6.4% of one 20 µm site pitch. Amplitude changes from drift of that
size are negligible.

Qualification: the estimate is rigid-only, its QC status is `NOT_EVALUATED`,
and `extra_summary.json` shows `weights_thresh` entirely non-finite, so the
estimate is unqualified and non-rigid motion is unmeasured. It is evidence
against the motion hypothesis for this session, not proof.

## 6. What this does and does not establish

**Establishes.** The post-curation comparison does not show a detection-quality
difference between the configurations. It shows that they admit different unit
populations. The completeness statistic is confounded by unit amplitude, and
firing-rate stratification does not control for it, because rate and amplitude
are largely independent.

**Does not establish.** Whether rescue's 50 additional units are genuine
neurons, fragments, or noise. The truncation metric cannot answer that: high
missingness for a small unit is expected in every pipeline. That question needs
waveform, refractory, CCG and spatial evidence on those specific units, which is
tracked separately.

**Does not overturn.** The original `reject_universal_default` verdict, which
rests on the prespecified gates that actually failed — similar good–good pairs
(27 against 8 and 11 post-curation) and edge-spike fraction (2.004%
pre-curation, ~2.066% post-curation, against a 2.000% threshold).

## 7. Consequences for the follow-up plan

- Item 1 (formal completeness gate) stands, but must be specified on
  **unit-matched** or amplitude-stratified comparisons. A population-median gate
  would penalize any configuration that recovers additional small units, which
  is the opposite of the intent.
- Item 2 (frozen matched recomputation) remains worthwhile for provenance, but
  is no longer urgent: the discrepancy it was meant to test is explained.
- Item 3 (validate the fitter) is **complete**; see section 2. Censored windows
  must be filtered, and the ceiling must never be treated as a measurement.
- Items 4–6 should be re-aimed at the 50 rescue-unique units rather than at a
  population-level completeness deficit that does not exist.
