# Cheap Kilosort-family plausibility and lighthouse comparison

## Headline

Five of the 24 depth-blind cosine families pass a simple post-hoc depth/time
coherence screen: **F001, F010, F015, F016, and F022**. The strictest tested
refractory cutoff leaves a stable core of **F001, F010, and F015**; a looser
cutoff adds F014. These are candidate tracklet groups, not verified cells.

The useful waveform-only predictor is not raw pairwise cosine by itself. The
best separator is an **isolation margin**: the family's weakest internal cosine
minus its closest similarity to any template outside the family (AUC 0.80,
unadjusted permutation p=0.048). Low template-amplitude imbalance is next (AUC
0.79, p=0.060). Neither survives correction across 12 exploratory predictors,
so this is a design hint rather than a confirmed rule. Kilosort's `good` label
is weak here (AUC 0.61), and the median internal cosine is nearly uninformative
(AUC 0.58).

The displacement comparison is **suggestive but not yet a replication**. The
five coherent families have best individual-cell correlations of 0.76–0.89,
but four fail a circular-shift null that corrects for choosing the best of the
17 previous lighthouse cells. F016 is the sole boundary result: best-cell
r=0.81 and prior-cell-median r=0.48, both exact shift p=0.05. F015 has the
strongest descriptive agreement with the prior-cell median (r=0.60), but its
shift p=0.20. F022 does not track the prior-cell median (r=-0.07).

## Frozen selection and post-hoc outcome

Family construction is unchanged from the v2 screen: every cached Kilosort
template pair with lag-tolerant, relative-patch cosine >=0.95 is connected, and
connected components form proposed families. Absolute depth and event time are
not used to create or rank those families.

After freezing membership, a family is called `depth_time_coherent` only if all
of these are supported and pass:

- at least five chronological cross-CID joins separated by no more than 2 s,
  with no more than 10% jumping at least 100 um;
- at least three 1-s bins in which two member CIDs each contribute at least two
  spikes, with no more than 10% showing CID median depths separated by at least
  80 um;
- merged 1-ms short-ISI fraction no more than three times the matched Poisson
  expectation.

An observed failure produces `implausible`; inadequate observations without a
failure produce `underpowered`. The result is 5 coherent, 17 implausible, and 2
underpowered families. This screen is intentionally cheap and legible. It is
not a claim that the cutoffs are optimal.

## Which pre-reveal properties predict coherence?

The predictor analysis excludes depth and event time. It compares the five
coherent families with the 17 decided-implausible families, leaving out the two
underpowered families. With only 22 decided examples, multivariate fitting
would overfit; the audit reports univariate AUCs and deterministic label-
permutation p-values instead.

The ranking suggests three practical refinements for the next cheap family
builder:

1. Prefer families separated from the rest of the template bank, not merely
   pairs with a high internal cosine.
2. Penalize single-linkage growth. Smaller complete-link groups separate better
   than large chained components, although that association is not independently
   significant here.
3. Prefer balanced member amplitudes and spike support. These are secondary
   hints, not hard filters yet.

The result argues against increasing the cosine threshold alone: several of the
most visibly impossible long-distance pairs have internal cosine near 0.98.

## Comparison with previous lighthouse displacement

For each coherent family, cached Kilosort spike depths are median-binned at 5 s
and centered on 930–940 s. Previous lighthouse traces use only
`strict_accepted` observations, the same binning, and the same seed centering.
No lag, sign, gain, or spatial matching is fit. Each family is compared both to
every previous cell and to the per-bin median across cells, requiring at least
three previous cells for that median.

| Family | Range (um) | Best prior cell | Matched bins | Best r | Library-shift p | Prior-cell median r | Median-shift p |
|---|---:|---:|---:|---:|---:|---:|---:|
| F001 | -18 to 42 | 317 | 12 | 0.80 | 0.15 | 0.39 | 0.20 |
| F010 | -48 to 10 | 675 | 6 | 0.81 | 0.15 | 0.24 | 0.30 |
| F015 | 0 to 90 | 527 | 5 | 0.89 | 0.15 | 0.60 | 0.20 |
| F016 | -5 to 40 | 527 | 5 | 0.81 | **0.05** | 0.48 | **0.05** |
| F022 | -18 to 20 | 527 | 5 | 0.76 | 0.25 | -0.07 | 0.70 |

The null circularly shifts the entire 17-cell reference library through the 19
nonzero offsets of the 20-bin grid, then repeats best-cell selection. Its p-value
resolution is therefore only 0.05. F016 is worth following, but the correct
summary is one boundary result, not five confirmations.

## Outputs and reproducibility

- `testing/luke_kilosort_family_plausibility_v3.py`
- `testing/test_luke_kilosort_family_plausibility.py`
- `testing/outputs/luke_kilosort_family_plausibility_v3/family_plausibility_audit.csv`
- `testing/outputs/luke_kilosort_family_plausibility_v3/classification_threshold_sensitivity.csv`
- `testing/outputs/luke_kilosort_family_plausibility_v3/waveform_only_predictor_associations.csv`
- `testing/outputs/luke_kilosort_family_plausibility_v3/coherent_family_by_lighthouse_comparisons.csv`
- `testing/outputs/luke_kilosort_family_plausibility_v3/coherent_family_motion_summary.csv`
- `testing/outputs/luke_kilosort_family_plausibility_v3/binned_displacement_tracks.csv`
- four figures as PNG and PDF, plus hashed `summary.json`

The audit reads no voltage, launches no sort, and refits no family. Seven tests
cover this analysis and the two preceding family-building stages.

## What this establishes—and what it does not

It establishes that the cheap depth-blind Kilosort-template idea contains a
small coherent subset and identifies a better candidate-ranking feature:
internal-versus-external waveform isolation. It also shows limited shared
movement with the prior lighthouse measurements, strongest for F016.

It does not establish cell identity, whole-probe false-positive rate, unbiased
depth, or a validated motion field. Both sides of the comparison use products
derived from the same recording, and Kilosort `spike_positions` are PC-weighted
sorter outputs. The next cheap test should inspect F001/F010/F015/F016/F022 at
the waveform/event level and repeat the displacement overlay with evidence
classes separated; raw-voltage re-extraction can remain deferred until that
screen says which families are worth it.
