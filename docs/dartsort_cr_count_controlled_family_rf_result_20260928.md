# CR count-controlled family RF result

## Verdict

The count-controlled W2 development RF comparison provides **no reproducible
functional advantage for no-force splitting**. It is also too support-limited
and noisy to prove that the force merges are biologically correct. The CP
recommendation therefore stands: retain force-merged grouping as the current
default while treating individual merge identity as unresolved.

## Frozen support

All 88 substantial W2 split parents and both preselected top children were
preserved, giving 176 comparisons. Only 25 comparisons passed the unchanged
post-thinning rule of at least 500 events total and 200 in each original inner
fold; 151 failed and remain in the table. Only four families had both top
children eligible. Before RF outcomes, those four families—parents 485, 455,
551 and 101—were frozen for display by descending minimum child count and then
parent ID.

Every actual final child in the 25 comparisons was the limiting train in both
folds. Median exact-parent purity was 0.9988 (minimum 0.9643), so the result is
not driven by appreciable mixing from other all-force parents. Uniform sampling
without replacement was repeated with seeds 1729, 1730 and 1731. All 300
selected row-ID arrays are saved and independently validate for count,
uniqueness, label, original fold, and exact common response support.

## Count-controlled outcome

| Seed | Comparisons | Child wins / losses | Median child-parent CV SNR | Mean difference |
| ---: | ---: | ---: | ---: | ---: |
| 1729 | 25 | 6 / 19 | -0.363 | -0.530 |
| 1730 | 25 | 7 / 18 | -0.465 | -0.507 |
| 1731 | 25 | 15 / 10 | +0.211 | -0.196 |

Across all three thinnings, eight comparisons were negative every time and
three were positive every time. The aggregate sign changes with the thinning
seed, so the seed spread is material and is not an independent-trial confidence
interval. Parent and child scores are dependent because the child events are a
subset of the parent lineage.

The four two-child families do not provide an interpretable child-versus-child
RF identity result. Child self-fold map cosines range from -0.024 to 0.053, and
cross-fold child-pair cosines range from -0.001 to 0.018. These are weak RF maps;
their low agreement is inconclusive, not evidence that the children are distinct
neurons. One high-CV comparison exists, but it was not selected or promoted by
an outcome threshold and does not alter the population result.

## Integrity and limits

- The accepted gaze receipt passes all six holdout-leakage checks.
- The original 12 complete development trials and 7/5 inner folds were used.
- The five overlapping outer-holdout trials remain unopened.
- The unchanged stimulus grid, lags, feature selection, and opposite-fold
  scoring were reused from the accepted evaluator.
- All 25 qualifying comparisons, every failure, actual child purity, original
  fold counts, selected row IDs, fold STAs, and the four-family agreement table
  are retained.
- Similar RFs cannot prove one neuron; weak RFs are inconclusive; reproducibly
  different, reliable RFs would be evidence against a merge. This data set
  contains too few reliable two-child families for that last test.

## Outputs

Shared packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cr_family_rf_count_control_20260928/`.

Local packet:
`testing/outputs/cr_family_rf_count_control_v1/`.

No sort, calibration, GPU work, voltage read, or outer-holdout evaluation was
performed.
