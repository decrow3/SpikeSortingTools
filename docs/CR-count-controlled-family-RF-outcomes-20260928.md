# CR: count-controlled family RF outcomes

Status: frozen before CR RF outcomes.

## Scope

Use the 88 substantial W2 all-force parent families already defined without RF
results: at least two exact-row no-force children, second child at least 20 rows,
and second-child share at least 0.10. Rank children by descending exact parent
row count, then integer child ID. Preserve all families and all failures.

For each parent/top-child comparison, map actual final-arm events to the same
accepted W2 development response frames. Within each original inner fold,
uniformly sample without replacement to the smaller parent/child event count.
Use seeds 1729, 1730, and 1731. Require at least 500 selected events total and
at least 200 in each fold after thinning. Save the selected dense event-row IDs.

Compute the accepted unchanged-grid/lag fold STAs and symmetric
cross-validated SNR. Report parent and child scores as dependent descriptive
outcomes because exact event lineage overlaps. Record actual final-child parent
purity and unthinned fold counts. For families where both top children qualify,
save both fold STAs, self-fold reliability, and cross-fold child agreement.

Before RF outcomes, freeze at most six display families by descending minimum
top-child valid event count, then parent ID. Displays use a fold-0-selected
parent feature and show fold-1 maps; no held-out peak optimization.

## Fixed inputs and limits

- Original 81-trial split; 12 complete W2 development trials (7/5 inner
  folds); five W2 outer-holdout trials remain unopened.
- Shared accepted gaze calibration only; no new calibration.
- Two CPU threads, one reader, no GPU, no raw voltage, no sorting.
- H1 allowance: 1,800 all-work CPU seconds within the remaining CP budget;
  cumulative ceiling 19,300 seconds.
- Seeds and eligibility thresholds may not change after outcomes are viewed.

Similar RFs do not prove one neuron. Weak RFs are inconclusive. Reproducibly
different RFs are evidence against a merge.
