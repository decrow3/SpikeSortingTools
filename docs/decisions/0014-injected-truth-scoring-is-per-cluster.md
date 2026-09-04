# 0014 — Injected-truth recovery is scored per output cluster, not against the pooled spike river

**Status:** Adopted 2026-09-03
**Corrects:** the v2 `ground_truth_scores` in `testing/ladder_score.py`
introduced under [0011](0011-cross-sort-event-matching-and-detection-evidence.md)
**Evidence:** the failed C2 v3 run,
[`../luke_20250804_c2_v3_scorer_validation_failure.md`](../luke_20250804_c2_v3_scorer_validation_failure.md)

## Problem

[0011](0011-cross-sort-event-matching-and-detection-evidence.md) correctly
retired the non-exclusive coincidence counter that let one output spike be
credited to two nearby truth events. The replacement (`_exclusive_event_matches`)
ran one global maximum-cardinality 1:1 match between **all injected truth events
and every spike in the sort**, then credited each matched pair to whichever
cluster the output spike belonged to.

On a dense recording that is the wrong contest. The imec1 strip carries ~10⁷
spikes; within ±0.5 ms of any injected event there are dozens of unrelated
spikes. The global matcher hands the injected event to whichever of them is
first in interval order, so an unrelated background cluster routinely "wins"
ownership. Those stolen events then count against the injected unit's own
cluster as **both** a false negative (it didn't get the event) and a false
positive (its real spike for that event is unmatched).

C2 v3 exposed this cleanly: all 14 compact donors, 73–295 µV, both polarities,
injected static (no motion) and scored, landed at accuracy **0.74–0.79** with a
near-constant **~80 FP + ~87 FN**. A 74 µV and a 295 µV neuron cannot fail a
detection benchmark identically; the ~10 % floor was the background-steal rate,
not a pipeline property. No drift penalty could be computed because no donor
cleared the 0.8 static-qualification gate.

## Decision

`ground_truth_scores` scores recovery **per candidate output cluster**:

1. For each injected truth train, the candidate clusters are those with at
   least one spike within tolerance of the train.
2. Within each candidate cluster, run exclusive 1:1 interval-order matching
   between the truth train and **that cluster's spikes only**. Compute
   `TP`, `FP`, `FN`, `accuracy = TP / (TP + FP + FN)`.
3. The **best cluster** is chosen by accuracy, then by greater `TP`, then by
   fewer `FP`. Its score is the primary recovery metric.
4. **Split burden** counts every cluster that satisfies **both**
   `capture = TP / N_truth > 5 %` **and** `TP > CHANCE_MARGIN × E[TP]`, where
   `E[TP] = N_truth × cluster_size × (2·tol+1) / total_samples` is the
   coincidence expected between two uniform-random trains and `CHANCE_MARGIN = 3`.
   Duplicated capture across genuine fragments is still allowed. The chance test
   was added after re-scoring the C2 v3 static arms: on the dense imec1 strip
   3–6 high-rate background clusters clip > 5 % of a 6 Hz injected train by pure
   luck, which without it flagged every clean donor as split. It replaces an
   earlier `precision = TP / cluster_size > 5 %` clause, which would also have
   suppressed a genuine but contaminated fragment (Codex review).
5. **Merge burden** asks whether the chosen best cluster captures > 5 % of any
   *other* injected truth train **and** by more than `CHANCE_MARGIN × E[TP]`.
   Chance coincidence between the best cluster and a sparse injected train is
   tiny, so an imbalanced real merge — a small train wholly swallowed by a large
   best cluster — is still caught.

Exclusivity is enforced only between one truth train and one cluster. It is not
enforced across clusters; cross-cluster competition is resolved by step 3.

The v2 invariant "one output event can never be credited to two truth events" is
withdrawn. Two injected neurons firing within tolerance that the sorter resolves
as one unit are now scored as a **merge for both trains** (neither is a clean
recovery), which is the more useful statement.

`SCORE_SCHEMA` is `luke-ladder-score-sort-v3`.

## Guardrail

These regression tests in `testing/test_ladder_score.py` must pass before any
injected-truth run:

- `test_dense_background_does_not_steal_from_the_best_cluster` — guaranteed
  early steal partners across 40 sub-threshold background clusters leave the
  best cluster's TP/FN and accuracy untouched; a v2-style pooled matcher on the
  same fixture would score < 0.8.
- `test_split_diagnostic_ignores_chance_coincidence_from_high_rate_background` —
  7 background clusters at ~67 Hz clip > 5 % of the train by chance; the
  precision clause flags only the real recovery, `n_capturing == 1`.
- `test_true_split_lowers_accuracy_and_is_flagged` — one train across two
  clusters drops primary accuracy and sets split count > 1.
- `test_duplicate_cluster_is_noticed_while_best_cluster_score_stays_sensible` —
  a partial copy of the train in a second cluster is caught by the split
  diagnostic while the clean best cluster still scores accuracy 1.0.
- `test_identity_continuity_match_across_a_bin_edge_never_exceeds_one` — a
  match straddling a 30 s bin boundary keeps per-bin accuracy in [0, 1].
- `test_ground_truth_normalises_unsorted_inputs` — a shuffled sort and truth
  train are normalised before matching.

## Consequence

The failed C2 v3 run is retained as a **scorer validation failure**, not a C2
result. Its static arms must be re-scored or equivalently validated first under
both configs for all 14 donors. Because the v3 prespec/output namespace is
frozen with stale trajectories, the moving rerun is C2 v4: a new frozen
Luke-calibrated rigid family per
[0013](0013-luke-imec0-has-appreciable-rigid-motion.md).
