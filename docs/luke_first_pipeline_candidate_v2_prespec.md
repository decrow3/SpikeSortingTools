# First pipeline candidate v2 — prespec

Date: 2026-09-06. Status: **written before v2 executes.** Nothing in this file
was chosen after seeing a v2 result.

Contract: [`configs/first_pipeline_candidate.v2.json`](../configs/first_pipeline_candidate.v2.json).
Supersedes: [`v1`](../configs/first_pipeline_candidate.v1.json), whose verdict was
**FAIL** — [result](luke_first_pipeline_candidate_v1_result.md). v1 is not
re-run, re-frozen or edited.

## 1. What this revision is for

v1 failed for a reason more fundamental than the calibration its own §8 named:
**a proposed linking operation destroyed existing identities when it accepted no
link.** Refused epoch-to-epoch continuity links fragmented 659 clusters into
2,038 units, in an arm that made zero cross-cluster merges. A candidate that
cannot justify a change must leave the input alone.

This is the one allowed revision, and its purpose is preservation, not links.

**It is explicitly not permitted** to weaken the waveform or contamination gates
to obtain a merge. If v2 still finds no defensible cross-cluster link, the
correct result is unchanged output and no recovery improvement, and the
candidate closes on cluster 37.

## 2. The invariant

The output partition is a partition of the **original clusters**. Epochs are
observations used to evaluate links; they are not output units.

1. Only an accepted link **between two different original clusters** changes the
   partition. A link between two observations of the same cluster is measured
   and reported as continuity evidence and can never be accepted or refused into
   an output effect.
2. With no accepted merge, the exported spike assignment reproduces the input
   partition **exactly**, apart from family renumbering.
3. Every original event in the processing interval reaches the export, through
   its cluster — including rows below the per-epoch minimum and rows in no whole
   epoch. There is no "unassigned" class and no extra family for one.
4. **Splitting an original cluster is not an operation this replay has.** A
   non-zero split count in the run's own preservation check is a defect, not a
   result. Adding a splitting operation would require its own gates and its own
   fixtures, and is out of scope here.

## 3. Preserving is not certifying, and a dirty baseline does not license a merge

These are two distinct statements and v2 keeps them separate.

- Retaining an imperfect input cluster does **not** certify it as a clean neuron.
  A single-cluster family is the input cluster unchanged; its absolute
  refractory-violation fraction is reported and may be high; it is never pruned
  and never declared clean.
- Conversely, a high baseline violation fraction does **not** license merging
  that cluster with another. The merge gate is the *increase* a merge causes over
  the worst contributing cluster's own baseline, checked on the epoch pair and
  again on the exported train. It **supplements** the depth, amplitude and
  waveform evidence; it is never sufficient by itself.

The increment threshold is `0.01`, the same magnitude v1 used as an absolute cap.
It is not a larger allowance — a test asserts the two numbers are equal.

## 4. What changed, and what deliberately did not

| | v1 | v2 |
|---|---|---|
| output partition | connected components of `(epoch, cluster)` | connected components of **original clusters** |
| same-cluster links | gated like merges; refusal split the cluster | evidence only, no output effect |
| refractory gate | absolute 0.01 on the pair union | **increase** ≤ 0.01, on the pair and the exported train |
| rows in no eligible epoch | new family id per cluster (an extra fragment) | belong to their cluster, like every other row |
| waveform representation | `templates.npy` (whitened) used directly | `templates.npy @ whitening_mat_inv.npy` |
| pruning loop | returned at the first unfixable family | continues while any breaching family has a removable link |
| healthy completeness | pass on whatever was measurable | coverage against a fixed eligible population; below 0.5 it is `inconclusive` |
| healthy identity | not measured | measured on **every** eligible cluster |

Unchanged, deliberately: `min_waveform_cosine` 0.9, `max_amplitude_ratio` 2.0,
`max_spatial_distance_um` 30.0, `ambiguity_threshold_ratio` 0.85, epoch duration
and overlap, the motion declaration (`declared_absent`), both processing and
endpoint intervals, the named practical failure, and **all four acceptance
margins**. The margins were set from baseline evidence before any candidate
result existed; v1's results have since been seen, so re-deriving them now would
be choosing an acceptance criterion after viewing an answer.

## 5. Fixtures that must pass before the run

Known-answer, not review. Each is aimed at one named failure.

1. **The combined invariant.** One fixture carrying all four v1 conditions at
   once: a high-violation input cluster whose self-links are all refused,
   overlapping epochs, ambiguous neighbouring clusters at matching depth and
   amplitude, and boundary/low-count rows in no eligible epoch. Every original
   event survives; no original cluster fragments.
2. **No accepted merge ⇒ identical partition.** Two rows share a family iff they
   shared a cluster, and the export's unit count equals the input's.
3. **Pruning continues past a family it cannot fix.** A filthy standalone cluster
   in one family and a genuinely bad merge in another: the standalone is left
   alone and reported, and the bad merge is still pruned.
4. **Preserved but not certified.** A high-baseline cluster survives with every
   event, is not pruned, and its absolute violation fraction is reported with an
   explicit note that this is not a cleanliness claim.
5. **De-whitening.** The fixture stores templates in a rotated (whitened) space;
   the loader must recover the physical templates, the stored and recovered peak
   channels must differ, and a missing `whitening_mat_inv.npy` must refuse.
6. **Preservation on real exported rows.** Every candidate unit is a union of
   whole original clusters; with all merges refused the exported partition equals
   the input's up to renumbering.

## 6. Decision rule for the run

Unchanged from v1, on the same case and the same healthy control, run once.

| gate | rule |
|---|---|
| completeness | baseline − candidate ≥ 20.0 pp on cluster 37 over [6590.32, 6810.18] s; `unevaluable` is not a pass |
| identity | retained fraction ≥ 0.8 |
| contamination | increase ≤ 0.01 on the exported train |
| healthy preservation | identity: no eligible cluster split; completeness: worst increase ≤ 2.0 pp at ≥ 50% coverage, else `inconclusive` |

**Predicted outcome, recorded before the run.** Under the invariant, with no
merge accepted, the candidate's exported train for cluster 37 *is* cluster 37.
Then completeness reads 0.0 pp improvement (fail, not unevaluable), identity
reads 1.0 (pass), contamination reads 0.0 (pass), and healthy identity
preservation passes. That is the honest shape of "the candidate did nothing" and
it is a **fail on the endpoint**, which is the correct result if no defensible
link exists. A merge would have to survive depth, amplitude, de-whitened
waveform and the increment gate to change any of it.

If that is what happens, **the candidate closes on cluster 37.** The remaining
open questions — whether cluster 37 is one neuron, and whether any fragment of
it exists elsewhere — are not settled by re-running this candidate, and the
correct move is to stop, not to loosen a gate.

## Related records

- [v1 result and its amendments](luke_first_pipeline_candidate_v1_result.md)
- [Improvement plan](pipeline_improvement_plan.md)
- [Audit result and nomination](luke_amplitude_dropout_audit_result.md)
- [Amplitude completeness gates promotion](decisions/0008-amplitude-completeness-gates-promotion.md)
