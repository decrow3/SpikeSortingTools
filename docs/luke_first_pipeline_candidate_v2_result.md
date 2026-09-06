# First pipeline candidate v2 — executed result

Date: 2026-09-06. Status: **executed once against the frozen v2 contract.**
**The candidate is safe and ineffective. It closes on cluster 37.**

Contract: [`configs/first_pipeline_candidate.v2.json`](../configs/first_pipeline_candidate.v2.json)
(digest `e758716a…`, frozen before any v2 result existed).
Prespec: [v2 prespec](luke_first_pipeline_candidate_v2_prespec.md), written first.
Supersedes: [v1, verdict FAIL](luke_first_pipeline_candidate_v1_result.md) — not
re-run, not edited.
Outputs: `/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_first_pipeline_candidate_v2`.

## 1. Verdict

| gate | margin (carried from v1, unchanged) | measured | verdict |
|---|---|---|---|
| completeness | ≥ 20.0 pp improvement | **0.0 pp** | **fail** |
| identity | retained fraction ≥ 0.8 | **1.0** | pass |
| contamination | increase ≤ 0.01 | **0.0** | pass |
| healthy — identity preservation | no eligible cluster split | **632 of 632 intact** | pass |
| healthy — completeness | ≤ 2.0 pp at ≥ 50% coverage | 0.0 pp at **1.7%** coverage | inconclusive |

The preservation invariant holds and the candidate does nothing for cluster 37.
Per the prespec §6, that closes the candidate on cluster 37.

## 2. The invariant holds on real data

| | v1 | v2 |
|---|---|---|
| original clusters (case interval) | 659 | 659 |
| exported units | **2,038** | **641** |
| original clusters **split** | ~all | **0** |
| cross-cluster merges | **0** | **18** |
| rows with no family | 150,788 given an extra fragment | **0** |
| overlap rows with conflicting family | 374,965 | not a category any more |

Same on the healthy interval: 673 clusters, **0 split**, 19 merges, 652 units.
Every candidate unit is a union of whole original clusters, checked on the
exported rows rather than asserted.

The extra fragment class and the overlap conflict both disappeared rather than
being fixed, because the partition no longer consults epochs at all: a row's
family is its cluster's family, so a row in no eligible epoch has one anyway and
a row in two epochs cannot disagree with itself.

## 3. Cluster 37: preserved exactly, and unchanged

| | baseline (cluster 37) | candidate (family 34) |
|---|---|---|
| contributing original clusters | 37 | **37** — nothing merged into it |
| spikes in the endpoint interval | 3,999 | 3,999 |
| finite-interior windows | 3 | 3 |
| median missing % | 21.794 | **21.794** |
| refractory violation fraction | 0.396 | 0.396 |

Improvement 0.0 pp against a 20.0 pp margin: a **measured** fail, not
`unevaluable`. Both arms fit the same three windows over the same 173.6 s of the
220 s interval, from the same production amplitude source.

This is the shape the prespec predicted and wrote down before the run. It is the
honest result when no defensible link exists: the candidate left the neuron
alone and therefore did not improve it.

## 4. Merges did appear — 18 of them, and they are not cluster 37

The prespec predicted **zero** merges. That prediction was wrong, and the reason
is mechanistic: in v1 the same-cluster continuity links participated in the
exclusivity matching and *claimed the destinations*, so genuine cross-cluster
candidates lost to them on `destination_already_claimed`. Removing self-links
from the matching freed those destinations. De-whitening contributed too —
waveform refusals fell from 10,470 to 10,230.

What the 18 merges look like, descriptively:

| | |
|---|---|
| accepted merge links | 133 (31 further pruned by the exported-train refractory increase) |
| waveform cosine | min 0.902, median 0.914, max 0.981 — all just above the 0.9 gate |
| depth separation | median 4.2 µm, max 23.4 µm |
| amplitude ratio | median 1.23, max 1.90 |
| refractory increase | median 0.0000, max 0.0070 |
| families formed | 18, every one a pair of two clusters, **all labelled `unvalidated`** |

**11 of the 18 pairs also merge in the healthy interval**, 3,500 s earlier and
570 s long — a separate observation of the same cluster pairs. That is a
reproducibility signal for the merge operation, not a validation of the merges:
nothing downstream has checked them, they sit close to the cosine threshold, and
they remain `unvalidated` in the export. No unit is labelled `good` by the
runner.

## 5. The healthy arm: preservation passes, completeness cannot be evaluated

Identity preservation is measured on the **fixed eligible population** — all 632
original clusters with rows in H1, chosen before measurement. All 632 survive
intact, 0 split, coverage 1.0. That is the gate that matters for this failure
mode and it passes with full coverage.

Completeness is a different matter. Of 632 eligible clusters, 100 have enough
spikes in a 120 s window to consider and **11** produced enough finite-interior
fits in both arms — 1.7% coverage against a required 50%. The comparison is
therefore **inconclusive**, not a pass, and the worst measured increase (0.0 pp)
is not evidence about the other 621 clusters. v1 reported the analogous number
as a pass on 4 of 100; that reading is what this coverage requirement removes.

## 6. Defects fixed, with the fixtures that hold them

| defect | fixture |
|---|---|
| refused continuity links fragmented clusters | the combined invariant fixture: high-violation cluster, refused self-links, overlapping epochs, ambiguous neighbours, rows in no eligible epoch — every event survives, nothing splits |
| no accepted merge still changed the partition | all-merges-refused run reproduces the input partition exactly, unit count included |
| pruning stopped at the first unfixable family | a filthy standalone cluster in one family and a bad merge in another: the standalone is reported, the merge is still pruned |
| whitened templates used as physical waveforms | fixture stores rotated templates; loader must recover the physical ones, peak channels must differ, missing `whitening_mat_inv.npy` refuses |
| boundary/low-count rows became an extra fragment | no unassigned class exists; a run producing one is a refusal |
| an imperfect cluster was implicitly certified or implicitly licensed a merge | preserved-not-certified fixture: high baseline reported, zero increase, never pruned |

On the real data, de-whitening moved the peak channel for **52 of 710
templates** — the review's point was material, not theoretical.

Tests: 21 linker fixtures, 37 contract-to-runner-to-export integration tests,
197 contract tests, 20 fitter tests. 255 passing.

## 7. What this does and does not establish

**Does:** the candidate now preserves what it cannot improve. On two independent
intervals it split nothing, dropped nothing, invented no label, and made a small
number of reproducible, conservatively gated merges. On cluster 37 it changed
nothing and the endpoint says so with a measured 0.0 pp.

**Does not:**

- It does not establish that identity redistribution is absent for cluster 37.
  It establishes that **this candidate found no acceptable link for it** under a
  de-whitened waveform rule at cosine ≥ 0.9. A different rule, or a motion-aware
  arm, might find one; neither was run.
- It does not establish that the 18 merges are correct. They are `unvalidated`,
  near the threshold, and their downstream trains have not been checked.
- It does not resolve what cluster 37 is. 38% of its inter-spike intervals are
  under 1.5 ms; it is a cluster, and nothing here shows it is one clean neuron.
  Its truncated-amplitude fit remains the observation that selected it and it
  inherits that ambiguity.
- The arm is **not motion-aware**. No `qualified-motion-field-v1` exists for this
  recording and none was fabricated.

## 8. Next

**Close this candidate on cluster 37.** The prespec committed to that outcome
before the run, and the run produced it: unchanged output, no recovery
improvement. Loosening the waveform or contamination gate to manufacture a link
is out of scope and was ruled out in advance.

Two things are worth carrying forward, neither requiring another run of this
candidate:

1. **The 18 merges are a separate, testable question.** They reproduce across
   intervals and are conservatively gated, but they are unvalidated. Whether
   they are real split-unit repairs belongs to its own bounded evaluation with
   its own endpoints — not to the unit-37 recovery question.
2. **What cluster 37 is remains open**, and is a question about the retained
   sort that predates this candidate. It is not settled by re-running a replay.

## Related records

- [v2 prespec](luke_first_pipeline_candidate_v2_prespec.md)
- [v1 result and its amendments](luke_first_pipeline_candidate_v1_result.md)
- [Improvement plan](pipeline_improvement_plan.md)
- [Audit result and nomination](luke_amplitude_dropout_audit_result.md)
