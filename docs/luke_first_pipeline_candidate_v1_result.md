# First pipeline candidate v1 — executed result

Date: 2026-09-06. Status: **executed against the frozen contract. Verdict: FAIL.**

Contract: [`configs/first_pipeline_candidate.v1.json`](../configs/first_pipeline_candidate.v1.json)
(digest `31b4fdad…`, frozen before any candidate result existed).
Nomination: [amplitude-completeness audit result](luke_amplitude_dropout_audit_result.md) §8,
case `rescue_luke0804_v2v1_g0_imec0__c37__failure1`.
Outputs: `/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_first_pipeline_candidate_v1`
(local, gitignored; this document points at that run and is not written into it).

## 1. Verdict

| gate | margin (frozen) | measured | verdict |
|---|---|---|---|
| completeness | ≥ 20.0 pp improvement | not applied | **unevaluable** |
| identity | retained fraction ≥ 0.8 | **0.481** | **fail** |
| contamination | increase ≤ 0.01 | **+0.076** | **fail** |
| healthy-interval preservation | increase ≤ 2.0 pp | +0.285 pp | pass, on 4 of 100 clusters |

The candidate is rejected. The healthy-arm pass does not offset this and should
not be quoted on its own: see §5.

## 2. What ran

One retained-sort replay of the accepted rescue sort, in two bounded arms:

| arm | processing interval | endpoint interval | rows read |
|---|---|---|---|
| case | [6350, 7050] s, inside development window [5760, 8700] | [6590.32, 6810.18] s | 2,119,013 |
| healthy control | [2880, 3480] s | H1 = [3120, 3240] s | 1,412,759 |

Plus a control arm (`--option control`) that replays the retained labels through
the same export interface, so "baseline" is a measured comparator.

27,108,816 of the sort's 29,227,829 rows lie outside the case arm's declared
interval and were not processed. No sorting ran; no production output was
written to; the sealed held-out panel and its 300 s buffer were not touched.

**Motion: `declared_absent`.** No `qualified-motion-field-v1` artifact exists for
this recording. The only motion artifact on disk,
`motion/dredge-rigid-sidecar/estimate.npz`, is a `dredge-rigid-estimate-v1`
carrying no qualification receipt, support/confidence grids or polarity
declaration. Manufacturing one would have been fabricating the qualification, so
this arm ran unregistered and is **not motion-aware**. That is defensible here
only because the audit found depth flat on this case (2.81 µm), and it is a
limitation on the run, not a result. A motion-aware arm remains blocked on an
artifact that does not exist.

## 3. Why it failed: the linker shredded the unit it was meant to rescue

Cluster 37 was split into **seven families, one per epoch**. Every one of its six
adjacent-epoch self-links (37→37) was refused, all by the same gate:

| epoch pair | depth Δ (µm) | amplitude ratio | waveform cosine | pair refractory fraction | outcome |
|---|---|---|---|---|---|
| 71→72 | 1.59 | 1.01 | 1.00 | 0.557 | refused |
| 72→73 | 1.00 | 1.02 | 1.00 | 0.536 | refused |
| 73→74 | 3.05 | 1.44 | 1.00 | 0.422 | refused |
| 74→75 | 1.77 | 1.07 | 1.00 | 0.242 | refused |
| 75→76 | 0.45 | 1.06 | 1.00 | 0.213 | refused |
| 76→77 | 3.29 | 1.09 | 1.00 | 0.086 | refused |

Depth, amplitude and waveform all say "same neuron" — cosine is exactly 1.0
because the epochs share the template. The refractory gate says no, every time,
because it is an **absolute** threshold of 0.01 applied to a unit whose own
baseline is 0.396: 39.6% of cluster 37's inter-spike intervals inside the
endpoint interval are shorter than 1.5 ms (38.0% across the whole cluster, ISI
mode at 1.1–1.3 ms, zero duplicate samples). Verified directly from
`spike_times.npy`, independently of this runner.

So `max_refractory_violation_fraction: 0.01` cannot be satisfied by this unit in
any epoch pair, and the same is true probe-wide: 659 original clusters in the
case interval became **2,038 candidate units**; 1,230 links were accepted and
**every one of them joined a cluster to itself**. `num_families_built_from_a_link
= 0`. The intervention performed no cross-cluster identity joining at all, in
either arm, while fragmenting the sort.

The number 0.01 is correct for the *contamination margin*, where the contract
defines it as a maximum tolerated **increase** over baseline. Reusing the same
number as the linker's **absolute** refractory gate silently changed what it
measures. That mistake was invisible to review and to every fixture; it took
executing on real data to expose it.

## 4. The near miss the guards caught

Read naively, the case arm looks like a success:

| | baseline (cluster 37) | candidate (family 614) |
|---|---|---|
| median missing % | **21.79** | **0.95** |
| finite-interior windows | 3 | 1 |
| spikes in the interval | 3,999 | 1,925 |
| interval duration covered | 173.6 s of 219.9 s | **38.8 s of 219.9 s** |

`21.79 − 0.95 = 20.84 pp`, against a frozen minimum improvement of 20.0 pp. That
would have **passed** the endpoint the whole candidate was built for.

It is not a rescue. The candidate reported a cleaner amplitude fit because it
kept the early, high-amplitude 48% of the neuron's spikes and discarded the rest
into six other families — the amplitude collapse the audit selected the case for
was fitted away by throwing out the spikes that showed it, on a train covering
18% of the interval.

Three guards refused it, and any one of them was enough:

1. **The fit-count requirement.** The candidate arm produced one finite-interior
   fit against the contract's minimum of two, so the gate reported `unevaluable`
   and did not compute the improvement. `unevaluable` is not a pass.
2. **The identity floor.** Retained fraction 0.481 against a floor of 0.8. This
   gate only works because the margin's decision rule was disambiguated before
   execution (§6): read as a `minimum_improvement`, 0.8 above a baseline that is
   1.0 by construction is unsatisfiable and the gate would have been meaningless.
3. **Contamination on the exported train.** +0.076 against a tolerance of 0.01,
   scored on the complete exported union rather than on an anchor.

## 5. The healthy-arm pass is thin

The healthy-interval gate passed: worst increase +0.285 pp against 2.0 pp.
It should not be quoted without its denominator. Of 100 clusters with enough
spikes in H1 to consider, **4** were measurable in both arms; the rest produced
too few finite-interior fits in a 120 s window at 1,000 spikes per window. The
same fragmentation is present in this arm — 673 clusters became 2,178 units, and
most clusters' retained fraction sits near 0.5 — and the completeness gate as
specified does not see it, because it compares each cluster against whichever
family kept the plurality of its spikes.

A gate that passes while the arm halves identity retention is measuring the wrong
thing for this failure mode. Recorded here, not fixed: changing it now would be
changing a rule after seeing its answer.

## 6. Contract and implementation corrections made before this run

All of these landed **before** the freeze, and none was made after any candidate
result was visible.

- **`execution_mode` was `resort`; the implementation is a replay.** Corrected to
  `retained_sort_replay`, with `inputs.source_sort_id` naming a declared
  comparator. The runner refuses to execute a contract that declares a re-sort.
- **The identity margin's decision rule was ambiguous.** 0.8 was labelled
  `minimum_improvement` but read as a retention floor. A third margin kind,
  `absolute_floor`, was added to the validator, and every margin now must carry a
  `decision_rule` naming both quantities and the comparison operator. The number
  0.8 and its source are unchanged.
- **The runner could be steered by flags.** It now executes the contract's own
  resolved settings and declared inputs and refuses a conflicting
  `--snippet-dir`, `--config`, `--truth` or `--motion-info-dir`.
- **Dependencies were marked resolved before their checks existed.**
  `thin_candidate_runner`, `option_b_unwarped_identity` and
  `retained_sort_replay_interface` were returned to `unresolved` and moved only
  when the integration and known-answer tests passed; each now records which test
  resolved it.
- **Fabricated evidence removed.** Depths default to zeros no longer — a missing
  `spike_positions.npy` is a refusal. Amplitudes come from
  `full_st[kept_spikes][:, 2]`, never `amplitudes.npy`. Missing motion is a
  refusal, never a zero displacement. No track is labelled `good` by the runner:
  originals are preserved as `cluster_group.original.tsv` and a family built from
  a link is `unvalidated`.
- **Identity linking made conservative.** Waveform compatibility on a common
  physical channel representation (mean template on the probe's 384 sites,
  compared over a shared micrometre neighbourhood of both peak channels);
  exclusivity enforced on **both** source and destination; ambiguous links left
  separate; overlapping epochs deduplicated by original spike-row id rather than
  timestamp, so two distinct spikes at one sample both survive and both stay
  visible to the refractory check; refractory cleanliness re-validated on the
  final exported train under the same overlap-assignment rule.
- **The control arm produces comparable outputs** rather than a bare "executed:
  true" flag.

Tests: 18 known-answer fixtures for the linker
(`testing/test_ladder_unwarped_identity.py`), 30 contract-to-runner-to-export
integration tests (`testing/test_luke_two_motion_pipeline_bakeoff.py`), 177
contract tests, and the pre-existing 210-test amplitude-audit suite, all passing.

## 7. What must not happen next

`max_refractory_violation_fraction` must **not** be retuned and this case re-run.
The contract is frozen, results exist under that freeze, and changing a gate
after seeing the answer it produced is precisely the failure the nomination
prespec was written to avoid ([audit result](luke_amplitude_dropout_audit_result.md)
§7a). This run's verdict stands as FAIL on the margins as frozen.

## 8. Next implementation action

One prespec change, written down before it is executed, for a *v2* contract:

**The linker's refractory gate must be relative to each unit's own baseline, not
absolute.** The quantity that means "this link merged two neurons" is the
increase in refractory violation fraction over what the two observations already
carry separately — the same shape as the contamination margin. As an absolute
threshold it is unsatisfiable for any unit whose baseline exceeds it, which on
this sort is most of them, and it fragments rather than links.

That change alone does not make the candidate work; it makes it testable. Two
things this run says about that, neither of which a re-run settles:

1. Even with the gate fixed, no cross-cluster link survived the waveform cosine
   on this interval (10,470 pairs refused there). Whether the amplitude collapse
   on c37 corresponds to any *other* cluster is still unestablished, and this run
   is evidence against it, not for it.
2. Cluster 37's 38% ISI-under-1.5 ms baseline means 1.5 ms is not describing a
   refractory period for this train. Whether it is a bursting unit or a merge is
   an open question about the retained sort that predates this candidate, and the
   completeness endpoint on such a unit inherits that ambiguity.

## Related records

- [Improvement plan](pipeline_improvement_plan.md)
- [Audit prescription](amplitude_completeness_next_step_prescription.md)
- [Audit result and nomination](luke_amplitude_dropout_audit_result.md)
- [Motion is estimated, never applied](decisions/0002-motion-is-estimated-never-applied.md)
- [Amplitude completeness gates promotion](decisions/0008-amplitude-completeness-gates-promotion.md)
