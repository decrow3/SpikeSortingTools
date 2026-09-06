# Corrected nomination rule — prespec

Date: 2026-09-06. Status: **PROPOSED, NOT EXECUTED.** Awaiting the user's
go/no-go. Nothing has been re-run against this rule.

## Why this document exists

The audit's first real run ([result](luke_amplitude_dropout_audit_result.md))
nominated `legacy…c424__failure2` because `_nominate` returns the first
eligible case in the freeze's order, and that order is alphabetical by sort ID.
The rule is arbitrary with respect to evidence: it passed over
`rescue…c37__failure1`, which has rank 1, roughly double the amplitude effect,
the largest depth shift of the supported set, and the only completed voltage
review.

That defect was **not** fixed before reporting, and this rule has **not** been
executed. Adjusting the rule that consumes a ranking, after its answer is
visible, voids the point of freezing constants before the ranking is read. So
the corrected rule is written down first, here, with its basis, and executed
only on an explicit decision.

## The honesty problem, stated plainly

**I already know this rule would nominate `rescue…c37__failure1`.** A rule
written by someone who knows its answer is a retrofit unless it can be
specified without reference to that answer, from constraints that were true
before the ranking existed. The rest of this document is that argument, and the
disclosure above is part of it.

## The a priori constraint

The legacy sort has no raw voltage and never will: `traces_cached_seg0.raw` was
deleted from `pipeline_results_Luke0804_V2V1_g0_imec0/preprocessed_recording/`
after sorting. A case from that sort cannot have voltage evidence collected —
not "was not collected", *cannot be*.

This was on record before any case existed, in two independent files:

| record | commit | committed |
|---|---|---|
| contract: `comparators.legacy.provenance.raw_voltage_available: false` | `701d890` | 2026-09-05 14:14:35 |
| audit config: legacy `provenance.num_samples_source` names the deletion | `bfce40c` | 2026-09-05 14:21:01 |
| **first ranking** (`run1/selection.json`) | — | **2026-09-05 19:35:11** |

`bfce40c` is the commit that froze case selection, titled "freeze the case
selection before anyone can see a ranking". The constraint predates the first
ranking by over five hours. It is not knowledge acquired from the result.

## The rule

Among cases already eligible for nomination — role `failure`, `case_status`
stable, and exactly one supported mechanism (an `ambiguous` reading is still
never a nomination) — order by:

1. **Executable evidence first.** Prefer a case whose sort has raw voltage
   available. A nominated next action rests on evidence the experiment will
   need; a case whose voltage limb is uncollectable in principle cannot have
   that evidence completed at any cost, so it is the weaker nomination
   regardless of its numbers.
2. **Frozen rank.** Then prefer the lower `rank` — the freeze's own ordering,
   computed from the cached QC before any evidence was read.
3. **Effect on the supported limb.** Then prefer the larger measured effect on
   whichever limb carries the supported verdict.
4. **Determinism.** Then `sort_id`, then numeric `cluster_id`.

Criteria 1 and 2 are properties of the inputs and the freeze. Criterion 3 is a
preference statable in the abstract ("prefer the larger effect") and is only a
tie-break behind them.

## What this rule does NOT do

- It does not change any selection constant, evidence constant, threshold or
  interval. The case set is fixed by the freeze and is untouched.
- It does not make an unsupported case nominable, rescue an `ambiguous`
  reading, or nominate an unstable case.
- It does not change what the evidence means. Every limitation in §6 of the
  result stands: depth is flat on every case, so the supported verdicts rest on
  the amplitude limb alone, and that limb is measured on the same distribution
  whose truncation defined the case.

## The check that makes the re-run auditable

Constants, intervals and input data are unchanged, so a corrected run must
reproduce **the same six case IDs and the same exclusion counts** as `run1` and
`run2`, differing only in which case `decision.md` nominates. If the case set
moves, something other than the nomination rule changed and the re-run is not
clean. It goes into a fresh output root; `select` and `inspect` refuse a second
run into an occupied one by design.

## What is still true either way

Nominating `rescue…c37__failure1` would not make the category label mean
motion. The registration branch of `motion_amplitude_change` has little to act
on — every depth shift is 0.84–3.01 µm against a frozen 8 µm threshold — while
the retained-row lineage independently shows 0 of 2,000 labelled rows dropped
by `kept_spikes` in every failing span, which is a cheap negative that does not
need KS4 semantics and removes curation repair from the remedy space. Two limbs
converge on identity handling as what remains standing. That steer does not
depend on which case is nominated, and it should be stated in whatever
experiment contract follows.
