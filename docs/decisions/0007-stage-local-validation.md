# 0007 — Stage-local validation governs advancement

**Status:** Adopted 2026-08-31 as the governing methodological rule

## The rule

> Each stage must pass an observable that the next stage cannot rescue or
> manufacture.

Corollary, and the reason the rule exists:

> **A better final sort does not prove that every upstream choice was correct.**

## Why this is a decision and not a platitude

Most of the expensive dead ends in this investigation came from judging an
upstream change by downstream unit count. Sorters are adaptive: they can absorb a
bad preprocessing choice and still produce more units, and they can manufacture
apparent yield out of fragmentation or duplication. Unit count is therefore not a
valid acceptance observable for an upstream stage.

This is also why [0001](0001-ks4-unwarped-is-the-production-sorter.md) reports
yield gains *alongside reduced assigned spikes* — the second number is what makes
the first interpretable.

## The rule applies to the gate set itself

Extended 2026-09-02. Stage-local validation was written about *stages*, but the
same failure occurred one layer up, in the **acceptance criteria**.

The frozen gate set measured yield, contamination, refractory violations,
presence, coincidence and edge burden — and contained no measure of per-unit
detection completeness. A configuration could therefore improve every gate it
was scored on while degrading a dimension nobody was scoring. It did:
[0008](0008-amplitude-completeness-gates-promotion.md).

Three specific substitutions that this proved invalid:

| Not a valid proxy for | Because |
|---|---|
| Aggregate spike count → per-unit recall | Fewer total spikes with more units is compatible with each unit being less completely detected |
| Contamination / refractory violations → completeness | These are contamination measures; they bound false positives, not false negatives |
| Firing-rate-bin occupancy ("stable") → amplitude completeness | Units count as stable by time-bin occupancy while being poorly captured by amplitude fitting |

So when adding a gate, ask what the gate set as a whole *cannot* see. A metric
that only ever moves in the favourable direction under a change is more likely
to be insensitive than confirmatory.

## Consequences

- Broad preprocessing searches are **paused**; reopen only for a demonstrated,
  stage-local failure.
- The active work sequence is: motion-estimator validation → coordinate-only
  motion application → bounded sorter experiments → *only then* voltage warping
  or suppressive mechanisms.
- Challengers clear guardrails (reviewed-event recovery, refractory violations,
  duplicate burden) before longitudinal continuity is interpreted
  ([0005](0005-dartsort-kiasort-deferred.md)).
- Screens are diagnostic: they do not silently merge or relabel units, and a
  frozen evaluator verdict is not waived retrospectively when follow-up looks
  favourable ([0001](0001-ks4-unwarped-is-the-production-sorter.md)).
- A gate cannot be formalized on top of an unvalidated estimator or unmatched
  recomputations ([0008](0008-amplitude-completeness-gates-promotion.md),
  follow-up items 2 and 3).

## Evidence pointers

- `docs/luke_pipeline_stage_local_validation_strategy.md` — full observable map,
  advancement gates, and reopening criteria
- `docs/luke_20250804_rescue_status_and_test_plan.md` § 2026-08-31 strategic update
- `docs/luke_20250804_imec0_postcuration_evaluation.md` — the acceptance-layer
  instance of this failure
