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

## Evidence pointers

- `docs/luke_pipeline_stage_local_validation_strategy.md` — full observable map,
  advancement gates, and reopening criteria
- `docs/luke_20250804_rescue_status_and_test_plan.md` § 2026-08-31 strategic update
