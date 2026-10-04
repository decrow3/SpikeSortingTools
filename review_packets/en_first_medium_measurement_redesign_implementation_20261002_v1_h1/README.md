# First-medium measurement redesign implementation v1

Verdict: `READY_FOR_H5_IMPLEMENTATION_REVIEW_EXECUTION_DISABLED_NO_OUTCOMES_COMPUTED`.

This fresh namespace implements the outcome-blind workstream-B redesign without
changing or rerunning the prior evaluator.

Fitted amplitude completeness is retained only as a conditional diagnostic. It
uses the existing `finite_interior` fit status and unchanged two-window / 50%
common-time requirements. It reports every fixed-cohort unit plus REF,
candidate, and common support over the full cohort-time denominator. It never
refits, imputes, loosens a threshold, or supplies an advancement verdict.

The proposed primary estimand is now mathematical: `R(X)` is the equal-unit
mean over a fixed REF cohort of the fraction of that REF unit's events matched
exclusively to its single reciprocal primary partner in X. An unmatched REF
unit contributes zero. `DeltaR = R(repaired B) - R(existing B)`. Complete sorts
participate in reciprocal-primary competition, candidate events receive no
double credit, and fixed half-open 60-second cells are diagnostics only. This
is a continuity proxy, not biological identity or novel-event recovery.

The proposed margin rule is preserved and rejected explicitly—not silently
changed. `DeltaR >= 0.05` plus a lower 95% bound above zero supports a positive
effect with a large point estimate, but does not provide 95% evidence that the
effect exceeds the stated 5-point minimum. The frozen replacement is a
one-sided paired-REF-unit bootstrap lower bound strictly above `0.05`. That
bootstrap is only a fixed-cohort composition robustness calculation; it does
not cover recording, session, animal, matching-rule, or biological-identity
uncertainty. No power claim is made without an outcome-blind variance plan.

All requested known-answer fixtures are included. No redesigned real cross-arm
result was computed.

Implementation checks
- Done: retained the accepted exclusive matcher hash, reviewed every new
  denominator and branch, and froze matching, cell, support, bootstrap, and
  margin semantics in JSON.
- Done: REF self, self-difference, relabel, duplicate, deletion, split, merge,
  unmatched-zero, no-double-credit, half-open boundary, conditional support
  union, and margin-counterexample fixtures pass.
- Done: 35 combined tests and 9 packet-only tests pass.
- Not done: no real arrays, cross-arm result, recording/voltage, sort,
  training/detection, RF/holdout, waveform, or biological identity work.
- Can establish: explicit outcome-blind implementation and synthetic semantic
  correctness in the reviewed environment.
- Cannot establish: DeltaR, power, repaired-arm efficacy, identity, novel-event
  recovery, waveform stability, or readiness to advance.
