# Independent review request: joint-model K10 evaluator v1

Proposed verdict: **GO_FIXTURE_QUALIFIED_SAVED_OUTPUT_EVALUATOR**.

Review the frozen contract and full evaluator implementation before any
joint-model comparison output is inspected. Trace the actual construction of
TP/FP/FN, best cluster, split, merge, many-to-one, frame and overlap endpoints.
Check the deterministic maximum-cardinality matcher against the brute-force
fixture and confirm both inclusive gates apply to every match.

Load-bearing questions:

- Is matching per truth unit and per output cluster, rather than against the
  pooled spike river?
- Can an output event be reused within one truth/cluster comparison?
- Does the spatial gate compare per-event placements rather than template
  anchors?
- Does the shifted null use the same temporal/spatial statistic and support?
- Are detection, assignment, frame, amplitude and residual overlap scores kept
  separate?
- Are session clocks explicit, exact-key event matching absent, and result
  files create-only?

The 19 fixtures are synthetic and were frozen before any joint-model arm output
was inspected. This review must not infer benchmark validity or arm benefit.

Implementation checks
- Done: task card, method v4, review v3, rank-1 result and historical scorer-correction decisions read and hash-bound.
- Done: source/contract/tests frozen; 19/19 local fixtures pass.
- Not done: K9 benchmark arrays, any real saved arm output, voltage, DARTsort integration, runtime-scale test or medium run.
- Can establish: evaluator construction and known-answer arithmetic on saved fixtures.
- Cannot establish: K9 validity, scientific benefit, biological identity/purity, host readiness or launch authorization.
