# Focused independent review: joint-model K10 evaluator v3

Proposed verdict: **GO_SYNTHETIC_DECISION_PLUMBING**.

This is the load-bearing evaluator amendment for the minimal bounded N0-N1
slice. Review the frozen v3 contract and full implementation. V1/v2 remain
historical evidence; v3 must not be treated as having inherited GO without
review.

Independently verify:

- all-truth/all-output global one-to-one matching against a brute-force oracle;
- per-unit pooled matching and the algebra separating cross-cluster duplicate
  credit from cross-unit competing credit;
- no truth or output row receives two global credits;
- split, merge, many-to-one, shifted nulls and overlap endpoints remain present;
- development-only arm selection, deterministic ties and rejection of a
  non-selected holdout arm;
- frozen higher/lower practical-loss margin boundaries;
- PASS/FAIL/INCONCLUSIVE aggregation, with no equivalence/superiority claim;
- positive support below ten units is accepted and reported, while zero,
  fractional, boolean, complex, object and out-of-range inputs fail closed;
- exact session clocks and create-only output remain intact.

Do not inspect K9, real arm outputs, recordings/voltage, DARTsort, H5, GPU,
services, sorts, RF or any held-out dataset. Publish a distinct immutable review.

Implementation checks
- Done: planner amendment frozen before implementation; 34/34 synthetic fixtures pass.
- Done: previous K10 clock repair and review packets preserved by hash.
- Not done: K9 validity, real outputs, statistical CI construction, power or runtime scale.
- Can establish: synthetic global-credit and selection-aware decision arithmetic.
- Cannot establish: arm benefit, equivalence, superiority, identity/purity or run readiness.
