# Focused independent rereview: joint-model K10 evaluator v4

Proposed verdict: **GO_SYNTHETIC_DECISION_PLUMBING**.

V3's independent review found one release blocker: an extra top-level
non-selected candidate holdout summary was silently ignored. V4 applies only
the requested fail-closed key-set guard and adds the preserved failing case to
the author suite.

Independently verify that:

- `selected_holdout` must be a mapping with exactly `candidate_arm`,
  `baseline_arm`, and `endpoints`;
- the preserved v3 failure now raises before holdout values are read;
- all 44 previously passing adversarial fixtures still pass against v4;
- all 35 author fixtures pass;
- the source delta from v3 is confined to this guard and regression fixture.

Do not inspect K9, real arm outputs, recordings/voltage, DARTsort, H5, GPU,
services, sorts, RF or any held-out dataset. Publish a distinct immutable review.

Implementation checks
- Done: v3 NO_GO and failing fixture preserved; exact requested repair implemented.
- Done: 35/35 author fixtures pass.
- Not done: K9 validity, real outputs, CI construction, power or runtime scale.
- Can establish: whether the last synthetic input-boundary blocker is repaired.
- Cannot establish: arm benefit, equivalence, superiority, identity/purity or run readiness.
