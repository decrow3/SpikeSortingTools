# Focused independent review: joint-model K9 synthetic core v1

Proposed verdict: **GO_SYNTHETIC_MECHANICS_ONLY**.

Review the frozen contract, full implementation and tests. Independently check:

- exact half-open state ownership at boundaries;
- tissue placement uses `base + true_state`, while probe placement uses `base`;
- supplied field state affects metadata only and cannot change voltage;
- only exact 40 µm same-column mappings are admitted;
- nonzero spatial support and temporal support cannot be truncated;
- overlap members add linearly and retain separate truth rows;
- decoy groups require exactly two identities with complementary state support;
- local/session clocks, placement, frame, state, interval and collision receipts;
- fail-closed handling of bool/complex/object/string/fractional/out-of-range clocks,
  duplicate IDs, unknown templates, invalid geometry/dtypes and boundary crossings;
- input arrays are not mutated and bytes outside injected supports are unchanged.

Use independent known-answer fixtures, including adversarial geometry and clock
cases. No reviewer production actions. Do not inspect or choose real windows,
read recordings/voltage, run DARTsort/GPU/sorts, access RF/holdout, edit C4 or
infer N0/N1 readiness. Publish a distinct immutable review.

Implementation checks
- Done: contract frozen before implementation; focused and reused-helper regressions pass.
- Not done: exact real input/window/cohort/field/checkpoint binding or reader-output recovery.
- Can establish: synthetic pre-CAR exact-lattice voltage/truth mechanics.
- Cannot establish: K9 real-benchmark validity or N0/N1 run readiness.
