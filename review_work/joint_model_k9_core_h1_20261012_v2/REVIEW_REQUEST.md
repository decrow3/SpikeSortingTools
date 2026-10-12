# Focused independent rereview: joint-model K9 synthetic core v2

Proposed verdict: **GO_SYNTHETIC_MECHANICS_ONLY**.

V1's independent review found exactly three failing cases: derived session-clock
overflow, complex geometry, and numeric-string geometry. V2 applies only the
requested input/clock guards and adds those cases to the author suite.

Verify the exact three preserved failures now pass against v2, all 19 previously
passing independent cases remain green, 19 focused author fixtures and the
38-test combined suite pass, and the source delta is confined to the requested
guards. Verify local/shared v1 NO-GO and v2 hash chains.

No reviewer production action. Do not inspect real input/window/voltage,
DARTsort, H5, GPU, sorts, RF/holdout or C4. Publish a distinct immutable review.

Implementation checks
- Done: v1 NO-GO preserved; exact requested guards and regressions implemented.
- Not done: real input/window/cohort/field/checkpoint binding or reader recovery.
- Can establish: whether the known synthetic input-boundary blockers are repaired.
- Cannot establish: K9 real-benchmark validity or N0/N1 readiness.
