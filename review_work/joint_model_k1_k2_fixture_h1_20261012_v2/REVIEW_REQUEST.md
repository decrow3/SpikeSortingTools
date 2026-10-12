# Focused independent rereview: joint-model K1/K2 fixture v2

Proposed verdict: **GO_SYNTHETIC_FIXTURE_QUALIFIED**.

V1 remains immutably NO_GO because K2 accepted complex-valued sample clocks.
V2 changes `_integer_1d` and its author regressions only. It now permits only
non-boolean integer arrays or real floating arrays whose finite values are
exact integers in the int64 domain; complex, boolean, object/string,
fractional, nonfinite, and out-of-range values fail closed.

Rerun the author suite and the included prior independent suite against frozen
v2. Verify the latter differs from the preserved v1 reviewer test only by the
explicit packet-directory binding. Add independent dtype/range cases and
confirm K1, detector, CCG, channel, lag, tie, and width behavior is unchanged.
Verify v1 author/review packets and local/shared hash chains remain valid.

No real K1 arrays, K9 output, recording voltage, H5, GPU, service, sort, RF or
holdout may be accessed.

Implementation checks
- Done: precise v1 failure preserved; scoped clock dtype/range repair and regressions frozen.
- Done: combined replay 33/33 passes against v2.
- Not done: six C4-0 windows, 0804 pulse rate, real lag or scale/integration.
- Can establish: exact v1 clock defect is repaired without K1 or K2 statistic changes.
- Cannot establish: real-window equality, scientific benefit, identity/purity or launch readiness.
