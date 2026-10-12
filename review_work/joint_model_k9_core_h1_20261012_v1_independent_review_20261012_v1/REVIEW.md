# Independent implementation-first review: joint-model K9 synthetic core v1

Verdict: **NO_GO_PENDING_GEOMETRY_DTYPE_AND_DERIVED_CLOCK_GUARDS**.

Do not bind this frozen core to a real-window contract yet. The requested
synthetic mechanics pass, but two fail-closed input/clock defects remain:

1. `session_start_frame` and local `sample_index` are each int64-validated, but
   their derived sum is not. With `session_start_frame = int64.max` and local
   sample 100, the truth row records `9223372036854775907`, outside the promised
   int64 session-clock domain (`IMPLEMENTATION.py:223-225`).
2. Geometry is coerced directly with `np.asarray(..., dtype=np.float64)` before
   dtype validation (`IMPLEMENTATION.py:160`). Numeric strings are accepted,
   and complex coordinates are accepted after their imaginary components are
   discarded with a warning. Both violate the requested fail-closed geometry
   and dtype boundary.

## Exact repairs and rereview condition

- Form the derived clock and pass it through `_int64_scalar("session_sample",
  session_origin + sample)` before storing it.
- Inspect the uncast geometry array first. Require a two-dimensional numeric
  real non-boolean dtype (`kind` in integer/unsigned/floating), the exact
  `(channels, 2)` shape, finite values, and unique sites; only then cast to
  float64 for mapping.
- Preserve the current failing fixtures, add author regressions for positive
  derived-clock overflow plus complex and string geometry, freeze a new packet,
  rerun both suites, verify hashes, and obtain focused rereview.

No injection algorithm, state rule, decoy rule, or real-data action is needed
for these repairs.

## Independent evidence

- Contract commit: `86f6a7a`.
- Implementation commit: `3947bed76fc0fbfd1f37a221bcd22bac75d2e9be`.
- Frozen packet commit: `6614c1f`.
- Packet implementation/tests are byte-identical to the executable copies and
  frozen commit; local/shared manifest and completion chains verify.
- Focused author suite: `16 passed in 0.10s`.
- Combined author/helper suite: `35 passed in 0.13s`.
- Initial independent collection did not run because the review harness omitted
  the repository root needed by the frozen module's reused helper imports. The
  harness-only path was corrected; no implementation or test expectation changed.
- Corrected independent suite: `19 passed, 3 failed in 0.16s`. Failures are
  derived session-clock overflow and the complex/string manifestations of the
  geometry dtype defect.
- Passing subset: `19 passed, 3 deselected in 0.11s`.

Passing coverage includes exact half-open state ownership, positive contiguous
state intervals, tissue `base + true_state` versus probe `base`, supplied-state
voltage invariance, exact same-column placement and support rejection, full
temporal support, linear overlaps with separate truth rows, half-open collision
ownership, complementary decoys, local clock dtypes/ranges, geometry
shape/finiteness/uniqueness, background/template validation, complete truth
receipts, input nonmutation, and unchanged bytes outside event support.

No real recording/window/voltage, DARTsort, H5, GPU, service, sort, RF, held-out
dataset, or C4 action was accessed or performed.

## Implementation checks

- Done: what actually ran -> full frozen contract/source/tests plus reused
  mapping/template helpers and cited commits inspected; source copies and hashes
  verified.
- Done: states/clocks -> half-open boundary ownership, contiguous support, local
  clock validation and receipts pass; derived session clock can exceed int64
  (`IMPLEMENTATION.py:52-108,193,223-225`; `ADVERSARIAL_TESTS.py`).
- Done: frames/coordinates -> tissue and probe placement formulas and
  supplied-state voltage invariance pass; nonreal/nonnumeric geometry is
  silently coerced rather than rejected (`IMPLEMENTATION.py:160,202-208`).
- Done: support/overlap -> nonzero spatial support and full temporal support
  fail closed; overlaps add linearly, keep separate rows, and share collision
  groups; touching half-open supports remain separate.
- Done: decoys -> exactly two biological identities with disjoint state support
  pass; one/three identities and shared states fail.
- Done: nonmutation -> background, geometry and templates remain byte-identical,
  including after a later invalid event; output outside injected support is
  unchanged.
- Not done: exact real input/window/cohort/field/checkpoint binding,
  reader-output recovery, recording/voltage access, DARTsort/H5 integration,
  GPU, services, sorts, RF, holdout, or C4 -> excluded by assignment.
- Can establish: the positive synthetic injection/state/overlap/decoy mechanics
  pass bounded independent fixtures, and two exact fail-closed defects block
  reuse.
- Cannot establish: real K9 benchmark validity, N0/N1 benefit, biological
  identity/purity, or medium/full-run readiness.

