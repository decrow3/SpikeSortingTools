# Independent implementation review: joint-model K1/K2 fixture v1

Verdict: **NO_GO**.

The K1 implementation passed the frozen author suite and all independent
synthetic checks.  The K2 CCG arithmetic, lag sign, inclusive support, channel
selection, deterministic peak, and label-free interface also passed an
independent brute-force oracle.  The packet as a whole is not eligible for GO,
because the K2 clock validator accepts complex arrays and silently discards
their imaginary component while casting them to `int64`.

## Load-bearing finding

`IMPLEMENTATION.py:192-199` applies `np.isfinite` and `np.rint` to every
non-integer dtype.  A value such as `100+0j` therefore passes the equality
test, reaches `astype(np.int64)`, emits only `ComplexWarning`, and is accepted
as acquisition sample 100.  The frozen contract requires integer acquisition
samples and names noninteger guards as an acceptance criterion.  The preserved
review test `test_k2_rejects_complex_clock_values_instead_of_silently_casting`
fails for this reason.

Exact repair: in `_integer_1d`, allow integer dtypes directly; for non-integer
dtypes, first require a real floating dtype, then require finite integral-valued
entries.  Reject complex, boolean, string and object dtypes with `ValueError`
before any cast.  Freeze a corrected v2 snapshot, retain this v1 failure, add
complex/boolean/object rejection fixtures for every clock array, and rerun the
author and independent suites.

## Evidence

- Frozen author hashes and manifest were recomputed and matched.
- Exact reader, implementation and tests matched their working-source copies.
- DARTsort worktree commit/tree: `d40c516e47451e688dd4484813b562fa5fc0b4ef` /
  `f2c80b47c9d190aac128f89a3d10010464d19840`.
- Worktree `uv.lock`: `68f3fa53642043313bc9a4e56b7fc35dbd27af81030e436ffdba82de683f5034`.
- Exact-worktree environment import resolved DARTsort from the worktree.
- Author suite: 17 passed.
- Independent suite: 11 passed, 1 failed; the sole failure is the preserved
  complex-clock rejection fixture.

No real K1 arrays, K9 output, recording voltage, H5 state, GPU, service, sort,
RF or holdout data were inspected or used.

Implementation checks
- Done: frozen source/contract/test/hash review -> all packet hashes and source copies matched.
- Done: K1 operation order and exact-reader trace equality -> first/interior/final batches, joins, subsets, sign, no-filter, FFT high-pass, lower median, padding/core crop and whole-batch artifact zeroing passed synthetic known answers.
- Done: K1 detector semantics -> strict threshold, strict-left/non-strict-right plateau alignment, refractory boundary, depth-priority suppression and sample origin passed.
- Done: K2 semantics -> all ch251/252 detections, no label parameter, acquisition-clock lag sign, inclusive support, deterministic tie rule and contiguous width agreed with an independent brute-force oracle.
- Done: adversarial dtype guard -> complex sample arrays are accepted after lossy cast; contract violation at `IMPLEMENTATION.py:192-199`.
- Not done: six real C4-0 windows, 0804 detector rate, real lag, sorter integration, performance/scale, GPU behavior or any scientific comparison; excluded by review scope.
- Can establish: K1 synthetic construction is fixture-qualified; most K2 CCG arithmetic is fixture-qualified; K2 clock-domain validation has a concrete fail-open defect.
- Cannot establish: K1 real-window bitwise equality, a real frozen lag, biological validity, sorting benefit, benchmark validity or medium-run readiness.

