# Independent implementation review: joint-model K1/K2 fixture v1

Proposed verdict: **GO_SYNTHETIC_FIXTURE_QUALIFIED**.

Review the full exact reader and implementation, not only the passing tests.
Trace K1 operation order and verify returned traces are bitwise identical to
the unmodified reader while HP(median) is the actual removed lower median with
the same high-pass, batch/core boundary and artifact-zero semantics. Exercise
first/interior/final batches, joins, channel subsets, sign, no-filter and
artifact paths with an independent known-answer fixture.

For K2, confirm the interface cannot accept unit/cluster labels, selects every
detection on channel IDs 251/252, uses integer acquisition clocks, defines lag
as detection minus pulse, searches the frozen inclusive ±1.5 ms support, and
reports a deterministic peak and contiguous half-max width. Check refractory
NMS, threshold strictness, plateau alignment, ties, clock failure and an
independent brute-force CCG oracle.

Do not inspect real K1 arrays, K9 output, recording voltage, H5, GPU, services,
sorts, RF or holdout. Preserve failures and publish a separate immutable review.

Implementation checks
- Done: exact baseline bundle, commit/tree, reader and lock hashes verified; exact worktree import proven.
- Done: frozen source/contract/tests; author suite 17/17 passes on synthetic bounded inputs.
- Not done: six C4-0 real windows, 0804 detector rate, real lag, sorter integration or scale.
- Can establish: K1/K2 construction on synthetic fixtures bound to the exact reader.
- Cannot establish: real-window equality, biological validity, sorting benefit or launch readiness.
