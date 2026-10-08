# Review request: H1 candidate3 re-entry metadata probe v3

Proposed status: **GO_REUSABLE_POST_REMEDIATION_GATE_LIVE_EXECUTION_DISABLED**.

V1 and v2 remain preserved NO-GOs. V1 did not enforce the snapshot predecessor
or gap. V2 repaired placement, timeout kill-race handling, and consumption of
the reap bound, but retained a check-then-`os.replace` receipt race. V3 changes
only receipt installation to a same-directory fsynced temporary plus atomic
`link(2)` create-if-absent, followed by directory fsync and temporary unlink.
An adversarial fixture verifies a pre-existing receipt remains byte-identical.

Please check the full effective v3, not only the delta: exact compact target and
members, zero member-content reads, no recursion, five-second worker timeout,
one-second cleanup, timeout/unreaped-worker failure, bound PASS predecessor and
600-second gap before snapshot 2 touches `/mnt`, local fail-closed receipts, and
source/contract binding. Fourteen local fixtures pass. No live runner or new
live mount lookup occurred.

Implementation checks
- Done: v1/v2 defects -> preserved with explicit NO-GO provenance.
- Done: v3 receipt install -> atomic create-if-absent; existing receipt preservation fixture passes.
- Done: full local suite -> 14/14 pass.
- Not done: live runner, remediation, real snapshots, kernel/server diagnosis, H5 or managed prestart -> outside scope.
- Can establish: v3 locally implements the exact bounded re-entry metadata-gate policy.
- Cannot establish: live syscall interruptibility, H1/H5 readiness, recording readability, launch safety or scientific performance.
