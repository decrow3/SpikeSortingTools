# Review request: H1 candidate3 re-entry metadata probe v4

Proposed status: **GO_REUSABLE_POST_REMEDIATION_GATE_LIVE_EXECUTION_DISABLED**.

V3 is preserved NO-GO because two concurrent invocations could both spawn a
worker before one lost atomic receipt installation. V4 atomically creates and
fsyncs a permanent per-snapshot `RECEIPT.json.CLAIM.json` before the snapshot-2
predecessor check or worker spawn. The claim binds contract, runner, snapshot,
PID and time. Existing or concurrent claims block before live access. Claims
are never removed; every final receipt binds its claim path and hash.

Please check full effective v4. Sixteen local fixtures pass, including exactly
one owner across two simultaneous claim attempts. All earlier controls for the
exact nonrecursive metadata operation, zero content reads, timeout/reap,
snapshot-1 PASS binding, 600-second gap, and atomic no-overwrite receipt remain.
No live runner or new live mount lookup occurred.

Implementation checks
- Done: v1-v3 failures -> preserved and named.
- Done: pre-spawn ownership -> permanent atomic claim; sequential and simultaneous adversarial fixtures pass.
- Done: full local suite -> 16/16 pass.
- Not done: live runner, remediation, real snapshots, H5 or managed prestart -> outside scope.
- Can establish: v4 locally enforces one owner per snapshot and the complete bounded metadata-gate policy.
- Cannot establish: live syscall interruptibility, H1/H5 readiness, recording readability, launch safety or science.
