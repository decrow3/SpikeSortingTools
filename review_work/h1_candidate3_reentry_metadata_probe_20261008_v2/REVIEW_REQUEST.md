# Review request: H1 candidate3 re-entry metadata probe v2

Proposed status: **FROZEN_V2_LOCAL_FIXTURES_PASS_LIVE_EXECUTION_DISABLED**.

V1 is preserved as `NO_GO_SNAPSHOT_ORDER_AND_GAP_NOT_ENFORCED_BY_RUNNER`.
Although its contract required snapshot 2 at least 600 seconds after snapshot 1,
the implementation accepted either snapshot name independently. V2 repairs that
load-bearing defect before any live runner execution.

Please verify that snapshot 2 cannot touch the live target unless the fixed
snapshot-1 receipt exists, is `PASS_METADATA_GATE`, binds this exact contract
and runner, and completed at least 600 seconds earlier. A failed ordering/gap
precondition must write a local fail-closed snapshot-2 receipt without spawning
the metadata worker. Also check the timeout process-exit race and contracted
one-second reap interval.

All v1 restrictions remain: exact compact project path and five members; one
nonrecursive enumeration plus five no-follow metadata stats; no member-content
reads; five-second aggregate operation timeout; atomic local no-overwrite
receipts; at most one attempt per snapshot and no retry; all other re-entry
gates remain conjunctive. Thirteen local fixtures pass. The live runner was not
executed and v2 performed no new live mount lookup.

Implementation checks
- Done: v1 defect preserved -> sealed v1 packet and commit `ac78a0e` remain unchanged.
- Done: v2 delta -> predecessor PASS/binding/gap are enforced before worker spawn; timeout cleanup consumes the contract field.
- Done: local fixtures -> 13/13 pass, including four adversarial predecessor variants.
- Not done: live runner, remediation, real ten-minute snapshots, kernel/server diagnosis, H5, managed prestart -> outside scope.
- Can establish: v2 locally enforces the exact operation, repetition and placement policy.
- Cannot establish: live syscall interruptibility, current/future H1 readiness, recording readability, H5 readiness or launch safety.
