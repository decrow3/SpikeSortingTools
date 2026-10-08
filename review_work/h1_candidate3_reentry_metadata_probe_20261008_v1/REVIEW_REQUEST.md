# Review request: H1 candidate3 re-entry metadata probe v1

Proposed status: **FROZEN_LOCAL_FIXTURES_PASS_LIVE_EXECUTION_DISABLED**.

Review whether the contract and runner close the one qualification in the H1
host-readiness review. In particular, verify the exact approved path and five
member sizes, the absence of recursion and member-content reads, the aggregate
five-second worker timeout, one attempt at each of two snapshots at least 600
seconds apart, local atomic no-overwrite receipts, and the fail-closed outcome
rules. Verify that a timed-out uninterruptible worker cannot yield PASS and that
its PID/reap state remain in the receipt for administrator handling.

The live runner was not executed. One pre-freeze shell `stat` of the five exact
compact packet members was used to bind type and size and completed in 0.2
seconds; it is disclosed in `PROVENANCE.json` and is not treated as a readiness
result. All eight local fixtures passed.

The requested conclusion is only that the metadata-gate specification is
reusable after separately authorized administrator remediation. It must not
clear H1, authorize a run, imply recording readability, or substitute for the
other conjunctive process/vmstat/search/sync/prestart gates.

Implementation checks
- Done: runner-source binding and contract hash -> exact frozen files in this packet.
- Done: local known-answer, negative, timeout, receipt and no-content-read fixtures -> 8/8 pass.
- Done: target and operation -> one compact immutable project directory, one enumeration, five no-follow stats, no file-content opens.
- Not done: live runner, administrator remediation, two ten-minute-separated snapshots, kernel/server diagnosis, H5 state, managed prestart -> outside this preparation task.
- Can establish: the proposed gate is exact, bounded and fail-closed in local fixtures.
- Cannot establish: current or future H1 readiness, live syscall interruptibility, H5 readiness, recording access, or scientific performance.
