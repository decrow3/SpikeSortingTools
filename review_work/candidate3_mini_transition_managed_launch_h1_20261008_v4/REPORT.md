# Candidate-3 mini transition final launch closure v4

Verdict: `FROZEN_AWAITING_CHANGED_BOUNDARY_REVIEW_DO_NOT_START`.

This is the final narrow closure under the coordinator's fixed checklist. It
reuses every accepted unchanged v3 provenance/resource boundary and changes only
outer launch evidence, actual manager-context verification, timestamped terminal
evidence, and consumption-time tree revalidation. No input, producer, reporter,
DARTsort setting, scientific window, resource limit, or interpretation changed.

## Final blocking checklist

1. **Actual consumed bytes.** The accepted complete manifests still bind all
   four input and 22 ordinary files by membership, size, and SHA-256, with exact
   baseline `185,292,925` bytes and a clean `edcfe1b5...` DARTsort tree. The
   runner now repeats this full verification immediately before and after the
   reporter consumes those mutable `/tmp` trees. Either mismatch writes a
   timestamped failed `FINAL_STATUS.json`; no result is accepted.
2. **Actual launch context and limits.** The user-facing argv invokes immutable
   shared `launch` source, which atomically writes `OUTER_LAUNCH.json` before
   calling the exact frozen `systemd-run` argv and records return code, stdout,
   stderr, and finish time afterward. The managed main verifies its expected
   cgroup, `INVOCATION_ID`, unit, and actual Type/Restart/runtime/memory/CPU/task/
   kill/stop properties before creating the attempt.
3. **Reliable stop/failure/completion evidence.** Launching, prelaunch, stage
   failure, success, and manager-final receipts are timestamped. Signal handling
   and the 512 MiB monitor remain unchanged. `ExecStopPost` writes systemd result,
   exit tuple, final-status presence, and a full terminal property snapshot (or
   its exact query error). A launcher interruption leaves an atomic `launching`
   receipt for supervisor reconciliation; systemd owns any accepted service.
4. **Valid accounting report.** The accepted split-root reporter remains frozen
   at `e627...`, reloads both native finals, and emits compact JSON/NPZ. Reporting
   or either surrounding tree revalidation failure makes the managed run fail.

Seven focused tests pass, including durable outer systemd failure capture. A
30-second no-data systemd property probe independently confirmed the exact
manager value representations frozen in the contract. The v6 unit, output, and
three external receipts remain absent. No transition was launched.

Implementation checks
- Done: v3 accepted tree/baseline/clean-worktree boundaries reused unchanged.
- Done: remaining v3 wrong-execution paths closed by atomic outer receipt, actual manager verification, timestamps/terminal snapshot, and before/after consumption revalidation.
- Done: seven focused tests passed in 2.16 seconds.
- Not done: changed-boundary independent review and normal exact prestart -> required before one launch.
- Can establish: controlled execution and valid compact accounting for the missing synthetic transition arm if all frozen gates pass.
- Cannot establish: real-input readiness, benefit, identity, purity, causal attribution, or full-session behavior.

Any further launch blocker must identify a concrete wrong-science or uncontrolled-
execution failure in these four checklist items. Optional metadata or formatting
preferences are nonblocking. If this changed-boundary review fails, stop the
custom launch-design repair chain and return the coordinator's minimal fallback.
