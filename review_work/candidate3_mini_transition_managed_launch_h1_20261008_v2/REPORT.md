# Candidate-3 mini transition managed-launch closure v2

Verdict: `FROZEN_AWAITING_INDEPENDENT_REVIEW_DO_NOT_START`.

This fresh v2 closes the five launch blockers in the preserved v1 review without
changing the synthetic input, window, preprocessing boundary, DARTsort source,
resolved sorter settings, or scientific interpretation.

1. The exact systemd argv now executes only immutable shared packet source.
2. A stdlib-only runner verifies its exact contract hash, all packet source
   hashes, input/native ordinary hashes, DARTsort commit, freshness,
   nonsymlink/containment, and free-space condition before creating the attempt.
   It writes `PRELAUNCH.json` before spawning DARTsort.
3. The runner handles termination, owns the child process group, writes
   `FINAL_STATUS.json`, and the frozen `ExecStopPost` finalizer writes a separate
   manager-terminal receipt from systemd's `SERVICE_RESULT`, `EXIT_CODE`, and
   `EXIT_STATUS`, including when the main wrapper cannot finish.
4. The Numba cache is inside the fresh attempt root. A managed two-second
   monitor counts all owned regular files plus the frozen retained
   ordinary/input baseline and terminates the child on a 512 MiB cap breach.
   This is a monitored stop bound, not a filesystem quota; a write may overshoot
   between checks and the observed byte count is preserved as failure evidence.
5. On transition success only, the runner creates a split-root composition from
   symlinks to the immutable ordinary and fresh transition roots, then invokes
   the frozen `e627...` reporter to reload both native finals and emit compact
   `ACCOUNTING.json` and `ACCOUNTING.npz`. Reporting failure makes the service
   fail and is not retried.

The exact producer remains the recovered `5d55...` source. The reporter is the
post-run `e627...` source independently shown to reproduce the sealed ordinary
accounting byte-for-byte. Five focused tests cover valid preflight, symlink
refusal, symlink-excluding byte accounting, cap termination, and separate
manager-final receipt; all pass.

Implementation checks
- Done: v1 independent blocker packet bound -> all five findings mapped to executable v2 behavior.
- Done: producer/reporter/runner snapshots frozen and hash-bound in the run contract.
- Done: focused runner tests -> 5 passed in 2.07 seconds.
- Done: no service/output/failure receipt exists and no launch occurred while authoring v2.
- Not done: independent v2 source/contract/argv review -> mandatory before start.
- Not done: transition execution and downstream accounting -> occur only after independent GO and exact prestart reconciliation.
- Can establish: a managed, bounded synthetic transition capability result if the reviewed service completes every stage.
- Cannot establish: real-input readiness, benefit, biological identity, purity, causal attribution, or full-session behavior.
