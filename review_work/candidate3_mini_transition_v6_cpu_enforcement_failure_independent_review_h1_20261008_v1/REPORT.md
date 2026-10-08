# Candidate-3 v6 CPU-enforcement failure independent review and correction

Verdict: `FAILED_RESOURCE_ENFORCEMENT_NO_RETRY`.

This packet corrects and supersedes only the live-limit conclusion in the earlier v4 independent GO review (MANIFEST `c85374eb...`, COMPLETE `97d9dc95...`). That review incorrectly treated the matching systemd `CPUQuotaPerSecUSec=8s` property as proof that the kernel enforced the 800% CPU bound.

During the exact v6 run, the leaf cgroup had no `cpu.max` file and the parent `app.slice/cgroup.subtree_control` contained only `memory pids`, not `cpu`. Thus the CPU controller was not delegated to the user-service subtree and the declared quota was not enforced. Live `cpu.stat` showed 11,292.822 CPU seconds after roughly ten minutes; terminal systemd accounting recorded 11,925.237 CPU seconds over about 680.25 seconds from prelaunch to failure—an average 17.53 cores, 2.19 times the intended eight-core ceiling. This is a concrete uncontrolled-execution failure, not a formatting caveat.

The coordinator stopped the unit at 01:26:11 PDT. Failure preservation worked correctly: `FINAL_STATUS.json` is `failed` at transition with return code `-15`; `MANAGER_FINAL.json` records `exit-code`, status `241`, the matching invocation, timestamp, and final-status presence; systemd records `NRestarts=0`. No accounting report was produced and no scientific outcome is accepted. The partial v6 attempt is preserved and must not be reused or overwritten.

The outer receipt, exact prelaunch hashes/baseline, cgroup identity, memory/task properties, signal handling, and terminal finalizer all behaved as designed. The defect is specifically that checking systemd's configured CPU property did not verify the effective kernel controller/file.

Implementation checks
- Done: configured versus effective CPU limit -> `CPUQuotaPerSecUSec=8s` was present, but leaf `cpu.max` was absent and parent delegation omitted `cpu`.
- Done: independent utilization cross-check -> terminal CPU time implies 17.53 average cores and 2.19x the frozen ceiling.
- Done: failure closure -> transition stopped by SIGTERM; timestamped FINAL_STATUS and MANAGER_FINAL agree; no restart.
- Done: outcome closure -> transition incomplete, reporter never ran, and no accounting/scientific conclusion is available.
- Not done: retry or alternative launcher -> prohibited for this failed attempt and outside the review.
- Can establish: the CPU bound was not enforced and the stopped attempt is invalid for completion/science.
- Cannot establish: transition capability, comparative accounting, or readiness for any larger input.

Required decision: preserve v6, `NO_RETRY`, and stop the custom launch-design repair chain. Any future managed contract must either verify the actual leaf controller and `cpu.max` value before work, run under a manager with CPU-controller delegation, or explicitly freeze CPU as an unbounded documented resource condition with separate authorization.
