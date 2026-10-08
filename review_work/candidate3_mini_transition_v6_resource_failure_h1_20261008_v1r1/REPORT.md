# Candidate-3 miniature transition v6 resource-contract failure

## Verdict

`FAILED_STOPPED_RESOURCE_CONTRACT_NOT_ENFORCED`

The one authorized synthetic transition launch was stopped at 01:26:11 PDT. Although `systemctl show` reported the requested `CPUQuotaPerSecUSec=8s`, the live leaf cgroup had no `cpu.max`, and its parent delegated only `memory pids`. The attempt accumulated 11,925.237 CPU-seconds over approximately 680 wall-seconds (17.54 average cores), exceeding the intended eight-core bound.

The stop was therefore required by the frozen resource condition. The managed worker preserved `FINAL_STATUS.json` as failed at the transition stage with child return code -15. `ExecStopPost` preserved `MANAGER_FINAL.json`; systemd ended failed/exit-code, status 241, with zero restarts. The partial native tree is retained in `/tmp/candidate3_mini_e2e_20261008_v6`.

No `matching1.h5`, final sorting, `ACCOUNTING.json`, or `ACCOUNTING.npz` exists. This attempt provides no transition scientific result and cannot support an ordinary-versus-transition conclusion.

No retry is authorized. Per the coordinator's prior stop rule, the custom-launch repair chain ends here. Any future transition demonstration must use an existing reviewed launcher or supervised bounded execution. If CPU is intended to be bounded, preflight must verify controller delegation and the actual leaf `cpu.max`, not only systemd's requested property. Alternatively, a future frozen contract must explicitly document CPU as an unbounded resource condition before launch.

## Implementation checks

- Done: what actually ran -> exact v4 contract hash, outer receipt, in-service prelaunch receipt, unit, invocation, source/config bindings, partial native tree, worker final status and manager final status were inspected.
- Done: caps and manager enforcement -> requested systemd CPU property matched the contract, but the live kernel cgroup lacked `cpu.max`; memory and pids limits were present.
- Done: completion and provenance -> the run was stopped, zero restarts were observed, both failure receipts were preserved, and the compact receipt/config/log hashes are recorded.
- Not done: transition matching, final sorting, split-root accounting or ordinary-versus-transition comparison -> the run ended before those artifacts existed.
- Can establish: v6 was an incomplete synthetic transition attempt whose intended CPU bound was not kernel-enforced; stopping it preserved the failed-run evidence.
- Cannot establish: transition stage capability through final output, an ordinary-versus-transition difference, any real-recording behavior, biological identity, purity, causal attribution or full-session readiness.
