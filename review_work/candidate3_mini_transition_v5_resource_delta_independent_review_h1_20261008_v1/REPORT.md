# Candidate-3 mini transition v5 resource-delta independent review

Verdict: `GO_EXACT_V5_ONE_V7_REPLACEMENT_AFTER_FRESH_PRESTART`.

This review releases only the exact immutable v5 packet (MANIFEST `87f741047d51e28d2bcf2adf2d8b646231566fa7bf82f8184c9826a5d86133c2`, contract `782f00f8798ce3758c152b00dd5a749efc07f0a89c044bd6b2dc4266d474549d`) for one fresh synthetic transition attempt in the v7 namespace. It does not reinterpret, resume, or authorize reuse of invalid v6.

The CPU policy is now honest and prospective: v5 makes no CPU-limit request, states `NO_ENFORCED_QUOTA`, rejects any finite live-leaf `cpu.max`, records process affinity, online CPUs, host memory, and load, and disclaims timing comparability and bitwise determinism. This is a resource-policy change, not a scientific change. The exact producer (`5d55bb1e...`), reporter (`e627b1aa...`), complete input/ordinary manifests, DARTsort commit, transition command, native config/seed, accounting semantics, and 512 MiB aggregate cap remain unchanged from reviewed v4/v6.

The new checks run before the attempt root or scientific child is created. They require the real leaf `memory.max=17179869184`, `pids.max=128`, no finite `cpu.max`, at least 20 CPUs in the runner's actual affinity, and at least 32 GiB host `MemAvailable`. The existing in-unit checks still bind the invocation and active manager plus Type, no-restart, 45-minute runtime, memory/task settings, control-group kill mode, and 30-second stop timeout. V6 independently demonstrated propagation of a manager stop to the child, failed terminal status, `ExecStopPost` finalization, and zero restart. The 2-second aggregate-output monitor and full-tree pre/report/post bindings are byte-identical unchanged code paths.

The launch remains conditional on a fresh operator reconciliation. Immediately before the single launch, all of the following must hold:

1. Rehash v5 MANIFEST and every member; rehash the full four-file input and 22-file ordinary trees; confirm clean DARTsort commit `edcfe1b51d672b4136eb13cc78c0875da804b851`.
2. Confirm the v7 unit, attempt root, outer receipt, preflight-failure receipt, and manager-final receipt are all absent; do not remove or overwrite anything to make this pass.
3. Confirm no other H1 spike sort or equivalent high-load scientific job is active, all 20 H1 CPUs are online and allowed to the service context, host `MemAvailable >= 34359738368`, current load is compatible with exclusive spike-sort use, and `/tmp` free space remains at least 2 GiB. Record the process/service scan, CPU online/affinity, memory, load, and storage values.
4. The in-service PRELAUNCH must, before creating `/tmp/candidate3_mini_e2e_20261008_v7`, record the matching invocation and manager properties plus live leaf `memory.max=17179869184`, `pids.max=128`, and absent or `max PERIOD` `cpu.max`; its allowed CPU count must be at least 20 and its host memory gate must pass.
5. If any gate fails, preserve the failure receipt and do not retry. Acceptance after execution additionally requires outer `started`/rc0, complete FINAL_STATUS with both retained-tree revalidations and accounting paths, successful MANAGER_FINAL with zero restarts, aggregate bytes at or below 512 MiB, and an implementation review of the result.

No genuine scientific prerequisite is reopened for this narrow synthetic transition capability check. The prior native-output review supports only miniature ordinary-arm stage capability and explicitly left the transition arm unestablished; v7 is the already-planned missing arm under unchanged science. The result still cannot establish real-input behavior, scientific benefit, biological identity or purity, causal attribution, timing equivalence, bitwise determinism, or full-session readiness.

## Implementation checks

- Done: what changed -> exact v4-to-v5 diff adds only live resource-policy checks, replacement provenance, fresh v7 paths, and removal of `CPUQuota=800%`; producer and reporter hashes are unchanged.
- Done: configured versus effective resources -> source reads the runner's actual cgroup-v2 leaf before attempt creation and checks effective memory and PID files; CPU is explicitly unbounded and any unexpected finite `cpu.max` fails closed.
- Done: launch/stop/output safeguards -> exact source ordering, v6 terminal receipts, and prior v4 review show invocation binding, no restart, manager timer/property checks, control-group termination/finalizer preservation, and unchanged 2-second aggregate-byte monitoring.
- Done: science/input equality -> complete contract manifests, producer/reporter hashes, DARTsort commit, transition/report commands, native configuration/seed lineage, and accounting boundary are unchanged.
- Not done: live v7 resource observation -> necessarily occurs inside the fresh service and is a mandatory pre-science gate.
- Not done: v7 execution or result -> outside this review; no service was launched or mutated.
- Can establish: exact v5 is a fail-closed, source-bound contract for one synthetic replacement with CPU explicitly unbounded and memory/PID/output/time protections retained.
- Cannot establish: v7 completion or scientific outcome, timing comparability, bitwise determinism, transition benefit, real-recording behavior, biological identity/purity, or larger-run readiness.
