# Candidate-3 miniature transition replacement contract v5

Status: `FROZEN_AWAITING_RESOURCE_DELTA_REVIEW_DO_NOT_START`.

This packet requests one fresh replacement attempt in `/tmp/candidate3_mini_e2e_20261008_v7`. It does not retry or reinterpret invalid v6. V6 remains preserved under failure MANIFEST `63b038...` and independent correction MANIFEST `05d090...`.

The only intended delta is resource policy and its live verification. The ineffective `CPUQuota=800%` request is removed. CPU is explicitly `NO_ENFORCED_QUOTA`: all 20 allowed online H1 CPUs may be used, load averages are recorded, and there is no timing-comparability or bitwise-determinism claim. Prestart must show no other H1 spike sort and at least 32 GiB host memory available.

Inside the real service, before the attempt root or scientific child is created, the runner now verifies the actual leaf `memory.max=17179869184` and `pids.max=128`, rejects any unexpected finite `cpu.max`, records allowed/online CPUs, host memory and load, and retains the exact manager invocation/property checks. The 45-minute manager timer, 30-second control-group stop, `ExecStopPost` finalizer, 512 MiB aggregate storage monitor and fail-closed receipts are unchanged. V6's actual controlled stop demonstrated SIGTERM propagation and finalizer preservation; focused tests and the earlier property probe cover the unchanged paths.

Producer SHA `5d55bb...`, reporter SHA `e627b1...`, input and ordinary complete-tree manifests, DARTsort commit `edcfe1b5...`, seed, native config, lineage/accounting semantics, 512 MiB cap and no-retry rule are unchanged.

## Launch gate

An independent local reviewer must accept only this resource/launch delta and confirm no genuine scientific prerequisite reopened. Immediately before launch: rehash the packet, input and ordinary trees; confirm v7 service/output/receipts absent; confirm no other H1 spike sort; capture allowed/online CPUs, host memory/load and storage headroom. The in-service checks must pass before any scientific child starts. If they fail, preserve the receipt and do not retry.

## Implementation checks

- Done: science/input equality -> producer, reporter, exact input/ordinary manifests, native config/seed and DARTsort commit are byte-identical to reviewed v4/v6 bindings.
- Done: effective resource delta -> code reads the real leaf cgroup files before creating the attempt and explicitly treats CPU as unbounded.
- Done: stop/finalizer evidence -> invalid v6 was actually terminated through the same unchanged signal/finalizer path with both terminal receipts preserved.
- Not done: v7 live effective-resource check -> necessarily occurs inside the fresh service and must pass before scientific work.
- Not done: transition result -> no v7 launch before independent GO and fresh reconciliation.
- Can establish: a focused, prospectively frozen replacement contract with honest CPU semantics and unchanged science.
- Cannot establish: completion, timing comparability, bitwise determinism, transition capability or any scientific result before the one replacement finishes and is reviewed.
