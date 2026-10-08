# Candidate-3 mini transition managed-launch v4 independent review

Verdict: `GO_EXACT_V4_ONE_SYNTHETIC_TRANSITION_LAUNCH` after the frozen normal prestart reconciliation. This review does not launch the service.

The final four changed boundaries are sufficient and fail closed:

1. The immutable `launch` mode writes an atomic timestamped outer receipt before invoking the exact contract-rendered `systemd-run` argv, then records its return code/stdout/stderr and finish time. An interrupted `launching` state is explicitly reconcilable against the unit; it is not treated as success by an operator without that reconciliation.
2. Before creating the attempt, managed `run` requires the expected unit substring in its own cgroup, a matching `INVOCATION_ID`, active manager state, and exact live Type/Restart/runtime/memory/CPU/task/kill/stop properties. A wrong manager context or effective limit stops before scientific output.
3. All stage/status receipts now have UTC timestamps. `ExecStopPost` persists the systemd result/exit tuple, final-status presence, and terminal property snapshot or exact query error. A query error is preserved evidence, not a successful completion; acceptance requires a successful snapshot.
4. The accepted exact input and ordinary tree manifests are reverified immediately before and after the reporter. Either drift writes a failed terminal status, so accounting is not accepted from changed retained inputs.

Independent checks: all six packet members match MANIFEST `ac55b926...`; COMPLETE `4d8e4ca6...` binds it and is last. The exact RUN_CONTRACT is `db34b9b9...`. Seven focused tests pass in 2.13 seconds, including durable outer failure capture. Full retained-tree preflight independently passes with exact 57,081,256 + 128,211,669 = 185,292,925 bytes and a clean `edcfe1b5...` DARTsort checkout. The v6 unit and all four attempt/external receipt paths are absent; no launch occurred.

Implementation checks
- Done: outer launch path -> exact argv is rendered from the hash-bound contract and both acceptance and failure leave an atomic receipt.
- Done: live manager/resource boundary -> cgroup, invocation, state, and all frozen effective properties are checked before attempt creation.
- Done: terminal closure -> timestamped inner status plus ExecStopPost systemd snapshot; post-run acceptance explicitly rejects missing/error evidence.
- Done: consumption provenance -> full tree hashes run at preflight and immediately before/after reporting.
- Done: unchanged v3 provenance/resource gates -> reused from independent v3 review rather than serially redesigning them.
- Not done: service launch or scientific outcome -> outside this code-review task and still pending exact prestart.
- Can establish: controlled execution and valid compact accounting for one missing synthetic transition arm if every frozen gate and post-run condition passes.
- Cannot establish: real-input readiness, scientific benefit, biological identity/purity, causal attribution, or full-session behavior.

Release scope: only the exact v4 `LAUNCH_REQUEST.json` against fresh v6 paths. Do not retry or alter parameters under this GO.
