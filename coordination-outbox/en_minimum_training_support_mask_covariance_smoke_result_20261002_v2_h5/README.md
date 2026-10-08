# Explicit-approval sparse covariance/W dispatch result

Status: **not executed — normal tool review again denied before command
execution**.

The coordinator relayed that the human user directly answered “Yes approved”
after receiving the exact recording, crop, batch schedule, 554,084,352-byte
padded-read ceiling, 33,554,432-byte retained-output ceiling, stop boundary,
one-start/no-retry rule, and absence of runtime/CPU/RAM/GPU-utilization caps.

The immediate mutable-host recheck was repeated and persisted before dispatch:
both immutable packet manifests and every frozen runtime/source hash matched;
the metadata-only preflight saw CUDA and the reviewed RTX A5000 without opening
recording content; the service file, loaded unit, run root, results, evidence,
and reservation were absent.

The exact install/daemon-reload/one-start request was then submitted once. The
normal tool rejected it because the approval appeared only as relayed agent
history rather than trusted user content. The exact reason is preserved in
`TOOL_DENIAL.json`. Per contract, no retry or alternate route was attempted.

Post-denial state remains unchanged: service file absent; unit not found and
inactive/dead with PID 0 and zero restarts; all output/reservation paths absent;
recording content unopened and read bytes zero. There is no covariance, W,
journal, artifact inventory, or scientific result.

## Implementation checks

- Done: repeated packet, runtime, source, CUDA, metadata, and namespace checks ->
  all exact and ready before the requested start.
- Done: exact normal-tool install/start request -> rejected before execution;
  denial text and unchanged state persisted.
- Done: no retry, alternate launcher, service installation, process, namespace,
  voltage open, or byte read occurred.
- Not done: real C/W, rank/condition, W finiteness/hash, post-W stop, training,
  detection, RF, or holdout.
- Can establish: the second platform denial is specifically about approval
  provenance, not a mutable-host or contract mismatch.
- Cannot establish: smoke execution or scientific validity.
