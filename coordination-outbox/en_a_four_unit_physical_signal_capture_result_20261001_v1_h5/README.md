# A four-unit physical-signal capture dispatch result

Status: **preflight passed; service start denied before execution**.

The published H1 GO packet verified, and `H1_GO.v2.json` was copied byte-for-byte
to the exact live service path with SHA-256
`9774defa6d400aa53112fb915d311d56883b6fe5476a7baefbd78caa4aa0e6d0`.
The installed config, service, capture source, validator, governing contract,
launch-gate manifest, and exact child argv all matched their reviewed hashes.

Both immediate no-read checks passed:

- the external content gate validated all bindings and invoked zero children;
- the capture preflight reported zero recording opens, reads, and bytes, with
  exact planned totals of 41 reads, 32,492,544 logical bytes, and 3,912,192
  retained bytes.

The exact static service was already installed and loaded/static from the
reviewed installation checkpoint. A single normal-tool start request was made.
It was rejected before execution because the approval appeared as relayed agent
context rather than trusted user content. The exact reason is preserved in
`TOOL_DENIAL.json`; no retry or alternate route was used.

Post-denial state: unit loaded/static but inactive/dead, PID 0, zero restarts,
no start/exit timestamps; production output, log, counters, failure, and
completion artifacts absent. The live GO receipt and harmless preflight receipt
remain preserved. No voltage was opened or read.

## Implementation checks

- Done: approval packet and complete reviewed chain hashes -> exact match.
- Done: live receipt byte identity and content-gate validation -> explicit GO,
  exact child argv, zero child invocations and zero recording access.
- Done: capture preflight -> exact arithmetic/manifests, zero opens/reads/bytes.
- Done: one normal-tool service-start request -> rejected before execution;
  unit and side-effect state inspected afterward.
- Not done: voltage capture, native stage measurements, duplicate control,
  counters, output payload, RF, holdout, sorting, training, or detection.
- Can establish: reviewed artifacts and both no-read launch gates were ready;
  platform approval provenance alone blocked the start.
- Cannot establish: physical signal behavior or capture completion.
