# Independent full review: H1 candidate-3 re-entry metadata probe v4

## Verdict

`GO_IMPLEMENTATION_READY_METADATA_GATE_ONLY`

V4 repairs the preserved v3 concurrency defect. The fixed per-snapshot claim is
installed atomically and permanently before snapshot-2 predecessor inspection or
worker spawn. A pre-existing or concurrent claim therefore prevents a second
invocation from reaching the live metadata operation. This verdict approves the
frozen implementation for a separately authorized, coordinator-controlled use
only after every external placement precondition passes. It is not evidence that
H1 is currently responsive, does not satisfy either snapshot, and is not launch
authorization.

Reviewed identities:

- author MANIFEST `ec20d7df77b10d83a604f90812d8a750a9e48835c5f06b05f91cc7238ed862ea`
- author COMPLETE `10ff4736b9c5ff9f73548f9d5583f4e4a6cc91b1eb21f8e4e8da0537c94908a9`
- contract `80d4e2944b2a81654c1f66441434092a1927db3e30eeb1fa92094d3ef71f4676`
- runner `64c4ef0cb7a19e44cf31409c4757ac84b66cb4a69a41fe21ecd72f591055d165`
- tests `ebf12b0a042a8fa03f07bb05d2311680ac7d7b5f6495def02aff7979620c547c`
- author commit `49d273c232a5d8af2d9605c8f00e91d27dd775c4`

## Full-source findings

The packet, committed copies and shared copy agree. The manifest binds every
member except COMPLETE, and filesystem chronology places MANIFEST before
COMPLETE. The contract binds the exact runner hash and fixed local receipt paths.

The live operation remains one directory open/fstat, one immediate enumeration,
and five dirfd-relative no-follow regular-file size checks. It performs no member
open, content read, hash, glob or recursion. The five-second bound covers worker
startup through completion; timeout sends SIGKILL to the worker process group,
allows the contracted one-second reap, and cannot PASS if the worker times out or
remains unreaped.

Snapshot 2 requires the exact fixed snapshot-1 receipt, valid JSON, exact schema,
PASS status, snapshot label, contract hash, runner hash, and an integer completion
epoch at least 600 seconds earlier. The check is performed before the live worker.
A failed predecessor/gap check writes a blocked local receipt and never touches
the live target.

The v4 claim closes the v3 race. `claim_attempt` creates the fixed
`RECEIPT.json.CLAIM.json` from a fsynced same-directory temporary using atomic
`link(2)` create-if-absent, fsyncs the directory, and never removes the installed
claim. `live_probe` acquires it before predecessor validation and before
`run_worker_bounded`. The claim binds snapshot, contract hash, runner hash, PID
and claim time; every terminal receipt binds the claim path and hash. A crash or
post-claim validation failure consumes the attempt, which is the contract's
explicit fail-closed/no-automatic-retry policy.

All 16 frozen local fixtures passed independently. The concurrency fixture starts
two simultaneous claim attempts and admits exactly one owner. Other fixtures
cover exact metadata, timeout/reap, predecessor binding/gap, immutable receipt
publication, and preservation of an existing receipt. These controls do not
execute `live_probe` against `/mnt` and do not establish current host state.

## Scope and placement caveat

The runner deliberately cannot establish the contract's external preconditions:
administrator remediation, absence of relevant D-state tasks, three acceptable
vmstat samples, and absence of whole-root search or blocked sync. Those remain
conjunctive coordinator/administrator placement gates for each snapshot. Both
metadata snapshots must PASS at least 600 seconds apart, and all other H1 re-entry
gates must pass, before H1 is only eligible for a fresh managed prestart.

The disclosed pre-freeze exact-member stat is construction evidence only. This
review performed no live runner, mount probe, process/service/mount change, H5
contact, voltage/recording access, sort, RF, or holdout operation.

## Implementation checks

- Done: full source, contract, packet seals, committed copies and shared identity
  checked -> exact binding and COMPLETE-last chronology pass.
- Done: metadata call surface, clocks, timeout/reap, snapshot placement, immutable
  receipt and claim paths traced -> frozen semantics are enforced.
- Done: independent execution of frozen local tests -> 16/16 pass, including
  simultaneous claim acquisition with exactly one owner.
- Not done: live runner, mount probe, external placement observations, remediation,
  or any process/service/mount/H5/voltage/recording/sort operation -> excluded.
- Can establish: v4 is implementation-ready for the narrowly bounded metadata gate
  when separately dispatched after all external preconditions pass.
- Cannot establish: current H1 readiness, either required snapshot, throughput,
  recording readability, scientific validity, managed-prestart success, or launch
  safety/authorization.
