# Independent full review: H1 candidate-3 re-entry metadata probe v3

## Verdict

`NO_GO_REPAIR_ATOMIC_PRESPAWN_ATTEMPT_CLAIM`

V3 correctly repairs the v1 and v2 defects in predecessor placement, timeout
handling and immutable receipt publication. It remains unsuitable for live use
because exclusivity is acquired only after the metadata worker finishes. Two
concurrent invocations can both perform the live probe, violating the frozen
one-worker/one-attempt resource bound, even though only one final receipt wins.

Reviewed identities:

- author MANIFEST `873a9773e15be22fba1e0bdd5a20f740f188a4be86bb11c12c04f5e31065d597`
- author COMPLETE `7888476d31f38b33f82a4bd3614f3faf17f31ad1bf77d6434be4f8df4c05a21a`
- contract `c5fa62db94da2d81b63bf30050d86e8abbbf295a7dcf3e41913d1f59e7d372a4`
- runner `0662fb11c131a27d485d112470e03bdd36c2255384d58ffa02c1c2f4e90a26f8`
- author commit `7d49afbbc5948fa9d3f549a8c97dc62f60e52ff8`

## Full-source checks that pass

The runner and contract are exactly bound. The live operation remains one exact
directory open/fstat/listing and five no-follow regular-file size checks, with no
member open/hash/content read, glob or recursion. The five-second worker timeout
covers child startup through exit; the contracted one-second post-timeout reap is
consumed; `ProcessLookupError` during kill is handled; timeout and unreaped states
cannot PASS and retain PID/reap evidence.

Snapshot 2 now requires the fixed snapshot-1 receipt to exist and be valid JSON,
with exact receipt schema, PASS status, snapshot label, contract hash, runner hash
and integer completion epoch. It enforces at least 600 seconds from predecessor
completion before the live worker starts. A failed precondition writes a local
blocked receipt and performs no mount probe.

Receipt bytes are written to a same-directory temporary file, flushed and
fsynced. `link(2)` then installs the final path only if absent, the directory is
fsynced and the temporary is removed. This is atomic no-overwrite publication;
the adversarial existing-receipt test confirms prior bytes remain identical.
All 14 author fixtures pass independently.

The single pre-freeze stat remains disclosed construction evidence, not either
snapshot. No live runner or mount operation was performed in this review.

## Blocking concurrency defect

`live_probe` checks `receipt_path.exists()` near entry, but does not acquire an
exclusive claim. Snapshot-2 predecessor checking and `run_worker_bounded` occur
before `write_receipt_atomic` calls `os.link`. Therefore two processes can both
observe an absent receipt and both spawn metadata workers. The final link makes
one receipt win and correctly gives the loser `BLOCKED_RECEIPT_WRITE`, but it
cannot undo the second live attempt.

An independent local control mocked `run_worker_bounded` so no live target was
accessed, synchronized two simultaneous snapshot-1 calls at the worker boundary,
and observed two worker calls with exit codes 0 and 30. Exactly one PASS receipt
remained and was not overwritten. This demonstrates that receipt integrity is
fixed while the contract's `concurrency: 1` and “at most one worker per declared
snapshot” constraints are not enforced. The 14 fixtures have no concurrent-start
case.

## Minimal repair

Before predecessor validation or worker spawn, atomically create and fsync a
fixed local per-snapshot immutable attempt-claim artifact using create-if-absent.
Any existing claim or final receipt must block before live access. Preserve the
claim on every success, failure and crash path, bind the final receipt to its
claim identity/hash, and do not permit automatic stale-claim recovery. Add a
two-invocation fixture proving exactly one call can cross the worker boundary.
Keep all v3 metadata, placement, timeout and receipt semantics unchanged.

## Implementation checks

- Done: complete source/contract, packet seals, metadata calls, placement logic,
  timeout/reap logic and receipt installation inspected; 14/14 fixtures pass.
- Done: mocked concurrent-start control with no live access -> two workers cross
  the boundary before one receipt publication loses.
- Not done: live runner, mount probe, remediation or any process/service/mount/
  H5/voltage/recording/sort operation -> prohibited.
- Can establish: v3 repairs earlier evidence integrity and placement defects.
- Cannot establish: the frozen single-attempt/concurrency bound; v3 remains NO-GO.
