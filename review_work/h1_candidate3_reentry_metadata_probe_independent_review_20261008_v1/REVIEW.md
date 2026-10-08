# Independent review: H1 candidate-3 re-entry metadata probe v1

## Verdict

`NO_GO_REPAIR_PLACEMENT_RECEIPT_AND_TIMEOUT_RACE_ENFORCEMENT`

The frozen target and metadata operation are appropriately narrow, and all eight
local fixtures pass, but v1 does not implement the contract's load-bearing
two-placement timing rule or its atomic no-overwrite receipt claim. A timeout
kill race can also escape before a durable fail-closed receipt. Preserve v1 and
repair these mechanics in a distinct namespace. Do not execute v1 live.

Reviewed identities:

- author MANIFEST `54791a9105f8bd97e0607c88a695642bf2ec3cbb29d09e9441b5811f50490b34`
- author COMPLETE `59f710727ef64d7cacbdaaaa4e0b9a992163bd4ea7c94062f8ac5012054e7252`
- contract `0ee4e95e055d7f478c3128f2f3f775a056fbcec77d31f464f8887db4f8c02ed1`
- runner `356c8427009d1ceaac12d3fba64df21c3891ee3c30ad07d386a48c0a639e8f89`
- author commit `ac78a0e1a20483cf8411b12907cd3f24b00d94b0`

## Checks that pass

1. **Exact target and operation.** The hash-bound contract names one compact
   immutable directory and five exact member names/sizes. The worker opens that
   directory, fstats it, enumerates immediate names exactly once and performs
   `dir_fd`-relative `os.stat(..., follow_symlinks=False)` on only those members.
   It rejects extra/missing names, nonregular members and wrong sizes.

2. **No content or recursion.** Member files are never opened or hashed. There
   is no glob, walk or recursive traversal. Directory enumeration and metadata
   stats are the only remote operations. The source fixture replaces
   `Path.open` with a failure and still passes the metadata operation.

3. **Worker timeout scope.** The five-second `communicate` timeout begins after
   the child is spawned and covers interpreter startup, child contract and
   runner hashing, the entire metadata operation, serialization and child exit.
   Receipt durability occurs afterward on local storage. Normal timeout paths
   kill the worker process group, wait up to one second, mark timeout regardless
   of reap success and preserve PID/reap state. An unreaped worker cannot PASS.

4. **Source and contract binding.** Parent and internal worker both rehash the
   exact contract and runner before the metadata operation. Snapshot choices are
   restricted to two names and map to two fixed local receipt paths.

5. **Local fixtures.** Independent rerun passes 8/8: accepted metadata, extra
   member, wrong size, symlink, bounded success, bounded timeout/reap, atomic
   write round trip and source no-content-read check. These fixtures do not cover
   the placement or no-overwrite defects below.

6. **Disclosure.** The single pre-freeze shell stat is explicitly identified as
   construction input, lasted 0.2 seconds, did not execute the runner and is not
   counted as either re-entry snapshot. That treatment is correct.

## Blocking defects

### 1. Snapshot-2 ordering and separation are not enforced

`--snapshot snapshot_2` is accepted independently. The runner does not require
the snapshot-1 receipt to exist, verify that it is a PASS from the exact same
contract and runner, or compare a completion timestamp against a minimum
600-second interval. V1 also records only a start string, not a robust completed
time usable for this gate. Consequently snapshot 2 can be executed first or
immediately after snapshot 1 while still producing a PASS receipt. Distinct
paths limit successful labels to two but do not enforce placement semantics.

### 2. Receipt no-overwrite is not atomic

The parent checks `receipt_path.exists()` before work, but
`write_receipt_atomic` later calls `os.replace(temporary, path)`. A receipt
created between those operations is overwritten. The write is atomic for
visibility, but it is not atomic no-overwrite publication. The receipt helper
itself has no `EEXIST` guard, and the fixture tests only round-trip durability.

### 3. Timeout kill race can bypass the receipt

After `TimeoutExpired`, `os.killpg(process.pid, SIGKILL)` is unguarded. If the
worker exits between timeout detection and the kill, `ProcessLookupError` can
escape `run_worker_bounded`; `live_probe` then writes no failure receipt. Also,
the frozen `post_timeout_reap_seconds` field is not consumed—the implementation
hard-codes `1.0`. Normal timeout/unreaped behavior is fail-closed, but this race
violates the promise that every attempt preserves a local outcome.

## Minimal repair

1. For snapshot 2, require and parse snapshot 1; verify receipt schema, snapshot,
   PASS status, target, contract SHA and runner SHA; record a completed epoch in
   both receipts and require snapshot-2 start at least 600 seconds later.
2. Publish receipts with a same-filesystem atomic no-replace primitive or
   equivalent exclusive claim, fsync file and directory, and classify `EEXIST`
   as fail-closed without modifying the prior receipt.
3. Catch the timeout kill race, always perform bounded communicate/poll, consume
   `post_timeout_reap_seconds` from the contract, and preserve kill/reap details.
4. Add fixtures for snapshot-2 missing/non-PASS/wrong-binding/too-early
   predecessor, no-overwrite race and already-exited-at-kill timeout handling.

The host-remediation and process/vmstat/search/sync preconditions remain external
conjunctive gates; this metadata runner need not infer them, but its eventual
invocation receipt must not be interpreted as evidence that they passed.

## Implementation checks

- Done: packet/shared seals, exact source and contract, metadata system calls,
  timeout/reap path, receipt publication and all eight fixtures inspected.
- Done: snapshot placement and receipt lifecycle traced -> 600-second predecessor
  PASS and atomic no-overwrite are not enforced; timeout kill race is unhandled.
- Not done: live runner, mount probe, remediation, process/service/mount/H5/
  voltage/recording/sort operations -> prohibited.
- Can establish: exact narrow metadata behavior and normal-path fail-closed local
  fixture behavior.
- Cannot establish: a reusable two-snapshot re-entry gate; v1 is NO-GO.
