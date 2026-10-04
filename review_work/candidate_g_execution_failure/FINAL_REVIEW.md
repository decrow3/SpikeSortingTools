# Candidate-G continuation execution-failure review

## Verdict

`NO_GO_RETRY_MONITOR_SNAPSHOT_AND_LEDGER_FINALIZATION_REPAIR_REQUIRED`

The one authorized start executed exactly once under the reviewed v6 release and units. Release SHA-256 is `fd09fd6a45a4740fbbb9a33d915d3f99259835c63bd7746b44efc82f4e1d4f10`; the production and finalizer unit snapshots match their reviewed template hashes. Invocation `2e33135c44a3443cba39ce697f8053f7` became live with PID 2491642, retained `NRestarts=0`, and later ended status 125. The worker was killed with `-9`; no retry or second start occurred.

## Failure mechanism

The storage classifier is not stable under transient deletion. It builds `owners` from one set of recursive scans, builds `actual` in a later scan, compares the sets for equality, but reports only `actual - owners`. The observed `unclassified writable files: []` can only occur when at least one file classified in the first scan disappeared before the second. An independent deterministic fixture deleted a classified file at that boundary and reproduced the exact empty-list exception.

The supervisor correctly treated this as monitor failure, killed the worker group, and finalized partial evidence. A separate post-publication sample then failed because a read-only SQLite connection encountered the preserved rollback journal. Compact publication itself had already sealed successfully.

## Artifact scope

There is no completed or resumable scientific result. The result directory contains only `RUN_FAILED.json`, the ledger database and its journal; it has no result `COMPLETE.json`, labels, templates or post-agglomeration outputs. The frozen worker also requires an exactly empty result directory, and the one-shot contract permits no retry.

The ledger is diagnostic only. Independent immutable inspection finds 223 reservations totaling 4,895,970,352 bytes: 222 completed and one left reserved at termination. This is below the 17,179,869,184-byte cap and short of the 14,973,326,368-byte expected successful total. It does not support a completion fraction, recovered sorting, or scientific comparison.

## Prerequisite for any fresh execution

A new execution requires an execution-disabled repair that:

1. Replaces the two-scan equality check with race-safe snapshot/classification semantics. Disappearing or renamed files must not cause a false monitor failure, while any presently existing unclassified or double-counted file must still fail closed and sampled bytes must remain conservative.
2. Adds deterministic create/delete/rename concurrency fixtures plus category and aggregate-stop controls.
3. Makes terminal/post-publication ledger sampling safe with a preserved/hot SQLite journal through an explicit close/checkpoint protocol or a bounded writable database+journal snapshot, with fixtures for both clean and hot-journal states.
4. Preserves the entire current failed release, units, attempt, ledger/journal, terminal and compact evidence unchanged.
5. Uses a new attempt ID and fresh release/run/compact namespaces, freezes the repaired closure and contract, and receives a new independent GO. The current attempt cannot be continued or retried.

Implementation checks
- Done: what ran -> inspected activation preflight/materialization/start source, release, installed unit snapshots, gate/start/systemd receipts, worker command and v6 lifecycle source.
- Done: start semantics -> one start command, one invocation, zero restarts, no retry, main status 125 and worker -9 agree across receipts.
- Done: monitor semantics -> traced the asymmetric two-scan set comparison and independently reproduced the exact empty-list deletion race.
- Done: limits/counts -> last completed sample was 6,276,383 storage bytes and 3,446,938,768 logical bytes; immutable ledger recomputation was 223 rows and 4,895,970,352 bytes, below the cap.
- Done: terminal provenance -> independently verified the one-product terminal seal and four-product compact seal, including source binding and COMPLETE-last hashes.
- Done: scientific completeness -> no result seal or scientific outputs exist; partial files are limited to failure and ledger evidence.
- Done: packet provenance -> verified MANIFEST `637dc5ab3a8bb9a5ae41a01e1a69a829a9555730b22df08270fd0368af37997c`, COMPLETE `736c2ffc5095f7ba591bcf02555d7e0f9760dce0a43a44816ce781e15e1f530b`, and request `46817541274b31bb79f6a47314c960f6e063e04a842d529ed8dd7edb67b055bb`.
- Not done: voltage, incomplete sorting evaluation or biological/scientific interpretation -> prohibited and unsupported by the artifacts.
- Can establish: exact one-shot execution identity, implementation failure mechanism, bounded partial reads/storage, kill/finalizer behavior and preserved terminal provenance.
- Cannot establish: a Candidate-G sorting result, performance, completion percentage, safe in-place continuation or fitness of an unfrozen repair.

No artifact was modified and no new execution, voltage read, GPU work or scientific evaluation was performed.
