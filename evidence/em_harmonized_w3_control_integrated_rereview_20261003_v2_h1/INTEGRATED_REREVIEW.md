# H1 integrated rereview — input-harmonized W3 control v2

## Decision

**NO_GO_LIFECYCLE_REPAIR_REQUIRED. Do not launch.**

V2 resolves the three v1 implementation findings at the component level:

- the scorecard helper closure is packet-local and hash-enforced before import;
- the one-shot score/seal worker validates the terminal sort, preserves scoring
  failure, prevents overwrite/retry, and writes `COMPLETE.json` last;
- the immediate release gate verifies manifest-bound H1/coordinator receipts,
  absent outputs, 25 GiB disk, 160 GiB `MemAvailable`, exactly one RTX A5000 at
  `cuda:0`, <=60-minute sort wall, zero retry, and cache-lineage metadata.

The cache view and scientific contract are unchanged. All 29 manifest members
verify, 10 focused tests pass, the repeated 1,119,664-byte cache preflight is
identical, and `COMPLETE.json` is last. No output, service, sort, GPU work, or
scientific scoring exists.

One load-bearing orchestration defect remains. The release template dispatches
the score service immediately after dispatching the sort and gives it
`After=`/`Requires=` dependencies on the sort service. For an ordinary
long-running systemd service, these directives order **service startup**; they
do not delay the dependent unit until the required service exits successfully.
The score service's `ExecCondition` can therefore run while the sort is still
active, find no terminal receipt, exit without starting, and never score a later
successful sort. No `FAILURE.json` is written in that path because the score
process itself never starts.

This contradicts the frozen release invariant that scoring starts only after a
terminal successful sort. The Python worker is correct; only the systemd
lifecycle trigger requires repair.

## Exact repair

Replace the startup-order-only relationship with a completion-triggered,
disconnect-safe mechanism, for example:

- install an `OnSuccess=` score service on the managed sort unit before the
  sort starts; or
- install a systemd path unit watching the exact H terminal marker and activate
  the one-shot score service only when it appears; or
- use a managed controller service that waits on the actual sort unit to reach
  terminal success, then invokes the existing score worker.

Retain the worker's independent `verify_sort` check, one-attempt/no-overwrite
behavior, 900-second score wall, failure preservation, and terminal sealing.
Demonstrate the new launch method with cheap dummy success and failure jobs:

- success: score/seal starts once only after the producer exits 0;
- producer failure: scoring never runs and the producer failure is preserved;
- premature/incomplete marker: scoring fails closed with durable evidence;
- launcher disconnection: producer and successful downstream scoring survive;
- retry/restart count remains zero.

No cache, sorter, endpoint, threshold, guardrail, or resource redesign is
needed. Publish a fresh immutable namespace; preserve v1, v2, and this review.

## Implementation checks

- Done: packet integrity -> all 29 v2 manifest members verified; manifest
  `b9e7b61d...8c191`, COMPLETE `6d78f1ac...62b58`, written last.
- Done: frozen dependency closure -> packet-local helper hashes are checked
  before import and tamper rejection is tested (`harmonized_scorecard.py`,
  lines 15-42).
- Done: worker semantics -> terminal sort receipt/hash, H path, one-shot output,
  score failure, manifest, and COMPLETE-last paths traced and tested
  (`post_sort_score_and_seal.py`, lines 41-104).
- Done: release gate -> receipts, freshness, 25-GiB disk, 160-GiB RAM, single
  intended GPU, <=60-minute wall, zero retry, and cache metadata are enforced
  (`release_preflight.py`, lines 77-109).
- Done: scientific/cache diff -> v2 changes only namespace, disk reserve, frozen
  helpers, managed workflow, and release checks; H-F/H-L rules, F/L inputs,
  cache view, sorter settings, and exclusions are unchanged.
- Not done: terminal lifecycle -> `After=`/`Requires=` was not tested against a
  real long-running dummy producer and does not imply completion ordering.
- Can establish: the repaired components are individually correct and the
  scientific/input contract remains frozen.
- Cannot establish: a disconnect-safe sort-to-score lifecycle, launch readiness,
  H outcome, identity/purity, or long/full/production suitability.
