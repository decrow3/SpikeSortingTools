# Independent implementation-first review: joint-model K10 evaluator v1

Verdict: **NO_GO_PENDING_EXACT_INTEGER_SESSION_BOUNDS_GUARD**

Do not use this frozen evaluator on joint-model arm output yet. The matcher and
the reviewed endpoint plumbing passed the bounded independent fixtures, but the
clock-support guard does not satisfy the frozen contract: `validate_inputs`
converts `session_bounds_samples` with `map(int, bounds)` before checking them
(`EVALUATOR.py:106-110`). Thus, for example, `[0.75, 10000.75]` is silently
accepted as `[0, 10000]`. These bounds define both the half-open event domain
and the circular shifted-null support, so silently changing them is a
load-bearing clock error.

## Exact repair and rereview condition

Validate the two supplied session bounds as exact integer samples before any
cast or normalization, using the same exact-integer rule applied to event times
and null shifts. Then require `lo < hi` and apply the existing half-open domain
checks. A minimal repair is equivalent to:

```python
bounds = _integer_samples("session_bounds_samples", bounds, 2)
lo, hi = map(int, bounds)
```

Preserve the present failing fixture. Freeze the repaired source in a new
packet, rerun the author suite and all independent fixtures, verify the new
hash chain, and obtain bounded independent rereview before inspecting any real
arm output. No matcher, endpoint, DARTsort, RF, holdout, or recording work is
needed for this repair.

## Independent evidence

- Source packet commit: `2cf2faf9a1bdfd68d388d80e0f03e4b974047d23`.
- Frozen packet and shared copy matched byte-for-byte for all eight packet
  files; the local manifest and completion hash chain verified.
- Frozen review `EVALUATOR.py` and `TESTS.py` matched the executable files at
  that commit byte-for-byte; no author file was modified.
- Author suite: `19 passed in 0.12s`.
- Independent adversarial suite: `10 passed, 1 failed in 0.20s`; the sole
  failure is the noninteger-session-bounds fail-closed fixture.
- Independent passing subset: `10 passed, 1 deselected in 0.16s`.
- Passing adversarial coverage included exhaustive maximum-cardinality checks,
  one-to-one exclusivity, both inclusive gates and per-event placement,
  per-cluster rather than pooled scoring, shifted-null statistic and spatial
  support, absence of event-key matching, many-to-one/merge/decoy separation,
  overlap endpoint separation, half-open event bounds, clock identity, and
  create-only preservation/cleanup.

No real joint-model arm output, K9 array, recording, voltage, DARTsort import or
edit, H5 contact, GPU, service, sort, RF, or holdout was accessed.

## Implementation checks

- Done: what actually ran -> frozen packet source/tests are byte-identical to
  commit `2cf2faf`; source, contract, tests, manifests, and local/shared hashes
  were inspected.
- Done: matching/counting -> exhaustive independent small-graph oracle and
  targeted fixtures confirm maximum cardinality, within-comparison output
  exclusivity, inclusive time/spatial gates, per-event placement, per-cluster
  scoring, and the frozen best-cluster order (`EVALUATOR.py:139-204,236-309`).
- Done: shifted null -> uses the same matcher, tolerance, spatial placement,
  cluster support, and circular declared session support; adversarial aligned
  and spatially swapped fixtures behaved as expected (`EVALUATOR.py:253-272`).
- Done: merge/many-to-one/overlap -> constructions and receipts were traced and
  independently exercised; detection, best-cluster assignment, frame,
  amplitude, and supplied residuals remain separate (`EVALUATOR.py:311-405`).
- Done: exact-key absence/create-only -> changing unique event IDs leaves the
  result unchanged; hard-link publication refuses overwrite and preserves the
  first bytes (`EVALUATOR.py:421-440`).
- Done: clocks -> event times, null shifts, clock frame/identity, and half-open
  event domain fail closed, but fractional session bounds are silently
  truncated (`EVALUATOR.py:98-135`; `ADVERSARIAL_TESTS.py`).
- Not done: real arm/K9 outputs, recordings, voltage, DARTsort integration,
  scale/runtime, GPU, RF, and holdout -> explicitly excluded by the review
  contract and assignment.
- Can establish: the reviewed matcher and named endpoint arithmetic pass
  synthetic known-answer/adversarial checks, and one precise clock guard defect
  blocks release.
- Cannot establish: benchmark validity, scientific arm benefit, biological
  identity or purity, runtime readiness, medium-run readiness, or authorization
  to inspect/launch a joint-model arm.

