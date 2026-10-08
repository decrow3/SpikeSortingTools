# Independent delta review: candidate-3 equality result evaluator v2

## Verdict

`NO_GO_REPAIR_MALFORMED_SCHEMA_EXCEPTION_CLASSIFICATION`

V2 correctly repairs all three v1 fail-open cases, but it still cannot guarantee
the contract's deterministic category and sealed minimum coordinator action for
malformed, correctly hashed compact members. Two independent malformed-schema
controls raise uncaught exceptions. Preserve v2 unchanged and repair only these
exception boundaries in a distinct version before consuming a real result.

Reviewed identities:

- packet MANIFEST `97247989e57f01bf77d2674fa35f8735b32bf1c1d4d994f1c110b41e70c78a19`
- packet COMPLETE `154c9dc6917768503a3d513177a8c873f181765130705fc84a4bbc568d16ba64`
- evaluator `c611acdd30bec755ef8d50ef60ae76b3684e567d66ad709edab8643be42ce678`
- evaluator contract `4aa189c0120749829cee1749ae5ef75ff9835779bb111f5fe41a67fcf48ce34f`

## V1 repairs that pass

1. **Output hashes.** Each lazy and eager digest must be a present lowercase
   64-hex string and the pair must match. Missing-both no longer passes.

2. **Real release reconstruction.** For well-typed valid JSON, v2 validates the
   exact interpreter, canonical argv including every flag/value and the runner
   path, six release path/hash fields, absolute receipt paths and strict digest
   syntax. It rehashes and reparses review, dispatch and resource receipts and
   cross-checks their schemas, statuses, hashes, paths, resource limits, manager
   identifier and execution mode against the exact accepted v3 contract.

3. **Handled failure semantics.** MANIFEST and COMPLETE status must agree; a
   handled failure additionally requires all three statuses to equal
   `FAILED_PRESERVED_NO_RETRY`, string type/message and finite nonnegative wall
   time. The formerly inconsistent pass/failure fixture now rejects at the
   seal/schema priority.

4. **Reachability and regression suite.** All ten frozen controls pass, including
   the three v1 adversarial regressions and a fully cross-bound synthetic
   real-schema construction. The latter proves category reachability only; it is
   not an H5 outcome.

5. **Unchanged valid-path logic.** Request reconstruction, zero tolerances,
   positive controls, resource checks, DARTsort boundary, seal membership/hashes,
   synthetic/real distinction and minimal decision text are unchanged and remain
   coherent for well-formed inputs.

## Remaining blocker

### Malformed FAILURE JSON escapes the category machine

A v3 synthetic-copy packet was changed to contain exactly one FAILURE member
whose bytes were malformed JSON, then correctly resealed with
`FAILED_PRESERVED_NO_RETRY`. `validate_seal` accepted its membership, size and
hash. `evaluate` then executes `json.loads` for FAILURE outside a rejection
guard, raising `JSONDecodeError` instead of returning
`REJECT_SEAL_OR_SCHEMA_STOP`.

### Ill-typed release paths escape provenance rejection

A resealed real-mode schema fixture supplied all six truthy release fields and
strict digest-shaped strings, but made `review_complete` an integer. The path
dictionary comprehension calls `Path(123)` outside a rejection guard, raising
`TypeError` instead of returning `REJECT_PROVENANCE_STOP`. Malformed receipt JSON
has the same unguarded parse pattern later in the block.

These cases do not falsely advance candidate3, but they are still blocking: the
evaluator's purpose is to deterministically translate one sealed packet into a
minimum coordinator action and seal that verdict. An uncaught exception produces
neither. There is no separately reviewed persistent evaluator wrapper whose
receipt supplies an equivalent deterministic action.

## Minimal repair

1. Catch FAILURE JSON read/parse/type errors and return
   `REJECT_SEAL_OR_SCHEMA_STOP` before handled-failure classification.
2. Require all release path fields to be strings before constructing `Path`.
   Catch path, file-read and receipt JSON parse/type errors and return
   `REJECT_PROVENANCE_STOP`.
3. Add the two malformed compact controls. Preserve all current scientific,
   resource, category-priority and coordinator-decision semantics.

## Implementation checks

- Done: v1-to-v2 source/contract delta and all ten controls -> all three original
  fail-open defects are repaired; suite passes 10/10.
- Done: malformed FAILURE JSON and ill-typed release path controls using only
  resealed v3 synthetic copies -> uncaught JSONDecodeError and TypeError.
- Not done: H5 contact, real outcome, voltage, services, GPU, sort, RF, holdout
  or execution framework -> excluded.
- Can establish: v2 is fail-closed against advancement and correct for tested
  well-formed packets.
- Cannot establish: deterministic classification and a sealed minimum action for
  all compact schema failures; evaluator v2 remains NO-GO.
