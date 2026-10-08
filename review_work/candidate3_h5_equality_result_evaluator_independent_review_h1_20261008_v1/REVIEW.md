# Independent review: candidate-3 H5 equality result evaluator v1

## Verdict

`NO_GO_REPAIR_FAIL_OPEN_SCHEMA_AND_REAL_PROVENANCE_VALIDATION`

The evaluator is deterministic and most arithmetic/seal checks are sound, but it
is not safe to interpret a real H5 result. Three compact adversarial controls,
using only temporary copies of the sealed v3 synthetic packet, reached categories
that are too permissive. Preserve this packet unchanged and repair the evaluator
under a distinct version before consuming any real result.

Reviewed identities:

- packet MANIFEST `64325df582aa5845d6cc73e0cc7bb3b3f3514bcc9610eb6d81a26668700ce669`
- packet COMPLETE `0773a040a7a8b848163ba23db52d89c533f6f051569ed54b020d06a9a035246f`
- evaluator `b2afc87ebcedb868f8a5804254a27fde251aada55e40dff615e08266420d3c64`
- evaluator contract `302d29fba22761dd968fa4f976656503cb6cfe1048235c73a8fd07d951efc0ae`
- accepted v3 execution contract `6cb5c29b806b63021bf9edef87535718f24efc9ff27842d9e4e34c3e06e56037`
- accepted v3 runner `0bed993b8718f29591b92884e88b65f2bf8a6780cce49490b869ba82823f1f7e`

## Checks that pass

1. **Deterministic priority.** Evaluation is a fixed early-return sequence:
   seal/schema, handled failure, provenance, resources, primary equality,
   positive control, DARTsort boundary, then mode-specific pass. The six stock
   controls pass and observe the intended categories they cover.

2. **Seal and contract identity.** The evaluator checks COMPLETE and MANIFEST
   schemas, COMPLETE-to-MANIFEST hash, exact top-level membership, every member
   size/hash, exactly one terminal member and the exact accepted execution
   contract copy. Its own output writes VERDICT and PROVENANCE, then MANIFEST,
   then COMPLETE.

3. **Request reconstruction.** Nine expected rows are independently constructed
   from the hash-bound execution contract in exact order. Global half-open frames
   and `[length,384]` shapes are reconstructed, including the transition-only
   boundary request, rather than trusted from RESULT.

4. **Primary scalar rules.** Each row must be float32, finite, zero mismatch,
   zero max absolute error and have no first mismatch index. This implements the
   frozen zero-tolerance rule for present scalar fields.

5. **Positive and DARTsort controls.** The evaluator requires exactly ordinary
   then transition positive-control arms with positive finite mismatch counts and
   errors. It also requires exactly two no-copy boundary dictionaries with both
   same-object flags and null work directory.

6. **Resource checks.** Cached and logical binary-read byte counts are exact by
   mode; wall time must be finite and within 1,800 seconds; temporary bytes must
   be an integer within 2 GiB. These are correctly ordered before scientific
   interpretation. External manager facts remain a runner/release concern and
   are not inferred here.

7. **Pass distinction.** Synthetic provenance can only reach the synthetic
   plumbing category, while real provenance plus `PASS_REAL_EQUALITY` is required
   for the real advancement category. The textual coordinator actions are
   minimal and do not themselves authorize a new experiment.

## Blocking defects reproduced

### 1. Missing output digests pass

For all nine request rows, both `lazy_sha256` and `eager_sha256` were removed and
the packet was validly resealed. The evaluator returned
`PASS_SYNTHETIC_FIXTURE_ONLY_NO_H5_DECISION`. At line 170 it compares the two
`dict.get` results; two missing values are both `None` and compare equal. Thus it
does not enforce the contract's requirement for identical SHA-256 values.

### 2. Incomplete real provenance advances

A temporary synthetic copy was consistently resealed as real, with correct
interpreter text, only the four argv tokens currently searched by membership,
and three nonempty receipt values that were deliberately not SHA-256 strings.
The evaluator returned `PASS_REAL_EQUALITY_ADVANCE_TO_FROZEN_NEXT_GATE`.

The real gate does not reconstruct exact argv, check `argv[0]` against the frozen
runner, require argument/value adjacency, require review/dispatch/resource paths,
compare receipt paths with the runner provenance, or validate the three receipt
digests as 64 lowercase hexadecimal values. Nonempty strings suffice. A sealed
packet is integrity evidence, not authority or authenticity, so the evaluator
must independently enforce the frozen v3 provenance schema before advancing.

### 3. Inconsistent failure status bypasses schema priority

A packet with exactly one validly hashed FAILURE member but MANIFEST and COMPLETE
status `PASS_REAL_EQUALITY` and a nonfrozen failure status was classified
`UPSTREAM_HANDLED_FAILURE_STOP`. The stop action is conservative, but it violates
the declared seal/schema-first priority and labels a semantically inconsistent
packet a valid handled failure. The evaluator returns on terminal name before
checking terminal/status agreement or the frozen failure schema/status.

## Minimal repair

1. Require both output digests to be present strings matching exactly 64 lowercase
   hexadecimal characters before comparing them.
2. For real mode, reconstruct the exact semantic v3 invocation from the accepted
   contract: runner in `argv[0]`; exact flag/value pairs for contract and its hash,
   output, Kilosort site, review COMPLETE/path/hash, dispatch path/hash, resource
   path/hash; and exactly one `--execute-real`. Cross-check all release paths and
   hashes recorded in provenance, and validate every digest syntax.
3. Before the handled-failure return, require MANIFEST and COMPLETE statuses to
   agree with each other and with the frozen failure status; validate FAILURE
   schema/required fields and reject inconsistencies as seal/schema errors.
4. Add these three controls to the frozen suite. Retain the current category
   order, request reconstruction, scientific thresholds and minimal coordinator
   actions unchanged.

## Implementation checks

- Done: executed evaluator, contract, packet seals, request construction,
  category ordering, output sealing and all six stock controls inspected; stock
  tests pass 6/6.
- Done: three independent compact adversarial controls on sealed synthetic-copy
  packets -> missing digests pass synthetic, incomplete real provenance advances,
  and inconsistent failure status bypasses schema rejection.
- Not done: real H5 result inspection/contact, voltage, services, GPU, sorts, RF,
  holdout or execution framework -> explicitly excluded.
- Can establish: deterministic behavior and correct arithmetic for present,
  well-formed fields, plus exact locations of three fail-open validation gaps.
- Cannot establish: safe deterministic interpretation of a later real H5 result;
  evaluator v1 is therefore NO-GO.
