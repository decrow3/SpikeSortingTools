# Independent final delta review: candidate-3 equality result evaluator v3

## Verdict

`GO_FROZEN_V3_RESULT_CONSUMER_ONLY`

The exact evaluator v3 closes the two remaining v2 deterministic-output defects
without changing scientific thresholds, category priority or coordinator
decisions. It is suitable as the frozen consumer of a future exact candidate-3
equality v3 packet. This review does not inspect, imply or authorize a real H5
outcome or any downstream experiment.

Reviewed identities:

- packet MANIFEST `480e5e079a792f4254b6664b9b86364215fdd211605be2300d9380e54ad7e4ba`
- packet COMPLETE `028f093b549d5ba8b66db061119a415f1c4516ee546894e90ba0ae61059ac028`
- evaluator `7af98e2b2652dd6fcb9ee657d82f4db2e98a8bc19b161ea14513c522c058027c`
- contract `f9a70c6885f994ec1654b2a4cfe456b1f53fcde1de630c17cec9e27aa30416a5`

## Delta findings

1. Malformed but correctly hashed FAILURE JSON is caught inside the failure
   branch and deterministically returns `REJECT_SEAL_OR_SCHEMA_STOP`.

2. All three release path fields must be strings before `Path` construction.
   Missing/non-file/hash-mismatched receipts, receipt read/JSON errors and
   non-object top-level receipt values deterministically produce
   `REJECT_PROVENANCE_STOP`.

3. The CLI wraps the complete `evaluate` call with a final fail-closed guard.
   Any unanticipated compact-schema exception becomes
   `REJECT_SEAL_OR_SCHEMA_STOP`, after which the normal output path seals
   VERDICT, PROVENANCE, MANIFEST and COMPLETE. This restores the evaluator's
   promised minimum coordinator action rather than relying on process failure.

4. The earlier strict protections remain intact: present 64-lowercase-hex
   output digests, exact canonical real argv, rehashed/reparsed review/dispatch/
   resource receipts with full v3 cross-bindings, exact failure statuses/schema,
   request reconstruction, zero tolerance, positive controls, resource counters,
   no-copy boundary and synthetic-versus-real decision separation.

5. The exact committed evaluator and contract match the immutable packet. The
   independently rerun suite passes all 12 controls, including both preserved v2
   malformed fixtures, all v1 regressions, accepted synthetic success and fully
   bound synthetic real-schema reachability. The reachability control validates
   logic only and is not evidence of real equality.

## Scope and coordinator action

Use only this exact evaluator/contract pair on the later exact sealed v3 runner
packet. Any evaluator, contract, packet, member, receipt, path, hash, schema,
resource or provenance mismatch must retain the emitted stop category. A real
pass permits only consideration of the already-frozen next gate; it does not
authorize that gate, a sort, RF/holdout access or a retry.

## Implementation checks

- Done: v2-to-v3 source and contract delta -> both reviewer exception fixtures
  are caught at their intended categories and CLI has a sealed fallback.
- Done: exact committed/packet identities and full control suite -> 12/12 pass.
- Done: earlier evaluator reviews reused -> strict equality, provenance,
  resources, positive controls, DARTsort boundary and category order unchanged.
- Not done: H5 contact, real result, voltage, service/GPU/sort/RF/holdout or
  execution framework -> excluded.
- Can establish: deterministic compact-schema interpretation and sealed minimum
  action for the reviewed evaluator v3 consumer.
- Cannot establish: real equality or any downstream scientific outcome.
