# H1 ready queue — refractory interval defect follow-up

Updated: 2026-10-07T21:17:06-07:00

## Completed: independent impact-map review and repair

- Status: GO after one repair cycle.
- Milestone / decision: validate whether the immutable impact map is adequate to route a bounded correction contract.
- Reviewer: native child `/root/review_refractory_impact` on `huklaban1-Precision-5820-Tower`.
- Accepted input: `/mnt/NPX/Luke/DARTsort_motion_experiments/refractory_interval_defect_impact_map_h1_20261007_v2`, MANIFEST `3bfb285271ac7f5991c48a48c995383710612648f44786bfe954f1f7bc009555`, COMPLETE `15e23b28e8e01b80c82b6ddd6a7baf93cbddbd4c16f888ae03af0b32b7ca9c6e`.
- Completion evidence: delta GO packet `/mnt/NPX/Luke/DARTsort_motion_experiments/refractory_interval_defect_impact_map_review_20261007_v2_child`, MANIFEST `17b1ea06fea685b5caf9e925747d31fe56a6dd231d7568fa77341927c7a5153d`, COMPLETE `b9226390e1985e2e3562993ba56ae24f18ec6c9b080def7942c1ae5ca325f443`.
- Invalidation: any mutation/hash drift of the reviewed snapshot or discovery that the bound R1c evaluator/source differs from the audited implementation.
- Safe checkpoint: after each primary artifact/source trace.

## Completed supporting task: exact R1c source-recovery audit

- Result: exact source was not recoverable from H1 local/shared/advertised remote history; ledger SHA-256 `fdccf32f1a7f4f6b94f40342474578652a63bff6557e15314a709ef77ad3d9b8`.
- Milestone / decision: decide whether exact executed evaluator SHA-256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423` can be recovered from an existing Git object/ref or immutable packet, eliminating the source-provenance gap without new scientific computation.
- Exact inputs: census `CONTRACT.json` SHA-256 `e90ad62778b5fa9422102c2f0d9ffeefa0c710a38ceba68d6d79ad9553bf5ea3`; declared commit `ba21bc97ad177ba5f2a1b79fb69d37f4f3eac3ad`; repository remotes and existing shared packets.
- Prerequisites: read-only Git/shared-tree access; network only for advertised remote refs if needed.
- Output: compact recovery ledger identifying every checked object/ref/path and either the exact source bytes/hash or a precise absence result.
- Recipient: correction-contract owner and independent reviewer.
- Invalidation: H5 publishes the exact source snapshot with verified SHA-256 first.
- Safe checkpoint: after local object/shared-tree search, before any remote fetch.

## Completed supporting task: sample-clock known-answer fixture specification

- Result: synthetic boundary/confidence fixture draft independently verified; spec SHA-256 `5141cb6f8f58b5fe74bbf3c22bc6081307054d11e502ad0da0a61b50a733ac98`, expected JSON SHA-256 `aeec9c32fcb2928369ddbb4d03585400b35a005d326651f51206a8232f249c2b`.
- Milestone / decision: freeze an implementation-neutral fixture that distinguishes 7/fs sample-derived edges from nominal 0.25-ms edges before any corrected real outcome is computed.
- Exact inputs: defect impact packet above; imec0 clock `29999.835983263598` Hz; imec1 clock `29999.759166666667` Hz; historical bin count 40 and bin width 7 samples; contamination grid 0.005–0.345; confidence threshold 0.9.
- Prerequisites: impact-map review may still be running; fixture is synthetic and must not encode a preferred scientific verdict.
- Output: frozen fixture cases/expected arithmetic and acceptance criteria only, no production implementation or real-data results.
- Recipient: correction-contract author and reviewer.
- Invalidation: independent review finds the defect semantics or required boundary rule misstated.
- Safe checkpoint: after expected values are independently derived, before implementation integration.

## Ready 1: independent ACG reference fixture implementation

- Priority: P1; independent of H5 output.
- Milestone / decision: provide a small implementation-independent oracle for exact bin boundaries, same-block/Boolean-state inclusion, no segment gate, and `d=0` behavior.
- Exact inputs: accepted impact-map v2 and fixture draft hashes above; no real spike arrays.
- Prerequisites: CPU Python/NumPy only.
- Output: reviewer-ready synthetic reference source, cases, and receipt in a fresh namespace.
- Recipient: correction-contract owner.
- Invalidation: corrected contract changes the accepted one-factor historical counting boundary.
- Safe checkpoint: after reference expected vectors are frozen, before comparing any production helper.

## Ready 2: corrected sufficient-stat consumer schema/validator

- Priority: P1; independent of H5 output.
- Milestone / decision: prove compact ACG/count/exposure artifacts are sufficient to recompute point and deterministic-bootstrap SRP/P/K without spike or voltage rereads.
- Exact inputs: accepted v2 recalculation plan; synthetic tensors only; exact clock/grid definitions.
- Prerequisites: the schema must remain outcome-neutral and must not encode real R1 values.
- Output: schema, closure validator, synthetic positive/negative fixtures, and resource estimate; no real outcomes.
- Recipient: correction-contract author/reviewer.
- Invalidation: source recovery reveals materially different R1c tensor/block semantics.
- Safe checkpoint: after schema closure tests, before integration with any real packet.

## Completed fallback: divergence production-contract review/release

- Status: activated after the H5 child returned technical acceptance but could not create a trusted-authorization release; completed by H1 without launching production.
- Routing amendment: H5's native child reviewer owns the normal production-contract review. H1 must not duplicate that review.
- Activation evidence: coordinator recorded the H5-child authorization-only failure and dispatched H1 fallback against exact D3 v2.
- Result: `GO_EXACT_D3_PRODUCTION_EXECUTION` for exact contract `81ac32ec...` and runner `fc304240...`, scoped `historical_panel_only`; no production launch by H1.
- Exact inputs: awaited immutable H5 production-contract packet plus H1 checklist and interpretation rules committed at `a696eea82730e02aaff2d6068cfad2f68775acde`.
- Prerequisites: recorded coordinator fallback dispatch plus immutable contract, source/config snapshots, fixtures, managed-runner bindings, explicit resources and stop condition.
- Output: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_production_release_20261007_v2_h1_fallback`, MANIFEST `44876744836db23a01576812ad10d0b55064ec51902ea417687533c0d42d3c45`, COMPLETE `3f5a6589779c9508def7a9d19857510015cc28a35c63d86d61e4afed50f00a34`, RELEASE `edd155483e31d3dd0ef49adf573e6243e7440af77300a80c252a7cbaf598fa12`.
- Recipient: H5 production owner and coordinator.
- Invalidation: changed contract/source/config hash, unresolved fixture failure, or altered scientific acceptance rule.
- Successor: H5 data-local execution owner after refreshed prelaunch reconciliation; awaiting `ACK_STARTED` or concrete blocker.

No additional conditional successor or fallback is queued.
