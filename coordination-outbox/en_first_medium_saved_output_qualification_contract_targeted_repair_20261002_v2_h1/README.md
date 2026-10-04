# Saved-output qualification orchestration targeted repair

Status: `READY_FOR_H5_REREVIEW_REAL_EXECUTION_DISABLED`.

This packet repairs only the orchestration/lifecycle findings from H5 review.
The accepted measurement contract, measurement implementation, spatial loader,
active matcher, and Kilosort source snapshots are unchanged byte-for-byte. The
H5 review is preserved exactly as
`bindings/h5_composition_review_{MANIFEST.sha256,COMPLETE.json}` with hashes
`b13dac9f7ad26df5be20e21ca8768d0d9a38504650f4b0bc7684db1e9de12f32`
and `b18a9235d992ebb4cc2aa3aa3f15fc91570eeb27a3a248683a6011aabd657d21`.

Repairs:

1. The production CLI has no request-selectable fixture branch and rejects
   fixture fields or any non-`real_saved` arm. A separate fixture entry point
   has no caller-supplied request/input arguments, verifies exact packet-owned
   fixture hashes and containment, and emits `COMPLETE_FIXTURE` with receipt type
   `PACKAGED_FIXTURE`.
2. Phase 2 accepts only a real `COMPLETE_ONE_SHOT` / `REAL_CONTROL` Phase-1
   packet, exact Phase-1 contract/request hashes, and every manifest member. It
   loads and compares Phase-2 REF with the Phase-1 REF receipt before opening B.
3. Start tokens are HMAC-SHA256-bound to phase, contract/config hash, sanitized
   request identity, absolute fresh output namespace, active source hashes, and
   arm input identities. `SHA256(token).claim.json` is created atomically with
   `O_CREAT|O_EXCL` in the fixed registry and persists on failure; sequential,
   namespace, and concurrent replay fixtures fail.
4. The dedicated launcher installs CPU affinity, `RLIMIT_AS`, GPU-invisibility,
   and an external wall timeout before the worker starts. The worker rejects a
   missing/mismatched authenticated resource envelope before input loading;
   every runtime write checks the aggregate persistent-output quota. Receipts
   retain the limits and observed resource state.
5. The exact outcome-blind REF cohort is frozen in
   `cohorts/REF384_COHORT.v1.json`: the 61 ascending cluster IDs with median
   bound physical depth in inclusive `[1600,2180]` um. The list binds the H5
   independent REF receipt. Only the REF-only baseline unit-metrics columns
   `cluster_id` and `median_depth_um` were read; no B artifact, cross-arm score,
   RF result, or holdout was opened.

Twelve bounded tests pass, including fixture isolation, Phase-2 fixture and REF
rejection, sequential/concurrent/namespace token replay, over-thread, GPU,
memory, output-quota, and external-timeout fixtures. No real phase ran.

Implementation checks
- Done: traced all five H5 findings to executed branches and added positive/negative lifecycle fixtures.
- Done: rehashed unchanged measurement/spatial snapshots and exact H5 review bindings.
- Done: independently recomputed the frozen cohort IDs from the specified REF-only depth columns and verified its digest.
- Done: ran the full packet suite natively and from a byte-identical stage.
- Not done: no real Phase 1/2, B read, score, voltage, sorter, RF, holdout, or managed production job.
- Can establish: the repaired disabled orchestration is fail-closed for the reviewed fixture, prerequisite, token, resource, and cohort boundaries.
- Cannot establish: real execution readiness until H5 independently re-reviews the packet; no scientific result, efficacy, repeatability, identity, purity, recovery, or advancement claim.

