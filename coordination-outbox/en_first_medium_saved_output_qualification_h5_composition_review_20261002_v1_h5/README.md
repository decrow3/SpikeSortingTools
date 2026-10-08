# H5 independent composition review

Status: `TARGETED_REPAIR_REQUIRED_NO_PHASE1_TOKEN`.

The H1 composition packet is byte-valid and preserves the accepted measurement
and spatial snapshots. The active spatial-superset matcher has the same
canonical AST for `coincidence_counts`, `exclusive_count`, and
`correspondence` as the reviewed measurement snapshot. The thin composition,
REF-only Phase-1 output schema, diagnostic labeling, absence of a Phase-1
cross-arm scalar/ranking/tuning path, fresh-namespace check, and saved-format
CLI fixture all pass.

Four fail-closed/lifecycle gaps block a real Phase-1 start token:

1. Fixture mode is selected by request text plus a public constant. Its arm is
   labeled `real_saved`, may point to any absolute saved-output path, and bypasses
   both real-execution flags. The adversarial fixture passed from an arbitrary
   path outside the packet.
2. The Phase-2 prerequisite verifier explicitly accepts a Phase-1
   `COMPLETE_FIXTURE` receipt. The adversarial proof presented such a receipt and
   the verifier accepted it.
3. A Phase-1 start token is compared as a string but never atomically consumed.
   The same token completed twice in two fresh namespaces in a synthetic replay
   proof. Same-namespace preservation works, but that is not one-start semantics.
4. Resource checks run only after input loading, all controls, and the actual
   binding receipt. A five-thread declaration completed those operations before
   failing. Wall and peak-RSS checks are likewise post hoc, and `gpu_count: 0`
   is not checked by the CLI.

The exact REF saved-output position/frame/clock/row/source binding was
independently recomputed read-only. It passes and is recorded in
`REAL_REF_BINDING_RECEIPT.json`. No cohort/start token is issued because the
composition gate did not pass. No voltage, B arm, scientific score, job, RF, or
sealed holdout was accessed.

Implementation checks
- Done: verified subject MANIFEST/COMPLETE and every member; compared accepted dependency bindings byte-for-byte with their source packets.
- Done: independently recomputed matcher AST equivalence and real REF spatial/clock/row/source identity.
- Done: reran six tests natively and from a byte-identical staged packet; ran bounded synthetic adversarial controls for fixture isolation, Phase-2 prerequisites, token replay, and resource timing.
- Not done: no real Phase 1, B-arm read, cross-arm scalar, voltage, job, RF, or holdout work; no start token because blocking gates remain.
- Can establish: the accepted modules and REF binding are intact, while the published orchestration does not yet enforce its claimed real/fixture, one-start, resource, and Phase-2 boundaries.
- Cannot establish: launch readiness, any existing-B result, DeltaR, efficacy, repeatability, biological identity, recovery, or advancement.
