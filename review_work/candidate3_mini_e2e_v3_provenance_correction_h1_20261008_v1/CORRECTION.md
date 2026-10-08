# Candidate-3 mini v3 provenance correction

Status: `SCOPED_RESULT_PRESERVED_WRAPPER_BYTE_IDENTITY_UNRESOLVED`

This packet supplements, and does not modify, the sealed packet
`candidate3_mini_e2e_stage_capability_h1_20261008_v3`.

The prelaunch/running coordinator receipt records the executed orchestrator
source SHA-256 as
`5d55bb1e0257afbb87b60d9d0d9dd5fcb4290a49ce2f67f8656969929e18801a`.
The source first committed with the result (`697d2bc`) has SHA-256
`e627b1aaba5355a6a156aab908bd6bc4e19f82cfffbb359ffcaf1ee1ef67d32e`.
The latter includes post-run changes to result summarization/loading. No exact
snapshot of the former source bytes was retained, so this packet does not claim
that the committed wrapper is byte-identical to the launched wrapper. The
author's recollection that only reporting code changed is not independently
verifiable and is therefore not used as evidence.

The ordinary-arm execution remains evidenced at the native-output level by the
resolved DARTsort configuration, timing receipt, final sorting NPZ, and compact
accounting export hashes recorded in `PROVENANCE_SUPPLEMENT.json`. These support
the scoped claim that the ordinary synthetic arm traversed the configured native
DARTsort stages and produced the reported final output. They do not establish
exact wrapper-byte reproducibility, transition-arm execution, real-voltage
readiness, sorter benefit, biological identity, purity, or causal attribution.

Future real-snippet or load-bearing runs must snapshot the exact executed source
before launch, hash that snapshot in the prelaunch receipt, and carry the same
hash into the final packet. A source hash without retained bytes is insufficient.

Implementation checks
- Done: compared prelaunch source binding to the first committed source -> hashes differ (`5d55...` versus `e627...`).
- Done: independently hashed resolved native config, timing receipt, final sorting NPZ, and sealed compact accounting artifacts -> bindings recorded in `PROVENANCE_SUPPLEMENT.json`.
- Done: preserved the original sealed v3 packet without edits -> correction is additive in a new namespace.
- Not done: byte-level review of the exact launched orchestrator -> its bytes were not snapshotted and are unavailable.
- Can establish: scoped ordinary synthetic native-stage completion and inspectable output/accounting provenance.
- Cannot establish: exact launched-wrapper reproduction, transition or real-voltage behavior, benefit, identity, purity, or causal stage attribution.
