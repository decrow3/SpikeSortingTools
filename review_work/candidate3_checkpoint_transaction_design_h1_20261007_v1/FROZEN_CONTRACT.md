# Candidate-3 transactional checkpoint repair design — H1 v1

Status: `FROZEN_DESIGN_AND_PATCH_SNAPSHOT_NO_PRODUCTION_EXECUTION`

## Decision

- Milestone: make candidate-3 interruption semantics truthful before a managed full-session launch.
- Decision changed: define the smallest source change and fixture matrix that can make standard DARTsort subtraction/matching peel checkpoints recover only fully committed chunks.
- Cheapest adequate test: design against the independently reproduced marker-before-append failure, then implement a source-bound patch snapshot and synthetic crash points before requesting a change to the DARTsort clone.
- Completion: immutable patch/design with passing zero-event, nonzero-event, multi-dataset, residual-HDF5, changed-schedule and legacy-file rejection fixtures, plus independent review. This task does not modify the DARTsort clone or authorize a sort.

## Bound dependencies

- Inventory MANIFEST/COMPLETE: `0370b607b3240fb4da6b31ba1defe4907e6956a47539c9f22e6efe6f74074f7e` / `cfda092933eef6484fa6ea6c05e734f8fab0b48b69011a115b3f1578edc9c05a`.
- Fault fixture MANIFEST/COMPLETE: `96f9b81f01471ba88564426ab86d3e9810d794894e8d8e8de6a87187f512d446` / `c4af65ec8c1188cccbcd055852fad137bb97518dd44a0c1dccf20f247a836284`.
- Inventory independent review MANIFEST/COMPLETE: `a3011e203c8392cdb75513a88883823a9bd56876183019baacaeaa1dcf1a72e2` / `66c0357abdb6a7e2dde6959a95270f19fd97f3881e54c2c48d597dca0999970b`; verdict `CORE_NOT_READY_VERDICT_ACCEPTED_BINDING_REPAIR_REQUIRED`.
- DARTsort Git `edcfe1b51d672b4136eb13cc78c0875da804b851`; `peel_base.py` SHA-256 `ac357ade5b6f23b690a403e297a2260ea93c2a1b35df13ee7798d8fc8591a2d4`.
- Later real-snippet work must additionally bind compact H5 `INPUT_BINDINGS.json` SHA-256 `7c045a88be1819d5411534867ee717ceadf76661c3bb3ba176dd525ed5c4e3b1`, accepted Arm-A materialization manifest SHA-256 `ef2071c9ead186689b804f0df52ff9eb5a25cf1f4bdd610ae14e6c81cab834b2`, binary SHA-256 `672938e942d105af6b1a3f7f1bddf206939535e0e4b57bc01153472a1ccb8b69`, and exact lattice adapter SHA-256 `0125394629af7bffc6d46a5c26903efd65cd244f30000ae45d3cfa8324abea3a`. This design does not read those payloads.

## Frozen repair semantics

1. Per-spike and residual payload writes occur before a chunk is published committed.
2. A single append-only `chunk_commit_log` row records `(chunk_index, chunk_start, committed_n_spikes, committed_n_residuals, committed_external_residual_bytes)` only after payload flush; the log is flushed before returning.
3. On resume, the last complete commit-log row is authoritative. Every variable-length per-spike dataset is truncated to `committed_n_spikes`; logical/variable residual arrays are restored to `committed_n_residuals`; an external residual file is truncated to its committed byte extent before append.
4. Commit rows must be consecutive and agree exactly with `chunk_starts_samples`. Gaps, duplicates, non-monotone extents, dataset shorter than a committed extent, or a legacy partial file without the new schema reject. A completed legacy file is not silently upgraded.
5. The existing scalar marker fields may be retained as derived compatibility fields, but they are written only after the commit log and never override it.
6. Standard candidate subtraction/matching uses variable-length spike datasets and no external residual file. Template-reduction files use `ignore_resuming=True` and fixed known counts and remain outside the resumable contract.

## Frozen crash fixtures

- no-event chunk commit and resume;
- nonzero-event two-dataset commit and resume;
- crash after first dataset append, before second;
- crash after all payload append, before commit row;
- crash after commit row append, before compatibility scalar update;
- residual-HDF5 logical count rollback;
- external residual byte rollback or explicit rejection if unsupported;
- changed chunk schedule, corrupted/nonconsecutive commit log and committed extent beyond dataset length rejection;
- fully completed positive reload and final-row equality against uninterrupted execution.

## Bounds and exclusions

2 CPU, 4 GiB RSS, 30 minutes, under 100 MiB `/tmp`, under 10 MiB durable output. Synthetic data only. No raw/real voltage, GPU, sort/training, RF/holdout, production namespace, package install, source mutation outside this packet, or launch. Preserve failed patch iterations and do not relax crash acceptance after observing failures.
