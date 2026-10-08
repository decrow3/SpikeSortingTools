# Candidate-3 readiness inventory — H1 v1

Status: `FROZEN_SOURCE_AND_METADATA_INVENTORY_ONLY`

## Decision and milestone

- Global milestone: prospective candidate-3 fully checkpointed DARTsort full-session training.
- Decision changed: identify whether H1 has the exact Arm-A pre-whitening producer/input, a usable DARTsort environment, and a checkpoint/save/resume path suitable for later snippet instrumentation and a reviewed full-session contract.
- Cheapest adequate test: inspect exact imported source/environment, saved configs/manifests and filesystem metadata; trace the implemented resume path before reading voltage or launching a sorter.
- Completion: publish exact source/config/input candidates and concrete readiness gaps, then freeze the separate snippet equality/instrumentation contract. This inventory cannot authorize a sort.

## Bound instructions

- Shared `AGENTS.md`: `/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/AGENTS.md`, SHA-256 `34414087180e5ba50812ae453c7cc2fd672e7c7b21e2f665d8a42f7a3e98cb08`.
- Umbrella plan: `/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/DARTSORT_DIVERGENCE_PLAN_20261007.md`, SHA-256 `39f55c319ca13b392fb04237829540d72aec6719336c2d43f0d67ab3a99684f8`.
- Dispatch: `candidate3_h1_sequence_20261007` in `DISPATCH_LEDGER_20261007.json`, dispatched `2026-10-07T23:26:40-07:00`.

## Inputs and allowed operations

- Read-only DARTsort clone `/home/huklab/Documents/DARTsort`, initially observed Git `edcfe1b51d672b4136eb13cc78c0875da804b851` with clean short status, `uv.lock` SHA-256 `68f3fa53642043313bc9a4e56b7fc35dbd27af81030e436ffdba82de683f5034`.
- Read-only project manifests, contracts, effective configs, source, file names/sizes and bounded metadata needed to identify Arm A and the corresponding pre-whitening boundary.
- Environment/import diagnostics and GPU/storage visibility checks; temporary compiler/cache output only under `/tmp`.

## Bounds and prohibitions

- At most 2 CPU, 4 GiB RSS, 30 minutes wall time, 1 GiB aggregate metadata/source reads, and 10 MiB durable output.
- No raw-voltage/sample reads, waveform extraction, HDF5/NPZ scientific-array inspection, sorter launch, training, fit, RF/holdout access, parameter change, package install, source mutation, output deletion, or transfer of large artifacts.
- Do not inspect the failed D3 production outcome namespace.
- Stop and record a precise gap if exact Arm-A boundary or source cannot be identified without broader data access. Preserve all historical runs.

## Frozen interpretation

This task may establish only operational/source/input candidates and missing prerequisites. It cannot establish sample equality, resume correctness under interruption, full-session readiness, sorting benefit, causal attribution, or biological identity. Those require separately frozen fixtures/snippets, managed-job survival checks, independent review, and ultimately the full-session experiment.
