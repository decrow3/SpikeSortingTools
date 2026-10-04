# Candidate G v2.1 integrated implementation-first review

## Verdict

`NO_GO_INTEGRATED_CANDIDATE_G_V2_1_PREPARATION`

Candidate G remains execution-disabled. This review did not run candidate G, install or start a service, perform the real input preflight, read feature/residual/recording-voltage payloads, use a GPU, or inspect RF/holdout data.

## Reviewed packet

- Path: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_f_fixed_event_candidate_g_preparation_20261004_v2_1_h5`
- MANIFEST.json SHA-256: `2ce97d278b3d40c0583e8d06ae2e74b3de651b2b99b156b583abe86b19a302e7`
- COMPLETE.json SHA-256: `4823ab489675d4387e565ea58f7bf8c3fc1808239a31eda3c64dba9a795b3708`
- Embedded H1 request SHA-256: `cbf9edac8449361ff8c56342e01bcf86ec9fa8a80b9e75b4f1f3b70b68b420a0`
- Coordination H1 request SHA-256: `de82469c59cee7abd09263ca202e39cf06a4c546f5d29708c50bfec594ffcdfc`
- Independent packet verification: all 190 listed products matched their byte counts and SHA-256 hashes; `COMPLETE.json` is physically newer than MANIFEST and every product.
- Independent synthetic rerun: 46 passed in 1.93 seconds with bytecode and pytest cache writes disabled.

## Blocking findings

### 1. The frozen release semantics and executable gate disagree

The immutable repair contract requires top-level release status `RELEASED_FOR_ONE_MANAGED_RUN`, coordinator schema `candidate-g-coordinator-release-v2`, and H1 status `GO_INTEGRATED_CANDIDATE_G_PREPARATION_EXECUTION_REMAINS_DISABLED` (`contract/candidate_g.execution_disabled.v2.json:8-16`). The executable gate instead requires the top-level release status `RELEASE_CANDIDATE_G_ONCE`, an embedded authorization schema `candidate-g-coordinator-authorization-v2`, and the old H1 status (`source/release_gate.py:11-12,22-29,65-94`). It validates only the v2 contract hash, not these frozen interlock fields (`source/release_gate.py:47-52`).

The coordination assignment permits only `GO_INTEGRATED_CANDIDATE_G_V2_1_PREPARATION_EXECUTION_REMAINS_DISABLED` for a GO, but `release_gate.H1_STATUS` requires the old status. Therefore the only authorized v2.1 GO artifact would be rejected, while a differently named artifact not authorized by this assignment is what the worker accepts. This is a load-bearing contract/gate mismatch, not a cosmetic label difference.

### 2. Failed-run evidence is not preserved on all consequential failure paths

The ordinary Python-exception path before sealing creates and publishes a sealed `failed_preserved` packet (`source/candidate_g_worker.py:104-116,242-246`). However:

- after `seal_directory(work)` succeeds, any copy, partial verification, atomic rename, or final verification failure skips `_publish_failure` because `work/COMPLETE.json` already exists;
- once `publish_partial` exists, the same handler also declines to publish a failure packet;
- `RuntimeMaxSec=3600` supplies no Python signal handler or `ExecStopPost` evidence-preservation helper, so systemd termination can bypass the exception handler entirely;
- work resides in the service-private tmpfs, so unhandled terminal evidence is not an independently durable failure record.

These paths violate the unchanged v1 completion condition requiring one preserved failure receipt and the project rule to preserve failed-run evidence before restart. A full candidate-G run must not be entrusted to this launcher until failure publication is independently durable and tested for timeout and publication-stage failures.

## Prior blocker review

1. **Semantic sealed approvals — repaired in substance, but blocked by incompatible frozen semantics.** Full packet membership, status, COMPLETE-newest, H1 verdict fields, no-blocker condition, coordinator scope, and installed-unit hash are checked. Free text fails. The contract/gate/status mismatch above prevents GO.
2. **Live managed context — implemented synthetically.** Before project input or output creation, code checks invocation ID/PID, active unit/MainPID/cgroup, installed-unit hash, live memory/CPU/runtime limits, thread variables, one visible CUDA token, empty bounded tmpfs, exact release CLI binding, and absent output (`source/managed_context.py:66-123`; worker lines 119-136). Actual host formatting, mount availability, and GPU availability remain explicitly untested real-run conditions.
3. **Input identity and boundaries — repaired for the specified bounded policy.** Production receipt hashes/semantics, exact stat identities, H5 schema/scalars/chunk endpoints, four boundary timestamps and searchsorted inequalities, cache completion/binary metadata, model/motion hashes, and retained descriptors are checked before payload access (`source/input_preflight.py:66-140`). Historical `template_data.npz` is stat-checked but never opened/copied. Preparation provenance is correctly disclosed: bounded H5 metadata and four timestamps were inspected; feature, residual, and voltage payload bytes remained zero.
4. **Timing/state/output/terminal seal — success path repaired; failure path still blocked.** Native PCMerge/TMM/agglomeration timing, coarse timings, TMM/noise/membership, fresh templates/agglomeration arrays, stage bindings, final rows/transitions, reconciled meter, exact inventory, and COMPLETE-newest are validated (`source/terminal_validation.py:32-107`; worker lines 223-241). Blocker 2 prevents acceptance of the overall consequential-run lifecycle.
5. **Transactional meter — repaired.** SQLite `BEGIN IMMEDIATE` atomically increments reserved bytes and inserts the reservation; completed, failed, and pending states remain authoritative, counter/rows must reconcile, terminal success rejects pending or failed reservations, and reserve occurs before the parent read (`source/fixed_event_meter.py:37-109,131-144`).
6. **Preparation COMPLETE-last — repaired.** All 190 products verify and the published v2.1 COMPLETE mtime is strictly newest (`source/seal_validation.py:29-69`). The preserved v2.1 reporting correction accurately supersedes only the earlier overbroad zero-open statement.

## Scientific identity and scope

The v1 scientific contract SHA-256 remains `5c02db316bff951676d5c062f46837fc24bde42b2de7fc536b704868812d58cd`. The evaluation freeze and mixture/agglomeration capture patches match v1; the only declared native delta is the timing wrapper around unchanged callbacks. No G outcome exists.

This review can establish that the input-preflight, managed-context, metering, success-terminal, and sealing repairs are substantially implemented and synthetically exercised. It cannot establish actual systemd-host compatibility, real stage completion within resources, real cache-read behavior, real state shapes, outcome validity, continuity, refractory behavior, waveform preservation, biological identity, purity, or G-vs-F superiority/inferiority.

## Required repair

Create a new immutable preparation namespace that makes the contract, review verdict, release packet, and gate use one exact set of schema/status strings. Add a durable failure finalizer outside the worker process/private tmpfs that records systemd timeout/signal/exit status and preserves or atomically quarantines publication partials; make publication-stage failures produce a sealed failure record without overwriting partial evidence. Add synthetic fixtures for the exact v2.1 GO status, runtime termination, post-seal copy failure, partial verification failure, and final verification failure. Preserve v2.1 unchanged.

## Implementation checks

- Done: what actually ran -> independently verified 190 packet products and COMPLETE-last; reran 46 synthetic tests; no candidate fit or real preflight was run.
- Done: arm differences -> v1 scientific contract/evaluation and mixture/agglomeration capture hashes remain unchanged; only implementation repairs and timing instrumentation differ.
- Done: axes/frames/coordinates -> exact parent rows `[18112500,18603444)`, W3 frames `[239998073,250197991)`, global/local times, channels, feature rows/content and neighbourhood bindings remain frozen.
- Done: clocks/time bases -> sampling frequency, completion scalars, chunk endpoints, four boundary timestamps, and half-open searchsorted inequalities are frozen and checked metadata-only.
- Done: silent caps/defaults -> live CPU/memory/runtime/thread/scratch declarations and the 12 GiB meter cap are checked; actual host/GPU availability remains a real-run gap.
- Done: matching/counting semantics -> no rematching, fixed native PCMerge/TMM/agglomeration order, all four exhaustive transition cells, descriptive counts, and no yield gate remain unchanged.
- Done: states/domains -> required TMM/noise/train/validation/neighbourhood/template/agglomeration states and frozen evaluation domains were inspected in source and fixtures.
- Done: circularity -> evaluation remains frozen before G; synthetic fixtures test helpers and failure injections but do not substitute for a managed real run.
- Done: reproduction/provenance -> corrected metadata-only disclosure, immutable packet hashes, source/runtime inventories, tests, and predecessor NO-GO bindings were verified.
- Done: defensible scope -> implementation preparation only; no scientific outcome.
- Not done: candidate fit, service installation/start, real input preflight, feature/residual/voltage payload access, GPU work, RF/holdout inspection, or outcome evaluation -> explicitly prohibited.
- Can establish: the six prior repair areas are largely addressed, but this exact v2.1 preparation remains unsafe to release because approval semantics are incompatible and failure evidence is not durable across all stop/publication paths.
- Cannot establish: releasable managed execution or any candidate-G scientific claim.
