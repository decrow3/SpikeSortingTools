# Candidate G v2.2.1 final focused H1 review

## Verdict

`GO_INTEGRATED_CANDIDATE_G_V2_2_PREPARATION_EXECUTION_REMAINS_DISABLED`

This is a blocker-free implementation GO for the exact v2.2.1 preparation only. Candidate G remains execution-disabled pending a separately sealed coordinator release and actual input/resource preflight. This review did not install or start the real service, run candidate G, access project payloads, use a GPU, or alter science/evaluation.

## Reviewed identities

- Preparation: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_f_fixed_event_candidate_g_preparation_20261004_v2_2_1_h5`
- MANIFEST SHA-256: `3b124fbe3c079d5f4a121fb196e6a0e510fc66449ba8c07705319bdd36621a69`
- COMPLETE SHA-256: `d3ef7220a7f36f5db8c007132c7fb57c72ab4750e0ce1f7be0048205acf6d1fb`
- H1 request SHA-256: `fc25e9d86fa7f21092f33318202815893af9d944ee08805d6d05bc11e0d06638`
- Coordination receipt SHA-256: `e4f19daa783fe458368d2135c5d0db10e9d7083dda3a3ae6e4ce6fd0e65dd4a6`
- Live huklaban5 unit-absence receipt SHA-256: `3c70770b88eeefc26e43618fe94af405df2fbe4332954e3e2f8b04f9a4982c86`
- Canonical protocol SHA-256: `1ef84ece32802cbcfd1b204c2463dd99c953741f1023b66f96b7989f0c7ffb91`
- Inherited v2.1 science MANIFEST/COMPLETE: `2ce97d278b3d40c0583e8d06ae2e74b3de651b2b99b156b583abe86b19a302e7` / `4823ab489675d4387e565ea58f7bf8c3fc1808239a31eda3c64dba9a795b3708`
- Scientific contract SHA-256: `5c02db316bff951676d5c062f46837fc24bde42b2de7fc536b704868812d58cd`

All 48 v2.2.1 preparation products passed the packet's exact content/inventory verifier. An independent focused rerun passed 22 tests. Independent legacy verification matched all 190 v2.1 products and rejected the added-file fixture.

## Focused blocker findings

### Restart/adoption/quarantine — resolved

Finalization now persists an fsynced checkpoint after science verification, durable staging, publication-partial verification, destination rename, destination verification, and incomplete-evidence verification (`source/attempt_lifecycle.py:10-15,31-90`). On retry it validates and adopts complete staged results, build directories, publication partials, and destinations. Invalid or incomplete artifacts are atomically moved to uniquely named quarantine paths before reconstruction; none is overwritten (`source/attempt_lifecycle.py:35-69`).

Injected interruption after each of the five science/publication checkpoints and the evidence checkpoint resumes to the same terminal state. Corrupt staged and publication-partial fixtures are quarantined and rebuilt. Calls after `TERMINAL.json` remain idempotent (`tests/test_seal_and_lifecycle.py:53-73`).

### Finalizer lifetime — resolved

The production `ExecStopPost` no longer hashes or copies results. It invokes only `record-stop` through a five-second hard timeout, with `TimeoutStopSec=10`; the path performs one atomic receipt write and one event append (`service/candidate-g-v2-2.service.template:7-15`; `source/attempt_lifecycle.py:26-30`).

Normal success performs resumable finalization while the worker remains under the original managed service. Recovery uses a separate oneshot helper bounded to 1,800 seconds, 200% CPU, 512 MiB, and `CUDA_VISIBLE_DEVICES=-1`; repeated helper runs resume from durable checkpoints without refitting (`service/candidate-g-v2-2-finalize@.service.template:4-13`). Thus bulk staging/hash/publication work is outside the tightly bounded stop hook and remains independently managed.

### Inherited exact inventory — resolved

`verify_legacy_exact()` now compares the complete recursive actual file inventory with the manifest after checking the exact MANIFEST/COMPLETE hashes and every member's size/hash (`source/seal_v22.py:64-76`). The actual sealed v2.1 science packet passes with 190 products; a synthetic added file fails.

### Scientific/publication states and provenance — resolved for preparation

Scientific and publication states remain separately checkpointed. A scientific result must verify before durable staging, and staging must verify before partial publication. The partial verifies before rename and the final destination verifies afterward. Invalid destinations are quarantined and rebuilt from the durable stage.

The authoritative huklaban5 receipt records, at `2026-10-04T02:35:31-07:00`, the real candidate service and both rehearsal units as not found, with zero matching installed unit files and zero loaded units. Candidate G was not installed or started by this review.

## Remaining release prerequisites, not H1 blockers

- A coordinator must issue a new immutable release packet binding this exact preparation, H1 verdict, scientific contract, main unit, run ID, output and managed resources.
- Before starting, the coordinator must verify actual production input identities, the 4-CPU/64-GiB/one-GPU/3,600-second/30-GiB-scratch conditions, installed main-unit bytes, absent output/partial, and the exact bounded helper installation or invocation path.
- The managed runner must preserve STARTED/STOP/CHECKPOINT/TERMINAL receipts and inspect actual service state rather than infer liveness from logs.
- Any failed or interrupted attempt must be reconciled through the bounded helper before retry; no scientific or publication artifact may be overwritten.

## Defensible scope

This GO establishes that the three prior launcher blockers have focused implementations, negative controls, known-answer fixtures, and interruption/recovery coverage adequate for a separately controlled release decision. It does not establish real host resource feasibility, actual input preflight success, candidate-G completion, cache-read behavior, scientific continuity, refractory behavior, waveform preservation, biological identity, purity, or G-vs-F superiority/inferiority.

## Implementation checks

- Done: what actually ran -> verified 48 preparation products, reran 22 focused tests, independently verified all 190 inherited v2.1 products, and inspected the exact unit-absence receipt.
- Done: intended delta -> source diff and contract bind changes to lifecycle/checkpoints, strict legacy inventory, and service finalization layout; science/evaluation/native refinement remain at the exact inherited hashes.
- Done: axes, clocks, matching, counting, states and domains -> unchanged from the sealed v2.1 science packet; no outcome was inspected.
- Done: caps/defaults -> stop hook has a five-second hard bound; recovery helper is capped at 1,800 seconds, two CPUs, 512 MiB, and no GPU. Real-run caps remain subject to coordinator preflight.
- Done: circularity -> synthetic interruption fixtures and unit-absence checks do not use candidate-G outcomes.
- Done: provenance/reproduction -> exact packet inventories, canonical protocol, predecessor NO-GO, atomic checkpoints, quarantine behavior, and authoritative H5 absence receipt were inspected.
- Not done: real service install/start, actual production input/resource preflight, payload reads, GPU work, fit, evaluation, or RF/holdout -> prohibited and reserved for a separate coordinator release.
- Can establish: the exact v2.2 NO-GO blockers are repaired sufficiently for preparation-level GO while execution stays disabled.
- Cannot establish: production execution success or any candidate-G scientific claim.
