# Candidate G v2.2 launcher delta review

## Verdict

`NO_GO_INTEGRATED_CANDIDATE_G_V2_2_PREPARATION`

Candidate G remains execution-disabled. This review did not run candidate G, install or start its real service, perform real input/resource preflight, access project payloads, use a GPU, or inspect RF/holdout data.

## Reviewed identities

- Preparation: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_f_fixed_event_candidate_g_preparation_20261004_v2_2_h5`
- Preparation MANIFEST SHA-256: `60f94197ac30c149d2eef59bef6037d7365fabd45e0047a36e26e2225d79ef53`
- Preparation COMPLETE SHA-256: `8fa2ca48ccf21a18a18a4fad3f5cb98fc306bcd9628af0e397275a1efd4b44ee`
- Canonical approval protocol SHA-256: `1ef84ece32802cbcfd1b204c2463dd99c953741f1023b66f96b7989f0c7ffb91`
- Coordination receipt SHA-256: `e912864afeca05cfc968c14101af1a8711893d41dd7816c50e50347e4100d596`
- Inherited v2.1 science MANIFEST/COMPLETE: `2ce97d278b3d40c0583e8d06ae2e74b3de651b2b99b156b583abe86b19a302e7` / `4823ab489675d4387e565ea58f7bf8c3fc1808239a31eda3c64dba9a795b3708`
- Scientific contract SHA-256: `5c02db316bff951676d5c062f46837fc24bde42b2de7fc536b704868812d58cd`

All 41 v2.2 manifest members matched their recorded sizes and SHA-256 hashes. The packet passed its content/inventory/nonce completion verifier. Acceptance no longer depends on mtimes.

## Blocking findings

### 1. Finalization is idempotent only after TERMINAL exists, not restart-safe while in progress

`finalize()` returns the existing terminal receipt under a lock, which makes completed calls idempotent. Before `TERMINAL.json` exists, however, it unconditionally creates or copies into fixed paths (`staged_result`, `incomplete_evidence`, and the publication partial) (`source/attempt_lifecycle.py:36-83`). If the finalizer is interrupted after staging, partial-evidence creation, or publication-copy creation, `reconcile()` calls `finalize()` again, but:

- `shutil.copytree(scratch, staged)` rejects an existing staged directory;
- `_copy_partial_evidence()` calls `dst.mkdir()` and rejects an existing evidence directory;
- an existing distinct publication partial triggers a failure rather than verifying/resuming/quarantining it.

The retry can then write a misleading terminal state such as `NOT_STARTED`/`NOT_ATTEMPTED` even when a complete staged scientific result exists. The synthetic idempotence test covers only a second call after `TERMINAL.json` already exists (`tests/test_seal_and_lifecycle.py:25-28`); it does not inject interruption at each state transition.

### 2. The production finalizer itself has no adequate lifetime guarantee

The real service places staging, verification, and publication in `ExecStopPost` but sets no `TimeoutStopSec` (`service/candidate-g-v2-2.service.template:5-15`). The actual rehearsals copied an 87-byte result. Candidate G can require copying and hashing a much larger sealed result to durable staging and then publication. Under the manager's default stop timeout, the finalizer can be killed before `TERMINAL.json`; blocker 1 then prevents reliable recovery. This does not yet satisfy the project requirement that consequential work and downstream publication survive launcher disconnection/termination with preserved final status.

### 3. Inherited-science “exact inventory” verification is not exact

`verify_legacy_exact()` validates the predecessor MANIFEST/COMPLETE hashes and every listed member but never compares the listed paths with the actual recursive file inventory (`source/seal_v22.py:64-73`). Added unlisted files therefore do not invalidate the inherited science root, even though the v2.2 worker imports runtime/source/configuration directly from that root. This contradicts the helper's exact-inventory claim and the requested inherited sealed binding. The v2.2 packet and new result verifier do perform exact inventory comparison; the legacy path needs the same unlisted-file rejection.

## Confirmed repairs and evidence

- The canonical protocol is used by contract, request, and gate. The gate requires exact field sets, exact GO/coordinator/release statuses, candidate G, one execution, exact scope, blocker-free H1, exact preparation/science/H1 hashes, installed-unit identity, and absent destination (`source/approval_gate_v22.py:10-52`). The valid-chain fixture passes; altered science hash, H1 NO-GO, wrong candidate, and missing scope fail.
- `seal_v22` binds exact content/inventory, product count, manifest hash, shared nonce, and explicit exclusive completion-write semantics without mtime acceptance (`source/seal_v22.py:21-62`). Content mutation fails while mtime mutation alone is accepted.
- Scientific and publication states are distinct. A sealed result is verified, copied to durable staging, reverified, copied to a run-specific partial, verified, atomically renamed, and finally verified; failed publication preserves staging and a distinct partial/quarantine (`source/attempt_lifecycle.py:43-82`). These paths work when the finalizer runs uninterrupted.
- The independent test rerun passed all 16 tests.
- The recorded huklaban5 systemd success rehearsal verifies `COMPLETE/PUBLISHED_VERIFIED`; its published and staged seals independently verify with MANIFEST `71ee2199ef49a38c46955017c5cb661b1896d3ce19345214cb2aa7e745df0b96` and COMPLETE `dde38d43dda17b2f161e2ebe16d73be3f2cf7f3d6cc6c5d2633742dcb65b2f40`.
- The forced two-second timeout rehearsal records systemd `Result=timeout`, `ExecMainCode=killed`, `ExecMainStatus=TERM`, scientific state `INTERRUPTED`, publication state `NOT_ATTEMPTED`, and preserves `progress.json`.
- Both rehearsal units record 200% CPU and 512 MiB memory limits, `CUDA_VISIBLE_DEVICES=-1`, zero project-payload bytes, and no GPU work. The packet reports 34,974 bytes in the source rehearsal evidence set; the compact published evidence subtree is 16,358 bytes.
- The saved `UNINSTALLED_STATE.txt` records that the real and temporary units were absent after rehearsal. A new live SSH check from huklaban1 was attempted but hostname resolution for `huklaban5` was unavailable, so current remote absence was not independently reconfirmed in this review.
- Worker diff inspection shows the PCMerge/TMM/template/agglomeration body, fixed rows, bindings, evaluation inputs, and resource contract remain inherited from sealed v2.1; the delta redirects science imports to v2.1 and changes approval/lifecycle/sealing only.

## Required repair

Create a new immutable delta. Make finalization a persisted state machine that validates and resumes each existing stage/partial after interruption, with tests that terminate after every durable transition and reconcile successfully. Give the production finalizer an independently managed lifetime and explicit resource/time bounds sufficient for worst-case staged verification/publication, or move it to a separate durable managed unit. Extend legacy verification to reject every unlisted file. Retain distinct partial/quarantine artifacts and never overwrite them. The real candidate-G service must remain absent until a new H1 GO and coordinator actual input/resource preflight and release.

## Implementation checks

- Done: what actually ran -> independently verified 41 preparation products, reran 16 tests, verified saved success/staged seals, and inspected systemd success/timeout logs and terminal receipts.
- Done: intended delta -> source diff confirms launcher approval/lifecycle/sealing changes while v2.1 scientific rows, pipeline, inputs, and evaluation remain unchanged.
- Done: axes/clocks/caps -> inherited unchanged; no project input was opened. Rehearsal systemd evidence confirms 2 CPUs, 512 MiB, 10-second success/2-second timeout, no GPU.
- Done: matching/counting/states/domains -> inherited from exact named v2.1 science hashes; no outcome or evaluation was run.
- Done: circularity -> launcher rehearsals use a synthetic result and do not inspect G outcomes.
- Done: provenance -> content seals, success and timeout receipts, journals, unit definitions, predecessor NO-GO, and source delta were inspected.
- Not done: real candidate-G service install/start, actual resource/input preflight, payload reads, GPU work, fit, evaluation, RF/holdout, or live remote service confirmation -> prohibited or unavailable.
- Can establish: canonical approval vocabulary and uninterrupted success/timeout lifecycle paths behave as reported; the rehearsals themselves are credible and bounded.
- Cannot establish: restart-safe finalization, inherited exact inventory, a sufficiently durable real finalizer, production release safety, or any candidate-G scientific result.
