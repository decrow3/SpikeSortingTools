# Candidate G production-wiring delta review

## Verdict

`NO_GO_INTEGRATED_CANDIDATE_G_PRODUCTION_WIRING`

Candidate G remains execution-disabled. This review did not install or start the real service, create a real run tree, access project payloads, use a GPU, run candidate G, or alter science/evaluation.

## Reviewed identities

- Packet: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_f_fixed_event_candidate_g_production_wiring_repair_20261004_v1_h5`
- MANIFEST SHA-256: `a29db22c88a5c7f953b0156fc7eeb8cd5327939cf793913d061de4547ebfc7d4`
- COMPLETE SHA-256: `02684f9ef4d85a19a150aaf4d09094b79ca96ab9a3c78be1f7c43936fbadb40d`
- RESULT SHA-256: `889ddbad4167c9cb34a77cc75a985b4a83a2a852d24120e9726fe9cc13926280`
- H1 request SHA-256: `9446b2ae9c8164a463be26c6f4c2bce2fd5f20ec580fa3a43a83c4f0f6476bc8`
- Repaired preflight GO COMPLETE SHA-256: `06473cfcc77502295e473564cd63b91527f2f4d0599addd5e219337bca3d349c`
- Coordination receipt SHA-256: `44d16b4eb277352395bb6ae44ef604031ae7025d607d531acee83000c5d98ca7`
- Scientific contract SHA-256: `5c02db316bff951676d5c062f46837fc24bde42b2de7fc536b704868812d58cd`

All 70 packet products passed the content/inventory verifier. Independent reruns passed all nine new production-wiring tests and all 22 inherited v2.2.1 fault tests.

## Blocking findings

### 1. The 30 GiB policy is not an enforced peak bound

`storage_guard_v1.check()` sums the current run-tree and three external paths only when invoked, then separately checks that the declared estimate plus 64 MiB is below 30 GiB (`source/storage_guard_v1.py:4-11`). The service invokes it once after creating the run tree and the worker invokes it once after the scientific result is complete, before compact publication (`service/candidate-g-production.service.template:23-27`; `source/candidate_g_worker.py:239-243`). There is no filesystem/project quota, reservation ledger, continuous accounting, per-write enforcement, or post-publication aggregate check.

Consequences:

- transient peak use by result, models, checkpoints, Numba cache, temp files, logs and compact-publication partials can exceed 30 GiB between samples;
- the release gate accepts any positive `native_required_output_estimate_bytes` up to 30 GiB rather than requiring the frozen `25,769,803,776` bytes (`source/release_gate_v1.py:27-28`), so the coordinator could lower the estimate without rejection;
- compact publication is created after `storage_postscience.json`, and a failed oversized compact partial is not included because accounting names only the final compact path;
- systemd journal output is outside the declared run-tree `logs` directory and is not counted.

The test at `tests/test_release_wiring.py:57-60` validates point-in-time addition of four small fixtures, not peak enforcement. This does not meet the requested aggregate peak cap across the full run/unit/receipt/cache/log/publication lifecycle.

### 2. H1 and preflight semantics are hash-bound but not identity-bound to the release

The acyclic release/unit/receipt construction is sound as far as content hashes go: release binds template/renderer; rendered unit binds release hash; launch receipt binds release/unit/template/renderer; pre-start rerenders and compares exact unit bytes (`source/release_gate_v1.py:34-41`). The release does not contain a self/transitive unit or receipt hash.

However, the gate accepts an H1 verdict after checking only `status`, `candidate`, and empty blockers (`source/release_gate_v1.py:22-26`). It does not require an exact H1 verdict schema/field set or verify that the verdict reviewed this production-wiring preparation's path, MANIFEST, COMPLETE, scientific contract, preflight, template, renderer, or resource policy. The positive fixture deliberately uses a three-field minimal verdict (`tests/test_release_wiring.py:13-16`). A blocker-free GO for another packet with the same status can therefore pass.

Likewise, the preflight RESULT is checked only for its sealed hash and GO status (`source/release_gate_v1.py:20-21`). Its candidate, preparation identity, frozen-input hash, resource values and paths policy are never compared with the release. Thus the requested preparation -> H1 -> preflight -> release semantic chain is incomplete even though each referenced artifact is individually sealed.

### 3. Runtime-origin evidence is recorded, not enforced, and does not describe the worker's DARTsort import

The unit correctly invokes the reviewed venv and sets run-specific `NUMBA_CACHE_DIR`, `TMPDIR`, thread variables and GPU token. `runtime_probe_v1.py` records the executable, prefix, PYTHONPATH, cache path and six module origins, but it never raises if `venv_verified` is false or an origin/version differs (`source/runtime_probe_v1.py:1-9`). The release gate and worker load the receipt but never validate those fields (`source/candidate_g_worker.py:127-135`).

The saved zero-data receipt reports DARTsort from `/home/huklaban5/Documents/DARTsort/src/dartsort`. The actual worker subsequently prepends the sealed v2.1 `source` and `runtime` directories before importing DARTsort (`source/candidate_g_worker.py:123,156-165`), so its DARTsort origin is the sealed v2.1 runtime, not the origin recorded by the pre-start probe. This may be the intended frozen science, but the receipt is not an accurate assertion of the worker's actual module origins. No live check verifies the running unit/cgroup, process argv, environment or imported module paths before payload access.

## Confirmed repairs

- Main execution uses `/home/huklaban5/Documents/DARTsort/.venv/bin/python`; Numba and temp paths are run-specific and writable in the host-local run tree.
- The finalizer template uses the exact `reconcile --attempt-dir PATH` CLI and `Type=exec`, so `RuntimeMaxSec` is effective.
- The layout has one host-local authoritative result tree and separate lifecycle records. Compact publication copies manifests and small receipts only; no bulk result is transferred to shared storage.
- Gate execution precedes run-tree creation in the systemd `ExecStartPre` order.
- Rendered unit bytes, launch-receipt hashes, release hash, template hash and renderer hash form an acyclic construction and are rechecked pre-start.
- Managed synthetic success reached a complete host-local result plus compact report. Terminal failure produced `FAILED_SEALED` and the exact finalizer argv reconciled it. The preserved journal documents and corrects the earlier oneshot timeout issue.
- The final absence receipt records the real and all rehearsal units as not found, with zero installed or loaded matching units.
- The repaired input preflight is separately sealed GO with zero project-payload bytes and binds the exact v2.2.1 preparation, input identity and declared resources.
- AST regression checks confirm unchanged materialization, pipeline loader, file copy, native cluster/config/input/terminal calls. Science, seed and evaluation hashes are unchanged; the predecessor NO-GO remains preserved.

## Required repair

Create a new immutable wiring packet. Enforce aggregate storage with a real quota/reservation design covering the run tree, all external lifecycle/unit/receipt/journal or bounded-log artifacts, compact partial/final publication, and transient peaks; bind the exact frozen estimate and run a final post-publication check. Define exact versioned H1 and preflight semantic schemas and compare their reviewed preparation/science/resource identities to the release. Make the runtime probe fail closed on the exact executable/prefix/environment/origin policy, or change it to record the worker's effective post-`sys.path` module origins before project payload access; validate the receipt in the worker. Preserve the acyclic construction and all failed artifacts.

## Implementation checks

- Done: what actually ran -> verified 70 packet products; independently reran 9 new and 22 inherited tests; inspected managed success/failure/finalizer journals and unit-removal receipt.
- Done: intended delta -> AST and hash evidence supports unchanged science/evaluation/native calls; wiring, lifecycle, storage and publication are the changed domain.
- Done: axes/clocks/matching/counting/states/domains -> inherited unchanged and not rerun; no G outcome inspected.
- Done: silent caps/defaults -> identified that CPU/RAM/runtime values are represented, but storage peak and runtime-origin policies are not fail-closed.
- Done: circularity/provenance -> unit/release/receipt hashes are acyclic; H1/preflight semantic identity links remain incomplete.
- Not done: real service installation/start, actual run tree, project payload read, GPU work, G fit, evaluation, RF/holdout, or final coordinator release -> prohibited.
- Can establish: several production wiring repairs are correct and synthetic managed paths behave as reported.
- Cannot establish: enforceable 30 GiB peak compliance, exact approval/preflight semantic chain, effective runtime provenance, production launch safety, or any candidate-G scientific result.
