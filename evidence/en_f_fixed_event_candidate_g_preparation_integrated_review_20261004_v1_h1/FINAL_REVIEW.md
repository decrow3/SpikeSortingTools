# Candidate G integrated implementation-first review

## Verdict

`NO_GO_INTEGRATED_CANDIDATE_G_PREPARATION`

Candidate G remains execution-disabled. This review did not run candidate G, install or start a service, read recording voltage, feature payloads, or residual payloads, use a GPU, or access RF/holdout data.

## Reviewed packet

- Path: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_f_fixed_event_candidate_g_preparation_20261004_v1_h5`
- MANIFEST.json SHA-256: `54f45746366c4c149f170364e3e38ad5a533e12dd363e5310f0df7a8d72e014b`
- COMPLETE.json SHA-256: `794c47e0d3903a0f63b846be8a037f8bc6544df8563528486c5ee417befd2543`
- Frozen execution-disabled contract SHA-256: `5c02db316bff951676d5c062f46837fc24bde42b2de7fc536b704868812d58cd`
- H1 review request SHA-256: `4cda5e407f042cbcda8f5976c5359a875ac8bd5e6b721a05e9610f039401509e`
- Independent manifest verification: all 159 listed products matched recorded byte counts and SHA-256 hashes.

## Blocking findings

1. The release gate is not semantically or operationally fail-closed. It hash-checks arbitrary H1/coordinator files but never parses their verdict/status; its own positive test accepts `go\n` and `release\n`. It compares a declared `managed_job` string but does not verify that execution is actually under the required systemd user service/cgroup or that the resource limits are active (`source/candidate_g_worker.py:42-66`; `tests/test_candidate_g.py:182-217`).
2. The preparation seal is incompletely verified before execution. The gate checks only the COMPLETE file hash and its manifest hash, not COMPLETE status/product count or every manifest member. The reviewed packet also fails the project publication invariant that `COMPLETE.json` be written last: its filesystem mtime precedes MANIFEST, README, source, tests, and configuration products, despite the marker's `written_last_utc` claim.
3. Frozen input identities are not validated before payload access. `materialize_fixed_h5` checks only the total row count, then reads datasets alphabetically. It does not first validate the frozen file size, exact schema, completed-chunk metadata, sampling frequency, or W3 boundary times (`source/candidate_g_worker.py:91-119`; contract lines 19-24, 47-56). The model directory is copied without content verification; motion, preprocessing-cache/checkpoint, and historical-F hashes/metadata are not checked before their use (`source/candidate_g_worker.py:143-177,211`). A substituted same-row-count H5 or changed auxiliary input can therefore cause payload access or a scientifically different run before rejection.
4. The run does not establish the required managed resource conditions or a complete terminal evidence seal. The resource receipt repeats release-declared limits rather than validating active cgroup/GPU/scratch conditions; no per-stage timing is emitted; `RUN_COMPLETE.json` covers only status, row count, and final NPZ hash, with no authoritative output inventory/manifest, state completeness validation, or COMPLETE-last seal (`source/candidate_g_worker.py:220-235`; contract lines 88-98, 131-140, 161-166).
5. Cache read reservation is process-shared, bounded, and occurs before the parent read, but the counter update and call-log append use separate locks. A crash between them can leave the final summary lower than the reserved counter, and `summary()` does not reconcile the two (`source/fixed_event_meter.py:25-48`). This is a repair requirement for authoritative accounting, although it remains cap-conservative.

## Confirmed implementation properties

- The frozen scientific contract specifies exact W3 frames `[239998073,250197991)`, exact parent rows `[18112500,18603444)`, 490,944 rows, PCMerge then TMM then fresh templates/agglomeration, no rematching, and seed 0.
- Materialization crops every row-aligned dataset to the fixed parent-row slice and preserves fixed datasets; native sorting order is retained rather than independently resorted (`source/candidate_g_worker.py:91-119,174-175`).
- Immutable stage bindings cover parent/global/local time, channel, matching label, feature row/content hash, and TMM neighbourhood; only candidate labels vary. Stage tables are validated across all three stages (`source/candidate_g_worker.py:187-210`; `source/fixed_event_bindings.py`).
- The reviewed patches capture TMM/noise/train/validation/neighbourhood state before deletion and capture fresh templates/agglomeration arrays on both native return paths (`source/fixed_event_state_capture.py`; `source/fixed_event_agglomeration_capture.py`; patched `mixture.py` and `agglomerate.py`).
- The metered recording rejects cache access outside `[239997994,250198070)` and reserves logical bytes before reading (`source/fixed_event_meter.py:104-115`). The frozen plan retains the 12 GiB logical cap without silent scientific reductions.
- The accepted origin fixture is byte-identical for three frozen 4,096-frame by 182-channel float32 windows. The duplicate successful run is disclosed and preserved: 8,945,664 bytes each, 17,891,328 bytes cumulative. It does not itself block the preparation design, but neither run is candidate-G scientific evidence.
- Seventeen synthetic tests passed. The primary G-F decision is frozen at strict CI bounds beyond +/-0.05 with 2,000 common-block draws/seed 1729; H is contextual, counts descriptive, there is no yield gate, lags 9-29 are segment-safe, RF/holdout are excluded, and waveform guardrails remain unmeasured (`evidence/EVALUATION_FREEZE.json`; `tests/test_candidate_g.py:220-229`).

## Required repair before another integrated review

Create a new immutable preparation namespace; preserve this v1 packet unchanged. The replacement must: parse and validate the allowed H1/coordinator statuses and full packet seal; verify actual systemd/cgroup/resource context before output or payload access; validate every frozen input identity and W3 boundary/schema condition before feature/residual/voltage/model use; reconcile meter reservations to the call log; validate every required state/output and stage timing; and publish an authoritative manifest with `COMPLETE.json` genuinely written last. Candidate G must remain disabled until that replacement receives a new integrated GO and a separate coordinator release derivative.

## Implementation checks

- Done: what actually ran -> verified all 159 packet members, frozen source/config/runtime snapshots, preserved fixture/test receipts, and confirmed no candidate-G run/service/GPU work is represented; preparation packet hashes above.
- Done: intended arm differences -> contract explicitly distinguishes exact-W3 PCMerge/TMM population, fresh native-noise fit, and fresh post-TMM templates/agglomeration from historical F; historical F is not a paired fresh-noise baseline (contract lines 58-87).
- Done: axes/frames/coordinates -> exact global frames, parent rows, local-time transform, channel and neighbourhood bindings are frozen and stage-table validation preserves native order (contract lines 47-56; worker lines 174-210).
- Done: clocks/time bases -> frozen sampling rate and half-open supports were inspected, but the worker does not validate them against actual inputs before payload access; blocker 3.
- Done: silent caps/defaults -> native plan/read derivation specify 500 spikes/unit, 121 samples, 182 channels, one template pass and a 12 GiB cap; worker checks the stage plan but not all frozen input/resource conditions.
- Done: matching/counting semantics -> no rematching is allowed; all four F/G transition cells are frozen and tested; counts remain descriptive.
- Done: states/domains -> negative excursion, outside-mask flat, and catalogue remainder are frozen; common later-endpoint blocks and segment-safe lags are specified.
- Done: circularity -> outcome rules were frozen before G; the cache-origin fixture tests independent direct-memmap equivalence, but synthetic gate tests do not establish real systemd/seal semantics.
- Done: reproduction/provenance -> source/config/tests and preparation receipts are present and hash-consistent, but no G outcome exists and terminal run sealing/state completeness are inadequate; blockers 2 and 4.
- Done: defensible scope -> preparation demonstrates several synthetic invariants and a bounded read design only.
- Not done: real feature/residual/voltage reads, candidate fit, service launch, GPU work, RF/holdout access, or outcome evaluation -> prohibited by the review request and unnecessary for finding the release blockers.
- Can establish: this exact preparation packet must not be released; specific fixed-row/binding/state-capture/meter-bound/evaluation-freeze components are promising but require a new sealed repair packet.
- Cannot establish: candidate-G executability under managed limits, complete input identity, scientific continuity, refractory improvement, waveform preservation, identity, purity, or superiority/inferiority versus F.
