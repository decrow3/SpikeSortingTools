# Execution-disabled trained support-mask entry point

Status: **implementation complete; execution disabled pending independent H1 delta review and a separate GO.** No recording, saved voltage, sort, training, detection, RF, holdout, parameter search, service, or real-data job was opened or launched while building this packet.

## Milestone and decision

- Milestone: freeze one runnable non-smoke trained source/config pair around the unchanged reviewed native pre-W consumer hook.
- Decision this packet can change: whether the trained orchestration delta is implementation-complete enough for H1 to authorize a later, separately enabled 600-second medium run.
- Cheapest adequate check completed: actual CLI -> full-structure synthetic config -> required-key gate -> reviewed metadata/ExactLattice validator adapter -> unchanged native I/O consumer hook -> stage-aware orchestration -> artifact inventory -> completion/failure receipts.
- Completion condition: runnable pair, exact recursive delta, controlled vertical success/failure/early-rejection checks, comparator bindings, saves, prerequisites, and resource estimate are all immutable here. Real execution remains out of scope.

## Runnable pair

- Source: `source/kilosort_support_mask_trained.py`
- CLI: `tests/en_minimum_training_support_mask_trained_launch.py`
- Full config: `config/en_minimum_training_support_mask.trained.review_pending.v1.json`
- Exact recursive delta from the reviewed smoke base: `config/en_minimum_training_support_mask.trained.review_pending.v1.delta.json`

The config is not a reduced toy: it retains the complete recording, crop, field, ops, frozen Kilosort settings, covariance gate, invocation, sources, and prohibitions. It adds trained mode, H1 approval binding, comparator provenance, resource limits, a complete save contract, and a never-used trained namespace. `execution_enabled=false` and the H1 manifest is null.

The reviewed consumer implementation is included byte-identically as `source/kilosort_support_mask_candidate.py` (`705b4af...`) plus its unchanged geometry helper (`9c9810...`). The trained module wraps rather than edits that consumer. It never imports a smoke C/W or learned bank: covariance, W, wPCA, wTEMP, universal detections/features, Wall3, iU/iCC/mask, learned extraction, final clusters, and final templates are declared fresh per run.

## Trained orchestration delta

The trained boundary:

1. hashes and parses the config, then validates every required structural key before hashing or opening any bound source, manifest, field, or ops path;
2. uses the reviewed metadata/ExactLattice validator through a schema-only adapter while retaining the complete config tree;
3. atomically reserves run, evidence, snapshot, and native-result directories;
4. installs the unchanged single pre-W support hook and actual audited covariance/W constructor without a smoke stop;
5. phase-stamps support reads for covariance/W, PCA/universal-template learning, universal detection/features, and learned-template extraction;
6. snapshots Wall3 and its wPCA/wTEMP/chanMap/iU/iCC/iCC_mask ancestry when native `prepare_extract` has constructed those mappings and before the learned batch loop;
7. validates 30 required native files and nine snapshot files, hashes them, and writes `COMPLETE.json` last;
8. on any exception, writes the exact stage, exception, phase, mask-read count and whitening audit, and never writes completion.

## Vertical fixture

`fixture/VERTICAL_FIXTURE_RECEIPT.json` records an actual CLI invocation using a full structural clone of the trained config with all data-bearing paths substituted beneath a fresh local fixture root. The substitutions are explicitly orchestration-only: an empty binary path (not opened), four-channel metadata, local field/ops/source marker, CPU, synthetic native runner, and small placeholder save files. It performed no sorting, training, detection, or scientific computation.

- Success returned 0, reached completion validation, inventoried all 30 native plus nine snapshot artifacts, and wrote completion last.
- Stage audit contains one controlled hook read in each of the four required phases.
- Injected post-whitening failure returned nonzero, preserved stage `native_trained_pipeline`, a passing covariance receipt and whitening hash, and wrote no completion.
- Missing `trained.required_snapshot_files` returned nonzero before either deliberately nonexistent source or manifest sentinel could be accessed.
- Focused suite: 54 passed in 9.58 seconds (`test_kilosort_support_mask_trained.py`, unchanged consumer tests, and support-helper tests).

The synthetic artifact arrays are not native sorter outputs and are not evidence of algorithm efficacy. Their only purpose is to prove orchestration, validation, save inventory, and terminal-state behavior.

## Comparator bindings

`COMPARATOR_PROVENANCE.json` binds the completed local REF384 and B384 roots by their existing receipts, status, ops, pre-extraction receipt, recording-manifest identities, common-screen completion and frozen config. No raw recording was rehashed and no comparator array was copied or opened for values.

## Future medium-run prerequisites

Before any native data execution:

1. H1 must inspect the exact delta and source and publish a review manifest.
2. A new enabled config may change only approval/status/execution gate and fresh run paths; its exact diff must be frozen.
3. Revalidate all bound source, recording-manifest, field, ops and comparator metadata hashes; stat but do not rehash the raw binary unless separately required.
4. Confirm the 600-second crop `[208498882,226498783)`, 384-row identity geometry, 29,999.835983263598 Hz clock, `nblocks=0`, CAR, thresholds 12/9/6, and no preprocessed copy.
5. Confirm one GPU, <=64 GiB RAM, <=16 CPU threads, >=8 GiB result space per fresh arm, managed-service durability, and a fresh namespace.
6. Run repaired B384 from scratch. If the comparison design retains REF384 repeatability gating, run the exact REF384 repeat separately from scratch; neither run is resumable within an interrupted sort.

Estimated repaired-arm runtime is 600–720 seconds, based on observed 120-second and full-session Kilosort runs. Its closest completed B384 output is 1,983,697,263 file bytes. Repaired B384 plus REF384 repeat is estimated at 1,200–1,440 seconds and about 3,965,904,536 file bytes before logs/evidence; the frozen 8 GiB per-arm ceiling is conservative.

## Preserved failure and parked work

The corrected waveform-QC retry remains parked. Its consumed one-shot failure is preserved by manifest `34a2b053f7c368a83400df607d1c418c98bf4a2bf8a06f997d2fa6470b7d2d15`; this packet neither repairs nor executes it.

## Implementation checks

- Done: actual unchanged consumer and installed Kilosort phase call sites inspected -> one pre-W hook remains shared by covariance, PCA learning, universal detection, learned extraction, and optional preprocessing export (source snapshots in this packet).
- Done: exact source/config/runtime delta frozen -> full config SHA and recursive 132-node diff are bound in the manifest.
- Done: early structural gate tested -> missing required key fails before deliberately nonexistent source/manifest sentinels.
- Done: actual CLI/config/validator/adapter boundary invoked -> full-structure local synthetic success reaches validated artifact construction and completion.
- Done: downstream failure tested -> stage, exception and whitening evidence survive; completion is absent.
- Done: original consumer regressions and trained tests -> 54 passed.
- Done: data-local comparator identities checked -> compact receipt/status/ops/snapshot-receipt hashes recorded without raw rehash or array copying.
- Not done: independent H1 source/delta review -> required before any enabled config.
- Not done: native medium data execution or output validation -> deliberately prohibited for this packet.
- Not done: current GPU/free-space/service preflight -> launch-time prerequisite, not evidence from this preparation.
- Can establish: the trained source/config pair is structurally complete, execution-disabled, vertically runnable under controlled orchestration, fail-closed before data access for missing keys, stage-aware, and completion-gated on the required saved artifacts.
- Cannot establish: successful native training/sorting on the 600-second crop, run-to-run comparator stability, scientific benefit, biological identity, purity, localization, motion correctness, long/full generalization, or RF performance.
