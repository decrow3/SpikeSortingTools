# Targeted trained-entry repair candidate v2

Status: **execution disabled; independent H1 repair review requested; no launch GO.** This packet answers H1 review manifest `ad2eaac82c8848fa994ab7b0dae42564479081370b9ec4453ea2e7cc73114c03` and COMPLETE `6a87a38a57552907f1363c19d1102946e90f2b627d15cedf9a6ad6809ef16725` in a fresh planned namespace ending `_v2`.

No recording, voltage, sorting, training, detection, RF/holdout, parameter search, service, enabled config, or data launch occurred.

## Repair 1: semantic completion validation

Completion now parses all 30 native outputs and nine snapshot artifacts. It rejects empty/unparseable/nonfinite or wrong-rank arrays; verifies integer/index dtypes; aligns exported spike arrays, features and positions; validates full-table/kept/exported counts; aligns channel map/positions/shanks and square whitening matrices; aligns templates/similarity/indices; checks `ops.chanMap`; and verifies Wall3/wPCA/wTEMP/channel axes, iU/iCC bounds, `chanMap[iU]`, detection-template IDs, snapshot receipt ancestry and per-file hashes. Cluster TSVs and text outputs must be nonempty and parseable.

Large arrays use read-only memory mapping. Validation adds one bounded full-content finiteness/semantic scan over the saved outputs, O(total output bytes), with low resident-memory overhead rather than materializing all arrays. No real-output timing was measured; the 1800-second job ceiling is unchanged and launch review must budget this terminal scan explicitly.

The controlled positive fixture validates 3 exported spikes, 4 full rows, 4 channels, 2 final templates and 2 Wall3 templates before completion. A corrupted fixture truncates `spike_clusters.npy`; the real completion boundary rejects it at `validate_and_inventory_saved_artifacts`, preserves a failure receipt, and writes no completion.

## Repair 2: actual downstream hook exercise

The durable native-hook fixture installs `installed_phase_and_snapshot_hooks` on Kilosort 4.0.27 and invokes the actual native `template_matching.extract`, which invokes the actual native `prepare_extract`. A checking bfile asserts the snapshot receipt already exists before the first learned-extraction batch read. Native `prepare_extract` supplies iU/iCC/iCC_mask, the wrapper saves Wall3/wPCA/wTEMP/chanMap and physical `chanMap[iU]`, and native extract returns normally with zero synthetic events under a high threshold.

`fixture/HOOK_FIXTURE_RESULT.json` records: native prepare invoked, native extract returned, first batch observed the durable snapshot, and snapshot receipt hash `9883acb5...`. This is a controlled local hook compatibility check, not sorting/training/detection evidence or a scientific output.

## Verification

- 56 focused tests passed in 12.83 seconds.
- Final vertical fixture: positive semantic completion passed; downstream failure preserved; corrupted alignment rejected; missing-key pre-data gate passed; actual native prepare/extract hook exercised.
- Source repair diff: `trained_source_repair.diff`.
- Config repair diff: `trained_config_repair.diff`.
- New disabled config SHA-256: `1cd8354556ec6c5d8077128d08dd4fdaa4a8f898c5bf11999ca3c22d23596cf9`.
- New trained source SHA-256: `d75db9794a069e0bbded1ea218ae275eb6107dcdced45050cdfa30b77d40d7df`.

## Implementation checks

- Done: H1 repair decision packet and all members verified.
- Done: semantic validator traced across every declared native/snapshot file and positive/corrupt fixtures executed through the real completion boundary.
- Done: actual native prepare/extract hook signatures exercised with snapshot-before-first-batch assertion.
- Done: prior candidate remains immutable; new config is disabled and uses a fresh `_v2` planned namespace.
- Not done: validation against a real completed trained output -> none exists and no data execution was authorized.
- Not done: real terminal-scan runtime benchmark -> requires a completed output; mmap bounds resident memory but not I/O time.
- Can establish: the two H1 implementation gaps are repaired under controlled local known-answer inputs.
- Cannot establish: real-data hook behavior beyond the exercised signatures, full 600-second runtime, launch readiness, efficacy, identity/purity, or generalization.
