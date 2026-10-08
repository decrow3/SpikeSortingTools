# H5 corrected waveform-QC one-shot execution failure

Status: **FAILED; the one authorized invocation was consumed and was not retried.**

The reviewed source raised `KeyError: 'unit_channel_identities'` at source line 259 after it had verified every configured input hash, loaded the metadata and both retained NPZ archives, checked archive keys, materialized the retained arrays, enforced the numeric-byte ceiling, and created the fresh output directory. The approved config contains no `unit_channel_identities` node. The output directory is empty; no sample, slot, comparison, decision, or completion artifact was produced.

This is a post-hoc implementation failure report. It does not change or reinterpret the immutable historical QC `FAIL`, its four unit-278 remap failures, any historical global-peak distances, visibility flags, or metric tables. There is no corrected `PASS`, `CAUTION`, or `FAIL` verdict and there are no corrected waveform-QC metrics.

No retry, source patch, config patch, recording read, waveform capture, waveform transfer, training, detection, sorting, RF/holdout access, or waveform publication occurred.

## Implementation checks

- Done: candidate packet and independent H1 GO packet hashes and members were verified before launch -> matched the authorized manifests and completion receipts (see `PROVENANCE.json`).
- Done: enabled-config delta was restricted to status, the H1 manifest binding, three execution gates, and a fresh output root; all other reviewed config nodes remained identical -> approved config SHA-256 `69e22f44f523acac2d3e765b3b84262cec797c0faec5e3b81264ab17a02883a9` (see `PROVENANCE.json`).
- Done: fresh-path, no-competing-process, interpreter, resource-limit, and historical-result preservation checks ran before the only invocation -> passed (launch receipt SHA-256 `4f5a7a1dd229da1145108b108f221cc3ada52b4bf7db10c3dd1c098d9aa3bd74`).
- Done: executed failure path was inspected -> source line 259 reads `config["unit_channel_identities"]`, while the exact approved config contains no such key; traceback SHA-256 `831437297c3f92eb9c3f6b5111e6a31a969c33b6c77622caa55a3cc3d76af064`.
- Done: managed service state was checked after exit -> `failed/failed`, result `exit-code`, status 1, PID 0, restart count 0, CPU 0.280813 seconds (see `UNIT_STATUS.json`).
- Not done: per-member shape/dtype checks, per-sample remap, controls, local detectability, QC metrics, or verdict -> execution stopped at the missing config key before the member loop.
- Not done: peak memory -> the transient unit did not retain a `MemoryPeak` value after exit.
- Can establish: reviewed source/config pair was not runnable on its authorized executed path because a required top-level config node was absent; the only GO invocation failed before metric computation.
- Cannot establish: corrected remap integrity, local detectability, a corrected technical verdict, biological identity, localization, purity, motion-estimator correctness, or sorter efficacy/benefit.

All hashes in `MANIFEST.sha256` were verified before `COMPLETE.json` was written last.
