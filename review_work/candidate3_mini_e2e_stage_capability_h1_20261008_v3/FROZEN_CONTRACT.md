# Candidate 3 miniature end-to-end stage capability v3

This is a two-wiring-change repair of failed preserved v2: explicitly set the internal `detection_type="threshold"` to match the already frozen `ThresholdingConfig`, and disable the stock `save_everything_on_error` copier because `work_in_tmpdir=false` makes its source `None`; the run-specific directory and traceback remain preserved normally. No data, source adapter, hashes, scientific thresholds, stage selection, resources, predictions, accounting, or acceptance criteria change.

- Reuse the v2 frozen inputs exactly: ordinary SHA-256 `290221b1b91e5a5110ee845c5e42d044b6f2d7b3eb85564554d9c3cd624d3523`; transition `dc78516734e154e45bec8f29f8f757bff904bbd32938fe0a2e0d1767f41a46f2`; truth `0525b9838e98b07e210876a12f957573996fce5a11c21ce05bf49bdb8a3a9dc9` (2,297 events).
- Input: 297,000 frames at 29,999.835983263598 Hz, 24 channels, seed 38172, 12 deterministic units; ordinary `[0,0]` and exact-half transition `[0,20]` um.
- Boundary: actual `ExactLatticeRemapRecording` source `01253946...`; installed Kilosort `BinaryFiltered.filter` source `767b76a0...`; CAR/centering/300 Hz HP/infinite artifact threshold/no whitening; DARTsort `preprocessing="none"`.
- DARTsort: commit `edcfe1b5...`; CPU; threshold detection 40; no motion; one matching iteration; intermediate features/labels; fixed fitting limits and seed from v2.
- Required outputs: stage paths/schemas/counts, reload, event/sample/label/template lineage where present, ambiguous/noise accounting, compact observed voltage statistics, and A-vs-Candidate3 capability matrix using `saved`, `qualified_replay_only`, `unavailable`.
- A: input and final exports saved; pre-extraction detection bank and bound matching/refinement transitions unavailable. No A recreation or inferred detection agreement.
- Stop: preserve first failed stage; no threshold/config tuning after v3 outcome. Transition runs only after ordinary is resource-safe.
- Acceptance: adapter and entrypoint execute; at least detection saves/reloads; all later stages reported honestly. Synthetic capability only, no benefit/transfer/identity claim.
- Resource and holds: <=120 MB temporary synthetic voltage/intermediates; 45 minutes/arm; no accepted voltage, GPU, RF, holdout, or D3.
- Active plan SHA-256 `c26d63dfe9ae94312485d57ec99f1a3c4d7d089806b84485099d4afb4f12554f`.

This contract was frozen before v3 execution.
