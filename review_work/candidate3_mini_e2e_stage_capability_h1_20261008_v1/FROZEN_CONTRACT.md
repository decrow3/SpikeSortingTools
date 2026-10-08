# Candidate 3 miniature end-to-end stage capability: frozen contract

- Milestone: candidate-3 snippet/readiness.
- Decision changed: identify which prospective Candidate-3 stage artifacts and lineage can actually be saved/reloaded, and which exact-A comparator stages exist, before any real-voltage snippet.
- Cheapest adequate test: two deterministic synthetic 9.9 s, 24-channel inputs (ordinary zero shift; one exact-lattice 20 um transition at 4.95 s) through the actual `ExactLatticeRemapRecording` -> installed Kilosort `BinaryFiltered.filter` prewhitening boundary -> DARTsort `preprocessing="none"` entrypoint. This is synthetic capability evidence only.
- Synthetic source: seed 38172; 29,999.835983263598 Hz; 297,000 frames; 24 channels at 20 um pitch; int16 Gaussian noise sigma 20 counts; 12 units; deterministic 20 Hz schedules; 61-sample negative waveforms; unit amplitudes 180..290 counts. Truth events within waveform margins only.
- Transition adapter: temporal centers `[2.475,7.425]` s, 4.95 s cells, shifts `[0,20]` um; ordinary shifts `[0,0]`. Geometry, channels, clock and time origin are unchanged. Unsupported edge sites are zero filled by the actual adapter.
- Kilosort boundary: channel map 0..23, no inversion, per-channel centering, global median CAR, installed 300 Hz high-pass, infinite artifact threshold, `whiten_mat=None`, no drift shift. Output is float32 and passed to DARTsort with `preprocessing="none"`.
- DARTsort run: exact editable checkout commit `edcfe1b51d672b4136eb13cc78c0875da804b851`; CPU development capability run; threshold detection at 40; no motion estimation; one matching iteration; save intermediate features/labels; 30,000-sample chunks; seed 0; bounded fitting (`n_waveforms_fit=2000`, `max_waveforms_fit=3000`, 512 residual snippets); no work tmpdir.
- Required snapshots/accounting: resolved config; input hashes/statistics; adapter and producer source hashes; every produced HDF5/NPZ/label snapshot path, schema and row count; reload result; final event samples/labels/template lineage where present; ambiguous/noise label counts; added/removed counts between consecutive candidate stages using exact same-row/sample semantics only; compact voltage min/max/RMS/finite counts.
- Exact-A capability matrix: use only saved bound artifacts. Operational states are `saved`, `qualified_replay_only`, or `unavailable`. Never infer detection agreement. A's post-sort `spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy`, `templates.npy`, and `ops.npy` are saved; its packet explicitly says the pre-extraction detection bank is unavailable.
- Resource bound: two 9.9 s x 24 channel synthetic inputs, at most about 120 MB temporary input/intermediate voltage; two CPU mini-runs; 45 minute wall stop per arm; no real/accepted voltage; no GPU; no RF/holdout.
- Stop: preserve and report the first failed arm/stage; do not tune scientific thresholds in v1. Continue the other arm only if failure is local and resource-safe.
- Acceptance: input adapter and DARTsort entrypoint execute; at least detection saves/reloads. Each later stage is reported as saved, qualified replay only, or unavailable from actual outputs. No performance/transfer claim.
- Active plan: `DARTSORT_DIVERGENCE_PLAN_20261007.md` SHA-256 `c26d63dfe9ae94312485d57ec99f1a3c4d7d089806b84485099d4afb4f12554f`.

## Frozen A comparator bindings

- `INPUT_BINDINGS.json` SHA-256 `7c045a88be1819d5411534867ee717ceadf76661c3bb3ba176dd525ed5c4e3b1`.
- Accepted Arm-A binary recorded SHA-256 `672938e942d105af6b1a3f7f1bddf206939535e0e4b57bc01153472a1ccb8b69`.
- Saved A final spikes: `spike_times.npy` SHA-256 `1c7643385dc3f128451880871a643b93fd343dfaa90c4cd7c65a229238ee3bcc`.
- Saved A final cluster/template labels: `spike_clusters.npy` and `spike_templates.npy`, each SHA-256 `84409cbbf3548939fc88d44b191824acceb5ebdcb5432a2faf0619960b50454a`.
- Saved A templates: SHA-256 `19133339863c9c8f58e07128e035fdf7099c45cba30f1342fa0acfac57599ba7`.
- Saved A ops: SHA-256 `07d61baf2c0da759db2ee2a21ef1d2d7a561bcb90293f568bccfe0a9e0906906`.

This contract is frozen before synthetic input generation and sorter execution.
