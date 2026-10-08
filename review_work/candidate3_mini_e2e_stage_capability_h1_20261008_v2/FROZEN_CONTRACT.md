# Candidate 3 miniature end-to-end stage capability v2

This is a one-change repair of failed preserved v1. Cell support is derived from the exact clock: `duration_s = 297000 / 29999.835983263598`; centers are duration/4 and 3*duration/4; each cell width is duration/2. This exactly covers every sample. All other inputs, predictions, schemas, processing, DARTsort settings, resource bounds, stop rules, A bindings, and interpretation are unchanged from v1.

- Synthetic input: seed 38172; 297,000 frames; 24 channels at 20 um pitch; int16 noise sigma 20; 12 deterministic units; 61-sample negative waveforms; amplitudes 180..290 counts.
- Arms: ordinary shifts `[0,0]`; transition shifts `[0,20]` um at the exact half-duration boundary.
- Actual adapter/boundary: `ExactLatticeRemapRecording`, installed Kilosort `BinaryFiltered.filter` with channel map 0..23, inversion false, centering, global median CAR, 300 Hz high-pass, infinite artifact threshold, `whiten_mat=None`; float32 into DARTsort `preprocessing="none"`.
- DARTsort: commit `edcfe1b51d672b4136eb13cc78c0875da804b851`; CPU; threshold 40; no motion; one matching iteration; saved intermediate features/labels; 30,000-sample chunks; seed 0; `n_waveforms_fit=2000`, max 3000, 512 residual snippets.
- Required accounting: resolved config; input hashes/statistics; produced stage schemas/counts; reload; event/sample/label/template lineage where present; noise/ambiguous labels; compact observed voltage accounting. No causal/identity claim.
- Stop: preserve first failed stage; no v2 threshold/config tuning. Continue other arm only if resource-safe.
- Acceptance: adapter and entrypoint execute; at least detection saves/reloads; later stages are reported `saved`, `qualified_replay_only`, or `unavailable` from actual outputs.
- A comparator: input and final exported spikes/labels/templates/ops saved; pre-extraction detection bank explicitly unavailable; no bound matching candidates or refinement membership transitions. Do not recreate A or infer initial detection agreement.
- Resources: two 9.900054 s x 24 synthetic arms, <=120 MB temporary voltage/intermediates; 45 minute stop per arm; no accepted voltage, GPU, RF, or holdout.
- Active plan SHA-256: `c26d63dfe9ae94312485d57ec99f1a3c4d7d089806b84485099d4afb4f12554f`.

Frozen A bindings remain: `INPUT_BINDINGS.json` `7c045a88...`; input `672938e9...`; final times `1c764338...`; final cluster/template labels `84409cbb...`; templates `19133339...`; ops `07d61baf...`.

This v2 contract was frozen before v2 input construction.
