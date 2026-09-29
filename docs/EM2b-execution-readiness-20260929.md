# EM.2b W2 execution readiness — 2026-09-29

**Status: stage-0 measurement complete; stage-1 packets ready but not
launched.** The measured result and bounded publication failure are reported in
`docs/EM2b-stage0-measurement-and-stage1-readiness-20260929.md`. The revised
scorecard, including the measured resource limits, is frozen in
`configs/em2b_w2_scorecard.v1.json` (SHA-256
`714fd29c41f9c4aa6142d534232f030438af0819133613177ca404f4331642c6`). A harmless systemd user-service dummy
continued after its launching shell exited and completed with status 0, so the
required persistent launch mechanism is available. Its receipt is
`testing/outputs/em2b_launcher_dummy_20260929/receipt.json` (SHA-256
`9a86ff4a75174761c1880c01656ac65763b993e7a7690097c38224c146b5027b`).

The exact-DD sort and sorting-only QC are complete: 477 assigned units and
603,125 accepted events. Its service used 1,066.229 wall seconds, 2,556.378 CPU
seconds, and 10.377 GiB peak scratch. The primary stage is two more sorts: unrounded
kriging and rounded kriging. They are compared with the existing S0 sort and
the new exact-DD sort. Unrounded IDW and nearest remain frozen optional arms
and run only if the primary stage cannot distinguish a field-rounding effect
from an operator effect.

The prior D2L W2 sort provides context only: its `sort-complete.json` reports
exit status 0 and 1,514.141 seconds, and its run directory occupies 1.3 GB. It
reused a shared recording and detection and therefore is not the required
measurement of a selected EM.2b arm. The first exact-DD arm had a conservative
10,800-second and 15-GiB ceiling, with one GPU and four CPU jobs. Its completed
measurement freezes 3,600 seconds wall and 13 GiB scratch for each later arm.
Timeout or cap breach is a preserved failure and does not
authorize a silent retry or larger budget. The network mount currently has
about 231 GB free, while local scratch has about 1.2 TB free; working
directories stay on local scratch.

The scorecard uses exact piecewise-linear W2 domains and arm-local units. A unit
is eligible with at least 100 flat-domain events in the full W2 point dataset;
that eligible set remains fixed through 2,000 common 5-second-block bootstrap
draws. Every draw uses the same block multiplicities for every arm. The primary
quantity is `rho_lattice - rho_unrounded_kriging`, where rho is an arm-local
rate-rank continuity proxy. No unit is matched across arms, and rho is not
interpreted as identity or purity.

A lattice advantage requires delta rho at least 0.05, a 95% CI lower bound
above zero, no more than 5% relative yield loss, and no domain's segment-safe
9--29-sample ISI fraction increasing by more than 0.01. Practical equivalence
requires the entire CI inside `[-0.05, 0.05]`, yield within 5% in either
direction, and every ISI fraction within 0.01. Fewer than three eligible units,
a constant rank vector, a non-finite point estimate, or fewer than 1,900 valid
bootstrap draws is unresolved rather than zero-filled.

The outcome implementation is `testing/em2b_w2_scorecard.py`. It requires a
post-sort manifest containing exact hashes for every saved sorting, the field,
mask, catalogue, and this contract. It writes point estimates, segment-safe ISI
counts, exact block/domain exposure, all valid paired bootstrap deltas,
comparison decisions, and a hashed completion manifest. Its common-resample,
domain-exposure, and undefined-statistic tests pass before any EM.2b outcome is
available.

The first-arm run packet is now concrete. It uses the accepted DD S_L
materialization through
`testing/inputs/em2b_w2/rounded_exact_dd_recording.json` (SHA-256
`b6fd2e162e93b5492b93885d0ee9e8e438cf44f33ed462abbdd3e3a1b3b9577e`)
and `configs/em2b_w2_rounded_exact_dd.v1.json` (SHA-256
`3d9a55d597721d1f4edde9fb3c2599b8457fe7077f623d95071b44b8df0240eb`).
The production runner SHA-256 is
`857614c3f711eb4e2270bd198be718c7ecd94c02a91e5dd8b5f25ec649167a4e`.
The descriptor loads as 10,199,918 float32 frames by 182 channels in both the
SpikeInterface and DARTsort environments, and the runner validates the config.
Its built-in dummy mode passed the same systemd/GPU preflight; the compact
receipt SHA-256 is
`fc738af1fa9ea7814a5881e01df6fd7037f7bae9a165f7744bb1892910aa402a`.

The completed stage-0 command was:

```bash
env NUMBA_CACHE_DIR=/tmp/numba-em2b \
  /home/huklaban5/Documents/DARTsort/.venv/bin/python \
  /mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/source/pipeline.py \
  launch \
  --config /home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/configs/em2b_w2_rounded_exact_dd.v1.json \
  --output /home/huklaban5/DARTsort_experiment_scratch/em2b_w2_20260929/rounded_exact_dd_v1
```

The runner completed every scientific stage but withheld its final receipt
after detecting that linked HDF5 reuse changed the detection artifact's mtime.
The preserved failure and independently hashed outputs are covered in the
stage-0 report. The two kriging packets now pass config validation,
cross-version kernel checks, and persistent systemd runtime preflights that
load the custom extractor under the exact future service environment. Their
sorts require separate authorization. No RF work, outer-holdout access, or
automatic W3 run is authorized.
