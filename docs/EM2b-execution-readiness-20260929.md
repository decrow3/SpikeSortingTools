# EM.2b W2 execution readiness — 2026-09-29

**Status: scorecard frozen and first measurement arm selected; no sort
launched.**
EM.2a selected a nonredundant comparison set. The exact scorecard is frozen in
`configs/em2b_w2_scorecard.v1.json` (SHA-256
`c944e48abf5bd9868dc398d47741a8224bf21c96f536f8219f6719312025458c`). A harmless systemd user-service dummy
continued after its launching shell exited and completed with status 0, so the
required persistent launch mechanism is available. Its receipt is
`testing/outputs/em2b_launcher_dummy_20260929/receipt.json` (SHA-256
`9a86ff4a75174761c1880c01656ac65763b993e7a7690097c38224c146b5027b`).

The exact-DD voltage materialization exists, but its EM.2b sort does not. It is
the first selected arm and will measure the end-to-end resource cost. After
that measurement, the cheaper primary stage is two more sorts: unrounded
kriging and rounded kriging. They are compared with the existing S0 sort and
the new exact-DD sort. Unrounded IDW and nearest remain frozen optional arms
and run only if the primary stage cannot distinguish a field-rounding effect
from an operator effect.

The prior D2L W2 sort provides context only: its `sort-complete.json` reports
exit status 0 and 1,514.141 seconds, and its run directory occupies 1.3 GB. It
reused a shared recording and detection and therefore is not the required
measurement of a selected EM.2b arm. The first exact-DD arm has a conservative
10,800-second and 15-GiB ceiling, with one GPU and four CPU jobs. Its completed
measurement will freeze the tighter limits for later arms before another sort
authorization. Timeout or cap breach is a preserved failure and does not
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

The authorized real command would be:

```bash
env NUMBA_CACHE_DIR=/tmp/numba-em2b \
  /home/huklaban5/Documents/DARTsort/.venv/bin/python \
  /mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/source/pipeline.py \
  launch \
  --config /home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools/configs/em2b_w2_rounded_exact_dd.v1.json \
  --output /home/huklaban5/DARTsort_experiment_scratch/em2b_w2_20260929/rounded_exact_dd_v1
```

This exact-DD sort requires explicit post-report run authorization under the EM
plan. Its completed receipt will set the later-arm caps before the kriging
packets are launched. No RF work, outer-holdout access, production change, or
automatic W3 run is authorized.
