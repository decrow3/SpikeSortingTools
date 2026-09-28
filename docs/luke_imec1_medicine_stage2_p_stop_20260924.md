# Correction P pre-fit stop report

No stage-2 MEDiCINe fit, spike sort, or voltage modification was run.

The P.1 quiet-heldout null rerun passed in all four quiet windows. The episode
tile ranking and eight selected windows were frozen before any fit. The first
rate-only interpretation produced zero accepted episode references and is
preserved as `references_episode_rate_only_zero/`. Before any fit, the episode
candidate union was corrected to include runs of the same saved
shared-recovery field used by P.2, at centred rigid displacement <= -40 um,
plus the raw low-rate candidates. Reference shifts remained the M raw,
unlabeled depth-by-x peak-map test.

The corrected episode references contain 31 accepted episodes:

| window | candidates | accepted | resolved nulls |
|---|---:|---:|---:|
| e01 | 6 | 4 | 6 |
| e02 | 6 | 3 | 6 |
| e03 | 7 | 4 | 7 |
| e04 | 7 | 5 | 7 |
| e05 | 8 | 6 | 8 |
| e06 | 7 | 3 | 7 |
| e07 | 6 | 4 | 6 |
| e08 | 6 | 2 | 5 |

Episode nulls passed in e01-e07. The e08 null failed the unchanged per-window
criterion: mode = 0 um, median = +40 um. Its resolved distribution was
`{0: 2, +40: 2, +120: 1}`. The reference service therefore stopped before
block scoring and before the pre-fit receipt was issued. Fit receipt count at
the stop was zero.

The result root is
`testing/outputs/luke_imec1_medicine_reference_sweep_v2/stage2_o/`.
Relevant files there are `episode_tile_ranking.csv`,
`references_heldout/null_validation.json`,
`references_episode/measured_episodes.csv`,
`references_episode/null_pseudo_episodes.csv`, and
`references_episode/null_validation.json`.
