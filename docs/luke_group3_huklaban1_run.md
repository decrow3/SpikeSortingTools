# Group 3 threshold-interaction execution on huklaban1

User authorization: on 2026-09-07, prepare, test, and launch Group 3 while the
separate Group 1 run is delayed.

## Scientific scope

The new motion-off arms are `rescue_10_8_motion_off`,
`rescue_10_10_motion_off`, and `rescue_12_10_motion_off`. The completed Group 2
`rescue_10_9_motion_off` arm is the shared reference and is validated/rebound,
not sorted again. Together these fill two adjacent threshold cells:

- universal 9/10 by learned 8/9: existing 9/8, 9/9, 10/9 plus new 10/8;
- universal 10/12 by learned 9/10: existing 10/9 plus new 10/10 and 12/10,
  with Group 1's 12/9 completing the cell when available.

The more aggressive 9/7 combination is deferred. These runs test threshold
interaction/nonlinearity; smoke results remain engineering checks and do not
establish scientific superiority.

## Data and execution

- Contract: `testing/configs/luke0804_threshold_interactions_v1.json`, digest
  `3cac8f9bb56ec643513ee01af5482a783fee6b611d9c0f8aa2be81f73c637e9c`.
- Host plan: `testing/configs/luke0804_group3_huklaban1_v1.json`.
- Reused full-duration depth strip:
  `/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group2/long/recording`.
- Output root:
  `/media/huklab/Data/SpikeSortingTools/luke0804_hindsight_ladder_v1/group3`.
- Smoke: first 120 seconds for the three new arms.
- Long run: 10,473.553728 seconds, processing depths 1400–2380 µm and
  scoring interior 1600–2180 µm.
- Controller: `testing/luke_group3_job.py`, run by the independent systemd user
  service `luke-group3-threshold-interactions-v1.service`.

The exact systemd launch argv is persisted as `manager_launch_command` in the
host plan and is copied into the run's `launch.json` by the controller.

Stages are `prepare_smoke`, `run_smoke`, `run_long`, and `finalize_long`. Each
has a durable managed-command receipt. Any failure stops the group, preserves
the evidence, and does not retry automatically.

## Preflight

The focused production-environment suite passed 66 tests, including Group 2
regression coverage and proof that smoke failure prevents long launch. An
earlier attempt with the host Python produced only old-SpikeInterface failures;
the locked production interpreter is the authoritative test environment.

The contract digest matched the plan. The reused strip passed content-bound
recording and spatial-contract validation with request digest
`4b4d2aeb7c08be0e93d9f2cbc4e3412a5675fb1ae2b324f5ddcdeaee71c8ecf1`.
The production environment saw CUDA, and the selected local disk had about
878 GiB free. The systemd launch mechanism was already tested with both success
and intentional-failure dummy jobs; its evidence remains in
`testing/outputs/group2_preparation/`.

## Recovery limits

Kilosort has no within-sort checkpoint. Completed arms and downstream stages
can be reused, but interruption of the active sort requires that arm to restart
from its beginning. The controller refuses an existing launch receipt, so an
unexpected termination must be investigated before any manual restart.

## Execution milestones

- The independent service started at 2026-09-07 03:59:14 PDT as
  `luke-group3-threshold-interactions-v1.service` (main PID 1335650).
- At 04:01:13 PDT, the controller completed its repeated launch-time recording
  validation, persisted `launch.json`, and entered `prepare_smoke`. The durable
  stage receipt records the production interpreter and all three intended arms.
