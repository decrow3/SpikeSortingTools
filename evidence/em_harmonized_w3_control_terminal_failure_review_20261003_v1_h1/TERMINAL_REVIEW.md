# H1 independent terminal review — harmonized W3 control

## Verdict

The released run performed one managed sort and one evaluator invocation with no
retry according to the sealed H5 execution record. The sort is reported complete
and hash-bound; the evaluator failed before loading domain data because the
frozen field path did not exist. No scientific H/F/L result is available.

A saved-output-only recovery is technically sufficient and is the cheapest
adequate next operation. It must not relaunch or retrain H. Release remains
blocked until H5 publishes a fresh derivative and real-input preflight meeting
`SAVED_OUTPUT_RECOVERY_CONTRACT.json`, including direct validation of the saved H
receipt, effective config, completion marker, array clock/bounds, and hash.

## Bound evidence

- Coordinator release `MANIFEST.sha256`:
  `f89982502850f5f6920b187bfd7ee9576bb4ee5a045864818970aaa6eb9eeec9`.
- Terminal `MANIFEST.sha256`:
  `e31e984a1e60beeab55ae8926409b3a14ea08d8e8a28c2733b0a12da2cfbba39`.
- Terminal `COMPLETE.json`:
  `9c47b4273d7d201f2456ef3e96ddcbc26fe4601d4a36a3ecd3e191b944803814`.
- Reported completed H sorting:
  `03966a706beeddf188951510444b703614f7eae25f1ece46be521b4c1c024361`.
- All release and terminal manifest members independently rehashed successfully.

The immutable timestamps order the recorded stages: workflow start
`1791079101.0867221`, score dispatch `1791080118.1506248`, score attempt start
`1791080118.200697`, and preserved failure `1791080119.3539593`. The terminal
report records controller `ExecMainStatus=1`, `NRestarts=0`, attempts 1, retries
0. The controller source permits only attempt 1 and refuses an existing state
directory; the post-sort worker likewise refuses an existing score directory.

The traceback is exact: `harmonized_scorecard.py:153` entered the domain-input
hash loop and `pathlib.Path.open` raised `FileNotFoundError` for the frozen field
path. The scorer creates its results directory only after this loop, so this
failure path cannot create point estimates, bootstrap draws, guardrails, or
`RESULT.json`. H5's terminal inventory separately reports no result.

## Input recovery evidence

H1 has an authoritative motion NPZ at
`/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/luke0804_imec1_two_layer_motion.npz`:
372,378 bytes, SHA-256
`85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9`.
An existing shared copy under `av_aw_handoff_20260926_v1/d2l_field/` is byte
identical.

H1 has the authoritative episode CSV at
`/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage3_q/episode_catalogue.csv`:
105,013 bytes, SHA-256
`46463cda1348c0e0b0c39ad22ae9c71bc46f7c7347797c1c10588944cfc52df1`.
No shared copy with that hash was found in the project tree. No file was
transferred by this review.

The frozen scorer hash-checks the catalogue but does not load or use it after
that check. Its operational `catalogue_outside_remainder` domain is constructed
from the field deviation and censor mask residual, as in the accepted frozen
helper; it is not literal catalogue membership. Recovery must retain this
established operational meaning rather than silently redesign the domain.

## Implementation checks

- Done: authorization and execution packets -> all manifest members rehashed;
  supplied release, terminal, COMPLETE, and H hashes bind consistently.
- Done: actual frozen source path -> controller, post-sort validator, scorer,
  helpers, configs, ordering, failure path, no-overwrite, and no-retry behavior
  traced from the manifest-bound source.
- Done: missing inputs -> authoritative H1 field and catalogue paths, sizes,
  schemas, and exact frozen hashes verified; shared field copy also verified.
- Done: clock contract -> frozen cache view and evaluator agree on global
  `[239998073,250197991)`, local `[0,10199918)`, and 29999.759166666667 Hz.
- Not done: direct H5 H receipt/config/NPZ read -> direct SSH is unavailable from
  H1; the current terminal packet exposes hashes and summaries, not those files.
  The derivative preflight must publish direct validation before release.
- Not done: current H5 prior-score tree inventory -> producer report says no
  result; direct absence proof is required in the derivative preflight.
- Can establish: the first run preserved one post-sort evaluator failure and a
  separately managed saved-output-only recovery can avoid retraining.
- Cannot establish: H's scientific result, independent direct validity of the H
  arrays until the derivative evidence is published, or any advancement claim.

No transfer, scoring, relaunch, GPU work, recording access, RF/holdout access,
or scientific interpretation was performed.
