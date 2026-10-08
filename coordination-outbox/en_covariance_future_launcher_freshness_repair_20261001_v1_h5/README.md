# Future covariance launcher freshness repair

This packet repairs the systemd-only defect observed in the consumed covariance/W smoke: systemd ignored `ConditionPathNotExists` because it is not a valid unit directive. The historical v2 source and failed warning evidence remain preserved; they were not overwritten.

The future v3 static service removes that invalid condition and changes only the description to identify the atomic namespace guard. Its `ExecStart`, positive path conditions, working directory, restart policy, and reviewed data command remain unchanged. Freshness authority stays in the executed launcher at `run_root.mkdir(parents=False, exist_ok=False)`, which reserves the complete run root before any native runner can read.

The repaired service is installed at `/home/huklaban5/.config/systemd/user/en_minimum_training_support_mask_covariance_smoke_enabled_v3.service`, hash `7cde6c198d97ff122ff9c62966c30174854c4d9b6271db30963ab69787b7b074`. Installed `systemd-analyze --user verify` exits zero. The unit is loaded/static and inactive/dead, PID 0, zero restarts. It was not started and carries no future execution authorization.

## Zero-voltage dummy

The managed dummy imports the actual `_reserve_fresh_output_namespace` helper with a config containing only scratch output paths. It has no recording path. The first attempt was preserved as a failure because its managed Python environment lacked the repository import path; it exited before creating output and made zero recording reads. After adding only an explicit repository-root import bootstrap, a new dummy unit succeeded.

The successful dummy atomically created the run root, evidence directory, results directory, and `namespace_reservation.json`. A second reservation against the same root raised `FileExistsError`, and `DUMMY_TEST.json` durably records the refusal. No recording, voltage, covariance, W, sorting, training, detection, RF, or holdout was accessed.

## Implementation checks

- Done: historical v2 and repaired v3 sources compared -> invalid unit condition removed; historical source retained.
- Done: installed v3 hash and syntax -> byte-identical to repaired source; systemd verification exit 0.
- Done: installed runtime state -> static, inactive/dead, PID 0, zero restarts; future data service never started.
- Done: actual atomic helper under a managed zero-voltage dummy -> first claim succeeded with evidence; second claim refused with `FileExistsError`.
- Done: unexpected dummy import failure -> evidence preserved, cause isolated, minimal bootstrap repair tested under a new unit.
- Not done: future covariance/data execution -> no GO exists and none was inferred.
- Can establish: future v3 no longer relies on the invalid systemd directive and the launcher-level atomic reservation fails closed on reuse.
- Cannot establish: any covariance/W result, recording behavior, or permission to launch the future service.
