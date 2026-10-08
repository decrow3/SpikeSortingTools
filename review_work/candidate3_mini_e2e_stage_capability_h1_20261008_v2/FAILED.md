# Frozen v2 execution failed before detection

Input construction passed with exact sample support. The ordinary DARTsort arm then stopped before detection because `DARTsortInternalConfig` retained its default `detection_type="subtract"` while the frozen configuration supplied a `ThresholdingConfig`. The stock assertion rejected this mismatch. Its error handler then exposed a secondary implementation defect: `save_everything_on_error=true` tries to copy a `None` work directory when `work_in_tmpdir=false`.

No candidate events or stages ran. Scratch inputs and the failed output/config/traceback remain under `/tmp/candidate3_mini_e2e_20261008_v2`.

V3 makes only two wiring corrections: explicitly set `detection_type="threshold"`, and set `save_everything_on_error=false` because the run-specific directory already preserves outputs and traceback. Inputs, hashes, thresholds, stages, resources, and acceptance remain unchanged.

Implementation checks
- Done: resolved saved config and traceback identify the pre-detection type assertion.
- Done: secondary error-handler traceback identifies `None` work-dir copying, not a sorter-stage result.
- Not done: detection or later stages -> none ran.
- Can establish: v2 input adapter/boundary completed and config dispatch was invalid.
- Cannot establish: any DARTsort stage capability.
