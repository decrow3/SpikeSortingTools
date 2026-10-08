# Candidate 3 whole-stage restart fallback v2: frozen contract

This is a one-change rerun of failed preserved v1. The only wrapper change is normalization of the attempt-root path to absolute before verifying that DARTsort's absolute returned output remains inside it. The failure injection, input, schemas, expected arrays, run path, resource bounds, stop rule, and acceptance rule are unchanged.

- Milestone: candidate-3 checkpoint readiness.
- Decision changed: whether a conservative whole-stage restart can replace unsafe automatic partial resume for Candidate 3 while transactional resume remains queued.
- Prediction: after a deterministic append failure advances the stock DARTsort chunk marker and leaves a partial HDF5, the wrapper preserves that failed attempt, never reopens it, and starts a fresh attempt from unchanged inputs. The fresh attempt must equal a clean baseline in event rows, samples, seconds, channels, template indices, construction memberships, and source rows with no gaps or duplicates.
- Actual path: `dartsort.util.peel_util.run_peeler` -> `BasePeeler.peel` -> stock HDF5 writer, invoked by `testing/candidate3_stage_restart_actual_path_fixture.py`. Wrapper: `testing/candidate3_stage_restart_fallback.py`.
- Frozen input: 300 frames x 2 channels float32 zeros at 1000 Hz; geometry `[[0,0],[20,0]]`; chunk starts `[0,100,200]`; two deterministic events per chunk at offsets 10 and 20.
- Frozen lineage schema: `times_samples`, stock `times_seconds`, `channels`, `template_indices`, `construction_memberships`, `source_rows`. Expected source rows `[0,1,2,3,4,5]`.
- Failure: incompatible `source_rows` shape at chunk 100, after prior datasets become appendable. Require exception, advanced failed marker, retained HDF5/receipts.
- Restart: new fresh attempt directory, same input binding, `overwrite=false`, no failed-file reuse. Compare with independent clean baseline attempt.
- Resources: CPU, zero worker pool, three tiny attempts, no accepted voltage read, no GPU allocation.
- Stop: after failure, restart, baseline and accounting, or immediately on invariant failure.
- Acceptance: `GO_WHOLE_STAGE_RESTART_FALLBACK`; all equality flags true; six exact unique source rows; failed attempt present; equal bindings; marker `(2,200)`.
- Scope: supports only this whole-stage orchestration fallback. It does not repair automatic resume, cover kill/power-loss durability, exercise every DARTsort pipeline stage, or establish real Arm-A behavior.
- Active plan: `DARTSORT_DIVERGENCE_PLAN_20261007.md` SHA-256 `c26d63dfe9ae94312485d57ec99f1a3c4d7d089806b84485099d4afb4f12554f`.

This contract was frozen before v2 execution.
