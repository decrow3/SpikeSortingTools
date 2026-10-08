# Candidate 3 whole-stage restart fallback: frozen contract

- Milestone: candidate-3 checkpoint readiness.
- Decision changed: whether a conservative whole-stage restart can replace unsafe automatic partial resume for Candidate 3 while transactional resume remains queued.
- Prediction: after a deterministic append failure advances the stock DARTsort chunk marker and leaves a partial HDF5, the wrapper preserves that failed attempt, never reopens it, and starts a fresh attempt from unchanged inputs. The fresh attempt must equal a clean baseline in event rows, samples, seconds, channels, template indices, construction memberships, and source rows with no gaps or duplicates.
- Actual path: `dartsort.util.peel_util.run_peeler` -> `BasePeeler.peel` -> stock HDF5 writer, invoked by `testing/candidate3_stage_restart_actual_path_fixture.py`. The wrapper is `testing/candidate3_stage_restart_fallback.py`.
- Frozen input: 300 frames x 2 channels of float32 zeros at 1000 Hz; channel geometry `[[0,0],[20,0]]`; chunk starts `[0,100,200]`; two deterministic events per chunk at offsets 10 and 20.
- Frozen lineage schema: `times_samples`, stock `times_seconds`, `channels`, `template_indices`, `construction_memberships`, and `source_rows`, all event-aligned. Expected source rows are exactly `[0,1,2,3,4,5]`.
- Failure: on chunk start 100, supply an incompatible `source_rows` shape only after earlier event datasets are appendable. Success requires an exception, an advanced failed marker, and retention of the failed HDF5 and receipts.
- Restart: a different fresh attempt directory, identical input binding, `overwrite=false`, no reuse of the failed HDF5. Compare it byte-for-array with a separate clean baseline attempt.
- Resources: 300 x 2 synthetic float32 frames; CPU; zero worker pool; three tiny attempts; no accepted voltage read; no GPU allocation.
- Stop condition: stop after injected failure, one fresh restart, one clean baseline, validation and equality accounting; stop immediately on any failed invariant.
- Acceptance: verdict `GO_WHOLE_STAGE_RESTART_FALLBACK`, all dataset equality flags true, six unique source rows in exact order, failed attempt present, bindings equal, final marker `(index=2,start=200)`.
- Interpretation: passing supports only the whole-stage orchestration fallback and the tested stock run path. It does not repair automatic resume, cover kill/power-loss durability, prove every DARTsort stage, or establish real Arm-A behavior.

This contract was written before the packet execution. An earlier development run in `/tmp` established fixture executability but is not the decision artifact.
