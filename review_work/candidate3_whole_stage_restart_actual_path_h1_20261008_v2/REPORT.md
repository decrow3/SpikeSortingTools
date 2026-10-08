# Candidate 3 whole-stage restart result

Verdict: `GO_WHOLE_STAGE_RESTART_FALLBACK` for the scoped DARTsort peel-stage path.

The injected second-chunk append failure advanced the stock marker to chunk index 1/start 100 and left four rows in every inspected event/lineage dataset. The wrapper retained that HDF5 under `injected.failed` and did not reopen it. A new attempt with the identical recording and chunk-schedule binding started from a fresh directory and completed all three chunks.

The restart contains exactly six rows with source rows `[0,1,2,3,4,5]`. Samples, seconds, channels, template indices, construction memberships and source rows are array-equal to a separate clean baseline. The complete HDF5 files are also SHA-256 identical (`66865c7d...`). The two wrapper unit tests pass.

Operational decision: Candidate 3 can use fresh whole-stage restart as the checkpoint fallback for the exercised peel path. Unsafe partial peel files must be quarantined and never passed back to stock automatic resume. Transactional within-stage resume remains useful future work, but it is not a prerequisite if the real snippet binds this wrapper, preserves lineage, and the stage recomputation cost is accepted.

This fixture does not exercise kill/power-loss durability, residual files, every append boundary, fitting/training, clustering/refinement, the real Arm-A adapter, or a full session. Those remain separate readiness links.

Implementation checks
- Done: actual run path -> executed `dartsort.util.peel_util.run_peeler` and stock `BasePeeler.peel`/HDF5 writer, not direct calls to fixture helpers.
- Done: failure semantics -> retained failed file has marker `(1,100)` and four event/lineage rows after the injected append error.
- Done: fresh restart isolation -> failed path remains present; restart uses a distinct directory, `overwrite=false`, and identical input bindings.
- Done: accounting and lineage -> six exact unique source rows; samples, seconds, channels, templates and construction memberships equal the clean baseline.
- Done: positive/negative controls -> clean baseline passes; attempt-ID reuse and invalid publication are covered by two passing unit tests.
- Not done: real Candidate-3 input adapter and downstream stages -> require the frozen ordinary/transition mini-input end-to-end deliverable.
- Not done: abrupt process kill and storage durability -> require a managed-job kill-boundary fixture if Candidate 3 will rely on durability beyond caught write failures.
- Can establish: a conservative fresh whole-peel-stage restart prevents the known marker-before-payload gap in the tested actual run path.
- Cannot establish: safe stock partial resume, every failure mode, full pipeline restart completeness, full-session readiness, or scientific performance.
