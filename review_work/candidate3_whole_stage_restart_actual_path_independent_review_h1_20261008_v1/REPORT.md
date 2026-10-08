# Candidate-3 whole-stage restart actual-path independent review

Verdict: `GO_SCOPED_WHOLE_STAGE_RESTART_FALLBACK`.

Commit `6cdfc1456134a99b3cb95291292ed2f28cd9c07d` implements and demonstrates a fresh-directory, whole-peel-stage retry for the exercised caught append-error path. It does not implement or validate partial HDF5 resume.

The injected error is in the stock writer path, not a fixture-only exception. `run_peeler` calls `BasePeeler.peel`; `BasePeeler.gather_chunk_result` advances `last_chunk_index`/`last_chunk_start` before iterating datasets, resizes each HDF5 dataset, and assigns its chunk payload. The incompatible final `source_rows` payload raises `TypeError: Can't broadcast (2, 2) -> (2,)` at that stock assignment. The preserved failed HDF5 has marker `(1,100)`, four rows in each inspected dataset, and corrupt resized `source_rows=[0,1,0,0]`, while earlier second-chunk datasets contain their appended values. This is direct evidence of marker-before-complete-payload failure.

An independent rerun through `dartsort.util.peel_util.run_peeler` reproduced the packet byte-for-byte for the result, failed HDF5, failed receipt, restart HDF5, and clean baseline HDF5. The fresh restart and clean baseline each contain exactly six rows: samples `[10,20,110,120,210,220]`, seconds equal to samples divided by 1000, channels `[0,1,0,1,0,1]`, templates `[100,101,110,111,120,121]`, memberships `[1000,1001,1010,1011,1020,1021]`, and source rows `[0,1,2,3,4,5]`. All arrays are equal between restart and baseline; there are no source-row gaps or duplicates.

The wrapper does not reopen the failed attempt in this exercised path. It gives each attempt a distinct `.partial` namespace, promotes only after validation, quarantines caught failures as `.failed`, and refuses any existing `.partial`, `.failed`, or `.complete` namespace. Independent negative checks confirmed all three conflict rejections and confirmed that `valid=false` is preserved as `.failed` without publishing `.complete`.

Packet integrity is valid. Failed v1 remains sealed as failed (`MANIFEST` SHA-256 `2deb486ccffa440262e024e22fc131beecb445098e9a1e547a3d70dd932e52bc`; `COMPLETE.complete=false`) and retains both failed attempts. Successful v2 has 14/14 manifest members valid (`MANIFEST` SHA-256 `ad5ae1d0eef0977d25109c2030d67e972238a8de54202b246b7a9661dd9d7cdf`; `COMPLETE.complete=true`). Filesystem birth times show `COMPLETE.json` was created after the payload and manifest in both shared packets.

Non-blocking boundaries: the executable file form (`python testing/candidate3_stage_restart_actual_path_fixture.py ...`) lacks the repository root on `sys.path`; the reproducible invocation is `python -m testing.candidate3_stage_restart_actual_path_fixture ...` (or an explicit `PYTHONPATH`). Also, `attempt_id` and symlink-resolved output containment are trusted-caller boundaries, not hardened interfaces. Before a real Candidate-3 launch, the adapter should use a generated slash-free attempt ID, a nonsymlink attempt root, a new namespace, and record the exact unchanged upstream bindings. General-purpose reuse should validate attempt IDs and compare resolved output containment. Abrupt kill/power-loss durability remains untested and outside this contract.

## Implementation checks

- Done: executed source and binding -> commit `6cdfc14`; all three local source hashes and both frozen DARTsort source hashes match v2 `PROVENANCE.json`; DARTsort commit is `edcfe1b51d672b4136eb13cc78c0875da804b851`.
- Done: actual failure location -> stock marker update at `peel_base.py:634-638`, dataset resize/assignment at `peel_base.py:671-676`, observed bad tail and broadcast exception.
- Done: preservation/no reuse -> v1 and v2 failed artifacts remain hash-valid; fresh restart uses a distinct namespace; all existing-state conflicts reject.
- Done: validation closure -> invalid validation is quarantined and not published; two committed unit tests pass.
- Done: exact lineage/time/accounting -> independently checked all six arrays, marker `(2,200)`, six unique contiguous source rows, restart-to-baseline equality, and `times_seconds=times_samples/1000`.
- Done: repeatability -> independent actual-path rerun reproduced `RESULT.json`, failed receipt, failed HDF5, restart HDF5, and baseline HDF5 hashes exactly.
- Done: packet seals -> v1 9/9 and v2 14/14 members match; each `COMPLETE.json` binds its manifest; COMPLETE creation follows manifest creation.
- Not done: real Arm-A adapter, fitting/training, downstream stages, kill/power-loss durability, or full-session behavior -> each needs its separately frozen actual-input or managed-job check.
- Can establish: for this synthetic three-chunk CPU peel path and a caught stock append exception, discarding the failed namespace and recomputing in a new validated namespace yields complete, exact output without failed-file reuse.
- Cannot establish: safe stock partial resume, crash-durable publication, arbitrary-caller path safety, whole-pipeline/full-session readiness, or scientific performance.
