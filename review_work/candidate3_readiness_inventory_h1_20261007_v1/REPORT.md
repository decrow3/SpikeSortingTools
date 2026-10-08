# Candidate-3 H1 readiness inventory

Verdict: **H1 is not currently full-session ready; CPU-only instrumentation can proceed.** The exact comparison target is the completed full-session Luke0804 imec0 Arm A. Its REF source, clock, geometry, gain, AM.3 field and accepted Arm-A identity are recoverable, but the accepted remapped Arm-A binary itself is H5-local. H1 can reconstruct the boundary from the exact REF cache and field, but sample equality must be established before treating it as the same input.

The exact candidate boundary is not the Arm-A int16 remapped cache alone. It is that cache after Kilosort's per-channel centering, global median CAR, FFT high-pass and disabled artifact gate, immediately before whitening. Kilosort `BinaryFiltered.filter` exposes this boundary by using the exact accepted settings with `whiten_mat=None`. DARTsort must then use `preprocessing="none"` on float traces; otherwise the comparison double preprocesses the voltage.

DARTsort's current resume implementation is not yet adequate evidence for “fully checkpointed.” Stage discovery and within-peel chunk resume exist, but `gather_chunk_result` advances `last_chunk_index` and `last_chunk_start` before residual and spike datasets are appended. An abrupt kill can therefore leave a checkpoint that skips an incompletely committed chunk. A fault-injection fixture through this actual path is the next cheapest test. If confirmed, either move/flush the commit marker transactionally and review the repair, or declare within-stage interruption non-resumable and budget whole-stage restart; do not describe current behavior as safely resumable.

H1 also has no usable GPU in the current check (`nvidia-smi` driver failure; DARTsort environment reports zero CUDA devices). The editable DARTsort import works when its Numba cache is redirected to `/tmp`, but its installed version string names an older Git identity than the imported clean checkout, so future contracts must bind Git and source hashes directly. If H1 GPU access is not restored, H5 is the concrete full-session host alternative because it holds the accepted Arm-A materialization and has the previously verified RTX A5000 path; current H5 state still requires refresh before launch.

## Implementation checks

- Done: exact target identity and input metadata -> completed imec0 Arm A binds 314,204,894 samples, 384 channels, 29,999.835983263598 Hz, gain 2.34375 µV/count and AM.3 field SHA `4c769125...` (`ARM_A_COMPLETE.json` SHA `4f49d6d4...`; REF manifest SHA `2d15cf9d...`).
- Done: actual Kilosort boundary source -> centering/CAR/high-pass/artifact gate precede optional whitening (`kilosort/io.py:952-985`, SHA `767b76a0...`); accepted Arm A disables the artifact gate and internal motion.
- Done: actual DARTsort source/environment -> clean Git `edcfe1b5...`, editable Python 3.12.4 environment imports with writable Numba cache; `preprocessing="none"` is the no-second-preprocessing route (`src/dartsort/main.py:127-160`; `src/dartsort/util/preprocess_util.py`, SHA `566ce1ab...`).
- Done: checkpoint source trace -> stage fast-forward exists (`src/dartsort/util/main_util.py:366-435`), but chunk markers are advanced before data append (`src/dartsort/peel/peel_base.py:630-678`) and later trusted as complete (`:975-1014`).
- Done: current host check -> no CUDA device is visible; exact accepted Arm-A materialized binary is not mounted on H1.
- Not done: voltage/sample equality -> prohibited by this inventory contract; requires a separately frozen ordinary/transition snippet contract and the actual producer/consumer wrapper.
- Not done: interruption/reload correctness -> requires fault injection through the actual HDF5 gather/resume path, including zero-event and nonzero-event chunks.
- Not done: H5 current readiness or managed-service disconnect survival -> refresh only after source/input/checkpoint repair is reviewable.
- Can establish: exact candidate target and boundary, source/environment candidates, and concrete blockers to a truthful full-session-ready claim.
- Cannot establish: sample equality, checkpoint safety, launch readiness, sorting benefit, causal attribution or biological identity.
