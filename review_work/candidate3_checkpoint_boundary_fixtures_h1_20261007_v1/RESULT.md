# Candidate-3 checkpoint and boundary fixture result

Verdict: **the current DARTsort within-peel checkpoint is unsafe under abrupt append failure; the synthetic pre-whitening handoff itself is exact.**

The frozen fault fixture invoked the imported `BasePeeler.gather_chunk_result` with a two-chunk schedule and injected an exception on the spike-dataset resize. The method had already advanced the HDF5 marker to chunk 0, wrote zero event rows, and `BasePeeler.check_resuming` then returned `next_chunk_index=1`. The positive append also resumed at chunk 1, and a changed chunk schedule rejected. This directly reproduces the source-order concern: the current marker is not a committed-data transaction boundary.

For both ordinary and synthetic channel-state-transition batches, installed Kilosort 4.0.27 produced float32 output using the actual `BinaryFiltered.filter` path with channel selection, centering, CAR, 300 Hz FFT high-pass, infinite artifact threshold and no whitening. The imported DARTsort path with `preprocessing="none"` returned array-identical samples (maximum absolute error 0), unchanged channel IDs and geometry. The same DARTsort guard rejected the original int16 input under `none`.

These fixtures justify a checkpoint repair requirement and the no-second-preprocessing handoff design. They do not establish equality to the real Arm-A materialization, full-batch boundary/context behavior, within-batch remap-transition correctness, or full-session readiness. The next implementation should make per-chunk data durable before publishing its commit marker, validate/trim row extents on resume, and fault-test zero-event, nonzero-event, residual and multi-dataset cases. Until that is reviewed, an interruption requires discarding/restarting the affected peel stage rather than claiming safe resume.

## Implementation checks

- Done: actual imported checkpoint methods -> injected append failure after marker update yields zero rows but resumes at the next chunk (`FIXTURE_RECEIPT.json`; DARTsort `peel_base.py` SHA `ac357ade...`).
- Done: positive and configuration-negative controls -> complete append resumes at the next chunk; changed `chunk_starts_samples` rejects.
- Done: actual installed Kilosort producer method plus DARTsort consumer boundary -> ordinary and transition arrays are exactly equal with stable IDs/geometry; int16/no-preprocessing rejects (`kilosort io.py` SHA `767b76a0...`; DARTsort `preprocess_util.py` SHA `566ce1ab...`).
- Not done: real Arm-A snippet equality -> requires separately frozen access to the exact materialization or reconstruction and ordinary/true transition sample selection.
- Not done: repaired transaction semantics -> requires changed source, crash fixtures at each append boundary, and independent review.
- Can establish: current within-peel abrupt-failure resume can skip unwritten data; float pre-whitening output can traverse DARTsort without a second transform under the tested synthetic paths.
- Cannot establish: current full-session safe resume, real-data equality, launch readiness, scientific benefit, causal attribution or biological identity.
