# Independent review: candidate-3 checkpoint and boundary fixtures

## Verdict

`GO_SCOPED_FIXTURE_CONCLUSIONS`

The packet's two bounded conclusions are supported. The current DARTsort
within-peel marker can cause a genuinely false resume after an append failure,
and the tested synthetic Kilosort pre-whitening float outputs pass through
DARTsort `preprocessing="none"` byte-for-byte unchanged. The packet keeps both
claims within their defensible scope.

No raw or real voltage, production scientific arrays, sorter, GPU, RF/holdout,
or D3 outcome was accessed.

## Checkpoint fault is a genuine false resume

The fixture calls the imported `BasePeeler.gather_chunk_result` against a real
temporary HDF5 checkpoint. Its synthetic spike dataset throws at the first
resize after the real HDF5 completion marker has advanced. Closing and reopening
the file leaves:

- `last_chunk_index = 0` and `last_chunk_start = 0`;
- zero rows in the real `times_samples` dataset;
- `check_resuming` accepting the saved two-chunk schedule and returning last
  chunk 0.

This is not merely a misleading receipt. The actual peel control path computes
`next_chunk_index = last_chunk_index + 1` and slices
`chunks_to_do = chunk_starts_samples[next_chunk_index:]`
(`peel_base.py:218-236`). It therefore skips failed chunk 0 and starts at chunk
1. The positive control produces the same next index only after writing its
row, and the changed-schedule control rejects.

The injection uses one fake failing dataset at a real method boundary rather
than simulating a process kill. That is sufficient to establish the source's
non-transactional ordering and one concrete false-resume state. It does not
characterize every HDF5 durability outcome. Repair validation still needs
failures after resize, during each dataset write, around residual-file writes,
and for zero-event chunks.

Until repaired and independently reviewed, interruption must restart the
affected peel stage; current within-stage state is not safely resumable.

## Synthetic boundary handoff

Independent reruns reproduce both 4-channel × 512-sample fixtures:

- installed Kilosort 4.0.27 executes the real `BinaryFiltered.filter` method
  with centering, CAR, 300-Hz FFT high-pass, infinite artifact threshold, and
  `whiten_mat=None`;
- DARTsort `preprocessing="none"` returns the float32 recording unchanged;
- ordinary and synthetic transition batches have exact array equality and zero
  maximum absolute error, with unchanged channel IDs and geometry;
- the integer-input negative control raises `DSPreprocessingError`.

DARTsort emitted its heuristic “looks unpreprocessed” warning for the synthetic
float amplitudes. This did not alter or reject the data, but future contracts
should record expected warning handling rather than treating warning absence as
an acceptance criterion.

The transition fixture changes channel content halfway through a synthetic
batch. It is not an exact Arm-A motion-field transition or a test of
Kilosort's padded batching context. The result supports the no-second-transform
handoff design only. Real Arm-A ordinary/transition/padding equality remains a
separate prerequisite with the exact materialization and adapter bindings.

## Seal and scope

- All seven packet members match size/hash; COMPLETE binds MANIFEST.
- Member → MANIFEST → COMPLETE chronology is valid.
- Frozen source hashes for DARTsort and Kilosort reproduce.
- Conclusions do not authorize a sort, claim H1 GPU readiness, or establish
  scientific benefit.

Implementation checks
- Done: immutable packet -> seven member hashes/sizes, COMPLETE binding, and seal chronology verified.
- Done: actual checkpoint path -> marker persisted with zero rows; `check_resuming` returns chunk 0 and `peel` advances to chunk 1 (`peel_base.py:218-236,630-678,973-1014`).
- Done: checkpoint controls -> completed append resumes at chunk 1; changed chunk schedule rejects.
- Done: installed-source boundary -> ordinary and transition synthetic arrays reproduce exact equality, stable IDs/geometry, and integer-input rejection (`CHECK_RECEIPT.json`).
- Not done: real Arm-A equality, padded-batch equivalence, true remap-transition behavior, or full crash-boundary matrix -> outside this synthetic packet.
- Can establish: one concrete false-resume state in current source and exact no-op DARTsort transport for the tested synthetic pre-whitening outputs.
- Cannot establish: safe repaired resume, full-session readiness, real-data equality, sorting benefit, causal attribution, or biological identity.
