# Matching1 times/labels adapter review request

Review commit `f9449eb` and the frozen snapshots in this directory.

Load-bearing checks:

1. `times_samples` and `labels` are read from the same matching1 HDF5 rows and preserved in order.
2. Session-clock integer samples and negative labels round-trip exactly.
3. The output directory is create-only.
4. The adapter does not invent amplitudes, positions, or KS labels.
5. The adapter is only a two-array bridge; it must not be interpreted as making every historical Kilosort evaluator metric applicable to DARTsort.

Implementation checks
- Done: producer-side check and one known-answer fixture -> reviewable, not accepted (`RESULT.json`).
- Not done: independent source review and H5 production-output round trip -> reviewer/H5 execution required.
- Can establish: bounded two-array interface behavior.
- Cannot establish: P1/P2/I2 or scientific outcome validity.
