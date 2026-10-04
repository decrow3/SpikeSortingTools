# Candidate-G full-readiness final acceptance

## Verdict

`GO_CONSTRUCT_RELEASE_INSTALL_REVIEWED_UNITS_START_ONCE`

The authorized parent creation and corrected full audit satisfy the accepted v6 activation prerequisites. The run parent is recorded as a real, empty, non-symlink directory owned by UID/GID 1000 with mode `0700`. The exact reviewed audit source ran in a fresh namespace and produced `PASS_FULL_READINESS_NO_START` with receipt SHA-256 `a06d125f65888c92cb1111b38b89904029c9b9c8ccd62ed0543cf355a8b6eb9f`.

The receipt verifies all 30 closure entries including both H5-local state files, the 14-array/490,944-row state with exact row order and native QDA, frozen input descriptors without voltage payload, CPU/RAM/local/shared-disk floors, one available CUDA device without compute, unit-definition and concrete-instance absence, and continued canonical release/attempt/compact absence. All three historical failures remain byte-identical to their preserved predecessors.

## Activation boundary

The coordinator may construct only the exact v6 canonical release, install/reload only the two byte-identical reviewed user units, and start `candidate-g-continuation-v1.service` once. Immediately beforehand it must recheck the exact empty UID/GID-1000 mode-0700 parent, all canonical absences, not-found/inactive systemd state, and the frozen CPU/RAM/disk/GPU floors. The v6 gate must pass against the final release and installed unit bytes before start.

The authorized scope remains attempt `candidate-g-v14-agglomerate-continuation-20261004a`, envelope `[239969921,250200079)`, 16 GiB logical-read ceiling, 3,416-second worker timeout, four CPUs, 64 GiB memory and one GPU. There is no retry. Any failure must be preserved and returned for review; no source, unit, path, parameter or resource substitution is allowed.

Implementation checks
- Done: parent identity -> inspected sealed pre/post receipts showing exact path, pre-absence, post UID/GID 1000, mode 0700, empty and non-symlink.
- Done: executed source -> source is byte-identical to the reviewed correction and hashes to `16330d5551a9a55c8c2b8a68d3ff24cd57ec5701bae5990c5f9905c91fcbbac2`.
- Done: closure/state -> the H5 audit reports all 30 entries verified and state identity of 14 arrays, 490,944 rows, row equality and native QDA; 28 shared entries independently reverify on huklaban1.
- Done: inputs/resources -> descriptor validation reports no voltage payload; the sealed receipt records 20 CPUs, 191,968,174,080 available memory bytes, 904,844,582,912 local free bytes, 165,928,431,616 compact free bytes and one available CUDA device without compute.
- Done: absence/failure provenance -> systemd/canonical absences are in the receipt, shared release/compact paths remain absent, and all three historical failures compare byte-identical.
- Done: packet provenance -> verified MANIFEST `7b10d4a6c1821fac89796f5bf52bd4cacb655b3997d28a6d6f74280b33012d2a`, COMPLETE `14054a3aa584caab99d89fa0ca79421a26192028222d05600c4ab299f0d44aa9`, request `a51d8e777a3dab5d129a93423223027029519308d4c60c1fd988e1fec8a1afa8`; supplied coordination hash `9153d2ead45f57499a72b05e433dd9ae24d45468be5096a55965ec059656f297`.
- Not done: independent H1 opening of the two H5-local state files or direct H5 systemd/resource query -> unavailable from huklaban1; accepted from the exact reviewed audit and guarded by mandatory just-in-time rechecks.
- Can establish: the frozen v6 continuation is ready for exact release construction, reviewed-unit installation and one managed start.
- Cannot establish: successful continuation completion, realized resource use or Candidate-G scientific outcome until terminal evidence is produced and reviewed.

No mutation, installation, reload, start, voltage payload access, agglomeration or GPU compute was performed during this acceptance.
