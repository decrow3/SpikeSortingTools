# Frozen contract: noise-cutoff semantics seal repair v2

Status: `FROZEN_BEFORE_V2_PUBLICATION`

Owner: H1

## Decision and scope

Repair only the v1 seal-chronology caveat in a fresh immutable namespace. The accepted `NON_DECISIVE` scientific verdict is carried forward unchanged. No outcome tables or scientific inputs will be re-read for interpretation, and no evaluator substitution, waveform/raw recording read, H5 dependency, RF/holdout access, sort, parameter sweep, or production launch is permitted.

## Immutable predecessors

- Original packet: `/mnt/NPX/Luke/DARTsort_motion_experiments/noise_cutoff_measurement_semantics_h1_20261007_v1`
  - MANIFEST SHA-256: `e1fc1f66fbffc95d39d4d7501deb0ef117b4d8cbccf705f50ec611541861edae`
  - COMPLETE SHA-256: `616e6fac5fd7131256d4be6e35db85047eb727d167b70257dead24f2078f176c`
- Independent review: `/mnt/NPX/Luke/DARTsort_motion_experiments/noise_cutoff_measurement_semantics_independent_review_h1_20261007_v1`
  - MANIFEST SHA-256: `33ee13bfc88e523135fc41081d19d6b44277e9f86bca6e2b9e165f7db295cec7`
  - COMPLETE SHA-256: `08a29cb7c066fb865a349c7aa279e748af02dc64cb70e4ed69740e23f4256e90`
- Frozen source-host preparation contract: `/mnt/NPX/Luke/DARTsort_motion_experiments/current_coordination_20261002_v1/NOISE_CUTOFF_SUFFICIENT_STATS_SOURCE_PREP_CONTRACT_20261007_V1.json`
  - SHA-256: `c95d1c3ab2578fa1585d32d8c7ff060e9304c9d8569274c5ea40e40b7dac1fb3`

All predecessor packets remain unchanged. V2 references them; it does not replace or overwrite them.

## Seal procedure and acceptance

1. Write every v2 member file first.
2. Hash and size every member into `MANIFEST.json`.
3. Verify all manifest bindings.
4. Write `COMPLETE.json` last and bind the manifest hash.
5. Publish members first, `MANIFEST.json` next, and `COMPLETE.json` last into a previously nonexistent destination.
6. Record nanosecond filesystem mtimes and hashes from the published destination in an independent review packet.
7. Accept only if every member hash/size matches, COMPLETE binds MANIFEST, `mtime(COMPLETE) > mtime(MANIFEST)`, and `mtime(COMPLETE) > max(mtime(member))`.

Filesystem mtime is a publication-order receipt, not a cryptographic ordering primitive. Hash bindings establish content integrity; the independent receipt establishes the requested observed chronology on this filesystem.

## Completion condition

A fresh immutable v2 packet and a separate focused reviewer packet report `GO`, or a precise blocker is preserved without rewriting either predecessor.
