# Independent review: candidate-3 H1 readiness inventory

## Verdict

`CORE_NOT_READY_VERDICT_ACCEPTED_BINDING_REPAIR_REQUIRED`

The inventory's main decision is correct: H1 is not ready for a candidate-3
full-session sort, while CPU-only wrapper and checkpoint instrumentation may
proceed. The target dimensions, Kilosort pre-whitening boundary, DARTsort
environment, current lack of CUDA, and checkpoint-order risk all reproduce.

The next snippet equality contract still needs two exact bindings that are
available in compact evidence but absent from this inventory: the authoritative
H5 Arm-A binary/materialization manifest and the reconstruction adapter source.
This is a handoff/provenance repair, not evidence that the core no-launch
verdict is wrong.

No raw voltage or scientific NPZ/HDF5 arrays were opened. No sort, RF/holdout,
or D3 outcome was accessed.

## Target and boundary

`ARM_A_COMPLETE.json` independently confirms imec0, 314,204,894 samples, 384
channels, 29,999.835983263598 Hz, int16 storage, gain 2.34375 µV/count, AM.3
field hash `4c769125...`, and 241,309,358,592 bytes. The byte count exactly
equals `samples × channels × 2`.

The original REF cache and its manifest are present on H1. The accepted Arm-A
binary is recorded at
`/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/arm_a_v1/full/recording/traces_cached_seg0.raw`
with binary hash `672938e9...` and materialization-manifest hash `ef2071c9...`
in compact `INPUT_BINDINGS.json` (SHA `7c045a88...`); that source-host path and
manifest hash should be copied into the next frozen contract.

Kilosort source confirms channel selection and optional sign inversion precede
per-channel mean removal, median CAR, FFT high-pass, artifact handling, then
optional whitening (`io.py:952-985`). Accepted settings supply CAR=true,
high-pass=300 Hz, artifact threshold infinity, inversion=false, and motion
correction=false. Setting `whiten_mat=None` exposes the intended pre-whitening
float boundary. DARTsort's `preprocessing="none"` returns the recording
unchanged and rejects non-float inputs (`preprocess_util.py:15-20,104-109`).

The adapter candidate at local and archived paths is byte-identical, SHA
`01253946...`, but `ARM_A_COMPLETE.json` names only the class/behavior and does
not bind those executed source bytes. The snippet contract should bind a frozen
adapter snapshot/hash and compare ordinary plus transition/padding cases before
claiming equivalence to accepted Arm A.

## Checkpoint interpretation

The source-only AST fixture verifies that `gather_chunk_result` writes
`last_chunk_index` and `last_chunk_start` at lines 635 and 638, before the first
residual write at line 641 and before spike-dataset writes at 671-676. No
explicit flush occurs inside the function. `check_resuming` later trusts the
saved marker at lines 987-1014.

This establishes a real marker-before-data hazard. It does not establish which
HDF5 writes survive a particular abrupt process or host termination. The
inventory appropriately requires fault injection before claiming safe resume.
That fixture must include zero-event and nonzero-event chunks and verify dataset
length/content after reopen, not merely marker values.

## H1 readiness

- DARTsort checkout is clean at Git `edcfe1b5...`; all four declared source
  hashes and `uv.lock` reproduce.
- Editable import resolves to the checkout and reports Python 3.12.4,
  SpikeInterface 0.104.8, Torch 2.6.0+cu124, and the documented stale
  distribution version string.
- Import succeeds with `NUMBA_CACHE_DIR` under `/tmp`; without it, the exact
  SpikeInterface Numba cache-locator failure reproduces.
- `nvidia-smi` cannot communicate with the driver; Torch reports CUDA false and
  zero devices.
- Root and shared storage have ample current free space, but storage does not
  close the missing-GPU or checkpoint blockers.

Implementation checks
- Done: immutable inventory packet -> three member sizes/hashes match; COMPLETE binds MANIFEST; member→MANIFEST→COMPLETE chronology is valid.
- Done: exact target metadata -> Arm-A dimensions, clock, gain, field/content identities, and byte arithmetic match compact accepted receipts.
- Done: boundary trace -> Kilosort operation order and DARTsort float/no-preprocessing requirement reproduce from bound source.
- Done: marker-before-data claim -> independent AST fixture confirms marker lines 635/638 precede data line 641 and finds no in-function flush (`checkpoint_order_fixture.py`, SHA `ff6a9493...`).
- Done: H1 environment -> clean editable source import with `/tmp` cache, default-cache failure, zero CUDA devices, driver failure, and available storage reproduced.
- Not done: actual abrupt-termination persistence -> requires separately frozen fault injection through HDF5 gather/reopen/resume.
- Not done: sample equality or H5 readiness -> requires bounded snippets on the source host and refreshed H5 environment checks.
- Can establish: H1 is not full-session ready; CPU-only instrumentation is reasonable; the checkpoint implementation has a source-level ordering hazard.
- Cannot establish: accepted-Arm-A sample equality, interruption-safe resume, launch readiness, sorting benefit, causal attribution, or biological identity.
