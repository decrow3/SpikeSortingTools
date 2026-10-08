# Independent implementation-first review: candidate-3 lazy prewhitening wrapper

## Verdict

`GO_SCOPED_SYNTHETIC_LAZY_WRAPPER_CAPABILITY_REAL_H5_EQUALITY_REQUIRED`

Reviewed evidence:

- packet `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_lazy_prewhitening_wrapper_h1_20261008_v1`
- packet manifest `c1c74ee649925a9cac8708247ae99f63748d9ec7a2c8e3836d75fb2990440ba6`
- packet complete `4d960c8f71b60db8e2e3af97eb2a2321ea8b1b31044c40849e157d64439e137b`
- source commit `aea7431c4026fa74e36ad01139d9e596900a1d25`

The wrapper is accepted for the packet's narrow synthetic capability claim. It
does not establish real Arm-A equality, H5 compatibility, throughput, resource
sufficiency, full-session readiness, or scientific performance.

## Findings

1. **Seal, source and rerun pass.** Every manifested member matches its recorded
   size and SHA-256 and `COMPLETE.json` binds the manifest. Shared and local birth
   order put `COMPLETE.json` after the manifest. All committed wrapper, export,
   consumer, test and contract files rehash to `SOURCE_BINDINGS.json`; `git show`
   at `aea7431c` reproduces the wrapper and test hashes. The DARTsort checkout is
   clean at `edcfe1b5...`, and all cited Kilosort/DARTsort sources match. An
   independent bounded rerun passed 22/22 tests in 10.67 seconds.

2. **Request-only laziness and FFT length are real.** Construction builds only
   metadata and a small high-pass kernel and performs no parent trace read.
   Each segment call reads exactly `[start_frame,end_frame)` once, across parent
   channels, and adds no cache or hidden temporal context. The wrapper passes a
   tensor with exactly that time length to installed `BinaryFiltered.filter`,
   whose source constructs `fft_highpass(..., NT=X.shape[1])`. The committed
   tracker test observes one exact interval read. The independent adversarial
   check additionally intercepted the actual helper call and observed `NT=53`
   for a 53-sample request.

3. **Padding/core semantics and positive control are appropriate.** A caller
   that needs context must request the padded interval and crop the returned
   core. The test compares both the complete padded output and its crop to the
   corresponding eager padded request. Independently filtering the unpadded core
   differs, as expected from request-length FFT and boundary context. Thus the
   adapter is explicitly request-context preserving, not chunk-invariant.

4. **Operation order matches installed Kilosort.** `BinaryRWFile.__getitem__`
   converts uint16 by subtracting `2**15`, then applies scale and shift before
   `BinaryFiltered.filter`. The filter then applies channel map/order, optional
   sign inversion, per-channel temporal mean removal, lower-median CAR, the
   request-length FFT high-pass, disabled artifact handling and no whitening.
   The wrapper spells out the same pre-helper conversion/order and freezes
   `artifact_threshold=inf`, `whiten_mat=None` and `dshift=None`.

   The committed eager comparisons use the same installed filter and therefore
   cannot independently validate that helper's scientific correctness. They are
   nevertheless noncircular for the narrower question of whether the wrapper
   delivers the same tensor and metadata to that frozen installed helper. To
   guard against helper-against-itself masking an adapter-order bug, this review
   added a manual expected path that does **not** call `BinaryFiltered.filter`.
   It combines uint16 input, distinct per-parent-channel scale/shift, reordered
   channels, `invert_sign=true`, centering, lower-median CAR and the lower-level
   FFT primitives. It is exactly array-equal to the wrapper and also verifies
   float32, selected IDs/geometry, nonzero clock and output shape.

5. **Axes, selection, geometry and clocks are preserved.** The parent trace is
   time-by-channel; filtering transposes to channel-by-time and transposes back.
   Kilosort channel mapping uses parent indices in the declared selected order;
   later SpikeInterface channel subrequests are applied to that selected output.
   `BasePreprocessor` carries selected channel IDs/properties, and each segment
   wraps its own parent segment, preserving nonzero per-segment `t_start`. Tests
   cover reordered string IDs, selected geometry, two segments with distinct
   nonzero origins, and exact output dtype `float32`.

6. **Input conversion and rejection behavior match the contract.** Non-uint16
   data is converted to float32; uint16 is converted to signed float32 first.
   Scalar or full-parent-channel vector scale/shift is applied before channel
   mapping. Duplicate and unknown channels, nonfinite or wrongly shaped
   scale/shift, nonpositive/nonfinite cutoff, unsupported non-CPU device and
   empty time intervals fail closed. The committed tests cover the contract's
   listed invalid cases; direct source inspection covers the cutoff guard.

7. **The DARTsort boundary is correctly scoped.** In `dartsort.main.dartsort`,
   `ds_will_copy_recording` is evaluated, `preprocess` is called, and
   `ds_all_to_workdir` receives the result. With the frozen float32 wrapper,
   `preprocessing="none"` returns the same object; `copy_recording_to_tmpdir="no"`
   makes the copy flag false; and `work_in_tmpdir=false` with no work directory
   returns the same object and `None`. The exact DARTsort interpreter consumer
   independently verifies object identity, values, dtype, clock, IDs and
   geometry. DARTsort still performs bounded sanity reads; the claim is no
   preprocessing/copy/materialization, not zero reads by the full entrypoint.

8. **Freeze chronology is transparently corrected.** The ten-rule preliminary
   contract is preserved verbatim as `PRELIMINARY_CONTRACT_SUPERSEDED.json`.
   There is no independent immutable timestamped receipt for the first 10-test
   run, so its historical pre-outcome freeze is author-reported rather than
   independently proven. It is not load-bearing: the strengthened contract was
   committed at 02:20:26 before the final combined run timestamped 02:21:14.
   `FREEZE_CORRECTION.md` explicitly calls the change a strengthening/redesign
   and says it is not retroactively preregistered. The JSON retains the earlier
   `frozen_at` field, so future summaries should cite commit time as the effective
   freeze for the strengthened clauses.

No blocking implementation defect was found for synthetic CPU capability.
Before any real candidate-3 run, the H5 checklist still requires exact actual
imports plus predeclared ordinary and transition padded-window equality on the
accepted path, full configuration/lineage, resources and managed-run gates.

## Implementation checks

- Done: exact executed source/commit, packet seal, external source hashes and
  clean DARTsort state -> all match.
- Done: request/read extent, FFT length, padding/core behavior and positive
  control -> source-traced, rerun and independently adversarially checked.
- Done: uint16, scale/shift, channel/sign/centering/CAR/filter order -> source
  traced and manual known-answer array-equal without using the installed filter
  method for the expected result.
- Done: IDs, geometry, multi-segment clocks, axes, float32 and invalid inputs ->
  source and fixtures agree.
- Done: DARTsort none/no-copy/no-workdir path -> exact source and interpreter
  consumer establish same-object/no-workdir behavior.
- Done: freeze chronology -> preliminary text retained; strengthened contract
  committed before final run and explicitly not called preregistered.
- Not done: real voltage/H5 equality, throughput, GPU/service/sort, full-session
  behavior or scientific validation of Kilosort's filter itself -> excluded.
- Can establish: exact synthetic request-context adapter capability at the
  frozen installed Kilosort immediate-prewhitening and DARTsort no-copy boundary.
- Cannot establish: real-data equality, chunk invariance, current H5 readiness,
  safe runtime/resources, full-session completion or scientific performance.
