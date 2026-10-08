# Candidate-3 lazy Kilosort prewhitening wrapper

Verdict: `GO_SYNTHETIC_LAZY_WRAPPER_CAPABILITY_REAL_H5_EQUALITY_STILL_REQUIRED`.

The committed adapter converts each exact SpikeInterface trace request into one installed Kilosort `BinaryFiltered.filter` call with whitening disabled. It adds no hidden padding or cache. Parent uint16 conversion, optional scale/shift, selected channel order, sign, per-channel centering, global lower-median CAR and the request-length-dependent 300 Hz FFT high-pass follow the installed source order. Output is float32 with selected channel IDs, geometry and segment clocks preserved.

This is intentionally request-context preserving, not chunk-invariant. Callers needing a padded core must request the padded interval and crop afterward. Because Kilosort constructs the FFT filter with `NT=X.shape[1]`, independently filtering a cropped core is expected to differ and is retained as a positive control.

The separate DARTsort interpreter consumer exercises the functions used by `dartsort.main.dartsort`: `ds_will_copy_recording`, `preprocess`, and `ds_all_to_workdir`. Under `preprocessing=none`, `copy_recording_to_tmpdir=no`, and `work_in_tmpdir=false`, it requires the exact same recording object and values with no work directory.

The frozen combined run passed 22/22 tests with zero failures, errors or skips. Ten directly exercise the new wrapper and cross-environment DARTsort boundary; twelve retain the existing exact-lattice remap coverage. This is synthetic capability only. The H5 checklist still requires the actual H5 imports and predeclared ordinary/transition real-window equality before a full-session launch.

## Implementation checks

- Done: executed source binding -> committed wrapper, contract, fixtures and consumer are hash-bound to commit `aea7431c`.
- Done: Kilosort semantics -> wrapper calls the exact installed `BinaryFiltered.filter`; request length is the exact parent interval length.
- Done: axes/clock/geometry -> output remains time-by-channel and tests bind selected IDs/order, geometry and nonzero per-segment origins; 22/22 combined tests passed.
- Done: scaling/dtype -> uint16 signed conversion and explicit pre-channel-map scale/shift are tested; output is float32.
- Not done: real Arm-A equality, throughput, H5 imports/resources or full-run context -> excluded and still required by the H5 checklist.
- Can establish: synthetic adapter-boundary capability if the frozen fixture and review pass.
- Cannot establish: current H5 readiness, real-voltage equality, runtime sufficiency, full-session completion or scientific performance.
