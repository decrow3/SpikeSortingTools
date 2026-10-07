# DARTsort descriptive scorecard freeze

This packet freezes the input request and scorecard before any outcomes are computed. It is a new descriptive analysis, not the rejected/original R1 production endpoint.

The state definition is not invented here. It is the existing imec1 Arm-A lattice: 41,895 quarter-second cells from AP frame zero, q=0 outside the canonical mask, and inside-mask displacement rounded half away from zero to the 40 µm lattice. The expected lattice SHA-256 is fixed in the request. DARTsort's separate four-depth MEDiCINe field may not replace it.

The amplitude and spatial names deliberately carry their implementation semantics. `dartsort_denoised_ptp_*` refers to maximum-channel PTP of DARTsort's denoised waveform in the noise-normalized recording. `amortized_*_z_abs` refers to model-derived observed point-source depth. Neither is relabeled as a Kilosort amplitude, PC norm, physical µV signal, voltage-peak depth, or motion truth.

Implementation checks
- Done: freeze order -> input schema, population, mappings, metrics, interpretation, exclusions, resources, and stop conditions were fixed before opening any requested q-lattice payload or computing outcomes.
- Done: clocks/state -> exact sample rate, frame count, AP-frame-zero origin, 0.25 s grid, half-open boundary rule, state values, mask binding, reference, rounding, and sign are explicit (`imec1_part2b_contract.py:33-98`; reviewed Part2b config field block).
- Done: source compatibility -> final HDF5 row count/schema, accepted-label rule, unit count, persistent amplitude/localization definitions, and final NPZ identity are bound to the prior compatibility audit.
- Done: axes/domains -> z_abs is column 2 of amortized point-source localizations; channel index is forbidden as measured depth; observed and registered frames are not mixed.
- Done: caps/defaults -> no unit-quality filter; state-support subset requires 100 events in both q0 and displaced states; 60 s presence retains the final partial bin; all missing ratios remain missing rather than receiving a pseudocount.
- Done: matching/counting -> comparisons are within the same DARTsort unit only. State-conditioned adjacent intervals require a common state and contiguous state segment; bootstrap-block boundaries are also exclusive if block summaries are later used.
- Done: circularity -> no DARTsort outcome arrays were aggregated or scored while drafting this contract.
- Not done: q-lattice binding -> the expected lattice is still absent from H1/shared storage and must be supplied with the frozen SHA-256.
- Not done: HDF5 immutable identity -> size/schema are known, but a full hash or reviewed chunk manifest must be provided and validated before execution.
- Not done: scoring/review -> execution waits for both compact bindings; an independent post-execution review remains required.
- Can establish: a bounded no-sort, no-voltage-read descriptive analysis can run once compact identities bind.
- Cannot establish: R1 P/K, production status, physical amplitude completeness, curated yield, biological identity/purity, or sorter-only causality.
