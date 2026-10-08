# Frozen contract: noise-cutoff measurement semantics

Status: `FROZEN_BEFORE_ADDITIONAL_SUFFICIENT-STAT_OUTCOME_INSPECTION`

Frozen: 2026-10-07 (America/Los_Angeles)

Owner: H1

## Milestone and decision

- Milestone: M1 to M2 diagnostic handoff.
- Decision: determine whether the persistent Arm-A noise-cutoff gap is presently attributable to metric construction/input sufficiency, or whether the saved evidence warrants a sorting-signal interpretation.
- Allowed recommendation: exactly one of `RETAIN`, `REPAIR`, or `NON_DECISIVE`, followed by the cheapest adequate next signal test.
- This packet cannot establish within-neuron deterioration, causal sorting stage, biological identity, purity, or arm superiority.

## Frozen inputs

1. `/mnt/NPX/Luke/DARTsort_motion_experiments/noise_cutoff_interpretation_audit_h1_20261007_v1`, MANIFEST SHA-256 `7c669a036277a7658d9a1d2380873dfe3a52e2febf21df8c772012e5cea233f5`.
2. `/mnt/NPX/Luke/DARTsort_motion_experiments/post_sort_structure_census_h5_20261007_v1`, MANIFEST SHA-256 `022ad0e1c558a283a97cabe7f27aaa318701bc228197ea7819ae75bcae2f1028`.
3. Declared accepted executed evaluator: commit `ba21bc97ad177ba5f2a1b79fb69d37f4f3eac3ad`, path `testing/full_session_r1_evaluation.py`, SHA-256 `415f29d7f26102606143d0bd4fbb8ed9a136199a6642e664d50c2a6716b7c423`.
4. Corroborating, explicitly nonbinding snapshot: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_full_session_r1_imec0_rescore_review_request_20261004_v1_h5/source/full_session_r1_evaluation.py`, SHA-256 `a31f1349854eff0e4c732063993be37e8cf53de5d206628f66d68a65ff7f341e`.

The corroborating snapshot must not be substituted for the declared accepted evaluator. If the exact accepted bytes remain unavailable, semantic claims are source-qualified and the final recommendation cannot be `RETAIN` on construction evidence alone.

## Frozen measurement definition to verify

The corroborating snapshot defines a per-unit scalar from saved Kilosort `amplitudes.npy` values:

1. cast to float64 and discard non-finite values;
2. return undefined for empty input, any retained negative value, or nonpositive/non-finite maximum;
3. use `np.linspace(0, max(amplitude), 100)` as histogram edges, which yields 99 bins;
4. set `peak_i = argmax(hist)`;
5. count occupied bins in `hist[peak_i:-1]` (last bin excluded);
6. set `high_start = ceil(2 * 0.25 * occupied_top + peak_i)`;
7. take positive-count bins from `hist[high_start:]` as the high tail;
8. use the count in the second occupied bin (`nonzero[1]`) as `first_low`;
9. return `(first_low - mean(high)) / std(high)` when the tail is nonempty, at least two histogram bins are occupied, and population standard deviation is positive; otherwise undefined;
10. pass only when the result is finite and strictly less than 5. Exactly 5 fails.

The output is a dimensionless function of histogram counts. It is not an amplitude quantile or an amplitude-space distance. Dynamic per-unit maxima imply different amplitude bin widths across units and state subsets. Finite-sample occupancy and tail-count variance create count dependence; uniform multiplication of a fixed count vector need not change the ratio, but sparse sampling, altered maxima, and redistributed counts can.

The separate historical conditional flag is not the R1 decision rule. The audit's `q0_spikes >= 100` and `displaced_spikes >= 100` criteria are analysis/state eligibility conditions, not gates inside the scalar function.

## Frozen tests

- Trace the declared source identity and the consuming call sites; distinguish exact executed bytes from corroboration.
- Build independent known-answer fixtures from explicit histogram count vectors and manual arithmetic. Do not import or call the production helper.
- Positive controls: a finite value below 5 and a finite value above 5.
- Negative/boundary controls: empty input, negative value, zero tail variance, strict threshold at exactly 5, and a maximum/outlier perturbation.
- Inventory saved artifacts against the minimum sufficient statistics: exact histogram counts and binning/max for each evaluated subset, plus retained-count/invalid-input diagnostics. A saved scalar and pass flag can reproduce only the downstream decision, not validate construction.
- Verify table grain, population, availability, missingness, caps/defaults, and state eligibility. Preserve unmatched-population and selection limitations; no cross-arm unit identity is assumed.

## Bounds and prohibitions

- CPU only; under 1 GiB peak RSS; under 64 MiB aggregate input reads; under 25 MiB outputs.
- No raw voltage, recording reads, waveform experiment, new H5 output, bulk transfer, RF/holdout access, spike sort, parameter sweep, service, or managed job.
- Preserve failed fixtures and conflicting evidence. Never overwrite a prior packet.

## Completion and stop condition

Complete after emitting a measurement DAG, source-line/hash ledger, independent fixture receipts, saved-data sufficiency table, implementation-check block, and one frozen recommendation. Stop early and preserve a failed namespace on any invariant or resource violation. Publish immutable compact evidence with `MANIFEST.json`, then write `COMPLETE.json` last.
