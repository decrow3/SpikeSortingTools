# DARTsort descriptive scorecard row-binding failure review

Status: **BLOCKED_BY_FROZEN_CONTRACT_MISMATCH**. The scorecard stopped before feature outcomes and the failed namespace was preserved.

## What actually ran

- Managed unit: `h1-dartsort-descriptive-scorecard-v1.service`, PID 4126715.
- Exact scorecard source SHA-256: `76674fa91bfb5b86f33de6c77c5f76ce6076dbb85c823bc40cde3405b3aabe6f`.
- Frozen scorecard contract SHA-256: `21f21e26404172e7a369d806d30ab2966421a3b473305253aea50141b854e992`.
- Bound HDF5 SHA-256: `cc70606126e94367bdab76fc46afee08164838de55f1801d367a83a660969731`.
- Bound sorting NPZ SHA-256: `772acc6a810bd65ea3dd8b99f197c4b8d3fcfb9248091027877d26fca50e7aa7`.
- Bound q lattice SHA-256: `92d7a28ccc808f76924a6ba392eb71c1a9b0b867796615a15307d1c59bc7df4f`.
- The run read only lattice metadata and NPZ/HDF5 `times_samples` and `labels` before stopping. It did not read amplitude/localization outcomes, raw voltage, RF, or holdout data.

## Failure and narrow diagnostic

The frozen input contract required NPZ and HDF5 `times_samples` and `labels` to be element-for-element equal. They are not:

- Rows: 66,286,937.
- Same-row time equality: 65,739,565 (99.1742385%); 547,372 time mismatches.
- Mismatching time deltas (`NPZ - HDF5`): -4 through +2 samples, median +1; five distinct deltas.
- The exact time multisets also differ, so this is not merely row permutation.
- Same-row label equality: 69,057 (0.1041789%); 66,217,880 label mismatches.
- Either time or label differs on 66,217,881 rows.

## Executed implementation interpretation

The executed wrapper saved the returned final `DARTsortSorting` to `dartsort_sorting.npz`. The DARTsort source at frozen repository commit `2162273edd83276e1698cf3b58a8766adf8d4be1` explains the mismatch:

- `DARTsortSorting.save` stores the current (final/ephemeral) `times_samples`, `channels`, and `labels` in the NPZ while retaining `parent_h5_path` for persistent features.
- `DARTsortSorting.load` reconstructs persistent features from that parent HDF5 and then replaces `times_samples`, `channels`, and labels with the NPZ values, preserving row-index association rather than requiring equal HDF5 label/time datasets.
- `clustering.cluster_util.apply_reclustering` may update both labels and event times through template alignment shifts. The observed small time deltas are consistent with that code path, but this diagnostic does not prove which individual refinement stage changed each row.
- Provenance records only `experiments/dataset_pipeline/*` changes relative to that commit, not changes under `src/dartsort`, so the committed library implementation is the applicable source for these semantics.

## Required correction before any scoring

A separately versioned, pre-outcome contract must replace the invalid equality rule with an explicit row-index binding:

1. Use NPZ `times_samples` and NPZ final `labels` for event time/state and unit membership.
2. Use HDF5 row `i` for `denoised_ptp_amplitudes[i]` and `point_source_localizations[i]`.
3. Verify NPZ `parent_h5_path` resolves to the bound `matching1.h5`.
4. Verify NPZ and HDF5 event counts agree and NPZ `channels` equals HDF5 `channels` element-for-element; this channel check has not yet been run.
5. Add an independent known-answer fixture showing that `DARTsortSorting.load` preserves persistent-feature row association while replacing final times/labels/channels.
6. Preserve this failed contract and namespace; execute a corrected contract only in a fresh namespace.

Implementation checks
- Done: exact managed command/source/config/input hashes inspected; authoritative journal and preserved failure receipt establish exit 1.
- Done: full-array same-row time/label comparisons, exact time-multiset comparison, mismatch counts, delta range, and array-byte hashes computed in a bounded diagnostic.
- Done: frozen DARTsort commit implementation inspected for save/load, persistent-feature, label-replacement, and time-realignment semantics.
- Not done: NPZ/HDF5 channel equality and a standalone loader known-answer fixture; both are prerequisites for a corrected contract.
- Not done: amplitude, localization, state scorecard, or scientific outcome review; the run stopped before those reads.
- Can establish: the original equality premise is false and HDF5 labels are not final labels; DARTsort's implementation is designed to pair HDF5 persistent features with NPZ final arrays by row index.
- Cannot establish: that every required row-index binding is valid until channels and the loader fixture pass, or any R1/WIU, production, physical-amplitude, curated-yield, biological-identity, or sorter-causal conclusion.
