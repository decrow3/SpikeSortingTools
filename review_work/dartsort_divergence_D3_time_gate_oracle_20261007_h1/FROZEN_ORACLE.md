# Frozen D3 time-gate oracle

Frozen at 2026-10-07T21:45:39-07:00 on huklaban1, before inspecting any H5 diagnosis beyond the coordinator-reported terminal error string `ContractError: DARTsort final/H5 time rows differ`. No production input or outcome artifact was opened.

## Verdict

`REPAIR_REQUIRED_EXACT_EQUALITY_GATE_INVALID`

The D3 v2 assertion that every final-NPZ `times_samples` value must equal the parent `matching1.h5` `times_samples` value at the same row is not a valid DARTsort row-binding invariant. At repository commit `fdaad62118d778eec63f8e755a0b0f1422574aff`, the final matching H5 is loaded into a `DARTsortSorting`, then optional clustering/refinement runs, and only afterward is the final NPZ saved. Template realignment adds label-dependent shifts to event times, while reclustering subtracts merge-alignment shifts from selected source-unit events. Saving writes the then-current time array while retaining the parent-H5 pointer. Loading a saved NPZ explicitly reloads persistent features from the parent H5 and replaces the H5 times/channels with NPZ times/channels. These are direct source semantics, not an inferred exception.

The expected general relation is therefore rowwise

`final_time[row] = matching_time[row] + cumulative_postmatching_shift[row]`.

It is **not** justified to identify that cumulative shift with `matching1.h5["time_shifts"]`, with either sign. Peeling applies the H5 `time_shifts` feature to `spike_times` before emitting the H5 row, so that feature is already reflected in `matching_time`. Later realignment and merge shifts are distinct operations. With `save_intermediates=false`, the reviewed compact evidence does not establish that every postmatching shift contributor was saved, so exact delta reconstruction is not a prerequisite for valid row-index binding unless new saved provenance proves it is possible.

The source-defined signs and indexing are fixed rather than fitted from the observed delta: peeling adds the per-row H5 feature before H5 emission; template realignment adds a per-current-label template shift to every valid row carrying that label; reclustering subtracts a source-label-pair merge shift only from rows of the nonreference source label before relabeling. Relabeling, depth reordering, deduplication, and boundary invalidation change labels (including setting some to `-1`) but the reviewed paths retain the row array and its order. No nearest-time alignment between final and H5 arrays is permitted as ancestry evidence.

For the frozen cross-pipeline question, final-NPZ time is the event clock for DARTsort-to-Kilosort correspondence and half-open support selection. Parent-H5 time is the earlier matching-stage clock used only to attach matching labels/features to the same source row. Using H5 time for final-event correspondence would undo legitimate postmatching alignment; using final time to invent a nearest H5 parent would make assignment attribution circular.

## Source-bound invariants for a repaired gate

The repaired gate must freeze and test all of these before consuming a matching feature row:

1. Resolve final-NPZ `parent_h5_path` relative to the NPZ parent and require it to equal the frozen H5 path.
2. Require identical row counts for final times, final labels, final channels, parent-H5 times, parent-H5 channels, parent-H5 labels when used, and every consumed feature dataset.
3. Require exact rowwise equality of final and parent-H5 `channels`. The reviewed postmatching paths alter times and labels, not row order or channels.
4. Treat row index as the ancestry key. Do not join final and H5 rows by timestamp.
5. Compute `delta = final_times - matching_times` for every row, chunkwise, and publish only compact diagnostics: dtype; row count; nonzero count and fraction; min/max; complete value counts when the unique set is bounded, otherwise a bounded histogram plus deterministic samples. Report the already-source-defined H5 fact that its `time_shifts` was added before H5 emission; do not choose a sign or relation by fitting observed `delta` to `+/- H5 time_shifts`.
6. Require final times to remain inside the frozen half-open recording support for final labels used in comparison. Report, rather than silently discard, shifted out-of-support rows and negative labels.
7. Before temporal matching, stably sort each final-label train by `(final_time, source_row_id)` and carry the original source row IDs through every match and emitted relation. Do not assume final NPZ rows remain globally or within-unit time-sorted after label-dependent realignment/merge shifts.
8. Add a known-answer fixture with at least two source labels merged into one final label, nonzero shifts of both signs, equal row/channel ancestry, unequal final/H5 times, and a local time-order inversion. The fixture must pass row binding, preserve source row IDs, and yield the expected sorted matching result. Negative fixtures must fail on parent-path drift, row-count drift, channel drift, and feature-row drift.
9. Preserve the failed v2 output namespace and contract. Publish any repair as v3 with new hashes and obtain independent review of the delta only before a new managed run.
10. If source-backed exact row ancestry cannot be established after the count/path/channel checks, the final-event DARTsort-to-Kilosort comparison may still answer the frozen final-output correspondence question using final times and labels, but matching-to-final assignment attribution must be omitted and explicitly marked unavailable. Never substitute nearest-time final/H5 alignment.

## Invalidated and retained claims

- Invalidated: D1/D2/D3 language making exact final/H5 time equality a necessary row-binding condition.
- Invalidated: D3 v2 `STAGE_VALIDATION` specification insofar as it requires that equality.
- Invalidated: any conclusion that the terminal failure demonstrates corrupt DARTsort ancestry or unusable final-to-feature correspondence.
- Not produced and therefore unavailable: D3 candidate rankings, event relations, pair summaries, or scientific divergence conclusions.
- Retained: frozen input hashes, parent-path resolution repair, resource/lifecycle controls, and the source-supported use of common row index after the corrected ancestry checks.
- Retained only as a reported fact pending direct artifact access on H1: H5 reported a terminal pre-evaluation failure with no raw-voltage or GPU work.

## Cheapest adequate correction validation

Patch only the adapter/runner gate and per-unit ordering, run the fixed-small adversarial tests, then perform one bounded full-row saved-time-relation pass on H5 that emits the compact delta/channel/count receipt before any candidate analysis. Do not rerun sorting, voltage processing, fit a sign relation, access RF/holdout, or relaunch production until the v3 contract, runner, tests, and independent review are frozen.

## Implementation checks

- Done: what actually ran -> reviewed accepted D3 contract `81ac32ec196aea9336f8883eca5cdfd868fde5c1a91011ec6c252a93b45dbfa1`, runner `fc304240486f48f2791159256f1320d51a37ea08fd03175387256e35b3ca5a23`, and adapter `60ecddfdd01d1fbcdeffaa77642d6e4a88bf689dc7a7b1d9275b303efd37526b`; exact equality is enforced by adapter lines 115-146 and called before evaluation by runner lines 421-440.
- Done: producer source semantics -> at DARTsort commit `fdaad62118d778eec63f8e755a0b0f1422574aff`, `main.py:358-419` loads matching output, optionally clusters/refines it, then saves; `data_util.py:647-707,737-783,785-847` establishes H5 load, current-state NPZ save, parent pointer, and NPZ-over-H5 replacement semantics.
- Done: time-shift signs/stages -> `realignment.py:63-87,303-346` adds template shifts; `cluster_util.py:62-101` subtracts selected merge shifts; `peel_lib.py:446-452` shows H5 feature `time_shifts` is applied before the H5 `times_samples` row is returned.
- Done: row/order risk -> reviewed D3 `candidate_index` lines 253-266; it groups rows by stable label order but does not sort shifted times. DARTsort itself explicitly sorts where time order is required (`data_util.py:148-175,233-249`), confirming that stored order is not a universal invariant.
- Not done: direct inspection of H5 failure packet or forthcoming H5 diagnosis -> intentionally excluded so this oracle was frozen independently; the H5 path is also not mounted on H1.
- Not done: full executed pipeline script/effective config bytes -> compact D0 packet provides their hashes but not bytes; exact presence and magnitude of each postmatching stage in this historical run remain to be checked from those frozen files on H5.
- Not done: actual final-minus-H5 delta distribution or channel equality -> requires the bounded saved-array diagnostic on H5; no production or scientific outcome inspection was authorized for this independent freeze.
- Can establish: exact final/H5 timestamp equality is not a valid general DARTsort row-binding gate, the v2 failure is non-scientific and pre-evaluation, and a bounded v3 repair/diagnostic is required.
- Cannot establish: the actual delta distribution, whether every row/channel ancestry condition passes, any D3 candidate or divergence result, biological identity, purity, or sorter-only causality.
