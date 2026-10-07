# Post-execution review: historical-MEDiCINe DARTsort saved-output scorecard v2

Verdict: **reviewed descriptive result; no production decision**.

## Execution and integrity

- Managed unit `h1-dartsort-historical-medicine-scorecard-v2.service` completed without a failure receipt.
- Runtime was 518.03 s; raw-voltage bytes 0; GPU false; 65,287,555 accepted events and 1,872 final uncurated units.
- Every run payload matched `RUN_MANIFEST.json`; `RUN_COMPLETE.json` matched the manifest hash.
- Exact output row counts: 18,720 unit-state rows and 1,872 unit-summary rows.
- State event counts sum to 65,287,555; state exposures sum to 10,473.553879363088 s; every aggregate rate recomputes as events/exposure.

## Scoped descriptive findings

- Aggregate final-time event rate was 6,285.6405 Hz during external `q=0` exposure and 5,985.3297 Hz across external displaced exposure, a ratio of 1.05017.
- Among 1,744 units with at least 100 events in each comparator group, the median within-unit `q0/displaced` rate ratio was 1.10185; the 10th–90th percentile range was 0.16017–5.11488. Heterogeneity is therefore much larger than the aggregate difference.
- In those same state-supported units, the median within-unit denoised-PTP median ratio was 1.00567 (10th–90th percentile 0.88931–1.14990). These are noise-normalized DARTsort denoised PTP values, not microvolts, Kilosort amplitudes, SNR, or physical completeness.
- The median per-unit displaced-minus-q0 model-derived observed-z shift was -36.08 um (10th–90th percentile -43.35 to -21.89 um). This is an amortized point-source field and is confounded with time/state support; it is not ground-truth motion or registered depth.
- Exactly one accepted event crossed an external-q boundary after final DARTsort time adjustment (`-200` to `-80`). It was reported and excluded from all state-sensitive HDF5 feature summaries. Features were not re-extracted at final adjusted times.
- Segment-safe totals were 0 exact duplicates, 27,143 intervals of 1–8 samples, and 567,968 intervals of 9–29 samples among 61,709,887 positive segment-safe intervals. These are contamination proxies, not proof of purity or bad merges.
- SlidingRP returned a finite minimum-contamination value for 511/1,872 units and 110 units met the saved 10%/90%-confidence flag. Missing values must not be interpreted as failures or passes.

## Interpretation limits

The archive was sorted with its historical four-depth MEDiCINe field. The Arm-A q lattice is only an external common time partition. The result does not evaluate lattice-remapped DARTsort, `em2f_full_20260929`, `rounded_kriging_v1`, sorter-only effects, training effects, R1/WIU, production benefit/harm, physical amplitude completeness, curated yield, biological identity, or purity. Extreme q states have little exposure (for example, 1–4 s), so their aggregate rates are descriptive and unstable.

Implementation checks
- Done: executed source, frozen v2 contract, effective configuration, exact input identities, pointer/row/channel/geometry/feature-precedence validation, and actual-loader positive/negative fixture inspected.
- Done: run-manifest hashes, completion hash, output row counts, state totals, exposure totals, aggregate rates, unit distribution summaries, interval totals, and boundary-crossing transition independently recomputed from saved outputs.
- Done: final NPZ times/labels govern timing metrics; original HDF5 extraction times govern HDF5 feature state labels; the one crossing event is excluded as frozen.
- Not done: raw-waveform or physical-amplitude validation, external-arm matching, inferential uncertainty, curation, biological identity, or any causal attribution.
- Can establish: exact state-conditioned final-timing and saved-feature descriptions for this one uncurated historical-MEDiCINe DARTsort archive.
- Cannot establish: lattice-remapped performance, sorter/training causality, R1/WIU, production value, physical amplitude completeness, curated yield, identity, purity, or generalization.
