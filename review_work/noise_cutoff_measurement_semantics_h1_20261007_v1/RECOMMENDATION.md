# Recommendation: NON_DECISIVE

The Arm-A noise-cutoff gap is real in the published scalar table, but the available evidence is non-decisive about whether it reflects metric construction/input effects or a sorting-signal defect.

Independent recomputation from `CENSUS_UNITS.csv` confirms in-domain finite-failure prevalence of 434/653 (66.46%) for imec0 Arm A versus 308/586 (52.56%) for REF, a +13.90 percentage-point gap, and 401/553 (72.51%) versus 287/451 (63.64%) on imec1, a +8.88-point gap. There are no duplicate `(probe, arm, unit_id)` keys and no scalar-to-pass-flag mismatches. This establishes the saved-table pattern only; populations are unmatched across arms.

The corroborating source shows that the metric is a dimensionless ratio of histogram **counts**, not an amplitude quantile or amplitude-space distance. It uses 100 edges (99 bins), a different maximum and therefore different bin width for every unit/subset, the count in the second occupied bin as `first_low`, and a tail-count population standard deviation. Independent fixtures confirm finite pass/fail behavior, undefined support/variance branches, strict failure at exactly 5, and that one extreme maximum can rescale all edges and change a finite pass into undefined.

Two load-bearing inputs needed to distinguish construction from signal are absent:

1. the exact accepted evaluator bytes (declared SHA-256 `415f29d7...`; the only located snapshot hashes to `a31f1349...` and is nonbinding); and
2. per-unit/subset histogram counts, edges/maxima, retained/invalid event counts, and intermediate peak/tail/first-low diagnostics.

The saved scalar and pass flag can reproduce the downstream decision, but they cannot reconstruct or audit its numerator, denominator, undefined branch, dynamic range, or sensitivity to an extreme maximum. Therefore do not use the adjusted gap as evidence of within-neuron deterioration, a causal sorting stage, biological identity/purity, or physical amplitude incompleteness.

## Cheapest next discriminating test

Before any waveform or raw-voltage work, run one bounded source-host export from the already existing curated `amplitudes.npy` arrays under a new frozen contract:

- publish the exact executed evaluator snapshot;
- for every full/q0/displaced unit subset, publish only compact sufficient statistics: finite and invalid counts, maximum, exact 100 edges or their defining range, 99 histogram counts, peak index/count, occupied-top count, high-start, high-tail count/mean/population standard deviation, second occupied-bin count, cutoff, margin to 5, and explicit undefined reason;
- independently reconstruct every scalar from those statistics and require exact/roundoff-bounded equality to the saved table;
- add two predeclared diagnostics on the same amplitudes: (a) recompute after removing only the single maximum to test dynamic-range leverage, and (b) deterministic count-matching within each unit/subset to a frozen support count to test sparse-occupancy dependence. Preserve original and diagnostic outputs separately; do not change the historical `<5` decision.

This test is compact, needs no recording or waveform read, and can distinguish implementation/provenance failure, extreme-maximum binning leverage, and finite-count occupancy sensitivity. Even if the gap survives, unmatched arm populations still prevent a within-neuron harm claim; a matched-identity signal test would then require a separately frozen correspondence design.

## Implementation checks

- Done: packet identity and integrity -> independently hashed 16 members in each input packet; both `COMPLETE.json` files bind the observed MANIFEST SHA-256 and no member failures were found (`INPUT_VERIFICATION.json`).
- Done: saved metric pattern -> independently read 2,766 unit rows, found zero duplicate `(probe, arm, unit_id)` keys and zero saved full-pass mismatches; reproduced +13.90 pp imec0 and +8.88 pp imec1 in-domain A-minus-REF gaps (`verify_inputs.py`; `INPUT_VERIFICATION.json`).
- Done: consuming measurement semantics -> traced the corroborating source at lines 59–87, 246–318, and 323–335; documented units, 100-edge/99-bin construction, caps/defaults, dynamic support, strict threshold, and state eligibility (`MEASUREMENT_DAG.md`; source SHA-256 `a31f1349...`).
- Done: independent known-answer fixtures -> seven cases passed without importing the production helper, including manually derived pass/fail values, undefined branches, exact-threshold failure, and dynamic-maximum perturbation (`independent_fixture.py` SHA-256 `58740e9b...`; `FIXTURE_RECEIPT.json` SHA-256 `a4cf5da9...`).
- Done: saved-data sufficiency -> confirmed all four probe/arm availability rows mark amplitude distributions absent; histogram and intermediate sufficient statistics are not published (`SAVED_DATA_SUFFICIENCY.csv`; `INPUT_VERIFICATION.json`).
- Not done: exact executed-source review -> declared commit `ba21bc97...` is absent locally and the configured Git remote returned `not our ref`; declared SHA-256 `415f29d7...` bytes remain unavailable.
- Not done: scalar reconstruction from saved sufficient statistics -> exact histogram counts/edges and retained-input diagnostics are not saved in the compact packet.
- Not done: within-neuron or causal attribution -> no cross-arm identity mapping is assumed, and the current task forbids expansion to waveform/voltage work.
- Can establish: the saved in-domain scalar failure gap and pass flags are internally consistent; the corroborating metric construction is count-, occupancy-, and dynamic-maximum-sensitive; current compact evidence cannot discriminate construction/input sensitivity from a sorting-signal defect.
- Cannot establish: exact accepted evaluator semantics, cause of the gap, within-neuron deterioration, physical amplitude completeness, causal pipeline stage, biological identity/purity, or arm superiority.
