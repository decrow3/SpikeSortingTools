# A four-unit saved-array QC correction: review candidate

This is a fresh, execution-disabled correction candidate. It does not revise the historical one-shot QC result: that result remains **FAIL**, including its four unit-278 exact-remap failures, twelve global-peak distances, visibility flags, and all source tables. This candidate uses a new output namespace and asks whether corrected implementation semantics warrant a separately authorized rerun.

The defect addressed here is the v1 assumption that one event-center state applies to every sample in a retained member. The executed A materializer instead used the AM.3 field's depth-median rigid projection, full-session median reference, 40-um half-away rounding, and one piecewise-constant state per sample. At the exact right edge of a half-open cell, `searchsorted(..., side="right")` assigns the sample to the later state. The bound clock is 29,999.835983263598 Hz, global frame/time origins are zero, field centers run from 0 to 10,473.5 seconds at 0.25-second spacing, and the nine observed rounded states are frozen in the config.

The corrected implementation reconstructs every retained sample's global frame and motion state, then remaps original voltage into corrected coordinates column by column. It disables fixed state-specific original anchors; support is channel-by-time; conditional whole-array q0 identity is evaluated only when every member sample is q0; supported local correlations are descriptive; and no-shift/wrong-sign controls are counted only when they demonstrably differ from the positive expected array.

Local detectability is non-tautological. The domain is the predeclared set of channels within 80 um of the frozen corrected anchor. The gate is local maximum CAR-centered PTP SNR at the unchanged threshold of 6. The all-channel global peak is descriptive only. There is no peak-distance gate, localization claim, identity claim, purity claim, motion-correctness claim, or efficacy claim.

Six synthetic known-answer fixtures pass, including an interior state change, an exact-edge state change, an unchanged state, a strong distant distractor plus known local signal, a distant-only/no-local-event case, and nonvacuous/disabled negative-control behavior. Metadata-only preflight verified the field, geometry, selection, and historical-result bindings while opening zero retained archives.

No recording was read. Neither retained waveform archive was opened. No real QC, sort, training, detection, tuning, RF/holdout access, or voltage transfer occurred. The packet contains no voltage.

## Implementation checks

- Done: traced the executed A source through rigid projection, median reference, half-away rounding, explicit zero time origin, per-sample frame-to-time conversion, half-open cell edges, exact-edge assignment, exact-coordinate remap, and zero-fill semantics.
- Done: bound the executed materializer source, rounding helper, remap helper, frozen materialization config, field, accepted recording manifest, geometry, selection receipt, and original FAIL summary by SHA-256.
- Done: replaced every identified single-state-dependent anchor/support/correlation/control assumption with per-sample semantics or an explicit disable rule.
- Done: froze a local-domain SNR metric whose result is independent of the descriptive global-peak location, and passed six synthetic fixtures.
- Not done: retained-array execution or validation of a corrected real-data result; independent H1 review and a separate launch decision are prerequisites.
- Can establish: this packet is a bounded, source-complete, outcome-independent correction candidate faithful to the executed A materialization clock and field semantics.
- Cannot establish: whether corrected real-data controls pass, local signals are detectable, biological identity/purity, localization, motion correctness, or sorter benefit.
