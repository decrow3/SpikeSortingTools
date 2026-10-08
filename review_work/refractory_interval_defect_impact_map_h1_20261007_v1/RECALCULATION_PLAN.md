# Cheapest bounded correction

Do not reread voltage, regenerate waveforms, rerun a sorter, or touch RF/holdout. Reuse the frozen curated spike-time/cluster arrays, exact probe clocks, field cells, 300-s blocks, and deterministic bootstrap indices.

1. Freeze a fresh correction contract before outcomes. Bind both R1c manifests, the exact evaluator source snapshot, spike-time/cluster hashes, field/timing receipts, state/segment rules, bootstrap seed/draw count, and the unchanged verdict map.
2. Add a known-answer fixture in which a lag at every 7-sample boundary proves the observed-bin support and the confidence upper edge are the same exact sample-derived value `(k+1)*7/fs`. Include a negative fixture demonstrating that nominal 0.25-ms edges fail at the real clocks.
3. Traverse saved spike times once per probe/arm and emit compact per-unit × 300-s-block × scope (`all`, `q0`, `displaced`) 40-bin positive-lag count tensors, per-block spike counts, and exposure matrices. Preserve exact block and field-segment restrictions.
4. Recompute sliding-RP using `upper = arange(1,41)*7/fs` and eligibility defined in the same physical clock. The contract must state whether the historical `center > 0.5 ms` rule becomes exact sample-time centers or an explicitly chosen sample-index boundary; this is a scientific rule that must be frozen, not inferred during execution.
5. Recompute point P/K and the original deterministic block bootstrap. Keep the 9-29-sample guardrail unchanged and verify bit-for-bit equality as a negative control.
6. Reapply the frozen verdict map without tuning. Publish old-versus-corrected unit transition counts, P/K point values and intervals, guardrail equality, and decision transitions. Never overwrite R1c or the historical census/selection packets.
7. Regenerate the post-sort census and deterministic panel selection only if corrected categories change. Rerun panel measurement only if changed membership affects the intended next decision.

Resource condition: CPU-only, no voltage or waveform reads, no GPU, no sorter, no RF/holdout. The exact byte/runtime caps should be derived from a timing-only fixture and frozen in the correction contract. Stop after sealed sufficient statistics and corrected scorecard, or preserve a failed namespace on any hash, closure, fixture, or resource violation.

The cheapest useful first stage is steps 1-4 on a small deterministic known-answer fixture plus one unit per probe/arm. It establishes implementation correctness and artifact sufficiency; it cannot establish corrected full-session decisions until the full saved spike-train traversal and frozen bootstrap complete.
