# Corrected bounded recalculation plan

No voltage, waveform, sorter, GPU, RF, or holdout access is required. Reuse frozen spike-time/cluster arrays, clocks, field cells, blocks, and deterministic bootstrap specification.

1. Bind the exact R1c evaluator bytes with SHA-256 `415f29d7...`. If unavailable, label the implementation `RECONSTRUCTED_R1C_CLOCK_REPAIR` and require independent reproduction of every unchanged saved field/point value that is reproducible before interpreting the edge-only delta.
2. Freeze historical observed-count semantics exactly: within unit; sorted samples; `j>i`; integer `d=t[j]-t[i]`; include `0 <= d < 280`; bin `d//7`; require same 300-s block; for q0/displaced require equal Boolean `state_nonzero`; do not add an exact field-segment gate; retain `d=0` same-sample pairs. Any segment-gated or zero-lag-excluding variant is a separately labeled redesign, not the correction.
3. Freeze corrected time edges as `upper[k]=7*(k+1)/fs` and centers as `7*(k+0.5)/fs`, with eligibility `center > 0.0005` seconds. Use exact per-probe clocks and the unchanged contamination/confidence/pass grids.
4. Before real outputs, run an independent boundary fixture and a discriminating zero-count fixture. The author fixture draft is in `review_work/r1c_sample_clock_fixture_spec_20261007_h1/`; it is not itself a frozen production contract.
5. Traverse saved spikes once and emit compact per-unit × 300-s-block × scope 40-bin integer ACG tensors, matching spike-count tensors, and block/scope exposures. Save exact bootstrap indices or bind the exact seed, RNG, and draw contract.
6. Recompute point P/K and frozen bootstrap intervals. Preserve the separate 9-29-sample guardrail and Allen-like outputs bit-for-bit as negative controls. Reapply the frozen verdict map without tuning.
7. Publish old-versus-corrected unit transitions, P/K intervals, guardrail equality, and verdict transitions in a fresh namespace. Never overwrite historical R1c/census/selection packets.
8. Regenerate census and deterministic selection only after the corrected compact tables exist. D1/D2/D3 runner mechanics may be reused, but affected anchors must be revised, independently shown unchanged, or explicitly labeled historical-panel-only before any production interpretation.

Stop on any source/input drift, fixture failure, unchanged-output mismatch, ambiguous reconstruction, resource-bound violation, or nonfresh namespace. Preserve failures without COMPLETE.

The cheapest first stage is the synthetic fixtures plus one unit per probe/arm, sufficient to validate implementation and provenance behavior but not full-session decisions.

## Implementation checks

- Done: one-factor boundary now matches historical counter semantics and excludes the erroneous segment gate.
- Done: `d=0` behavior and exact-source/reconstructed path are explicit.
- Not done: real spike traversal or corrected results.
- Can establish: a bounded contract-ready correction design.
- Cannot establish: corrected scientific decisions until the full frozen recalculation completes.
