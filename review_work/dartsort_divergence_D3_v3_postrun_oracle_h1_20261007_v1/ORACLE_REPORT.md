# D3 v3 post-run oracle

## Frozen verdict space

This packet was sealed without inspecting any future D3 v3 production namespace or outcome.

The consumer has three possible dispositions:

1. `GO_FINAL_REF_A_TO_DARTSORT_CORRESPONDENCE_ONLY`: the exact saved-output schema, bindings, clocks, support, ordering, resource bounds, and final-only attribution policy pass.
2. `FAILURE_CLOSED_NO_SCIENTIFIC_RESULT`: a preserved failure record is present, COMPLETE is absent, and prohibited-work accounting is zero.
3. rejection/blocker: any missing or stale member, schema/field error, mixed COMPLETE/FAILURE state, ordering error, false parent attribution, unsupported clock/support, provenance drift, reconciliation mismatch, or resource violation.

Even a GO permits only direct final-event REF-to-DARTsort and A-to-DARTsort correspondence for the exact eight arm-specific frozen anchors over `[0, 314204094)` acquisition samples with inclusive ±15-sample matching. Intermediate DARTsort matching-to-final assignment is unavailable. No transitive identity, causal stage localization, biological identity/purity, or sorter-only causality may be inferred.

## Consumer coverage

- Verifies both bound predecessor packets, all 15 declared members, and both COMPLETE-to-MANIFEST bindings.
- Requires the exact twelve-file successful output namespace or a failure-only closure.
- Verifies every output member byte size/hash and COMPLETE binding.
- Enforces the final NPZ event clock, half-open support, and zero shifted final-label rows outside support.
- Requires unavailable row ancestry in the time relation, stage validation, provenance, and every stage-event row; parent row and matching label must be null.
- Requires exactly eight uncapped candidate rankings, frozen supports, deterministic metric ordering, exact dense ranks, and passing-label reconciliation.
- Reconciles ranking matches to pair summaries, pair summaries to event rows, exclusive reciprocal matches, unmatched counts, neighbor-degree lists, and exact expected contexts.
- Enforces stable `(sample_time, source_row_id)` serialization within every pair context/side and unique increasing final source-row IDs.
- Checks runtime, RSS, input-read, logical-read, output, raw-voltage, GPU, fit/replay/sort/RF/holdout accounting.

## Synthetic fixture result

Eight tests pass: exact predecessor binding, positive final-only success, missing clock field, stale member hash, reordered relation rows, false parent attribution, valid failure closure, and mixed COMPLETE/FAILURE rejection. A stronger ranking-to-event reconciliation initially exposed an inconsistent positive fixture (25 claimed versus 2 serialized); that failed run is preserved in `FAILED_TEST_RECEIPT_V0.json`, and the repaired fixture now serializes 25 reciprocal matches.

## Implementation checks

- Done: exact executable derivative and technical review bindings -> 11 derivative and 4 review members independently matched sizes/hashes; both COMPLETE files bind their MANIFEST (`postrun_oracle.verify_predecessors`).
- Done: executed source trace -> output schema and seal at runner lines 679–840; failure closure at 843–867; final clock/support and ancestry gate at 252–356; stable unit/event ordering at 209–234, 422–449, and 497–553; stage null-attribution serialization at 556–577; resource accounting at 786–821 (runner SHA-256 `725d811d...`).
- Done: consumer known-answer/adversarial fixtures -> 8 passed in 0.12 s; production helper was not imported (`postrun_oracle.py` SHA-256 `5e0fdf65...`; `test_postrun_oracle.py` SHA-256 `01415890...`).
- Done: output semantics -> exact membership, hashes, completion, final clock/support, stable serialization, summary/event/ranking reconciliation, unavailable ancestry, and resource/prohibited-work checks are encoded before outcomes.
- Not done: future production output inspection or scientific outcome validation -> prohibited until this oracle is sealed.
- Not done: independent recomputation from production input arrays -> would require the separately controlled future output review and is outside this pre-outcome oracle.
- Can establish: whether a future namespace is structurally/provenance-valid for the exact limited final correspondence claim, or is a closed failure/blocker.
- Cannot establish: any future correspondence value now, internal matching-to-final attribution, transitive or biological identity, purity, causal stage, or sorter-only causality.
