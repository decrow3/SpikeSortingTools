# Frozen post-run invocation

This is a preparation artifact, not an executable request. Do not invoke it until the producer publishes authoritative final receipts and the reporting-compatibility conflict is resolved before any outcome inspection.

Mechanical completion order:

1. Verify the enabled producer packet and H1 enablement review, then verify the pair `COMPLETE`, terminal attempt receipt, enabled contract, and both artifact inventories at the exact paths frozen in `FROZEN_REQUEST_TEMPLATE.json`.
2. Fill only the five named SHA-256 values under `future_final_receipt_hash_fill`. Do not change paths, arm order, source hashes, cohort, matcher settings, bootstrap seed/count, primary threshold, output namespace, or interpretation language.
3. Recompute host-local binding structures for the four already-fixed paths without inspecting scores. REF must equal the successful Phase1 REF binding before any candidate arm is opened.
4. A managed launcher may generate the one-time start token and its HMAC identity binding. These are lifecycle controls, not outcome-selected fields.
5. Stop before invocation unless a separately reviewed reporting-only correction makes REF repeat diagnostic-only with threshold/pass-fail `NOT_DEFINED`, or the coordinator explicitly resolves the visible contract conflict before outcomes are opened.
6. If invoked after those gates, run exactly once, with no automatic retry, into the frozen absent output namespace. Preserve `STARTED.json` and any failure evidence; write `COMPLETE.json` last only on full success.

Reporting rules are frozen in `INTERPRETATION_RULES.json`. Missing waveform or truncation-QC evidence is `UNMEASURED`. R, DeltaR, and the REF repeat are engineering continuity proxies; no identity, purity, recovery, or generalization claim is permitted.
