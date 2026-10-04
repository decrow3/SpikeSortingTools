# Workstream-B measurement redesign targeted repair

Status: `READY_FOR_H5_REVIEW_EXECUTION_DISABLED`.

This fresh packet repairs only H5 findings H5-B1 and H5-B2. It preserves the original implementation packet and H5 review by hash and includes their manifest/COMPLETE receipts under `bindings/`.

The repaired measurement module now rejects non-finite or non-positive duration, validates every included amplitude interval against `[0,duration_s]`, rejects malformed and within-unit overlapping intervals, and constructs all support fractions through an explicit `[0,1]` invariant. When diagnostic cell bounds are supplied, both complete event streams are checked against the half-open domain before correspondence, so out-of-domain candidate or non-cohort REF events cannot alter reciprocal primaries.

The accepted matcher, R/DeltaR definitions, paired bootstrap, strict margin gate, execution-disabled contract, and previous diagnostics are byte-identical or semantically unchanged. Thirty packet tests and the 56-test compatibility set pass. No real scoring or job ran.

`IMPLEMENTATION_REVIEW.json` contains the exact hashes, scoped claims, and implementation-check block. Deltas against the preserved original packet are under `delta/`.
