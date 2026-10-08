# Implementation review

## Result

The focused R2 repair closes H1's demonstrated successful-return escape. The
last aggregate check is read-only and occurs after all counted pending terminal
receipts exist. Success is then a no-growth rename and no counted write follows.

The 1 MiB reserve is inside, not in addition to, the unchanged 16 GiB cap. Its
code-derived allocation is 262,144 bytes each for counted late service/child
logs, mutable receipt final-plus-partial coexistence, two pending success
receipts, and failure receipt/partial allowance. Actual JSON serialization is
bounded at 131,072 bytes per control receipt and exception text is truncated.

H1's v2 counterexample is preserved verbatim at
`evidence/H1_R2_FINAL_RECEIPT_CAP_COUNTEREXAMPLE.py`. The production live guard
cannot provide a strict instantaneous quota between polling observations; this
is explicitly not claimed. The final namespace is fail-closed for success.

## Implementation checks

- Done: executed-source trace -> `run_single_attempt` alone invokes
  `finalize_success` after `execute_pair` returns with both arms stopped; no
  `ATTEMPT_COMPLETE` write follows.
- Done: cap semantics -> total 17,179,869,184; reserve 1,048,576; live threshold
  17,178,820,608; validation requires their exact relation.
- Done: temp/replacement and failures -> reserve derivation includes the peak
  categories; bounded serializer rejects oversized control receipts.
- Done: final audit -> `aggregate_output_bytes` is read-only; final publication
  is same-directory `os.replace` of `COMPLETE.pending.json` and directory fsync.
- Done: actual outer-path fixtures -> 14 passed; independent file sum covers
  below, exactly equal, and one byte above total; above produces durable failure
  and no authoritative success.
- Done: accepted R1/R3 -> sequential/concurrent single-start and compact voltage
  continuity fixtures remain passing; no scientific source changed.
- Done: validation only -> execution disabled, approval null, enabled binding
  absent, service not installed/started, no voltage/science performed.
- Not done: independent review of this load-bearing delta -> H1 review is the
  next gate and no enabled derivative is authorized.
- Not done: strict instantaneous filesystem quota -> polling/process termination
  cannot prevent arbitrary write bursts; the defensible guarantee is no
  authoritative success above the read-only final aggregate cap.
- Can establish: in the tested final namespace and failure paths, v3 cannot
  publish authoritative success when the final counted total exceeds 16 GiB.
- Cannot establish: actual scientific performance, actual managed-run resource
  behavior, or enabled execution readiness before independent review.

