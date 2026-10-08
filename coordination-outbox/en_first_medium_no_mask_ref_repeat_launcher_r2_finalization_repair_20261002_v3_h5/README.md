# No-mask REF-repeat launcher R2 finalization repair v3

Fresh immutable candidate repairing only the v2 terminal-accounting blocker.
The v2 packet and H1 counterexample remain preserved.

- Pair-wide total remains exactly 17,179,869,184 bytes (16 GiB).
- A 1,048,576-byte terminal/control allowance is reserved inside that total;
  ordinary live output is limited to 17,178,820,608 bytes.
- The reserve derivation includes bounded counted late logs, atomic
  final-plus-partial coexistence, success pending receipts, and failure
  receipts. Every control JSON is serialization-bounded.
- Both arms and ordinary writers stop before the single outer finalizer.
- Terminal receipts are first written under counted pending/non-success names.
- The final aggregate audit is read-only. Authoritative `COMPLETE.json` is
  published last by same-directory rename of the already-counted pending file,
  with no later counted write.
- The monitor is process-boundary protection, not a strict instantaneous
  filesystem quota: a child can overshoot between polls. No success can be
  published above the final-namespace 16 GiB limit.

Fourteen tests traverse the actual `run_single_attempt` path and independently
sum final counted files for below/equal/above-cap cases. They also cover
finalization failure/no false success, live and second-arm crossings, and all
accepted R1/R3 regressions.

Scientific sources and settings remain unchanged except required v3 launcher
bindings/paths. This packet is execution-disabled. The proposed service was not
installed or started; no recording voltage, sort, training, evaluator, RF, or
holdout work ran. Independent H1 review remains required before any enabled
derivative.

