# Focused independent rereview: joint-model K10 evaluator v2

Verdict: **GO_FIXTURE_QUALIFIED_SAVED_OUTPUT_EVALUATOR**, with an explicit
author-packet replay-provenance correction.

The v1 `NO_GO_PENDING_EXACT_INTEGER_SESSION_BOUNDS_GUARD` remains immutable.
V2 makes only the required source change: it validates exactly two session
bounds with `_integer_samples` before converting them to Python integers
(`EVALUATOR.py:106-111`). The contract is unchanged, and the only author-test
delta is the frozen fractional-bounds regression. The repaired evaluator passes
the full author suite, every preserved v1 adversarial test when independently
bound to frozen v2, and seven additional clock-boundary cases.

This fixture-qualified GO authorizes only use of the reviewed saved-output
evaluator construction. It does not validate K9, any real arm output, benchmark
design, biological identity/purity, runtime scale, or a medium/full run.

## Author-packet replay-provenance correction

The v2 `FIXTURE_RESULTS.json` says the prior adversarial suite passed 11/11
against the repaired snapshot. That receipt is not reproducible from the frozen
packet as written: `PRIOR_ADVERSARIAL_TESTS.py:17` still hard-codes the v1
packet and is byte-identical to the immutable v1 review test. Direct execution
therefore targets v1 and reproduces the preserved failure (`10 passed, 1
failed`). This does not negate the evaluator repair, because the review-owned
`PRIOR_ADVERSARIAL_REPLAY_V2.py` changes only that module binding and exposes
the exact preserved test callables; all 11 pass against hash-bound v2.

Treat this rereview packet, not the author packet's prior-replay receipt, as the
authoritative evidence for adversarial replay against v2. Preserve the flawed
receipt and test unchanged.

## Evidence

- Reviewed source commit: `0132bb1845e93cb68cc7a53b390bfb17f4da8512`.
- V1-to-v2 source delta: exact-bounds validation only; contract byte-identical.
- V1-to-v2 author-test delta: fractional-bounds regression only.
- Author suite: `20 passed in 0.12s`.
- Frozen bundled prior file, executed directly: `10 passed, 1 failed in 0.20s`
  because it targets v1, preserving the original NO_GO.
- Exact prior test callables rebound to frozen v2: `11 passed in 0.17s`.
- New clock-boundary suite: `7 passed in 0.08s`.
- New cases cover fractional/nonfinite bounds, exact integral-float
  normalization, lower-inclusive/upper-exclusive semantics, and circular null
  support with a nonzero session origin.
- V2 author packet, v1 author packet, and v1 NO_GO review packet each have valid
  local manifests/completion chains and byte-identical shared copies.

No real joint-model arm output, K9 array, recording, voltage, DARTsort import or
edit, H5 contact, GPU, service, sort, RF, or holdout was accessed.

## Implementation checks

- Done: what changed -> frozen v2 differs from v1 only at the exact-integer
  session-bound guard; its contract is unchanged and author regression is
  scoped to the rejected case (`EVALUATOR.py:106-111`; `TESTS.py:202`).
- Done: clock semantics -> fractional and nonfinite bounds fail closed; exact
  integral bounds normalize to integers; lower bound remains inclusive, upper
  bound exclusive; circular null wrapping respects a nonzero exact origin
  (`BOUNDS_ADVERSARIAL_TESTS.py`).
- Done: prior load-bearing checks -> exact preserved v1 test functions pass
  11/11 after binding only their evaluator global to frozen v2
  (`PRIOR_ADVERSARIAL_REPLAY_V2.py`).
- Done: provenance -> direct bundled replay was checked and found to target v1;
  its contradictory author receipt is corrected here rather than silently
  accepted or overwritten.
- Done: hash/provenance -> v1 NO_GO, v1 author packet, and v2 author packet
  local/shared hashes and manifest/completion chains verified.
- Not done: K9 arrays, real arm outputs, recordings, voltage, DARTsort
  integration, runtime scale, H5, GPU, services, sorts, RF, and holdout ->
  excluded by assignment and contract.
- Can establish: the precise v1 bounds defect is repaired without a matcher,
  endpoint, or contract change; reviewed synthetic evaluator plumbing is fit
  for fixture-qualified saved-output use.
- Cannot establish: benchmark validity, arm benefit, biological identity or
  purity, host/runtime readiness, or authorization for a medium/full arm run.

