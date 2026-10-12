# Focused independent rereview: joint-model K10 evaluator v4

Verdict: **GO_SYNTHETIC_DECISION_PLUMBING**.

V4 repairs the sole v3 release blocker exactly as requested. Before reading any
holdout identity or endpoint value, `selection_aware_arm_report` now requires
`selected_holdout` to be a mapping with exactly `candidate_arm`, `baseline_arm`,
and `endpoints` (`EVALUATOR.py:536-540`). The frozen v3 failing fixture passes
against v4, all 44 previously passing v3 adversarial fixtures remain green, and
the additional exact-key cases pass.

This GO is limited to reviewed synthetic global-credit and selection-aware
decision plumbing. It does not establish K9 validity, arm benefit, equivalence,
superiority, biological identity/purity, CI validity, power, runtime scale, or
readiness for a medium/full run.

## Evidence

- Exact repair commit: `ebcb8f44c58f6f5f556cd3a420472714fddf5bfe`.
- Frozen packet commit: `a911116`.
- V3 NO_GO review commit/COMPLETE remain preserved:
  `3cb5a27fae8d737c39c20adc392e9bbab54f764f` /
  `2b7cbb6eceb52b6d67e2b47cc473e83b995cddc07642517debeaeaf0bf27d754`.
- V3-to-v4 source delta is only the exact five-line top-level holdout guard;
  the contract is byte-identical. The author-test delta is only the preserved
  regression.
- Author suite: `35 passed in 0.15s`.
- All exact preserved v3 adversarial callables rebound to frozen v4:
  `45 passed in 0.19s`.
- Focused guard cases: `6 passed in 0.08s`, covering the exact valid key set,
  non-mappings, missing keys, arbitrary extras, and an explicit non-selected
  candidate summary.
- V4 and preserved v3 NO_GO local/shared manifest and completion chains verify.

The preserved adversarial file itself remains byte-identical and names v3, as
required for historical preservation. `PRIOR_ADVERSARIAL_REPLAY_V4.py` changes
only its module-level evaluator binding, exposing all original test callables
unchanged to pytest.

No K9 or real arm output, recording, voltage, DARTsort, H5, GPU, service, sort,
RF, or held-out dataset was accessed.

## Implementation checks

- Done: actual delta -> full contract/source/tests and commits read; v3→v4 is
  confined to the requested key-set guard and author regression
  (`EVALUATOR.py:536-540`; `TESTS.py`).
- Done: exact guard -> valid three-key mapping is accepted; missing/extra keys
  and non-mappings fail with the frozen single-summary error before holdout
  values are read (`GUARD_TESTS.py`).
- Done: preserved blocker -> the exact v3 failing callable now passes against
  frozen v4; all other 44 preserved callables also pass
  (`PRIOR_ADVERSARIAL_REPLAY_V4.py`).
- Done: provenance -> frozen packet matches executable source/tests and commit;
  v4 plus v3 NO_GO local/shared hash chains verified.
- Not done: K9, real outputs, recordings, voltage, DARTsort/H5 integration, CI
  construction, power, scale, GPU, services, sorts, RF, and held-out data ->
  excluded by assignment and contract.
- Can establish: the final known synthetic holdout-boundary blocker is repaired
  without changing prior matching, accounting, endpoint, or decision behavior.
- Cannot establish: benchmark validity, arm benefit, equivalence, superiority,
  biological identity/purity, statistical power, or run readiness.

