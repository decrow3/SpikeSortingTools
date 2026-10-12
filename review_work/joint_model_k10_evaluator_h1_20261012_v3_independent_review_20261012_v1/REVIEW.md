# Independent implementation-first review: joint-model K10 evaluator v3

Verdict: **NO_GO_PENDING_EXACTLY_ONE_HOLDOUT_SUMMARY_GUARD**.

Do not use the v3 selection-aware decision report on real arm summaries yet.
The global event-credit implementation and all other reviewed synthetic
decision arithmetic passed, but the holdout input does not enforce the frozen
single-summary boundary. `selection_aware_arm_report` reads the three expected
fields with `.get(...)` but never rejects extra top-level fields
(`EVALUATOR.py:536-542`). A payload containing the selected N1-versus-N0
summary plus an explicit `non_selected_candidate_holdout` N2 summary is
accepted and the second summary is silently ignored. This violates the
contract's “exactly one selected-arm-versus-baseline summary; non-selected
candidate holdout summaries are rejected” requirement.

## Exact repair and rereview condition

Before reading the holdout identity or endpoints, require `selected_holdout` to
be a mapping whose key set is exactly:

```python
{"candidate_arm", "baseline_arm", "endpoints"}
```

Otherwise raise `ValueError` stating that exactly one selected-arm-versus-
baseline summary is allowed. Preserve the current failing fixture, add it to
the author suite, freeze a new packet, rerun the full author and independent
suites, verify hashes, and obtain focused independent rereview. No real output,
matching redesign, DARTsort, RF, or holdout access is needed.

## Independent evidence

- Frozen contract commit: `b56c4a1b22b99c26f3b2085f6d43a787bea46b6b`.
- Implementation commit: `1648a9b1b84bd86d78e2a52735e039402e0bbe17`.
- Frozen evidence commit: `af3968a4477e144be7b840f87d343bfae101d3aa`.
- Frozen packet source/tests are byte-identical to the executable files at the
  reviewed commit; local and shared manifest/completion chains verify.
- Author suite: `34 passed in 0.14s`.
- Initial independent run: `43 passed, 2 failed in 0.23s`. One failure was a
  review-fixture null shift wrapping onto one compact synthetic event, not an
  evaluator defect; the fixture was corrected by moving its frozen synthetic
  null shifts away from its event span.
- Corrected independent run: `44 passed, 1 failed in 0.24s`; the sole failure
  is acceptance of an added non-selected holdout summary.
- Passing subset: `44 passed, 1 deselected in 0.19s`.

Passing adversarial coverage includes an independent exhaustive oracle for
all-truth/all-output and per-unit pooled matching; unique truth/output receipts;
cross-cluster versus cross-unit credit algebra; preserved split, merge,
many-to-one, shifted-null and overlap outputs; development-only deterministic
selection; wrong-arm rejection; exact higher/lower boundaries; fail-dominant
PASS/FAIL/INCONCLUSIVE aggregation; one-unit support with no floor; numeric,
dtype and range rejection; no equivalence/superiority claim; exact clocks; and
create-only output.

No K9 or real arm output, recording, voltage, DARTsort, H5, GPU, service, sort,
RF, or held-out dataset was accessed.

## Implementation checks

- Done: what actually ran -> full frozen contract/source/tests and the three
  cited commits were read; packet source/tests match the executable copies and
  frozen commit.
- Done: matching/counting -> independent brute-force oracle agrees with global
  and per-unit pooled cardinalities across randomized small graphs; every
  global receipt is gate-valid and unique; credited-by-unit/cluster sums equal
  global TP (`EVALUATOR.py:315-358,432-447`; `ADVERSARIAL_TESTS.py`).
- Done: accounting semantics -> known answers separate cross-cluster duplicate
  credit from cross-unit competition, while gross pairwise TP is explicitly
  retained as non-recovery evidence.
- Done: legacy endpoints -> split, merge, many-to-one, shifted nulls and overlap
  endpoints remain present and pass a combined synthetic fixture
  (`EVALUATOR.py:251-475`).
- Done: selection arithmetic -> development-only selection, arm-name tie break,
  wrong-arm rejection, both margin boundaries, decision aggregation, `n_units`
  support/validation, claims, and numeric guards pass (`EVALUATOR.py:478-622`).
- Done: holdout schema -> selected identity and exact endpoint names are checked,
  but extra top-level non-selected summaries are silently accepted; this is the
  release-blocking contract defect (`EVALUATOR.py:536-542`).
- Done: clocks/create-only -> exact int64 clocks, half-open session bounds, and
  overwrite refusal remain green (`EVALUATOR.py:43-62,120-150,633-652`).
- Not done: K9, real outputs, recordings, voltage, DARTsort/H5 integration,
  statistical CI construction, power, runtime scale, GPU, services, sorts, RF,
  and held-out data -> excluded by assignment and contract.
- Can establish: v3 global-credit, legacy endpoint, and bounded decision
  arithmetic pass synthetic independent checks; one precise input-boundary
  defect blocks selection-aware release.
- Cannot establish: benchmark validity, arm benefit, equivalence, superiority,
  biological identity/purity, power, or medium/full-run readiness.

