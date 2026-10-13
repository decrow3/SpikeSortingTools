# Focused review: A2 field-only subwindows v2

Proposed verdict: **GO_FIELD_ONLY_SUBWINDOW_BOUNDS**.

Review commit `a4852f3` and the result in this directory. A4 permits one small
subwindow selector and requires reuse of existing field code. Confirm that the
selector calls `rounded_rigid_field` from the shared frozen source at SHA-256
`823e788d...`, reproduces rounded-q SHA-256 `3174326c...`, evaluates every
30,000-sample-grid candidate fully contained in each frozen parent, ranks by
switch count then samples at `|q| >= 80 um` then earlier start, and returns 300
chunks. Run the single known-answer test.

Check that v1 and `FAILURE_1.json` remain preserved and that v2 did not inspect
arm, injection, sorter, RF or holdout outcomes. Do not expand review into A4's
accepted dependencies or add tests.

Task statement:

- Existing code called: frozen `rounded_rigid_field`, AM.3 field, and C4 chunk grid.
- Smallest missing capability: deterministic selection of one 300-chunk interval per parent window.
- Scientific decision enabled: which chunks may enter P4 preparation; no arm effect is decided.

Implementation checks
- Done: hashes, exact session clock, parent containment, full candidate enumeration, ranking semantics and one known-answer test are available to inspect.
- Not done: independent review, retained-bank donor selection, reader injection, adapter, matching or scoring.
- Can establish: deterministic field-only high/medium subwindow bounds.
- Cannot establish: P1 equivalence, donor validity, P4 readiness, runtime or scientific benefit.
