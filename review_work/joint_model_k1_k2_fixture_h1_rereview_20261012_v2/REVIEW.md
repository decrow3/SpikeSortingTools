# Focused independent rereview: joint-model K1/K2 fixture v2

Verdict: **GO_SYNTHETIC_FIXTURE_QUALIFIED**.

The v1 `NO_GO` remains preserved and reproducible: its `_integer_1d` accepts
`100+0j`, warns, and silently casts it to acquisition sample 100.  V2 repairs
that exact defect by admitting only non-boolean integer arrays or real floating
arrays containing finite, integral values inside the int64 domain.

The implementation diff is scoped to `_integer_1d`; K1, the detector and the
K2 statistic are otherwise AST-identical to v1.  On valid inputs, v1 and v2
produce identical detector arrays and identical complete K2 result mappings.

## Evidence

- V1 author packet, v1 independent `NO_GO` packet, and v2 author packet all
  verify member-by-member against their local manifests, `COMPLETE.json`, and
  shared-storage copies.
- V2 exact reader and working sources match the frozen packet.
- Author suite: 21 passed.
- Prior independent suite rebound to v2: 12 passed.  Its only source change
  from the preserved v1 reviewer suite is the explicit author-packet path.
- Fresh rereview suite: 25 passed.
- Fresh cases cover complex values with zero/nonzero imaginary parts, boolean,
  object, Unicode, bytes, `uint64` overflow, both floating int64-domain edges,
  fractional and nonfinite floats, valid int64 extrema, valid `uint64`, valid
  integral floats, all three public K2 companion positions, the preserved v1
  failure, scoped-source invariance, and valid-output invariance.

No real K1 arrays, K9 output, recording voltage, H5 state, GPU, service, sort,
RF or holdout data were inspected or used.

Implementation checks
- Done: v1 failure preservation -> complex zero-imaginary clock remains accepted only by preserved v1 and is rejected by v2.
- Done: repair scope -> AST comparison shows the implementation changes only `_integer_1d`; commit `d73a9ff` contains that repair and four author regressions only.
- Done: dtype/domain behavior -> complex, bool, object/string/bytes, fractional/nonfinite float and out-of-int64 values fail closed; valid boundary values return int64.
- Done: unchanged K1/K2 behavior -> K1 source and detector are unchanged; complete K2 mappings on valid inputs are identical between v1 and v2.
- Done: frozen suites and provenance -> 21 author, 12 rebound prior-review and 25 fresh rereview tests pass in the exact worktree environment; all three local/shared evidence chains match.
- Not done: six real C4-0 windows, 0804 pulse rate, real lag, sorter integration, performance/scale, GPU behavior or any scientific comparison; excluded by review scope.
- Can establish: the v1 K2 clock-validation defect is repaired without changing K1 or valid-input K2 statistics; K1/K2 v2 is synthetic-fixture-qualified.
- Cannot establish: K1 real-window bitwise equality, a real frozen lag, biological validity, sorting benefit, benchmark validity or medium-run readiness.

