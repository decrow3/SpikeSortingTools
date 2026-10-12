# Independent K9 core v2 rereview

Verdict: **GO_SYNTHETIC_MECHANICS_ONLY**.

The exact v1 `NO_GO` is preserved by review commit
`1aae00ee6d388125f7e99eda326b1778c9d6c437` and COMPLETE SHA-256
`df2621f0e71e3f4eb57311efeedb1871825ce759bfc13175f53d6790a31168da`.
Frozen v2 repairs its three findings without changing the contract or the
underlying synthetic placement/counting mechanics.

## Findings

- The implementation delta from frozen v1 is limited to validation guards:
  geometry is inspected before conversion and must have a real numeric,
  non-boolean dtype; derived `session_sample` is passed through the existing
  exact-integer/int64 scalar guard.
- The v1 and v2 contracts are byte-identical. The preserved adversarial suite
  is byte-identical to the v1 review fixture. The working implementation and
  focused test file are byte-identical to their v2 frozen copies.
- Rebinding only the preserved fixture's implementation module from v1 to v2
  yields 22/22 passing cases: the 19 cases that passed under v1 remain green,
  and positive derived-clock overflow, complex geometry, and numeric-string
  geometry now fail closed as required.
- Nine independent controls also pass. They cover boolean/object/complex/string
  rejection, accepted integer/float geometry, exact upper-int64 derived-clock
  acceptance, positive overflow rejection, and low-boundary preservation.
- Author tests pass: 19 focused and 38 in the requested combined suite.
- Local and authorized shared copies of v2 and the prior v1 review are
  byte-identical; their supplied MANIFEST and COMPLETE hashes match the
  requested values.

This is a narrow approval of deterministic synthetic K9 mechanics and guard
behavior. It is not evidence that a real-data K9 benchmark will be feasible,
valid, or scientifically decisive, and it does not authorize C4 or any real
input access.

## Implementation checks

- Done: executed-source and delta read -> repair commit
  `ab403e6dd9bd10b8e1f799ecf281870d328bef66` changes only the two validation
  paths and adds their focused regressions; contract unchanged.
- Done: matching/counting, state/domain, clocks, coordinates, overlap, decoy,
  support, dtype, and nonmutation fixtures -> preserved suite 22/22 and added
  controls 9/9 pass against frozen implementation SHA-256
  `5c9077148d6371426c1973ae8932914d556067215fa2f78122cf49d2e313e77e`.
- Done: author regression check -> 19 focused and 38 combined tests pass.
- Done: provenance/hash check -> local/shared source packet and prior review
  compare byte-for-byte; v2 MANIFEST SHA-256 `f4f59b1bea804e3957bbffb95bc0971522603ba6b2ac432471e0098d3e463dfb`
  and COMPLETE SHA-256 `d45187e922fe955ef255d92fe0df39f1cc7186e9b0c6a3b610e2e984aade6e15`.
- Not done: real input/window or voltage inspection, DARTsort/H5 access,
  GPU/service/sort/RF/holdout execution, or C4 action -> explicitly excluded.
- Can establish: the exact three v1 guard defects are repaired in frozen v2,
  with the previously passing synthetic mechanics unchanged under the tested
  fixtures.
- Cannot establish: real-data benchmark validity, biological identity or
  purity, sorter performance, C4 readiness, or any downstream scientific
  claim.

