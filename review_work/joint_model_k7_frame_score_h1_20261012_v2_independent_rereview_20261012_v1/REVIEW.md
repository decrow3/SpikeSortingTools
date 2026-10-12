# Independent K7 frame-score fixture v2 rereview

Verdict: **GO_SYNTHETIC_SCORE_ORACLE_ONLY**.

The v1 `NO_GO` remains immutable at review commit
`2aa5569760036a9a0e9afe0029be537bb61b9ca8` and COMPLETE SHA-256
`3b1a9953e416a97eaaf4869143f430c37ec71c520702faa782e7a67ca046681f`.
Frozen v2 implements the requested repair without changing the contract or
unrelated candidate-union/group logic.

## Findings

- The source delta exactly enforces the frozen 80-µm and 50-event floors,
  separates the fixed 80-µm training rule from the configurable conservative
  held-out cutoff, calculates training separation in Python integers, validates
  every held-out residual row before eligibility filtering, and computes the
  log ratio without intermediate division overflow. Added receipt fields expose
  effective support settings.
- The original immutable v1 adversarial file, rebound independently to frozen
  v2, passes 33/33: the exact six v1 failures pass, and all 27 formerly passing
  cases remain green.
- Six added controls pass for exact-floor acceptance, stricter conservative
  settings, reverse extreme-ratio stability, excluded-row validation, and valid
  excluded-row abstention.
- Author tests pass 34/34; the packet-local author plus adversarial suites pass
  67/67.
- The v2 manifest members and hash chain match locally, in the existing offline
  coordination copy, and in the additional top-level shared copy. The local and
  shared v1 `NO_GO` chains also verify. The coordination copy contains an
  unmanifested `__pycache__`; all immutable manifest members themselves match.

This verdict approves only the frozen synthetic candidate-union, score,
abstention, and member-local group-rule oracle. It does not validate template
learning, detection, whitening, calibration, real classification, or any
production/C4 decision.

## Implementation checks

- Done: executed-source and exact-delta read -> repair commit
  `fc3a83d2e8418d7053e7e2d0a77ae7b92bba9ef2` is confined to the requested
  support, separation, residual-validation, numeric-stability, and receipt
  changes plus focused regressions.
- Done: immutable failure replay -> exact six former failures pass; all 27 prior
  passing cases remain green against implementation SHA-256
  `e3675fc15d54ce2745b3903a13ed2f8637c9558c51a1fdcb19000a4aafabcfe2`.
- Done: author and additional boundary checks -> 34 focused, 67 combined frozen,
  and 6 added controls pass.
- Done: provenance/hash checks -> local and both shared v2 manifest-member
  chains verify at MANIFEST SHA-256
  `ce899b44a437ce6604c11ce9eff913545e5cc6209e912c74ddf59498da776f1d`
  and COMPLETE SHA-256
  `4d2b460d2477c27f76cd1904e43ed066785534be35ed8ccd54f29cc357e2e0da`;
  v1 `NO_GO` remains preserved.
- Not done: template learning, detection, whitening, calibration, production or
  real-data access, DARTsort/H5, GPU/sorts, RF/holdout, or C4 -> explicitly
  excluded.
- Can establish: the five v1 defect causes and exact six failing fixtures are
  repaired while the 27 prior passing synthetic cases remain unchanged.
- Cannot establish: real frame-test validity, calibrated labels, biological
  identity, N3/N4 or N0/N1 readiness, or any production/C4 decision.

