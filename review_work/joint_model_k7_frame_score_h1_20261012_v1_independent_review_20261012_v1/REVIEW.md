# Independent K7 frame-score fixture review

Verdict: **NO_GO_PENDING_SCORE_SUPPORT_AND_NUMERIC_GUARDS**.

The nominal oracle behavior is correct under the author fixtures, and 27
independent adversarial cases pass. Six independent fixtures fail across five
closely bounded defects, so this frozen v1 packet should not yet be used as the
load-bearing score oracle for later template-learning integration.

## Findings

1. **Frozen support minima can be weakened.** The contract fixes held-out
   displacement support at at least 80 µm and state contribution at at least 50
   events. `informative_abs_state_um=40` and `minimum_events_per_state=1` are
   accepted and can produce a reported classification from weaker support.
2. **Training and held-out gates are coupled.** Raising the held-out cutoff to
   160 µm also raises the training-separation requirement to 160 µm. Contractual
   training states separated by exactly 80 µm then abstain even when a valid
   160-µm held-out state contributes 50 events.
3. **Training separation can overflow.** `np.ptp` in int64 arithmetic wraps for
   valid, exact-lattice states near opposite int64 boundaries. The oracle
   reports inadequate training support instead of their large positive
   separation.
4. **The residual statistic is numerically unstable at its admitted domain
   boundary.** Both residuals can be finite and strictly positive while
   `probe/tissue` overflows. Direct division before `log` returns an infinite
   score; separate logarithms return the finite mathematical result.
5. **Residual validation is conditional on contribution.** A noninformative
   held-out state is skipped before its residual fields and values are checked,
   allowing missing, nonfinite, or nonpositive residuals to pass silently even
   though the contract requires residuals to be finite and strictly positive.

Stable candidate union, duplicate-key rejection, residual direction, inclusive
classification boundaries, median-of-state-medians balance, default eligibility
and count boundaries, abstention with tissue operational default, member-local
pulse/classification evidence, no automatic merge, and the other tested dtype,
lattice, range, and support checks pass.

## Exact repair

- Keep the 80-µm training separation rule independent of the held-out cutoff.
  Compute separation in Python integers, for example
  `int(train_unique[-1]) - int(train_unique[0])`, rather than with int64
  `np.ptp`.
- Reject `informative_abs_state_um < 80` and
  `minimum_events_per_state < 50` (or remove their configurability from this
  frozen oracle); higher held-out/count requirements may remain conservative.
- Validate the residual fields, scalar dtypes, finiteness, and strict positivity
  for every held-out residual row before deciding whether its state contributes.
- Compute each event statistic as `log(probe) - log(tissue)` and require the
  derived statistic to be finite.
- Preserve this v1 failure packet and add the six failing fixtures as
  regressions in a new version; rerun all 27 currently passing adversarial cases
  and the 26 author cases unchanged.

## Implementation checks

- Done: full contract/source/test read and parameter trace -> support arguments
  feed both training and held-out gates at implementation lines 93–106; no
  hidden default classification thresholds exist.
- Done: candidate-union, scoring, eligibility, abstention, group, dtype,
  lattice, and numeric known-answer checks -> author suite 26/26; independent
  suite 27 pass and 6 fail.
- Done: source/provenance check -> working implementation and tests match frozen
  packet SHA-256 values; local and authorized shared packets are byte-identical;
  supplied MANIFEST and COMPLETE hashes verify.
- Not done: template learning, detection, whitening, calibration, real units,
  recordings/voltage, DARTsort/H5, GPU/sorts, RF/holdout, or C4 -> explicitly
  excluded and required for claims beyond the fixture oracle.
- Can establish: nominal candidate-union and member-local group behavior work,
  but the frozen score oracle violates its support and residual-domain contract
  at independently reproduced boundaries.
- Cannot establish: real frame-test validity, calibrated labels, biological
  identity, N3/N4 or N0/N1 readiness, or any C4 decision.

