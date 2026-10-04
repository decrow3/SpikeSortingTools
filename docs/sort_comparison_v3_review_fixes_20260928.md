# Comparison evaluator v3: review corrections

The generic evaluator now uses `long-sort-comparison-v3`. These are code
corrections, not a new rigid or nonrigid scientific result. Existing v2 output
directories and publication packets remain historical evidence and were not
rewritten. Any corrected cached comparison must use a fresh output namespace.

## Spatial eligibility

The complete unit inventory and correspondence graph remain inspectable.
Population refractory and edge-unit guardrails use units whose median depth
lies in the common processing domain. Refractory metrics retain those units'
complete spike trains, avoiding artificial gaps from event-level filtering.
Coincidence and edge-spike guardrails use only in-domain spikes belonging to
those units. Missing depths fail closed when a spatial contract is supplied;
an empty supported population reports unavailable guardrails rather than zero.
The effective policy is recorded in the comparison manifest.

## Fit trust

The comparator and operational completeness timeline now share
`classify_fit_trust`. Paired support requires both the legacy structural status
`finite_interior` and the sidecar status `measured`. Nonfinite, boundary-censored,
and disagreeing fits stay in the saved window table but cannot provide support.
The default disagreement boundary remains 5 percentage points, and the resolved
boundary is recorded in the manifest. Missing fit parameters fail closed.

The pre-fix review of the saved rigid window table found that applying this
trust rule reduced 23 supported pairs to 11. This is a review calculation,
not a replacement full comparison report.

## Input identity

`spike_positions.npy` is read once; those exact bytes are hashed and parsed.
Depth changes now change the comparison input fingerprint. Loaded positions
are detached from subsequent file modifications. Nonfinite depths are rejected.

## Validation

45 focused and related tests passed under the existing DARTsort Python test
runtime, including completeness timeline, operational-stage verification,
development runner, generic comparison and handoff tests. The locked production
environment was not modified to install pytest. Regression tests cover spatial
cohort invariance, depth excursions, unavailable spatial support, rejected fits,
fit-policy boundaries, depth identity changes, and refusal to overwrite prior
comparison outputs. `git diff --check` passed.
