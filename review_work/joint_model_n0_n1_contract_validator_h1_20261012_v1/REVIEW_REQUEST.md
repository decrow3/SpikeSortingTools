# Focused independent review: N0/N1 run-contract validator v1

Proposed verdict: **GO_METADATA_VALIDATOR_ONLY**.

Review the frozen contract, implementation, and tests. Independently verify the
validator fails closed on malformed/unresolved identities, source drift,
invalid window/padding/state bounds, cross-arm input/checkpoint/config drift,
unlisted arm deltas, missing practical endpoint margins, equivalence claims,
unbounded resources, non-managed execution, and unsafe output lifecycle.

Check exact scalar dtypes, overflow boundaries, unknown/duplicate fields and
adversarial mapping/sequence inputs. Check whether the declared fields actually
bind the required clock, geometry, cohort, evaluator, source closure and only
intended N0/N1 delta. A structurally valid result must remain metadata-only and
must not release execution or claim referenced bytes exist.

No reviewer production action. Do not open referenced recordings or voltage,
allocate GPU, launch services/sorts, inspect arm outcomes, access RF/holdout, or
edit C4. Publish a distinct immutable review.

Implementation checks
- Done: metadata schema, cross-arm invariants, bounded resources/output lifecycle and 20 author fixtures.
- Not done: referenced-byte existence/hash checks, live resource preflight or scientific execution.
- Can establish: whether the frozen validator is a fail-closed structural gate.
- Cannot establish: input correctness, implementation behavior beyond metadata, resource readiness, or N0/N1 outcome.
