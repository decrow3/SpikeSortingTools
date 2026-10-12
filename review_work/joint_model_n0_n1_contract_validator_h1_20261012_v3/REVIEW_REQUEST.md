# Focused independent rereview: N0/N1 run-contract validator v3

Proposed verdict: **GO_METADATA_VALIDATOR_ONLY**.

V2 repaired all v1 findings but independently admitted NUL-containing paths
and invalid simple systemd service-unit names. V3 changes only those guards,
adds exact regressions plus the 255-byte positive boundary, and transparently
corrects the contract's draft timestamp to its pre-implementation Git commit
time. Both earlier `NO_GO` packets remain immutable.

Verify the four formerly failing v2 controls now pass, all ten prior v2 controls
remain green, semantic v1 replay remains 44/44, all 43 author tests pass, and
the code delta is exactly the two frozen guards. Independently test the service
grammar and byte bound. Verify local/shared hash chains for v1-v3.

No reviewer production action. Do not open referenced recordings or voltage,
allocate GPU, launch services/sorts, inspect arm outcomes, access RF/holdout, or
edit C4. Publish a distinct immutable review.

Implementation checks
- Done: v1 and v2 NO_GO packets preserved; exact two-guard repair and regressions implemented.
- Not done: referenced-byte existence/hash checks, live resource preflight or scientific execution.
- Can establish: whether all known metadata-validator boundary defects are repaired.
- Cannot establish: input correctness, resource readiness, implementation behavior beyond metadata, or N0/N1 outcome.
