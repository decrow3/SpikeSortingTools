# Implementation review

## Result

The derivative is internally activation-eligible but remains an uninstalled
review candidate. H1 must review this exact compact delta before any install or
start.

Implementation checks
- Done: accepted-base identity -> disabled MANIFEST/COMPLETE/contract `5b3b780e...` / `6fbc0064...` / `712ec10c...` verify and are explicitly bound.
- Done: reviews kept distinct -> launcher review `7fde2fc6...` remains in `approval.h1_review_manifest_sha256`; activation review `d815d0ea...` is separately bound in `approval.activation_review_manifest_sha256` with COMPLETE/status `c2b1301b...` / `96e4c4ca...`.
- Done: decision provenance -> packet-local HubPlanner decision `92c37aea...` allows only preparation of this enabled derivative and explicitly forbids install/start before H1 review.
- Done: Phase3 status -> literal token `GO_PROVENANCE_REPAIR_DELTA` is present in contract, environment, tests, and provenance.
- Done: exact delta -> contract and arm flags/status, review/decision bindings, relocated packet/config/service paths, and derived hashes are the only intended changes; scientific config values are equal after removing the permitted flag/status fields.
- Done: fixed experiment -> REF384_repeat then repaired_B384, 600 seconds each, unchanged clocks/input/science, no search, and sealed RF/holdout.
- Done: resources -> one GPU, 16 CPUs, 68,719,476,736 bytes RAM, 2,700 seconds, 17,179,869,184-byte cap, 1,048,576-byte reserve, 214,748,364,800-byte minimum free storage, and Restart=no.
- Done: lifecycle -> unused v2 root/cache/log/attempt namespaces remain absent; one-start/no-retry and finalizer implementations are unchanged.
- Done: read-only validation -> launcher validates enabled; activation preflight verifies every review/decision edge, reports all namespaces fresh and `INSPECTED_ELIGIBLE_NOT_LAUNCHED`; 23 tests pass.
- Not done: H1 review of this exact enabled delta -> required before installation or start.
- Not done: service install/enable/start or scientific execution -> prohibited for this task and not performed.
- Can establish: a compact, provenance-complete, internally eligible enabled derivative exists for independent review.
- Cannot establish: authority to install/start, real execution success, scientific efficacy, or advancement.

