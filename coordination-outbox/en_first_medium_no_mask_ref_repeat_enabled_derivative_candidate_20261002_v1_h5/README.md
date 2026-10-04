# EN first-medium enabled derivative candidate v1

Status: `COMPLETE_ENABLED_CANDIDATE_NOT_INSTALLED_PENDING_H1_EXACT_DELTA_REVIEW`.

This immutable packet is the narrow enabled derivative of accepted disabled
packet `5b3b780e...` / `6fbc0064...`. It binds the exact H1 activation review and
the HubPlanner decision to prepare—but not install or start—this derivative.

The contract and both arm configs have `execution_enabled=true`. The packet-
local service and environment pass read-only activation preflight and report
`activation_eligible=true`. They have not been installed, enabled, or started;
the unused v2 runtime namespace and start-claim absence are preserved for H1's
compact-delta review.

The earlier launcher review remains a distinct provenance edge from the later
activation review. The Phase3 machine token is preserved literally as
`GO_PROVENANCE_REPAIR_DELTA`.

Science, two 600-second arms and REF-then-repaired order, clocks, input identity,
one-start/no-retry semantics, no-growth finalizer, RF/holdout seal, and resource
bounds are unchanged. Twenty-three tests pass. No service action, voltage or
outcome read, sort, training, RF, or holdout access occurred.

