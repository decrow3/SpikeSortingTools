# Frozen scope

- Milestone: candidate-3 full-session host/storage readiness.
- Decision: determine the minimum safe execution topology on H1 versus H5 without a full float32 materialization.
- Cheapest adequate test: exact input/source/config binding, existing H5 resource receipts, and current H1 mount/free-space metadata only.
- Completion: two concrete host options with feasibility status, blockers, resource arithmetic, placement and invalidation conditions, followed by local independent review.
- Exclusions: no transfer, voltage-payload read, throughput benchmark, materialization, sort, service launch, RF/holdout access, or new H5 dependency.
