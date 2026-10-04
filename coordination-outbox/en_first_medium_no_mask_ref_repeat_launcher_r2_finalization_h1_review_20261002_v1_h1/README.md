# H1 independent R2 finalization delta review

Verdict: `GO_R2_FINALIZATION_REPAIR_ONLY_EXECUTION_REMAINS_DISABLED`

The v3 repair closes the v2 terminal-receipt accounting escape. The accepted
pair-wide cap remains 17,179,869,184 bytes. Ordinary writers are restricted to
17,178,820,608 bytes, leaving a derived 1 MiB bounded terminal/control reserve.
The outer `run_single_attempt` finalizer stages bounded receipts, performs a
read-only aggregate audit, and publishes authoritative `COMPLETE.json` by a
same-directory no-growth rename with no later counted write.

The v2 counterexample reproduces. The v3 outer path independently sums below
and exactly at the cap, while an above-cap late write preserves failure and
cannot publish authoritative success.

This GO accepts only the R2 bookkeeping delta. The packet remains execution-
disabled with the enabled binding absent. No service, voltage, or science ran.
