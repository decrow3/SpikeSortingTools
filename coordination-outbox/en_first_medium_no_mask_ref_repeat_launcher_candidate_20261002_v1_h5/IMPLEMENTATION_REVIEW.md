# H5 implementation review

Verdict: **READY FOR INDEPENDENT H1 REVIEW; EXECUTION DISABLED**.

The launcher uses two deliberately distinct paths. `REF384_repeat` calls the
frozen historical REF384 runner with no bad channels and no support hook, then
augments only its evidence receipt and applies the clock-v5 terminal validator.
`repaired_B384` calls the support-aware clock-v5 trained entry. The orchestrator
accepts only execution state and a hashed COMPLETE file between arms; it does
not read counts, scores, evaluator outputs, or other scientific outcomes.

The pair contract, both arm configs, and the proposed service remain disabled.
The service file is review evidence only and was not installed or started.

## Implementation checks

- Done: what would run -> traced the exact wrapper, historical REF runner, clock-v5 trained entry, Kilosort source snapshots, three configs, and proposed systemd unit; all are hash-bound by the disabled contract.
- Done: intended arm distinction -> REF requires `support_policy_enabled=false` and exact historical REF384 input; B uses the support-aware trained entry with independently recomputed adaptive state.
- Done: clocks/ancestry -> both configs bind crop `[208498882,226498783)` and exact `spike_times == full_st[kept,0] + crop.imin`; the terminal validator checks half-open bounds, ordering, cluster ancestry, and amplitude ancestry.
- Done: sequencing/counting semantics -> fixed REF-then-B order; only completion state is inspected between arms; scientific results cannot change whether B is attempted.
- Done: failure and retry semantics -> existing namespaces fail closed; REF failure prevents B; B failure preserves REF state; no pair COMPLETE on failure; Restart=no and one-start limit are frozen.
- Done: gates/resources -> 2,700 s, 64 GiB, 16 CPU threads, one GPU, 16 GiB persistent output, 200 GiB free-space floor, RF/holdout seal, and launch-time GPU recheck are explicit.
- Done: known-answer tests -> 9/9 passed; validate-only opened no recording or voltage and created no namespace.
- Not done: independent H1 review, enabled derivative, service installation, detached dummy survival proof for this exact service, or scientific execution.
- Can establish: a concrete, fail-closed heterogeneous pair launcher now exists and preserves the no-mask REF versus support-aware B distinction under an execution-disabled contract.
- Cannot establish: launch readiness, runtime success, repeatability, efficacy, waveform preservation, biological identity/purity, or evaluator advancement.
