# EN first-medium Phase3-bound activation derivative v2

Status: `COMPLETE_EXECUTION_DISABLED_PENDING_H1_ACTIVATION_BINDING_REVIEW`.

This immutable derivative binds the exact Phase3 v3 producer/consumer repair
candidate and its independent GO review. The proposed active pair launcher is
the accepted inventory-binding delta `9e7290c1...`, whose base is the previously
reviewed launcher-v3 `238c9c99...`. One-start, cap, reserve, finalization, arm
order, clocks, scientific sources/settings, and RF/holdout seal remain frozen.

The packet is deliberately not activatable. `execution_enabled` is false, the
enabled environment file is absent, and `ExecStartPre --require-activation`
rejects it. Only the proposed-not-installed unit and a non-installable review
template are included.

All proposed runtime outputs now use the single fresh, absent
`en_first_medium_trained_pair_20261002_v2` namespace. This derivative corrects
four stale v1 paths retained by the earlier activation candidate: REF snapshot,
repaired native results, repaired launch evidence, and repaired snapshot.

Verification produced 22 passing tests, a successful launcher validate-only
receipt, and a read-only activation inspection with exact Phase3 bindings,
fresh namespaces, and `activation_eligible=false`. No service was installed or
started; no recording, voltage, outcome, sort, training, RF, or holdout access
occurred.

