# CL: portable CJ-v2 force gate

`testing/cl_force_gate.py` packages the frozen CJ-v2 computation without W2 or
CE path, origin, mask-hash or unit-count assumptions. Callers provide one
same-run post-TMM state, route packet, exact interval metadata, canonical mask
and its preregistered hash. SI must be declared explicitly as `absent` or
provided as an array; it is unioned unchanged after gated force connectivity,
like QDA.

The statistic, thresholds, seed rule and draws are unchanged from commit
`48868fc`: central annulus 9--29 samples, shoulders 45--89, 100 events/unit, 20
common complete 5-s blocks, expected central count 20, 1,000 bootstrap draws,
1,000 same-segment nonwrapping derangements, bootstrap upper <=0.50 and observed
ratio <= null q05. Unknown support stays unresolved. Direct edges are gated
before connectivity. No consumer aggregation/deduplication is implemented here.

The W2 invocation must declare SI `absent`. A focused SI fixture confirms that a
provided SI edge survives the force gate even when no force edge passes. The
portable W2 equivalence receipt and finite-exposure audit are published with the
shared package. This helper is the computation dependency for W3; it does not
authorize capture, fitting, sorting or threshold changes.

Routine downstream work described by the accepted CL scope proceeds under the
existing broad authority; it does not require another user approval. New
scientific parameter changes or expanded resource envelopes still require an
explicit scoped instruction.
