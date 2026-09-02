# Decision records

Short, durable records of the choices the production pipeline embodies: what was
adopted, the evidence, what was rejected, and what would justify reopening.

These are the layer that must survive extraction into a production repository.
They are deliberately thin. The full investigative record — hypotheses, failed
approaches, intermediate results, figures — stays in `docs/` and `testing/` in
this research repository and is not duplicated here.

| # | Decision |
|---|---|
| [0001](0001-ks4-unwarped-is-the-production-sorter.md) | KS4 on the unwarped frozen graph is the production sorter |
| [0002](0002-motion-is-estimated-never-applied.md) | Motion is estimated and recorded, never applied to voltage |
| [0003](0003-saturation-blanking-and-artifact-sidecar.md) | Bilateral 500 µV blanking with a raw artifact sidecar |
| [0004](0004-bad-channel-191-interpolation.md) | Physical channel 191 is interpolated and sorted |
| [0005](0005-dartsort-kiasort-deferred.md) | DARTsort and KIAsort are deferred, not rejected |
| [0006](0006-recovery-axis-is-post-sort-mua-reconciliation.md) | The recovery axis is post-sort MUA family reconciliation |
| [0007](0007-stage-local-validation.md) | Stage-local validation governs advancement |
| [0008](0008-amplitude-completeness-gates-promotion.md) | Amplitude completeness gates promotion; yield alone never does |

Each record states the evidence commit or artifact it rests on. When a record is
superseded, add a new record and mark the old one superseded rather than editing
its conclusion — the point is to keep the reasoning trail intact.

[0008](0008-amplitude-completeness-gates-promotion.md) is the worked example:
it narrows the interpretation of [0001](0001-ks4-unwarped-is-the-production-sorter.md)
and extends [0007](0007-stage-local-validation.md), and 0001's measurements were
left exactly as recorded.
