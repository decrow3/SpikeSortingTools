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
| [0009](0009-cross-sort-comparisons-must-be-unit-matched.md) | Cross-sort quality comparisons must be unit-matched (corrects 0008) |
| [0010](0010-rescue-yield-is-relabelling-not-detection.md) | The rescue yield difference is relabelling, not detection |
| [0011](0011-cross-sort-event-matching-and-detection-evidence.md) | Exclusive event identity and null-controlled detection evidence (retracts 0009/0010 empirical results) |
| [0012](0012-c2-uses-compact-donor-cohort.md) | C2 uses compact D2b-2 donors; pilot plateau donors are forbidden |

The active work plan derived from these records is
[`../pipeline_improvement_plan.md`](../pipeline_improvement_plan.md).

Each record states the evidence commit or artifact it rests on. When a record is
superseded, add a new record and mark the old one superseded rather than editing
its conclusion — the point is to keep the reasoning trail intact.

[0008](0008-amplitude-completeness-gates-promotion.md) and
[0009](0009-cross-sort-comparisons-must-be-unit-matched.md) are the worked
example. 0008 narrowed [0001](0001-ks4-unwarped-is-the-production-sorter.md);
0009 then retracted 0008's own central claim after a unit-matched re-analysis.
Nothing was edited away — each record keeps its measurements, and the
corrections are additive. Read 0008 and 0009 together.
