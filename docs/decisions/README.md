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
| [0013](0013-luke-imec0-has-appreciable-rigid-motion.md) | Luke imec0 has appreciable rigid motion; the 1.28 µm sidecar is withdrawn |
| [0014](0014-injected-truth-scoring-is-per-cluster.md) | Injected-truth recovery is scored per output cluster, not against the pooled spike river |
| [0015](0015-corrected-cross-sort-audits-do-not-establish-equivalence.md) | Corrected cross-sort audits show no confirmed detection difference, not equivalence |

The active work plan derived from these records is
[`../pipeline_improvement_plan.md`](../pipeline_improvement_plan.md).
Its current opening governs development priority: reliable spike recovery over
time, amplitude-completeness-led real failure cases, bounded experiments and
explicit decisions. The [next diagnostic prescription](../amplitude_completeness_next_step_prescription.md)
implements that direction. These priorities do not amend historical evidence or
weaken frozen production-promotion rules; dated next-step lists in older records
must be read in the context of the current plan.

Each record states the evidence commit or artifact it rests on. When a record is
superseded, add a new record and mark the old one superseded rather than editing
its conclusion — the point is to keep the reasoning trail intact.

The correction chain is itself part of the record. 0008's population-level
completeness claim was challenged by 0009, but 0009's empirical matched-unit
result was then retracted by 0011 because the matcher was not exclusive. The
corrected Phase A/A2 interpretation is recorded in 0015. Nothing is edited out
of the older decisions; read the newer record before citing their empirical
claims.
