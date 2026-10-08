# Failed packet execution

The frozen execution did not reach its scientific acceptance checks. DARTsort returned an absolute output path while the wrapper retained a relative attempt root; the containment check therefore rejected the valid fresh output and quarantined the otherwise successful restart as `restart.failed`.

This is a wrapper path-normalization defect, not evidence for or against event completeness. The entire failed execution is retained. The repair normalizes the attempt root before containment comparison and will run under a new `v2` packet; this packet will not be overwritten or promoted.

Implementation checks
- Done: executed traceback and retained attempt receipts -> failure occurred at `Path.relative_to`, after DARTsort returned an absolute path.
- Not done: restart/baseline equality -> execution stopped before the baseline and final accounting.
- Can establish: v1 is an orchestration-path handling failure.
- Cannot establish: whether the whole-stage restart fallback meets the event-completeness gate.
