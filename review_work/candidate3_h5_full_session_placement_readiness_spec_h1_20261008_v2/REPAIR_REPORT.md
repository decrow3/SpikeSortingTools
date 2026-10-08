# Candidate-3 H5 placement/readiness v2 repair

Verdict: `REPAIRED_CHECKLIST_FROZEN_H5_STILL_NOT_CURRENTLY_READY`.

V1 remains immutable and blocked. V2 changes only two gates and one ambiguous stop-condition phrase.

H5-06 now defines the disk metric, enumeration domain and cadence. The manager-external monitor records both allocated bytes (`st_blocks*512`) and apparent bytes (`st_size`), deduplicates hard links and open descriptors by device/inode, explicitly accounts for open-unlinked files, refuses filesystem/symlink escapes, samples at prestart/60-second/stage/promotion/terminal boundaries, and stops the whole managed cgroup on either cap, reserve, path or materialization breach. Any additional full-recording copy/materialization is a breach; the earlier ambiguous word “second” is removed.

H5-11 now distinguishes manager terminal receipts from state correctness. Before a real launch, the exact installed production path must survive both managed timeout and external control-group-kill fixtures during a nonzero append: retain and refuse the partial, publish no stage completion, restart the whole peel stage in a distinct namespace with unchanged bindings, keep the original immutable, and reproduce an independent baseline exactly. Caught-exception evidence cannot substitute for this fixture.

No H5 check was executed. The accepted input, 482,618,717,184-byte aggregate cap, 120,000,000,000-byte reserve, 602,618,717,184-byte free-space gate, placement roles and all other v1 content remain unchanged.

## Implementation checks

- Done: blocking review mapping -> each defect has one exact JSON-pointer replacement; all unrelated v1 fields are inherited unchanged.
- Done: storage semantics -> linked, hard-linked, sparse and open-unlinked files, filesystem escapes, available bytes, cadence and atomic receipts are defined.
- Done: kill-boundary semantics -> timeout and external cgroup kill both require partial preservation/no promotion/no reopen and baseline-equal fresh whole-stage completion.
- Not done: fixture/H5 execution -> expressly excluded and remains a prelaunch hard gate.
- Can establish: the two checklist defects are repaired at specification level without changing the placement decision or capacity envelope.
- Cannot establish: monitor/fixture implementation correctness, current H5 readiness, safe memory demand, wrapper equality, launch authorization, full-session completion or scientific performance.

