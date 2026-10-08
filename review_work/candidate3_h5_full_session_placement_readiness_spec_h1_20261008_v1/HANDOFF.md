# Candidate-3 H5 full-session placement/readiness handoff

Verdict: `H5_PREFERRED_CHECKLIST_FROZEN_NOT_CURRENTLY_READY`.

H5 remains the preferred execution host because the accepted 241,309,358,592-byte Arm-A binary is already local there. This packet turns the accepted host/storage feasibility result into an executable prestart checklist; it does not claim the host, wrapper, environment or run contract is ready today.

The placement is intentionally simple: read the accepted input in place; put output, scratch, caches, failure retention, logs and receipts beneath one fresh persistent H5-local run root on one named writable filesystem; configure DARTsort to avoid both second preprocessing and recording copy; publish only compact evidence. Full float32 materialization is prohibited.

The conservative placement gate requires at least 602,618,717,184 free bytes before attempt creation. The managed run may allocate at most 482,618,717,184 bytes across all run-root roles, including one quarantined failed whole-stage attempt, while retaining 120,000,000,000 free bytes. These values are placement safeguards, not expected consumption. A later evidence-based smaller envelope requires a separately frozen review; it cannot be improvised at prestart.

The wrapper and full config remain hard blockers. Kilosort's FFT high-pass length follows each request, so an H5 lazy wrapper must freeze request size, padding, crop and transition context and pass predeclared real ordinary/transition equality. DARTsort must consume `preprocessing=none`, `copy_recording_to_tmpdir=no`, and `work_in_tmpdir=false`. The current stock partial-peel checkpoint remains untrusted; an incomplete peel stage is quarantined and recomputed from unchanged inputs in a new namespace.

The exact twelve fresh checks are in `READINESS_SPEC.json`. They cover current input identity, wrapper equality, actual imports, complete config/lineage, path placement, free space, memory/oomd, GPU conflicts, effective manager controls, duplicate-run prevention, whole-stage restart behavior and durable terminal publication. Any failure blocks launch without stopping unrelated work.

## Implementation checks

- Done: input/source/config transcription -> exact accepted metadata, source hashes and supported no-copy settings are bound to the accepted feasibility and active-path evidence.
- Done: placement arithmetic -> `482,618,717,184 + 120,000,000,000 = 602,618,717,184` bytes; materialization remains prohibited.
- Done: checkpoint semantics -> only fresh whole-stage restart is permitted; stock partial resume is not represented as safe.
- Done: manager/resource lessons -> the H1 unenforced CPU-quota failure and H5 candidate-2 oomd caveat are converted into effective-cgroup and parent-pressure gates.
- Not done: H5 checks -> excluded from this task; H5 was not contacted and no payload, GPU, service or filesystem was touched.
- Can establish: the exact minimum handoff checklist and placement policy required before a separate H5 run contract can be released.
- Cannot establish: current H5 readiness, input payload identity today, wrapper equality, safe memory requirement, full-session completion, scientific performance or launch authorization.

