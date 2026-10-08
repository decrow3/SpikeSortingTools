# Independent review: candidate-3 H5 full-session placement/readiness specification

## Verdict

`BLOCK_CHECKLIST_REPAIR_REQUIRED_KILL_BOUNDARY_AND_STORAGE_ACCOUNTING`

Reviewed packet:

- `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_h5_full_session_placement_readiness_spec_h1_20261008_v1`
- manifest SHA-256 `43c670308750c00f9d45e2167267196dd494b74798dd0c883e6c6609aff38f19`
- complete SHA-256 `d51ebe4f9881764408fc7326bbf34afb2de105cb2eeda2242fa724a453084609`

The author packet remains valid immutable evidence and must not be overwritten.
Its placement direction is sound, but two exact repairs are required before the
checklist can be treated as the minimum sufficient release gate for a managed
full-session attempt.

## Passing findings

1. **Seal and chronology pass.** All five manifested members match their sizes
   and SHA-256 values; `COMPLETE.json` binds the exact manifest. Local birth
   times show the manifest was created after the payload and validation and
   `COMPLETE.json` after the manifest. Shared birth times independently show
   `COMPLETE.json` copied last.

2. **Accepted predecessor evidence is exactly transcribed.** Both host/storage
   packets, the active-path map, readiness inventory, whole-stage-restart v2
   packet and its independent review rehash to the values in `PROVENANCE.json`.
   The active plan rehashes to `c26d63df...`. Accepted input path, support,
   clock, channels, dtype, gain, size, metadata-recorded hashes, DARTsort commit
   and source hashes, Kilosort source hash, wrapper absence and config absence
   agree with those predecessors. The specification correctly distinguishes
   metadata bindings from a fresh H5 payload verification.

3. **The placement arithmetic and conservative policy are defensible.**
   `482618717184 + 120000000000 = 602618717184` exactly. Reusing the full
   float32 byte count as an aggregate run-root cap is not an output estimate,
   but it is a conservative placement safeguard when paired with the reserve,
   continuous enforcement and a no-materialization rule. The text repeatedly
   says it is not expected consumption, allows a smaller envelope only after a
   separate evidence-based review, and does not promise that the cap is enough
   to finish. It includes output, scratch, cache, receipts, visible temporary
   files and one retained failed whole-stage attempt.

4. **Placement roles are appropriately separated.** The accepted input is
   read-only and may occupy another H5-local filesystem. All write roles resolve
   under one new nonsymlinked persistent run root and one named local filesystem;
   shared publication is compact only. Input, output, scratch, cache, failure
   retention, receipts and publication each have distinct roles.

5. **Present blockers and non-authorization are unmistakable.** The wrapper,
   real-window equality, full effective config/runtime, exact run root/current
   capacity, and current memory/oomd/GPU/process/manager state are explicitly
   unresolved. Status, scope, handoff, release rule and `COMPLETE.json` all say
   that H5 is not currently proven ready and this packet neither authorizes nor
   starts a run.

6. **Resource/manager gates are proportionate.** H5-07 requires a separately
   evidenced `MemoryMax`, headroom above it, parent and leaf pressure policy,
   swap and effective leaf enforcement rather than MemAvailable alone. H5-08
   binds the actual runtime and GPU UUID, tiny allocation, free device-memory
   envelope and conflict absence. H5-09 requires launcher-disconnection
   survival, exact installed bytes, effective live-leaf memory/tasks/wall/kill
   controls, explicit bounded-or-unbounded CPU policy, and terminal receipts for
   normal and abnormal exits. H5-10 prevents duplicate/conflicting full sorts.

7. **All twelve checks and invalidations are structurally present.** IDs H5-01
   through H5-12 are unique. Input, wrapper, environment, config/lineage,
   placement, storage, memory/oomd, GPU, manager, deduplication, checkpoint and
   publication/stop are covered. Invalidation conditions cover every bound
   input, source/config, filesystem, resource, conflict, restart and freshness
   domain.

## Blocking defects and exact repair

### 1. H5-11 does not close the predecessor's abrupt-kill boundary

The accepted restart review establishes only a **caught append exception** and
states that abrupt kill/power-loss durability remains untested
(`candidate3_whole_stage_restart_actual_path_independent_review.../REPORT.md:15,26`).
The new policy nevertheless requires quarantine of `manager-killed` stages
(`READINESS_SPEC.json:81`) and describes an interrupted stage as safely replaced
by whole-stage restart. H5-09 requires terminal receipts for timeout, oomd,
cgroup kill and user stop, but receipt capture alone does not demonstrate that
the partial scientific namespace is preserved, remains unpublished, is never
reopened, and can be replaced from unchanged bindings. H5-11 only says
`failure-injection receipt`; it does not require an abrupt managed-process kill
or these artifact invariants (`READINESS_SPEC.json:127-139`).

**Required repair:** replace/extend H5-11 with a cheap installed-entrypoint,
managed-job kill-boundary fixture. Kill or time out the process group during an
active peel after a partial stage exists, then verify:

- terminal manager/finalizer evidence survives;
- the original partial namespace and hashes are retained, has no successful
  completion marker, and is never reopened or promoted;
- the next authorized run refuses namespace reuse and selects a distinct,
  slash-free attempt namespace beneath a resolved nonsymlink run root;
- unchanged upstream input/source/config bindings are revalidated; and
- fresh whole-stage recomputation completes and equals an independent clean
  baseline for the fixture's declared arrays and lineage.

Power-loss durability may remain explicitly unestablished; the run contract
must then disclose that boundary. This repair tests the actual full-session
policy without requiring a real sort.

### 2. The aggregate cap is not yet machine-defined

The numeric envelope is acceptable, but `sum of allocated bytes` does not name
an accounting primitive, treatment of sparse/preallocated files or open-deleted
files, sample cadence, race margin, or breach response deadline
(`READINESS_SPEC.json:54-75`). H5-06 says the monitor is installed and exercised,
but a fixture cannot establish an exact cap until those semantics are frozen.
Also, `a second full-recording materialization appears` is ambiguous beside
`materialization_policy=PROHIBITED`: it could be read as allowing one additional
full copy.

**Required repair:** define in the specification or require the run contract to
freeze:

- the exact aggregate metric (for example a filesystem/project quota, or an
  explicit conservative combination of logical size, allocated blocks,
  open-deleted files and statvfs consumption);
- monitor cadence, breach margin and maximum stop latency;
- handling of sparse, preallocated, renamed and open-unlinked files and
  concurrent filesystem consumption;
- a fixture that breaches both aggregate cap and free-reserve paths and proves
  process-group stop plus durable terminal evidence; and
- unambiguous text that **any additional full-recording copy/materialization,
  including a float32 materialization, is prohibited and immediately breaches
  the gate**.

These are focused checklist edits, not a request for H5 contact, a new launcher,
or a resource experiment. After correction, retain this blocked packet and
publish a new immutable author version for focused rereview.

## Implementation checks

- Done: author seal and COMPLETE-last chronology -> exact hashes/sizes and local
  plus shared birth order pass.
- Done: predecessor and active-plan bindings -> every cited manifest/complete,
  source and identity value checked against the accepted artifacts.
- Done: storage arithmetic and interpretation -> exact; defensible only as a
  conservative placement guard, not measured usage or completion capacity.
- Done: exact roles, current blockers, 12 checks, manager/resource gates and
  invalidations -> structurally complete except for the two semantics above.
- Done: checkpoint attribution -> accepted fixture covers caught append failure
  only; abrupt managed kill remains explicitly untested.
- Not done: any H5 check, payload read/hash, wrapper equality, transfer,
  materialization, service installation/launch, sort, GPU work or RF/holdout ->
  prohibited by this review scope.
- Can establish: the packet is sealed, correctly scoped as not-ready/no-launch,
  and provides a strong draft placement policy with exact conservative
  arithmetic.
- Cannot establish: a sufficient machine-checkable storage stop policy or safe
  manager-kill whole-stage restart; therefore it cannot yet be the final release
  checklist for a full-session run.
