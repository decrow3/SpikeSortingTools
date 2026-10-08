# Independent review: H1 candidate-3 transition host readiness

## Verdict

`PASS_SCOPED_OPERATIONAL_RECOMMENDATION_WITH_REENTRY_PROBE_FREEZE_REQUIRED`

The evidence supports the operational recommendation
`ABANDON_H1_FOR_CANDIDATE3_TRANSITION_USE_EXISTING_H5_SUPERVISED_PATH_LATER`.
“Abandon” is correctly scoped to the current transition attempt and present H1
condition, not permanent host decommission. “Use H5 later” refers to the already
reviewed supervised path and does not claim that H5 is currently ready.

Reviewed packet:

- local and shared MANIFEST `caafac6bc3a258437aed3f8b1e913ce8e84b9a5bae2607ec145841777caf8680`
- local and shared COMPLETE `f062c8a8c23e45562a78f09bc59c9d7c19c7604206ed0fc9e3849ac30649c46e`
- author commit `f92d9d7`

## Findings

1. **The launch prohibition is directly supported.** The author snapshot shows
   load near 52, 52 D-state threads, 36 blocked tasks and 77–79% I/O wait with
   negligible block throughput. An independent read-only host-namespace recheck
   reproduced 52 D-state threads: 35 Python threads in
   `folio_wait_bit_common`, 10 bfs I/O workers in `fuse_lock_inode`, and six sync
   processes. Four fresh `vmstat` interval samples again showed 36 blocked tasks
   and 77–78% I/O wait with zero or small block output. This is incompatible with
   a controlled transition run and is not trending toward a bounded passive
   recovery.

2. **The causal wording is appropriately limited.** The April Python/SpikeGLX
   family is the dominant measured D-state contributor at 35 of 52 threads and
   predates the August/September bfs searches by roughly five months. Open-file
   evidence joins that family to the cited `/mnt/NPX/Luke/...ap.bin`, while the
   wait channels and CIFS statistics locate the pressure in remote/FUSE/CIFS
   paths rather than sustained local-device throughput. This supports calling
   the Python family the primary observed blocker. It does not prove the exact
   kernel/server defect, and the report explicitly preserves that uncertainty.

3. **Whole-root bfs jobs are contributors, not a sole-cause story.** Ten blocked
   bfs I/O threads are material and should not be repeated. Their later start
   dates rule them out as the origin of the April waiters, and 10 of 52 threads
   is not the dominant measured share. The report accurately calls them
   causative contributors while rejecting them as the original or largest
   blocker. The six old sync waiters are also preserved as a separate family.

4. **H5 wording is safe.** The recommendation defers to a later coordinator
   priority and existing supervised path. It does not contact H5, assert current
   H5 resources or authorize a launch. Therefore H1 unsuitability is not being
   misused as evidence of H5 readiness.

## Re-entry gates

The two-snapshot, ten-minute separation; absence of project-relevant `/mnt`
D-state tasks; three `vmstat` samples with `b=0` and I/O wait below 10%; absence
of whole-root searches and blocked syncs; and a full fresh managed-launch
resource/dedup prestart are defensible conservative gates. They require both
process-level and system-level recovery and prevent a single transient sample
from clearing H1.

One gate needs implementation detail before it can govern re-entry: “a bounded
`/mnt` metadata probe completes within its declared timeout” does not yet freeze
the exact operation, target, timeout, repetition count or success metric. Before
administrator remediation is assessed, freeze a cheap nonrecursive probe against
an approved compact project path, a numeric timeout, two-snapshot placement and
fail-closed exit/status capture. Do not use a whole-root traversal or voltage
file. This is a future gate-specification qualification, not a reason to weaken
the current H1 abandonment decision.

The `<10%` I/O-wait threshold is a conservative operational safeguard rather
than an empirically estimated performance boundary. It is defensible for re-entry
because current values are 77–79%, but passing it would not alone prove H1 ready;
all gates and the fresh managed prestart remain conjunctive.

## Implementation checks

- Done: local/shared seals and commit identity -> match exactly.
- Done: host-namespace process/thread census, ages/wait families, `vmstat`, CIFS
  counters and mount identity -> reproduce the load-bearing snapshot facts.
- Done: contributor attribution -> April Python family is older/dominant;
  September bfs and sync families remain material contributors, not sole cause.
- Done: re-entry rules -> directionally conservative and conjunctive; metadata
  probe lacks a frozen operation and numeric timeout.
- Not done: kernel stack/server diagnosis, administrator remediation, ten-minute
  post-remediation snapshots, metadata probe, H5 state or any mutation -> outside
  this read-only review.
- Can establish: H1 is presently unsuitable for the candidate-3 transition and
  passive waiting is not a bounded readiness plan.
- Cannot establish: exact server/kernel root cause, permanent H1 unsuitability,
  current H5 readiness or successful remediation.
