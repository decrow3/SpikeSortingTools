# Candidate-3 missing-transition resource/launch contract independent review

Verdict: `BLOCKED_PENDING_MANAGED_LAUNCH_CLOSURE`.

The scientific scope, resource arithmetic, systemd resource properties, no-retry policy, and whole-stage restart semantics are coherent. The contract is not yet sufficient to launch because its declared preflight, storage stop, terminal manager receipt, and post-run accounting requirements are not implemented by the frozen argv.

## Blocking findings

1. **The executable source path does not match the frozen immutable binding.** `CONTRACT.md` binds `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_mini_e2e_executed_source_recovery_h1_20261008_v1/source/candidate3_mini_e2e_stage_capability.py`, but `LAUNCH_REQUEST.json` hashes and executes `/home/huklab/Documents/RyanSorting/SpikeSortingTools/review_work/.../source/candidate3_mini_e2e_stage_capability.py`. Both currently hash to `5d55bb1e...`, but the latter is a mutable workspace path. The repaired argv must execute the immutable shared packet source, or separately freeze and bind a committed executable copy.

2. **Preflight and outer-launch receipts are specifications, not implementation.** The exact argv calls `systemd-run` directly around `testing/managed_job.py`; neither component verifies the listed hashes, DARTsort commit/clean state, current free space, service-name freshness, output containment, or nonsymlink requirements. No exact checker command, schema, or immutable prelaunch receipt destination is frozen. A reviewed preflight/launcher must perform these checks after creating only the job directory, atomically save their values plus the exact outer argv, and refuse launch on any mismatch. It must also require fresh stdout/stderr and a fresh in-root NUMBA cache.

3. **Timeout, OOM, cancellation, and outer-launch failure do not have durable failure closure.** `managed_job.py` writes a running receipt and updates it only after `subprocess.run` returns or raises in Python. `RuntimeMaxSec`, `MemoryMax`, or user stop can terminate the wrapper cgroup before that update, leaving `managed_receipt.json` at `state="running"`. The launch request has no `ExecStopPost` or equivalent finalizer and no persisted destination for `systemctl show` terminal properties. It also does not save the `systemd-run` return code if unit creation fails. Add a reviewed manager finalizer/receipt that records at least service result, exit code/status, timestamps, and terminal properties for normal exit, timeout, OOM, cancellation, and launch failure.

4. **The 512 MiB storage stop is not enforced by the managed job.** The arithmetic is exact, but no service or companion monitor measures the combined retained inputs, ordinary output, v4 job/output/logs, and cache while the run is active. The current NUMBA cache is outside the v4 root and outside the stated accounting. Either add a managed cap watcher that stops the service and writes its observation, or explicitly redesign the cap as a post-run classification under a documented free-space condition. If it remains a hard stop, monitoring cannot depend on the chat session.

5. **The completion condition has no frozen downstream implementation.** The service argv runs only `run ... transition`; it does not generate compact accounting/report artifacts. Ordinary and transition outputs are in separate v3 and v4 roots, while the available summarizer accepts one `runs_dir`. Freeze the exact post-run composition/summarization source, argv, inputs, output namespace, schemas, resource limit, and failure rule. Any downstream job that waits for the sort must itself use the independent manager rather than depend on the chat.

## Verified portions

- Contract packet: 3/3 members match; MANIFEST `847eee86d2dc461b1a6fd88ceb229d8f78440ae90faaf416035eb08e57bb134c`; COMPLETE `fb48e3fe9e54432de3f04675da74982ebb15bb1d9eecc983129ba0288842bfd2`; payload-before-MANIFEST-before-COMPLETE birth ordering passes.
- Dependencies: executed-source recovery `74761855...` / `11602c0b...` and mini independent review v2 `3e3aa9b...` / `dd33c8d...` match their shared files.
- All five hash-preflight targets currently match. The DARTsort checkout is at `edcfe1b51d672b4136eb13cc78c0875da804b851`; `managed_job.py` hash is `5867440e...`; its two focused tests pass.
- Arithmetic independently recomputes exactly: projection `160,279,947`; subtotal `412,698,120`; headroom `103,174,530`; derived need `515,872,650`; cap `536,870,912`; remaining margin `20,998,262` bytes.
- Systemd properties are internally consistent: 45-minute runtime, 16 GiB memory, 800% CPU, 128 tasks, control-group kill, 30-second stop timeout, and `Restart=no`. Existing success/failure dummy receipts and logs show survival of their disposable launcher, but do not test this contract's missing finalizer or storage monitor.
- The service is currently `not-found`/inactive; v4 root, job directory, managed receipt, and transition output are absent. These observations are current readiness evidence, not a durable launch-time preflight.
- Whole-stage recovery is correct: `run_arm` uses a fresh output directory and `overwrite=False`; `managed_job.py` refuses receipt reuse; failure must remain in v4 and any retry must use a new namespace after investigation. There is no within-sort checkpoint.

## Implementation checks

- Done: resource arithmetic and units -> every frozen byte total independently reproduced.
- Done: exact argv/source bindings -> found mutable-path mismatch despite equal current content hashes.
- Done: managed-job code and focused tests -> normal success/failure receipts work; abrupt cgroup termination closure is absent.
- Done: fresh/nonsymlink/containment contract -> current paths are absent and service is not loaded, but no launch-time executable check exists.
- Done: stop/restart semantics -> no automatic retry and fresh whole-stage restart are correctly stated and partly enforced by receipt/output refusal.
- Done: dependency seals -> recovery and mini-review hashes match; scope and detection-default caveats carry forward.
- Not done: transition execution, real voltage, RF/holdout, D3, or any service mutation -> excluded by this review.
- Can establish: the proposed one-arm transition is scientifically scoped and has a plausible, correctly calculated resource envelope.
- Cannot establish: safe launch/failure closure or complete managed delivery until the five implementation blockers are repaired and independently reviewed.

Required decision: `NO_START`. Preserve this v1 contract and publish a new repaired namespace; do not edit or promote it.
