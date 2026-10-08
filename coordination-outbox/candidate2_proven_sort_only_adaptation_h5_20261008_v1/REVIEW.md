# Focused independent implementation review — Candidate2 proven sort-only adaptation

## Verdict

**GO_TO_INSTALL_AND_AUTHORIZE_ONE_SORT_ONLY_INVOCATION.** The frozen adaptation preserves the successfully executed October 6 bounded-pread, ordinary-oneshot sort-only structure and changes the scientific join to the exact saved Candidate2 native-rigid request. The literal final entrypoint/config/source/environment/service boundary independently passes without reading recording content or starting a sort.

This is a technical release verdict for one separately authorized invocation. Installation, installed-unit reconciliation, live disk/memory/pressure/GPU checks, authorization, and execution remain future launch steps.

## Load-bearing findings

- The exact predecessor runner was recovered from bound commit `7ed3e2e7b376adfbe4668259d19b62a4830b0c15` as `testing/imec1_part2b_sort_retry_v2.py`; its SHA-256 is the contracted `62905e2df8cfb93869c653a821e4c0a5a26300b919c17ed5604bb4403a3cc90f`. The preserved success receipt records one completed 14,168.845-second sort-only invocation and the preserved ordinary oneshot service uses the same bounded-reader environment, cache-settle structure, CPU quota, persistent logging, and no service memory cap.
- The adaptation retains the predecessor control flow and adds frozen source/environment/service authorization, exact imec0 manifest/clock/channel-extent checks, a complete saved-request join, full content validation, and saved-effective-settings validation. The source diff does not add downstream QC or another sort invocation.
- `CONFIG.json` binds the accepted imec0 manifest (`2d15cf9...`), 241,309,358,592-byte binary receipt, 314,204,894 samples, 384 channels, and 29,999.835983263598 Hz. The exact saved request hashes to `58efcdb5...`, identifies Kilosort 4, and contains 46 sorter fields.
- The runner computes the current `RESCUE_RIGID.params()` and requires exact equality with all 46 saved fields before the production path. Those bytes request `do_correction=true`, `nblocks=1`, `do_CAR=true`, `Th_universal=12`, and `Th_learned=9`; the same saved request is passed directly to `run_sorter_config`. It does not substitute REF defaults or an Arm-A remap. Completion additionally reads `ops.npy` through `check_effective_settings("rescue_rigid", ...)` and rejects any effective `nblocks` other than 1.
- The three required cache operations are present in execution order: `validate_accepted_recording` hashes the full accepted binary in 8 MiB blocks and issues exact-target `POSIX_FADV_DONTNEED` per block; the runner then performs whole-file `DONTNEED`, waits five seconds, and rechecks pressure; Kilosort subsequently uses the exact-target bounded `preadv` reader with one reusable batch buffer and drops each batch from cache after its GPU copy.
- All 30 source members rehash exactly to `SOURCE_MANIFEST.json`; the stable launch copy is byte-identical. The uv lock, installed Kilosort I/O source, and SpikeInterface Kilosort wrapper match their frozen hashes. Authorization binds status, runner, source manifest, config, and literal service bytes. The runner verifies the full source inventory, environment files, installed-unit bytes, exact request, manifest metadata, binary size/receipt, and cache target before sorting.
- I reran the literal final boundary with the service interpreter, stable working directory/source tree, exact config/authorization/service, and service environment. It returned `PASS_FINAL_ENTRYPOINT_CONFIG_BOUNDARY`; the independently written receipt SHA-256 is exactly `5a78dde14b7ed74c8739ba4d6c7f2870caf9a040e170b7091515b8fefa6ba7e5`, with `recording_bytes_read=0` and `sort_started=false`.
- The retained environment negative changes the expected Kilosort I/O hash and fails before recording content access. Static tracing confirms an environment-file mismatch raises during `verify_contract`, before output creation or `validate_accepted_recording`.
- The service is ordinary `Type=oneshot`, has `CPUQuota=800%` and `TimeoutStartSec=28800`, and contains neither `MemoryHigh` nor `MemoryMax`. `Restart` is absent (systemd default: no automatic restart). The runner requires a fresh output root, emits one `sort_invocations` value, preserves failure evidence/partial state, and marks completion `complete_sort_only_downstream_pending`; QC is not called.
- The output root is absent, the target unit is not installed, and no authorization exists in the launch root. This review performed no content hash, voltage read, sorter execution, service installation/start, or GPU work.

## Evidence-hygiene note (non-blocking)

`evidence/CHANGED_BOUNDARY.stderr.txt` is an older changed-clock/channel transcript: its traceback line numbers do not match the final runner. It is not referenced by `CONTRACT.json`, `FINAL_BOUNDARY_PASS.json`, or `ENVIRONMENT_BINDING_NEGATIVE.json`, and it was not used to support this verdict. The current final positive boundary was independently reproduced, and the environment-negative failure is directly supported by final source order plus its focused receipt. Removing or labeling the stale auxiliary transcript in a future publication would reduce ambiguity; no executable repair is required.

## Implementation checks

- Done: recovered and hashed the exact executed predecessor runner at its bound commit; inspected predecessor config, service, and successful `SORT_COMPLETE` receipt -> successful bounded-pread sort-only lineage is real.
- Done: diffed the predecessor runner against the adaptation and traced exact saved-request consumption through the real sorter call and saved `ops.npy` gate -> Candidate2 native rigid is the intended science delta and effective `nblocks=1` is mandatory.
- Done: inspected full-hash cache dropping, whole-file eviction/settle/recheck, installed Kilosort bounded reader, and their target-path guards -> all three memory operations are intact and ordered.
- Done: rehashed source closure, environment files, config, service, request, manifest metadata, and reran the final zero-read entrypoint boundary -> exact final boundary passes and reproduces the frozen receipt hash.
- Done: inspected service and failure lifecycle -> finite timeout/CPU quota, no service memory cap, fresh output, one invocation, no restart/retry, and downstream QC separate.
- Not done: full recording content hash, live launch-resource check, installation/loaded-unit comparison, authorization, service start, sort, GPU computation, or QC -> outside this review and required only at the later launch/execution stages.
- Can establish: the exact frozen adaptation is technically defensible for installation and one separately authorized Candidate2 native-rigid sort-only invocation, subject to fresh installed-byte and live resource checks.
- Cannot establish: content-hash success, runtime/memory outcome, sort completion, output validity, QC, or scientific benefit before execution.
