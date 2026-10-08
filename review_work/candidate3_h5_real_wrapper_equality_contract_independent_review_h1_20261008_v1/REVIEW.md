# Independent implementation-first review: candidate-3 H5 real wrapper equality contract

## Verdict

`NO_GO_EXECUTION_READINESS_REPAIR_LAUNCH_BINDINGS_AND_RESOURCE_ENFORCEMENT`

Reviewed packet:

- `/mnt/NPX/Luke/DARTsort_motion_experiments/candidate3_h5_real_wrapper_equality_contract_h1_20261008_v1`
- manifest SHA-256 `b3e84dbaa8ef83ce2d387824fdca1f739fc98854761d91489e5fae9709ac7f9b`
- complete SHA-256 `670176d8cb69783b7da74f4791eaeb00291c9bc5ee9881ae3676494984d9c998`
- contract SHA-256 `2eeae1524c27d95e19f93dcdf8f9e2e0bb954e639ecb0d459c2fdd4da08c6afd`
- runner SHA-256 `c3904ee88043699c8ac1e270ec8f2c0b4f693a71d7fc73622b2b0a0d35356628`

The packet is scientifically well-scoped and most comparison plumbing is
correct, but it is not execution-ready. The exact runner/environment/review
authority are not cryptographically joined by the release gate, several stated
resource bounds are not enforced, and the exact packet runner fails its local
synthetic invocation unless an undocumented Numba-cache environment variable is
supplied. Preserve this packet and the reproduced failure; publish a new
contract/runner version after the focused repairs below.

## Passing findings

1. **Packet integrity and top-level COMPLETE-last pass.** Every top-level and
   nested self-test member matches the manifest. Both COMPLETE files bind their
   manifests. Local top-level birth order puts COMPLETE after MANIFEST. The
   runner's `manifest()` function writes MANIFEST and then COMPLETE for handled
   success/failure paths.

2. **Window arithmetic is exact.** Each central interval is 299,998 frames;
   each padded interval is 419,998 frames with exactly 60,000 frames on both
   sides. Every request length equals `relative_end-relative_start`. The
   transition padded origin is 217,431,311. The expected step boundary is at
   relative frame 206,249 and field-center change at 209,999. The extra request
   `[191249,221249)` maps exactly to global `[217622560,217652560)`, centers the
   expected step boundary with 15,000 frames on either side, and contains the
   field-center change with 18,750/11,250-frame support. All frames lie inside
   accepted recording support.

3. **The eager-versus-lazy comparison matches the reviewed capability.** Both
   receive the same cached int16 request, convert to float32 through the frozen
   wrapper path, use channels 0:384, no scale/shift, no sign inversion, temporal
   centering, lower-median CAR, a 300 Hz request-length FFT high-pass, disabled
   artifact gate and no whitening. Each primary request independently filters
   exactly its own length. Equality requires shape, float32, joint finite mask,
   elementwise identity, zero max error and byte-hash identity. This establishes
   same-installed-producer compatibility only; it does not independently
   validate Kilosort filtering or scientific benefit.

4. **Logical recorded-voltage access is bounded and sequential.** Real mode
   creates one read-only memmap, and the generator copies each of the two
   419998x384 int16 padded slices exactly once before yielding it. All lazy and
   eager requests then operate on that cached array. The exact logical read
   total is `2*419998*384*2 = 645,116,928` bytes. Iteration processes one window
   at a time. Compact metadata/geometry hashes occur before memmap access and no
   full binary hash or copy exists in the runner.

5. **Positive-control semantics are correct.** The runner compares the cropped
   `[60000,359998)` core from the padded 419,998-frame filter with the same
   299,998 frames filtered alone. Any equality fails closed as noninformative;
   a difference is reported only as proof of request-context sensitivity. Both
   synthetic controls differed strongly. No voltage-quality, transition or
   sorting interpretation follows.

6. **Durable numeric output is compact.** Result output contains hashes, shapes,
   counts, scalar errors and first mismatch coordinates, not voltage or
   prewhitened arrays. The synthetic result is 5,525 bytes and the entire
   evidence packet is compact. The actual array lifetimes remain in memory only.

## Blocking defects and minimal repair

### 1. The exact executable and import composition are not enforced

`PROVENANCE.json` and the packet manifest bind the runner, but `CONTRACT.json`
does not contain the runner SHA-256 and real-mode release checks bind only the
contract SHA-256. The runner records its own hash after output creation but
never compares it to an expected value. Consequently another runner can satisfy
the same contract/review/dispatch content.

The contract names exact H5 interpreter, repository and extra Kilosort-site
paths, but the code does not compare `sys.executable`, repository root or the
`--extra-kilosort-site` argument to them. It only checks imported source hashes.
Moreover, the packet copy derives `REPOSITORY_ROOT` from its packet location,
while the committed repository copy derives it correctly from `testing/`; no
exact execution path or command chooses between them.

**Repair:** bind the runner SHA and exact execution path/command in the contract
or a separately immutable release packet; verify self-hash before output and
before binary access. In real mode compare the resolved interpreter,
repository/root runner and extra site path to the frozen H5 paths, verify path
precedence, and record them. Prefer the committed repository runner at the
frozen commit, or add an explicit `--repository-root`; do not infer repository
identity from an arbitrarily relocated packet copy.

### 2. The exact packet invocation has an unbound environment dependency

One local synthetic execution of the exact packet runner with the declared
DARTsort interpreter and extra Kilosort site failed during DARTsort import:
SpikeInterface's Numba-cached function had no cache locator. The earlier wrapper
work already identified the requirement for a writable `NUMBA_CACHE_DIR`, but
this contract, runner, provenance and review request do not freeze it. The
author self-test therefore does not reproduce from the packet's declared inputs
alone. The runner correctly preserved this handled failure with MANIFEST then
COMPLETE and no array creation; no retry was made.

**Repair:** freeze a fresh bounded writable Numba/cache/temp directory and exact
environment in the execution contract and launch receipt; create it before
imports, include it in temporary accounting, and make the self-test use the same
composition. Then reproduce the exact immutable execution command.

### 3. The focused-review release gate is substitutable

The runner accepts any JSON file whose status starts `GO_` and which contains a
`manifest_sha256`, as long as the caller also supplies that file's own hash.
The dispatch receipt is contract-bound but does not bind the focused-review
COMPLETE hash or runner hash. Thus the two CLI `--expected-*` values are
self-asserted by the launcher rather than joined to coordinator authority.

**Repair:** make the coordinator dispatch/release receipt bind the exact
contract SHA, runner SHA, focused-review MANIFEST and COMPLETE hashes, exact
command/environment and fresh output path. The runner must verify every binding
and the focused review's reviewed-contract/runner fields before any binary
access. Use a specific accepted review status/schema rather than arbitrary
`GO_` prefix acceptance.

### 4. Declared resource and output bounds are not effectively enforced

`memory_max_bytes` is never used. `wall_seconds_max` is checked only after all
filtering finishes, so it cannot stop an overrun. `temporary_bytes_max` is never
used and cache/temp paths are not bound. `torch.set_num_threads(8)` does not
enforce a process/cgroup CPU ceiling. `persistent_output_bytes_max` is checked
before MANIFEST and COMPLETE are added, and failure outputs are not checked.
The runner therefore cannot claim these are stop conditions.

**Repair:** bind a persistent managed launch that proves effective MemoryMax,
wall timeout, task/CPU policy and group kill, plus an external output/temp/cache
monitor or filesystem quota. Put all writable cache/temp paths under the
monitored namespace. Measure final success or failure bytes including MANIFEST
and COMPLETE, preserve terminal receipts, and stop the whole process group on a
breach. A cheap no-voltage service fixture may validate enforcement before real
dispatch.

### 5. Pre-output and final-cap failure evidence needs an outer receipt

Release/hash/site failures occur before output creation and therefore cannot
produce the contract's promised failure packet. This is reasonable only if a
persistent launcher owns a prestart receipt and terminal status. Also, a
persistent-cap failure after `RESULT.json` would produce both RESULT and FAILURE,
while the schema says `RESULT.json or FAILURE.json`.

**Repair:** have the bound launcher preserve pre-output failures and terminal
status; make result promotion atomic only after final bounds pass, or explicitly
allow a provisional result plus failure. Verify COMPLETE-last and final byte
accounting on both success and injected failure paths.

No H5 check, voltage read/hash, transfer, materialization, service/GPU/sort,
RF/holdout or full-session launch was performed. These blockers are operational
and provenance-critical; they do not invalidate the window selection or the
synthetic equality evidence.

## Implementation checks

- Done: packet/nested seals, exact contract and runner, source imports, window
  arithmetic, boundary placement and logical read total -> pass.
- Done: eager/lazy construction, per-request FFT context, exact primary metrics,
  positive control and compact output -> source-traced and synthetic evidence
  reconciled.
- Done: exact packet-runner local synthetic launch -> failed before arrays on an
  unbound Numba-cache environment dependency; failure preserved without retry.
- Done: release/resource/output paths -> runner hash/import paths/review authority
  and multiple bounds are not effectively enforced.
- Not done: H5 state, recorded voltage/hash, real equality, service/GPU/sort,
  RF/holdout or full launch -> prohibited.
- Can establish: correct bounded comparison design, frame arithmetic, source
  semantics and compact handled-failure sealing.
- Cannot establish: reproducible exact H5 invocation, authoritative release,
  effective resource containment or execution readiness; therefore NO-GO.
