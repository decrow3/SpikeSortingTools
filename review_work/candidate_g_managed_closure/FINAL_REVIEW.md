# Candidate-G continuation managed-closure final review

## Verdict

`NO_GO_PRODUCTION_BOOTSTRAP_RUNTIME_AND_CANONICAL_BINDING_INCOMPLETE`

The managed lifecycle is materially stronger and the six real user-systemd rehearsals establish that the supervisor/finalizer design can survive launcher return and preserve bounded terminal evidence. They do not establish that the exact production service can start or that the real DARTsort worker can initialize inside its mount namespace. Three release-boundary defects remain blocking.

## Blocking findings

1. **Fresh writable roots are circular.** The production unit declares the exact run and compact roots in `ReadWritePaths` before any command runs, but the launch preflight requires them absent and creates them in `ExecStartPre`. The rehearsal's pre-existing broad writable base does not reproduce this condition.
2. **The real runtime lacks writable temp/cache locations.** Bubblewrap makes `/` read-only and binds only the run and compact roots writable. The unit does not point Python/DARTsort temporary files or JIT/framework caches into those roots. The dummy worker never exercises this path.
3. **Authorization and executable paths are not canonical.** The gate verifies hashes for release-declared preparation, H1 and unit paths but does not require the exact reviewed path identities. Internally consistent substitute packets or units can satisfy the gate, and the declared preparation packet can differ from the gate's own packet.

The repair should remain execution-disabled. Pre-provision and receipt exact empty writable roots outside the production service, bind all runtime temp/cache locations beneath the run root, and hard-pin the reviewed packet, H1, release, installed unit paths/names and executable-packet identity. Then run one disposable production-shaped managed rehearsal that uses the exact mount/environment path and performs real environment imports, temp/cache writes and CUDA discovery without starting the scientific continuation.

## Preserved accepted evidence

- The packet seal and exact inventory verify, including 20 explicit closure entries.
- Ten closure-focused tests independently pass on huklaban1.
- H5 evidence records six real systemd modes, asynchronous launcher return, live process identity, zero restarts, terminal seals and compact completion.
- The failed ReadOnlyPaths-only rehearsal is preserved, and the Bubblewrap rehearsal reports all 36 protected write probes rejected.
- The exact 14-array state design, accepted direct-agglomeration science, cache envelope, byte ceiling, expectation and runtime/resource bounds are unchanged by this review.

Implementation checks
- Done: what ran -> inspected the immutable packet, v3 gate/preflight/rendering/worker/lifecycle/publisher sources, production/finalizer and rehearsal units, contracts and real-systemd receipts; independently reran 10 focused tests.
- Done: arms/inputs -> no scientific arm or state change was introduced; this packet changes only execution closure.
- Done: caps/defaults -> retained exact half-open envelope `[239969921,250200079)`, 17,179,869,184-byte logical read ceiling, 14,973,326,368-byte expectation, 3,416-second worker timeout, 4 CPU, 64 GiB and one GPU.
- Done: provenance -> verified MANIFEST `da75f1d195fbb6146a62558494c877c7502e53eb9429c68fb652c691e17faebb`, COMPLETE `7b9a63ef2fe01915a74d77a46dd98a1af66a936cce007c140c0ee2454bbf760f`, request `fe3543a8122e9778684f6a993643de00b467de9c927a5bb0a1f18a07a0c08371` and coordination `7d5fc9fa7ab46be64d6a3f7ff52c58e36bb71898abb977698476990d558e1d95`.
- Not done: exact production service execution -> prohibited by the review scope and currently unsafe because the fresh writable-path contract is circular.
- Not done: real DARTsort initialization under the production Bubblewrap environment -> requires the bounded non-scientific rehearsal described above.
- Not done: systemd evidence regeneration on huklaban1 -> no unit was installed or started; H5 immutable receipts were inspected.
- Can establish: the revised supervisor, finalizer, containment probes and compact publication work for the six dummy-worker rehearsals recorded on huklaban5, and the packet has substantially improved source closure.
- Cannot establish: that the exact production unit reaches its preflight, that real DARTsort has sufficient writable runtime paths, that only the exact reviewed authorization/executable identities can pass, or that the continuation itself will complete scientifically.

No release, unit installation, start, voltage access or GPU work is authorized.
