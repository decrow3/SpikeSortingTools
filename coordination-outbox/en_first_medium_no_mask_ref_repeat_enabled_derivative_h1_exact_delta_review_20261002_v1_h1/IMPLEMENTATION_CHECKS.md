# Implementation checks

- Done: verified every subject manifest member and exact subject `COMPLETE.json`, contract, and coordination-status hashes.
- Done: recursively compared the enabled contract and both arm configs with the accepted disabled parent. Changes are limited to execution/status fields, added provenance gates, packet-local relocations, and derived hashes.
- Done: confirmed launcher review and activation review remain separate exact bindings, and the literal Phase3 token is `GO_PROVENANCE_REPAIR_DELTA` in contract, preflight, environment, status, and completion metadata.
- Done: confirmed science sources and active launcher are byte-identical; arm order, v2 runtime/output namespace, one-start behavior, no-retry policy, terminal reserve, finalizer, CPU/GPU/RAM/wall/storage limits, and `Restart=no` are preserved.
- Done: verified relative contract/config/service paths resolve to the sealed packet and exact hashes. With only the unavailable exact H5 runtime verifier stubbed, the actual outer preflight reports `INSPECTED_ELIGIBLE_NOT_LAUNCHED`, all namespaces fresh, and no voltage/outcome access.
- Done: ran 21 host-portable tests successfully; the two H5-runtime-only subprocess tests cannot execute on H1. The sealed H5 receipt reports all 23 passing on the bound runtime.
- Done: followed the repaired-arm call chain from `execute_repaired_arm` into `run_native_trained` and the exact consumed `validate_trained_contract`. Direct validation of the sealed enabled config reproduces `PermissionError: enabled trained execution requires an H1 review manifest` because `approval.h1_review_manifest_sha256` remains null.
- Done: traced environment consumption. Contract path/hash and service hash affect command-line expansion, but the exact environment file hash is not runtime-verified, and the `REQUIRED_*` provenance variables are unused by both entry points.
- Not done: H1 cannot query H5's user systemd manager. The exact H5 coordination status and packet receipt state that the service is not installed/started; no H1 fragment exists at the H5-home path.
- Not done: no service installation/start, start claim, recording/voltage access, sorting/training, outcome access, RF, or sealed holdout operation was performed.
- Can establish: the outer enablement delta preserves previously reviewed science and lifecycle mechanics, but it is not consumably runnable because the enabled repaired config deterministically fails its own validator; environment identity also lacks the claimed runtime check.
- Cannot establish: a successful two-arm run, repaired-arm execution readiness, H5 service-manager state by direct query, or any scientific result.
