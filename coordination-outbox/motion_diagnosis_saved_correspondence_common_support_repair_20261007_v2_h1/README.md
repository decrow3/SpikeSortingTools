# Corrected correspondence v2: cache-isolated execution handoff

Status: `READY_FOR_HASH_BOUND_H5_EXECUTION_AFTER_EXACT_CACHE_PREFLIGHT`.

This packet is the launch-only successor to the immutable H5 v1 failure. The matching source and its five scientific fixture tests are byte-identical to v1. The new contract changes only the fresh output namespace and Numba cache isolation:

- fixture tests populate `/tmp/corrected_correspondence_numba_fixture_h5_20261007_v2` only;
- the exact managed execution uses the distinct `/tmp/corrected_correspondence_numba_execution_h5_20261007_v2`;
- both paths must be absent before the v2 procedure starts;
- the exact H5 interpreter and copied script path must pass `tests/cache_isolation_preflight.py` before the service starts;
- no v1 cache or failed output is reused or removed.

H1 reran the unchanged fixtures in a fresh fixture-only cache: 5 passed in 2.78 seconds, and confirmed that the dynamic-import test creates cache records. H1 cannot execute the exact H5-path preflight because `/home/huklaban5`, the H5 dependency tree, selection, and saved arrays are not mounted here. H5 must preserve the resulting preflight receipt and managed-service launch/status/log evidence.

The v1 failure packet remains at `/mnt/NPX/Luke/DARTsort_motion_experiments/motion_diagnosis_saved_correspondence_common_support_repair_h5_20261007_v1_FAILURE` (MANIFEST `77a50197...`, COMPLETE `f67471e9...`). It emitted no scientific outcomes.

Implementation checks

- Done: unchanged source and test hashes verified against v1; five production-path fixtures pass in a fresh fixture-only cache.
- Done: preflight fails closed on identical cache paths, absent/empty fixture cache, nonempty execution cache, wrong interpreter/source, source hash drift, or failed exact-path `--help` execution.
- Done: new contract binds the predecessor failure, exact output v2 namespace, exact cache paths, interpreter, source/test/preflight hashes, and unchanged scientific semantics.
- Not done: exact H5 interpreter/script-path preflight and managed real-array execution; H5 owns these data-local actions.
- Can establish: the new packet prevents the known dynamic-import cache from being reused by normal script execution.
- Cannot establish: corrected real-cohort correspondence or any biological identity, purity, recovery, mechanism, or sorting claim until H5 executes and the result is independently reviewed.
