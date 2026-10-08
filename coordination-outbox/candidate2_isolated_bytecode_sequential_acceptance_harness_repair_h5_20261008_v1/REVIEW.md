# Focused independent review — Candidate2 sequential-acceptance harness repair

## Verdict

**NO_GO_ACTUAL_PRODUCTION_BINDING_DEFECT.** The single retained harness traversal is valid as a harness-only source/bytecode test, but it did not run the exact frozen service argv. All three production service bootstrap self-hash arguments are stale, and the service contract-hash argument is also not the reviewed packet contract hash. Exact production argv would fail before the demonstrated preflight/runner-import boundaries.

No repair or second traversal was performed.

## Blocking finding: dynamic harness bindings differ from frozen service argv

The harness constructs its bootstrap invocations with live hashes:

- preflight: `--self-sha256 sha256(root / "preflight_bootstrap.py")` (`tests/test_full_preflight_main.py:85-93`);
- main: `--self-sha256 sha256(root / "bootstrap.py")` (`lines 96-108`);
- both use a freshly written synthetic contract and its live hash.

Those dynamic bindings pass, but the exact service files contain different values:

| Component | Actual packet SHA-256 | Frozen service `--self-sha256` |
|---|---|---|
| `bootstrap.py` | `672c9fa088f8523adbdde7bf391ae2effe0ead9eefde79c4a5065fb1d8820faa` | `a96ecc3c2cc2a2a921bdd6ab9c9b26591561d02b67da5da73a27c6b59a14b428` |
| `preflight_bootstrap.py` | `d7b6f8573f46d46ca7348c703ea5849d8ac9d7fc805fb6891146c540fad9b81f` | `333a9c72d5e940d7e60a9a4666c604bd6c2fb2a4b82dae5c0e62a065cf86eb7f` |
| `monitor_bootstrap.py` | `043abbff58a31fe66ece4a7c6266ea9b1830deb15cd91362a413b1c62a98fe9c` | `3c8f6f73ff71069bc985b2afe69963728d57842e5238aad159ea6747f10de16d` |

The target and monitor services also pass contract SHA-256 `4769968e7bef087d07097bb61b745de7882afaa9e38c2d20f7c75eb4071d6c68`, while this exact execution-disabled packet contract hashes to `8c918b84f8ccd4aecaf8e5f148a14bbadb0941ffc3512a10b46c8a8c0067b8da`. The contract's `source_and_environment` section likewise retains the old three bootstrap hashes.

Each bootstrap verifies its own or contract hash before project import. Concrete failure: using the frozen target `ExecStartPre`, target `ExecStart`, or monitor `ExecStart` argv causes a hash mismatch before the valid preflight receipt or real runner import demonstrated by the harness. Thus `JOB.json` proves the runner imported under the same interpreter flag shape (`-I -B -S -u`), but not under the exact deployed argv/bindings required by this review.

Minimum repair: in a fresh immutable successor, bind each service's self-hash to the exact carried bootstrap byte and bind the service contract hash to the exact intended executable contract. Add a retained negative/positive join that executes or statically validates the literal service argv without substituting dynamically calculated values. Preserve this packet and do not rerun its traversal.

## Harness checks that passed

- Preserved v9 failure packet MANIFEST SHA-256 `a4fd0c8df3e10178afacabd7ba8fbd6a9c0401deabe8aff27933b3cba61605e8`; all 50 members rehash exactly and its COMPLETE preserves the failure/no-production-attempt state.
- All 38 production contract/service/bootstrap/source members compared are byte-identical between preserved v9 and this packet. The only executable harness delta is `tests/test_full_preflight_main.py`, from `1ceb5b5a...` to `0f89e26d...`, adding receipt directories and retained evidence.
- `TEST_RECEIPT.json` SHA-256 is `22befc07ba7e3df901060d675beaa08bc6ee45b14ab606405da64835d66f398f`; all 20 retained artifacts match its byte counts and hashes.
- The retained preflight-bootstrap subprocess serialized a valid receipt (`ab1b1a1d...`) with `recording_opened=false` and zero bytes read.
- The retained main-bootstrap receipt (`ff06fc35...`) proves installed-dependency validation and a successful import of the real `candidate2_cache_managed_full` module in explicit import-only mode before calling its `main()` or reaching recording access.
- The 31-member source identity equals `SOURCE_MANIFEST.json` after preflight, after runner import, and after cleanup. During mutation it differs by exactly `SEQUENTIAL_ACCEPTANCE_MUTATION.txt`; the bootstrap rejects it with `runtime source inventory mismatch`, and the final identity returns exactly to 31 members.
- Path, argv-boundary, and `ignore_errors` authorization corruptions each fail with `manager-loaded typed execution identity differs from authorization`.
- Retained timestamps, fixture path, and receipts are consistent with one traversal; `traversal_count` is 1. No second traversal was run during review.

## Scope clarification

No production recording, sort, service install/start, or authorization occurred. The retained full preflight did create its expected small CUDA self-observation context (202 MiB reported) to test the GPU gate, but no sorter/GPU computation began. Temporary dummy services belong to the harness, not production.

## Implementation checks

- Done: diffed exact preserved-v9 production bytes against this packet -> all production bytes unchanged; only the harness/evidence changed.
- Done: inspected harness invocations, exact service argv, bootstrap verification order, and retained receipts -> dynamic harness hashes pass, exact deployed hashes would fail before import.
- Done: rehashed all retained artifacts and compared every source-tree boundary -> 31 members preserved; one-file mutation rejected and removed.
- Done: inspected preflight/JOB and authorization-negative evidence -> pre-recording import-only flow and all three negatives behave as reported.
- Not done: another traversal, repair, install, authorization, production start, recording, sort, or GPU workload -> explicitly prohibited.
- Can establish: the harness repair validly demonstrates bytecode-safe sequential preflight/import behavior when supplied correct dynamic bindings.
- Cannot establish: exact deployed-argv acceptance or production readiness, because the frozen literal service bindings are wrong.
