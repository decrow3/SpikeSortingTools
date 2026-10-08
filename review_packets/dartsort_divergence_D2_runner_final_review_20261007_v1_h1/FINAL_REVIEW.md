# H1 final incremental review — D2 managed runner preparation

## Verdict

**GO_TO_FREEZE_SEPARATE_PRODUCTION_CONTRACT_WITH_GATES.**

The three runner defects found in the provisional D2 snapshot are repaired.
The final packet demonstrates a fail-closed, synthetic-only managed runner and
is suitable as the implementation basis for a separately frozen bounded
production saved-output contract. This verdict does not enable production or
authorize any production-array traversal by itself.

## Review timeline

- Preparation began before `2026-10-07 20:29:23 PDT`; its exact start was not
  tool-observed, and H1 withdraws an earlier inconsistent prose estimate.
- H5 snapshot `reviewable_at`: `2026-10-07 20:27:03 PDT`.
- Direct review tool-observed start: `2026-10-07 20:29:23 PDT`.
- Final incremental review tool-observed start: `2026-10-07 20:48:26 PDT`.

## Bound evidence

Final packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D2_runner_preparation_h5_20261007_v1`

- MANIFEST:
  `ed86629295c0c33509959455e62281bd7f4bdd5b273483d178182916f86fb000`
- COMPLETE:
  `fa884dab92679debe920b630dca3795ba642e29927a4a5a46d663884625b6198`
- RUN_CONTRACT:
  `6cdb54829765857fbf4bc61b37850603c5f16a503e086988cab14d833881e696`
- Final runner:
  `1c75253f29ff43fdd93c460223137b96131a121dd3103c5878417d7fc832d2b9`

All final packet members and the nested managed-success manifest/COMPLETE seal
independently verify. The managed v3 source packet is sealed by MANIFEST
`efde3d6a...3577` and COMPLETE `edb8b7a...407e`; its contract, runner, tests,
requests, and fixture are byte-identical to the corresponding final members.

## Repaired runner boundaries

1. **Contract authentication before policy parsing.** The runner now reads raw
   contract bytes, checks them against a literal external `--contract-sha256`,
   and only then parses request whitelists, support, pairings, resources, or
   dependency paths. The managed argv contains the reviewed literal contract
   hash, and output provenance records both observed and externally pinned
   hashes. A wrong pin fails with `contract hash drift` before policy use.
2. **Immutable packet-relative fixture.** Absolute payload paths are rejected.
   The request path resolves relative to the authenticated contract's packet
   root, must equal the contract's frozen fixture path, and must match both the
   contract fixture hash and request hash. The final published packet directly
   reruns without the provisional staging-path failure.
3. **No-write refusal for non-fresh namespaces.** The runner claims its output
   with `mkdir(..., exist_ok=False)` before input processing. A pre-existing
   namespace returns status 3 without writing or replacing its historical
   `FAILURE.json`. New failed attempts receive their own fresh namespace,
   preserve `FAILURE.json`, and never receive `COMPLETE.json`.

## Direct and managed evidence

- H1 reran the final direct suite unchanged: `5 passed`.
- Managed success used the sealed v3 packet root, exact contract hash, frozen
  success request, one CPU quota, 512 MiB `MemoryMax`, 60-second
  `TimeoutStartSec`, and the five frozen environment variables. It completed
  with `Result=success`, `ExecMainStatus=0`, `NRestarts=0`, and produced a
  verified eight-member output plus COMPLETE written last.
- Managed expected failure used the same source/contract/environment and the
  frozen wrong-payload-hash request. It completed with `Result=exit-code`,
  `ExecMainStatus=2`, `NRestarts=0`, error `payload hash drift`, one preserved
  `FAILURE.json`, and no COMPLETE.
- H1 independently reverified all eight success members and recomputed pair
  event/count closures for all three pair names. DARTsort matching labels differ
  from final labels in the fixture; REF ancestry remains explicitly empirical.
- The packet preserves three superseded failures: the ephemeral absolute path,
  ineffective `RuntimeMaxSec` on `Type=oneshot`, and mutable staging-root launch.
  The final launch uses `TimeoutStartSec` and a sealed published v3 root.
- Services were cleaned after terminal evidence was captured; the packet
  preserves exact planned/claimed properties, environment, argv, compact
  journals, terminal statuses, and output hashes. Live unit state is therefore
  intentionally unavailable after cleanup, not evidence of an unfinished job.

## Provisional-to-final delta

The original provisional snapshot MANIFEST was
`416b0520...f495`. Its D1 adapter/matcher dependencies and accepted D1 review
remain unchanged. Only the D2 runner boundary was invalidated and rereviewed.

- Runner changed from `a0a07f01...d81e3` to `1c75253f...d2b9` to implement
  pre-parse contract authentication, packet-relative fixture binding, and
  no-write namespace claiming.
- Contract changed from `f52d4012...bdec` to `6cdb5482...e696`, updating request
  hashes, binding the fixture, defining external contract pinning, correcting
  the failure/no-overwrite rule, and replacing ineffective `RuntimeMaxSec` with
  `TimeoutStartSec`.
- Tests changed from `7dfabcbf...ed7f` to `48c69747...0970`, adding contract-pin
  and non-fresh no-write cases; the prior three tests remain and published
  replay now passes.
- Final-only members add managed launch, journals, terminal receipts, preserved
  failure history, nested success/failure outputs, readiness, and cleanup.

## Gates for any production contract

- Freeze a new production contract, request, exact production inputs and
  hashes, immutable runner/adapter/matcher sources, literal contract pin,
  systemd unit/environment/argv, fresh output namespace, and stop conditions.
- Recalculate and justify production resource bounds from the actual arrays and
  planned columns; the D2 `production_draft_resources` are review drafts, not an
  execution authorization or demonstrated peak.
- Require full DARTsort final/H5 row-time equality before matching features are
  consumed and empirical original/curated ancestry separately for REF and A.
  Never upgrade REF to recovered historical executable semantics.
- Hard-bind `100 / 25 / 0.01`, four anchors, complete uncapped candidate tables,
  three separate pair relations, no transitive identifier, and no depth,
  waveform, state, or presumed-identity candidate gate.
- Preserve managed terminal receipts, resource accounting, failures, and
  COMPLETE-last seals. A retry must use a new namespace.

## Implementation checks

- Done: final and nested integrity -> all hashes, sizes, manifests, and COMPLETE
  bindings independently verified.
- Done: changed source review -> contract pin precedes JSON parsing; payload is
  packet-relative and doubly hash-bound; non-fresh namespace returns without a
  write.
- Done: direct fixtures -> 5/5 pass from the published final packet.
- Done: managed success/failure -> exact recorded source packet, contract,
  environment, argv, unit properties, terminal statuses, journals, outputs,
  and cleanup inspected.
- Done: output closure -> nested success members rehashed; pair counts,
  unmatched counts, pair names, stage fields, REF scope limit, resource flags,
  and COMPLETE-last binding checked.
- Done: delta scope -> accepted D1 dependencies unchanged; only changed D2
  contract/runner/tests and managed evidence rereviewed.
- Not done: live systemd query -> units were deliberately cleaned after terminal
  receipts; review relies on sealed launch/status/journal/output evidence.
- Not done: production arrays or scientific outcomes -> prohibited and absent.
- Can establish: synthetic runner fail-closed behavior and readiness to freeze a
  separate bounded production contract with the gates above.
- Cannot establish: production resource sufficiency, real event divergence,
  exact historical REF executable semantics, biological identity/purity,
  sorter-only causality, or production benefit.

