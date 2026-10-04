# Workstream-A spatial-provenance targeted repair

Status: `READY_FOR_H1_REVIEW_EXECUTION_DISABLED`.

This fresh namespace preserves the original H5 candidate and H1 review while
repairing their three requested gaps. It does not modify or overwrite either
historical packet.

The fixture and packaged CLI no longer contain an H5-private Kilosort path.
They resolve the packet-bundled dependency root named by aggregate digest
`d1e5a446...`, then verify the exact accepted `io.py` and
`postprocessing.py` hashes before spatial validation. A future execution may
supply another absolute root, but the same aggregate and per-file hashes are
mandatory. Altered declared hashes and altered source bytes both fail closed.

Spatial identity is now derived from the bytes and semantics that were actually
validated: position and ops hashes, shape, event-row lineage, frame, units,
columns, row semantics, clock, geometry, spatial region, and Kilosort dependency
digest. Any redundant `provenance.spatial_identity` claim is compared to that
derived digest and disagreement fails. If absent, the derived value is emitted.

The waveform sketch is replaced by a fully frozen execution-disabled contract.
It fixes the original voltage identity, 384-channel physical geometry/order,
sample rate and crop, exact three epoch edges, 15-frame matcher tolerance,
endpoint partition, deterministic SHA-256 sampling, 1,145-sample padded reads,
filter state/settings, reference/channel rules, support and exclusion semantics,
Wilson/zero-N denominators, outputs, hashes, first-result preservation, and hard
resource/stop conditions. RF and the sealed holdout are explicitly excluded.
The repaired/repeat arm and anchor-ledger hashes remain prospective; therefore
this packet cannot read voltage or launch a job.

H5 result: 22 bounded tests passed and the packaged CLI passed from packet
source. No real arm, evaluator, waveform, voltage, sorter, RF, or holdout work
ran. H1 must rerun the unchanged packet before GO.

Implementation checks
- Done: verified original candidate MANIFEST `710723d0...`/COMPLETE
  `8beba945...` and H1 review MANIFEST `00d95f01...`/COMPLETE `8cb04260...`.
- Done: traced and repaired the exact source dependency and redundant identity
  paths; positive and negative fixtures cover source hashes/bytes and identity
  contradiction.
- Done: froze the waveform contract and checked crop/epoch/resource arithmetic.
- Not done: no H1 rerun, real H5-local REF/B recomputation, prospective-arm
  binding, evaluator execution, or voltage access.
- Can establish: portable synthetic execution and fail-closed spatial binding in
  the packet environment; a complete non-executable waveform protocol.
- Cannot establish: real REF/B receipt truth, repaired/repeat availability,
  physical waveform support, scientific efficacy, identity, purity, or launch
  readiness.
