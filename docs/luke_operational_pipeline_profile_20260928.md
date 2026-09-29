# Luke imec0 operational pipeline profile

Date: 2026-09-28. Status: **validated operational reference; external-rigid
challenger still pending paired comparison.**

## Current pipeline

The machine-readable profile
[`configs/luke_operational_pipeline.v1.json`](../configs/luke_operational_pipeline.v1.json)
consolidates the best-supported current pipeline for the complete
Luke0804_V2V1_g0 imec0 recording:

1. Neuropixels phase correction when present.
2. Bilateral samplewise blanking at 500 uV.
3. Interpolation of the frozen bad channel AP191.
4. Kilosort 4.0.27 with 12/9 detection thresholds, CAR enabled, and internal
   motion correction disabled (`do_correction=false`, effective `nblocks=0`).
5. Identity-bound legacy-compatible curation and QC.
6. Identity-bound standard unit-quality metrics with zero automatic curation
   changes.
7. The amplitude-completeness timeline as a screening-only analysis sidecar.

This is a dataset-specific operational reference. It does not claim universal
superiority, and warning or completeness flags do not remove or relabel units.

## Validation result

`python -m testing.validate_operational_pipeline` passes against the actual
recording, sort, curation, and QC artifacts. It verifies full-file SHA-256
digests, recording and sort request identities, the pinned sort identity,
critical saved sorter settings, every required stage receipt, and the compact
sidecar package inventory.

Validated inventory:

- sort identity: `22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`;
- recording content: `2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372`;
- 710 curated units and 29,227,829 curated spikes;
- 689 units with at least one lab-warning flag, retained for review rather
  than automatically excluded;
- 26,227 completeness windows, of which 9,204 are measured and the rest are
  explicitly censored or poor-fit rather than treated as numerical values.

The compact 5,330,189-byte QC package is published at
`/mnt/NPX/Luke/20250804/shared_analysis/luke_operational_pipeline_v1`.
Its 13-file manifest SHA-256 is
`e4641037f894df14e7ea2847fca4788d7dba6d51ae6793790ce674897ae5eeba`.
The profile verifies every payload hash and can use packaged outputs if the
original local analysis paths are absent.

Future production runs now finish the QC block with an
`operational_pipeline_validation` stage. Its receipt is written only after
curation, legacy QC, standard QC, and the completeness timeline all have
complete receipts for the same pinned sort identity, consistent unit/spike
inventories, zero automatic QC curation changes, and a screening-only
completeness policy. The visible run plan lists these stages individually.
The profile validation above supplies the equivalent backfilled proof for the
historical reference without modifying its completed output directory.

## Candidate boundary

The profile records why no silent parameter change is allowed:

- 8/8 and 9/9 threshold candidates did not pass the frozen donor-level gate;
- external nonrigid correction and its high-field-only use failed full-session
  paired evidence;
- cluster-553 artifact, wider-depth, motion, local-template, and lag variants
  did not support a repair;
- DARTsort remains a diagnostic under the architecture gate.

The completed external-rigid sort remains `comparison_pending`. Its persistent
consumer will update the candidate decision only after the atomically published
handoff passes content and identity verification. Until then, this profile is
the operational pipeline and no new sort is needed.
