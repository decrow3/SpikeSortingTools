# D3 diagnosis versus frozen H1 oracle

Verdict: **ACCEPT_DIAGNOSIS_REPAIR_REQUIRED_NO_RETRY**.

The H5 saved-array diagnosis agrees with the H1 oracle frozen before diagnosis outcomes were inspected. It establishes that the v2 failure was caused by invalid final/H5 exact-time and global-chronology gates, not corrupt saved-output ancestry. It does not produce a D3 scientific result or authorize a retry.

## Immutable evidence

- Frozen H1 oracle: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_time_gate_oracle_20261007_h1`; MANIFEST `57b6acea805908b8fef988ef4047897f8942e9244080571d91e468cb05b673f9`; COMPLETE `288063642f2fcc83d9b9397713ae53afa04eb2140db6f9cccb381f5e31c987c7`.
- H5 failure diagnosis: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_production_failure_h5_20261007_v1`; MANIFEST `ae33b15f1cd35412f5acbd271c7761a6f5d51cb8d959332cd8c2be6502d1d38e`; COMPLETE `f6692ce436b870ec709cca099dc9248daf2ec2733c4188c3db9b548bf450c7c8`.
- Independent H5 review: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_production_failure_review_h5_20261007_v1`; MANIFEST `fceea1b3545e936447f60e95acc27f2b5feadcf90ed829349f9a08b19d3acbd9`; COMPLETE `4e2cb24a7ff85ce748ef2e0ddcafc42ebc38bfd94b356c326207c25b0bedc942`.

All 12 diagnosis members and all 4 review members independently match their manifests and COMPLETE bindings on H1.

## Oracle comparison

| Frozen source-bound expectation | H5 observed/reproduced result | Assessment |
|---|---|---|
| Exact final/H5 time equality is invalid | 21,804,714 rows have delta 0; 17,849 have delta +1 | Confirmed |
| General relation is rowwise final time = matching time + cumulative postmatching shift | Four complete saved source groups carry +1; all others carry 0 | Confirmed for this output |
| H5 `time_shifts` is already pre-H5 and is not the later delta | Delta equals neither H5 shifts nor their negative | Confirmed |
| Parent path/count/feature-count/channel checks are required | Resolved parent, 21,822,563 row/feature counts, and rowwise channels pass | Confirmed |
| Final NPZ time drives final-label support/correspondence | H5 diagnosis and review adopt this clock rule | Confirmed |
| Global row order need not be chronological | 287,811 final global inversions; table is row-aligned feature storage | Confirmed |
| Each final train must be checked or stably sorted with source IDs | All 981 current >=100-event trains happen to be chronological, but v2 consumers lack an explicit contract | Repair still required |
| No nearest-time final/H5 ancestry join | H5 retains source row index and rejects nearest-time alignment | Confirmed |

The observed +1 has the exact source sign predicted by reclustering: `final_time = matching_time - template_pair_shift`; the four changed groups imply template-pair shift `-1` sample. The matching H5 shift field is a different earlier stage.

## Additional load-bearing qualification

The complete constant-delta-by-`gmm_candidates[:,0]` relation is a strong consistency check for the exact hash-bound source/output pair, but is not a standalone row-ancestry proof. The independent same-channel block-permutation adversary preserves row count, channel equality, and constant integer delta within each permuted group while breaking parent-row identity. V3 must explicitly bind ancestry to the reviewed row-preserving source transformation (and disclose that basis), or add a high-cardinality fingerprint common to parent and final representations.

Do not freeze the already-observed four group IDs as the generic gate. The prospective gate should require a complete source-consistent group-to-delta receipt over all groups, with exact source/config/hash binding, plus adversarial limits. Otherwise it would validate the inspected outcome rather than the construction.

The included `diagnose_saved_times.py` reproduces the core row count, channel equality, delta counts, group constancy, H5-shift comparisons, and within-unit order counts. `TIME_MISMATCH_DIAGNOSIS.json` contains additional extended fields not emitted by that included script. The independent review reproduces the load-bearing core relation, so this does not overturn the diagnosis; v3 provenance should bind every emitted extended receipt to the exact generating source/command.

## Required v3 delta

Before any fresh freeze or launch:

1. Replace exact-time/global-order gates with source-bound parent path, row count, feature count, rowwise channel, complete group-delta, and explicit ancestry-basis checks.
2. Preserve final and matching clocks as distinct fields; final time governs candidate support and cross-pipeline correspondence.
3. Normalize every matcher input as a chronological `(time, source_row_id)` train or fail closed; preserve IDs through stable equal-time ordering and all emitted relations.
4. Add same-channel/all-fields row-permutation, within-group tamper, globally unordered-but-unit-sorted, within-unit inversion, equal-time duplicate, and tolerance-boundary fixtures.
5. Recalculate memory/read bounds for final channels, H5 channels, and source-group reads.
6. Use a fresh v3 contract, hashes, independent review, release, service, and output namespace. Preserve failed v1 and consumed v2 evidence.

## Implementation checks

- Done: packet integrity -> all diagnosis/review members and both COMPLETE bindings independently rehashed on H1.
- Done: frozen prediction comparison -> sign, stage, units, row counts, channel binding, H5-shift nonrelation, clock role, and ordering requirements compared against the pre-outcome H1 oracle.
- Done: executed-source consistency -> H5 trace hashes match the same DARTsort core files reviewed for the oracle; reclustering subtracts the template-pair shift and save preserves row order.
- Done: matching-consumer semantics -> the v2 candidate and pair consumers require chronological inputs; current eligible trains are observed sorted but the implementation contract is absent.
- Done: circularity/adversary -> exact four observed groups are not promoted to a generic prospective rule; same-channel permutation shows the proposed structural gate's ancestry limit.
- Not done: direct H1 reproduction from the 21.8M-row production arrays -> arrays are not mounted on H1; exact core counts were independently reproduced in the immutable H5 review packet.
- Not done: historical shift-matrix reconstruction -> matrix was not saved; source consistency and the complete observed group receipt do not regenerate it.
- Can establish: the exact pre-evaluation failure cause, source-consistent +1-sample relation, false global-order premise, and concrete v3 implementation requirements.
- Cannot establish: standalone ancestry from channel/group-delta gates, a reviewable v3 implementation, any D3 divergence result, biological identity/purity, sorter-only causality, or production benefit.
