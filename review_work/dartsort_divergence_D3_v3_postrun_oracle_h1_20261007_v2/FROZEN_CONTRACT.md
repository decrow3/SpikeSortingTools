# Frozen D3 v3 post-run interpretation oracle v2

Status: `FROZEN_REPAIR_AFTER_INDEPENDENT_V1_FALSE_ACCEPT_REVIEW_BEFORE_ANY_V3_PRODUCTION_OUTCOME_IS_VISIBLE`

Owner: H1

## V2 repair boundary

V1 remains immutable and rejected for outcome interpretation at `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_postrun_oracle_h1_20261007_v1` (MANIFEST `3162ed5c3b5145abedecb30a43a891a6e61015ca30260d7352420cdbfc6ef807`). Its accepted blocking review is `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_postrun_oracle_independent_review_h1_20261007_v1r1` (MANIFEST `8a68c8cec143de76b35b4bfd8b6fa8247162b019a29ff1e0af9b21a4650ee9e9`, COMPLETE `baa292ec7c25aaf1169605ebe2d7f12f0f79e13445fd4cf293ec2b69c5fe848c`).

V2 repairs exactly three independently reproduced false accepts:

1. every exclusive matched pair and complete neighbour list is recomputed against inclusive ±15-sample final-event time support;
2. every anchor ranking must expose the same canonical eligible final-label set, and its row count must equal `eligible_final_units`;
3. candidate/anchor event totals, exclusive matches, smaller-train retention, F1, and pair-summary totals are recomputed and reconciled.

All other bindings, claim limits, and prohibitions below are unchanged. No future production namespace was inspected while making this repair.

## Bindings

- Executable derivative: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_executable_derivative_prep_h5_20261007_v2`
  - MANIFEST SHA-256: `10cdbcd20afdd665840496fa3549b1e5119b4c21f8e32ffb519e03ea49b0dee7`
  - COMPLETE SHA-256: `24228709fe0d032112f6224d4fe69ba0117d74c9da631e096ea1aebe4c89eb90`
  - contract SHA-256: `c41c9c521accf0fcd3c95778afd0511a4e4b799394491a6c5e3134302fd29dd5`
  - runner SHA-256: `725d811d2793a2d9a0fabd5ae20880a5cfaab80057dab0b8b5562e25657b5bb3`
  - request SHA-256: `1d8ba4ebf7576b4c8aa1f481ef4901adc28599e613e029d874e4aa53106c40a4`
- Technical review: `/mnt/NPX/Luke/DARTsort_motion_experiments/dartsort_divergence_D3_v3_executable_derivative_review_h5_20261007_v2`
  - MANIFEST SHA-256: `7874adc42c7aadf9da6bd93be0bb4d635ed7458874ab27c98a49195c3387e3fe`
  - COMPLETE SHA-256: `71be7fbab5e36892d9882cfaba850ff3a11a3c910d2af14f52072512626d37b6`
  - release SHA-256: `d830db10c9841046ac26183015086c502dedbdd13ca6d0f07b469245545ca684`

No future production output path or scientific outcome is an input to freezing this oracle.

## Frozen success interpretation

A successful output is reviewable only if:

1. the namespace contains exactly the ten named scientific/provenance members plus `MANIFEST.json` and `COMPLETE.json`, and no `FAILURE.json`;
2. all member byte sizes/hashes match MANIFEST and COMPLETE binds MANIFEST;
3. COMPLETE has the accepted schema/status and reports zero raw-voltage and GPU work;
4. provenance binds the exact contract, request, runner, adapter, matcher, and review release, and declares saved-output execution only;
5. final-event correspondence uses int64 acquisition samples from `final_npz.times_samples`, half-open support `[0, 314204094)`, inclusive tolerance 15 samples, and no event used for final matching is outside support;
6. `row_ancestry_established == false`, matching-stage assignment attribution is `unavailable`, and every emitted DARTsort stage row has null `parent_row_id` and `matching_label`; numerical time coincidence is never ancestry evidence;
7. the only scientific scope is direct final REF-to-DARTsort and A-to-DARTsort correspondence for the eight frozen arm-specific anchors, plus the four direct REF-A contexts. No transitive identity, biological identity/purity, or internal matching-to-final break localization is allowed;
8. candidate rankings include exactly eight anchors, the complete uncapped eligible final-label universe, frozen minimum supports, deterministic ordering, and exact dense-rank semantics;
9. event relations retain source IDs, and each context/side is serialized in stable lexicographic `(sample_time, source_row_id)` order with consecutive local indices; DARTsort stage rows have unique, strictly increasing final source-row IDs;
10. resource accounting stays within 2 CPU / 8 GiB RSS / 14.0 GB logical processing / 48 GiB aggregate reads / 4 GiB output / 14,400 s, and reports zero raw voltage, GPU, fit, replay, sort, RF, and holdout work.

Passing this oracle permits only the statement: **for the exact frozen anchors and support, the saved final REF/A event trains have the reported direct correspondence to DARTsort final units. Intermediate DARTsort matching-to-final attribution is unavailable.**

## Frozen failure interpretation

A failed namespace is closed only when `FAILURE.json` has the accepted failure schema/status, reports zero raw-voltage and GPU work, and `COMPLETE.json` is absent. Partial outputs are not scientific evidence. A stale hash, missing field/member, reordered relation row, false parent attribution, resource overrun, or mixed COMPLETE/FAILURE state is a blocker, not a result.

## Synthetic fixtures frozen before outcomes

- positive complete final-only correspondence packet -> accept;
- required field/member missing -> reject;
- member content changed after MANIFEST -> reject stale hash;
- relation rows reordered within one context/side -> reject;
- unavailable ancestry paired with non-null parent/matching attribution -> reject;
- valid preserved failure without COMPLETE -> accept as failure-closed, not scientific success.

## Bounds and stop

CPU-only synthetic data under 10 MiB. No production launch, production namespace inspection, new H5 output read, raw/waveform access, RF/holdout, sort, parameter change, or outcome interpretation. Seal the oracle packet before any future production outcome becomes visible. Because this task adds substantive consumer code, one bounded local reviewer must independently review only the frozen source/fixtures and packet.
