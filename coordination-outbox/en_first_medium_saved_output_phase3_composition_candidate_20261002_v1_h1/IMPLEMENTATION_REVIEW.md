# Implementation review

## What actually runs

The active worker is `source/testing/first_medium_saved_output_qualification.py`.
It imports byte-frozen accepted snapshots for loading/spatial validation,
matching, `R`/`DeltaR`, truncation-QC interpretation, resource enforcement, and
source dependency validation. The contract binds every active module hash.

Phase 3 performs these operations in order:

1. Validate disabled/enabled gates and exact prerequisite-token equality.
2. Perform managed resource preflight.
3. Validate successful real Phase 1 receipts and method-freeze receipt.
4. Validate authoritative bounded launcher-v3 pair receipts, then claim the HMAC-bound single-use
   start token.
5. Write `STARTED.json`, then validate the active implementation boundary.
6. Load REF and compare its curated identity, QC request identity, and spatial
   identity with Phase 1 before opening any candidate arm.
7. Load the other three arms, compute accepted fixed-cohort measurements, and
   emit a compact report with separate execution, binding, metric,
   repeatability, diagnostic, waveform, and decision states.
8. Write MANIFEST, then COMPLETE last. Any post-start exception writes FAILURE
   and cannot write COMPLETE.

## Arm and estimator controls

- Exact arm order is enforced.
- The fixed cohort is content-digested and must be present in REF.
- Unmatched cohort units remain in the denominator with retention zero.
- Candidate events receive exclusive credit through one reciprocal-primary
  partner; matching remains bijective.
- Bootstrap seed/count and strict threshold are read from the frozen contract.
- Fixed-cell outputs aggregate accepted per-unit diagnostic rows; they do not
  publish a unit gain/loss ranking.
- QC absence is explicit `UNMEASURED`; no empty-table default, imputation, or
  refit occurs.
- Waveform is explicit `UNMEASURED` until accepted pair-bound evidence is
  available. It cannot pass by omission.

## Saved-array dependency check

The accepted loader hashes and reads only `spike_times.npy`,
`spike_clusters.npy`, `full_st.npy`, and `kept_spikes.npy` for curated event
arrays. It obtains amplitudes from `full_st[kept_spikes][:,2]` and separately
binds `spike_positions.npy`, `ops.npy`, labels, clock, geometry, and Kilosort
source bytes. The managed known-answer fixtures intentionally omit
`amplitudes.npy` and pass.

## Scope

Implementation checks
- Done: executed-source syntax and content-hash boundary -> all eight active module hashes match the disabled contract.
- Done: managed four-arm known answers -> below, exactly-equal, and above the strict 0.05 boundary produce fail, fail, and pass metric states respectively; advancement remains false.
- Done: receipt and mutation negative controls -> unfinished pair state stops before output; post-start input mutation writes FAILURE with no report or COMPLETE.
- Done: REF-first ordering -> a Phase 1 REF mismatch opens only REF and no candidate.
- Done: missingness and dependency controls -> missing QC/waveform are UNMEASURED; no `amplitudes.npy` is needed.
- Not done: independent H5 source/artifact rereview -> required before filling prospective activation tokens.
- Not done: real Phase 3 execution or real-outcome inspection -> prohibited for this authoring task and disabled in the contract.
- Can establish: the execution-disabled composer enforces the frozen orchestration and produces the specified states on tiny saved-format known answers.
- Cannot establish: real-arm success, real metric values, scientific efficacy, biological identity/purity/recovery, or an advancement decision.
