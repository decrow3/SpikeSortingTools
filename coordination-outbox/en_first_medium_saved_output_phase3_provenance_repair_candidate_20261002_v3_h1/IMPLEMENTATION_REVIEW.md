# Implementation review: v3 producer/consumer provenance repair

V1 and v2 are preserved. H5's durable interface finding showed that existing
pair COMPLETE authenticated arm COMPLETE hashes but not either artifact
inventory. A fresh consumer rehash would validate current bytes without proving
they were the producer-validated run bytes, so consumer-only repair was
rejected.

The coordinated producer delta adds deterministic `artifact_inventory_path`
and `artifact_inventory_sha256` fields to both execution states before outer
finalization. The unchanged authoritative finalizer serializes the execution
state map into pair COMPLETE, closing the transitive chain. The consumer delta
requires and validates that chain before scoring.

## What actually runs

The active worker is `source/testing/first_medium_saved_output_qualification.py`.
It imports byte-frozen accepted snapshots for loading/spatial validation,
matching, `R`/`DeltaR`, truncation-QC interpretation, resource enforcement, and
source dependency validation. The contract binds every active module hash.

Phase 3 performs these operations in order:

1. Validate disabled/enabled gates and exact prerequisite-token equality.
2. Perform managed resource preflight.
3. Validate successful real Phase 1 receipts and method-freeze receipt.
4. Validate the exact bound producer contract/configs, exactly two execution
   states, arm-specific completion and inventory paths/hashes, completion arm
   association, exact curated-output locations, and every consumed inventory
   member. Then claim the HMAC-bound single-use
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
- REF repeat is bound to `<pair root>/REF384_repeat/sorter_output`; repaired B
  is separately bound to its reviewed config's `native_invocation.results_dir`.
- A same-basename path outside the deterministic arm namespace is rejected.
- The expected saved-array identity is derived from authenticated inventory
  hashes and is then checked by the accepted same-buffer hash/parse loader.
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
- Done: frozen report-schema validation -> every successful managed known answer validates against the accepted byte-frozen compact-report schema.
- Done: producer-compatible transitive binding -> lightweight arm inventories are bound into producer states and authoritative pair COMPLETE by the accepted outer finalizer.
- Done: negative controls -> absent/swapped states, unrelated same-basename inventory, wrong completion hash, unfinished pair state, and saved-output mutation all stop before consumer output/scoring.
- Done: REF-first ordering -> a Phase 1 REF mismatch opens only REF and no candidate.
- Done: missingness and dependency controls -> missing QC/waveform are UNMEASURED; no `amplitudes.npy` is needed.
- Not done: independent H5 source/artifact rereview -> required before filling prospective activation tokens.
- Not done: real Phase 3 execution or real-outcome inspection -> prohibited for this authoring task and disabled in the contract.
- Can establish: the execution-disabled v3 delta closes the identified producer-to-inventory edge and rejects tested path/hash substitution before unchanged scoring.
- Cannot establish: real-arm success, real metric values, scientific efficacy, biological identity/purity/recovery, or an advancement decision.
