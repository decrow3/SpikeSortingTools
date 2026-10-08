# H5 real saved-output integration review of the H1 v2 evaluator

Status: **read-only traversal attempted and fail-closed; scientific REF384 → existing B384 row remains UNMEASURED.** No recording voltage, sorter, training, detection, RF/holdout, prospective arm, or external service was accessed.

The reviewed candidate packet verified completely:

- candidate: `/mnt/NPX/Luke/DARTsort_motion_experiments/en_first_medium_four_arm_evaluator_review_candidate_20261002_v2_h1`
- manifest SHA-256: `67a607934be67e9856299e5f65637797c64352b68a04438caefb7ee165f88fd2`
- COMPLETE SHA-256: `f79caa55a2bf519e83f2dccc37bbb3e2dbc534a8083254e6590e71f7827b7aa9`
- every candidate manifest member passed.

## Actual traversal result

The byte-identical reviewed `first_medium_evaluator.py` (`f5ff0050...`) and `sort_comparison.py` (`ce2896f3...`) were bound in an isolated `testing` package and invoked through their actual CLI. The run returned 1 before creating an output root or `FOUR_ARM_REPORT.json`:

`ValueError: REF384: full_st[kept_spikes] times do not match spike_times.npy`

For both arms, the row counts align, ordering is monotonic, and the discrepancy is a single exact offset:

- REF384: 1,856,371 rows; `spike_times - full_st[kept,0] == 208498882` for every row.
- existing B384: 1,854,340 rows; the same exact offset for every row.

The saved `spike_times.npy` arrays are global acquisition samples, while `full_st[kept_spikes]` is crop-local. The evaluator config declares crop-local `[0,17999901)`, but its loader requires direct equality before normalizing either clock. This blocks all matching/counting and therefore every scientific endpoint.

A second independent blocker remains: neither arm contains the required `qc/amp_truncation/truncation_qc.npz`. Exact requested paths are recorded in `evidence/INVENTORY.json` with `exists=false`, digest null, and status `UNMEASURED_ABSENT`. No alternative A/full-session/fixed-bank/smoke/RF cache was substituted.

Required adapter repair before a future traversal: validate that `full_st[kept,0] + 208498882 == spike_times`, emit exactly one crop-local time representation for evaluation, and bind a compatible real cached truncation-QC artifact for each arm. Until both prerequisites exist, correspondence, coverage, split/merge, amplitude missingness, and the four-arm scientific scorecard are not measurable.

## Bound saved artifacts

`evidence/INVENTORY.json` gives absolute path, digest, shape, and dtype for every loader-consumed `spike_times.npy`, `spike_clusters.npy`, `full_st.npy`, and `kept_spikes.npy`; it also binds `cluster_KSLabel.tsv`, optional `spike_positions.npy`, `ops.npy`, arm receipt, screen config, clock, geometry, runtime, and input identities.

Both geometries are `(384,2)`, share file SHA-256 `0469ca92...`, share float64 value digest `d126fb24...`, and have identity `chanMap`. Spike positions are present and aligned with spike rows. No raw binary was rehashed or opened.

## Adapter review

The v1→v2 load-bearing delta adds mandatory real-arm provenance fields and mandatory expected saved-array/QC digests before data loading, then compares both observed identities exactly. The new negative test confirms unbound real arms fail before path reads. This delta is fail-closed.

One packaging issue was independently reproduced: invoking the packet file directly resolves `testing.sort_comparison` to the repository package because the packet snapshot has no `testing/__init__.py`. The repository adapter hash is `d107acee...`, not the reviewed `ce2896f...`. The authoritative traversal therefore used an isolated package shim with byte-identical reviewed files; `evidence/COMMAND.json` proves runtime and candidate hashes agree. Future launch packaging should make this binding intrinsic rather than rely on orchestration.

Synthetic rows are excluded correctly: CLI specs reject `synthetic_fixture`; scorecard eligibility requires both arms to be `real_saved`; fixture rows are stored separately. Missing evidence uses `UNMEASURED`; mathematically undefined identity continuity uses `UNRESOLVED`; neither can carry a numeric value or pass flag. Prospective absent arms force the overall decision to inconclusive.

Matching semantics are explicit: candidate edges are screened with all temporal neighbours, then each cluster pair uses chronological greedy one-to-one exclusive matching. Primary matches require unique reciprocal maximum Jaccard and ≥0.5 retention in both directions. Lost and insufficient-fit baseline units stay in the full eligible-unit denominator. Split/merge burden is the fraction of eligible baseline units implicated by non-primary or multi-degree edges. Chance-aware coincidence uses the same marked-spike statistic for observation and deterministic per-cluster circular-shift null. These are spike-train engineering measures, not biological identity or purity.

## Implementation checks

- Done: candidate manifest/COMPLETE and every member verified -> pass.
- Done: exact real loader inputs hashed and structurally inspected -> all requested saved sorter/label/position files present; both QC archives absent.
- Done: actual reviewed CLI/adapter invoked from isolated byte-identical bindings -> fail-closed at REF384 clock mismatch, no report/output root.
- Done: v1→v2 adapter delta inspected -> mandatory provenance and two expected digests added before loader access.
- Done: matching/counting, synthetic exclusion and missing-state semantics inspected in executed source and tests -> behavior described above; 23 candidate tests pass under isolated bindings.
- Not done: scientific pair matching or endpoint computation -> blocked before normalized sort construction.
- Not done: truncation-QC validation/digest -> files do not exist at the required paths.
- Can establish: exact saved-array provenance; constant clock-frame mismatch; missing QC; fail-closed traversal; adapter import-binding risk; scoped semantics of code and fixtures.
- Cannot establish: correspondence edges, primary pairs, coverage, missingness, split/merge result, guardrails, scorecard verdict, biological identity/purity, or prospective-arm performance.
