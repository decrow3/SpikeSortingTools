# Full-session external-nonrigid comparison — 2026-09-28

## Decision

Do not promote the tested full-session external-nonrigid MEDiCINe correction.
Keep rescue 12/9 with motion disabled as the operational reference. The
candidate fails prespecified contamination and boundary guardrails, produces
far fewer curated and KS-good units, and leaves the amplitude-completeness
endpoint infeasible. This result closes this exact field/application/sorter
combination; it does not establish that motion correction is generally harmful.

No spike sort was launched for this review. The already-completed 2026-09-09
candidate sort was resumed only through the common curation and QC path. Its
source sort identity is
`8909d5bda865b320b258b1f3dcab048a1b7507ee59b4eb384a6227cbcd1849af`.

## Comparable outputs

Both arms use thresholds 12/9, internal Kilosort correction disabled, and the
legacy-compatible cosine 0.90 / CCG 0.5 curation profile. The candidate applies
the frozen external nonrigid field before sorting and uses the common supported
AP20--AP365 domain (200--3640 um). The paired evaluator restricts both arms to
that processing range and excludes 100 um at each boundary for its interior
cohort. This channel restriction remains a comparison limitation.

| Result | Motion-off reference | External nonrigid |
|---|---:|---:|
| Curated units | 710 | 526 |
| KS-good curated units | 301 | 129 |
| Curated spikes | 29,227,829 | 26,550,885 |
| Standard-QC units with any lab warning | 689 | 518 |
| Units with legacy truncation windows | 314 | 313 |

The candidate has 9.16% fewer curated spikes, 25.9% fewer curated units, and
57.1% fewer KS-good units. Counts alone do not identify neurons or prove loss,
so the decision also uses exclusive correspondence, amplitude support, and
population guardrails.

## Paired evidence

The full correspondence graph contains 16,378 edges but only 72 reciprocal
primary matches; 65 are interior in both arms. Median interior-primary Jaccard
is 0.579, with median baseline retention 0.799 and candidate retention 0.722.
The graph is highly ambiguous: 617 baseline units and 503 candidate units have
multiple edges. These are spike-train correspondences, not verified biological
identities.

Only 20 of 586 baseline-eligible interior units have two supported fits in both
arms over at least half the session. Coverage is therefore 3.41%, far below the
frozen 50% floor. The endpoint is `infeasible_insufficient_coverage`. Within
that selected measurable subset, median baseline-minus-candidate missingness is
-2.55 percentage points, which points away from improvement but is not promoted
to a population efficacy estimate.

Three available prospective guardrails regress beyond their allowed 0.01:

| Guardrail | Reference | Candidate | Candidate - reference | Limit | Result |
|---|---:|---:|---:|---:|---|
| Median refractory fraction below 1.5 ms | 0.00517 | 0.01892 | +0.01375 | +0.01 | fail |
| Chance-aware near-coincident excess | 0.14837 | 0.15939 | +0.01101 | +0.01 | fail |
| Edge-unit fraction | 0.05493 | 0.06654 | +0.01161 | +0.01 | fail |

Candidate edge-spike fraction improves by 0.00286, but one favorable guardrail
does not cancel three failures. Sliding-RP, nearest-neighbor, SD-ratio and
noise-cutoff fields were unavailable and remain explicit gaps.

## Time-resolved completeness sidecar

The candidate's identity-bound screening sidecar lists 526 units and 23,379
windows. It contains 5,622 measured windows, 16,060 boundary-censored windows,
1,697 poor-fit windows and no nonfinite windows; 213 units have no supported
window. Its 2,600 deterioration nominations remain screening flags only. They
do not repair the infeasible paired endpoint or alter curation.

## Execution and recovery record

The first downstream service was killed by `systemd-oomd` while materializing
large curated feature arrays. The journal records user-manager pressure at
72.07%, above its 50% threshold for more than 20 seconds, while the host still
had substantial physical memory. The partial output and logs are retained under
`/media/huklaban5/Data/luke_improved_motion_20260909_v1/downstream/failed_attempts/v1_oomd_20260928`.
After investigation, v3 reused the proven persistent user systemd manager with
`ManagedOOMPreference=avoid` and completed all receipt-backed stages. No sorter
was restarted.

The first comparison attempt exposed quadratic per-unit full-array scans in the
generic evaluator and was stopped with its evidence retained. The evaluator now
groups spikes once in stable unit order for unit metrics and the shift null.
All nine generic comparison tests pass, implementation hashes are request-bound,
and the optimized comparison completed in about two minutes.

Primary artifact roots:

- downstream summary:
  `/media/huklaban5/Data/luke_improved_motion_20260909_v1/downstream/summary.json`
- paired comparison:
  `/media/huklaban5/Data/luke_improved_motion_20260909_v1/downstream/comparison_vs_rescue/`
- frozen configs:
  `configs/luke_improved_nonrigid_downstream.v1.json` and
  `configs/luke_improved_nonrigid_comparison.v1.json`

The cheaper conclusion is already decisive: this candidate has no supported
efficacy endpoint and fails multiple guardrails. Do not spend on another sort
using this unchanged field and application. A future motion candidate needs a
qualified field and a feasible identity/coverage plan before sorting.
