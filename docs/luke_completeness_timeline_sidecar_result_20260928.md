# Time-resolved completeness sidecar

Date: 2026-09-28. Status: **implemented and validated on the completed Luke
imec0 rescue output.**

## Pipeline change

The downstream pipeline now writes an independent, identity-bound
`amplitude-completeness-timeline-v2` stage after cached amplitude QC. It reads
no voltage and changes no spike or label. Its four artifacts are:

- `amplitude_completeness_timeline.csv`: one physical-time row per cached fit;
- `amplitude_completeness_units.csv`: every curated unit, including units with
  no supported fit window;
- `amplitude_completeness_policy.json`: machine-readable interpretation and
  caveats;
- `amplitude_completeness_summary.json`: counts tied to the sort identity.

Each window is classified as `measured`, `censored_at_least_50pct`,
`poor_fit_disagreement`, or `nonfinite`. A deterioration nomination requires a
trustworthy measured window at least 10 percentage points above the unit's
10th-percentile trustworthy baseline. The nomination is explicitly
`screening_only`: it cannot alter labels, exclude an interval, or claim lost
spikes.

The sidecar preserves the historical QC convention transparently. Stored
window endpoints span 1,000 spikes, while the historical fitter consumed the
stop-exclusive 999-amplitude slice. Both counts are exported. A missing row
means no supported fit, never zero missingness or reliable detection.

## Full-session Luke result

The completed rescue sort identity is
`22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`.
The v2 stage receipt is complete with request digest
`e93b538ede47a5099eec6c22ad18c248f988a42c6b61a3e181a34c0e9dc1c540`.

| measure | count | fraction of fitted windows |
|---|---:|---:|
| curated units | 710 | -- |
| units without a supported window | 396 | -- |
| fitted windows | 26,227 | 100.0% |
| trustworthy measured | 9,204 | 35.1% |
| boundary-censored | 14,394 | 54.9% |
| poor estimator agreement | 2,629 | 10.0% |
| nonfinite | 0 | 0.0% |
| deterioration nominations | 3,373 | 12.9% |

The known cases behave as required. Cluster 553's two independently confirmed
failing windows are nominated at +24.24 and +44.39 points over baseline.
Cluster 21's confirmed failing windows are nominated at +14.39 and +25.33
points. Cluster 452 also receives nominations despite its independent anchor
showing only about 2.2 points of loss. This is direct validation that the
timeline is useful for triage and unsafe as an automatic reliability mask.

The initial v1 development output omitted curated units with no fit window and
is superseded. V2 lists all 710 units and reports 396 explicitly unsupported
units.

## Verification

Known-answer tests cover time mapping, historical 999/1,000 semantics,
censoring, fit disagreement, screening, explicit unsupported units, atomic
artifact writing, identity-bound receipts, and exact reuse. Development-arm
tests cover additive manifest migration. The relevant suite passes 23 tests in
the base environment. Imports of the new stage plus the existing Kilosort and
PyTorch-dependent production modules succeed in the locked production runtime.

No sort was launched and no production output was overwritten. The executed
validation artifacts remain under
`testing/outputs/luke_rescue_completeness_timeline_v2/`.
