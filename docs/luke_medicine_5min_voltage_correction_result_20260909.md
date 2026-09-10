# Five-minute MEDiCINe voltage-correction matrix result

**Interval:** Luke0804 imec0, 930--1230 s  
**Field:** frozen 6-sigma/full-screen MEDiCINe, 10,000 steps  
**Status:** seven sorts and first matched evaluation complete; development evidence only

## Result

No tested voltage correction clearly beats the unwarped 12/9 KS4 baseline.
The strict frozen lighthouse events are already almost completely detected in
every arm, so the useful endpoint is whether their events remain concentrated
in one sorted cluster without added contamination, refractory burden, or
duplicate assignment.

The apparent raw-count winner, rigid 0.25x p=2/sigma10, is not an identity
winner. It adds one KS-good unit (91 versus 90) but reduces macro single-cluster
recovery by 7.4 percentage points across all 17 lighthouses, raises the fraction
of events near multiple clusters by 4.5 points, and slightly worsens median
contamination and refractory burden.

Among the 13 lighthouses with at least 20 held-out strict events:

| Arm | Macro recovery | Single-cluster recovery | Change vs unwarped | Duplicate-event fraction | KS-good | Median contamination | Median refractory fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unwarped | 0.9931 | 0.6857 | -- | 0.1948 | 90 | 45.3% | 0.0181 |
| Rigid 0.25x, p2/sigma10 | 0.9934 | 0.6272 | -0.0585 | 0.2408 | 91 | 46.0% | 0.0205 |
| Rigid 1x, p2/sigma10 | 0.9994 | 0.6743 | -0.0114 | 0.2047 | 77 | 61.9% | 0.0272 |
| Nonrigid 0.25x, p2/sigma10 | 0.9934 | 0.6936 | +0.0080 | 0.1990 | 85 | 49.8% | 0.0193 |
| Nonrigid 1x, p2/sigma10 | 0.9983 | 0.6918 | +0.0061 | 0.2175 | 85 | 49.7% | 0.0249 |
| Nonrigid 1x, p2/sigma20 | 0.9983 | 0.6977 | +0.0120 | 0.2097 | 79 | 56.5% | 0.0237 |
| Historical nonrigid 1x, p1/sigma20 | 0.9983 | 0.7258 | +0.0401 | 0.2267 | 57 | 63.3% | 0.0303 |

The historical p1/sigma20 arm illustrates why lighthouse concentration cannot
be used alone: it improves that scalar average while losing 33 good units and
substantially worsening contamination, refractory burden, and duplicate
assignment. Its apparent benefit is mixed across cells (five improve, six
worsen, two tie), rather than a replicated population-wide rescue.

The best balanced corrected arm is nonrigid 0.25x p=2/sigma10, but its advantage
is small and heterogeneous: six of 13 supported lighthouses improve, three
worsen, and four tie. It loses five good units and slightly worsens global
quality metrics. This is insufficient to replace the unwarped baseline.

## Evaluation contract

- Primary events are the previously frozen `strict_accepted` waveform-only
  lighthouse matches after the 930--940 s training interval: 2,228 events from
  17 candidates. Lower-score and identity-ambiguous events do not enter the
  primary result.
- Events match sorted spikes within 0.5 ms and 100 um. The unwarped arm uses
  observed waveform-centroid depth. Each corrected arm uses observed depth
  minus exactly the displacement applied to that arm. Matching is one-to-one
  within each lighthouse and does not use sorter labels to define identity.
- Sensitivity checks at 0.25/0.5 ms and 60/80/100/120 um preserve the ordering
  and qualitative tradeoffs.
- Near-complete event recovery is not evidence of continuous identity. It is
  saturated here and cannot distinguish the candidates.

## Decision

Retain the unwarped voltage baseline. Do not advance rigid 0.25x based on its
raw unit count. Preserve nonrigid 0.25x p=2/sigma10 as the least harmful
voltage-correction candidate, but require a separate interval or field and
replication across lighthouses before another long sort. The current result
does not justify tuning more kernels or gains on the same five-minute outcome.

Artifacts are in `testing/outputs/luke_medicine_5min_matrix_analysis_v1/`.
The seven materialized recordings and sorts are in
`/media/huklab/Data/luke_medicine_5min_matrix_v1/`.
