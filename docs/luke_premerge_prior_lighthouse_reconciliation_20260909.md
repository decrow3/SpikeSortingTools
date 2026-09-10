# Reconciliation of the 17 prior lighthouse candidates with the Kilosort premerge screen

## Headline

The prior lighthouse candidates were not lost by Kilosort detection. All 742 of
742 frozen seed spikes from the 17 prior candidate labels have an exact-sample
counterpart in the cached Kilosort premerge detections.

They disappeared from the reported result because the premerge screen was a
**multi-template family finder**, not a census of known lighthouse candidates.
At cosine 0.97, 15 of the 17 prior candidate labels are represented primarily by
a premerge cluster that has no edge above threshold. Those clusters were retained
in the waveform inventory but omitted from the family plots as singletons.

![Reconciliation buckets](../testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/01_reconciliation_buckets.png)

## Where the 17 labels went

| Outcome at cosine 0.97 | Prior labels | Interpretation |
| --- | ---: | --- |
| Dominant premerge cluster is a singleton | 15 | Present and eligible, but not emitted by the family-only screen |
| Dominant cluster is in a coherent family | 1: unit 705 | The 267/272 pair is spatially stationary, so it is a likely duplicate/split rather than a moving lighthouse family |
| Dominant cluster is in an implausible family | 1: unit 632 | It entered the large pathological component and failed the depth/time plausibility check |

The 17 labels map to only 16 distinct dominant premerge clusters because prior
labels 673 and 675 both map primarily to premerge cluster 335. This agrees with
the earlier warning that the 17 candidates are not 17 demonstrated-independent
biological cells.

The new moving premerge family `c097_F009` is not a straightforward recovery of
one of the 17 old identities. Unit 657 has the best posthoc motion correlation,
but its seed spikes map to clusters 321, 320, and 317, not to `c097_F009`.
Unit 555 contributes four seed spikes to family member 251, but its dominant
cluster is 243. The overlap is therefore weak and ambiguous.

## Method

For every frozen seed frame in `templates.npz`, the audit searched the cached
premerge detection times within nine samples. All seed matches were exact in
time. When Kilosort supplied more than one detection at the same sample, prior
seed depth selected the closest one solely for this retrospective reconciliation;
depth was not used to construct the waveform families.

The mapped dominant cluster was then checked against the exact eligibility list
and the 0.95 and 0.97 family memberships from the premerge screen. Strict prior
lighthouse events were also mapped as a secondary diagnostic.

## What this establishes

- There was no loss of the old candidates at Kilosort's detection stage.
- There was no minimum-spike or waveform-inventory exclusion of their dominant clusters.
- The apparent loss was caused mainly by an output-definition error: singleton clusters were not shown.
- Stopping before Kilosort's final merge does not, by itself, trace a singleton identity through depth. A separate whole-probe matcher is still needed to decide whether a singleton has displaced waveform counterparts.

## Limits

- The original 17 labels are provisional candidates, not 17 proven-independent cells.
- Exact seed-event recovery shows detection survival, not correct identity assignment outside the seed interval.
- Prior depth was used to resolve same-time duplicate detections, so this audit is a reconciliation rather than a new depth-blind discovery result.
- Several old strict tracks contain biologically implausible excursions; agreement with them is not a sufficient validation target.

## Cheapest correction

Keep the current family discovery unchanged, but add a second output containing
every eligible singleton and explicitly seed-trace the 17 known candidates into
those clusters. This makes the screen exhaustive without lowering the 0.97 edge
threshold or inventing new merges. Only after that bookkeeping fix should the
singleton templates be searched across the whole probe for displaced matches.

## Do the 16 distinct representations move together?

Only weakly. Using 5-second median observed depths from 930--1030 s and centering
each of the 16 distinct dominant premerge clusters independently, the median of
all 120 pairwise Pearson correlations is 0.023; 50.8% are positive. That is not
direct evidence of broad cell-to-cell agreement.

A robust leave-one-out consensus is mildly more encouraging: the median
correlation of each cluster with the median of the other 15 is 0.228, 11/16 are
positive, and 7/16 exceed 0.3. Circularly shifting each cluster's binned track
independently gives an upper-tail p-value of 0.016 for that statistic (500 fixed-
seed draws). The consensus excursion is only 4.83 um peak-to-peak and is not
unusual under the same null (p=0.106).

Thus there may be a small shared component, but these static-cluster centroids do
not recover a convincing common motion field. This is expected to be a difficult
readout: Kilosort clusters are spatially localized and can hand an identity to a
different cluster as it moves.

![Shared movement agreement](../testing/outputs/luke_premerge_lighthouse_agreement_v1/01_shared_movement_agreement.png)

## Reproducible files

- `testing/luke_premerge_prior_lighthouse_reconciliation_v1.py`
- `testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/prior_unit_reconciliation.csv`
- `testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/prior_seed_cluster_mapping.csv`
- `testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/prior_event_to_premerge_mapping.csv`
- `testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/02_prior_unit_premerge_tracks.pdf`
- `testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1/summary.json`
