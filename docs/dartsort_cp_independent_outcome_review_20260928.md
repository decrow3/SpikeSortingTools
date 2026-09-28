# CP independent outcome review and W2 RF accounting

## Verdict

**Retain force-merged D2L grouping as the current default at this saved
grouping boundary.** Removing force edges mostly partitions already assigned
parent event trains into more labels; it does not provide evidence for recovery
of many additional useful neurons. This conclusion is bounded to the saved W2
and W3 D2L states. Static versus D2L remains contextual because those arms
differ upstream.

The H5 corrected packet is ready for use within that scope. Its 112-product
completion marker is present. The source snapshots match the code-review
receipt, all corrected-table manifest entries match, and an independent H1
recalculation passes the unpadded episode-domain, common-support shifted-null,
exclusive-control, and `z_abs` invariants.

## Direct saved-state result

| Window | All-force assigned / units | No-force assigned / units | Net assigned rows | Substantial split parents |
| --- | ---: | ---: | ---: | ---: |
| W2 | 635,515 / 552 | 637,068 / 731 | +1,553 (+0.24%) | 88 |
| W3 | 565,063 / 488 | 566,629 / 589 | +1,566 (+0.28%) | 60 |

Across 148 substantial exact-lineage split families, median assignment
retention is 1.000 and the top two children carry a median 0.986 of assigned
parent rows. The top two children usually occupy the same behavioral periods:
their median absolute difference in actual, unpadded episode fraction is 0.0087
in W2 and 0.0304 in W3. Exclusive near-coincidences are not enriched over the
like-for-like shifted control (pooled median 0.00226 observed versus 0.00451
shifted). Thus additional labels must not be reported as additional neurons.

The old depth selection used localization column 1 (`y`) and is superseded.
The corrected selection uses column 2 (`z_abs`) directly, with no channel-depth
offset. It remains template-conditioned display support, not identity evidence.
The old shifted null also used unequal supports and is superseded; independent
W2 recomputation at 0 and +/-0.5 s exactly reproduces the corrected supports,
denominators, and maximum-cardinality match counts.

## Frozen W2 development RF endpoint

The comparison uses the accepted dots stimulus and gaze reconstruction, the
explicit imec1-to-NIDAQ clock, the original 81-trial split, and only the 12
complete W2 development trials. The five overlapping outer-holdout trials were
not evaluated. The accepted gaze receipt independently passes all six checks:
CSV hash, exact 61-trial development training set, holdout exclusion, excluded
set coverage, no holdout use in neural calibration, and no outer-holdout
evaluation. Inner RF scores remain conditional on the one shared gaze
calibration fitted to all 61 development trials.

| Arm | Labels | RF eligible | Eligible fraction | Median CV SNR |
| --- | ---: | ---: | ---: | ---: |
| Static | 538 | 67 | 12.5% | 0.388 |
| All-force | 552 | 77 | 13.9% | 0.290 |
| No-force | 731 | 55 | 7.5% | -0.052 |

Eligibility is frozen at at least 500 development spikes total and at least 200
in each inner fold. Every unit, both fold counts, and every exact nonnoise
all-force/no-force parent-child relation are retained in the accounting tables.

- All 731 no-force units have positive exact lineage from an all-force parent;
  none is formed only from all-force noise rows.
- Every one of the 55 eligible no-force units has an eligible all-force parent
  as its dominant row source. There is no eligible child whose dominant parent
  was ineligible.
- Of 77 eligible all-force parents, 61 have any eligible child, 16 have none,
  and 51 have an eligible dominant child.
- In those 51 comparable parent/dominant-child families, the median RF change
  is zero: 10 child wins, 23 losses, and 18 exact ties; mean change is -0.167.
- The 30 three-arm reciprocal candidate triplets give 9 no-force wins, 12
  losses, and 9 ties against all-force, with median change zero. These
  high-ambiguity timing links are candidate correspondences, not identities.
- Broad frozen spike-count bins do not reveal a hidden no-force advantage. They
  are descriptive only and were not used to tune a threshold.

An RF-map example was not reconstructed: the compact endpoint did not save STA
maps, and a selected example is optional and not decision-critical. This does
not weaken the complete eligibility, fold-count, and family accounting.

## Interpretation and limits

No-force sharply lowers short-ISI fractions, but splitting mechanically does
that, so it is not positive evidence for the split. Its all-neighbor duplicate
candidate burden is higher at the 0.10 threshold in both episode and rest time.
The representative exact-parent voltage panels are exploratory: child-pair
waveform similarity is not high enough to prove a universal merge identity and
does not establish that every force edge is biologically correct.

The corrected primary episode results use catalogue intervals exactly as saved,
without padding. The older `ISI_CONTIGUOUS_SEGMENTS.csv` uses the superseded
padded-AS state definition; it is retained only as descriptive evidence and is
not used for the verdict. Accepted and unresolved episodes, padded AS, and the
canonical hub mask remain separately reported.

## One next decisive procedure

If the force policy is revisited, use a precommitted **family-level independent
identity test** on the existing saved/cached rows: trace every substantial
parent's dominant children through depth using waveform geometry independent of
the evaluated merge, preserve rivals and unmatched events, and score temporal
replication on held-out blocks. Do not select examples by RF outcome, do not use
absolute depth as identity evidence, and do not rerun a sorter. This directly
tests whether any no-force children are reproducible distinct cells; another
aggregate label-count or ISI comparison cannot answer that question.

## Outputs

- H5 packet: `/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-cp-direct-outcomes-v1/`
- H1 packet: `/mnt/NPX/Luke/DARTsort_motion_experiments/cp_independent_outcome_review_20260928/`
- Local H1 working packet: `testing/outputs/cp_h1_independent_outcome_review_v1/`

No sort, motion fit, calibration, GPU task, or source-voltage read was run on H1.
