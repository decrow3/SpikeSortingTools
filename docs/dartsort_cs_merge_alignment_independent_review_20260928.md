# CS merge-alignment independent review

## Verdict

**Merge-time template alignment does not materially explain the saved CE W2
9--29-sample intervals.** Of 6,250 final consecutive intervals in that range,
only one began within the executed dedup radius and was moved beyond it by
alignment. The other 6,249 were already separated before alignment. This is a
stage-lineage conclusion, not evidence that the intervals are biological
spikes or physical duplicates.

The H5 packet is complete and its manifest and all 16 product hashes validate.
The scientific counts and signed timing result pass independent checks. The
independent review did find a seven-row provenance annotation defect: bridged
dedup drops were assigned an ultimate survivor 8--9 samples away and called
unique direct winners, despite the 7-sample radius. The corrected rows are in
`CS_DEDUP_RELATION_CORRECTION.csv`; they must be interpreted as partner unknown.
This correction does not change any event, interval, route, state, or mechanism
count.

## Independent checks

| Check | Result |
| --- | --- |
| Immutable event identity | 641,588 dense, unique source rows |
| Signed timing identity | `t_post = t_pre - applied_offset` for every row |
| Exact saved replay | times, labels, channels, merge map, candidates, likelihoods and responsibilities match |
| Executed sampling rate | 29,999.759166666667 Hz |
| Dedup conversion | 0.25 ms = 7 samples, inclusive |
| Score aggregation | 3,923 reassignments; no assigned/noise-state changes |
| Dedup closure | 6,072 dropped; 635,515 retained/final |
| Direct dedup relations | 6,062 unique; 3 ambiguous; 7 bridged/unknown |
| Candidate envelope | all 6,250 final 9--29 intervals covered; pre-lag envelope 32 samples |
| Candidate formula | predicted and observed post-alignment lag agree for all 14,040 rows |
| Null offset-grid completeness | all 9,192 published groups have all eight frozen offsets |
| Null exposure denominator | **not valid for rate comparison**: products of counts pooled across segments were used instead of sums of within-segment products |
| Physical waveform evidence | unavailable: existing compatible cache contains neither row of the 61 target pairs |

The real source behavior agrees with the replay. `apply_reclustering` chooses
the first maximum-SNR constituent and applies the signed subtraction at
`cluster_util.py:91--99`. Merged candidate scores are stably sorted at
`agglomerate.py:835--871`. Dedup selects `merged_log_liks[:, 0]` first, stably
sorts unsorted event times, and applies its radius inclusively at
`agglomerate.py:936--987`. Milliseconds are rounded by
`floor(samples + 0.5)` at `internal_config.py:57--60`, yielding seven samples
at the saved clock. The dedup loop can bridge across a run while eliminating
lower-scored events (`agglomerate.py:1005--1038`), which is why the seven
ultimate-survivor annotations cannot be treated as direct partners.

## Mechanism accounting

The applied constituent offsets span -1 to +2 samples, so the maximum pairwise
offset difference is only 3 samples. Among 634,963 final consecutive
intervals, 6,250 (0.984%) are 9--29 samples. Their routes are:

| Route | Pairs |
| --- | ---: |
| same source constituent | 2,822 |
| expanded force | 1,685 |
| direct force | 1,536 |
| post-score new coassignment | 175 |
| transitive only | 20 |
| direct force plus QDA overlap | 9 |
| QDA accept | 3 |

Only 61 of the 6,250 have nonzero offset differences, and only one is the
specific proposed route: originally within seven samples, then displaced into
9--29 samples. By state, 6,074 occur in rest, 170 in accepted episodes, and 6
in unresolved time. The all-near-pair denominator (14,040 candidate pairs) is
kept separate from the consecutive-ISI denominator (13,441 pre-dedup and
634,963 final consecutive pairs).

The fixed nonwrapping null contains 73,536 rows: all 9,192 published groups at
-4, -2, -1, -0.5, +0.5, +1, +2 and +4 seconds. Its `possible_*_pairs`
denominators multiply event counts pooled across state segments rather than
summing within-segment count products. That admits impossible cross-segment
exposure and invalidates normalized rate comparisons. The null is also a
**pre-dedup association null**, not a dedup-survival null. Its raw table is
therefore retained only as a qualified sensitivity product; it is not used in
the mechanism verdict.

The analysis source used nominal 900 s as W2's origin and rounded catalogue
boundaries rather than using exact source-frame membership. The saved manifest
gives start frame 26,999,783 and 29,999.759166666667 Hz, so the exact origin is
899.9999916666 s (0.25 sample earlier). A direct reclassification of all
641,588 saved events found zero state changes, and the 6,250 target pairs retain
the reported 6,074 rest / 170 accepted / 6 unresolved split. Thus the
implementation should still be fixed for reuse, but this tiny origin error did
not change the current result. The null products do not expose per-state
available duration or unavailable short segments, so those denominators cannot
be independently certified from the packet.

Route labels and deterministic lag agreement do not establish neuronal
identity.

## Boundary audit

The published candidate envelope closes exactly: reconstructing final
within-unit consecutive intervals directly from the event ledger gives the same
6,250 row-ID pairs as the candidate table, with 3,428 cross-constituent and
2,822 same-constituent pairs. The sole originally-close target is rows 246684
and 246685: pre-lag 7, offset difference -2, and post-lag 9 samples.

The excluded boundary values were checked separately:

| Final lag | Pairs | Same / cross constituent | Nonzero offset difference | Originally within 7 samples |
| ---: | ---: | ---: | ---: | ---: |
| 8 samples | 227 | 115 / 112 | 6 | 3 |
| 30 samples | 265 | 132 / 133 | 7 | 0 |

Thus alignment moved three previously-close pairs to exactly 8 samples and one
to 9 samples. Even for the broader 8--30-sample neighborhood, the current
alignment step creates only four previously-close separations. This statement
is deliberately narrow: it rejects the current merge-alignment step as the
source of most intervals, but does **not** rule out pre-existing detector timing
offsets that a different alignment model might correct.

The alignment-only counterfactual was correctly not triggered. With an offset
span of only three samples it cannot materially remove this 9--29-sample
population, and widening the production dedup radius was not authorized.

## Integration with CR

CR found no reproducible count-controlled RF advantage for the no-force split:
25 of 176 child comparisons qualified, only four families had both children,
and the median child-minus-parent CV SNR changed sign across seeds (-0.363,
-0.465, +0.211). Those RF scores remain too support-limited and are not
duplicate-aware neuronal-yield measures. Together, CR and CS support retaining
the force-merged grouping at the saved boundary, while leaving individual
merge identity unresolved.

## Outputs and limits

H5 packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cs_merge_alignment_lineage_20260928/`

Independent review packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cs_merge_alignment_independent_review_20260928/`

Local audit:
`testing/outputs/cs_merge_alignment_independent_review_v1/`

No GPU, sort, calibration, raw-voltage read, matching, template construction,
or outer holdout was used. The compatible existing cache was inspected before
waveform evidence was considered; it contained neither row of the 61
nonzero-offset targets, so no waveform claim was made.

H5's authoritative conservative CS charge is 450 s. H1's bounded source/product
audit consumed less than 30 CPU-s. The combined CS charge is therefore 480 s,
bringing the running cumulative charge from 16,253.22 to 16,733.22 s before CU.
