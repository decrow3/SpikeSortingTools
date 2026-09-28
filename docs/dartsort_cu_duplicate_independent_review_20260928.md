# CU duplicate-physics independent review

## Verdict

The current merge-alignment step is not a material source of the saved close
interval population, but its sole W2 boundary-crossing case is physically
**compatible with a single duplicated event**. Rows 246684/246685 moved from a
7-sample pre-alignment lag to 9 samples after the applied -2-sample offset
difference. In the frozen fixed-time residual description, the second
component has amplitude 0 and reduces SSE by exactly 0.

That is suggestive, not proof. The two constituent reference split-half
cosines are 0.667 and 0.768, while the second event's own-reference cosine is
only 0.059. References and constituent identities are sorting-derived, and the
two close event windows overlap. CU therefore does **not** justify deleting
events, widening deduplication, changing production parameters, or calling any
individual pair a physical duplicate.

## Independent integrity review

| Item | Result |
| --- | --- |
| Frozen targets | 96: 32 same-source, 32 cross-source/zero-offset, 32 cross-source/nonzero-offset |
| Boundary case | sole originally-close 9--29 pair included |
| References | 1,500 unique rows across 79 constituents; max 20/constituent; all pass the saved +/-30-sample, +/-8-channel isolation proxy |
| Comparisons | 32, all 185--562 samples (5--20 ms) and matched to the target parent/constituent/state stratum |
| Voltage | 182 channels; 128 pair windows and 1,500 reference windows; 195,776,672 logical saved bytes |
| Window geometry | exactly 42 samples before the first event and 79 after the second; NaN-only padding outside valid support |
| Physical support | exactly 16 construction-template PTP channels per participating constituent; registered geometry and requested physical channel IDs match |
| Residual outputs | nonnegative amplitudes, nested two-event SSE, exact fractional-reduction algebra, exact support unions |
| Manifest | all 34 products and `COMPLETE.json` manifest hash pass |

The selection was frozen before waveform outcomes. The full eligible table is
exactly the 6,250 CS final 9--29-sample pairs, and the selected row/source-frame
identities close against the 29,999.759166666667 Hz W2 source clock.

The H5 analysis source itself is not in the published manifest. H1 therefore
independently validated its selection, saved-window geometry, channel/support
mapping, metric identities, summaries, and hashes, but cannot publish a hash of
the residual-operator implementation from this packet. The receipt records a
reporting-only path failure after extraction; finalization reused saved arrays
and did not reread voltage.

## Saved shift orientation and boundaries

The source fixture establishes the orientation: `shifts[reference, source]` is
subtracted from the source event time. For each pair H1 records:

`L_post = L_pre - (u_b - u_a)` and
`e_ab = s_ab - (u_b - u_a)`.

The identity holds for every reconstructed W2 boundary pair. The saved shift
matrix happens to be exactly antisymmetric, but this was checked rather than
assumed. Across 914 ordered within-component constituent pairs, `e_ab` is 0 for
888 and +/-1 for 13 each.

| W2 final lag | Pairs | Same / cross source | Nonzero applied difference | Pre-lag <= 7 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 227 | 115 / 112 | 6 | 3 |
| 9--29 | 6,250 | 2,822 / 3,428 | 61 | 1 |
| 30 | 265 | 132 / 133 | 7 | 0 |

The independently reconstructed 6,250 row-ID pairs exactly equal the published
candidate table. Thus the current alignment creates four previously-close
separations over the broader 8--30 neighborhood: three at exactly 8 and one at
9. This does not rule out pre-existing detector timing offsets that a different
alignment model might correct.

W3 provides a saved-array-only ready check, with no replay: all 571,935 rows
satisfy the signed timing identity. None of its 5,685 final 9--29 intervals came
from a pre-lag <=7; ten of 174 exact-8 intervals did. State is reported only
from the captured `fixed_state_episode` array; none was reconstructed.

## Physical evidence

The frozen descriptive operator gives:

| Group | n | Median 2-vs-1 SSE reduction | IQR | Median reference reliability | Median event/reference cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| same constituent | 32 | 0.0061 | 0.0012--0.0153 | 0.427 | 0.137 |
| cross, zero offset | 32 | 0.0164 | 0.0039--0.0376 | 0.433 | 0.172 |
| cross, nonzero offset | 32 | 0.0193 | 0.0012--0.0530 | 0.464 | 0.168 |
| 5--20 ms comparison | 32 | 0.0044 | 0.0006--0.0274 | 0.464 | 0.207 |

The improvements are generally modest and overlap heavily. Independent visual
review shows many targets with voltage structure at both markers, but the
reference matches are variable. The sole boundary case instead presents as a
single broad event across the two markers and receives no second fitted
component.

The comparison arm is context, not a calibrated null. Its windows are
systematically longer (306--683 samples versus 128--151 for targets), while the
fractional SSE denominator covers the full lag-dependent window. It cannot be
used as a direct null distribution for target-group effect sizes. No second
operator was introduced after seeing outcomes.

## Decision and relation to CR/CS

- CS rejects the current alignment-displacement mechanism as an explanation
  for most close intervals.
- CU makes the one actual boundary-crossing pair a focused
  single-event-compatible case, while leaving it unproven.
- CR found no reproducible count-controlled RF advantage for no-force splits.

The combined recommendation remains: retain the saved force-merged grouping at
the current boundary, preserve the close-pair evidence for review, and make no
automatic event-level intervention.

## Outputs and resource accounting

H5 evidence packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cu_duplicate_voltage_evidence_20260928/`

H1 independent packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cu_duplicate_independent_review_20260928/`

Local H1 packet:
`testing/outputs/cu_duplicate_independent_review_v1/`

H5 charged 120 conservative CPU-s and read 195,864,760 voltage bytes including
preflight, below the 512 MiB cap. H1 used no new raw read or GPU and charges 60
conservative CPU-s. CU therefore adds 180 s to the 16,733.22 s post-CS total,
for a cumulative 16,913.22 s, below the 19,300 s ceiling.
