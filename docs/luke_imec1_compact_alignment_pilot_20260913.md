# Luke0804 imec1 compact alignment and identity-evidence pilot

## Decision

Retain compact-core proposal templates. Do not add integer member alignment
before averaging by default: it did not improve the frozen qualification result.
Separated-time reproduction is more discriminating than odd/even reproduction
and should be retained in the next proposal builder.

## Design

All three arms used the same 47,013 detections from 310--320 s and competed
against the same complete external 36.46--56.46 s bank:

1. broad proposal templates;
2. compact-core templates averaged without applying member lags;
3. the same compact members shifted to a training-only reference before averaging.

Template construction used 300--310 s. Core templates were constructed
independently in odd/even partitions and in separated 300--305 versus 305--310 s
blocks. Winning and runner-up identity, real/decoy kind, score, margin, gain,
detector phase, and lag were retained for every qualification event. Sorter
labels, motion estimates, and absolute depth were excluded from construction and
identity scoring. Depth centroids were computed only after matching.

## Results

| arm | all strict | s300 strict | external strict | decoy winners |
|---|---:|---:|---:|---:|
| broad | 458 | 180 | 278 | 8,713 |
| compact | 520 | 249 | 271 | 8,671 |
| compact + alignment | 511 | 247 | 264 | 8,677 |

Compact selection added a net 62 strict matches over broad templates. Applying
member alignment to compact templates lost a net 9 strict matches relative to
unaligned compact templates. The alignment comparison added 25 strict events
but lost 34: all changes were score or margin crossings, with no strict event
reassigned to another strict identity.

For broad versus compact, 106 events became strict and 44 ceased to be strict.
The gains were 47 margin crossings, 16 score crossings, 6 decoy-to-real changes,
and 37 changes from a different non-strict or ambiguous winner. There were zero
strict-to-different-strict identity reassignments. This supports cleaner identity
separation, while also showing that much of the numerical gain is threshold
proximal.

## Core reproducibility

All 20 valid odd/even cores produced independently constructed compact templates
with cosine >=0.90 (median 0.985, minimum 0.947). Sixteen had sufficient support
in both separated time blocks; 15/16 reproduced above 0.90 (median 0.976).
`p08_f033` was the exception, with time-block template cosine 0.851 and strongly
asymmetric reciprocal evidence despite an odd/even cosine of 0.974. It remains a
useful proposal lead but should not be treated as temporally established.

The clearest compact gains were `p08_f025` (0 to 41 strict matches), `p08_f029`
(0 to 21), and `p08_f033` (24 to 37). Alignment reduced `p08_f025` to 34, left
`p08_f029` at 21, and raised `p08_f033` to 39. Thus the aligned arm does not
provide a consistent candidate-level benefit.

Training proposal rates, training core rates, and qualification match rates are
reported separately. No ratio between training and qualification windows is
called a recovery probability.

## Integrity checks

Each arm has exactly 47,013 unique event IDs, no duplicate keys, no missing
required or scoring values, and zero time/phase/channel key mismatches across
arms. The enhanced scorer exactly reproduced v3 winner, score, margin, and gain
on a cached-template check before the raw run.

## Remaining limitation and next cheapest test

The original v3 proposal assignments saw all of 300--320 s, so proposal discovery
is not independent of qualification even though template construction is. No
candidate is promoted to a lighthouse identity from this result alone.

The next cheapest test is residual-core discovery inside the broad proposals,
starting with `p08_f025` and `p08_f029`: remove the first compact core, permit at
most one additional reciprocal core with a prespecified minimum size, and make
both compete globally while leaving residual events unassigned. `p08_f033`
should be carried as a temporal-reproduction failure. Wider channel support is
not yet justified; first inspect whether nearest-rival waveform differences are
truncated on the current common channels. Multiple genuinely independent seed
windows should follow once the proposal rule is frozen.

## Reproducibility

The managed job is `testing/outputs/luke_imec1_compact_alignment_pilot_v2_job`.
It completed with process return code 0 and systemd service result `success`.
The corrected event audit is
`testing/outputs/luke_imec1_compact_alignment_audit_v2`.
