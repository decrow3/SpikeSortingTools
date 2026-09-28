# CO exposure qualification and fixed-unit 400-waveform result

## Verdict

**Neither unresolved-edge default is selected for production, and neither CO
unit passes the new larger-sample primary rule.** The frozen pass-only gate is
preserved as an experimental definition, not recommended as a production
winner. CN's full-session counts are conditional support screens only.

CO materially increases the point reproducibility of both predetermined units,
but the prespecified 400-event median-template lower 95% bounds remain below
0.90: 0.840 for unit 23 and 0.892 for unit 120. Therefore the result is a finite
0/2 failure. CK remains 0/8 and unchanged. Rest reproducibility does not
establish neuronal identity, episode continuity, or residual-field accuracy.

## Qualified CN exposure

The AP metadata independently gives 314,204,094 frames at 29,999.7591667 Hz,
or 10,473.553879 s. The hash-verified canonical mask contains 422 merged
intervals covering 2,263.5 s, leaving 8,210.053879 s rest. Tiling each rest
segment separately gives 1,442 complete 5 s blocks; 1,382 blocks in 249
multi-block segments are structurally eligible for the gate. They cover
6,909.999939 s before and 6,901.800006 s after the 89-sample symmetric trim.

| Rate scenario | W2 independent / shoulder | W3 independent / shoulder |
| --- | ---: | ---: |
| 0.5x | 108 / 142 | 65 / 78 |
| 1.0x | 180 / 166 | 98 / 85 |
| 2.0x | 195 / 173 | 99 / 88 |

These counts apply common-positive blocks and the two-block-within-segment rule
in expectation under stationary independent Poisson rates. Relative to CN's 1x
all-rest screen, five W2 and one W3 independent-rate edges drop; exact shoulder
exposure adds one edge in each window. The result still is not a probability of
realizing all minima, gate power, gate passage, identity, recovery or yield.

The state-complementary table is also exposure-sensitive. In the frozen primary
822/824 pairs have fewer than 20 spikes in at least one named child; the counts
are 833/835 for no-force and 122/122 for accept-unresolved. These are pairwise
cross-products within 36, 38 and eight all-force parents under unequal state
exposure—not recovered or lost biological fragments.

## CO selection integrity

The h5 selection was sealed before voltage access. It contains the two
predetermined CE-W2 units only, with 1,600 unique rows and zero overlap with all
1,600 old CK scoring rows. Each unit/half has exactly 400 rows and a nested 100;
the nested subsets span all 28–34 selected 5 s blocks. Both saved CK noise
arrays match byte-for-byte and each unit keeps the same 30-channel physical
support. The raw pass was one sequential reader and cached 140,940,928 bytes;
GPU use was zero.

| Unit | Eligible pools after CK exclusion, h1 / h2 | Selected blocks, h1 / h2 |
| --- | ---: | ---: |
| 23 | 1,028 / 1,057 | 29 / 34 |
| 120 | 467 / 522 | 28 / 34 |

## Reproducibility result

| Unit | Old CK 100 point / lower | CO nested 100 point / lower | CO 400 median point / 95% CI | CO 400 mean point / 95% CI |
| --- | ---: | ---: | ---: | ---: |
| 23 | 0.767 / 0.573 | 0.758 / 0.559 | 0.927 / [0.840, 0.887] | 0.945 / [0.881, 0.917] |
| 120 | 0.839 / 0.685 | 0.835 / 0.673 | 0.950 / [0.892, 0.924] | 0.963 / [0.920, 0.946] |

The mean result for unit 120 clears 0.90 descriptively, but mean was explicitly
secondary and cannot replace the frozen median rule. The nested-100 results are
close to old CK despite disjoint rows. Four support-first point/first-draw checks
are array-exact. Independent h1 recomputation from saved conditioned templates
matches every point to at most `1.1e-16`; a separately recomputed unit-120,
400-event first bootstrap draw is 0.9105880761 in both copies (absolute error
`1.1e-16`, 67.9 MB saved-cache read).

The physical-channel panel is internally legible and shows broad improvement at
400 events, but not uniform stability: median channelwise temporal cosine at
400 events is 0.732 for unit 23 and 0.933 for unit 120. CO and CK use different,
disjoint time/block compositions, so the change cannot be attributed purely to
sample count. The 1,000 block-bootstrap draws quantify resampling of these
observed blocks; they are not 1,000 independent 400-event experiments.

## Episode inventory and next action

| Unit | Episode events h1 / h2 | Collision-free h1 / h2 | Episode blocks h1 / h2 |
| --- | ---: | ---: | ---: |
| 23 | 119 / 45 | 109 / 41 | 18 / 6 |
| 120 | 545 / 180 | 313 / 98 | 19 / 7 |

Episode events exist for the same fixed IDs, but neither unit passed the rest
qualification and half-2 episode support is limited. Do not advance to an
episode identity/residual test yet. The smallest next diagnostic is saved-cache
only: for units 23 and 120, plot fixed-support per-block median templates and
leave-one-block-out correlations on the existing 400-event CO cache. That can
show whether a few blocks/channels drive the low bound versus broad finite-
sample noise, without selecting new units, reading voltage again, fitting a
field, or changing the threshold.

## Integrity and accounting

The h5 packet's 29 products, six source snapshots and 14 pre-voltage selection
products pass size/SHA-256 verification. The panel was visually inspected. The
h5 worker elapsed time was 339.57 s with one reader, four CPU threads and zero
GPU. H1 performed
no source-voltage read or duplicate 400-event rescore; its only waveform access
was the 67.9 MB saved-cache first-draw control.

H5 packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-co-two-unit-400waveforms-v1/`.
H1 packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/co_exposure_and_sample_size_review_20260928/`.
