# Luke0804 imec1 compact-core lighthouse pilot

## Decision

Improve waveform proposals before enlarging the seed bank. The bounded test
supports timing-aligned compact cores, but it does not yet validate any family
as a lighthouse cell.

## Cached premise check

The two v3 banks contain 192 proposals each. Member-similarity failures affect
178/192 in the 36.46--56.46 s bank and 179/192 in the 300--320 s bank. Only
7 broad proposals in the first bank and none in the second contain compact
post-hoc local cores under the earlier diagnostic; the second bank has three
post-match motion-scale depth spans. Across the combined banks, 48 selected
families acquire a cross-bank template alias at cosine >=0.95. This makes broad
proposal construction and identity ambiguity the immediate bottlenecks.

## Bounded comparison

- Raw AP voltage: 300--320 s.
- Template construction: 300--310 s.
- Frozen qualification: 310--320 s.
- Proposal assignments: existing v3 assignments, held fixed.
- Compact core: reciprocal odd/even split-half support after +/-3-sample timing
  alignment, cosine >=0.86, at least five members in each half.
- Competition: the same complete 36.46--56.46 s external template bank and the
  same real/decoy global competition for both arms.
- Excluded from construction and matching: sorter labels, absolute depth, and
  every motion estimate.

Of 190 proposals with training-half support, 20 contained a reciprocal compact
core. In those 20, strict frozen matches increased from 132 for broad templates
to 198 for compact templates: 7 families improved, 9 tied, and 4 worsened.
Across all proposal identities, own-bank strict matches increased from 180 to
249; total strict matches increased from 458 to 520; decoy winners decreased
from 8,713 to 8,671; external-bank strict winners decreased from 278 to 271.

The largest compact gains were p08_f025 (0 to 41), p08_f029 (0 to 21), and
p08_f033 (24 to 37). Their revealed qualification depth spans were 42.7,
204.1, and 32.6 um respectively. Depth was revealed only after matching. These
are proposal-level leads, not verified cell identities or motion ground truth.

## Corrected denominator interpretation

The original pilot summary reported 2 broad versus 1 compact
`qualifies_depth_blind` flags. That flag divided qualification matches by the
full broad-proposal membership in both arms. It therefore penalizes the smaller
compact template support and is not a fair arm comparison. It is retained in
the immutable run output for provenance, but paired strict counts and separate
proposal/core support rates are the decision metrics.

## Remaining validity risk and next implementation

The v3 proposal assignments used the entire 300--320 s interval before this
pilot split it into construction and qualification halves. Thus template
construction was split, but proposal discovery was not independent of the
qualification interval. The next method should form proposals independently in
multiple seed windows, retain only timing-aligned reciprocal cores, combine
those compact variants into a seed bank, and then qualify on untouched windows.
It should preserve broad, compact, ambiguous, lower-score, decoy, and unmatched
evidence separately. Absolute depth remains post-match evidence; spans up to
about 250--300 um remain a motion-scale category rather than an automatic
identity failure.

## Reproducibility and job status

The successful persistent job is
`testing/outputs/luke_imec1_compact_core_pilot_v1_v2_job`; its managed receipt
records exit code 0 and the systemd service result is success. The preceding v1
job is preserved with exit code 1 and its traceback: its bundled source had
resolved cached bank paths relative to the bundle. The corrected v2 config uses
explicit absolute cached inputs and records their SHA-256 hashes. The bounded
stage has no within-stage checkpoint.
