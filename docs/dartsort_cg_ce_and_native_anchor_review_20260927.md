# CG: CE paired-grouping audit and same-state anchor contract

## Verdict

The new CE packet is an internally valid **prospective same-state construction
comparison**. Its construction rows, post-TMM state, basis, RNG and row-aligned
outputs pass the independent checks below. The shorter construction changes 103
expanded-force relations, and the exact same 103 relations remain changed in
the final union. QDA requested-pair coverage contracts from 2,438 to 489, but
the accepted mask is exactly unchanged at 16 relations. This supports a causal
effect of construction chunk length on this prospective grouping state, not a
historical reconstruction or biological identity claim.

The post-TMM state has 747 units; old BZ has 748. No integer-ID join between
them is valid or used. The next anchor pilot must select and score within CE's
747-unit namespace only. A tested helper and frozen contract are ready; no raw
snippets or residual field were read or fitted here.

Packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ce-paired-grouping-v1/`.

## Independent CE audit

### Integrity and isolated construction state

- Every `MANIFEST.json` product matches its declared SHA-256 and byte size.
- `POST_TMM_STATE.npz` has 641,588 row IDs exactly `0..641587`, dense nonnoise
  labels `0..746`, channels `0..181`, and 747 finite unit log proportions.
- The 299,625 construction rows are unique, strictly ordered, within that
  domain, and their saved times, labels and channels exactly equal the indexed
  post-TMM arrays.
- Both arms contain dense unit IDs `0..746`; their total per-unit spike counts
  exactly equal construction-row counts and sum to 299,625. Registered geometry,
  TSVD components/mean and declared whitening state are identical. Per-channel
  counts are allowed to differ because construction registration is the
  treatment. Python, NumPy, torch and CUDA RNG start/end digests are identical.
- The two effective template configurations differ in the declared construction
  chunk length, 30,000 versus 3,000 samples. No prefix rerun occurred between
  arms (`prefix_invocations=1`), and both post-arm input digests match.

### QDA, force and final assignment lineage

Saved-array checks independently reproduce:

| Quantity | 30,000 | 3,000 |
|---|---:|---:|
| QDA requested upper-triangle pairs | 2,438 | 489 |
| QDA accepted pairs | 16 | 16 |
| final units | 551 | 552 |
| actual-clock shifted rows | 6,847 | 12,928 |

The accepted QDA masks are array-identical. QDA score and minimum-ratio arrays
are also exact; IoU/coverage differ on the changed requested domain, without an
accepted-relation change. The expanded-force masks differ on 103 upper-triangle
relations, and final-union masks differ on the same 103. Connected-component
comparison gives 106 changed source co-memberships. Among rows assigned in both
arms, 85,496 inherit a changed source-component signature and 549,911 do not.

The row-aligned final partition has 635,407 rows assigned in both arms, 199 only
in the long arm, 108 only in the short arm, and 5,874 noise in both, totaling
641,588 exactly. The separately reported one `ambiguous_unavailable_source` row
is nested in that accounting, not a fifth disjoint category. On the common
assigned rows, adjusted Rand index independently reproduces
`0.9725026489098947`.

Final labels cannot be reconstructed by merely applying `merge_mapping` (or
`depth_reorder`) to each row's pre-grouping label: reassignment and
deduplication occur in the grouping path. The saved final-label vectors are
row-aligned to the exact source row IDs, have contiguous ranges `0..550` and
`0..551`, and match the supplemental copies. This is valid final-assignment
lineage; inventing a cross-run ID mapping would not be.

### Clock lineage

Both arms retain the exact post-TMM physical channel vector. Relative to the
fixed post-TMM sample clock, the long-arm actual-clock shift histogram is
`{-2:31, -1:4289, 0:634741, +1:2527}` and the short-arm histogram is
`{-2:1225, -1:1108, 0:628660, +1:10595}`. Long-arm timing moves one row from
rest to episode; short-arm timing moves none. The supplemental fixed-clock and
actual-clock metrics are therefore correctly distinguished. Fixed-clock results
remain sensitivity analyses, not substitutes for actual-clock output.

## Native same-state anchor contract

Implementation: `testing/cg_anchor_helper.py`. It is intentionally a small
consumer helper rather than a new framework.

Published packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cg_anchor_helper_20260927/`.
Its `SHA256SUMS` verifies the helper, tests, contract, README and this report;
`COMPLETE.json` was written last.

### Selection, frozen before waveform outcomes

1. Use only CE stage IDs `0..746`, CE row IDs and the corrected 3,000-sample CE
   template bank. Never join to AW/static or old BQ/BZ integer IDs.
2. Exclude every unit belonging to an off-diagonal force-connected component in
   either native arm before ranking.
3. Apply canonical rest, valid-waveform-bounds and collision masks before any
   event cap. A missing pool is unresolved; it is never filled from episode or
   masked time.
4. Determine the corrected-3,000 peak using DARTsort's channel SNR convention
   (`ptp × sqrt(per-channel count)`). Freeze physical contacts within 150 µm of
   that peak. Read all 182 contacts, then score only this fixed physical support.
5. Require at least 100 rest events and 10 distinct nonoverlapping 5-s blocks in
   each contiguous half. Rank within eight physical-depth strata by the minimum
   half block count, minimum half event count, support coverage, construction
   count and stable unit ID. Choose one per occupied stratum; deterministic
   farthest-depth backfill supplies at most eight total.
6. Freeze exactly 100 events per half with deterministic block-balanced sampling
   and publish row IDs, blocks, masks/hashes, split sample and support coordinates
   before reading outcomes.

### Reproducibility score

- Read `8 × 2 × 100 × 121 × 182` float32 samples: 140,940,800 bytes (134.4 MiB)
  logical payload. The support radius is applied only during scoring, not read.
- Estimate one robust per-channel noise vector from half-1 waveform edges and
  freeze it. Use fixed support, zero lag and unit gain. There is no held-out
  lag, gain or support optimization.
- The point statistic is cosine between noise-scaled median waveforms from the
  two halves. Bootstrap each half's exact point-estimand blocks independently
  for 1,000 draws. Accept only if all 1,000 draws are finite/accounted and the
  95% interval's lower bound is at least 0.90.
- Save signal energy, coverage, block/event counts, point value, interval and
  replicate accounting. Any missing pool/support/nonfinite draw is unresolved.
- Save nearest competing-template cosines as diagnostics only. They are not an
  identity gate and cannot turn rest reproducibility into identity or episode
  validation.

The helper test suite exercises force-union exclusion, deterministic
depth-stratified selection, pre-cap masks, exact row repeatability, block gates,
complete replicate accounting and missing-support failure. Fourteen combined CG
and existing BZ tests pass.

### Executor envelope

The authorized execution remains bounded to two CPU threads, one reader, 20 GB
RAM, 30 GB free-space guard, about 1 GB physical reads and 500 MB final output.
The 1,600-snippet payload plus receipts is comfortably inside it. Estimated
elapsed after input/mask resolution is 4–6 CPU minutes. Selection and contract
are ready for H5; this task did not authorize or perform the raw read itself.

## CD two-pair sidecar

At the one permitted check, neither
`cd_receipts_v1/CD_COMPATIBLE_PAIR_REPRODUCTION.json` nor its NPZ was published.
Therefore the two compatible CD pairs remain worker-reported only; intervals and
finite replicate counts were not independently certified here. No repeated poll
or 18-pair rescore was performed.
