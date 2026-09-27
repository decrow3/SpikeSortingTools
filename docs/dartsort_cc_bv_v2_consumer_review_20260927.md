# CC independent consumer review of BV identity v2

## Verdict

**The packet is provenance-strong but its single rest compatibility label is
not valid under the declared v3 consumer contract.** The worker used helper v2
(SHA-256 `96af...`), omitted the declared `min_events=100` and did not publish
accepted/requested bootstrap counts. All corrected labels remain limited
waveform evidence, never identity or merge truth.

## Saved-subset check

The sole `limited_waveform_compatibility` row is pair 1, units 6/9, parent 91.
Its saved raw banks each contain 100 waveforms. After applying the same common
5-second block intersections used by the estimator, counts are A1/B1 75/100
and A2/B2 73/100 across 19 and 21 common blocks. One direct v3 API call on only
this saved subset, with the declared `min_events=100`, returns
`ValueError: insufficient_events`. No full rescore was run.

Its v2 intervals would otherwise be descriptively compatible: within-unit
lower bounds 0.944/0.963 and cross-minus-within gap interval [-0.0084, 0.0163].
That observation does not override the frozen event gate or absent v3 replicate
accounting.

## Metadata and support audit

The cache has 9,797 float32 waveforms `[121,18]` and an exactly aligned metadata
table. All 144 pair/side/half/state groups are time-sorted; event row IDs are
unique within group; session times are finite; `block5` equals
`floor((time_session_s - 900)/5)`. Each pair has one constant explicit source
AP range of exactly 18 channels, plus capture index, event row ID, sample time,
session time and collision-exclusion count. This is substantially better
provenance than v1.

However, the packet does not save physical channel-ID vectors/geometry, frozen
noise vectors, declared support masks, the fitted frozen-state receipt, or
accepted/requested replicate counts. `supported_channels=18` only establishes
finite values in the crop; it does not establish complete waveform-energy
coverage. Episode field excursions span about 90–230 um, while every episode
row has fewer than 10 common blocks in half 2 (range 1–7) and is already
unresolved. The 18-site episode arrays are therefore limited crop evidence and
cannot justify full-waveform episode inference.

Rest common-block counts are 12–30 in half 1 and 6–34 in half 2; two rest pairs
fail the 10-block gate. The other point rows used default `min_events=4`, not the
declared 100.

## One next hypothesis

Test whether the event shortfall is caused by independently capping each bank
before intersecting common blocks. If the already-accepted worker cache retains
the underlying event pool, select up to 100 events *after* freezing and
intersecting eligible common blocks, then apply unchanged v3 gates. This can
establish whether parent-91 compatibility survives the declared sample support
without relaxing a threshold or rereading voltage. If the underlying event
pool is unavailable, leave the row unresolved.

Reviewed packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-bv-identity-v2/`.
No packet file was modified.
