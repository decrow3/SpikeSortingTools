# CA read-only review of the completed BV v1 packet

## Verdict

**The packet is complete and useful descriptively, but its one
`supports_merging` call is not valid under the corrected BX/BZ-v2 design.** Do
not convert any BV v1 row into an identity or merge decision. A worker-side v2
rescore needs retained event-level waveforms, 5-second block IDs, frozen support
and noise scales; those are not present in this published packet.

## Evidence

`COMPLETE.json` was written last and its four product hashes match. The packet
contains 18 fixed pairs, rest and episode rows, 6,990 selected snippets in the
accounting, 144 mean waveforms of shape `[121,18]`, and no raw voltage.

The implementation predates CA's repair:

- six of 18 pairs use different lags in rest and episode (pairs 0, 9, 11, 13,
  16 and 17), proving that state-specific refitting occurred rather than reuse
  of one rest-half-1 lag;
- gains also vary by state, and the CSV has no immutable rest-frozen state;
- its interval columns cover `difference_projection`, not the required
  within-unit reliability, cross-minus-within gap, cross deficit and signed-
  difference metrics;
- no common 5-second block IDs/counts or event arrays are published, so point
  estimates cannot be restricted to the same >=10-block domain as bootstrap
  intervals and BZ v2 cannot reconstruct those intervals from the mean-only
  NPZ;
- all 18 rest rows are marked qualified, but only 9/18 episode rows are. Under
  BX, episode is a separate supplement using the rest-frozen state.

The sole non-inconclusive v1 row is pair 5 (units 535/538, parent 12) at rest.
Its point metrics are descriptively compatible: within cosines 0.987/0.963 and
cross cosines 0.959/0.973. That is useful evidence for prioritizing a corrected
rescore, but it lacks the required block intervals and therefore remains
**inconclusive under BZ v2**. No stable-difference call was made.

## Smallest next step

H5 should retain or republish the already-extracted per-event waveform arrays,
their 5-second block IDs, declared support and noise vectors, then call the BZ
v2 API. No new raw read or GPU construction is justified until checking whether
those worker-side arrays still exist. Mean waveforms alone are insufficient.

Reviewed packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-bv-identity-v1/`.
This was read-only; no packet content was changed.
