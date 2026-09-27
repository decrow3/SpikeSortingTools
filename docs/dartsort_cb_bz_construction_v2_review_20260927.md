# CB read-only review of BZ construction v2

## Verdict

**The construction packet is internally complete and shows that the 3,000 vs
30,000-sample template construction chunk changes downstream force structure,
but it does not repair or explain away BQ's failed numerical gate.** Treat this
as a real construction-sensitivity result, not a final merge/QDA comparison.

Both arms are fresh paired builds over the same 748 unit IDs, identical exact
spike counts and the same saved 299,635 sampled rows. The finite distance mask
is unchanged. Nevertheless, templates are not equivalent: per-unit template
cosine is 0.9898 median, 0.9461 P5 and 0.0296 minimum. Finite pair-distance
absolute changes are 0.0073 median, 0.1984 P95 and 1.014 maximum.

Reading the comparison from the short construction to the long construction,
upper-triangle direct force edges change 229 -> 236; 37 edges flip (Jaccard
0.8526). Force-linkage pair relations change 483 -> 611, with 206 flips, and
force components change 559 -> 560. The experimental intervention was the
opposite direction, long -> short: 236 -> 229 edges, 611 -> 483 linkage pairs
and 560 -> 559 components. Thus a small typical template difference crosses
thresholds and propagates through connected components.

The packet correctly preserves the original BQ gate as `failed_unchanged` and
does not manufacture QDA/final labels: required post-TMM responsibilities and
unit proportions were not saved. It also documents clean relevant DARTsort
sources, exact configs, source/input hashes and fresh builds for both arms.

Interpretation is bounded: construction chunk length is a consequential hidden
operator for this run. The result does not identify 3,000 or 30,000 as
scientifically correct, and cannot adjudicate QDA. No additional construction
rerun is justified from this packet alone. The next useful step is to report
the changed-edge/component sensitivity alongside the failed BQ gate and, if
final QDA comparison remains necessary, save the actual post-TMM score state in
a prospective run rather than substitute matching-stage scores.

Reviewed packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-bz-construction-v2/`.
No packet file was modified and no raw data, voltage, GPU or sort was used.
