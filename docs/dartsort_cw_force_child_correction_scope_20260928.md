# CW H1 independent review scope

CW preserves CV v1 and audits the corrected H5 v2 packet. The correction is
limited to saved-data accounting; it does not authorize voltage access,
sorting, merging, deduplication, threshold changes, matching, or calibration.

The independent audit will verify:

1. every actual force-parent family and every actual no-force child are retained,
   with thresholded/substantial subsets reported only as secondary strata;
2. the primary child is the immutable saved amplitude-proxy child and all other
   children are separate targets, rather than an unlabelled top-two-by-count pair;
3. the signed-shift null uses the common target domain
   `C = I intersect (I + delta)`: observed queries are in `C`, shifted queries
   originate in `C`, and reference events use the same `C` with the declared
   maximum-lag trim;
4. 200 frozen signed offsets, explicit supported event/exposure counts, zero and
   unavailable cells, and the 8, 9--29, and 30 sample boundaries;
5. whole-force adjacent-pair closure at 9--29 samples independently equals
   W2 6,250 and W3 5,685, with 8 and 30 samples separate;
6. actual no-force survivors, force-only/lost rows, state-boundary pairs, and the
   common-row-clock domain remain distinct;
7. per-parent/per-child cumulative contributions close at top 1/5/10 and include
   the small--small/all-other remainder;
8. native constituent morphology remains distinct from final-child event
   profiles, and immutable saved matching/amplitude joins are not substituted
   by array-length alignment.

H1 will use a small boundary/rate-step fixture plus one-family and endpoint-count
checks. It will not duplicate the full v2 null computation.

Limits: existing CV allowance plus a 600 CPU-second CW extension, two threads,
one reader, 20 GB RAM, 500 MB final output, at least 30 GB free, zero raw-voltage
bytes and zero GPU time. H1 and H5 charges remain separate.

