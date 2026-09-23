# Saved motion-array audit and export — 23 September 2026

Only existing documents, field arrays, template arrays, and event assignments were read. No estimation, raw-voltage extraction, sorting, or injection was performed. `analyze_saved.py` contains the arithmetic and export procedure; `build_report.py` renders this report. Source artifacts were not modified.

## Definitions and important corrections

- All times here are seconds from AP recording frame zero. Intervals are half-open. The old DREDGE absolute clock is converted by subtracting 3057.6775463558583 s.
- Displacement convention: corrected depth = observed depth − displacement. All motion magnitudes are µm.
- **Window-centred rigid** means subtract each depth trace's temporal median within the selected window, then take the median over depths at each time. This reproduces the pilot's published rigid definition. The columns named `rigid` in the machine-readable files instead preserve the other convention used in the preceding answer: median over the saved, uncentred depth traces. `rigid_after_depth_centering` identifies the pilot-consistent convention explicitly. The two are not interchangeable because the cross-depth median is nonlinear.
- Excursion means P95 minus P5 over time, not full range. Per-depth excursions are unaffected by subtracting constant offsets. Reported per-depth P5/P95 endpoints preserve saved offsets; their zero is arbitrary.
- Increment SD is population SD (`ddof=0`) of d(t+lag)−d(t), without removing a fitted trend. For uniform 0.25 s fields, lags use exact index differences 1,4,8,20,80 and pair counts 479,476,472,460,400 in each 480-sample window. It measures variability of changes, not average speed or physical ground truth.
- There is no pilot field covering 8160–8280 s. Both full-session fields cover both requested windows. No pilot field was extrapolated.
- The 930–1030 s 348-channel matrix is **imec0**. The p10 and late-window analyses are **imec1**.

## 1. imec1 p10 alone

The pilot interval is [987.3553929363488,1107.3553896029887) s. Actual field samples run from 987.356092949996 to 1107.106092949996 s.

| component                   |   depth_um |    p5_um |   p95_um |   excursion_um |
|:----------------------------|-----------:|---------:|---------:|---------------:|
| rigid                       |    nan     | -136.585 |   31.555 |        168.140 |
| rigid_after_depth_centering |    nan     | -152.969 |   11.407 |        164.377 |
| depth_0                     |     -5.727 |  -75.780 |   26.755 |        102.535 |
| depth_1                     |   1270.957 | -176.022 |   58.713 |        234.735 |
| depth_2                     |   2547.642 | -230.283 |   75.029 |        305.312 |
| depth_3                     |   3824.326 |  -94.004 |   42.368 |        136.372 |

Largest absolute 0.25 s steps:

| component                   |   max_abs_step_um |   signed_step_um |   from_s |     to_s |
|:----------------------------|------------------:|-----------------:|---------:|---------:|
| rigid                       |            78.632 |           78.632 | 1099.356 | 1099.606 |
| rigid_after_depth_centering |            73.069 |           73.069 | 1099.356 | 1099.606 |
| depth_0                     |            76.853 |          -76.853 | 1014.356 | 1014.606 |
| depth_1                     |           120.550 |         -120.550 | 1097.606 | 1097.856 |
| depth_2                     |           139.223 |         -139.223 | 1031.356 | 1031.606 |
| depth_3                     |            75.993 |           75.993 | 1105.856 | 1106.106 |

The largest local step is a model-output change, not a validated tissue displacement. The pilot and full-session fits differ greatly in the same p10 interval; the two full-session fits agree much more closely with one another. This prevents treating the pilot's magnitude as an established recording property.

## 2. Increment time scales

Window-centred rigid increment SD:

| field           | window    |   0.25 |    1.0 |    2.0 |    5.0 |   20.0 |
|:----------------|:----------|-------:|-------:|-------:|-------:|-------:|
| merge_bias      | 8160_8280 |  5.423 | 13.185 | 15.585 | 22.144 | 27.155 |
| merge_bias      | p10       |  5.009 | 12.565 | 16.024 | 20.529 | 19.954 |
| pilot_p10       | p10       | 15.858 | 41.494 | 59.858 | 74.628 | 72.482 |
| shared_recovery | 8160_8280 |  5.211 | 13.332 | 16.652 | 22.232 | 26.678 |
| shared_recovery | p10       |  5.263 | 13.173 | 18.006 | 22.079 | 21.660 |

Window-centred rigid excursions:

| field           | window    |    p5_um |   p95_um |   excursion_um |
|:----------------|:----------|---------:|---------:|---------------:|
| pilot_p10       | p10       | -152.969 |   11.407 |        164.377 |
| shared_recovery | p10       |  -34.552 |   13.711 |         48.263 |
| shared_recovery | 8160_8280 |  -40.031 |   17.052 |         57.083 |
| merge_bias      | p10       |  -32.526 |   14.150 |         46.676 |
| merge_bias      | 8160_8280 |  -41.831 |   17.245 |         59.076 |

### Faster-than-two-second sensitivity

A percentile range has no unique additive frequency decomposition. As an explicit descriptive convention, high-pass the selected rigid trace with a fourth-order Butterworth at 0.5 Hz using forward/backward SOS filtering; omit 2 s at each end and compare P95−P5 of the high-pass component with P95−P5 of the original on that same interior. This is a smooth cutoff (with the forward/backward squared response), not an exact division at 2 s. The ratio is **not a fraction of total excursion explained**, and the field's 1 s estimation kernel already limits fast-motion representation. No field was refit.

| field           | window    |   full_excursion_same_interior_um |   highpass_excursion_um |   excursion_ratio |   variance_ratio |
|:----------------|:----------|----------------------------------:|------------------------:|------------------:|-----------------:|
| pilot_p10       | p10       |                           164.393 |                  32.189 |             0.196 |            0.031 |
| shared_recovery | p10       |                            48.246 |                  11.043 |             0.229 |            0.048 |
| shared_recovery | 8160_8280 |                            58.519 |                  10.059 |             0.172 |            0.029 |
| merge_bias      | p10       |                            47.340 |                  10.579 |             0.223 |            0.047 |
| merge_bias      | 8160_8280 |                            60.666 |                  11.486 |             0.189 |            0.036 |

### Saved lighthouse observations

The eight-proposal compact expansion has **no events in p10 or 8160–8280 s**. Historical imec1 tables do cover short pieces of p10. A scan of 76 scored event tables found no events in the late interval; see `event_inventory.csv`. These are candidate measurements, not verified independent cells. Revisions and hypotheses are preserved separately and never pooled.

For irregular event depths, use medians in 0.25 s bins separately for each candidate and source table. Difference only occupied endpoints within the same recorded acquisition interval: 998–1002, 1014–1025, 1071–1075, or 1097–1112 s, clipped to p10. No gap interpolation, filling, or cross-window differences. Thus lags are bin-grid lags (individual event separations have sub-bin jitter), unlike exact field sample lags. SD requires at least two pairs. Missing statistics are unavailable, not zero. No interval supports a 20 s lag.

The following are the historical **broad-template s300** assignments for IDs that later entered the eight-proposal expansion; they are not the expansion's final compact-template tracks:

| candidate   |    0.25 |     1.0 |     2.0 |     5.0 |   20.0 |
|:------------|--------:|--------:|--------:|--------:|-------:|
| p06_f010    | nan     | nan     | nan     | nan     |    nan |
| p06_f045    |  43.865 |  86.164 |  92.742 | nan     |    nan |
| p08_f025    | nan     | nan     | nan     | nan     |    nan |
| p08_f029    | nan     | nan     | nan     | nan     |    nan |
| p08_f033    |   2.025 |  22.634 |  24.197 | 109.885 |    nan |
| p08_f044    | nan     | nan     | nan     | nan     |    nan |
| p09_f020    |  59.693 | 148.966 | 118.186 | 103.324 |    nan |
| p09_f024    |  39.426 | 196.809 | 122.789 | 121.568 |    nan |

Pair counts for the same rows:

| candidate   |    0.25 |     1.0 |     2.0 |     5.0 |    20.0 |
|:------------|--------:|--------:|--------:|--------:|--------:|
| p06_f010    |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
| p06_f045    |  11.000 |   8.000 |   5.000 |   0.000 |   0.000 |
| p08_f025    |   0.000 |   0.000 |   0.000 |   0.000 |   0.000 |
| p08_f029    | nan     | nan     | nan     | nan     | nan     |
| p08_f033    |   7.000 |   6.000 |   2.000 |   3.000 |   0.000 |
| p08_f044    | nan     | nan     | nan     | nan     | nan     |
| p09_f020    |   8.000 |   6.000 |   4.000 |   3.000 |   0.000 |
| p09_f024    |   7.000 |   3.000 |   5.000 |   2.000 |   0.000 |

The very sparse, candidate-specific and sometimes enormous changes do not establish how fast the tissue moved. A high-frequency excursion decomposition is not warranted for these gapped, unverified tracks. `lighthouse_increment_sd.csv` also reports the other historical sources separately.

## 3. Across-depth spread and offsets

The previously quoted 24.59 µm DREDGE and 20.69 µm shared-recovery MEDiCINe spreads were computed from **uncentred saved depths**. They include constant depth-dependent offsets. In contrast, the pilot's published spread used per-depth temporal centring. For 8160–8280 s:

| field           |    window | depth_centered   |   median_spread_um |   p95_spread_um |
|:----------------|----------:|:-----------------|-------------------:|----------------:|
| shared_recovery | 8160_8280 | False            |             20.694 |          40.763 |
| shared_recovery | 8160_8280 | True             |             19.114 |          44.738 |
| merge_bias      | 8160_8280 | False            |             20.891 |          43.727 |
| merge_bias      | 8160_8280 | True             |             18.700 |          43.696 |
| original_dredge | 8160_8280 | False            |             24.592 |          38.266 |
| original_dredge | 8160_8280 | True             |             14.331 |          35.984 |

The temporal depth-median offsets have full depth ranges of 21.47 µm (original DREDGE), 9.31 µm (shared recovery), and 11.80 µm (merge bias). Subtracting them changes the median spread but does not change each depth's own temporal excursion. A depth-varying static offset in a field applied directly to voltage causes a permanent depth deformation; a common offset merely translates. The original upstream interpolation passed the saved DREDGE array directly into `Motion`, so these offsets were not removed there. This analysis does not establish that the offsets are biologically real or spurious. `depth_spread.csv` retains all offsets.

## 4. Are p06_f010 and p08_f044 duplicates?

The final strict-event table contains 115 p06_f010 events and 23 p08_f044 events. There are **0 one-to-one temporal matches within ±0.5 ms**: 0/115 and 0/23. These are all qualification and expansion windows combined, not just common-support increments. The global winner assignment is exclusive, so lack of coincident assignments does not rule out one cell being partitioned into two labels.

The compact-template geometry-aware cosine is **0.926689**, maximizing temporal lag from −3 to +3 waveform samples over ten exactly overlapping relative channels; minimum bidirectional energy coverage is **0.730265**. Different peak-channel phases make a naive flattened-array cosine inappropriate. This is the cosine between this pair, not either candidate's similarity to its nearest rival. Both the frozen compact templates and geometry are exported.

These results do not prove duplication, but they also do not validate independence. Their correlated trajectories should not be counted as independent cellular replication until that identity question is resolved. Moreover, the 2.9 µm separation is between broad seed-depth summaries, not independently verified instantaneous cell depths.

## 5. imec0 930–1030 s matrix fields

Both LFP candidates are rigid traces. Their per-depth excursion is therefore the same at every channel; neither has four or nine independent depth estimates. SG25 is the saved native-250-Hz trace after the documented Savitzky–Golay operation.

The saved matrix AP-rigid arm is the equal-depth rigid projection and has a 61.341 µm excursion. Taking a cross-depth median of the saved nonrigid AP field instead gives 63.291 µm. The applied arm and this diagnostic median are explicitly separated below.

| field                       | component   |   depth_um |     n |    p5_um |   p95_um |   excursion_um |
|:----------------------------|:------------|-----------:|------:|---------:|---------:|---------------:|
| lfp_native                  | rigid       |    nan     | 25000 | -196.460 |   33.298 |        229.759 |
| lfp_native                  | depth_0     |    nan     | 25000 | -196.460 |   33.298 |        229.759 |
| lfp_savgol                  | rigid       |    nan     | 25000 | -196.439 |   33.242 |        229.682 |
| lfp_savgol                  | depth_0     |    nan     | 25000 | -196.439 |   33.242 |        229.682 |
| fast_medicine_nonrigid      | rigid       |    nan     |   400 |  -57.626 |    5.665 |         63.291 |
| fast_medicine_nonrigid      | depth_0     |   -126.058 |   400 |  -19.902 |   19.131 |         39.033 |
| fast_medicine_nonrigid      | depth_1     |   1201.767 |   400 |  -79.137 |   11.895 |         91.032 |
| fast_medicine_nonrigid      | depth_2     |   2529.591 |   400 |  -70.739 |   11.232 |         81.971 |
| fast_medicine_nonrigid      | depth_3     |   3857.416 |   400 |  -62.160 |    7.108 |         69.268 |
| fast_medicine_applied_rigid | rigid       |    nan     |   400 |  -55.774 |    5.567 |         61.341 |
| fast_medicine_applied_rigid | depth_0     |    nan     |   400 |  -55.774 |    5.567 |         61.341 |
| native_ap                   | rigid       |    nan     |   100 |  -57.577 |    1.101 |         58.677 |
| native_ap                   | depth_0     |    312.000 |   100 |  -17.109 |    0.508 |         17.616 |
| native_ap                   | depth_1     |    712.000 |   100 | -144.581 |    0.729 |        145.309 |
| native_ap                   | depth_2     |   1112.000 |   100 |  -57.577 |    1.328 |         58.904 |
| native_ap                   | depth_3     |   1512.000 |   100 |  -16.470 |    1.220 |         17.690 |
| native_ap                   | depth_4     |   1912.000 |   100 |  -85.510 |    4.251 |         89.761 |
| native_ap                   | depth_5     |   2312.000 |   100 |  -41.185 |    1.710 |         42.896 |
| native_ap                   | depth_6     |   2712.000 |   100 |  -17.288 |    1.219 |         18.507 |
| native_ap                   | depth_7     |   3112.000 |   100 | -104.980 |    3.153 |        108.133 |
| native_ap                   | depth_8     |   3512.000 |   100 | -118.033 |    3.506 |        121.540 |

### The actual 10 September quiet-period result

This is a **different comparison**: its LFP candidate was the frozen `band_0p5_8` rigid trace, not the matrix's native-250-Hz/SG25 pair. Do not assign these errors to SG25 or native AP. The AP candidates are the full-session MEDiCINe field and its rigid projection.

Reference = consecutive 5 s increments of frozen, strict depth-aware lighthouse family medians, on common support, excluding predeclared concern units and retaining the 5 µm lattice-node sensitivity. Quiet = absolute observed lighthouse increment below 20 µm. Each family has equal weight; quiet strata permit at least one increment per family. The table distinguishes RMS of the field's predicted increments from RMSE of those increments against the lighthouse reference.

|    window | candidate   |   families |   increments |   candidate_increment_rms_um |   lighthouse_increment_rms_um |   residual_rmse_um |
|----------:|:------------|-----------:|-------------:|-----------------------------:|------------------------------:|-------------------:|
| 1150_1200 | ap_nonrigid |          4 |           28 |                        9.055 |                         1.426 |              9.032 |
| 1150_1200 | ap_rigid    |          4 |           28 |                        4.400 |                         1.426 |              4.917 |
| 1150_1200 | lfp_rigid   |          4 |           28 |                       21.666 |                         1.426 |             21.855 |
| 1150_1200 | zero        |          4 |           28 |                        0.000 |                         1.426 |              1.426 |
|  930_1030 | ap_nonrigid |          6 |           24 |                       10.138 |                         1.437 |             10.383 |
|  930_1030 | ap_rigid    |          6 |           24 |                        8.043 |                         1.437 |              8.276 |
|  930_1030 | lfp_rigid   |          6 |           24 |                       48.766 |                         1.437 |             48.863 |
|  930_1030 | zero        |          6 |           24 |                        0.000 |                         1.437 |              1.437 |

In particular, LFP predicts 48.766 µm RMS quiet increments against 1.437 µm RMS reference increments in 930–1030 s, producing 48.863 µm residual RMSE. In 1150–1200 s those values are 21.666, 1.426, and 21.855 µm. These are 5 s change statistics, not full-window excursions or direct proof of physical estimator error.

## 6. Export inventory

`export/` contains all requested field files, source configurations/receipts where available, original KS trace, frozen identity templates, the original 10 September comparison arrays/tables, and 76 original scored imec1 event tables. Event CSVs are losslessly gzip-compressed; selected canonical tables also have normalized copies adding `candidate` and `depth_um` while retaining original time, score, status, bank, event IDs, and per-event runner-up fields where present. No rival score was invented where absent. Older variants remain explicitly named and are not independent datasets.

- `source_manifest.csv`: source absolute path → exported path, byte count, exported SHA256, and decompressed original SHA256 for gzip CSVs. Copied source bytes were verified against originals.
- `SHA256SUMS`: checksums of every delivered file except the checksum file itself.
- `increment_sd.csv`: **all four depths**, raw-median rigid, and window-centred rigid at every requested lag.
- `matrix_excursions.csv`: exact P5/P95 endpoints, depths and excursions for all matrix fields.
- `fast_component.csv`, `depth_spread.csv`, `max_steps.csv`, `identity_overlap.json`, `lighthouse_increment_sd.csv`: complete numerical results.

The export is a directory rather than a duplicate archive because this filesystem has limited free space. No source files were deleted. All source paths remain in the manifest.

## Appendix: per-depth increment SDs

### merge_bias: 8160_8280

| component   |   0.25 |    1.0 |    2.0 |    5.0 |   20.0 |
|:------------|-------:|-------:|-------:|-------:|-------:|
| depth_0     |  6.518 | 15.766 | 17.137 | 20.912 | 26.586 |
| depth_1     |  7.540 | 18.107 | 20.990 | 30.550 | 37.204 |
| depth_2     |  7.544 | 18.827 | 20.822 | 27.832 | 31.620 |
| depth_3     |  7.496 | 18.420 | 19.737 | 23.582 | 25.942 |

### merge_bias: p10

| component   |   0.25 |    1.0 |    2.0 |    5.0 |   20.0 |
|:------------|-------:|-------:|-------:|-------:|-------:|
| depth_0     |  6.043 | 15.069 | 18.690 | 22.344 | 23.260 |
| depth_1     |  8.253 | 21.563 | 26.392 | 33.416 | 32.971 |
| depth_2     |  6.695 | 15.908 | 18.125 | 22.231 | 22.105 |
| depth_3     |  7.980 | 19.178 | 20.619 | 20.941 | 24.033 |

### pilot_p10: p10

| component   |   0.25 |    1.0 |     2.0 |     5.0 |    20.0 |
|:------------|-------:|-------:|--------:|--------:|--------:|
| depth_0     | 11.913 | 26.449 |  35.925 |  47.426 |  49.167 |
| depth_1     | 25.008 | 62.806 |  85.815 |  99.479 |  92.949 |
| depth_2     | 29.287 | 76.756 | 106.092 | 133.723 | 131.430 |
| depth_3     | 17.676 | 37.671 |  50.406 |  60.798 |  59.134 |

### shared_recovery: 8160_8280

| component   |   0.25 |    1.0 |    2.0 |    5.0 |   20.0 |
|:------------|-------:|-------:|-------:|-------:|-------:|
| depth_0     |  6.630 | 15.184 | 17.894 | 20.124 | 22.204 |
| depth_1     |  7.488 | 18.742 | 22.392 | 33.216 | 39.223 |
| depth_2     |  7.045 | 17.238 | 21.480 | 27.076 | 31.164 |
| depth_3     |  6.849 | 16.384 | 17.958 | 21.831 | 24.066 |

### shared_recovery: p10

| component   |   0.25 |    1.0 |    2.0 |    5.0 |   20.0 |
|:------------|-------:|-------:|-------:|-------:|-------:|
| depth_0     |  6.266 | 14.578 | 18.189 | 23.188 | 23.695 |
| depth_1     |  8.820 | 22.746 | 28.755 | 35.134 | 35.232 |
| depth_2     |  7.616 | 18.157 | 21.788 | 24.409 | 23.060 |
| depth_3     |  7.962 | 18.927 | 20.961 | 21.500 | 23.295 |

