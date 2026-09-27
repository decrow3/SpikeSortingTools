# CL: independent CK saved-cache measurement audit

## Verdict

CK's saved result is internally reproducible, but **no anchor is qualified**.
All eight point cosines (0.767--0.839) and half-1 noise arrays reproduce exactly;
the first 100 bootstrap draws for independently selected units 23 and 120 are
bitwise identical. The 1,600 cached rows, event IDs, fixed-clock samples,
channels and 900-s session-time conversion all match their saved selection.
Thus the 0/8 outcome is not a row join, clock-origin, cache-order or scoring-code
error.

The audit does not support attributing the low bootstrap lower bounds solely to
unequal block weights. Repeated-block resampling lowers the bound even in a
known-stable equal-block synthetic control; unequal sizes amplify the effect.
The raw-cache diagnostics also show plausible measurement variation across
halves, especially spatial profile changes, without establishing whether the
cause is residual drift, additive noise or misgrouping. The frozen 0.90 rule is
unchanged and no post-hoc diagnostic is promoted to a pass.

## Source and reproducibility checks

| Check | Result | Interpretation |
| --- | ---: | --- |
| CK manifest products | all valid | sizes and SHA-256 values match |
| Raw cache | `[1600,121,182]` float32 | full physical-channel saved cache |
| Selected row/index/time/channel | 1,600/1,600 exact | no join or order defect |
| Session-time origin | all exact | `900 + local_sample / 29999.7591667` |
| Point scores | 8/8 exact | saved 0.767--0.839 reproduced |
| Half-1 noise arrays | 8/8 exact | support conditioning reproduced |
| Bootstrap subset | 200/200 exact | first 100 draws for units 23 and 120 |

Visual inspection of `ANCHOR_RELIABILITY_PANEL.png` agrees with the files: the
two half traces substantially overlap on peak profiles and waveform shapes,
while every bootstrap distribution remains well below the 0.90 qualification
line. This is waveform compatibility at best, not identity proof.

## Measurement diagnostics

The registered template determines a fixed physical support, but the cached raw
snippets are not themselves motion-registered. On the fixed full support:

- median absolute half-to-half raw PTP-depth-centroid change is 5.41 µm; the
  maximum is 24.58 µm (unit 508);
- median observed event channel does not change for any of the eight units;
- peak-channel trough timing differs by at most one sample;
- each score uses 3,388--3,630 conditioned features, while signal participation
  dimension ranges from 26.95 to 411.56 (median 219.16);
- half-difference energy is 0.349--0.527 of mean-template energy (median 0.427).
  Under a simple independent additive-error decomposition this corresponds to a
  per-half noise-to-signal energy proxy of 0.191--0.303 (median 0.239).

These quantities make additive noise/high-dimensional median variability and
unregistered spatial profile change concrete alternatives to biological
misgrouping. They do not determine which mechanism dominates.

## Frozen synthetic control

A known-stable 121-sample, eight-channel waveform with independent Gaussian
noise, exactly 100 events per half, ten blocks, seed 701 and 300 bootstrap draws
was specified in commit `e66d25d` before output.

| Block design | Point cosine | Bootstrap lower | Point minus lower |
| --- | ---: | ---: | ---: |
| Equal, 10 events in each block | 0.9734 | 0.9334 | 0.0401 |
| Unequal, sizes 1/1/1/1/1/5/10/20/25/35 | 0.9734 | 0.8774 | 0.0960 |

Repeated-block median estimation therefore creates a lower-bound gap even with
equal block sizes. Unequal block sizes increase the gap in this controlled
example, but CK's earlier specific causal attribution to unequal weights alone
was unproved.

## Conclusion and smallest next test

The finite current conclusion is **zero qualified native anchors under the
frozen CK rule**. The audit neither rescues an anchor nor justifies lowering the
threshold. If another anchor test is needed, the smallest justified next test is
a cached-only, predeclared per-event physical-channel recentering sensitivity on
the same two deterministic units, followed by all eight only if its direction
is coherent. It must preserve support and zero temporal lag and remain a labeled
measurement sensitivity, not an identity pass. No new raw read, field fit or
infrastructure expansion is justified first.

Outputs:
`/mnt/NPX/Luke/DARTsort_motion_experiments/cl_anchor_measurement_audit_20260927/`.
No raw voltage was read; only the retained cache was memory-mapped.
