# Screening candidates for direct motion review

This shortlist uses the existing nine-cell overlays and per-cell scores, not peak attenuation or retained fraction as a quality score. These are candidates for review, not validated better motion configurations. All listed variants belong to the compensated5σ screening sweep; the later3σ study is separate.

| Candidate | Why retain it for review | Specific limitation |
|---|---|---|
| Center-energy only (≥0.65) | Retains84.8%; overall1.352 versus1.353µm baseline and movement1.420 versus1.433. Most trajectories remain close to baseline. | Numerical changes are tiny; no established improvement. Unit154 overall2.133→1.801 but its drop/recovery1.327→1.966, illustrating why averages are insufficient. |
| Relaxed combination (amplitude6σ, center0.50, neighbor0.6; other gates unchanged) | Retains56.2%; overall closer for154,246,317,463. Unit1542.133→1.229µm. | Unit4451.669→3.003 and deeper510/549/587 regress; drop/recovery overall1.433→2.359. A local comparison candidate, not a global winner. |
| Full screen without amplitude gate | Retains48.7%; unit154 overall2.133→0.766, and2460.794→0.674. | Unit154 drop/recovery worsens1.327→2.656; central/deep losses remain. Similar to5σ amplitude-gate relaxation, so do not spend duplicate follow-ups on both. |

Keep the full aggressive screen as a useful contrasting case: it improves shallow unit80 overall2.779→2.334 and drop/recovery2.076→0.392, while central445/463 become much worse. Amplitude-only is another contrast, preserving several deep trajectories but badly changing shallow80. Neither is a general recommendation. Stationary-looking references and incomplete identity qualification could mislead any local preference.

The new depth/time atlas is `testing/outputs/luke_screen_depth_time_atlas_v1/README.md`, with an all21page PDF and per-variant PNG/PDF links paired to the nine-lighthouse overlays. Each page compares unscreened, retained and removed populations, with counts and absolute detected-amplitude mass. Localized-depth bins are10µm×1s; all pages share explicit log1p color scales with no per-arm rescaling or quantile clipping. Counts outside the plotted range are reported. Mask hashes, partition checks, source hashes and plotted counts are saved. No detection, localization or motion fitting was rerun.

Priority change: expand biological reference depth/time coverage before another broad conditioning sweep. The existing broader candidate inventory should support a bounded qualification batch and300s then600s contiguous tracking, keeping local identity/sensitivity gaps explicit. A larger cohort should adjudicate local screening improvements rather than being selected for agreement with one candidate motion field.
