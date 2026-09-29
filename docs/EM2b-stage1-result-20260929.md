# EM.2b stage-1 W2 result — 2026-09-29

## Verdict

**Use the rounded field with production-order SpikeInterface kriging.** On the
frozen Luke0804 imec1 W2 scorecard, rounded exact-DD and rounded kriging are
practically equivalent. Rounded kriging has a meaningful advantage over the
same kriging operator driven by the unrounded field. The stage-1 result
therefore attributes the observed difference to field rounding, not to the
choice between exact-DD and kriging remapping.

The preregistered stage-2 trigger is not met. The IDW and nearest sorts were
not launched. This preserves the staged resource rule instead of spending two
diagnostic sorts after the field-rounding question has been resolved.

## Frozen scorecard result

| Arm | Assigned units | Eligible units | rho, negative versus flat |
|---|---:|---:|---:|
| no correction S0 | 538 | 447 | 0.209377 |
| rounded exact-DD | 477 | 419 | 0.696309 |
| rounded kriging bridge | 474 | 416 | 0.702656 |
| unrounded kriging | 486 | 410 | 0.651093 |

The paired 2,000-draw common block bootstrap gave:

| First minus second | Delta rho | 95% CI | Frozen decision |
|---|---:|---:|---|
| rounded exact-DD minus unrounded kriging | +0.045216 | [0.014336, 0.077813] | mixed or inconclusive |
| rounded exact-DD minus rounded kriging | -0.006347 | [-0.020942, 0.001258] | practical equivalence |
| rounded kriging minus unrounded kriging | +0.051563 | [0.030267, 0.081549] | meaningful first-arm advantage |

All directional and equivalence yield and segment-safe short-ISI guardrails
passed in all three comparisons. The result remains an arm-local rate-rank
continuity proxy; it does not establish cross-arm unit identity or purity.

The frozen arms manifest is
`testing/inputs/em2b_w2/arms_manifest_stage1_20260929.json` (SHA-256
`86e73a5766c13aad443f32bdd0522e5cba4d36cfa0ef1d24eab9b73965617c27`).
The scorecard packet is
`testing/outputs/em2b_w2_scorecard_stage1_20260929`; its manifest SHA-256 is
`eba5947aabcd9f67fc6e04106170e2c4070fb2015d3a74e2daa07daf9e0d60db`.

## Execution and resources

Both arms ran sequentially as independent systemd user services with a
58-minute service ceiling, four CPU jobs, and one RTX A5000. Both published
successful final production-runner receipts, and the manifest builder verified
that each receipt attests the saved sorting bytes.

| Arm | Service wall | CPU | Preprocess | Detect | Sort | QC | Final scratch | Sorting SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| unrounded kriging | 1165.2 s | 3041.4 s | 227.5 s | 407.4 s | 507.0 s | 9.1 s | 9.124 GiB | `e6c521cb...87605b` |
| rounded kriging | 1280.9 s | 3170.1 s | 213.1 s | 512.2 s | 531.9 s | 9.0 s | 9.275 GiB | `c19db164...def45e` |

Periodic disk checks reached `11G` for each run. Both stayed below the frozen
13 GiB saved-scratch cap and the 3,600 s wall cap. The unrounded arm produced
486 assigned units and 535,788 accepted events; the rounded arm produced 474
assigned units and 592,472 accepted events. No RF was evaluated, and no outer
holdout was accessed.

The compact resource and receipt audit is
`testing/outputs/em2b_w2_stage1_measurement_20260929/AUDIT.json` (SHA-256
`8bafd68dc60eb8850171ec8235e4e38cb4acc289fd4e676161ec23bdac8cb69c`);
its packet manifest SHA-256 is
`7133f00324a5f1c79a35409117001c2763cf796d1858d0ba3bd52ae5f9d9eae2`.

## Production implication

The usable W2 implementation is the already-tested production order:

1. preprocess the accepted 383-channel source;
2. interpolate AP191;
3. round the rigid displacement field to the accepted 40-micrometer lattice;
4. apply SpikeInterface kriging remapping;
5. crop to the 182 target sites.

This removes the custom exact-DD remapper from the production path without a
detectable loss on the frozen W2 scorecard. The unrounded-field variant should
not replace it: its continuity proxy is lower by 0.0516 relative to rounded
kriging, with the entire paired 95% interval above zero in favor of rounding.
