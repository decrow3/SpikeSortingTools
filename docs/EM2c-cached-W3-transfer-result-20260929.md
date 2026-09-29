# EM.2c cached W3 transfer result — 2026-09-29

## Verdict

**The rounded-field benefit transfers to W3 in the existing exact-lattice
experiment.** The cached, preregistered W3 packet reports `ADVANCE`: exact
lattice minus fresh static increases the population rate-rank continuity proxy
by 0.5292, with 95% CI [0.4885, 0.5481]. The yield ratio is 0.9902, and both
short-ISI guardrails pass.

This supports the rounded-field choice selected on W2. It does not directly
demonstrate rounded-kriging sorting equivalence on W3. The evidence chain is:

1. rounded exact-DD and rounded kriging are practically equivalent on W2;
2. exact lattice strongly improves the frozen W3 transfer endpoint over fresh
   static;
3. W3 adds only the -280 micrometer rounded state, whose production kriging
   kernel metrics all lie inside the W2 state envelope.

Given that triangulation, another W3 GPU sort has low marginal value and was
not launched.

## Cached W3 result

The verified external packet is
`/home/huklaban5/DARTsort_experiment_scratch/ef_w3_lattice_20260928/ej_v1/analysis`.
Its manifest SHA-256 is
`984c33cbfe230d3dc5767d52d74db3d160229eaf214f200e23295872a547484e`.
All ten manifest products, `COMPLETE.json`, and both bound sorting hashes pass.

| Measure | Point | 95% CI |
|---|---:|---:|
| rho, exact lattice minus static | 0.529232 | [0.488508, 0.548062] |
| yield ratio, exact lattice / static | 0.990211 | [0.988295, 0.992245] |
| negative-excursion short-ISI difference | -0.000385 | [-0.001478, 0.000659] |
| flat short-ISI difference | 0.000727 | [0.000308, 0.001141] |

The packet labels W3 as an exposed prospective new-arm transfer, not an
untouched validation window. It uses a population continuity proxy and makes
no biological identity or purity claim. No RF was used.

The compact import audit is
`testing/outputs/em2c_w3_cached_lattice_transfer_20260929/AUDIT.json` (SHA-256
`8de9ecfd80a10d2e70b7f2784eeb74294ab4eea565618c740bfc91854878a91c`);
its packet manifest SHA-256 is
`03f0f2650b154594afe6370c24b5389fc0f99e717b5616337d3e90e473a4edef`.

## New rounded state

W2 exercised shifts `{-240, -200, -160, -120, -80, -40, 0, 40}` micrometers.
W3 uses `{-280, -240, -200, -160, -120, -80, -40, 0}` and therefore adds only
-280 micrometers. Under the required SpikeInterface 0.104.7 interpreter, the
-280 state has the same kernel-delta profile as the already tested -240,
-200, and -160 states: one exact-DD unsupported target, no kriging structural
zero, kernel RMS delta 0.0014174, and maximum absolute kernel delta 0.17315.
Every frozen metric lies within the W2 range.

This was a geometry/kernel check with no voltage read and no sort. Its packet
is `testing/outputs/em2c_w3_new_state_kernel_audit_20260929`; manifest SHA-256
`0280df53b8204f9a08a4fb0fca0f541675362e4e79c688dbb8389b850c2a6544`.

## Pipeline implication

The imec1 rounded-kriging remap now has W2 direct method-comparison evidence
and cached W3 rounded-field transfer support. The remaining scientific gap is
full-session behavior and direct waveform/identity evidence, not another
interpolation-method sweep on the same two windows.
