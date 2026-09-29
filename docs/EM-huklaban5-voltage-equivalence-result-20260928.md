# EM.1 huklaban5 voltage-equivalence result — 2026-09-28

**Verdict: EM.1 exact-adapter qualification failed; stop before EM.2a.** The
stock SpikeInterface 0.104.7 nearest operator remains disqualified by its four
AP191 interior-hole substitutions. After repairing two handoff-runner defects,
the exact adapter reproduced the flat `q=0` control byte for byte and reproduced
all zero positions in the two nonzero-motion snippets, but it did not reproduce
all float32 bytes under the required SpikeInterface 0.104.7 execution path.
No sort was launched, no voltage was exported, and no outer holdout was read.

## Frozen actual-voltage result

The final preserved attempt is
`/home/huklaban5/DARTsort_experiment_scratch/em_spikeinterface_lattice_20260928/voltage_equivalence_v4`.
It used the three preselected 3-second windows and the original DD 30,000-frame
parent-read chunks.

| Window | Nominal start (s) | Byte equal | Zero mask equal | Maximum absolute error |
|---|---:|---:|---:|---:|
| low-deviation flat (`q=0`) | 1118.625 | yes | yes | 0 |
| episode core | 1031.625 | no | yes | 1.4901161e-8 |
| transition | 938.375 | no | yes | 2.9802322e-8 |

The first episode-core difference was at relative frame 23,468, target AP330,
with `q=-200` um. DD selected AP310. SpikeInterface 0.104.7 reconstructed the
source value as float32 bytes `9133f33d`; the DD materialization contains
`8f33f33d`. The first transition difference was at relative frame 18,350,
target AP360, with `q=-120` um. DD selected AP348; 0.104.7 produced
`09f59dbe` and DD contains `08f59dbe`.

The same DD-aligned source samples reconstructed under the original
SpikeInterface 0.104.8 producer environment exactly match the stored DD bytes.
Thus the residual is upstream preprocessing-version sensitivity, not a different
lattice source index or zero-fill decision. This diagnosis does not turn the
required 0.104.7 result into a pass.

## Preserved execution history

1. `voltage_equivalence_v1` stopped before reading voltage because the frozen
   runner loaded a portable recording without its required `base_folder`.
2. `voltage_equivalence_v2` stopped before reading voltage because the runner
   used the parent's retained acquisition clock instead of DD's source-frame
   clock (`START / FS`).
3. `voltage_equivalence_v3` reached all three snippets but requested arbitrary
   parent ranges. The lazy preprocessing chain is chunk-boundary sensitive, so
   even `q=0` differed by up to 0.0406 uV.
4. `voltage_equivalence_v4` reproduced DD's original 30,000-frame parent reads.
   Its flat control passed exactly; the two nonzero-motion snippets retained the
   1–2-ULP 0.104.7 versus 0.104.8 differences reported above.

Every failed attempt has a `FAILURE.json`; v3 and v4 also have `SELECTIONS.json`
and `RESULT.json`. `COMPLETE.json` is absent because equivalence did not pass.
The repaired runner now persists failures, resolves portable recording paths,
uses the DD receipt's source-frame clock, and reproduces the materializer's
parent-read chunks. Twenty-one focused adapter/runner tests pass.

The 1 GB EM.1 voltage-read target was exceeded during diagnosis. The first
scientific comparison and its chunk-corrected repeat together read roughly
1.36 GB of logical parent-plus-reference trace data, before small direct sample
checks. This was caused by the two runner defects and the required chunk-boundary
diagnosis; it is disclosed rather than silently resetting the budget.

## Decision

Do not start EM.2a, EM.2b, RF work, or W3 from this result. EM.2a is explicitly
conditional on an EM.1 pass for the selected implementation, and that gate is
not met. A future plan could freeze source values produced by the original
0.104.8 preprocessing environment and test only the adapter boundary under
0.104.7, or standardize the full pipeline on one preprocessing version. That
would be a new experiment and must not be described as the completed EM.1 run.

Key compact evidence hashes from v4:

- `RESULT.json`: `ed8c621e1a9fafa9659e28309b74f5ed15549de5db1b85b8dd16e42885eb60d7`
- `SELECTIONS.json`: `06ec30141cb7651f029d0d7623ac7a661417e1725e3d44fe8222a90a4f898a1a`
- `FAILURE.json`: `a592c2290bff127e058e626a0c5ff9e15e3941f2ed46debe75ddfec1bf9b8be5`

