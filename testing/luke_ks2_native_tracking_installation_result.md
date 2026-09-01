# Luke KS2 native tracking installation result

## Decision

**Stop before the six-segment scientific panel.** The native MATLAB KS2
installation is reproducible and its internal waveform-state diagnostics are
preserved, but neither audited biological smoke configuration passes the
preregistered no-periodic-detection-trough gate.

This is a software-integrity failure, not evidence for or against KS2's
waveform-state tracking mechanism. The accepted KS4 no-motion sort remains the
production comparator.

## Pinned implementation

- upstream: MouseLand Kilosort2 `v2.0.2`;
- commit: `0ce102799e69b97e3364ae47b403a809712d7e15`;
- tracked source: clean;
- source/MEX manifest: [`luke_ks2_v2.0.2_pin.json`](luke_ks2_v2.0.2_pin.json);
- MATLAB: R2022b (`9.13.0.2049777`), licensed;
- GPU: NVIDIA RTX A5000, compute capability 8.6;
- CUDA compiler: 11.5 with GCC/G++ 10.5;
- eight MEX modules: load-tested and hash-pinned; and
- runner: [`luke_ks2_native_tracking.py`](luke_ks2_native_tracking.py) with
  [`run_luke_ks2_native.m`](matlab/run_luke_ks2_native.m).

The official tag differs from the earlier `v2.0` source in two relevant
places: `configFile384.m` changes `NT` from `65536 + ntbuff` to
`65536 - ntbuff`, and `mexMPnu8.cu` adds the two boundary guards described by
the patch. The local SpikeInterface 0.102.1 wrapper's stale automatic
`65536 + ntbuff` default was never accepted silently; every executed `ops.NT`
was explicit and receipt-recorded.

## Deterministic fixtures

The 60-s, 32-channel fixture contains strong deterministic events at boundary
and matched interior phases. Both configurations completed twice with
byte-identical input and output hashes within configuration.

| Configuration | Raw stride | Pooled boundary ratio | Result |
|---|---:|---:|---|
| Published-minus grid: `NT=65472`, `ntbuff=64` | 65408 | 0.667 | Reproducible, but later disproven on biological high-rate data |
| Coverage-aligned grid: `NT=65600`, `ntbuff=64` | 65536 | 1.013 | Reproducible fixture pass |

The aligned fixture's six frozen output hashes were identical across its two
runs. Its detected phases extended through sample 65,506, demonstrating that
the gross coverage gap seen below was eliminated. The fixture pass was
necessary but not sufficient: its sparse, deliberately balanced events did
not reproduce the remaining biological boundary loss.

## Frozen rapid-motion input

Both biological smoke runs used the same accepted, unwarped 5910–6030 s
rescue slice:

- 3,599,971 samples;
- 384 channels;
- int16;
- 29,999.7591667 Hz;
- 2,764,777,728 bytes; and
- SHA-256
  `c68cab45f7828c3b09763b492065e48d4bd9e32e5020a9c8b7d1b23b632a5096`.

No voltage motion correction, spatial interpolation, or skipped KS2 frontend
was used.

## Published-minus-grid smoke

Artifact:
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/sorter_bakeoff/windows/rapid_motion-8b4978262d/ks2_native`

- executed batching: `NT=65472`, `ntbuff=64`, raw stride 65,408;
- 56 batches;
- 243 retained channels after KS2's native good-channel check;
- 404,019 pre-cutoff events;
- 233,001 exported events;
- 107 KS2-good units;
- pooled 7.03-ms boundary/interior ratio: 0.3878;
- pseudo-boundary percentile: 1.31%; and
- pre-cutoff spike phases ended at 64,553, leaving phases 64,554–65,407
  empty in every complete stride (854 samples, 28.47 ms).

The auxiliary conjunctive cutoff required a percentile below 1%, so this arm
narrowly missed that numerical fail condition. It nevertheless fails the
installation gate's controlling requirement that there be no material
periodic trough. The gap is already present in `rez_tracking.mat`, before
merges, splits, cutoff, or export.

The source mechanism is inspectable. In v2.0.2, the raw-reader stride is
`NT-ntbuff=65408`, while the 1024-thread convolution loop does not cover the
last partial block under the published-minus grid. This is not a clock-analysis
artifact.

## Coverage-aligned smoke

Artifact:
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/sorter_bakeoff/windows/rapid_motion-8b4978262d/ks2_native_aligned`

- executed batching: `NT=65600`, `ntbuff=64`, raw stride 65,536;
- 55 batches;
- 244 retained channels;
- 411,299 pre-cutoff events;
- 240,268 exported events;
- 103 KS2-good units;
- pooled boundary/interior counts: 552/805;
- pooled boundary ratio: 0.6857;
- pseudo-boundary percentile: 0.10%; and
- overall empirical batch-phase gate: **fail**.

Major subgroup results confirm that this is not driven by one small family:

| Subgroup | Boundary ratio | Pseudo-boundary percentile | Gate |
|---|---:|---:|---|
| Pooled | 0.686 | 0.10% | Fail |
| High-rate units | 0.680 | 0.10% | Fail |
| KS2 MUA | 0.665 | 0.10% | Fail |
| Depth quartile 1 | 0.753 | 0.80% | Fail |
| Depth quartile 2 | 0.614 | 0.10% | Fail |
| Depth quartile 3 | 0.642 | 0.00% | Fail |
| KS2 good | 0.792 | 8.40% | Pass |
| Depth quartile 4 | 0.749 | 2.60% | Pass |

The coverage-aligned grid therefore fixes the gross 28-ms empty interval but
does not clear the historical-scale boundary-detection bias.

## Preserved diagnostics

Each biological smoke retains:

- exact `ops.mat` and `chanMap.mat`;
- input binary and SHA-256 receipt;
- 0.5-ms pooled/subgroup phase histogram CSV and PNG;
- circular pseudo-boundary audit JSON;
- chronological and reordered batch matrices, batch order, and latent state;
- `rez_pretracking.mat`;
- `rez_tracking.mat`, including time-varying template state;
- `rez_final.mat`;
- standard Phy/NumPy exports; and
- complete MATLAB and launcher logs.

For the aligned run the three `rez` checkpoints are approximately 120 KiB,
253 MiB, and 326 MiB.

## Consequence

Do not run the relative-quiet, moderate-motion, sustained-noise,
support-dropout, motion-plus-anomaly, or 10-min large-motion segments with
this installation. Do not interpret the smoke's unit count, good labels, or
apparent family continuity as biological evidence.

A future restart requires a separately pinned KS2 implementation or audited
source patch that first passes:

1. the deterministic fixture twice with identical hashes;
2. a high-rate stationary/biological boundary fixture;
3. the 120-s Luke smoke with pooled ratio at least 0.98 unless it is not below
   the 1st-percentile null, and every major subgroup at least 0.95 under the
   same rule; and
4. direct confirmation that no contiguous raw-stride phase interval is
   structurally uncovered.

Only after those software gates pass should the frozen six-segment panel be
reopened.
