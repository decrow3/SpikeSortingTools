# AW CPU preparation for AU

Status: **AV handoff valid; full-probe worker remains blocked.**

This preparation performed no sorting, raw-voltage read, GPU work, new motion
fit, or continuous-trajectory injection. It used at most two CPU threads and
the sealed, compact AV handoff.

## Frozen preparation

| Item | Frozen value |
|---|---|
| Evaluation interval | imec1 W2, 900--1240 s (340 s) |
| Donor target | 30 accepted static DARTsort S units |
| Scorer fixture train | synchronized regular 5 Hz, 1 s guards, 1,690 events |
| Fixture array SHA-256 | `17e3cd15fb4965d973dc7d328deda5d0c3525387e17b207f4aefb770c9a13314` |
| Intended population | independent seeded dead-time renewal train per donor; mean 5 Hz; minimum 3 ms refractory; base seed 20260926 |
| Injection states | exact same-column multiples of 40 µm only |
| Matching tolerance | inclusive +/-0.4 ms = 12 samples at 30 kHz |
| Qualification | every donor/state: energy retention >0.99, round-trip PTP ratio 1.00 +/-0.02, centred cosine >=0.99 |
| Resources | CUDA disabled; <=2 CPU threads; one reader; nice 10; target <=8 GB working set |
| Persistent-output ceiling | <=5 GB |
| Disk guard | stop below 30 GB free |

The donor rule remains frozen before injected outcomes: accepted static-bank
label `good`, refractory-violation fraction <=0.005, rest-spike fraction >=0.80
and isolation score >=0.80. Six equal depth strata contribute five donors each,
ranked by static PTP descending, refractory fraction ascending, then unit id.
Fewer than 30 eligible donors is a hard gate.

## Handoff and field validation

The sealed bundle is
`/mnt/NPX/Luke/DARTsort_motion_experiments/av_aw_handoff_20260926_v1/`.
All 21 files pass their recorded size and SHA-256. The manifest hash is
`3cc7b0a3840e3d48cdcb0fe5b5707dd2eeea06942cab792294b1ab3e459ac843`;
`COMPLETE.json` is
`71ee43721c7e952d677a32a3f4b7b07b9c54579c11389c382ebe8d766656ff46`.

The D2L-v1 source is authoritatively confirmed at
`85062a37f38b3c5212393d627fe387b629fedd95068a4136a72f330b5aa4c2d9`,
on the AP-frame-zero 0.25 s grid with sign
`corrected = observed - displacement`. The exact local adapter copy differs by
0 µm at copied samples and applied neither filtering nor resampling.

## Actual matching operator and trajectory

At DARTsort commit `edcfe1b51d672b4136eb13cc78c0875da804b851`,
`peel/matching.py:260-269` evaluates the field at the matching-chunk centre.
Both S and D2L used `template_type=drifty`, template channel selection and
thin-plate kriging with a 200 µm neighborhood. S used 30,000-sample chunks;
D2L used 7,500-sample chunks. Thus D2L used the unrounded field at 0.25 s chunk
centres to continuously spatially interpolate the registered template basis.
Integer pitch support selection is a separate path and does not imply that
fractional interpolation was absent.

W2 has 1,360 D2L chunk centres. Quantized injection-state counts are 0: 1,140;
-40: 42; -80: 29; -120: 34; -160: 67; -200: 41; -240: 7. Median absolute
quantization error is 7.53 µm (P95 17.81, maximum 19.99); the episode median is
9.47 µm and rest median 7.34 µm. Local-adapter versus authoritative source
sampling differs by at most `4.24e-11` µm at these centres.

AW's injection operator remains intentionally lattice-safe: it rounds half away
from zero, preserves x-column identity, and requires an exact target at
`(x, y + state)`. Missing, ambiguous or many-to-one mappings fail. This injected
operator is not mis-described as DARTsort's continuous matcher; their difference
is explicitly measured as quantization error.

## Fixture versus intended population

The regular synchronized 5 Hz train is retained only to test the exclusive
per-cluster scorer and its +/-0.4 ms boundary. It must not become the eventual
population truth. The production primitive now creates distinct, immutable,
order-independent per-unit trains using a unit-specific seeded renewal process,
mean 5 Hz and minimum 3 ms refractory. Ten focused preparation tests and 40
combined AW/injected-truth/scorer regressions pass.

## Donor columns and crop feasibility

The saved S bank is 654 x 121 x 182, AP202--AP383, SHA-256
`a99b12c3075f038fbad8c05c36c96f63221fd0eac5ba71caee8f18ff97acb75c`;
the final sorting hash is
`be6106ed0cb0f99759fd23629dd3bcb5f50657dbffa238ad3721c15d1b21fba7`.

The crop directly supplies `unit_id`. Depth, template PTP, refractory fraction
and rest fraction are potentially derivable, but are not present as sealed
columns. Template PTP cannot silently be relabelled physical `ptp_uv`, and the
refractory definition must be frozen before calculation. The required `quality`
label and `isolation_score` are absent and not uniquely derivable. The receipt
references small `qc-phy` outputs; staging those exact files is preferable to
redefining quality.

With a 440 µm edge margin (240 µm maximum state plus the actual 200 µm matcher
neighborhood), only 7/654 templates have >99% of their observed crop energy in
the interior. 375/654 pass every occupied exact-state round trip; only 7 pass
both screens. This cannot reach the frozen 30-donor target, even as a
crop-specific cohort. The crop also cannot establish full-probe energy because
AP0--AP201 are unobserved, not zeros.

## Smallest bounded full-probe extraction

After the missing quality table selects 30 donors, reuse their accepted S spike
times and the exact W2 preprocessing graph, stopping before any sort:

1. Read only imec1 AP 900--1240 s once (about 7.83 GB raw).
2. Apply the recorded 384-channel `ibllikecmr` float32 reference before crop;
   hold the approximately 15.67 GB temporary cache in RAM.
3. Extract the recorded 121-sample waveforms for at most 500 accepted S spikes
   per selected donor in small batches, reproduce the median/SVD template
   recipe, and persist only the 30 full-probe templates, geometry, channel ids
   and receipt (raw float32 template payload about 5.58 MB).
4. Delete the RAM cache after hashes and crop-overlap equivalence pass.

The saved W2 preprocessing took 634 s while already referencing all 384
channels. Allow 10--15 minutes for preprocessing and 10--20 minutes for bounded
waveform/template extraction: about 20--35 minutes CPU wall time, one 7.83 GB
source read, 15.67 GB temporary RAM and under 0.2 GB persistent output. This is
an estimate and recipe, not authorization to run it.

## Remaining order and interpretation limits

1. Stage the exact S/W2 quality/isolation table.
2. Select 30 donors under the frozen rule.
3. Perform the bounded full-probe template extraction or receive an equivalent
   saved full-probe source.
4. Qualify every selected donor in every occupied state.
5. Materialize and hash the independent per-unit truth trains and worker assets.

`worker_ready` remains false until these gates pass. AV's chronological split,
fixed left/right builder self-review and unpaired representative templates are
accepted corrections. AV continuity observations remain exploratory, and T6
is not used to infer unit quality.
