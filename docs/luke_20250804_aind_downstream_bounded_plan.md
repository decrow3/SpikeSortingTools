# Luke bounded pinned-AIND downstream comparison

> **COMPLETED HISTORICAL PLAN.** The bounded design and its measurements remain
> usable. Describing rescue as the “locked reference” identifies the comparator
> chosen for this experiment; it is not a claim that rescue is biologically or
> detection-wise superior to legacy. Current interpretation is in
> [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md).

## Decision being tested

The frozen rescue graph is the locked downstream Luke reference. The historical
imec0 evaluator remains `reject_universal_default`, but detailed follow-up
localized its substantive failure to four artifact-associated questionable
units; conservatively discounting all four still leaves 297 KS-good units,
14.2% above legacy.

This comparison asks whether the independently selected pinned AIND
preprocessing branch improves downstream sorting enough to replace that strong
reference. It is not described as a CMR-only test because the AIND branch also
omits the rescue blanker and removes AP191 instead of interpolating it.

## Frozen conditions

All conditions use the rescue Kilosort settings, internal high-pass at 300 Hz,
`skip_kilosort_preprocessing=false`, no Kilosort motion correction, no voltage
motion interpolation, no claim mask, and no Kilosort batch artifact rejection.
Only the declared preprocessing graph and mechanistic KS-CAR ablation differ.

1. `rescue_ks_car_on`: accepted rescue voltage—phase correction, bilateral
   samplewise 500 µV blanking and AP191 interpolation—then KS CAR on.
2. `pinned_aind_ks_car_on`: pinned AIND voltage—phase shift, external 300 Hz
   high-pass, frozen AP191 removal and global median reference—with KS CAR on.
3. `pinned_aind_ks_car_off`: identical pinned AIND voltage with KS CAR off.

The artifact sidecar remains an annotation and never changes sorter input.

## Exact pinned AIND boundary

- Benchmark repository HEAD:
  `306aa05f1491b48d58433164ad5797cd1d1358e1`
- AIND upstream commit:
  `653de8f0471de65cd138f1815dd6ca770673b7cf`
- Python `3.12.4`; SpikeInterface `0.104.7`
- Branch-config digest:
  `8cce687e31337f2e9869e5756110f9a5b26d245248b56ab3d7b57213eae81d68`
- Dataset-manifest digest:
  `82d933695c95535f404ee108f3cb856ea5329784a0fd37ace31d67073a06f661`
- Bad-channel-manifest digest:
  `d084759fb30791c445fa2aa5be0d0ebe4da6e7c9545f85b494059b1fd7a2db55`

The exact graph is:

1. `phase_shift(margin_ms=100.0)`;
2. `highpass_filter(freq_min=300.0)` with SpikeInterface 0.104.7 defaults;
3. `detect_and_remove_bad_channels` with frozen precomputed identity AP191;
4. `common_reference(reference="global", operator="median")` over 383 channels.

AP191 was independently labeled dead on both probes. Bounded runs do not
redetect channels; they remove `imec0.ap#AP191` or `imec1.ap#AP191`
deterministically. No other channel was rejected. There is no saturation
blanking, clipping, interpolation, repair or artifact removal. Materialization
uses `get_traces(return_in_uV=True)` and stores float32 µV.

## Bounded panel

The panel uses windows frozen independently of these sorter outcomes:

| Panel | Range | Sealed cohorts |
|---|---:|---|
| `T1_high_motion` | 1200–1320 s | T1 high motion |
| `T2_combined` | 4680–4920 s | T2 relative quiet and high motion |
| `T3_combined` | 9000–9240 s | T3 high motion and relative quiet |

This is 600 seconds per probe and retains five of six sealed windows, totaling
360 of 432 sealed events per probe. `T1_relative_quiet` is omitted because it is
temporally isolated and would add six more sorts. The panel still spans all
three session thirds and both motion strata.

Three conditions × three windows × two probes produce 18 sorts. The primary
endpoint families are sealed-event recovery, detection expansion, coincidence,
refractory behavior, residuals, duplicate burden and continuity. KS-good count
and contamination remain secondary diagnostics.

## Execution and safeguards

The machine-readable plan is
`testing/configs/luke_aind_downstream_bounded_v1.json`; the guarded runner is
`testing/luke_aind_downstream_bounded.py`. Recording and sort outputs use
partial directories followed by atomic acceptance. Every prepared binary has a
size check, SHA-256 and request digest. Every completed sort validates saved
`do_CAR`, `nblocks=0` and `highpass_cutoff=300` before acceptance.

The run was launched as persistent user service
`luke-aind-downstream-bounded-v1.service`. Outputs are written under
`/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_aind_downstream_bounded_v1`.
The durable log is `logs/run.log` under that root.
