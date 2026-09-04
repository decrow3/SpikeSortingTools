# Luke 2025-08-04 imec1 upstream ablation

> **SCOPE NOTE — 2026-09-03.** The intervention result remains valid for the
> tested external DREDGE warp in the prespecified 120 s imec1 window: removing
> that warp improved the measured outcomes. It does not show that motion
> correction as a class is harmful, that unwarped rescue is globally better
> than legacy, or that the present pipeline handles motion correctly.

## Technical summary

The tested external DREDGE correction is the leading verified upstream cause of the
pathological claim-off peeling behavior in the 8,160--8,280 s imec1 window.
Removing external motion correction reduced learned-template detections by
57%, reduced the cross-unit near-coincident spike fraction from 0.770 to
0.365, improved visually neural unmatched-event recovery from 70.4% to 85.2%,
and increased KS-good units from 74 to 95. This is not a generic reduction in
sorter yield: the cleaner condition returned more good units and recovered
more fixed events despite producing far fewer spikes.

Preserving floating-point precision through the filter and reference stages
changed waveform-level fragmentation metrics but did not materially change the
sorter outcome. It therefore remains a worthwhile implementation cleanup, but
it does not explain the large peeling failure. The evidence identifies the
external motion-correction branch, not yet whether the failure originates in
the estimated displacement field or in how that field is interpolated.

## Motion correction, not claim masking, controls the failure

All four Kilosort runs used the same 120-second raw frame range, Kilosort
settings, probe geometry, and disabled claim mask. The factors were external
DREDGE correction on/off and intermediate conditioning precision int16/float.
The `current_motion` row reuses the already completed claim-off baseline.

| Condition | Learned detections | Final spikes | Units | KS-good | Median contamination | Cross-unit coincidence | Median refractory fraction | Neural unmatched recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current conditioning + DREDGE | 1,298,050 | 905,605 | 256 | 74 | 51.55% | 0.770 | 0.0218 | 70.4% |
| Current conditioning, no motion | 562,417 | 544,553 | 242 | 95 | 38.65% | 0.365 | 0.0147 | 85.2% |
| Float conditioning, no motion | 555,292 | 538,917 | 250 | 91 | 38.35% | 0.358 | 0.0150 | 85.2% |
| Float conditioning + DREDGE | 1,293,480 | 867,057 | 254 | 87 | 46.05% | 0.762 | 0.0207 | 70.4% |

Relative to the current DREDGE input, removing motion correction produced:

- 56.7% fewer learned-template detections;
- 39.9% fewer final spikes;
- 52.5% lower cross-unit near-coincident burden;
- 25.0% lower median contamination;
- 32.5% lower median refractory-violation fraction;
- 14.8 percentage points more reviewed-neural event recovery; and
- 28.4% more KS-good units.

This combination is inconsistent with a simple sensitivity loss. It is
consistent with DREDGE-corrected waveforms or residuals being repeatedly fit by
learned templates and then distributed across clusters.

Float conditioning did not rescue the DREDGE condition: learned detections
fell only 0.35%, collision burden fell only 0.95%, and reviewed-neural recovery
did not change. Likewise, without motion, float conditioning changed learned
detections by only 1.3% and did not change reviewed-neural recovery.

## The motion field is extreme where evidence is weaker

The reviewed event times are frame-relative, whereas the saved DREDGE temporal
bins use the raw extractor's 3,057.678 s start time. Metrics below apply that
offset. Omitting it silently inspects the wrong part of the displacement field.

| Window | Detected peaks/s for motion | Median peaks per 1 s × depth bin | Bins below 20 peaks | Median nonrigid spread | P95 nonrigid spread | Maximum spread |
|---|---:|---:|---:|---:|---:|---:|
| 7,095--7,335 s | 4,760 | 80 | 12.4% | 21.6 µm | 38.0 µm | 52.7 µm |
| 8,160--8,280 s | 3,329 | 55 | 18.8% | 24.6 µm | 38.3 µm | 52.3 µm |

The pathological window is not devoid of peaks: only 0.27% of its 1 s × depth
bins are empty. It nevertheless has 30% fewer motion-detection peaks per second
and 51% more low-support bins than the shared/template window, while reaching
an equally extreme nonrigid spread. This supports, but does not prove, a
poorly constrained nonrigid warp.

## Conditioning precision is a secondary issue

SpikeInterface preserves the raw `int16` dtype through the current 12th-order
filter and local-reference stages. Across the 200 fixed reviewed events, median
exact-zero occupancy rose from 2.1% after bandpass to 15.8% after the current
local reference and 17.1% after motion correction. A float-preserving
counterfactual reduced pre-motion zeros to zero and, for visually neural events
in the pathological window, reduced the median number of extra temporal
extrema by three and spatial peaks by two without reducing median event
amplitude.

Those waveform changes did not translate into a meaningful claim-off sorter
improvement. Intermediate quantization should therefore be cleaned up and
tested, but it is not the primary explanation for this dataset.

## Scope and metric definitions

- **Cohort:** 120 s of Luke 2025-08-04 imec1, frame-relative 8,160--8,280 s.
- **Fixed events:** 74 blinded-review events in this window, including 27
  visually neural events originally unmatched to the patched sort.
- **Learned detections:** rows in Kilosort `full_st.npy`, before final duplicate
  removal and export.
- **Final spikes:** exported spikes within the real 120 s recording boundary;
  Kilosort batch-padding samples are excluded.
- **Cross-unit coincidence:** fraction of final spikes participating in a pair
  from different clusters within 0.5 ms and 75 µm.
- **Recovery:** fraction of fixed raw events with a final sorted spike within
  0.5 ms and 100 µm. The visual-neural result is descriptive because the
  five-channel review overcalled some broad events; the conservative automatic
  unmatched subset is empty in this window.

## Methodology

`testing/luke_upstream_stage_ablation.py` reconstructs the production
conditioning graph lazily and measures every reviewed event after incremental
conditioning stages, local/global reference controls, DREDGE interpolation,
and float-preserving controls. It also measures DREDGE displacement and peak
support using the recording's absolute time origin.

`testing/luke_upstream_sorter_ablation.py` materializes only the predefined
120-second window and runs the 2 × 2 causal contrast. Claim masking is disabled
in every condition so it cannot hide the failure being measured. Production
pipeline defaults are unchanged.

## Limitations and robustness

- The causal result applies directly to one prespecified pathological window
  on Luke imec1. It should be replicated in the shared/template window and on
  imec0 before changing full-session defaults.
- Disabling external DREDGE here also leaves Kilosort internal correction off.
  The result therefore establishes that the current DREDGE-corrected input is
  harmful relative to no correction, not that no motion correction is optimal.
- The saved DREDGE run discarded correlation/weight diagnostics, so estimator
  support is approximated by detected-peak density rather than judged from the
  native DREDGE objective.
- Unit counts and KS labels are not ground truth. The fixed-event recovery,
  contamination, collision, and refractory results point in the same direction
  and make the conclusion substantially stronger than unit count alone.
- The universal-template detection counts for the new controls were not
  persisted by Kilosort. The result localizes the failure upstream from the
  claim mask and demonstrates learned-detection inflation, but does not yet
  partition the full change between Kilosort's universal and learned passes.

## Recommended next steps

1. Run no external correction with Kilosort internal motion correction enabled
   on this same window. Compare it with both current DREDGE and no correction;
   this is the most decision-relevant candidate for a full-session finalist.
2. Split DREDGE into identity interpolation, rigid-only displacement, current
   nonrigid displacement, and a smoothed/clipped nonrigid field. This will
   distinguish interpolation mechanics from estimator overfitting.
3. Re-estimate DREDGE in a cache-safe experiment that preserves correlation
   weights and peak counts. Test whether high-gradient bins are reproducible
   under nearby temporal and spatial regularization settings.
4. Replicate the winning contrast in the 7,095--7,335 s window and on imec0.
   Only then advance at most two finalists to a full-session sort.
5. Keep a mild claim mask as a downstream safeguard until the upstream motion
   branch is corrected; do not tune it as the primary remedy.

## Further questions

- Does the sorter failure track displacement magnitude, spatial gradient, or
  low DREDGE support across shorter windows?
- Does rigid-only correction preserve event recovery without recreating the
  collision burden?
- Is Luke unusually sensitive to this interpolation compared with a matched
  Yates window processed through the same exact graph?
