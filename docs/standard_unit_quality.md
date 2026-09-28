# Standard per-unit quality bundle

The identity-guarded downstream pipeline now adds a cheap, auditable unit-QC
stage after the historical QC stage. It is additive: legacy NPZ/PDF/MAT files
and curation labels remain unchanged, and no raw voltage is read.

## Run it

```python
from pipeline.downstream import run_standard_qc_stage

receipt = run_standard_qc_stage(
    recording_dir,
    curated_output,
    legacy_qc_dir,
    legacy_qc_dir / "standard",
    sort_identity,
)
```

This can backfill a completed identity-bound sort without rerunning curation or
legacy QC. The stage request hashes every available interpretation-critical
Kilosort array and legacy truncation cache. A changed input or parameter set is
therefore refused rather than silently reusing stale output.

## Outputs

| File | Meaning |
|---|---|
| `unit_quality_metrics.csv` | Measurements, one row per curated unit |
| `unit_quality_flags.csv` | Separate policy annotations and warnings |
| `metric_definitions.json` | Units, definitions, parameters, and caveats |
| `quality_policy.json` | Which external profiles are currently evaluable |
| `quality_summary.json` | Counts, recording metadata, and identity |
| `standard_qc_request.json` | Content-bound request |
| `standard_qc_receipt.json` | Completion receipt and required-file list |

## Metrics implemented without waveform recomputation

- Kilosort label and contamination estimate;
- spike count, full-recording firing rate, active lifetime;
- presence in 60 s and 300 s bins and rate stability in 5 s/300 s bins;
- SpikeInterface-formula ISI violation ratio at 1.5 ms;
- SpikeInterface/Hill refractory contamination at 1 ms, counting all close
  pairs rather than only consecutive ISIs;
- Kilosort feature-amplitude distribution, explicitly labeled as a
  dimensionless feature scale rather than microvolts;
- per-spike depth distribution, binned drift, and probe-edge burden;
- template peak depth/polarity and nearby high-similarity template burden;
- legacy amplitude-truncation coverage, uncensored estimates, 50% boundary
  saturation, and disagreement between the two fit-derived estimates.

The warning table never changes curation. Thresholded warnings are review aids,
not a scalar score or claim that a unit is biologically isolated.

## Relationship to common profiles

Kilosort status is directly evaluable. Allen-like, IBL-like, and common
SpikeInterface example profiles are deliberately marked `not_evaluable` in
this first release because a faithful decision needs metrics that the legacy
artifacts do not contain: calibrated waveform amplitudes/noise, amplitude
cutoff, SNR/noise cutoff, and IBL sliding-refractory output. Kilosort
`amplitudes.npy` is not substituted for those quantities.

That distinction prevents two common errors: calling a feature coefficient
microvolts, and presenting a partially reconstructed institutional policy as
an exact pass/fail label. A later waveform-analysis stage can add those fields
from a pinned `SortingAnalyzer` while preserving this schema and the current
legacy outputs.

## Known boundaries

- Presence is defined over the verified full recording duration; empty periods
  count against presence.
- The legacy truncation fitter requires roughly 1,000 spikes in a continuous
  block. No eligible window means unavailable, not zero missing spikes.
- A legacy truncation value of exactly 50% is boundary-censored (at least 50%),
  so it is counted but excluded from measured medians and percentiles.
- Drift bins require 100 finite spike depths. A unit with fewer than two valid
  bins has no drift-range estimate.
- Template similarity is a duplicate/over-splitting warning only; it is not a
  merge instruction without coincidence/CCG and identity evidence.
