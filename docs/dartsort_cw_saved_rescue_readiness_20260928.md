# CW saved rescue-sort readiness

## Verdict

The completed imec1 rescue Kilosort 12/9 no-motion sort is available and is
eligible for a bounded saved-output comparison on the exact CV W2 and W3 source
frame domains. No new sort or raw-voltage read is needed. It is a sorter-derived
control, not independent biological truth.

## Identity and clock

- Sort root: `/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1/kilosort4/`
- `Th_universal=12`, `Th_learned=9`, `do_correction=false`.
- The recording manifest starts at AP frame zero, covers 314,204,094 frames, and
  reports 29,999.759166666667 Hz—the exact frequency used by CV.
- `spike_times.npy` is monotonically sorted, spans frames 1--314,204,078, and is
  therefore directly sliceable on CV source frames without a new clock mapping.
- W2 `[26,999,783, 37,199,702)` contains 1,411,980 spikes from 530 units; 193
  of the 216 saved KS-good units occur there (123,352 good-labelled spikes).
- W3 `[239,998,073, 250,197,992)` contains 1,194,681 spikes from 551 units; 202
  KS-good units occur there (151,431 good-labelled spikes).

Saved `spike_clusters.npy`, `spike_positions.npy`, `templates.npy`, channel
geometry, and cluster labels are present and row-compatible with the event
array. The large per-spike feature arrays are unnecessary for this bounded use.

## Eligibility limits

The sort used a distinct Kilosort pipeline and its per-spike positions/templates
are sorting-conditioned. Any comparison must freeze the W2/W3 source-frame
cuts above, keep KS-good versus all-unit denominators explicit, and must not call
stationarity, waveform agreement, or cross-sort correspondence biological
identity. The saved outputs establish availability, not validity of a new
classifier or permission to tune one after CV/CW outcomes.

## Hashes

- rescue sort manifest: `20bd9c282721205107018b39e6b9945e075162394d40e3779d26b1eda0e001c7`
- recording manifest: `9ff4ed1e59c9528696d7d7a99dd16ba0749fe71b820d31738da411091b767f75`
- spike times: `ecc6fe1244757fdaf4c326190b84930396a0b6e20ccb03e28ab9ca6e50c334e5`
- spike clusters: `483f04d365b221b70bb39d87f407c7082bc2d9f39397e4851d44dd29dd944b82`
- spike positions: `5153b87aa0b1b301e2c5a8ecc8420ff0c65c8e570a1a6416acfe9bc15f43fe6b`
- templates: `63828382337ffda64a2d61aef03b639ffabfad0723294a56148aabeb61d710cf`
- cluster labels: `b81d1a2e5597599cf3a9a263793334c45c72ebd967be94317239973f90588eb2`

