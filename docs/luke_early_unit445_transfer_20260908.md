# Unit 445: bounded transfer to 930–1030 s

The frozen unit445 matcher accepted141events. Seven of ten10-second windows have≥10matches; only3adaptive≤2s windows are supported and no100-spike window satisfies the existing gap/span rules. This is provisional template transfer, not confirmation that these are the same neuron as at4080–4260s.

The figure overlays original-input waveform energy centroids and10-second median-waveform centroids on original/compensated peak-count rasters. The same compensated5σ DREDGE field at2260µm is repeated in both columns. Its vertical offset aligns the median DREDGE prediction at all accepted spike times to the median waveform centroid of those spikes (2253.7896µm). The usual970–975s reference lacks support. No sign, gain or lag is fitted. Bootstrap intervals are conditional and do not capture identity, selection, or displacement-calibration uncertainty.

DREDGE shows large brief negative excursions while accepted waveform summaries usually remain near their reference position. The matcher uses fixed spatial support and can lose a moving cell; the sparse observations cannot validate or refute those excursions. Particularly sparse970–990s bins contain1and4matches. The990–1000s supported summary differs by about12.4µm from DREDGE sampled at its accepted events; this is a discrepancy to inspect, not established motion error.

Frozen controls: original300–6000Hz/global median referenced voltage,4080–4090s template, existing threshold0.885, five competitor templates, gain0.4–2.5, competitor margin0.03, ±3sample timing, ±20µm detection and±60µm waveform support. Reconstructed training template exactly equals the saved unit445 template. No DREDGE information entered matching. Saved per-chunk waveforms and full scored decisions permit review; there is no automatic within-stage resume.

Detached service `luke-early-unit445-v1` completed successfully (exit0,126.6seconds); post-launch liveness verified, final MainPID0/inactive checked. Durable launch/settings/logs/receipts reside in `testing/outputs/luke_early_unit445_v1_job`. No sorting or new motion estimation was run.

Figure: [Sparse support and DREDGE](../testing/outputs/luke_early_unit445_v1/02_sparse_support_dredge.png). [PDF](../testing/outputs/luke_early_unit445_v1/02_sparse_support_dredge.pdf). Extraction: `testing/luke_early_unit445_v1.py`; cached renderer: `testing/luke_early_unit445_report_v1.py`; numerical observations and traces: `testing/outputs/luke_early_unit445_v1/`.
