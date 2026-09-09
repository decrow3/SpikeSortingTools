# Early lighthouse temporal support:930–1030s

Extended the two existing frozen candidate matchers through930–1030s, retaining original300–6000Hz/globalmedianreferenced waveform processing. No DREDGE field entered matching. The970–990s acceptedframes were reused exactly; all supported original5s medianwaveforms were bit-identical and centroids reproduced within1e-8µm. Two provisional candidates yielded1389matches at2920µm and2963at3380µm. The960–970s template-training period lies inside this view and is shaded explicitly. No new identity qualification is implied.

Recomputed the centroid of each window's median waveform directly from accepted per-event waveforms. Fixed bins at0.25,0.5,1,2,and5s require10events. Also evaluated adaptive0.25–2s windows with10events minimum, and non-overlapping groups of100consecutive spikes. Event-count groups split at interspike gaps>2s and cannot span>20s; incomplete tails remain unsupported. All window memberships freeze before reading baseline DREDGE, which is summarized at exactly the same accepted spike times. Both measurements use fixed970–975s reference offsets.

| Depth | Accepted events | Supported0.25s bins | Supported adaptive windows | Supported100-spike groups | Median supported100-spike duration |
|---|---:|---:|---:|---:|---:|
|2920µm|1389|43/400|93/108|11|6.01s|
|3380µm|2963|128/400|188/196|29|2.52s|

The100-spike support duration ranges2.61–11.55s at2920µm and1.21–9.49s at3380µm. Thus constant-event windows give more consistent event counts but variable temporal resolution; they can average over brief movement. No points are connected across unsupported periods. A maximum span and gap rule remain necessary. Future overlapping windows could update more often, but would introduce dependent estimates rather than new independent evidence.

Vertical uncertainty bars use100within-window event bootstrap samples. They condition on accepted identity and a fixed reference offset, omit reference/model/identity uncertainty, and are not calibrated physical error bounds. Fixed-template support and sparse competitor libraries can reject movers or confuse cells. More densely sampled centroids do not resolve those limitations.

Both extraction and reporting completed with exit0 under independent service `luke-early-lighthouse-events-v2`. Actual final state inactive/MainPID0; commands, receipts, logs and manager snapshots in `testing/outputs/luke_early_lighthouse_events_v2_job/`. Core outputs are versioned and preserve per20s waveforms, but there is no automatic/within-stage resume. Synthetic bin partition, maximum duration and interspike-gap invariants passed.

[Motion comparison](../testing/outputs/luke_early_lighthouse_binning_v2/01_motion_binning.png) · [PDF](../testing/outputs/luke_early_lighthouse_binning_v2/01_motion_binning.pdf) · [Quarter-second spike counts](../testing/outputs/luke_early_lighthouse_binning_v2/02_spike_support.png). Tables for every tested width are in `windows_with_motion.csv` and `support_summary.csv` in the same output directory.
