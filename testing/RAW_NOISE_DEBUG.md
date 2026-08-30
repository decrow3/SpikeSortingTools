# Raw probe-noise debugging analysis

`analyze_raw_probe_noise.py` is an exploratory analysis only. It does not import
or modify the canonical `pipeline`, write a recording, or change a Kilosort
input. It reads deterministic windows from the raw SpikeGLX AP binary and uses
the existing shallow-sweep crop as its spatial selection.

## Primary 2026-03-16 run

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
/home/huklaban5/anaconda3/envs/spikeinterface/bin/python \
testing/analyze_raw_probe_noise.py \
  --data-dir /mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0 \
  --stream-id imec1.ap \
  --sweep-dir /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep
```

The defaults sample 100 one-second windows. Use `--n-windows 2
--window-duration-s 0.2 --output-dir /tmp/raw_noise_smoke` for a smoke test.
The first run writes `channel_time_metrics.npz`; later invocations reuse that
cache unless `--recalc` is supplied. A settings or source mismatch is rejected
rather than silently reusing the wrong cache.

## Simultaneous-probe control

The saved crop contains `imec1` channel IDs, so those IDs cannot select `imec0`.
Use the same physical depth bounds instead and skip the `imec1` KS4-pair join:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
/home/huklaban5/anaconda3/envs/spikeinterface/bin/python \
testing/analyze_raw_probe_noise.py \
  --data-dir /mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0 \
  --stream-id imec0.ap \
  --sweep-dir /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep \
  --selection-mode saved-depths \
  --skip-duplicate-link \
  --output-dir /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep/raw_noise_debug_imec0
```

The same commands can be applied to the 2026-03-02 raw directory and its
`shallow_sweep` folder.

## Reference-safety diagnostic

`analyze_reference_safety.py` reconstructs the current conditioning chain as a
lazy view and measures the spike-triggered decomposition `y = x - r` using an
existing KS4 result. The local 40--140 µm median is explicitly formed on all
384 AP channels before the 36-channel shallow crop is requested. It reports
reference overlap (`alpha`), a jittered-time null, amplitude and SNR ratios,
waveform correlation, raw/referenced footprint width, and ordinary versus
high-common-state values.

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/huklaban5/anaconda3/envs/spikeinterface/bin/python \
testing/analyze_reference_safety.py \
  --data-dir /mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0 \
  --sweep-dir /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep \
  --output-dir /tmp/reference_safety_smoke --max-windows 2
```

This is a survivor-conditioned diagnostic: a KS4-good label does not prove a
cluster is neural, and units eliminated by the current preprocessing cannot be
measured from that sorter output.

## Sync-aligned cross-probe common trace

`analyze_cross_probe_common.py` compares the two phase-corrected, AP-filtered,
full-probe median traces sample by sample. It maps sample coordinates between
the probes from their matched one-second `riseSent` sync edges, thereby
correcting both stream offset and clock drift before calculating correlation,
short-lag cross-correlation, coherence, and PSD.

```bash
MPLCONFIGDIR=/tmp/mplconfig \
/home/huklaban5/anaconda3/envs/spikeinterface/bin/python \
testing/analyze_cross_probe_common.py \
  --data-dir /mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0 \
  --window-metrics /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep/raw_noise_debug/channel_time_metrics.npz \
  --output-dir /tmp/cross_probe_common_smoke --max-windows 2
```

## Interpretation limits

- `ap_mad_uv_local_masked` is a robust AP-band scale, not pure electrode noise.
- `masked_fraction` must be considered with the masked estimate; heavy masking
  can otherwise make a problematic channel appear quiet.
- Transient fractions describe the sampled windows, not unsampled portions of
  the recording.
- Rows in `duplicate_pairs_with_raw_noise.csv` are not independent statistical
  replicates because pairs can share units, depths, and recording intervals.
- Global and local references are diagnostic views only; they are not saved or
  passed to sorting.
- Cross-probe correlation without sync-edge alignment is invalid for these
  streams because their calibrated AP clocks drift relative to one another.
