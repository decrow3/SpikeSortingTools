#!/usr/bin/env python3
"""Sample-level comparison of the phase-corrected AP common traces on two probes.

This standalone diagnostic distinguishes correlated *window amplitudes* from
instantaneous shared waveforms.  It reads matched raw windows, computes the
full-probe median after phase correction and a diagnostic 300--6000 Hz filter,
then measures zero-lag correlation, short-lag cross-correlation, coherence,
and common-trace spectra.  It does not alter pipeline data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, coherence, correlate, correlation_lags, sosfiltfilt, welch

from analyze_raw_probe_noise import robust_scale


def normalized_short_xcorr(a: np.ndarray, b: np.ndarray, fs: float, max_lag_ms: float):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return np.nan, np.nan
    values = correlate(a, b, mode="full", method="fft") / denominator
    lags = correlation_lags(len(a), len(b), mode="full")
    keep = np.abs(lags) <= int(round(max_lag_ms * 1e-3 * fs))
    selected = np.flatnonzero(keep)
    best = selected[np.argmax(np.abs(values[keep]))]
    return float(values[best]), float(1000 * lags[best] / fs)


def band_mean(freq: np.ndarray, values: np.ndarray, low: float, high: float) -> float:
    keep = (freq >= low) & (freq < high)
    return float(np.nanmean(values[keep])) if keep.any() else np.nan


def map_sample_frames(frames: np.ndarray, source_edges: np.ndarray,
                      target_edges: np.ndarray) -> np.ndarray:
    """Map sample positions between streams using simultaneous sync edges.

    Linear endpoint extrapolation covers the short intervals before the first
    and after the final one-second edge.
    """
    frames = np.asarray(frames, dtype=float)
    source_edges = np.asarray(source_edges, dtype=float)
    target_edges = np.asarray(target_edges, dtype=float)
    if len(source_edges) != len(target_edges) or len(source_edges) < 2:
        raise ValueError("Sync edge arrays must have the same length and contain >=2 edges")
    if np.any(np.diff(source_edges) <= 0) or np.any(np.diff(target_edges) <= 0):
        raise ValueError("Sync edges must be strictly increasing")
    mapped = np.interp(frames, source_edges, target_edges)
    before = frames < source_edges[0]
    after = frames > source_edges[-1]
    first_slope = ((target_edges[1] - target_edges[0]) /
                   (source_edges[1] - source_edges[0]))
    last_slope = ((target_edges[-1] - target_edges[-2]) /
                  (source_edges[-1] - source_edges[-2]))
    mapped[before] = target_edges[0] + first_slope * (frames[before] - source_edges[0])
    mapped[after] = target_edges[-1] + last_slope * (frames[after] - source_edges[-1])
    return mapped


def load_sync_edges(path: Path) -> np.ndarray:
    data = loadmat(path, squeeze_me=True)
    if "riseSent" not in data:
        raise KeyError(f"No riseSent array in {path}")
    return np.asarray(data["riseSent"], dtype=np.float64).ravel()


def find_sync_file(data_dir: Path, stream_id: str, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    matches = sorted(data_dir.glob(f"*.{stream_id}.sync.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *.{stream_id}.sync.mat in {data_dir}; found {matches}"
        )
    return matches[0]


def phase_corrected_common(recording, start_frame: int, n_frames: int, padding_s: float,
                           band: tuple[float, float], order: int):
    from spikeinterface.preprocessing import phase_shift

    fs = float(recording.get_sampling_frequency())
    shifts = recording.get_property("inter_sample_shift")
    shifted = phase_shift(recording) if shifts is not None and np.any(shifts) else recording
    start = int(start_frame)
    pad = int(round(padding_s * fs))
    n = int(n_frames)
    traces = shifted.get_traces(
        start_frame=start - pad, end_frame=start + n + pad, return_scaled=True
    ).astype(np.float32, copy=False)
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, traces, axis=0)
    scale = robust_scale(filtered, axis=0)
    valid = np.isfinite(scale) & (scale > 0)
    typical = np.median(scale[valid])
    valid &= scale < 5 * typical
    common = np.median(filtered[pad:pad + n, valid], axis=1)
    frames = start + np.arange(len(common), dtype=np.float64)
    return frames, common, fs, float(np.mean(valid))


def run(args):
    from spikeinterface.extractors import read_spikeglx

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec0 = read_spikeglx(data_dir, load_sync_channel=False, stream_id=args.stream0)
    rec1 = read_spikeglx(data_dir, load_sync_channel=False, stream_id=args.stream1)
    sync0_path = find_sync_file(data_dir, args.stream0, args.sync0)
    sync1_path = find_sync_file(data_dir, args.stream1, args.sync1)
    sync0 = load_sync_edges(sync0_path)
    sync1 = load_sync_edges(sync1_path)
    if len(sync0) != len(sync1):
        raise ValueError(f"Sync edge count differs: {len(sync0)} versus {len(sync1)}")
    fs0 = float(rec0.get_sampling_frequency())
    fs1 = float(rec1.get_sampling_frequency())
    source = np.load(Path(args.window_metrics))
    starts_s = source["window_start_s"][: args.max_windows]
    rows, coherences, psd0s, psd1s = [], [], [], []
    spectral_freq = None

    for index, start_s in enumerate(starts_s):
        print(f"Window {index + 1}/{len(starts_s)} at {start_s:.3f} s", flush=True)
        start1 = int(round(float(start_s) * fs1))
        n1 = int(round(args.window_duration_s * fs1))
        bounds0 = map_sample_frames(
            np.array([start1, start1 + n1 - 1], dtype=float), sync1, sync0
        )
        start0 = int(np.floor(bounds0[0]))
        stop0 = int(np.ceil(bounds0[1])) + 1
        frames0, c0, _, valid0 = phase_corrected_common(
            rec0, start0, stop0 - start0, args.padding_s,
            (args.low_hz, args.high_hz), args.filter_order,
        )
        frames1, c1, _, valid1 = phase_corrected_common(
            rec1, start1, n1, args.padding_s,
            (args.low_hz, args.high_hz), args.filter_order,
        )
        # Convert every probe-0 sample coordinate onto probe 1's clock using
        # simultaneous one-second sync pulses, then interpolate onto the exact
        # probe-1 samples. This handles both start offset and clock drift.
        frames0_on_1 = map_sample_frames(frames0, sync0, sync1)
        c0 = np.interp(frames1, frames0_on_1, c0)
        fs = fs1
        zero_corr = float(np.corrcoef(c0, c1)[0, 1])
        peak_corr, peak_lag_ms = normalized_short_xcorr(c0, c1, fs, args.max_lag_ms)
        nperseg = min(args.welch_nperseg, len(c0))
        freq, coh = coherence(c0, c1, fs=fs, nperseg=nperseg)
        _, p0 = welch(c0, fs=fs, nperseg=nperseg)
        _, p1 = welch(c1, fs=fs, nperseg=nperseg)
        spectral_freq = freq
        coherences.append(coh); psd0s.append(p0); psd1s.append(p1)
        rows.append(dict(
            window_index=index, start_s=float(start_s),
            zero_lag_correlation=zero_corr,
            peak_abs_correlation=peak_corr, peak_lag_ms=peak_lag_ms,
            coherence_300_1000=band_mean(freq, coh, 300, 1000),
            coherence_1000_3000=band_mean(freq, coh, 1000, 3000),
            coherence_3000_6000=band_mean(freq, coh, 3000, 6000),
            common_scale_uv_imec0=float(robust_scale(c0, axis=0)),
            common_scale_uv_imec1=float(robust_scale(c1, axis=0)),
            valid_channel_fraction_imec0=valid0,
            valid_channel_fraction_imec1=valid1,
            mapped_start_frame_imec0=float(bounds0[0]),
            start_frame_imec1=start1,
        ))

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "cross_probe_common_windows.csv", index=False)
    coherence_array = np.asarray(coherences)
    np.savez_compressed(
        output_dir / "cross_probe_common_spectra.npz",
        frequency_hz=spectral_freq,
        coherence=coherence_array,
        psd_imec0=np.asarray(psd0s), psd_imec1=np.asarray(psd1s),
        sync_edges_imec0=sync0, sync_edges_imec1=sync1,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(table.start_s, table.zero_lag_correlation, marker=".")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Pearson r",
                   title="Sample-level zero-lag correlation")
    axes[0, 1].hist(table.peak_lag_ms, bins=25)
    axes[0, 1].set(xlabel="Peak absolute-correlation lag (ms)", ylabel="Windows",
                   title="Short-lag alignment")
    axes[1, 0].plot(spectral_freq, np.nanmedian(coherence_array, axis=0))
    axes[1, 0].set(xlim=(args.low_hz, args.high_hz), ylim=(0, 1),
                   xlabel="Frequency (Hz)", ylabel="Coherence",
                   title="Median magnitude-squared coherence")
    axes[1, 1].scatter(table.common_scale_uv_imec0, table.zero_lag_correlation,
                       c=table.start_s, cmap="viridis", s=22)
    axes[1, 1].set(xlabel="imec0 common scale (µV)", ylabel="Zero-lag correlation",
                   title="Correlation versus common-mode scale")
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(output_dir / "fig_cross_probe_common.png", dpi=220)
    plt.close(fig)
    print(table.describe().to_string())
    return table


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--stream0", default="imec0.ap")
    p.add_argument("--stream1", default="imec1.ap")
    p.add_argument("--sync0", help="Probe-0 AP sync .mat (auto-detected by default)")
    p.add_argument("--sync1", help="Probe-1 AP sync .mat (auto-detected by default)")
    p.add_argument("--window-metrics", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-windows", type=int, default=100)
    p.add_argument("--window-duration-s", type=float, default=1.0)
    p.add_argument("--padding-s", type=float, default=.1)
    p.add_argument("--low-hz", type=float, default=300.0)
    p.add_argument("--high-hz", type=float, default=6000.0)
    p.add_argument("--filter-order", type=int, default=4)
    p.add_argument("--max-lag-ms", type=float, default=5.0)
    p.add_argument("--welch-nperseg", type=int, default=2048)
    return p


if __name__ == "__main__":
    run(make_parser().parse_args())
