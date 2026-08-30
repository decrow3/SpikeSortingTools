#!/usr/bin/env python3
"""Spike-triggered test of the current full-probe local median reference.

This is debugging code only.  It reconstructs the existing conditioning view
(phase shift, saturation blanking, bad-channel interpolation, 300--6000 Hz
filter, then 40--140 um local median reference), samples spikes from an
existing shallow KS4 run, and decomposes y = x - r at every sampled spike.
No recording or sorter input is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_raw_probe_noise import robust_scale


def baseline_waveform(waveform: np.ndarray, n_baseline: int) -> np.ndarray:
    return waveform - np.mean(waveform[:n_baseline], axis=0, keepdims=True)


def waveform_alpha(reference_waveform: np.ndarray, raw_waveform: np.ndarray) -> float:
    denominator = float(np.dot(raw_waveform, raw_waveform))
    if denominator <= np.finfo(float).eps:
        return np.nan
    return float(np.dot(reference_waveform, raw_waveform) / denominator)


def waveform_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def footprint_width_um(waveforms: np.ndarray, depths_um: np.ndarray) -> float:
    energy = np.sum(np.square(waveforms), axis=0)
    total = float(np.sum(energy))
    if total <= np.finfo(float).eps:
        return np.nan
    center = float(np.sum(energy * depths_um) / total)
    return float(np.sqrt(np.sum(energy * np.square(depths_um - center)) / total))


WAVEFORM_ARRAY_KEYS = {"waveform_x", "waveform_r", "waveform_y", "waveform_null"}


def reconstruct_conditioning(recording, channel_metrics_path: Path, uV_thresh: float):
    """Rebuild the exact pre-motion sorting/reference view used by the pipeline."""
    from spikeinterface.preprocessing import (
        blank_staturation,
        common_reference,
        filter as si_filter,
        interpolate_bad_channels,
        phase_shift,
    )

    shifts = recording.get_property("inter_sample_shift")
    shifted = phase_shift(recording) if shifts is not None and np.any(shifts) else recording
    gains = np.unique(recording.get_channel_gains())
    if len(gains) != 1:
        raise ValueError(f"Expected one AP gain, found {gains}")
    saturated = blank_staturation(shifted, uV_thresh / float(gains[0]), direction="both")
    similarity, high_frequency_noise = np.load(channel_metrics_path)
    bad = (similarity < -0.5) | (high_frequency_noise > 0.3)
    bad_ids = recording.get_channel_ids()[bad]
    interpolated = interpolate_bad_channels(saturated, bad_ids)
    filtered = si_filter(
        interpolated, band=[300.0, 6000.0], btype="bandpass",
        filter_order=12, ftype="butter", direction="forward-backward",
    )
    referenced = common_reference(
        filtered, reference="local", operator="median", local_radius=(40, 140)
    )
    return filtered, referenced, bad_ids


def load_sorting_inputs(sorter_dir: Path):
    spike_times = np.load(sorter_dir / "spike_times.npy").astype(np.int64).ravel()
    spike_clusters = np.load(sorter_dir / "spike_clusters.npy").astype(np.int64).ravel()
    templates = np.load(sorter_dir / "templates.npy")
    labels = pd.read_csv(sorter_dir / "cluster_KSLabel.tsv", sep="\t")
    label_map = dict(zip(labels.iloc[:, 0].astype(int), labels.iloc[:, 1].astype(str)))
    return spike_times, spike_clusters, templates, label_map


def _state_metrics(mean_x, mean_r, mean_y, mean_null, peak_channel, noise_x, noise_y, n_baseline):
    wx_all = baseline_waveform(mean_x, n_baseline)
    wr_all = baseline_waveform(mean_r, n_baseline)
    wy_all = baseline_waveform(mean_y, n_baseline)
    wn_all = baseline_waveform(mean_null, n_baseline)
    wx, wr, wy, wn = (a[:, peak_channel] for a in (wx_all, wr_all, wy_all, wn_all))
    amp_x = float(np.ptp(wx))
    amp_y = float(np.ptp(wy))
    return dict(
        alpha=waveform_alpha(wr, wx),
        alpha_jitter_null=waveform_alpha(wn, wx),
        amplitude_x_uv=amp_x,
        amplitude_y_uv=amp_y,
        amplitude_ratio=amp_y / amp_x if amp_x > 0 else np.nan,
        snr_x=amp_x / noise_x if noise_x > 0 else np.nan,
        snr_y=amp_y / noise_y if noise_y > 0 else np.nan,
        snr_ratio=(amp_y / noise_y) / (amp_x / noise_x)
        if amp_x > 0 and noise_x > 0 and noise_y > 0 else np.nan,
        waveform_correlation=waveform_correlation(wx, wy),
        waveform_x=wx_all,
        waveform_r=wr_all,
        waveform_y=wy_all,
        waveform_null=wn_all,
    )


def run(args):
    from spikeinterface.extractors import read_spikeglx

    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    sorter_dir = Path(args.sorter_dir) if args.sorter_dir else (
        sweep_dir / "run_default" / "kilosort4" / "sorter_output"
    )
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "reference_safety"
    output_dir.mkdir(parents=True, exist_ok=True)

    recording = read_spikeglx(data_dir, load_sync_channel=False, stream_id=args.stream_id)
    fs = float(recording.get_sampling_frequency())
    subset = np.load(sweep_dir / "channel_subset_info.npz")
    selected_ids = subset["channel_ids"].astype(str)
    all_ids = recording.get_channel_ids().astype(str)
    id_to_index = {cid: i for i, cid in enumerate(all_ids)}
    selected_indices = np.asarray([id_to_index[cid] for cid in selected_ids])
    selected_depths = recording.get_probe().contact_positions[selected_indices, 1]

    conditioning_path = Path(args.conditioning_metrics) if args.conditioning_metrics else (
        sweep_dir.parent / "conditioning" / "channel_metrics.npy"
    )
    x_recording, y_recording, bad_ids = reconstruct_conditioning(
        recording, conditioning_path, args.saturation_uv
    )
    # Critical safety assertion: common_reference wraps the full 384-channel
    # extractor.  The crop is requested only during get_traces below.
    if y_recording.get_num_channels() != recording.get_num_channels():
        raise RuntimeError("Reference was unexpectedly formed after channel cropping")

    st, clu, templates, label_map = load_sorting_inputs(sorter_dir)
    units = np.arange(templates.shape[0], dtype=int)
    template_rms = np.sqrt(np.mean(np.square(templates), axis=1))
    template_peak = np.argmax(template_rms, axis=1)

    noise_cache = np.load(sweep_dir / "raw_noise_debug" / "channel_time_metrics.npz")
    starts = noise_cache["window_start_frames"].astype(np.int64)[: args.max_windows]
    common_scale = noise_cache["common_mode_ap_mad_uv"][: len(starts)]
    window_frames = int(round(args.window_duration_s * fs))
    pre = int(round(args.pre_ms * 1e-3 * fs))
    post = int(round(args.post_ms * 1e-3 * fs))
    offsets = np.arange(-pre, post + 1)
    n_samples = len(offsets)
    n_channels = len(selected_ids)
    n_units = len(units)
    states = ("all", "ordinary", "high_common")
    sums = {
        state: {view: np.zeros((n_units, n_samples, n_channels), dtype=np.float64)
                for view in ("x", "r", "y", "null")}
        for state in states
    }
    counts = {state: np.zeros(n_units, dtype=np.int64) for state in states}
    noise_x_windows, noise_y_windows = [], []
    rng = np.random.default_rng(args.seed)
    jitter_min = int(round(args.jitter_min_ms * 1e-3 * fs))
    jitter_max = int(round(args.jitter_max_ms * 1e-3 * fs))

    for wi, start in enumerate(starts):
        print(f"Window {wi + 1}/{len(starts)} at {start / fs:.3f} s", flush=True)
        read_start = int(start - pre)
        read_stop = int(start + window_frames + post)
        x = x_recording.get_traces(
            start_frame=read_start, end_frame=read_stop,
            channel_ids=selected_ids.tolist(), return_scaled=True,
        ).astype(np.float32, copy=False)
        y = y_recording.get_traces(
            start_frame=read_start, end_frame=read_stop,
            channel_ids=selected_ids.tolist(), return_scaled=True,
        ).astype(np.float32, copy=False)
        r = x - y
        # The final sampled window can extend beyond the recording when a
        # longer debugging duration is requested. Use only the complete core
        # for which pre/post waveform margins are actually available.
        available_window_frames = min(window_frames, len(x) - pre - post)
        if available_window_frames <= 2 * jitter_max:
            continue
        core = slice(pre, pre + available_window_frames)
        noise_x_windows.append(robust_scale(x[core], axis=0))
        noise_y_windows.append(robust_scale(y[core], axis=0))
        state = "high_common" if common_scale[wi] >= args.high_common_uv else "ordinary"

        lo = start + jitter_max + pre
        hi = start + available_window_frames - jitter_max - post
        left, right = np.searchsorted(st, [lo, hi])
        event_times = st[left:right]
        event_units = clu[left:right]
        for uid in np.unique(event_units):
            remaining = args.max_spikes_per_unit - counts["all"][uid]
            if remaining <= 0:
                continue
            times = event_times[event_units == uid][:remaining]
            if not len(times):
                continue
            centers = times - read_start
            indices = centers[:, None] + offsets[None, :]
            jitter = rng.integers(jitter_min, jitter_max + 1, size=len(times))
            jitter *= rng.choice(np.array([-1, 1]), size=len(times))
            null_indices = centers[:, None] + jitter[:, None] + offsets[None, :]
            views = {"x": x[indices], "r": r[indices], "y": y[indices], "null": r[null_indices]}
            for state_name in ("all", state):
                for view, snippets in views.items():
                    sums[state_name][view][uid] += np.sum(snippets, axis=0)
                counts[state_name][uid] += len(times)

    noise_x = np.nanmedian(np.asarray(noise_x_windows), axis=0)
    noise_y = np.nanmedian(np.asarray(noise_y_windows), axis=0)
    rows = []
    waveform_store = {}
    n_baseline = max(2, int(round(0.25 * pre)))
    for uid in units:
        if counts["all"][uid] < args.min_spikes_per_unit:
            continue
        peak = int(template_peak[uid])
        row = dict(
            unit_id=int(uid), ks_label=label_map.get(int(uid), "unknown"),
            n_spikes=int(counts["all"][uid]), peak_channel_index=peak,
            peak_channel_id=selected_ids[peak], peak_depth_um=float(selected_depths[peak]),
        )
        for state in states:
            n = counts[state][uid]
            if n < args.min_state_spikes:
                continue
            means = {view: sums[state][view][uid] / n for view in sums[state]}
            metrics = _state_metrics(
                means["x"], means["r"], means["y"], means["null"], peak,
                float(noise_x[peak]), float(noise_y[peak]), n_baseline,
            )
            prefix = "" if state == "all" else f"{state}_"
            for key, value in metrics.items():
                if key not in WAVEFORM_ARRAY_KEYS:
                    row[prefix + key] = value
            row[prefix + "n_spikes"] = int(n)
            if state == "all":
                row["raw_footprint_width_um"] = footprint_width_um(
                    metrics["waveform_x"], selected_depths)
                row["referenced_footprint_width_um"] = footprint_width_um(
                    metrics["waveform_y"], selected_depths)
                for view in ("x", "r", "y", "null"):
                    waveform_store[f"unit_{uid}_{view}"] = metrics[f"waveform_{view}"]
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("unit_id")
    results.to_csv(output_dir / "reference_safety_units.csv", index=False)
    np.savez_compressed(output_dir / "reference_safety_waveforms.npz", **waveform_store)
    good = results.loc[results.ks_label == "good"]

    def attenuation_summary(frame):
        return {
            "n": int(len(frame)),
            "fraction_losing_gt_10pct": float(np.mean(frame.amplitude_ratio < 0.9)),
            "fraction_losing_gt_20pct": float(np.mean(frame.amplitude_ratio < 0.8)),
            "fraction_losing_gt_30pct": float(np.mean(frame.amplitude_ratio < 0.7)),
            "median_alpha": float(np.nanmedian(frame.alpha)),
            "median_alpha_jitter_null": float(np.nanmedian(frame.alpha_jitter_null)),
            "median_amplitude_ratio": float(np.nanmedian(frame.amplitude_ratio)),
            "median_snr_ratio": float(np.nanmedian(frame.snr_ratio)),
        }

    summary = {
        "full_probe_channel_count": int(recording.get_num_channels()),
        "selected_channel_count": int(n_channels),
        "reference_before_crop_verified": True,
        "bad_channel_ids": bad_ids.astype(str).tolist(),
        "all_units": attenuation_summary(results),
        "good_units": attenuation_summary(good),
    }
    (output_dir / "reference_safety_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = results.ks_label.map({"good": "#276749", "mua": "#2B6CB0"}).fillna("gray")
    axes[0, 0].scatter(results.alpha, results.amplitude_ratio, c=colors, alpha=.75)
    axes[0, 0].axhline(.9, color="black", ls="--", lw=.8)
    axes[0, 0].set(xlabel="Spike-triggered reference α", ylabel="Referenced/raw amplitude",
                   title="Reference overlap and attenuation")
    axes[0, 1].scatter(results.raw_footprint_width_um, results.amplitude_ratio,
                       c=colors, alpha=.75)
    axes[0, 1].axhline(.9, color="black", ls="--", lw=.8)
    axes[0, 1].set(xlabel="Raw footprint width (µm)", ylabel="Amplitude ratio",
                   title="Spatial footprint dependence")
    axes[1, 0].hist(results.alpha, bins=20, alpha=.65, label="spike-triggered")
    axes[1, 0].hist(results.alpha_jitter_null, bins=20, alpha=.65, label="jitter null")
    axes[1, 0].set(xlabel="α", ylabel="Units", title="Reference waveform versus null")
    axes[1, 0].legend(frameon=False)
    if {"ordinary_amplitude_ratio", "high_common_amplitude_ratio"}.issubset(results.columns):
        valid = results[["ordinary_amplitude_ratio", "high_common_amplitude_ratio"]].dropna()
        axes[1, 1].scatter(valid.ordinary_amplitude_ratio, valid.high_common_amplitude_ratio,
                           c=colors.loc[valid.index], alpha=.75)
        axes[1, 1].plot([.5, 1.5], [.5, 1.5], "k--", lw=.8)
    axes[1, 1].set(xlabel="Ordinary-period amplitude ratio",
                   ylabel="High-common-period amplitude ratio", title="State dependence")
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(output_dir / "fig_reference_safety.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return results, summary


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--stream-id", default="imec1.ap")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--sorter-dir")
    p.add_argument("--conditioning-metrics")
    p.add_argument("--output-dir")
    p.add_argument("--max-windows", type=int, default=100)
    p.add_argument("--window-duration-s", type=float, default=1.0)
    p.add_argument("--max-spikes-per-unit", type=int, default=500)
    p.add_argument("--min-spikes-per-unit", type=int, default=30)
    p.add_argument("--min-state-spikes", type=int, default=10)
    p.add_argument("--pre-ms", type=float, default=1.0)
    p.add_argument("--post-ms", type=float, default=2.0)
    p.add_argument("--jitter-min-ms", type=float, default=20.0)
    p.add_argument("--jitter-max-ms", type=float, default=100.0)
    p.add_argument("--high-common-uv", type=float, default=30.0)
    p.add_argument("--saturation-uv", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=20260806)
    return p


if __name__ == "__main__":
    run(make_parser().parse_args())
