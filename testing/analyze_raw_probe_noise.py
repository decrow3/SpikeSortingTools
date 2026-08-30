#!/usr/bin/env python3
"""Exploratory raw-recording QC for existing SpikeSortingTools debug datasets.

This module is deliberately isolated from :mod:`pipeline`.  It reads a raw
SpikeGLX AP stream lazily, reuses the channel IDs saved by a shallow parameter
sweep, samples deterministic windows, and writes diagnostic metrics/figures.
It never writes a recording or changes a sorter input.

Example
-------
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/mplconfig \
python testing/analyze_raw_probe_noise.py \
    --data-dir /mnt/NPX/Luke/20260316/Luke03162026_V2V1_RH_g0 \
    --stream-id imec1.ap \
    --sweep-dir /mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt, welch


MAD_NORMALIZER = 0.6744897501960817


def robust_scale(values: np.ndarray, axis=0) -> np.ndarray:
    """Gaussian-equivalent MAD, preserving NaNs and the requested axes."""
    values = np.asarray(values, dtype=float)
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(values - center), axis=axis) / MAD_NORMALIZER


def deterministic_window_starts(
    n_frames: int,
    sampling_frequency: float,
    window_duration_s: float,
    n_windows: int,
    edge_margin_s: float,
) -> np.ndarray:
    """Return reproducible, approximately evenly spaced window start frames."""
    window_frames = int(round(window_duration_s * sampling_frequency))
    margin_frames = int(round(edge_margin_s * sampling_frequency))
    first = margin_frames
    last = n_frames - margin_frames - window_frames
    if window_frames < 2:
        raise ValueError("window_duration_s is too short")
    if last < first:
        raise ValueError("recording is shorter than the requested window and margins")
    if n_windows < 1:
        raise ValueError("n_windows must be at least 1")
    starts = np.rint(np.linspace(first, last, n_windows)).astype(np.int64)
    return np.unique(starts)


def make_local_neighbor_indices(
    positions_um: np.ndarray,
    inner_radius_um: float,
    outer_radius_um: float,
) -> list[np.ndarray]:
    """Build geometry-based local-reference neighborhoods for every channel."""
    positions_um = np.asarray(positions_um, dtype=float)
    distance = np.linalg.norm(positions_um[:, None, :] - positions_um[None, :, :], axis=2)
    return [
        np.flatnonzero((distance[i] >= inner_radius_um) & (distance[i] <= outer_radius_um))
        for i in range(len(positions_um))
    ]


def local_median_reference(
    traces: np.ndarray,
    neighbors: list[np.ndarray],
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract a per-channel local median from time-by-channel traces."""
    traces = np.asarray(traces, dtype=float)
    if eligible is None:
        eligible = np.ones(traces.shape[1], dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    output = traces.copy()
    for channel, neighborhood in enumerate(neighbors):
        usable = neighborhood[eligible[neighborhood]]
        if usable.size:
            output[:, channel] -= np.median(traces[:, usable], axis=1)
    return output


def dilated_excursion_mask(
    traces: np.ndarray,
    sigma: np.ndarray,
    threshold_sigma: float,
    dilation_samples: int,
    global_participation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel and probe-wide masks for large AP-band excursions."""
    centered = traces - np.median(traces, axis=0, keepdims=True)
    safe_sigma = np.maximum(np.asarray(sigma, dtype=float), np.finfo(float).eps)
    excursion = np.abs(centered) > threshold_sigma * safe_sigma[None, :]
    participation = np.mean(excursion, axis=1)
    global_mask = participation >= global_participation_fraction
    size = max(1, 2 * int(dilation_samples) + 1)
    if size > 1:
        excursion = maximum_filter1d(excursion.astype(np.uint8), size=size, axis=0) > 0
        global_mask = maximum_filter1d(global_mask.astype(np.uint8), size=size, axis=0) > 0
    return excursion | global_mask[:, None], global_mask


def masked_robust_scale(traces: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAD after masking, plus the removed fraction for interpretability."""
    masked = np.asarray(traces, dtype=float).copy()
    masked[np.asarray(mask, dtype=bool)] = np.nan
    return robust_scale(masked, axis=0), np.mean(mask, axis=0)


def integrate_psd_band(freq: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    keep = (freq >= low) & (freq < high)
    if np.count_nonzero(keep) < 2:
        return np.full(psd.shape[1], np.nan)
    # np.trapezoid was introduced after the NumPy version in the lab's
    # spikeinterface environment; np.trapz is equivalent here.
    return np.trapz(psd[keep], freq[keep], axis=0)


def line_noise_ratio(
    freq: np.ndarray,
    psd: np.ndarray,
    line_frequency: float = 60.0,
    half_width_hz: float = 1.0,
    neighborhood_hz: float = 5.0,
) -> np.ndarray:
    """Harmonic line power divided by nearby non-line power per Hz."""
    line = np.zeros_like(freq, dtype=bool)
    nearby = np.zeros_like(freq, dtype=bool)
    harmonic = line_frequency
    while harmonic < freq[-1]:
        line |= np.abs(freq - harmonic) <= half_width_hz
        nearby |= np.abs(freq - harmonic) <= neighborhood_hz
        harmonic += line_frequency
    background = nearby & ~line
    if not line.any() or not background.any():
        return np.full(psd.shape[1], np.nan)
    line_density = np.mean(psd[line], axis=0)
    background_density = np.mean(psd[background], axis=0)
    return line_density / np.maximum(background_density, np.finfo(float).eps)


def contiguous_intervals(mask: np.ndarray, start_s: float, fs: float) -> list[tuple[float, float]]:
    """Convert a Boolean sample mask into half-open intervals in seconds."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(start_s + a / fs, start_s + b / fs) for a, b in zip(starts, stops)]


def parse_spikeglx_meta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.lstrip("~")] = value
    return result


def find_stream_files(data_dir: Path, stream_id: str) -> tuple[Path | None, Path | None]:
    probe_token, band = stream_id.split(".", 1)
    candidates = sorted(data_dir.glob(f"**/*.{probe_token}.{band}.bin"))
    if not candidates:
        return None, None
    binary = candidates[0]
    meta = binary.with_suffix(".meta")
    return binary, meta if meta.exists() else None


def build_metadata_report(recording, data_dir: Path, stream_id: str) -> dict:
    binary, meta_path = find_stream_files(data_dir, stream_id)
    fs = float(recording.get_sampling_frequency())
    report = {
        "data_dir": str(data_dir),
        "stream_id": stream_id,
        "sampling_frequency_hz": fs,
        "n_channels_loaded": int(recording.get_num_channels()),
        "n_frames_loaded": int(recording.get_num_frames()),
        "duration_s_loaded": float(recording.get_num_frames() / fs),
        "dtype": str(recording.get_dtype()),
        "binary_path": str(binary) if binary else None,
        "meta_path": str(meta_path) if meta_path else None,
    }
    try:
        report["channel_gains_uv"] = np.unique(recording.get_channel_gains()).astype(float).tolist()
        report["channel_offsets_uv"] = np.unique(recording.get_channel_offsets()).astype(float).tolist()
    except Exception as exc:  # metadata report should not prevent the analysis
        report["scaling_error"] = str(exc)
    if binary and meta_path:
        meta = parse_spikeglx_meta(meta_path)
        n_saved = int(meta.get("nSavedChans", recording.get_num_channels()))
        size = binary.stat().st_size
        bytes_per_timepoint = 2 * n_saved
        report.update(
            binary_size_bytes=size,
            n_saved_channels_meta=n_saved,
            whole_timepoints=bool(size % bytes_per_timepoint == 0),
            trailing_bytes=int(size % bytes_per_timepoint),
            duration_s_from_file=float(size // bytes_per_timepoint / fs),
            first_sample=meta.get("firstSample"),
            probe_type=meta.get("imDatPrb_type"),
            probe_part_number=meta.get("imDatPrb_pn"),
            saved_channel_subset=meta.get("snsSaveChanSubset"),
            ap_gain=meta.get("imChan0apGain"),
            file_create_time=meta.get("fileCreateTime"),
            file_time_secs_meta=meta.get("fileTimeSecs"),
            file_size_bytes_meta=meta.get("fileSizeBytes"),
        )
        report["duration_difference_s"] = (
            report["duration_s_from_file"] - report["duration_s_loaded"]
        )
        report["warning_or_error_fields"] = {
            key: value for key, value in meta.items()
            if "error" in key.lower() or "warn" in key.lower()
        }
    return report


def _gross_reference_eligibility(ap_traces: np.ndarray, raw_traces: np.ndarray) -> np.ndarray:
    scale = robust_scale(ap_traces, axis=0)
    finite_positive = np.isfinite(scale) & (scale > 0)
    typical = np.median(scale[finite_positive]) if finite_positive.any() else np.nan
    zero_diff_fraction = np.mean(np.diff(raw_traces, axis=0) == 0, axis=0)
    eligible = finite_positive & (zero_diff_fraction < 0.99)
    if np.isfinite(typical) and typical > 0:
        eligible &= scale < 5 * typical
    return eligible


def compute_window_metrics(
    raw_uv: np.ndarray,
    shifted_uv: np.ndarray,
    selected_indices: np.ndarray,
    positions_um: np.ndarray,
    fs: float,
    config: dict,
) -> tuple[dict[str, np.ndarray | float], np.ndarray]:
    """Compute one sampled window. Inputs include padding; outputs exclude it."""
    sos = butter(
        config["filter_order"], config["ap_band_hz"], btype="bandpass", fs=fs, output="sos"
    )
    raw_ap_all = sosfiltfilt(sos, raw_uv, axis=0)
    shifted_ap_all = sosfiltfilt(sos, shifted_uv, axis=0)
    eligible = _gross_reference_eligibility(shifted_ap_all, shifted_uv)
    usable = np.flatnonzero(eligible)
    if usable.size:
        global_ap_all = shifted_ap_all - np.median(shifted_ap_all[:, usable], axis=1, keepdims=True)
    else:
        global_ap_all = shifted_ap_all.copy()
    neighbors = make_local_neighbor_indices(
        positions_um, config["local_inner_um"], config["local_outer_um"]
    )
    local_ap_all = local_median_reference(shifted_ap_all, neighbors, eligible)

    pad = int(config["padding_frames"])
    core = slice(pad, -pad if pad else None)
    raw_ap = raw_ap_all[core][:, selected_indices]
    shifted_ap = shifted_ap_all[core][:, selected_indices]
    global_ap = global_ap_all[core][:, selected_indices]
    local_ap = local_ap_all[core][:, selected_indices]
    raw_core = raw_uv[core][:, selected_indices]
    shifted_core = shifted_uv[core][:, selected_indices]

    local_sigma = robust_scale(local_ap, axis=0)
    mask, crop_transient_mask = dilated_excursion_mask(
        local_ap,
        local_sigma,
        config["mask_threshold_sigma"],
        int(round(config["mask_margin_ms"] * 1e-3 * fs)),
        config["global_participation_fraction"],
    )
    probe_ap = shifted_ap_all[core]
    _, probe_transient_mask = dilated_excursion_mask(
        probe_ap,
        robust_scale(probe_ap, axis=0),
        config["mask_threshold_sigma"],
        int(round(config["mask_margin_ms"] * 1e-3 * fs)),
        config["global_participation_fraction"],
    )
    # A widespread full-probe event should be excluded even if it does not
    # cross threshold on enough channels inside the shallow crop itself.
    mask |= probe_transient_mask[:, None]
    local_masked_sigma, masked_fraction = masked_robust_scale(local_ap, mask)

    nperseg = min(config["welch_nperseg"], len(local_ap))
    freq, local_psd = welch(local_ap, fs=fs, nperseg=nperseg, axis=0)
    _, shifted_psd = welch(shifted_core, fs=fs, nperseg=nperseg, axis=0)
    diff = np.diff(shifted_core, axis=0)

    metrics: dict[str, np.ndarray | float] = {
        "ap_mad_uv_raw": robust_scale(raw_ap, axis=0),
        "ap_mad_uv_shifted": robust_scale(shifted_ap, axis=0),
        "ap_mad_uv_global": robust_scale(global_ap, axis=0),
        "ap_mad_uv_local": local_sigma,
        "ap_mad_uv_local_masked": local_masked_sigma,
        "masked_fraction": masked_fraction,
        "upper_3_6khz_power_uv2_local": integrate_psd_band(freq, local_psd, 3000.0, 6000.0),
        "highfreq_power_uv2_shifted": integrate_psd_band(freq, shifted_psd, 0.8 * fs / 2, fs / 2),
        "line_noise_ratio_local": line_noise_ratio(freq, local_psd, config["line_frequency_hz"]),
        "large_derivative_fraction": np.mean(np.abs(diff) >= config["large_derivative_uv"], axis=0),
        "zero_derivative_fraction": np.mean(diff == 0, axis=0),
        "absolute_artifact_fraction": np.mean(np.abs(raw_core) >= config["absolute_artifact_uv"], axis=0),
        "reference_benefit_global": 1.0 - np.square(robust_scale(global_ap, axis=0))
        / np.maximum(np.square(robust_scale(shifted_ap, axis=0)), np.finfo(float).eps),
        "reference_benefit_local": 1.0 - np.square(local_sigma)
        / np.maximum(np.square(robust_scale(shifted_ap, axis=0)), np.finfo(float).eps),
        "common_mode_ap_mad_uv": float(robust_scale(np.median(shifted_ap_all[core][:, usable], axis=1), axis=0))
        if usable.size else np.nan,
        "crop_transient_fraction": float(np.mean(crop_transient_mask)),
        "probe_transient_fraction": float(np.mean(probe_transient_mask)),
        "reference_eligible_fraction": float(np.mean(eligible)),
    }
    return metrics, probe_transient_mask


def _config_for_cache(args, fs: float) -> dict:
    return {
        "n_windows": args.n_windows,
        "window_duration_s": args.window_duration_s,
        "padding_s": args.padding_s,
        "padding_frames": int(round(args.padding_s * fs)),
        "ap_band_hz": [args.ap_low_hz, args.ap_high_hz],
        "filter_order": args.filter_order,
        "local_inner_um": args.local_inner_um,
        "local_outer_um": args.local_outer_um,
        "mask_threshold_sigma": args.mask_threshold_sigma,
        "mask_margin_ms": args.mask_margin_ms,
        "global_participation_fraction": args.global_participation_fraction,
        "welch_nperseg": args.welch_nperseg,
        "line_frequency_hz": args.line_frequency_hz,
        "large_derivative_uv": args.large_derivative_uv,
        "absolute_artifact_uv": args.absolute_artifact_uv,
    }


def analyze_recording(args) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Run or load the sampled-window analysis."""
    # Imports stay local so estimator unit tests do not require SpikeInterface.
    from spikeinterface.extractors import read_spikeglx
    from spikeinterface.preprocessing import phase_shift

    data_dir = Path(args.data_dir)
    sweep_dir = Path(args.sweep_dir)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "raw_noise_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    recording = read_spikeglx(
        folder_path=data_dir, load_sync_channel=False, stream_id=args.stream_id
    )
    fs = float(recording.get_sampling_frequency())
    config = _config_for_cache(args, fs)

    metadata = build_metadata_report(recording, data_dir, args.stream_id)
    (output_dir / "metadata_check.json").write_text(json.dumps(metadata, indent=2) + "\n")

    subset_path = sweep_dir / "channel_subset_info.npz"
    if not subset_path.exists():
        raise FileNotFoundError(f"Missing existing shallow-crop metadata: {subset_path}")
    subset = np.load(subset_path)
    all_ids = recording.get_channel_ids().astype(str)
    positions = np.asarray(recording.get_probe().contact_positions, dtype=float)
    if args.selection_mode == "saved-ids":
        selected_ids = subset["channel_ids"].astype(str)
        id_to_index = {channel_id: index for index, channel_id in enumerate(all_ids)}
        missing = [channel_id for channel_id in selected_ids if channel_id not in id_to_index]
        if missing:
            raise ValueError(
                f"Crop IDs are absent from raw stream: {missing[:5]}. "
                "Use --selection-mode saved-depths for a simultaneous probe."
            )
        selected_indices = np.asarray([id_to_index[channel_id] for channel_id in selected_ids], dtype=int)
    elif args.selection_mode == "saved-depths":
        crop_lo = float(subset["crop_lo"])
        crop_hi = float(subset["crop_hi"])
        selected_indices = np.flatnonzero(
            (positions[:, 1] >= crop_lo) & (positions[:, 1] <= crop_hi)
        )
        if not selected_indices.size:
            raise ValueError(f"No raw channels fall within saved depth bounds {crop_lo}–{crop_hi} µm")
        selected_ids = all_ids[selected_indices]
    else:
        selected_indices = np.arange(len(all_ids), dtype=int)
        selected_ids = all_ids.copy()
    selected_depths = positions[selected_indices, 1]
    config.update(
        data_dir=str(data_dir.resolve()),
        stream_id=args.stream_id,
        sweep_dir=str(sweep_dir.resolve()),
        n_frames=int(recording.get_num_frames()),
        sampling_frequency_hz=fs,
        selection_mode=args.selection_mode,
        selected_channel_ids=selected_ids.tolist(),
    )

    cache_path = output_dir / "channel_time_metrics.npz"
    config_path = output_dir / "analysis_config.json"
    if cache_path.exists() and config_path.exists() and not args.recalc:
        cached_config = json.loads(config_path.read_text())
        if cached_config != config:
            raise RuntimeError("Cached metrics use different settings; rerun with --recalc")
        loaded = np.load(cache_path)
        arrays = {key: loaded[key] for key in loaded.files}
        intervals_path = output_dir / "sampled_transient_intervals.csv"
        intervals = pd.read_csv(intervals_path) if intervals_path.exists() else pd.DataFrame()
        return arrays, intervals

    shifts = recording.get_property("inter_sample_shift")
    shifted_recording = phase_shift(recording) if shifts is not None and np.any(shifts) else recording
    starts = deterministic_window_starts(
        recording.get_num_frames(), fs, args.window_duration_s,
        args.n_windows, args.padding_s,
    )
    window_frames = int(round(args.window_duration_s * fs))
    pad_frames = config["padding_frames"]
    collected: dict[str, list] = {}
    interval_rows: list[dict] = []

    for window_index, start in enumerate(starts):
        print(f"Window {window_index + 1}/{len(starts)} at {start / fs:.3f} s", flush=True)
        read_start = int(start - pad_frames)
        read_stop = int(start + window_frames + pad_frames)
        raw_uv = recording.get_traces(
            start_frame=read_start, end_frame=read_stop, return_scaled=True
        ).astype(np.float32, copy=False)
        shifted_uv = shifted_recording.get_traces(
            start_frame=read_start, end_frame=read_stop, return_scaled=True
        ).astype(np.float32, copy=False)
        metrics, transient_mask = compute_window_metrics(
            raw_uv, shifted_uv, selected_indices, positions, fs, config
        )
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value)
        start_s = start / fs
        for transient_start, transient_stop in contiguous_intervals(transient_mask, start_s, fs):
            interval_rows.append({
                "window_index": window_index,
                "start_s": transient_start,
                "stop_s": transient_stop,
                "duration_ms": 1000 * (transient_stop - transient_start),
            })

    arrays = {key: np.asarray(value) for key, value in collected.items()}
    arrays.update(
        channel_ids=selected_ids,
        channel_depths_um=selected_depths,
        window_start_frames=starts,
        window_start_s=starts / fs,
        window_stop_s=(starts + window_frames) / fs,
        sampling_frequency_hz=np.asarray(fs),
        recording_duration_s=np.asarray(recording.get_num_frames() / fs),
    )
    np.savez_compressed(cache_path, **arrays)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    intervals = pd.DataFrame(interval_rows, columns=["window_index", "start_s", "stop_s", "duration_ms"])
    intervals.to_csv(output_dir / "sampled_transient_intervals.csv", index=False)
    return arrays, intervals


def write_channel_summary(arrays: dict[str, np.ndarray], output_dir: Path) -> pd.DataFrame:
    per_channel_keys = [
        key for key, value in arrays.items()
        if isinstance(value, np.ndarray) and value.ndim == 2
        and value.shape[1] == len(arrays["channel_ids"])
    ]
    summary = pd.DataFrame({
        "channel_id": arrays["channel_ids"].astype(str),
        "depth_um": arrays["channel_depths_um"],
    })
    for key in per_channel_keys:
        summary[f"{key}_median"] = np.nanmedian(arrays[key], axis=0)
        summary[f"{key}_p95"] = np.nanpercentile(arrays[key], 95, axis=0)
    summary.sort_values(["depth_um", "channel_id"]).to_csv(
        output_dir / "channel_summary.csv", index=False
    )
    return summary


def _style_axes(axes: Iterable[plt.Axes]) -> None:
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def plot_overview(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    order = np.argsort(arrays["channel_depths_um"])
    depths = arrays["channel_depths_um"][order]
    times = arrays["window_start_s"]
    noise = arrays["ap_mad_uv_local_masked"][:, order]
    median_depth = np.nanmedian(noise, axis=0)
    p95_depth = np.nanpercentile(noise, 95, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    if len(times) == 1:
        time_half_width = max(0.5, float(arrays["window_stop_s"][0] - times[0]) / 2)
    else:
        time_half_width = float(np.median(np.diff(times))) / 2
    duration_s = float(arrays.get("recording_duration_s", arrays["window_stop_s"][-1]))
    extent = [
        max(0.0, times[0] - time_half_width),
        min(duration_s, times[-1] + time_half_width),
        depths[0], depths[-1],
    ]
    image = axes[0, 0].imshow(
        noise.T, aspect="auto", origin="lower", extent=extent, cmap="magma",
        vmin=np.nanpercentile(noise, 5), vmax=np.nanpercentile(noise, 95),
    )
    axes[0, 0].set(xlabel="Sampled-window time (s)", ylabel="Depth from tip (µm)",
                   title="Spike-masked local-reference AP scale")
    fig.colorbar(image, ax=axes[0, 0], label="µV")

    axes[0, 1].plot(median_depth, depths, label="median")
    axes[0, 1].plot(p95_depth, depths, label="95th percentile")
    axes[0, 1].set(xlabel="AP scale (µV)", ylabel="Depth from tip (µm)",
                   title="Noise versus depth")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(times, arrays["common_mode_ap_mad_uv"], marker=".")
    axes[1, 0].set(xlabel="Sampled-window time (s)", ylabel="AP scale (µV)",
                   title="Probe-wide median trace")
    axes[1, 1].plot(times, arrays["probe_transient_fraction"], label="full-probe transient")
    axes[1, 1].plot(times, arrays["crop_transient_fraction"], label="crop transient")
    axes[1, 1].plot(times, np.nanmedian(arrays["masked_fraction"], axis=1),
                    label="median channel mask")
    axes[1, 1].set(xlabel="Sampled-window time (s)", ylabel="Sample fraction",
                   title="Sampled artifact burden")
    axes[1, 1].legend(frameon=False)
    _style_axes(axes.ravel())
    fig.savefig(output_dir / "fig1_time_depth_noise.png", dpi=200)
    plt.close(fig)


def plot_reference_comparison(arrays: dict[str, np.ndarray], output_dir: Path) -> None:
    labels = ["raw", "delay-corrected", "global median", "local median", "local masked"]
    keys = [
        "ap_mad_uv_raw", "ap_mad_uv_shifted", "ap_mad_uv_global",
        "ap_mad_uv_local", "ap_mad_uv_local_masked",
    ]
    data = [arrays[key].ravel() for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].boxplot(data, tick_labels=labels, showfliers=False)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(ylabel="Robust AP-band scale (µV)", title="Diagnostic signal views")
    axes[1].hist(arrays["reference_benefit_global"].ravel(), bins=40, alpha=0.6,
                 label="global")
    axes[1].hist(arrays["reference_benefit_local"].ravel(), bins=40, alpha=0.6,
                 label="local")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="Reference benefit 1 − σref²/σshifted²", ylabel="Count",
                title="Reference benefit")
    axes[1].legend(frameon=False)
    _style_axes(axes)
    fig.savefig(output_dir / "fig2_reference_comparison.png", dpi=200)
    plt.close(fig)


def relate_duplicate_screens(
    arrays: dict[str, np.ndarray], sweep_dir: Path, output_dir: Path, radius_um: float = 100.0
) -> pd.DataFrame:
    rows = []
    depths = arrays["channel_depths_um"]
    channel_noise = np.nanmedian(arrays["ap_mad_uv_local_masked"], axis=0)
    for csv_path in sorted(sweep_dir.glob("within_run_screen_*.csv")):
        run = csv_path.stem.removeprefix("within_run_screen_")
        try:
            screen = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            continue
        required = {"depth_a", "depth_b", "near_zero_frac", "zero_peak_ratio"}
        if not required.issubset(screen.columns):
            continue
        for _, pair in screen.iterrows():
            midpoint = 0.5 * (float(pair["depth_a"]) + float(pair["depth_b"]))
            local = np.abs(depths - midpoint) <= radius_um
            if not local.any():
                local[np.argmin(np.abs(depths - midpoint))] = True
            row = pair.to_dict()
            row.update(
                run=run,
                pair_midpoint_depth_um=midpoint,
                local_raw_noise_uv=float(np.nanmedian(channel_noise[local])),
                flagged_duplicate=bool(
                    pair["near_zero_frac"] >= 0.05 and pair["zero_peak_ratio"] >= 1.25
                ),
            )
            rows.append(row)
    enriched = pd.DataFrame(rows)
    enriched.to_csv(output_dir / "duplicate_pairs_with_raw_noise.csv", index=False)
    if enriched.empty:
        return enriched

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for run, group in enriched.groupby("run"):
        ax.scatter(group["local_raw_noise_uv"], group["near_zero_frac"], s=15,
                   alpha=0.55, label=run)
    ax.axhline(0.05, color="black", ls="--", lw=0.8)
    ax.set(xlabel="Local spike-masked raw AP scale (µV)",
           ylabel="Near-zero-lag fraction", title="Existing duplicate screen versus local raw QC")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style_axes([ax])
    fig.savefig(output_dir / "fig3_noise_vs_duplicate_depth.png", dpi=200)
    plt.close(fig)
    return enriched


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stream-id", default="imec1.ap")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--selection-mode", choices=("saved-ids", "saved-depths", "full"),
        default="saved-ids",
        help=("Use exact shallow-sweep channel IDs (default), reuse its physical depth "
              "bounds on another probe, or analyze the full probe."),
    )
    parser.add_argument("--skip-duplicate-link", action="store_true")
    parser.add_argument("--n-windows", type=int, default=100)
    parser.add_argument("--window-duration-s", type=float, default=1.0)
    parser.add_argument("--padding-s", type=float, default=0.1)
    parser.add_argument("--ap-low-hz", type=float, default=300.0)
    parser.add_argument("--ap-high-hz", type=float, default=6000.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--local-inner-um", type=float, default=40.0)
    parser.add_argument("--local-outer-um", type=float, default=140.0)
    parser.add_argument("--mask-threshold-sigma", type=float, default=5.0)
    parser.add_argument("--mask-margin-ms", type=float, default=1.0)
    parser.add_argument("--global-participation-fraction", type=float, default=0.25)
    parser.add_argument("--welch-nperseg", type=int, default=2048)
    parser.add_argument("--line-frequency-hz", type=float, default=60.0)
    parser.add_argument("--large-derivative-uv", type=float, default=100.0)
    parser.add_argument("--absolute-artifact-uv", type=float, default=500.0)
    parser.add_argument("--recalc", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.sweep_dir) / "raw_noise_debug"
    arrays, _ = analyze_recording(args)
    write_channel_summary(arrays, output_dir)
    plot_overview(arrays, output_dir)
    plot_reference_comparison(arrays, output_dir)
    if not args.skip_duplicate_link:
        relate_duplicate_screens(arrays, Path(args.sweep_dir), output_dir)
    print(f"Raw-noise debugging outputs written to {output_dir}")


if __name__ == "__main__":
    main()
