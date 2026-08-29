"""Matched raw-AP voltage audit for Luke and the known-good Yates session.

Both recordings are converted to microvolts and receive the same in-memory
300--6000 Hz filter.  Results are reported both before referencing and after a
geometry-matched local median reference.  Events are local extrema, thresholded
by either per-channel robust sigma or absolute microvolts, and greedily
deduplicated within 0.5 ms and 100 um.

The audit is diagnostic rather than a spike sorter.  It asks whether Luke's
lower strong-negative-event density is already present before the historical
session-specific preprocessing graphs, and whether it is broad or concentrated
by time, depth, polarity, or channel.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "testing/outputs/luke_motion_candidate_results/raw_voltage_audit"
LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
YATES_ROOT = Path("/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16")
SIGMA_THRESHOLDS = (5.0, 6.0, 8.0)
UV_THRESHOLDS = (50.0, 75.0, 100.0)


@dataclass(frozen=True)
class RecordingSpec:
    name: str
    binary: Path
    n_channels_file: int
    neural_channels: int
    n_frames: int
    sampling_rate_hz: float
    gain_uv_per_count: float
    locations_um: np.ndarray
    shanks: np.ndarray
    window_kind: str
    window_start_s: float | None = None
    window_duration_s: float | None = None


def robust_sigma(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values)
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis)
    return np.maximum(mad / 0.6744897501960817, np.finfo(np.float32).eps)


def depth_exposure_mm(locations: np.ndarray, shanks: np.ndarray) -> float:
    exposure = 0.0
    for shank in np.unique(shanks):
        y = locations[shanks == shank, 1]
        if y.size > 1:
            exposure += float(np.ptp(y)) / 1000.0
    return exposure


def spatial_neighbors(
    locations: np.ndarray, shanks: np.ndarray, radius_um: float
) -> list[np.ndarray]:
    delta = locations[:, None, :] - locations[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    same_shank = shanks[:, None] == shanks[None, :]
    return [np.flatnonzero(same_shank[i] & (distance[i] <= radius_um)) for i in range(len(locations))]


def local_median_reference(values: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for channel, neighborhood in enumerate(neighbors):
        result[:, channel] = values[:, channel] - np.median(values[:, neighborhood], axis=1)
    return result


def shank_median_reference(values: np.ndarray, shanks: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    for shank in np.unique(shanks):
        mask = shanks == shank
        result[:, mask] = values[:, mask] - np.median(values[:, mask], axis=1, keepdims=True)
    return result


def extrema_candidates(values: np.ndarray, negative: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = values[1:-1]
    if negative:
        mask = (center < values[:-2]) & (center <= values[2:])
        amplitude = -center[mask]
    else:
        mask = (center > values[:-2]) & (center >= values[2:])
        amplitude = center[mask]
    times, channels = np.nonzero(mask)
    return times.astype(np.int64) + 1, channels.astype(np.int64), amplitude.astype(np.float32)


def collapse_candidates(
    times: np.ndarray,
    channels: np.ndarray,
    scores: np.ndarray,
    neighbors: list[np.ndarray],
    n_samples: int,
    temporal_radius: int,
) -> np.ndarray:
    if not len(times):
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros((len(neighbors), n_samples), dtype=bool)
    keep: list[int] = []
    for index in order:
        time = int(times[index])
        channel = int(channels[index])
        if suppressed[channel, time]:
            continue
        keep.append(int(index))
        lo = max(0, time - temporal_radius)
        hi = min(n_samples, time + temporal_radius + 1)
        suppressed[neighbors[channel], lo:hi] = True
    kept = np.asarray(keep, dtype=np.int64)
    return kept[np.argsort(times[kept])]


def normalized_depth(locations: np.ndarray, shanks: np.ndarray) -> np.ndarray:
    result = np.zeros(len(locations), dtype=np.float32)
    for shank in np.unique(shanks):
        mask = shanks == shank
        y = locations[mask, 1]
        span = float(np.ptp(y))
        result[mask] = (y - y.min()) / span if span else 0.5
    return result


def select_batch_starts(spec: RecordingSpec, n_batches: int, batch_s: float, padding_s: float) -> np.ndarray:
    total_s = spec.n_frames / spec.sampling_rate_hz
    if spec.window_start_s is None:
        lo = padding_s
        hi = total_s - batch_s - padding_s
    else:
        lo = spec.window_start_s
        hi = spec.window_start_s + float(spec.window_duration_s) - batch_s
    if hi < lo:
        raise ValueError(f"Window is too short for {spec.name}: {lo=} {hi=}")
    return np.linspace(lo, hi, n_batches)


def load_specs() -> list[RecordingSpec]:
    yates_meta = json.loads((YATES_ROOT / "ephys_metadata.json").read_text())
    yates_locations = np.asarray(yates_meta["probe_geometry_um"], dtype=np.float32)
    yates_shanks = np.repeat(np.arange(2), 32)
    yates_binary = YATES_ROOT / "recording.dat"
    expected_yates_bytes = int(yates_meta["n_samples"]) * int(yates_meta["n_channels"]) * 2
    if yates_binary.stat().st_size != expected_yates_bytes:
        raise RuntimeError("Yates recording.dat size does not match ephys_metadata.json")

    specs: list[RecordingSpec] = []
    for probe in ("imec0", "imec1"):
        raw_dir = LUKE_ROOT / "Luke0804_V2V1_g0" / f"Luke0730_V2V1_g0_{probe}"
        binary = raw_dir / f"Luke0730_V2V1_g0_t0.{probe}.ap.bin"
        processed = LUKE_ROOT / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}" / "preprocessed_recording"
        locations = np.load(processed / "properties/location.npy").astype(np.float32)
        n_channels_file = 385
        n_frames = binary.stat().st_size // (n_channels_file * 2)
        common = dict(
            binary=binary,
            n_channels_file=n_channels_file,
            neural_channels=384,
            n_frames=n_frames,
            sampling_rate_hz=29999.759166666667,
            gain_uv_per_count=2.34375,
            locations_um=locations,
            shanks=np.zeros(384, dtype=np.int16),
        )
        specs.extend(
            [
                RecordingSpec(name=f"Luke {probe} pathological", window_kind="pathological", window_start_s=8160.0, window_duration_s=120.0, **common),
                RecordingSpec(name=f"Luke {probe} shared", window_kind="shared", window_start_s=7095.0, window_duration_s=240.0, **common),
                RecordingSpec(name=f"Luke {probe} session", window_kind="session-wide", **common),
            ]
        )
    specs.append(
        RecordingSpec(
            name="Yates raw session",
            binary=yates_binary,
            n_channels_file=64,
            neural_channels=64,
            n_frames=int(yates_meta["n_samples"]),
            sampling_rate_hz=float(yates_meta["sample_rate"]),
            gain_uv_per_count=float(yates_meta["uV_per_bit"]),
            locations_um=yates_locations,
            shanks=yates_shanks,
            window_kind="session-wide",
        )
    )
    return specs


def read_batch(spec: RecordingSpec, start_s: float, batch_s: float, padding_s: float) -> np.ndarray:
    start = int(round((start_s - padding_s) * spec.sampling_rate_hz))
    stop = int(round((start_s + batch_s + padding_s) * spec.sampling_rate_hz))
    start = max(0, start)
    stop = min(spec.n_frames, stop)
    raw = np.memmap(
        spec.binary,
        dtype="int16",
        mode="r",
        shape=(spec.n_frames, spec.n_channels_file),
    )
    return np.asarray(raw[start:stop, : spec.neural_channels], dtype=np.float32) * spec.gain_uv_per_count


def footprint_metrics(
    values: np.ndarray,
    sigma: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    polarity: str,
    locations: np.ndarray,
    shanks: np.ndarray,
    neighbors_100: list[np.ndarray],
    max_events: int,
) -> list[dict]:
    if not len(times):
        return []
    chosen = np.linspace(0, len(times) - 1, min(max_events, len(times))).astype(int)
    rows: list[dict] = []
    radius = 3
    for index in chosen:
        time = int(times[index])
        peak_channel = int(channels[index])
        lo, hi = max(0, time - radius), min(len(values), time + radius + 1)
        wave = values[lo:hi]
        if polarity == "negative":
            amplitude = np.maximum(0.0, -np.min(wave, axis=0))
        else:
            amplitude = np.maximum(0.0, np.max(wave, axis=0))
        same_shank = shanks == shanks[peak_channel]
        within_500 = same_shank & (np.abs(locations[:, 1] - locations[peak_channel, 1]) <= 500.0)
        energy = amplitude * amplitude
        total = float(np.sum(energy[within_500]))
        local = float(np.sum(energy[neighbors_100[peak_channel]]))
        weights = energy[within_500]
        depths = locations[within_500, 1]
        if weights.sum() > 0:
            center = float(np.sum(depths * weights) / weights.sum())
            depth_sd = float(np.sqrt(np.sum((depths - center) ** 2 * weights) / weights.sum()))
        else:
            depth_sd = float("nan")
        rows.append(
            {
                "local_energy_fraction_100um": local / total if total else float("nan"),
                "footprint_depth_sd_um": depth_sd,
                "active_channels_4sigma": int(np.sum(amplitude >= 4.0 * sigma)),
                "peak_amplitude_uv": float(amplitude[peak_channel]),
            }
        )
    return rows


def analyze_stage(
    spec: RecordingSpec,
    values: np.ndarray,
    stage: str,
    batch_index: int,
    start_s: float,
    batch_s: float,
    neighbors_100: list[np.ndarray],
    footprint_limit: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    sigma = robust_sigma(values, axis=0).astype(np.float32)
    temporal_radius = int(round(0.0005 * spec.sampling_rate_hz))
    exposure = depth_exposure_mm(spec.locations_um, spec.shanks)
    norm_depth = normalized_depth(spec.locations_um, spec.shanks)
    event_rows: list[dict] = []
    depth_rows: list[dict] = []
    footprint_rows: list[dict] = []
    for polarity, negative in (("negative", True), ("positive", False)):
        times, channels, amplitudes = extrema_candidates(values, negative)
        thresholds = [("sigma", threshold, amplitudes >= threshold * sigma[channels]) for threshold in SIGMA_THRESHOLDS]
        thresholds += [("absolute_uv", threshold, amplitudes >= threshold) for threshold in UV_THRESHOLDS]
        selected_six: tuple[np.ndarray, np.ndarray] | None = None
        for threshold_kind, threshold, select in thresholds:
            candidate_indices = np.flatnonzero(select)
            keep_local = collapse_candidates(
                times[candidate_indices],
                channels[candidate_indices],
                amplitudes[candidate_indices],
                neighbors_100,
                len(values),
                temporal_radius,
            )
            kept = candidate_indices[keep_local]
            kept_times = times[kept]
            kept_channels = channels[kept]
            if threshold_kind == "sigma" and threshold == 6.0:
                selected_six = kept_times, kept_channels
            base = {
                "dataset": spec.name,
                "window_kind": spec.window_kind,
                "stage": stage,
                "batch_index": batch_index,
                "batch_start_s": start_s,
                "polarity": polarity,
                "threshold_kind": threshold_kind,
                "threshold": threshold,
                "event_count": len(kept),
                "event_rate_per_s": len(kept) / batch_s,
                "event_rate_per_mm_s": len(kept) / batch_s / exposure,
                "median_event_amplitude_uv": float(np.median(amplitudes[kept])) if len(kept) else float("nan"),
                "p90_event_amplitude_uv": float(np.percentile(amplitudes[kept], 90)) if len(kept) else float("nan"),
                "median_channel_sigma_uv": float(np.median(sigma)),
                "p90_channel_sigma_uv": float(np.percentile(sigma, 90)),
                "depth_exposure_mm": exposure,
            }
            event_rows.append(base)
            if threshold_kind == "sigma" and threshold == 6.0:
                for depth_bin in range(8):
                    lo, hi = depth_bin / 8, (depth_bin + 1) / 8
                    count = int(np.sum((norm_depth[kept_channels] >= lo) & (norm_depth[kept_channels] < hi + (depth_bin == 7))))
                    depth_rows.append(
                        {
                            **{key: base[key] for key in ("dataset", "window_kind", "stage", "batch_index", "batch_start_s", "polarity")},
                            "depth_bin": depth_bin,
                            "normalized_depth_center": (lo + hi) / 2,
                            "event_count": count,
                            "event_rate_per_mm_s": count / batch_s / (exposure / 8),
                        }
                    )
        if selected_six is not None:
            selected_times, selected_channels = selected_six
            for row in footprint_metrics(
                values,
                sigma,
                selected_times,
                selected_channels,
                polarity,
                spec.locations_um,
                spec.shanks,
                neighbors_100,
                footprint_limit,
            ):
                footprint_rows.append(
                    {
                        "dataset": spec.name,
                        "window_kind": spec.window_kind,
                        "stage": stage,
                        "batch_index": batch_index,
                        "batch_start_s": start_s,
                        "polarity": polarity,
                        **row,
                    }
                )
    return event_rows, depth_rows, footprint_rows


def run_audit(n_batches: int, batch_s: float, padding_s: float, footprint_limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sos = butter(3, (300.0, 6000.0), btype="bandpass", fs=30000.0, output="sos")
    all_events: list[dict] = []
    all_depths: list[dict] = []
    all_footprints: list[dict] = []
    channel_rows: list[dict] = []
    manifest_specs: list[dict] = []
    for spec in load_specs():
        print(f"Analyzing {spec.name}", flush=True)
        neighbors_100 = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
        reference_neighbors = spatial_neighbors(spec.locations_um, spec.shanks, 100.0)
        starts = select_batch_starts(spec, n_batches, batch_s, padding_s)
        manifest_specs.append(
            {
                **{key: value for key, value in asdict(spec).items() if key not in {"locations_um", "shanks"}},
                "binary": str(spec.binary),
                "locations_um": spec.locations_um.tolist(),
                "shanks": spec.shanks.tolist(),
                "batch_starts_s": starts.tolist(),
                "depth_exposure_mm": depth_exposure_mm(spec.locations_um, spec.shanks),
            }
        )
        trim = int(round(padding_s * spec.sampling_rate_hz))
        for batch_index, start_s in enumerate(starts):
            raw_uv = read_batch(spec, float(start_s), batch_s, padding_s)
            filtered = sosfiltfilt(sos, raw_uv, axis=0).astype(np.float32)
            if trim:
                filtered = filtered[trim:-trim]
                raw_trimmed = raw_uv[trim:-trim]
            else:
                raw_trimmed = raw_uv
            stages = {
                "common_bandpass": filtered,
                "common_bandpass_shank_median": shank_median_reference(filtered, spec.shanks),
                "common_bandpass_local_reference": local_median_reference(filtered, reference_neighbors),
            }
            raw_centered = raw_trimmed - np.median(raw_trimmed, axis=0, keepdims=True)
            for channel in range(spec.neural_channels):
                channel_rows.append(
                    {
                        "dataset": spec.name,
                        "window_kind": spec.window_kind,
                        "batch_index": batch_index,
                        "batch_start_s": start_s,
                        "channel": channel,
                        "shank": int(spec.shanks[channel]),
                        "x_um": float(spec.locations_um[channel, 0]),
                        "y_um": float(spec.locations_um[channel, 1]),
                        "normalized_depth": float(normalized_depth(spec.locations_um, spec.shanks)[channel]),
                        "raw_sigma_uv": float(robust_sigma(raw_centered[:, channel], axis=0)),
                        "bandpass_sigma_uv": float(robust_sigma(filtered[:, channel], axis=0)),
                        "fraction_abs_raw_over_500uv": float(np.mean(np.abs(raw_centered[:, channel]) > 500.0)),
                    }
                )
            for stage, values in stages.items():
                event_rows, depth_rows, footprint_rows = analyze_stage(
                    spec,
                    values,
                    stage,
                    batch_index,
                    float(start_s),
                    batch_s,
                    neighbors_100,
                    footprint_limit,
                )
                all_events.extend(event_rows)
                all_depths.extend(depth_rows)
                all_footprints.extend(footprint_rows)
            print(f"  batch {batch_index + 1}/{n_batches}", flush=True)

    events = pd.DataFrame(all_events)
    depths = pd.DataFrame(all_depths)
    footprints = pd.DataFrame(all_footprints)
    channels = pd.DataFrame(channel_rows)
    events.to_csv(OUT / "raw_event_batch_metrics.csv", index=False)
    depths.to_csv(OUT / "raw_event_depth_metrics.csv", index=False)
    footprints.to_csv(OUT / "raw_event_footprint_sample.csv", index=False)
    channels.to_csv(OUT / "raw_channel_batch_metrics.csv", index=False)
    summarize(events, depths, footprints, channels)
    render(events, depths, footprints, channels)
    manifest = {
        "metric_definition": "Same 300--6000 Hz filter; optional 100 um local median reference; physical deduplication within 0.5 ms and 100 um.",
        "n_batches": n_batches,
        "batch_duration_s": batch_s,
        "padding_s": padding_s,
        "sigma_thresholds": SIGMA_THRESHOLDS,
        "absolute_uv_thresholds": UV_THRESHOLDS,
        "footprint_sample_limit_per_batch_polarity": footprint_limit,
        "recordings": manifest_specs,
    }
    (OUT / "raw_voltage_audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def summarize(events: pd.DataFrame, depths: pd.DataFrame, footprints: pd.DataFrame, channels: pd.DataFrame) -> None:
    event_summary = (
        events.groupby(["dataset", "window_kind", "stage", "polarity", "threshold_kind", "threshold"], as_index=False)
        .agg(
            median_event_rate_per_mm_s=("event_rate_per_mm_s", "median"),
            p10_event_rate_per_mm_s=("event_rate_per_mm_s", lambda x: np.percentile(x, 10)),
            p90_event_rate_per_mm_s=("event_rate_per_mm_s", lambda x: np.percentile(x, 90)),
            median_event_amplitude_uv=("median_event_amplitude_uv", "median"),
            median_channel_sigma_uv=("median_channel_sigma_uv", "median"),
            n_batches=("batch_index", "nunique"),
        )
    )
    event_summary.to_csv(OUT / "raw_event_summary.csv", index=False)
    depth_summary = (
        depths.groupby(["dataset", "window_kind", "stage", "polarity", "depth_bin", "normalized_depth_center"], as_index=False)
        .agg(
            median_event_rate_per_mm_s=("event_rate_per_mm_s", "median"),
            p10_event_rate_per_mm_s=("event_rate_per_mm_s", lambda x: np.percentile(x, 10)),
            p90_event_rate_per_mm_s=("event_rate_per_mm_s", lambda x: np.percentile(x, 90)),
        )
    )
    depth_summary.to_csv(OUT / "raw_depth_summary.csv", index=False)
    footprint_summary = (
        footprints.groupby(["dataset", "window_kind", "stage", "polarity"], as_index=False)
        .agg(
            sampled_events=("peak_amplitude_uv", "size"),
            median_peak_amplitude_uv=("peak_amplitude_uv", "median"),
            median_local_energy_fraction=("local_energy_fraction_100um", "median"),
            median_footprint_depth_sd_um=("footprint_depth_sd_um", "median"),
            median_active_channels_4sigma=("active_channels_4sigma", "median"),
            compact_fraction=("local_energy_fraction_100um", lambda x: np.mean(np.asarray(x) >= 0.5)),
        )
    )
    footprint_summary.to_csv(OUT / "raw_footprint_summary.csv", index=False)
    channel_summary = (
        channels.groupby(["dataset", "window_kind", "channel", "shank", "x_um", "y_um", "normalized_depth"], as_index=False)
        .agg(
            median_raw_sigma_uv=("raw_sigma_uv", "median"),
            median_bandpass_sigma_uv=("bandpass_sigma_uv", "median"),
            median_fraction_abs_raw_over_500uv=("fraction_abs_raw_over_500uv", "median"),
        )
    )
    channel_summary.to_csv(OUT / "raw_channel_summary.csv", index=False)


def render(events: pd.DataFrame, depths: pd.DataFrame, footprints: pd.DataFrame, channels: pd.DataFrame) -> None:
    datasets = list(events["dataset"].drop_duplicates())
    palette = dict(zip(datasets, plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))))
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for stage, linestyle in (
        ("common_bandpass", ":"),
        ("common_bandpass_shank_median", "--"),
        ("common_bandpass_local_reference", "-"),
    ):
        for dataset in datasets:
            subset = events[(events.dataset == dataset) & (events.stage == stage) & (events.polarity == "negative") & (events.threshold_kind == "sigma")]
            summary = subset.groupby("threshold").event_rate_per_mm_s.median()
            stage_label = {"common_bandpass": "no ref", "common_bandpass_shank_median": "shank median", "common_bandpass_local_reference": "100 um median"}[stage]
            axes[0, 0].plot(summary.index, summary.values, marker="o", linestyle=linestyle, color=palette[dataset], label=f"{dataset} | {stage_label}")
    axes[0, 0].set(title="Negative event density by robust threshold", xlabel="Threshold (sigma)", ylabel="Events/mm/s")
    axes[0, 0].legend(fontsize=7, ncol=2)

    stage = "common_bandpass_shank_median"
    for polarity, marker in (("negative", "o"), ("positive", "s")):
        for dataset in datasets:
            subset = events[(events.dataset == dataset) & (events.stage == stage) & (events.polarity == polarity) & (events.threshold_kind == "absolute_uv")]
            summary = subset.groupby("threshold").event_rate_per_mm_s.median()
            axes[0, 1].plot(summary.index, summary.values, marker=marker, color=palette[dataset], alpha=0.85, label=f"{dataset} | {polarity}")
    axes[0, 1].set(title="Absolute-amplitude event density", xlabel="Threshold (uV)", ylabel="Events/mm/s")

    noise = channels.groupby("dataset").bandpass_sigma_uv.median().reindex(datasets)
    axes[0, 2].bar(np.arange(len(datasets)), noise.values, color=[palette[d] for d in datasets], edgecolor="0.25")
    axes[0, 2].set_xticks(np.arange(len(datasets)), [d.replace(" ", "\n", 1) for d in datasets], fontsize=7)
    axes[0, 2].set(title="Raw bandpass noise", ylabel="Median channel sigma (uV)")

    for dataset in datasets:
        subset = depths[(depths.dataset == dataset) & (depths.stage == stage) & (depths.polarity == "negative")]
        summary = subset.groupby("normalized_depth_center").event_rate_per_mm_s.median()
        axes[1, 0].plot(summary.index, summary.values, marker="o", color=palette[dataset], label=dataset)
    axes[1, 0].set(title="Strong negative events across normalized depth", xlabel="Normalized shank depth", ylabel="6-sigma events/mm/s")
    axes[1, 0].legend(fontsize=7)

    compact = footprints[(footprints.stage == stage) & (footprints.polarity == "negative")].groupby("dataset").local_energy_fraction_100um.median().reindex(datasets)
    axes[1, 1].bar(np.arange(len(datasets)), compact.values, color=[palette[d] for d in datasets], edgecolor="0.25")
    axes[1, 1].set_xticks(np.arange(len(datasets)), [d.replace(" ", "\n", 1) for d in datasets], fontsize=7)
    axes[1, 1].set(title="Spatial compactness of sampled 6-sigma events", ylabel="Median energy within 100 um")

    saturation = channels.groupby("dataset").fraction_abs_raw_over_500uv.median().reindex(datasets)
    axes[1, 2].bar(np.arange(len(datasets)), saturation.values, color=[palette[d] for d in datasets], edgecolor="0.25")
    axes[1, 2].set_xticks(np.arange(len(datasets)), [d.replace(" ", "\n", 1) for d in datasets], fontsize=7)
    axes[1, 2].set(title="Large raw excursions", ylabel="Median channel fraction |raw| >500 uV")
    for ax in axes.flat:
        ax.grid(axis="y", color="0.9")
    fig.suptitle("Luke--Yates matched raw-voltage audit", fontsize=16)
    fig.savefig(OUT / "luke_yates_raw_voltage_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-batches", type=int, default=30)
    parser.add_argument("--batch-duration-s", type=float, default=2.0)
    parser.add_argument("--padding-s", type=float, default=0.1)
    parser.add_argument("--footprint-limit", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_audit(args.n_batches, args.batch_duration_s, args.padding_s, args.footprint_limit)
    print(f"Wrote matched raw-voltage audit to {OUT}")


if __name__ == "__main__":
    main()
