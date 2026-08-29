"""Dumb raw-data recovery check for high-amplitude Luke units.

The diagnostic deliberately avoids re-running a sorter.  It builds a small
multi-channel matched filter from raw AP snippets at the sorted spike times of
selected units, applies that filter to other raw-data windows, and asks whether
each obvious template-like event has a nearby spike in the curated sorting.

Candidates are classified as:

* ``target``: the selected unit has a spike within the tolerance;
* ``other``: some sorted spike is nearby, but not one from the selected unit;
* ``missed``: the curated sorting has no spike nearby at all.

This is a screening test, not a new spike sorter.  Interpret a miss fraction
only when the held-out recovery of known target spikes is high.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")


@dataclass(frozen=True)
class ProbeConfig:
    probe: str
    sample_rate_hz: float
    raw_path: Path
    n_saved_channels: int
    sorting_path: Path


PROBES = {
    "imec0": ProbeConfig(
        probe="imec0",
        sample_rate_hz=29999.835983263598,
        raw_path=LUKE_ROOT
        / "Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec0/"
        "Luke0730_V2V1_g0_t0.imec0.ap.bin",
        n_saved_channels=385,
        sorting_path=LUKE_ROOT
        / "patched_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_sorter_output",
    ),
    "imec1": ProbeConfig(
        probe="imec1",
        sample_rate_hz=29999.759166666667,
        raw_path=LUKE_ROOT
        / "Luke0804_V2V1_g0/Luke0730_V2V1_g0_imec1/"
        "Luke0730_V2V1_g0_t0.imec1.ap.bin",
        n_saved_channels=385,
        sorting_path=LUKE_ROOT
        / "patched_pipeline_results_Luke0804_V2V1_g0_imec1/cur/cur_sorter_output",
    ),
}


DEFAULT_WINDOWS = {
    "imec0": [
        ("template", 7215.0, 120.0),
        ("quiet", 3951.0, 120.0),
        ("pre_shared", 7095.0, 120.0),
    ],
    "imec1": [
        ("template", 7215.0, 120.0),
        ("pre_shared", 7095.0, 120.0),
        ("registration_outlier", 8160.0, 120.0),
    ],
}


def robust_sigma(values: np.ndarray, axis=None) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    return np.median(np.abs(values - median), axis=axis) / 0.6744897501960817


def nearest_event_mask(candidates: np.ndarray, events: np.ndarray, tolerance: int) -> np.ndarray:
    """Return whether each candidate is within tolerance of a sorted event."""
    if events.size == 0:
        return np.zeros(candidates.size, dtype=bool)
    indices = np.searchsorted(events, candidates)
    left = np.clip(indices - 1, 0, events.size - 1)
    right = np.clip(indices, 0, events.size - 1)
    distance = np.minimum(np.abs(candidates - events[left]), np.abs(candidates - events[right]))
    return distance <= tolerance


def load_sorting(config: ProbeConfig) -> dict[str, np.ndarray]:
    path = config.sorting_path
    return {
        "times": np.load(path / "spike_times.npy", mmap_mode="r").reshape(-1),
        "clusters": np.load(path / "spike_clusters.npy", mmap_mode="r").reshape(-1),
        "positions": np.load(path / "spike_positions.npy", mmap_mode="r"),
        "channel_positions": np.load(path / "channel_positions.npy"),
    }


def unit_channels(
    sorting: dict[str, np.ndarray], unit_ids: list[int], radius_um: float
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, float]]:
    positions = sorting["positions"]
    clusters = sorting["clusters"]
    channel_positions = sorting["channel_positions"]
    per_unit: dict[int, np.ndarray] = {}
    depths: dict[int, float] = {}
    for unit_id in unit_ids:
        mask = clusters == unit_id
        if not np.any(mask):
            raise ValueError(f"Unit {unit_id} has no spikes")
        depth = float(np.median(positions[mask, 1]))
        depths[unit_id] = depth
        channels = np.flatnonzero(np.abs(channel_positions[:, 1] - depth) <= radius_um)
        if channels.size == 0:
            channels = np.array([np.argmin(np.abs(channel_positions[:, 1] - depth))])
        per_unit[unit_id] = channels
    union = np.unique(np.concatenate(list(per_unit.values())))
    return union, per_unit, depths


def load_conditioned_window(
    config: ProbeConfig,
    union_channels: np.ndarray,
    start_s: float,
    duration_s: float,
    chunk_s: float = 2.0,
    pad_s: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Load a raw window with only a global median and 300 Hz high-pass."""
    fs = config.sample_rate_hz
    n_total = config.raw_path.stat().st_size // (2 * config.n_saved_channels)
    raw = np.memmap(
        config.raw_path,
        mode="r",
        dtype="<i2",
        shape=(n_total, config.n_saved_channels),
    )
    start = int(round(start_s * fs))
    stop = min(n_total, start + int(round(duration_s * fs)))
    chunk = int(round(chunk_s * fs))
    pad = int(round(pad_s * fs))
    highpass = butter(3, 300.0, btype="highpass", fs=fs, output="sos")
    output = np.empty((stop - start, union_channels.size), dtype=np.float32)

    for core_start in range(start, stop, chunk):
        core_stop = min(stop, core_start + chunk)
        read_start = max(0, core_start - pad)
        read_stop = min(n_total, core_stop + pad)
        block = np.asarray(raw[read_start:read_stop, :384], dtype=np.float32)
        block -= np.median(block, axis=1, keepdims=True)
        local = sosfiltfilt(highpass, block[:, union_channels], axis=0).astype(np.float32)
        source_start = core_start - read_start
        source_stop = source_start + (core_stop - core_start)
        dest_start = core_start - start
        output[dest_start : dest_start + core_stop - core_start] = local[
            source_start:source_stop
        ]
    return output, start


def extract_waveforms(
    traces: np.ndarray, centers: np.ndarray, channel_indices: np.ndarray, half_width: int
) -> np.ndarray:
    valid = centers[(centers >= half_width) & (centers < traces.shape[0] - half_width)]
    if valid.size == 0:
        return np.empty((0, 2 * half_width + 1, channel_indices.size), dtype=np.float32)
    offsets = np.arange(-half_width, half_width + 1)
    return traces[valid[:, None] + offsets[None, :]][:, :, channel_indices]


def matched_filter_score(
    traces: np.ndarray, channel_indices: np.ndarray, template: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    whitened_template = template / noise[None, :]
    denominator = float(np.sum(whitened_template**2))
    score = np.zeros(traces.shape[0], dtype=np.float32)
    for local_index, trace_index in enumerate(channel_indices):
        whitened_trace = traces[:, trace_index] / noise[local_index]
        kernel = whitened_template[::-1, local_index]
        score += fftconvolve(whitened_trace, kernel, mode="same").astype(np.float32)
    return score / max(denominator, np.finfo(float).eps)


def waveform_cosine(
    waveforms: np.ndarray, template: np.ndarray, noise: np.ndarray
) -> np.ndarray:
    """Noise-scaled cosine similarity to the multi-channel template."""
    if waveforms.shape[0] == 0:
        return np.empty(0, dtype=float)
    scaled_waveforms = waveforms / noise[None, None, :]
    scaled_template = template / noise[None, :]
    flat_waveforms = scaled_waveforms.reshape(waveforms.shape[0], -1)
    flat_template = scaled_template.reshape(-1)
    numerator = flat_waveforms @ flat_template
    denominator = np.linalg.norm(flat_waveforms, axis=1) * np.linalg.norm(flat_template)
    return numerator / np.maximum(denominator, np.finfo(float).eps)


def plot_missed_candidates(
    path: Path,
    waveforms: np.ndarray,
    template: np.ndarray,
    scores: np.ndarray,
    max_examples: int = 16,
) -> None:
    n = min(max_examples, waveforms.shape[0])
    if n == 0:
        return
    peak_channel = int(np.argmax(np.ptp(template, axis=0)))
    order = np.argsort(scores)[::-1][:n]
    fig, axes = plt.subplots(4, 4, figsize=(12, 9), sharex=True, sharey=True)
    for axis, index in zip(axes.flat, order):
        axis.plot(waveforms[index, :, peak_channel], color="0.25", lw=1)
        axis.plot(template[:, peak_channel], color="tab:red", lw=1.5)
        axis.set_title(f"score={scores[index]:.2f}", fontsize=8)
    for axis in axes.flat[n:]:
        axis.set_visible(False)
    fig.suptitle("Raw candidates with no nearby sorted spike; red = unit template")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(PROBES), default="imec1")
    parser.add_argument("--units", type=int, nargs="+", default=[338, 265, 294])
    parser.add_argument("--channel-radius-um", type=float, default=100.0)
    parser.add_argument("--waveform-ms", type=float, default=2.0)
    parser.add_argument("--match-tolerance-ms", type=float, default=0.5)
    parser.add_argument("--threshold-sigma", type=float, default=6.0)
    parser.add_argument(
        "--detector",
        choices=["peak", "matched"],
        default="peak",
        help="Use the deliberately simple peak detector by default.",
    )
    parser.add_argument("--max-template-spikes", type=int, default=250)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testing/outputs/luke_raw_high_amplitude_recovery"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PROBES[args.probe]
    output_dir = args.output_dir / args.probe
    output_dir.mkdir(parents=True, exist_ok=True)
    sorting = load_sorting(config)
    union, per_unit_raw_channels, depths = unit_channels(
        sorting, args.units, args.channel_radius_um
    )
    union_lookup = {int(channel): index for index, channel in enumerate(union)}
    per_unit_trace_channels = {
        unit: np.array([union_lookup[int(channel)] for channel in channels])
        for unit, channels in per_unit_raw_channels.items()
    }

    fs = config.sample_rate_hz
    half_width = int(round(args.waveform_ms * 1e-3 * fs / 2))
    tolerance = int(round(args.match_tolerance_ms * 1e-3 * fs))
    windows = DEFAULT_WINDOWS[args.probe]
    source_name, source_start_s, source_duration_s = windows[0]
    source_traces, source_start = load_conditioned_window(
        config, union, source_start_s, source_duration_s
    )

    templates: dict[int, np.ndarray] = {}
    template_noise: dict[int, np.ndarray] = {}
    calibration: dict[int, dict[str, float]] = {}
    source_validation_samples: dict[int, np.ndarray] = {}
    for unit_id in args.units:
        absolute = np.asarray(sorting["times"][sorting["clusters"] == unit_id], dtype=np.int64)
        centers = absolute - source_start
        centers = centers[(centers >= half_width) & (centers < source_traces.shape[0] - half_width)]
        if centers.size < 5:
            raise RuntimeError(
                f"Unit {unit_id} has only {centers.size} usable spikes in template window"
            )
        if centers.size > args.max_template_spikes:
            take = np.linspace(0, centers.size - 1, args.max_template_spikes).astype(int)
            centers = centers[take]
        channels = per_unit_trace_channels[unit_id]
        train_centers = centers[::2]
        validation_centers = centers[1::2]
        train_waveforms = extract_waveforms(
            source_traces, train_centers, channels, half_width
        )
        templates[unit_id] = np.median(train_waveforms, axis=0)
        noise = robust_sigma(source_traces[:, channels], axis=0)
        template_noise[unit_id] = np.maximum(noise, np.finfo(np.float32).eps)
        validation_waveforms = extract_waveforms(
            source_traces, validation_centers, channels, half_width
        )
        validation_cosine = waveform_cosine(
            validation_waveforms, templates[unit_id], template_noise[unit_id]
        )
        scaled_template = templates[unit_id] / template_noise[unit_id][None, :]
        scaled_validation = validation_waveforms / template_noise[unit_id][None, None, :]
        amplitude = (
            scaled_validation.reshape(validation_waveforms.shape[0], -1)
            @ scaled_template.reshape(-1)
        ) / np.sum(scaled_template**2)
        calibration[unit_id] = {
            "cosine_min": float(np.quantile(validation_cosine, 0.05)),
            "amplitude_min": float(max(0.0, 0.5 * np.quantile(amplitude, 0.02))),
            "amplitude_max": float(1.5 * np.quantile(amplitude, 0.98)),
            "validation_cosine_median": float(np.median(validation_cosine)),
        }
        peak_channel = int(np.argmax(-np.min(templates[unit_id], axis=0)))
        validation_peak_amplitude = -np.min(
            validation_waveforms[:, :, peak_channel], axis=1
        )
        calibration[unit_id]["peak_channel"] = peak_channel
        calibration[unit_id]["peak_amplitude_min"] = float(
            np.quantile(validation_peak_amplitude, 0.10)
        )
        source_validation_samples[unit_id] = validation_centers + source_start

    summary_rows: list[dict] = []
    candidate_rows: list[dict] = []
    for window_name, start_s, duration_s in windows:
        if window_name == source_name:
            traces, window_start = source_traces, source_start
        else:
            traces, window_start = load_conditioned_window(
                config, union, start_s, duration_s
            )
        window_stop = window_start + traces.shape[0]
        all_mask = (sorting["times"] >= window_start) & (sorting["times"] < window_stop)

        for unit_id in args.units:
            channels = per_unit_trace_channels[unit_id]
            template = templates[unit_id]
            edge = max(half_width, int(round(0.05 * fs)))
            limits = calibration[unit_id]
            if args.detector == "peak":
                peak_trace_channel = channels[int(limits["peak_channel"])]
                score = -traces[:, peak_trace_channel]
                score_noise = float(robust_sigma(score[edge:-edge]))
                threshold = max(
                    float(limits["peak_amplitude_min"]),
                    args.threshold_sigma * score_noise,
                )
            else:
                score = matched_filter_score(
                    traces, channels, template, template_noise[unit_id]
                )
                score_noise = float(robust_sigma(score[edge:-edge]))
                threshold = args.threshold_sigma * score_noise
            candidates, properties = find_peaks(
                score,
                height=threshold,
                distance=max(1, int(round(0.00067 * fs))),
            )
            candidates = candidates[(candidates >= edge) & (candidates < traces.shape[0] - edge)]
            candidate_scores = score[candidates]
            candidate_waveforms = extract_waveforms(
                traces, candidates, channels, half_width
            )
            candidate_cosine = waveform_cosine(
                candidate_waveforms, template, template_noise[unit_id]
            )
            if args.detector == "peak":
                keep = np.ones(candidates.size, dtype=bool)
            else:
                shape_ok = candidate_cosine >= limits["cosine_min"]
                amplitude_ok = (candidate_scores >= limits["amplitude_min"]) & (
                    candidate_scores <= limits["amplitude_max"]
                )
                keep = shape_ok & amplitude_ok
            candidates = candidates[keep]
            candidate_scores = candidate_scores[keep]
            candidate_cosine = candidate_cosine[keep]
            candidate_waveforms = candidate_waveforms[keep]
            absolute_candidates = candidates + window_start
            target_all = np.asarray(
                sorting["times"][sorting["clusters"] == unit_id], dtype=np.int64
            )
            target = target_all[(target_all >= window_start) & (target_all < window_stop)]
            local_mask = all_mask & (
                np.abs(sorting["positions"][:, 1] - depths[unit_id])
                <= args.channel_radius_um
            )
            local_sorted = np.sort(
                np.asarray(sorting["times"][local_mask], dtype=np.int64)
            )
            validation_target = (
                source_validation_samples[unit_id]
                if window_name == source_name
                else target
            )
            matched_target = nearest_event_mask(absolute_candidates, target, tolerance)
            matched_any = nearest_event_mask(
                absolute_candidates, local_sorted, tolerance
            )
            recovered_target = nearest_event_mask(
                validation_target, absolute_candidates, tolerance
            )
            classification = np.where(matched_target, "target", np.where(matched_any, "other", "missed"))

            counts = pd.Series(classification).value_counts()
            summary_rows.append(
                {
                    "probe": args.probe,
                    "unit_id": unit_id,
                    "unit_depth_um": depths[unit_id],
                    "window": window_name,
                    "start_seconds": start_s,
                    "duration_seconds": duration_s,
                    "n_sorted_target": target.size,
                    "n_sorted_local": local_sorted.size,
                    "n_validation_target": validation_target.size,
                    "n_candidates": candidates.size,
                    "n_target": int(counts.get("target", 0)),
                    "n_other": int(counts.get("other", 0)),
                    "n_missed": int(counts.get("missed", 0)),
                    "target_spike_recovery": float(np.mean(recovered_target)) if target.size else np.nan,
                    "candidate_missed_fraction": float(np.mean(classification == "missed")) if candidates.size else np.nan,
                    "score_noise_sigma": score_noise,
                    "score_threshold": threshold,
                    "cosine_threshold": limits["cosine_min"] if args.detector == "matched" else np.nan,
                    "amplitude_min": limits["amplitude_min"] if args.detector == "matched" else threshold,
                    "amplitude_max": limits["amplitude_max"] if args.detector == "matched" else np.inf,
                }
            )
            for sample, value, cosine, label in zip(
                absolute_candidates, candidate_scores, candidate_cosine, classification
            ):
                candidate_rows.append(
                    {
                        "probe": args.probe,
                        "unit_id": unit_id,
                        "window": window_name,
                        "sample_index": int(sample),
                        "time_seconds": float(sample / fs),
                        "score": float(value),
                        "cosine": float(cosine),
                        "classification": label,
                    }
                )

            missed = candidates[classification == "missed"]
            missed_scores = score[missed]
            missed_waveforms = candidate_waveforms[classification == "missed"]
            plot_missed_candidates(
                output_dir / f"{window_name}_unit{unit_id}_missed.png",
                missed_waveforms,
                template,
                missed_scores,
            )
        # A 120 s conditioned window can approach 1 GB.  Release it before
        # loading the next window, including the source/template window.
        del traces
        if window_name == source_name:
            del source_traces

    summary = pd.DataFrame(summary_rows)
    candidates = pd.DataFrame(candidate_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    manifest = {
        "probe": args.probe,
        "units": args.units,
        "raw_path": str(config.raw_path),
        "sorting_path": str(config.sorting_path),
        "sample_rate_hz": fs,
        "windows": windows,
        "conditioning": "raw counts; per-sample global median; 3rd-order 300 Hz zero-phase high-pass",
        "channel_radius_um": args.channel_radius_um,
        "waveform_ms": args.waveform_ms,
        "match_tolerance_ms": args.match_tolerance_ms,
        "threshold_sigma": args.threshold_sigma,
        "detector": args.detector,
        "interpretation_guardrail": "Do not interpret miss fraction unless target_spike_recovery is high.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
