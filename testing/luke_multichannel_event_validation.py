"""Build a blinded multichannel review set for Luke raw-event candidates.

This is the measurement-control step for the raw-event recovery diagnostic.  It
does not call an unmatched event a missed spike.  Instead, it samples locally
matched and unmatched candidates without exposing that status to the reviewer,
extracts small waveforms directly from the original AP binary, computes simple
spatial/temporal features, and writes blinded contact sheets plus a blank label
form.  The key is saved separately and should only be joined after review.

The automatic gate is deliberately advisory.  Its thresholds are explicit in
the manifest and the primary result is the blinded human label.
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

try:
    from .luke_raw_high_amplitude_recovery import PROBES, load_sorting, robust_sigma
except ImportError:  # Allow direct execution: python testing/<script>.py
    from luke_raw_high_amplitude_recovery import PROBES, load_sorting, robust_sigma


@dataclass(frozen=True)
class GateThresholds:
    peak_snr_min: float = 6.0
    local_energy_fraction_min: float = 0.50
    common_mode_ratio_max: float = 0.50
    active_channels_min: int = 1
    active_channels_max: int = 24
    trough_width_ms_min: float = 0.067
    trough_width_ms_max: float = 0.80
    peak_offset_ms_max: float = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=sorted(PROBES), default="imec1")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "testing/outputs/luke_raw_high_amplitude_recovery/imec1/candidates.csv"
        ),
    )
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20250804)
    parser.add_argument("--local-channels", type=int, default=5)
    parser.add_argument("--read-pad-ms", type=float, default=10.0)
    parser.add_argument("--display-ms", type=float, default=2.0)
    parser.add_argument("--alignment-search-ms", type=float, default=0.4)
    parser.add_argument("--local-radius-um", type=float, default=100.0)
    parser.add_argument("--active-snr", type=float, default=2.5)
    parser.add_argument("--events-per-sheet", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("testing/outputs/luke_multichannel_event_validation"),
    )
    return parser.parse_args()


def stratified_sample(
    candidates: pd.DataFrame, n_per_class: int, seed: int
) -> pd.DataFrame:
    """Sample equal matched/unmatched sets while spreading rows across strata."""
    required = {"unit_id", "window", "sample_index", "classification"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")

    work = candidates.copy()
    work["status"] = np.where(work["classification"] == "missed", "unmatched", "matched")
    # One physical event can be detected for multiple representative units.  Keep
    # one row per status/sample so the blinded set contains independent events.
    work = work.drop_duplicates(["status", "sample_index"])
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    used_samples: set[int] = set()
    for status in ("matched", "unmatched"):
        pool = work[
            (work["status"] == status) & ~work["sample_index"].isin(used_samples)
        ].copy()
        if len(pool) < n_per_class:
            raise ValueError(
                f"Requested {n_per_class} {status} events but only {len(pool)} are available"
            )
        groups = [group for _, group in pool.groupby(["window", "unit_id"], sort=True)]
        order = [rng.permutation(group.index.to_numpy()) for group in groups]
        chosen: list[int] = []
        cursor = 0
        while len(chosen) < n_per_class:
            made_progress = False
            for group_order in order:
                if cursor < len(group_order) and len(chosen) < n_per_class:
                    chosen.append(int(group_order[cursor]))
                    made_progress = True
            if not made_progress:
                break
            cursor += 1
        chosen_rows = pool.loc[chosen]
        selected.append(chosen_rows)
        used_samples.update(chosen_rows["sample_index"].astype(int))

    result = pd.concat(selected, ignore_index=True)
    result = result.iloc[rng.permutation(len(result))].reset_index(drop=True)
    result.insert(0, "review_id", [f"E{i:04d}" for i in range(1, len(result) + 1)])
    return result


def unit_centers(sorting: dict[str, np.ndarray], unit_ids: np.ndarray) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for unit_id in np.unique(unit_ids):
        mask = sorting["clusters"] == unit_id
        if not np.any(mask):
            raise ValueError(f"Unit {unit_id} has no sorted spikes")
        centers[int(unit_id)] = np.median(sorting["positions"][mask], axis=0)
    return centers


def nearest_channels(
    channel_positions: np.ndarray, center: np.ndarray, count: int
) -> np.ndarray:
    scale = np.array([1.0, 1.0], dtype=float)
    distance = np.linalg.norm((channel_positions - center[None, :]) * scale, axis=1)
    return np.argsort(distance)[:count]


def half_amplitude_width_ms(waveform: np.ndarray, trough: int, fs: float) -> float:
    amplitude = -float(waveform[trough])
    if not np.isfinite(amplitude) or amplitude <= 0:
        return float("nan")
    below = waveform <= -0.5 * amplitude
    left = trough
    right = trough
    while left > 0 and below[left - 1]:
        left -= 1
    while right + 1 < waveform.size and below[right + 1]:
        right += 1
    return float((right - left + 1) * 1e3 / fs)


def evaluate_gate(metrics: dict[str, float], gate: GateThresholds) -> tuple[bool, str]:
    checks = {
        "low_snr": metrics["peak_snr"] >= gate.peak_snr_min,
        "diffuse": metrics["local_energy_fraction"] >= gate.local_energy_fraction_min,
        "common_mode": metrics["common_mode_ratio"] <= gate.common_mode_ratio_max,
        "active_channel_count": gate.active_channels_min
        <= metrics["active_channels"]
        <= gate.active_channels_max,
        "trough_width": gate.trough_width_ms_min
        <= metrics["trough_width_ms"]
        <= gate.trough_width_ms_max,
        "peak_offset": abs(metrics["peak_offset_ms"]) <= gate.peak_offset_ms_max,
        "saturation": not bool(metrics["near_saturation"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, ";".join(failed)


def extract_event(
    raw: np.memmap,
    sample: int,
    fs: float,
    channel_positions: np.ndarray,
    search_channels: np.ndarray,
    local_channel_count: int,
    highpass: np.ndarray,
    read_pad: int,
    display_half: int,
    search_half: int,
    local_radius_um: float,
    active_snr: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    start = sample - read_pad
    stop = sample + read_pad + 1
    if start < 0 or stop > raw.shape[0]:
        raise ValueError(f"Sample {sample} is too close to the recording boundary")

    raw_block = np.asarray(raw[start:stop, :384], dtype=np.float32)
    common_mode = np.median(raw_block, axis=1)
    filtered_common_mode = sosfiltfilt(highpass, common_mode).astype(np.float32)
    conditioned = raw_block - common_mode[:, None]
    conditioned = sosfiltfilt(highpass, conditioned, axis=0).astype(np.float32)
    center = read_pad
    baseline_mask = np.ones(conditioned.shape[0], dtype=bool)
    exclusion = int(round(1.5e-3 * fs))
    baseline_mask[max(0, center - exclusion) : center + exclusion + 1] = False
    noise = robust_sigma(conditioned[baseline_mask], axis=0)
    noise = np.maximum(noise, np.finfo(np.float32).eps)
    search = conditioned[
        center - search_half : center + search_half + 1, search_channels
    ] / noise[search_channels][None, :]
    local_flat = int(np.argmin(search))
    local_time, local_channel_index = np.unravel_index(local_flat, search.shape)
    aligned = center - search_half + local_time
    peak_channel = int(search_channels[local_channel_index])
    local_channels = nearest_channels(
        channel_positions, channel_positions[peak_channel], local_channel_count
    )
    wave_start = aligned - display_half
    wave_stop = aligned + display_half + 1
    if wave_start < 0 or wave_stop > conditioned.shape[0]:
        raise ValueError("Display window exceeds conditioned event block")

    # Measure the spatial footprint at the aligned trough.  Taking an
    # independent 0.8 ms minimum on every channel strongly inflates the number
    # of active channels through multiple comparisons.
    event_half = 1
    event_slice = conditioned[aligned - event_half : aligned + event_half + 1]
    negative_amplitude = np.maximum(0.0, -np.min(event_slice, axis=0))
    snr = negative_amplitude / noise
    peak_snr = float(snr[peak_channel])
    active = snr >= active_snr
    weights = np.where(active, np.maximum(snr**2 - active_snr**2, 0.0), 0.0)
    peak_position = channel_positions[peak_channel]
    local = np.abs(channel_positions[:, 1] - peak_position[1]) <= local_radius_um
    weight_total = float(np.sum(weights))
    local_fraction = float(np.sum(weights[local]) / weight_total) if weight_total else 0.0
    if weight_total:
        depth_center = float(np.sum(weights * channel_positions[:, 1]) / weight_total)
        depth_sd = float(
            np.sqrt(np.sum(weights * (channel_positions[:, 1] - depth_center) ** 2) / weight_total)
        )
    else:
        depth_sd = float("nan")

    local_waveforms = conditioned[wave_start:wave_stop, local_channels]
    peak_waveform = conditioned[wave_start:wave_stop, peak_channel]
    trough = int(np.argmin(peak_waveform))
    common_peak = float(
        np.max(np.abs(filtered_common_mode[aligned - event_half : aligned + event_half + 1]))
    )
    peak_amplitude = max(float(-np.min(peak_waveform)), np.finfo(float).eps)
    metrics = {
        "aligned_sample_index": int(sample + aligned - center),
        "peak_channel": peak_channel,
        "peak_depth_um": float(peak_position[1]),
        "peak_snr": peak_snr,
        "peak_amplitude_counts": peak_amplitude,
        "peak_offset_ms": float((aligned - center) * 1e3 / fs),
        "trough_width_ms": half_amplitude_width_ms(peak_waveform, trough, fs),
        "positive_to_negative_ratio": float(max(0.0, np.max(peak_waveform)) / peak_amplitude),
        "active_channels": int(np.sum(active)),
        "local_energy_fraction": local_fraction,
        "footprint_depth_sd_um": depth_sd,
        "common_mode_ratio": float(common_peak / peak_amplitude),
        "near_saturation": bool(np.any(np.abs(raw_block) >= 32760)),
    }
    profile = np.column_stack((channel_positions[:, 1], snr)).astype(np.float32)
    return local_waveforms, profile, metrics


def plot_review_sheets(
    output_dir: Path,
    review_ids: list[str],
    waveforms: np.ndarray,
    fs: float,
    events_per_sheet: int,
) -> list[str]:
    sheet_paths: list[str] = []
    time_ms = (np.arange(waveforms.shape[1]) - waveforms.shape[1] // 2) * 1e3 / fs
    for sheet_index, start in enumerate(range(0, len(review_ids), events_per_sheet), 1):
        stop = min(len(review_ids), start + events_per_sheet)
        rows = 4
        cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True)
        for axis, event_index in zip(axes.flat, range(start, stop)):
            event = waveforms[event_index]
            scale = max(float(robust_sigma(event)), np.finfo(float).eps)
            offsets = np.arange(event.shape[1]) * 5.0
            axis.plot(time_ms, event / scale + offsets[None, :], lw=0.8)
            axis.axvline(0, color="0.7", lw=0.6)
            axis.set_title(review_ids[event_index], fontsize=9)
            axis.set_yticks([])
        for axis in axes.flat[stop - start :]:
            axis.set_visible(False)
        fig.supxlabel("Time from aligned trough (ms)")
        fig.supylabel("Five nearby channels (normalized, vertically offset)")
        fig.suptitle(f"Blinded raw-event review — sheet {sheet_index}")
        fig.tight_layout()
        filename = f"review_sheet_{sheet_index:02d}.png"
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)
        sheet_paths.append(filename)
    return sheet_paths


def main() -> None:
    args = parse_args()
    if args.local_channels < 3 or args.local_channels % 2 == 0:
        raise ValueError("--local-channels must be an odd integer of at least 3")
    if args.events_per_sheet > 20:
        raise ValueError("--events-per-sheet must be 20 or fewer")

    config = PROBES[args.probe]
    output_dir = args.output_dir / args.probe
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    selected = stratified_sample(candidates, args.n_per_class, args.seed)
    sorting = load_sorting(config)
    channel_positions = np.asarray(sorting["channel_positions"])
    centers = unit_centers(sorting, selected["unit_id"].to_numpy())
    per_unit_search_channels = {
        unit_id: np.flatnonzero(
            np.abs(channel_positions[:, 1] - center[1]) <= args.local_radius_um
        )
        for unit_id, center in centers.items()
    }

    n_total = config.raw_path.stat().st_size // (2 * config.n_saved_channels)
    raw = np.memmap(
        config.raw_path,
        mode="r",
        dtype="<i2",
        shape=(n_total, config.n_saved_channels),
    )
    fs = config.sample_rate_hz
    read_pad = int(round(args.read_pad_ms * 1e-3 * fs))
    display_half = int(round(args.display_ms * 1e-3 * fs / 2))
    search_half = int(round(args.alignment_search_ms * 1e-3 * fs))
    if read_pad <= display_half + search_half:
        raise ValueError("--read-pad-ms must exceed the display and alignment windows")
    highpass = butter(3, 300.0, btype="highpass", fs=fs, output="sos")
    gate = GateThresholds()

    metrics_rows: list[dict] = []
    waveforms: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    for row in selected.itertuples(index=False):
        waveform, profile, metrics = extract_event(
            raw=raw,
            sample=int(row.sample_index),
            fs=fs,
            channel_positions=channel_positions,
            search_channels=per_unit_search_channels[int(row.unit_id)],
            local_channel_count=args.local_channels,
            highpass=highpass,
            read_pad=read_pad,
            display_half=display_half,
            search_half=search_half,
            local_radius_um=args.local_radius_um,
            active_snr=args.active_snr,
        )
        gate_pass, gate_failures = evaluate_gate(metrics, gate)
        metrics_rows.append(
            {
                "review_id": row.review_id,
                **metrics,
                "automatic_neural_like": gate_pass,
                "automatic_gate_failures": gate_failures,
            }
        )
        waveforms.append(waveform)
        profiles.append(profile)

    metrics = pd.DataFrame(metrics_rows)
    review_form = metrics[["review_id"]].copy()
    review_form["review_label"] = ""
    review_form["review_confidence"] = ""
    review_form["review_notes"] = ""
    key = selected[
        [
            "review_id",
            "status",
            "classification",
            "window",
            "unit_id",
            "sample_index",
            "time_seconds",
        ]
    ].merge(metrics, on="review_id", validate="one_to_one")

    waveform_array = np.stack(waveforms)
    profile_array = np.stack(profiles)
    np.savez_compressed(
        output_dir / "review_waveforms.npz",
        review_id=review_form["review_id"].to_numpy(),
        waveforms=waveform_array,
        spatial_profiles=profile_array,
        sample_rate_hz=np.array(fs),
    )
    review_form.to_csv(output_dir / "review_labels.csv", index=False)
    metrics.to_csv(output_dir / "blinded_metrics.csv", index=False)
    key.to_csv(output_dir / "review_key.csv", index=False)
    sheets = plot_review_sheets(
        output_dir,
        review_form["review_id"].tolist(),
        waveform_array,
        fs,
        args.events_per_sheet,
    )

    summary = (
        key.groupby("status", observed=True)
        .agg(
            n_events=("review_id", "size"),
            automatic_neural_like_fraction=("automatic_neural_like", "mean"),
            median_peak_snr=("peak_snr", "median"),
            median_local_energy_fraction=("local_energy_fraction", "median"),
            median_active_channels=("active_channels", "median"),
            median_common_mode_ratio=("common_mode_ratio", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "automatic_screen_summary.csv", index=False)
    manifest = {
        "probe": args.probe,
        "raw_path": str(config.raw_path),
        "sorting_path": str(config.sorting_path),
        "candidate_source": str(args.candidates),
        "seed": args.seed,
        "sampling": {
            "n_per_class": args.n_per_class,
            "classes": {
                "matched": "candidate classified target or other",
                "unmatched": "candidate classified missed",
            },
            "strata": ["window", "unit_id"],
            "deduplication": ["sample_index across both review arms"],
        },
        "conditioning": "original int16 AP; per-sample global median; 3rd-order 300 Hz zero-phase high-pass",
        "review": {
            "labels": ["neural", "artifact", "uncertain"],
            "primary_endpoint": "fraction manually labeled neural in unmatched versus matched sets",
            "key_file": "review_key.csv (do not open before labeling)",
            "label_file": "review_labels.csv",
            "sheets": sheets,
        },
        "automatic_gate": {
            "role": "advisory prioritization only; not ground truth",
            "thresholds": asdict(gate),
            "active_channel_snr_threshold": args.active_snr,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved blinded review set to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
