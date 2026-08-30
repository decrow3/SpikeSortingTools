"""Audit Luke's external voltage-resampling implementation on fixed events.

This is a paired, sorter-independent comparison.  Every interpolation variant
uses the same conditioned recording, motion field, geometry, and reviewed event
times.  The variants isolate the differences between the Luke pipeline's
low-level SpikeInterface defaults and SpikeInterface's calibrated DREDGE
interpolation preset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_upstream_stage_ablation import (
    DEFAULT_REVIEW,
    MOTION_ROOT,
    RAW_ROOT,
    STREAM_ID,
    build_recording_stages,
    max_channel_shift_correlation,
    robust_sigma,
    selected_events,
)


DEFAULT_OUTPUT = REPO_ROOT / "testing/outputs/luke_interpolation_implementation_audit"


VARIANTS = {
    "pipeline_p1_zero_float": {
        "border_mode": "force_zeros",
        "sigma_um": 20.0,
        "p": 1,
        "cast_int16": False,
    },
    "pipeline_p1_zero_int16": {
        "border_mode": "force_zeros",
        "sigma_um": 20.0,
        "p": 1,
        "cast_int16": True,
    },
    "p2_zero_int16": {
        "border_mode": "force_zeros",
        "sigma_um": 20.0,
        "p": 2,
        "cast_int16": True,
    },
    "p1_extrapolate_int16": {
        "border_mode": "force_extrapolate",
        "sigma_um": 20.0,
        "p": 1,
        "cast_int16": True,
    },
    "official_p2_extrapolate_float": {
        "border_mode": "force_extrapolate",
        "sigma_um": 20.0,
        "p": 2,
        "cast_int16": False,
    },
    "official_p2_extrapolate_int16": {
        "border_mode": "force_extrapolate",
        "sigma_um": 20.0,
        "p": 2,
        "cast_int16": True,
    },
    # SI uses exp(-(d/sigma)^2), whereas native KS4 uses
    # exp(-d^2/(2*sigma^2)); sqrt(2)*20 um matches only kernel width.
    "ks4_width_proxy_p2_extrapolate_int16": {
        "border_mode": "force_extrapolate",
        "sigma_um": float(np.sqrt(2.0) * 20.0),
        "p": 2,
        "cast_int16": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--variant", action="append", dest="variants")
    return parser.parse_args()


def build_variants(current, motion, names: list[str]):
    from spikeinterface.preprocessing import astype
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    result = {"conditioned_baseline": current}
    current_float = astype(current, "float32")
    for name in names:
        spec = VARIANTS[name]
        recording = interpolate_motion(
            current_float,
            motion,
            border_mode=spec["border_mode"],
            spatial_interpolation_method="kriging",
            sigma_um=spec["sigma_um"],
            p=spec["p"],
        )
        if spec["cast_int16"]:
            recording = astype(recording, "int16")
        result[name] = recording
    return result


def anchored_event_metrics(
    traces: np.ndarray,
    channel_depths_um: np.ndarray,
    fs: float,
    review_depth_um: float,
    depth_half_width_um: float = 150.0,
) -> tuple[dict[str, float], np.ndarray]:
    """Measure an event only near its reviewed depth.

    A global argmax of channelwise SNR is unsafe after interpolation because a
    flat/quantized edge channel can have an approximately zero noise estimate.
    The reviewed depth is fixed before looking at interpolation variants and
    the 150-um neighborhood comfortably contains Luke's observed motion.
    """
    traces = np.asarray(traces, dtype=np.float32)
    center = traces.shape[0] // 2
    local_channels = np.flatnonzero(
        np.abs(np.asarray(channel_depths_um) - review_depth_um) <= depth_half_width_um
    )
    if local_channels.size == 0:
        raise ValueError(f"No channels near reviewed depth {review_depth_um}")

    search_half = int(round(0.6e-3 * fs))
    search = traces[center - search_half : center + search_half + 1][:, local_channels]
    local_time, local_channel_index = np.unravel_index(int(np.argmin(search)), search.shape)
    aligned = center - search_half + int(local_time)
    peak_channel = int(local_channels[local_channel_index])
    peak_amplitude = float(max(0.0, -search[local_time, local_channel_index]))

    baseline_exclusion = int(round(2.0e-3 * fs))
    baseline_mask = np.ones(traces.shape[0], dtype=bool)
    baseline_mask[center - baseline_exclusion : center + baseline_exclusion + 1] = False
    noise = robust_sigma(traces[baseline_mask], axis=0)
    local_noise_floor = max(
        float(np.median(noise[local_channels])) * 0.1, np.finfo(float).eps
    )
    peak_noise = max(float(noise[peak_channel]), local_noise_floor)

    core_half = int(round(1.5e-3 * fs))
    core_wave = traces[aligned - core_half : aligned + core_half + 1].copy()
    if core_wave.shape[0] != 2 * core_half + 1:
        raise ValueError("Snippet is too short around the aligned event")
    local_traces = traces[:, local_channels]
    return {
        "aligned_offset_ms": float((aligned - center) * 1e3 / fs),
        "anchor_peak_channel": peak_channel,
        "anchor_peak_depth_um": float(channel_depths_um[peak_channel]),
        "anchor_peak_depth_error_um": float(channel_depths_um[peak_channel] - review_depth_um),
        "anchor_peak_amplitude_counts": peak_amplitude,
        "anchor_peak_snr": float(peak_amplitude / peak_noise),
        "snippet_rms_counts": float(np.sqrt(np.mean(np.square(traces, dtype=np.float64)))),
        "local_snippet_rms_counts": float(
            np.sqrt(np.mean(np.square(local_traces, dtype=np.float64)))
        ),
        "zero_fraction": float(np.mean(traces == 0)),
        "local_zero_fraction": float(np.mean(local_traces == 0)),
    }, core_wave


def extract_metrics(recordings: dict, events: pd.DataFrame, fs: float) -> pd.DataFrame:
    half_samples = int(round(5.0e-3 * fs))
    depths = np.asarray(next(iter(recordings.values())).get_channel_locations())[:, 1]
    baseline_waves: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for variant, recording in recordings.items():
        print(f"Extracting {len(events)} events from {variant}", flush=True)
        for event in events.itertuples(index=False):
            sample = int(event.sample_index)
            traces = recording.get_traces(
                start_frame=sample - half_samples,
                end_frame=sample + half_samples + 1,
                return_scaled=False,
            )
            metrics, core_wave = anchored_event_metrics(
                traces, depths, fs, float(event.peak_depth_um)
            )
            if variant == "conditioned_baseline":
                baseline_waves[event.review_id] = core_wave
                correlation = 1.0
            else:
                correlation = max_channel_shift_correlation(
                    baseline_waves[event.review_id], core_wave
                )
            rows.append(
                {
                    "review_id": event.review_id,
                    "sample_index": sample,
                    "time_seconds": float(event.time_seconds),
                    "window": event.window,
                    "status": event.status,
                    "review_label": event.review_label,
                    "variant": variant,
                    "correlation_to_conditioned_baseline": correlation,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def add_paired_changes(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add within-event changes relative to the conditioned baseline."""
    baseline = metrics.loc[metrics["variant"] == "conditioned_baseline"].set_index(
        "review_id"
    )
    if baseline.index.has_duplicates:
        raise ValueError("review_id must be unique in the baseline")
    result = metrics.copy()
    numeric = [
        "anchor_peak_amplitude_counts",
        "anchor_peak_snr",
        "snippet_rms_counts",
        "local_snippet_rms_counts",
        "anchor_peak_depth_error_um",
        "zero_fraction",
        "local_zero_fraction",
    ]
    for column in numeric:
        reference = result["review_id"].map(baseline[column])
        result[f"delta_{column}"] = result[column] - reference
    for column in (
        "anchor_peak_amplitude_counts",
        "anchor_peak_snr",
        "snippet_rms_counts",
        "local_snippet_rms_counts",
    ):
        reference = result["review_id"].map(baseline[column])
        result[f"ratio_{column}"] = result[column] / reference.replace(0, np.nan)
    return result


def summarize(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ratio_anchor_peak_amplitude_counts",
        "ratio_anchor_peak_snr",
        "ratio_snippet_rms_counts",
        "ratio_local_snippet_rms_counts",
        "correlation_to_conditioned_baseline",
        "delta_anchor_peak_depth_error_um",
        "zero_fraction",
        "local_zero_fraction",
    ]
    group_columns = ["window", "review_label", "variant"]
    grouped = paired.groupby(group_columns, dropna=False)
    summary = grouped[columns].median().add_prefix("median_").reset_index()
    counts = grouped.size().rename("n_events").reset_index()
    return counts.merge(summary, on=group_columns, validate="one_to_one")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/luke-interpolation-audit-numba")
    import spikeinterface
    import spikeinterface.extractors as se
    from spikeinterface.core.motion import Motion
    from spikeinterface.preprocessing.motion import get_motion_parameters_preset

    names = args.variants or list(VARIANTS)
    unknown = set(names) - set(VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")

    raw = se.read_spikeglx(
        folder_path=RAW_ROOT, load_sync_channel=False, stream_id=STREAM_ID
    )
    stages, bad_ids, gain = build_recording_stages(raw)
    current = stages["current_conditioned"]
    motion_dir = MOTION_ROOT / "dredge-motion"
    motion = Motion(
        displacement=np.load(motion_dir / "motion.npy"),
        temporal_bins_s=np.load(motion_dir / "time_bins.npy"),
        spatial_bins_um=np.load(motion_dir / "depth_bins.npy"),
    )
    recordings = build_variants(current, motion, names)
    events = selected_events(args.review_events, args.max_events)
    fs = float(raw.get_sampling_frequency())
    metrics = extract_metrics(recordings, events, fs)
    paired = add_paired_changes(metrics)
    summary = summarize(paired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_event_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "paired_event_summary.csv", index=False)
    manifest = {
        "spikeinterface_version": spikeinterface.__version__,
        "raw_root": str(RAW_ROOT),
        "stream_id": STREAM_ID,
        "review_events": str(args.review_events),
        "motion_dir": str(motion_dir),
        "baseline": "current_conditioned",
        "n_events": len(events),
        "bad_channel_ids": [str(value) for value in bad_ids],
        "gain_uv_per_bit": gain,
        "variants": {name: VARIANTS[name] for name in names},
        "installed_official_dredge_preset": get_motion_parameters_preset("dredge"),
        "scope_note": (
            "Paired waveform audit only; no variant has passed sorter-level validation. "
            "The KS4-width proxy matches kernel width, not regularization, normalization, "
            "preprocessing order, or streaming implementation."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda value: value.tolist()) + "\n"
    )
    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
