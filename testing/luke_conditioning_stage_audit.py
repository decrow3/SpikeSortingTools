"""Atomic, motion-free conditioning audit for the Luke imec1 recording.

This program deliberately stops before template learning and sorting.  It
replays fixed two-second batches from good, neutral, and pathological epochs
through one operation at a time and records polarity, compact event density,
shared correlation, saturation-boundary ringing, covariance, and reviewed-event
waveform metrics.

The stage graph is:

    raw -> phase -> saturation policy -> KS CAR -> KS high-pass
        -> channel-191 policy -> whitening

No motion estimate or voltage registration is loaded anywhere in this module.
Use ``--smoke`` first; the default panel is still snippet-scale, not a sort.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, maximum_filter1d


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_two_axis_pilot import (
    DTYPE,
    N_CHANNELS,
    N_SAVED_CHANNELS,
    RAW_BINARY,
    SOURCE_FRAMES,
    SOURCE_PROBE,
    SOURCE_PROVENANCE,
)
from testing.luke_claimmask_window_sweep import load_reference_settings
from testing.luke_upstream_stage_ablation import DEFAULT_REVIEW, event_metrics
from testing.luke_yates_detection_stage_audit import spatial_neighbors


DEFAULT_OUTPUT = Path("testing/outputs/luke_conditioning_stage_audit")
DEFAULT_SATURATION_INDEX = Path(
    "testing/outputs/luke_motion_candidate_results/raw_voltage_audit/"
    "raw_channel_batch_metrics.csv"
)
SATURATION_UV = 500.0


@dataclass(frozen=True)
class AuditWindow:
    name: str
    start_s: float
    duration_s: float = 120.0


WINDOWS = (
    AuditWindow("good", 7095.0),
    AuditWindow("neutral", 7215.0),
    AuditWindow("pathological", 8160.0),
)


CORE_STAGES = (
    "raw",
    "phase_int16",
    "phase_float32",
    "phase_car",
    "phase_clip_car",
    "phase_car_highpass",
    "phase_clip_car_highpass",
    "phase_car_highpass_postblank_1ms",
    "phase_car_highpass_interp191",
    "phase_interp191_float_car_highpass",
    "current_source_int16_car_highpass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-events", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--batch-duration-s", type=float, default=2.0)
    parser.add_argument("--batches-per-window", type=int, default=12)
    parser.add_argument("--max-events-per-window", type=int, default=40)
    parser.add_argument("--saturation-index", type=Path, default=DEFAULT_SATURATION_INDEX)
    parser.add_argument("--saturation-batches", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="Recompute only reviewed-event metrics and preserve existing batch outputs",
    )
    return parser.parse_args()


def robust_sigma(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis)
    return np.maximum(mad / 0.6744897501960817, np.finfo(float).eps)


def ks_center_car(values: np.ndarray) -> np.ndarray:
    """Kilosort's per-channel mean removal followed by across-channel median."""
    values = np.asarray(values, dtype=np.float32)
    centered = values - values.mean(axis=0, keepdims=True)
    return centered - np.median(centered, axis=1, keepdims=True)


@lru_cache(maxsize=8)
def _ks_highpass_fft(fs: float, n_samples: int) -> torch.Tensor:
    from kilosort.io import fft_highpass
    from kilosort.preprocessing import get_highpass_filter

    hp = get_highpass_filter(fs=fs, cutoff=300, device=torch.device("cpu"))
    return fft_highpass(hp, NT=n_samples)


def ks_highpass(values: np.ndarray, fs: float) -> np.ndarray:
    """Apply Kilosort4's own FIR/FFT high-pass implementation on CPU."""
    x = torch.as_tensor(np.asarray(values).T, dtype=torch.float32)
    fwav = _ks_highpass_fft(float(fs), x.shape[1])
    result = torch.real(torch.fft.ifft(torch.fft.fft(x) * torch.conj(fwav)))
    result = torch.fft.fftshift(result, dim=-1)
    return result.T.numpy()


def clip_saturation(values: np.ndarray, threshold_counts: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    result[np.abs(result) >= threshold_counts] = float(np.median(result))
    return result


def dilate_time_mask(mask: np.ndarray, radius_samples: int) -> np.ndarray:
    if radius_samples <= 0:
        return np.asarray(mask, dtype=bool).copy()
    structure = np.ones((2 * radius_samples + 1, 1), dtype=bool)
    return binary_dilation(np.asarray(mask, dtype=bool), structure=structure)


def apply_postfilter_blank(
    values: np.ndarray, saturation_mask: np.ndarray, radius_samples: int
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    mask = dilate_time_mask(saturation_mask, radius_samples)
    channel_fill = np.median(result, axis=0)
    rows, columns = np.nonzero(mask)
    result[rows, columns] = channel_fill[columns]
    return result


def interpolate_191(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply the saved channel-191 kernel to an already conditioned array."""
    result = np.asarray(values, dtype=np.float32).copy()
    good = np.delete(np.arange(N_CHANNELS), 191)
    if weights.shape != (N_CHANNELS - 1, 1):
        raise ValueError(f"Unexpected interpolation weights: {weights.shape}")
    result[:, 191] = result[:, good] @ weights[:, 0]
    return result


def materialize_int16(values: np.ndarray) -> np.ndarray:
    info = np.iinfo(np.int16)
    return np.clip(np.rint(values), info.min, info.max).astype(np.int16).astype(np.float32)


def load_raw_recordings():
    from probeinterface import read_probeinterface
    from spikeinterface.core import BinaryRecordingExtractor
    from spikeinterface.extractors.neuropixels_utils import get_neuropixels_sample_shifts
    from spikeinterface.preprocessing import astype, phase_shift

    graph = json.loads(SOURCE_PROVENANCE.read_text())
    blank_graph = graph["kwargs"]["recording"]
    phase_graph = blank_graph["kwargs"]["recording"]
    raw_graph = phase_graph["kwargs"]["recording"]
    _, fs = load_reference_settings()
    raw = BinaryRecordingExtractor(
        file_paths=[RAW_BINARY],
        sampling_frequency=fs,
        dtype=DTYPE,
        num_channels=N_SAVED_CHANNELS,
        channel_ids=np.arange(N_SAVED_CHANNELS),
        is_filtered=False,
    ).channel_slice(channel_ids=np.arange(N_CHANNELS))
    raw = raw.set_probegroup(read_probeinterface(SOURCE_PROBE))
    for name, values in raw_graph.get("properties", {}).items():
        if name not in {"location", "group"}:
            raw.set_property(name, np.asarray(values))
    raw.set_property(
        "inter_sample_shift", get_neuropixels_sample_shifts(N_CHANNELS, 12, 13)
    )
    shifted_int = phase_shift(raw)
    shifted_float = phase_shift(astype(raw, "float32"), dtype="float32")
    weights = np.asarray(graph["kwargs"]["weights"], dtype=np.float32)
    gain = float(np.unique(raw.get_property("gain_to_uV")).item())
    return raw, shifted_int, shifted_float, fs, gain, weights


def stage_arrays(
    raw_values: np.ndarray,
    phase_int: np.ndarray,
    phase_float: np.ndarray,
    fs: float,
    threshold_counts: float,
    weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    saturation_mask = np.abs(raw_values) >= threshold_counts
    phase_car = ks_center_car(phase_float)
    phase_clip = clip_saturation(phase_float, threshold_counts)
    phase_clip_car = ks_center_car(phase_clip)
    phase_car_hp = ks_highpass(phase_car, fs)
    phase_clip_car_hp = ks_highpass(phase_clip_car, fs)
    postblank = apply_postfilter_blank(
        phase_car_hp, saturation_mask, int(round(1e-3 * fs))
    )
    phase_interp_float = interpolate_191(phase_float, weights)
    phase_interp_float_car_hp = ks_highpass(ks_center_car(phase_interp_float), fs)
    current_source = materialize_int16(interpolate_191(phase_clip, weights))
    current_source_car_hp = ks_highpass(ks_center_car(current_source), fs)
    stages = {
        "raw": np.asarray(raw_values, dtype=np.float32),
        "phase_int16": np.asarray(phase_int, dtype=np.float32),
        "phase_float32": np.asarray(phase_float, dtype=np.float32),
        "phase_car": phase_car,
        "phase_clip_car": phase_clip_car,
        "phase_car_highpass": phase_car_hp,
        "phase_clip_car_highpass": phase_clip_car_hp,
        "phase_car_highpass_postblank_1ms": postblank,
        "phase_car_highpass_interp191": interpolate_191(phase_car_hp, weights),
        "phase_interp191_float_car_highpass": phase_interp_float_car_hp,
        "current_source_int16_car_highpass": current_source_car_hp,
    }
    return stages, saturation_mask


def event_counts(
    values: np.ndarray,
    positions: np.ndarray,
    fs: float,
    thresholds_sigma: tuple[float, ...] = (6.0, 8.0),
) -> dict[str, float]:
    sigma = robust_sigma(values, axis=0)
    standardized = (values / sigma[None, :]).T.astype(np.float32, copy=False)
    neighbors = spatial_neighbors(positions, 100.0)
    temporal_radius = int(round(0.5e-3 * fs))
    result: dict[str, float] = {}
    for negative in (True, False):
        sign = "negative" if negative else "positive"
        scores = -standardized if negative else standardized
        time_max = maximum_filter1d(
            scores,
            size=2 * temporal_radius + 1,
            axis=1,
            mode="constant",
            cval=-np.inf,
        )
        keep_scores = []
        for channel, nearby in enumerate(neighbors):
            neighborhood_max = np.max(time_max[nearby], axis=0)
            center = scores[channel, 1:-1]
            temporal_peak = (center > scores[channel, :-2]) & (center >= scores[channel, 2:])
            keep_scores.append(center[temporal_peak & (center >= neighborhood_max[1:-1])])
        kept = np.concatenate(keep_scores) if keep_scores else np.empty(0)
        for threshold_sigma in thresholds_sigma:
            result[f"{sign}_{threshold_sigma:g}sigma_events_per_s"] = float(
                np.sum(kept >= threshold_sigma) / (values.shape[0] / fs)
            )
    return result


def batch_metrics(
    values: np.ndarray,
    saturation_mask: np.ndarray,
    positions: np.ndarray,
    fs: float,
) -> dict[str, float]:
    sigma = robust_sigma(values, axis=0)
    sampled = values[::30]
    correlation = np.corrcoef(sampled.T)
    offdiag = correlation[np.triu_indices(values.shape[1], 1)]
    original = dilate_time_mask(saturation_mask, 1)
    boundary = dilate_time_mask(saturation_mask, int(round(3e-3 * fs))) & ~original
    boundary_z = np.abs(values / sigma[None, :])
    metrics = {
        "rms": float(np.sqrt(np.mean(np.square(values, dtype=np.float64)))),
        "median_mad_sigma": float(np.median(sigma)),
        "exact_zero_fraction": float(np.mean(values == 0)),
        "median_abs_channel_correlation": float(np.nanmedian(np.abs(offdiag))),
        "saturated_sample_fraction": float(np.mean(saturation_mask)),
        "boundary_fraction_over_6sigma": float(np.mean(boundary_z[boundary] >= 6.0))
        if boundary.any()
        else 0.0,
    }
    metrics.update(event_counts(values, positions, fs))
    return metrics


def covariance_metrics(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    covariance = (x.T @ x) / max(1, len(x))
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > np.finfo(float).eps]
    return {
        "covariance_condition": float(positive[-1] / positive[0])
        if positive.size
        else np.inf,
        "covariance_effective_rank": float(
            np.square(eigenvalues.sum()) / np.square(eigenvalues).sum()
        )
        if np.any(eigenvalues)
        else 0.0,
        "covariance_max_eigen_fraction": float(eigenvalues[-1] / eigenvalues.sum())
        if eigenvalues.sum()
        else np.nan,
    }


def kilosort_whitening_metrics(
    fit_values: np.ndarray, apply_values: np.ndarray, positions: np.ndarray
) -> dict[str, float]:
    """Fit Kilosort's regularized local ZCA matrix and audit its output."""
    fit = np.asarray(fit_values, dtype=np.float32)
    apply = np.asarray(apply_values, dtype=np.float32)
    covariance = (fit.T @ fit) / max(1, len(fit))
    matrix = np.zeros_like(covariance)
    # This is Kilosort4's whitening_local/whitening_from_covariance algorithm:
    # local 32-channel ZCA, with its fixed 1e-6 singular-value regularizer.
    for channel in range(covariance.shape[0]):
        distance = np.sum(
            np.square(positions - positions[channel], dtype=np.float64), axis=1
        )
        nearby = np.argsort(distance)[:32]
        left, singular, _ = np.linalg.svd(covariance[np.ix_(nearby, nearby)])
        local = (left / np.sqrt(singular + 1e-6)) @ left.T
        matrix[channel, nearby] = local[0]
    whitened = apply @ matrix.T
    covariance_after = covariance_metrics(whitened)
    correlation = np.corrcoef(whitened.T)
    offdiag = correlation[np.triu_indices(whitened.shape[1], 1)]
    sigma = robust_sigma(whitened, axis=0)
    return {
        "whitening_matrix_max_abs": float(np.max(np.abs(matrix))),
        "whitening_matrix_max_row_norm": float(
            np.max(np.sqrt(np.sum(np.square(matrix, dtype=np.float64), axis=1)))
        ),
        "whitened_rms": float(
            np.sqrt(np.mean(np.square(whitened, dtype=np.float64)))
        ),
        "whitened_median_mad_sigma": float(np.median(sigma)),
        "whitened_median_abs_channel_correlation": float(
            np.nanmedian(np.abs(offdiag))
        ),
        "whitened_covariance_condition": covariance_after["covariance_condition"],
        "whitened_covariance_effective_rank": covariance_after[
            "covariance_effective_rank"
        ],
        "whitened_covariance_max_eigen_fraction": covariance_after[
            "covariance_max_eigen_fraction"
        ],
    }


def choose_batch_starts(window: AuditWindow, fs: float, duration_s: float, n: int) -> np.ndarray:
    first = int(round(window.start_s * fs))
    last = int(round((window.start_s + window.duration_s - duration_s) * fs))
    return np.unique(np.linspace(first, last, n, dtype=np.int64))


def saturation_enriched_starts(path: Path, n: int) -> list[tuple[str, float]]:
    """Select complementary broad and focal high-voltage batches from prior audit indices."""
    if n <= 0:
        return []
    frame = pd.read_csv(path)
    frame = frame[frame["dataset"].astype(str).str.contains("Luke imec1")]
    grouped = (
        frame.groupby(["dataset", "window_kind", "batch_index", "batch_start_s"])[
            "fraction_abs_raw_over_500uv"
        ]
        .agg(["mean", "max"])
        .reset_index()
    )
    candidates: list[tuple[str, float]] = []
    for label, column in (("saturation_broad", "mean"), ("saturation_focal", "max")):
        for start in grouped.sort_values(column, ascending=False)["batch_start_s"]:
            value = float(start)
            if all(abs(value - prior) > 1.0 for _, prior in candidates):
                candidates.append((label, value))
            if sum(name == label for name, _ in candidates) >= int(np.ceil(n / 2)):
                break
    return candidates[:n]


def build_plan(args: argparse.Namespace, fs: float) -> dict:
    n = 1 if args.smoke else args.batches_per_window
    saturation = [] if args.smoke else saturation_enriched_starts(
        args.saturation_index, args.saturation_batches
    )
    duration_s = min(args.batch_duration_s, 0.25) if args.smoke else args.batch_duration_s
    return {
        "motion_enabled": False,
        "raw_binary": str(RAW_BINARY),
        "sampling_frequency_hz": fs,
        "windows": [asdict(window) for window in WINDOWS],
        "batch_duration_s": duration_s,
        "batches_per_window": n,
        "saturation_enriched_starts": [
            {"kind": kind, "start_s": start} for kind, start in saturation
        ],
        "stages": list(CORE_STAGES),
        "outputs": [
            "batch_metrics.csv",
            "stage_summary.csv",
            "reviewed_event_metrics.csv",
            "covariance_metrics.csv",
            "whitening_metrics.csv",
            "manifest.json",
        ],
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [column for column in frame.columns if column not in {"window", "stage", "batch"}]
    return (
        frame.groupby(["window", "stage"], as_index=False)[numeric]
        .median(numeric_only=True)
        .sort_values(["window", "stage"])
    )


def review_channel_indices(
    event: pd.Series,
    channel_depths_um: np.ndarray,
    radius_um: float = 150.0,
    excluded_channels: tuple[int, ...] = (191,),
) -> np.ndarray:
    """Return a fixed, review-centered channel neighborhood for stage comparisons."""
    reference_depth = float(event["peak_depth_um"])
    indices = np.flatnonzero(np.abs(channel_depths_um - reference_depth) <= radius_um)
    indices = indices[~np.isin(indices, excluded_channels)]
    if not len(indices):
        raise ValueError(f"No usable channels near reviewed depth {reference_depth:g} um")
    return indices


def extract_reviewed_event_metrics(
    args: argparse.Namespace,
    raw,
    phase_int,
    phase_float,
    fs: float,
    threshold_counts: float,
    weights: np.ndarray,
    positions: np.ndarray,
    max_events: int,
) -> pd.DataFrame:
    """Compare stages inside the same reviewed depth neighborhood for every event."""
    events = pd.read_csv(args.review_events)
    event_rows: list[dict] = []
    half = int(round(10e-3 * fs))
    depths = positions[:, 1]
    for window in WINDOWS:
        lo = int(round(window.start_s * fs))
        hi = int(round((window.start_s + window.duration_s) * fs))
        selected = events[events["sample_index"].between(lo + half, hi - half - 1)]
        if len(selected) > max_events:
            selected_indices = np.unique(
                np.linspace(0, len(selected) - 1, max_events, dtype=int)
            )
            selected = selected.iloc[selected_indices]
        for _, event in selected.iterrows():
            center = int(event["sample_index"])
            channel_indices = review_channel_indices(event, depths)
            raw_values = raw.get_traces(
                start_frame=center - half, end_frame=center + half + 1
            )
            phase_i = phase_int.get_traces(
                start_frame=center - half, end_frame=center + half + 1
            )
            phase_f = phase_float.get_traces(
                start_frame=center - half, end_frame=center + half + 1
            )
            stages, _ = stage_arrays(
                raw_values, phase_i, phase_f, fs, threshold_counts, weights
            )
            for stage, values in stages.items():
                metrics, _ = event_metrics(
                    values[:, channel_indices], depths[channel_indices], fs
                )
                local_peak = int(metrics["peak_channel"])
                metrics["peak_channel"] = int(channel_indices[local_peak])
                event_rows.append(
                    {
                        "window": window.name,
                        "review_id": event.get("review_id", event.name),
                        "review_label": event.get("review_label", "unknown"),
                        "status": event.get("status", "unknown"),
                        "reference_peak_channel": int(event["peak_channel"]),
                        "reference_depth_um": float(event["peak_depth_um"]),
                        "neighborhood_radius_um": 150.0,
                        "channel_191_excluded": True,
                        "n_neighborhood_channels": len(channel_indices),
                        "stage": stage,
                        **metrics,
                    }
                )
    return pd.DataFrame(event_rows)


def run(args: argparse.Namespace) -> None:
    raw, phase_int, phase_float, fs, gain, weights = load_raw_recordings()
    plan = build_plan(args, fs)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_batches = 1 if args.smoke else args.batches_per_window
    saturation = [] if args.smoke else saturation_enriched_starts(
        args.saturation_index, args.saturation_batches
    )
    max_events = 2 if args.smoke else args.max_events_per_window
    batch_duration_s = min(args.batch_duration_s, 0.25) if args.smoke else args.batch_duration_s
    n_samples = int(round(batch_duration_s * fs))
    positions = raw.get_channel_locations().astype(float)
    threshold_counts = SATURATION_UV / gain
    if args.events_only:
        event_frame = extract_reviewed_event_metrics(
            args,
            raw,
            phase_int,
            phase_float,
            fs,
            threshold_counts,
            weights,
            positions,
            max_events,
        )
        event_frame.to_csv(args.output_dir / "reviewed_event_metrics.csv", index=False)
        print(f"Wrote {len(event_frame)} fixed-neighborhood reviewed-event rows")
        return
    rows: list[dict] = []
    covariance_rows: list[dict] = []
    whitening_rows: list[dict] = []
    covariance_data: dict[tuple[str, str], list[np.ndarray]] = {}

    for window in WINDOWS:
        for batch, start in enumerate(
            choose_batch_starts(window, fs, batch_duration_s, n_batches)
        ):
            stop = min(start + n_samples, SOURCE_FRAMES)
            raw_values = raw.get_traces(start_frame=int(start), end_frame=int(stop))
            phase_i = phase_int.get_traces(start_frame=int(start), end_frame=int(stop))
            phase_f = phase_float.get_traces(start_frame=int(start), end_frame=int(stop))
            stages, saturation_mask = stage_arrays(
                raw_values, phase_i, phase_f, fs, threshold_counts, weights
            )
            for stage, values in stages.items():
                rows.append(
                    {
                        "window": window.name,
                        "batch": batch,
                        "start_s": start / fs,
                        "stage": stage,
                        **batch_metrics(values, saturation_mask, positions, fs),
                    }
                )
                if stage in {
                    "phase_car_highpass",
                    "phase_clip_car_highpass",
                    "phase_car_highpass_postblank_1ms",
                    "phase_car_highpass_interp191",
                    "phase_interp191_float_car_highpass",
                    "current_source_int16_car_highpass",
                }:
                    covariance_data.setdefault((window.name, stage), []).append(values[::30])
                    if stage == "phase_car_highpass":
                        clean = ~dilate_time_mask(
                            saturation_mask, int(round(1e-3 * fs))
                        ).any(axis=1)
                        clean_values = values[clean][::30]
                        if len(clean_values):
                            covariance_data.setdefault(
                                (window.name, "phase_car_highpass_artifact_excluded"), []
                            ).append(clean_values)

    for batch, (kind, start_s) in enumerate(saturation):
        start = int(round(start_s * fs))
        stop = min(start + n_samples, SOURCE_FRAMES)
        raw_values = raw.get_traces(start_frame=start, end_frame=stop)
        phase_i = phase_int.get_traces(start_frame=start, end_frame=stop)
        phase_f = phase_float.get_traces(start_frame=start, end_frame=stop)
        stages, saturation_mask = stage_arrays(
            raw_values, phase_i, phase_f, fs, threshold_counts, weights
        )
        for stage, values in stages.items():
            rows.append(
                {
                    "window": kind,
                    "batch": batch,
                    "start_s": start / fs,
                    "stage": stage,
                    **batch_metrics(values, saturation_mask, positions, fs),
                }
            )
            if stage in {
                "phase_car_highpass",
                "phase_clip_car_highpass",
                "phase_car_highpass_postblank_1ms",
                "phase_car_highpass_interp191",
                "phase_interp191_float_car_highpass",
                "current_source_int16_car_highpass",
            }:
                covariance_data.setdefault((kind, stage), []).append(values[::30])
                if stage == "phase_car_highpass":
                    clean = ~dilate_time_mask(
                        saturation_mask, int(round(1e-3 * fs))
                    ).any(axis=1)
                    clean_values = values[clean][::30]
                    if len(clean_values):
                        covariance_data.setdefault(
                            (kind, "phase_car_highpass_artifact_excluded"), []
                        ).append(clean_values)

    batch_frame = pd.DataFrame(rows)
    batch_frame.to_csv(args.output_dir / "batch_metrics.csv", index=False)
    summary = summarize(batch_frame)
    summary.to_csv(args.output_dir / "stage_summary.csv", index=False)

    for (window, stage), blocks in covariance_data.items():
        fit_values = np.concatenate(blocks)
        covariance_rows.append(
            {"window": window, "stage": stage, **covariance_metrics(fit_values)}
        )
        apply_stage = (
            "phase_car_highpass"
            if stage == "phase_car_highpass_artifact_excluded"
            else stage
        )
        apply_values = np.concatenate(covariance_data[(window, apply_stage)])
        whitening_rows.append(
            {
                "window": window,
                "fit_stage": stage,
                "apply_stage": apply_stage,
                **kilosort_whitening_metrics(fit_values, apply_values, positions),
            }
        )
    pd.DataFrame(covariance_rows).to_csv(
        args.output_dir / "covariance_metrics.csv", index=False
    )
    pd.DataFrame(whitening_rows).to_csv(
        args.output_dir / "whitening_metrics.csv", index=False
    )

    event_frame = extract_reviewed_event_metrics(
        args,
        raw,
        phase_int,
        phase_float,
        fs,
        threshold_counts,
        weights,
        positions,
        max_events,
    )
    event_frame.to_csv(args.output_dir / "reviewed_event_metrics.csv", index=False)
    manifest = {
        **plan,
        "gain_uv_per_count": gain,
        "saturation_uv": SATURATION_UV,
        "saturation_counts": threshold_counts,
        "source_frames": SOURCE_FRAMES,
        "source_channels": N_CHANNELS,
        "review_events": str(args.review_events),
        "n_reviewed_event_stage_rows": len(event_frame),
        "whitening_status": (
            "Kilosort local regularized ZCA matrices are fit diagnostically for every "
            "terminal upstream branch; no branch is selected automatically"
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
