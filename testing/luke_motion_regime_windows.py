"""Select estimation-only Luke windows from input and estimator signatures.

This script never applies a motion field and never reads sorter labels.  It
reduces the full-session cached peak population to exact 10-second support and
artifact features, aligns saved DREDGE and decentralized estimates to the same
frame-relative clock, and scores fixed 120-second windows for four regimes:

* quiet / stable support;
* rapid supported estimate change without an input anomaly;
* sustained anomalous input with little estimated motion;
* peak-support dropout or instability with little estimated motion;
* anomalous input coincident with estimated motion.

The selected windows are discovery material for a controlled estimator
factorial.  They are not physical ground truth labels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LUKE_ROOT = Path("/mnt/NPX/Luke/20250804")
OUTPUT = Path("testing/outputs/luke_motion_regime_windows")
FS = 30_000.0
PROBES = ("imec0", "imec1")
BIN_S = 10.0
WINDOW_S = 120.0
STRIDE_S = 30.0
DEPTH_EDGES_UM = np.linspace(0.0, 3840.0, 17)
SYNC_MULTIPLICITY = 8


@dataclass(frozen=True)
class Window:
    regime: str
    start_s: float
    duration_s: float
    score: float
    selection_basis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--chunk-peaks", type=int, default=2_000_000)
    return parser.parse_args()


def motion_root(probe: str) -> Path:
    return LUKE_ROOT / f"dredge_pipeline_results_Luke0804_V2V1_g0_{probe}" / "motion"


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - median))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return np.zeros_like(values)
    return (values - median) / scale


def relative_times(native_times: np.ndarray) -> np.ndarray:
    """Convert saved absolute extractor time bins to frame-relative seconds."""
    native_times = np.asarray(native_times, dtype=float)
    dt = float(np.median(np.diff(native_times)))
    acquisition_start_s = float(native_times[0] - dt / 2)
    return native_times - acquisition_start_s


def rigid_trace(probe: str, method_dir: str, centers_s: np.ndarray) -> np.ndarray:
    root = motion_root(probe) / method_dir
    field = np.load(root / "motion.npy", mmap_mode="r")
    times = relative_times(np.load(root / "time_bins.npy"))
    rigid = np.nanmedian(np.asarray(field, dtype=float), axis=1)
    return np.interp(centers_s, times, rigid, left=np.nan, right=np.nan)


def peak_bin_features(probe: str, chunk_peaks: int) -> pd.DataFrame:
    root = motion_root(probe)
    peaks = np.load(root / "peaks.npy", mmap_mode="r")
    locations = np.load(root / "peak_locations.npy", mmap_mode="r")
    duration_s = float(peaks["sample_index"][-1]) / FS
    n_bins = int(np.ceil(duration_s / BIN_S))
    n_channels = int(np.max(peaks["channel_index"])) + 1
    counts = np.zeros(n_bins, dtype=np.int64)
    amplitude_sum = np.zeros(n_bins, dtype=np.float64)
    amplitude_sq_sum = np.zeros(n_bins, dtype=np.float64)
    high_amplitude = np.zeros(n_bins, dtype=np.int64)
    synchronous_peaks = np.zeros(n_bins, dtype=np.int64)
    channel_counts = np.zeros((n_bins, n_channels), dtype=np.int64)
    depth_counts = np.zeros((n_bins, len(DEPTH_EDGES_UM) - 1), dtype=np.int64)

    for start in range(0, len(peaks), chunk_peaks):
        stop = min(start + chunk_peaks, len(peaks))
        peak = peaks[start:stop]
        bin_index = np.floor(peak["sample_index"] / (FS * BIN_S)).astype(np.int64)
        amplitude = np.abs(np.asarray(peak["amplitude"], dtype=float))
        valid = (bin_index >= 0) & (bin_index < n_bins) & np.isfinite(amplitude)
        bins = bin_index[valid]
        amps = amplitude[valid]
        counts += np.bincount(bins, minlength=n_bins)
        amplitude_sum += np.bincount(bins, weights=amps, minlength=n_bins)
        amplitude_sq_sum += np.bincount(bins, weights=amps**2, minlength=n_bins)
        high_amplitude += np.bincount(bins, weights=amps >= 50.0, minlength=n_bins).astype(np.int64)

        channels = np.asarray(peak["channel_index"][valid], dtype=np.int64)
        flat_channel = bins * n_channels + channels
        channel_counts += np.bincount(
            flat_channel, minlength=n_bins * n_channels
        ).reshape(n_bins, n_channels)

        depth = np.asarray(locations["y"][start:stop], dtype=float)[valid]
        depth_index = np.searchsorted(DEPTH_EDGES_UM, depth, side="right") - 1
        good_depth = (depth_index >= 0) & (depth_index < depth_counts.shape[1])
        flat_depth = bins[good_depth] * depth_counts.shape[1] + depth_index[good_depth]
        depth_counts += np.bincount(
            flat_depth, minlength=n_bins * depth_counts.shape[1]
        ).reshape(depth_counts.shape)

        samples = np.asarray(peak["sample_index"], dtype=np.int64)
        unique_sample, multiplicity = np.unique(samples, return_counts=True)
        sync = multiplicity >= SYNC_MULTIPLICITY
        if np.any(sync):
            sync_bins = np.floor(unique_sample[sync] / (FS * BIN_S)).astype(np.int64)
            good_sync = (sync_bins >= 0) & (sync_bins < n_bins)
            synchronous_peaks += np.bincount(
                sync_bins[good_sync],
                weights=multiplicity[sync][good_sync],
                minlength=n_bins,
            ).astype(np.int64)

    safe_counts = np.maximum(counts, 1)
    depth_probability = depth_counts / safe_counts[:, None]
    depth_entropy = -np.sum(
        np.where(depth_probability > 0, depth_probability * np.log(depth_probability), 0.0),
        axis=1,
    ) / np.log(depth_counts.shape[1])
    centers = (np.arange(n_bins) + 0.5) * BIN_S
    result = pd.DataFrame(
        {
            "probe": probe,
            "bin_start_s": centers - BIN_S / 2,
            "bin_center_s": centers,
            "peak_count": counts,
            "peak_rate_hz": counts / BIN_S,
            "mean_abs_amplitude": amplitude_sum / safe_counts,
            "rms_amplitude": np.sqrt(amplitude_sq_sum / safe_counts),
            "fraction_abs_amplitude_ge_50": high_amplitude / safe_counts,
            "max_channel_fraction": channel_counts.max(axis=1) / safe_counts,
            "channel_191_fraction": (
                channel_counts[:, 191] / safe_counts if n_channels > 191 else np.nan
            ),
            "synchronous_peak_fraction": synchronous_peaks / safe_counts,
            "depth_entropy": depth_entropy,
            "occupied_depth_bands": np.sum(depth_probability >= 0.01, axis=1),
        }
    )
    result["dredge_rigid_um"] = rigid_trace(probe, "dredge-motion", centers)
    result["decentralized_rigid_um"] = rigid_trace(
        probe, "decentralized-motion", centers
    )
    return result


def window_features(bin_features: pd.DataFrame) -> pd.DataFrame:
    n_window = int(round(WINDOW_S / BIN_S))
    stride = int(round(STRIDE_S / BIN_S))
    rows = []
    for probe, values in bin_features.groupby("probe", sort=True):
        values = values.sort_values("bin_start_s").reset_index(drop=True)
        for start in range(0, len(values) - n_window + 1, stride):
            part = values.iloc[start : start + n_window]
            dredge = part.dredge_rigid_um.to_numpy(float)
            decentralized = part.decentralized_rigid_um.to_numpy(float)
            valid = np.isfinite(dredge) & np.isfinite(decentralized)
            agreement = (
                float(np.corrcoef(dredge[valid], decentralized[valid])[0, 1])
                if valid.sum() >= 4
                and np.std(dredge[valid]) > 0
                and np.std(decentralized[valid]) > 0
                else np.nan
            )
            rate = part.peak_rate_hz.to_numpy(float)
            rows.append(
                {
                    "probe": probe,
                    "start_s": float(part.bin_start_s.iloc[0]),
                    "duration_s": WINDOW_S,
                    "median_peak_rate_hz": float(np.median(rate)),
                    "peak_rate_cv": float(np.std(rate) / max(np.mean(rate), 1e-12)),
                    "median_synchronous_peak_fraction": float(
                        np.median(part.synchronous_peak_fraction)
                    ),
                    "p95_synchronous_peak_fraction": float(
                        np.percentile(part.synchronous_peak_fraction, 95)
                    ),
                    "median_high_amplitude_fraction": float(
                        np.median(part.fraction_abs_amplitude_ge_50)
                    ),
                    "median_max_channel_fraction": float(
                        np.median(part.max_channel_fraction)
                    ),
                    "median_depth_entropy": float(np.median(part.depth_entropy)),
                    "dredge_excursion_um": float(np.ptp(dredge)),
                    "dredge_p95_step_um": float(np.percentile(np.abs(np.diff(dredge)), 95)),
                    "decentralized_excursion_um": float(np.ptp(decentralized)),
                    "dredge_decentralized_r": agreement,
                }
            )
    result = pd.DataFrame(rows)
    # Score within probe so different absolute peak yields do not choose a regime.
    scored = []
    for probe, values in result.groupby("probe", sort=True):
        values = values.copy()
        noise = (
            robust_z(values.p95_synchronous_peak_fraction)
            + robust_z(values.median_high_amplitude_fraction)
            + robust_z(values.median_max_channel_fraction)
            + robust_z(values.peak_rate_cv)
            - robust_z(values.median_depth_entropy)
        ) / 5.0
        motion = (
            robust_z(values.dredge_excursion_um)
            + robust_z(values.dredge_p95_step_um)
            + robust_z(values.decentralized_excursion_um)
            + robust_z(values.dredge_decentralized_r.fillna(-1.0))
        ) / 4.0
        values["input_anomaly_score"] = noise
        values["persistent_noise_score"] = (
            robust_z(values.median_synchronous_peak_fraction)
            + robust_z(values.p95_synchronous_peak_fraction)
            + robust_z(values.median_high_amplitude_fraction)
            + robust_z(values.median_max_channel_fraction)
            - robust_z(values.median_depth_entropy)
        ) / 5.0
        values["support_instability_score"] = (
            robust_z(values.peak_rate_cv) - robust_z(values.median_peak_rate_hz)
        ) / 2.0
        values["supported_motion_score"] = motion
        scored.append(values)
    return pd.concat(scored, ignore_index=True)


def choose_windows(features: pd.DataFrame) -> tuple[pd.DataFrame, list[Window]]:
    # Select on both simultaneous probes.  Max anomaly protects against a
    # one-probe artifact; minimum motion support requires the rapid class to be
    # expressed on both probes instead of merely leaking through one input.
    eligible = features[np.isfinite(features.dredge_decentralized_r)].copy()
    values = (
        eligible.groupby(["start_s", "duration_s"], as_index=False)
        .agg(
            max_input_anomaly=("input_anomaly_score", "max"),
            max_persistent_noise=("persistent_noise_score", "max"),
            max_support_instability=("support_instability_score", "max"),
            min_supported_motion=("supported_motion_score", "min"),
            mean_supported_motion=("supported_motion_score", "mean"),
        )
    )
    values["quiet_score"] = (
        -values.max_input_anomaly
        - values.max_support_instability
        - values.mean_supported_motion
    )
    values["rapid_score"] = (
        values.min_supported_motion
        - values.max_persistent_noise
        - values.max_support_instability
    )
    values["noise_score"] = (
        values.max_persistent_noise
        - np.abs(values.max_support_instability)
        - values.mean_supported_motion
    )
    values["dropout_score"] = (
        values.max_support_instability
        - values.max_persistent_noise
        - values.mean_supported_motion
    )
    values["mixed_score"] = values.max_persistent_noise + values.min_supported_motion
    definitions = {
        "quiet": ("quiet_score", "low input anomaly and low supported motion"),
        "rapid_motion": (
            "rapid_score",
            "high DREDGE/decentralized-supported change with relatively normal input",
        ),
        "sustained_noise": (
            "noise_score",
            "persistent synchrony/amplitude/channel anomaly with relatively little supported motion",
        ),
        "support_dropout": (
            "dropout_score",
            "unstable or depleted peak support with relatively little supported motion",
        ),
        "noise_plus_motion": (
            "mixed_score",
            "input anomaly coincident with a supported estimate change",
        ),
    }
    selected: list[Window] = []
    used: list[tuple[float, float]] = []
    for regime, (column, basis) in definitions.items():
        candidates = values.sort_values(column, ascending=False)
        row = None
        for candidate in candidates.itertuples(index=False):
            interval = (candidate.start_s, candidate.start_s + candidate.duration_s)
            overlap = any(max(interval[0], old[0]) < min(interval[1], old[1]) for old in used)
            if not overlap:
                row = candidate
                break
        if row is None:
            row = next(candidates.itertuples(index=False))
        selected.append(
            Window(regime, float(row.start_s), float(row.duration_s), float(getattr(row, column)), basis)
        )
        used.append((row.start_s, row.start_s + row.duration_s))
    selected_frame = pd.DataFrame([asdict(value) for value in selected])
    selected_frame = selected_frame.merge(values, on=["start_s", "duration_s"], how="left", validate="one_to_one")
    return selected_frame, selected


def make_figure(features: pd.DataFrame, selected: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for probe, values in features.groupby("probe"):
        ax.scatter(
            values.input_anomaly_score,
            values.supported_motion_score,
            s=18,
            alpha=0.45,
            label=probe,
        )
    for row in selected.itertuples(index=False):
        ax.scatter(
            row.max_input_anomaly,
            row.mean_supported_motion,
            s=110,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            label=f"{row.regime}: {row.start_s:.0f} s",
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_xlabel("Input anomaly score (robust within-probe units)")
    ax.set_ylabel("Mean supported estimate-change score (robust within-probe units)")
    ax.set_title("Prespecified Luke windows span input and estimate regimes")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bins = pd.concat(
        [peak_bin_features(probe, args.chunk_peaks) for probe in PROBES],
        ignore_index=True,
    )
    windows = window_features(bins)
    selected, selections = choose_windows(windows)
    bins.to_csv(args.output_dir / "ten_second_input_features.csv", index=False)
    windows.to_csv(args.output_dir / "candidate_window_features.csv", index=False)
    selected.to_csv(args.output_dir / "selected_windows.csv", index=False)
    windows.merge(
        selected[["regime", "start_s", "duration_s"]],
        on=["start_s", "duration_s"],
        how="inner",
        validate="many_to_one",
    ).to_csv(args.output_dir / "selected_window_probe_features.csv", index=False)
    make_figure(windows, selected, args.output_dir / "selected_regimes.png")
    manifest = {
        "source": str(LUKE_ROOT),
        "probes": list(PROBES),
        "peak_bin_s": BIN_S,
        "window_s": WINDOW_S,
        "stride_s": STRIDE_S,
        "synchronous_peak_multiplicity": SYNC_MULTIPLICITY,
        "selected_windows": [asdict(value) for value in selections],
        "sorter_labels_accessed": False,
        "motion_applied": False,
        "interpretation": (
            "Discovery-window labels describe cached input and estimator signatures; "
            "they are not physical ground truth."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
