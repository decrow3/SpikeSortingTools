"""Audit geometry dependence and sigma sensitivity of motion interpolation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import event_local_samples, events_in_window, load_reference_settings
from testing.luke_motion_candidate_sort import field_displacement
from testing.luke_motion_replication_sort import DEFAULT_REVIEW, WINDOW, recording_path, shared_motion_path

OUTPUT_ROOT = REPO_ROOT / "testing/outputs/luke_motion_candidate_results"
YATES_ROOT = Path("/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16")


def kernel_metrics(locations: np.ndarray, shift_um: float, sigma_um: float) -> dict[str, float]:
    from spikeinterface.preprocessing import get_spatial_interpolation_kernel

    target = locations.copy()
    target[:, 1] += shift_um
    kernel = get_spatial_interpolation_kernel(
        locations,
        target,
        method="kriging",
        sigma_um=sigma_um,
        p=1,
        force_extrapolate=False,
        dtype="float64",
    )
    inside = ~np.isclose(kernel.sum(axis=0), 0)
    return {
        "median_peak_weight": float(np.median(np.max(np.abs(kernel[:, inside]), axis=0))),
        "median_l2_gain": float(np.median(np.sqrt(np.sum(kernel[:, inside] ** 2, axis=0)))),
        "zeroed_channel_fraction": float(np.mean(~inside)),
        "n_channels": int(len(locations)),
    }


def geometry_audit(luke_locations: np.ndarray) -> pd.DataFrame:
    yates_meta = json.loads((YATES_ROOT / "ephys_metadata.json").read_text())
    yates_locations = np.asarray(yates_meta["probe_geometry_um"], dtype=float)[:32]
    rows = []
    for probe, locations, sigma_um in (
        ("Luke imec1", luke_locations, 20.0),
        ("Yates Nandy64 shank", yates_locations, 20.0),
        ("Luke imec1", luke_locations, 10.0),
        ("Luke imec1", luke_locations, 5.0),
    ):
        for shift_um in np.linspace(0, 10, 21):
            rows.append(
                {
                    "probe": probe,
                    "sigma_um": sigma_um,
                    "shift_um": float(shift_um),
                    **kernel_metrics(locations, float(shift_um), sigma_um),
                }
            )
    return pd.DataFrame(rows)


def event_audit(baseline, locations: np.ndarray) -> pd.DataFrame:
    from spikeinterface.core.motion import Motion
    from spikeinterface.sortingcomponents.motion import interpolate_motion

    events = events_in_window(pd.read_csv(DEFAULT_REVIEW), WINDOW).copy()
    _, fs = load_reference_settings()
    samples = event_local_samples(events.sample_index.to_numpy(), WINDOW, fs)
    motion_path = shared_motion_path()
    displacement = field_displacement(np.load(motion_path / "motion.npy"), "rigid_gain_025")
    motion = Motion(
        displacement=displacement,
        temporal_bins_s=np.load(motion_path / "time_bins.npy"),
        spatial_bins_um=np.load(motion_path / "depth_bins.npy"),
    )
    variants = {"identity": baseline}
    for sigma_um in (5.0, 10.0, 20.0):
        variants[f"kriging_sigma_{sigma_um:g}"] = interpolate_motion(
            baseline.astype("float"),
            motion,
            border_mode="force_zeros",
            spatial_interpolation_method="kriging",
            sigma_um=sigma_um,
        ).astype("int16")

    half = int(round(1e-3 * fs))
    rows = []
    for (_, event), center in zip(events.iterrows(), samples):
        local = np.flatnonzero(np.abs(locations[:, 1] - event.peak_depth_um) <= 120)
        base_trace = baseline.get_traces(
            start_frame=int(center - half),
            end_frame=int(center + half + 1),
            channel_ids=baseline.channel_ids[local],
        ).astype(float)
        base_peak = float(np.max(np.abs(base_trace)))
        base_rms = float(np.sqrt(np.mean(base_trace**2)))
        for variant, recording in variants.items():
            trace = recording.get_traces(
                start_frame=int(center - half),
                end_frame=int(center + half + 1),
                channel_ids=recording.channel_ids[local],
            ).astype(float)
            rows.append(
                {
                    "review_id": event.review_id,
                    "variant": variant,
                    "sigma_um": np.nan if variant == "identity" else float(variant.rsplit("_", 1)[1]),
                    "review_label": event.review_label,
                    "status": event.status,
                    "automatic_neural_like": bool(event.automatic_neural_like),
                    "peak_ratio": float(np.max(np.abs(trace)) / base_peak) if base_peak else np.nan,
                    "rms_ratio": float(np.sqrt(np.mean(trace**2)) / base_rms) if base_rms else np.nan,
                    "trace_correlation": float(np.corrcoef(base_trace.ravel(), trace.ravel())[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def render(geometry: pd.DataFrame, events: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    styles = {
        ("Luke imec1", 20.0): ("#32688e", "-", "Luke, sigma 20 µm"),
        ("Yates Nandy64 shank", 20.0): ("#d17a22", "-", "Yates shank, sigma 20 µm"),
        ("Luke imec1", 10.0): ("#32688e", "--", "Luke, sigma 10 µm"),
        ("Luke imec1", 5.0): ("#32688e", ":", "Luke, sigma 5 µm"),
    }
    for key, (color, linestyle, label) in styles.items():
        subset = geometry[(geometry.probe == key[0]) & (geometry.sigma_um == key[1])]
        axes[0].plot(
            subset.shift_um,
            100 * subset.median_peak_weight,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
    axes[0].set(
        title="Kriging impulse retention by probe geometry",
        xlabel="Rigid displacement (µm)",
        ylabel="Median peak weight retained (%)",
        ylim=(50, 101),
    )
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", color="0.9")

    populations = {
        "All reviewed (n=126)": np.ones(len(events), dtype=bool),
        "Visual neural misses (n=35)": (events.review_label == "neural") & (events.status == "unmatched"),
        "Automated neural-like misses (n=12)": events.automatic_neural_like & (events.status == "unmatched"),
    }
    variants = ["identity", "kriging_sigma_5", "kriging_sigma_10", "kriging_sigma_20"]
    labels = ["No resampling", "Sigma 5 µm", "Sigma 10 µm", "Sigma 20 µm"]
    colors = ["#6b7280", "#b7cde1", "#6d9bc3", "#32688e"]
    x = np.arange(len(variants))
    for offset, (population, mask) in zip((-0.22, 0, 0.22), populations.items()):
        medians = []
        for variant in variants:
            subset = events[(events.variant == variant)]
            population_mask = mask[events.variant == variant]
            medians.append(100 * subset.loc[population_mask, "peak_ratio"].median())
        axes[1].plot(x + offset, medians, marker="o", linewidth=1.8, label=population)
    axes[1].set(
        title="Luke reviewed-event peak retention",
        xlabel="Spatial interpolation configuration",
        ylabel="Median peak retained (%)",
        xticks=x,
        xticklabels=labels,
        ylim=(94, 101),
    )
    axes[1].grid(axis="y", color="0.9")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Motion interpolation transfer by geometry and kriging scale", fontsize=14, fontweight="bold")
    fig.savefig(OUTPUT_ROOT / "luke_yates_interpolation_kernel_audit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import spikeinterface.core as sc

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = sc.load(recording_path("no_external_correction"))
    locations = baseline.get_channel_locations()
    geometry = geometry_audit(locations)
    events = event_audit(baseline, locations)
    geometry.to_csv(OUTPUT_ROOT / "luke_yates_interpolation_kernel_metrics.csv", index=False)
    events.to_csv(OUTPUT_ROOT / "luke_interpolation_sigma_event_metrics.csv", index=False)
    render(geometry, events)
    summary = events.groupby("variant").agg(
        n_events=("review_id", "size"),
        median_peak_ratio=("peak_ratio", "median"),
        median_rms_ratio=("rms_ratio", "median"),
        median_trace_correlation=("trace_correlation", "median"),
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
