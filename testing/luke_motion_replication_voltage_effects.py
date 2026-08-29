"""Quantify voltage attenuation from 0.25x rigid interpolation for all reviewed events."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import event_local_samples, events_in_window, load_reference_settings
from testing.luke_motion_replication_sort import DEFAULT_REVIEW, WINDOW, recording_path, shared_motion_path

OUTPUT_ROOT = REPO_ROOT / "testing/outputs/luke_motion_candidate_results"


def main() -> None:
    import spikeinterface.core as sc

    events = events_in_window(pd.read_csv(DEFAULT_REVIEW), WINDOW).copy()
    _, fs = load_reference_settings()
    samples = event_local_samples(events.sample_index.to_numpy(), WINDOW, fs)
    baseline = sc.load(recording_path("no_external_correction"))
    corrected = sc.load(recording_path("rigid_gain_025"))
    locations = baseline.get_channel_locations()
    half = int(round(1e-3 * fs))
    motion_path = shared_motion_path()
    rigid = 0.25 * np.load(motion_path / "motion.npy").mean(axis=1)
    time_bins = np.load(motion_path / "time_bins.npy")
    raw_t_start = float(baseline.get_times()[0] - WINDOW.start_s)
    displacements = np.interp(events.time_seconds.to_numpy(float) + raw_t_start, time_bins, rigid)

    rows = []
    for (_, event), center, displacement in zip(events.iterrows(), samples, displacements):
        local = np.flatnonzero(np.abs(locations[:, 1] - event.peak_depth_um) <= 120)
        traces = []
        for recording in (baseline, corrected):
            trace = recording.get_traces(
                start_frame=int(center - half),
                end_frame=int(center + half + 1),
                channel_ids=recording.channel_ids[local],
            ).astype(float)
            traces.append(trace)
        before, after = traces
        before_peak = float(np.max(np.abs(before)))
        after_peak = float(np.max(np.abs(after)))
        rows.append(
            {
                "review_id": event.review_id,
                "review_label": event.review_label,
                "status": event.status,
                "automatic_neural_like": bool(event.automatic_neural_like),
                "rigid_displacement_um": float(displacement),
                "peak_ratio": after_peak / before_peak if before_peak else np.nan,
                "rms_ratio": float(np.sqrt(np.mean(after**2)) / np.sqrt(np.mean(before**2))),
                "trace_correlation": float(np.corrcoef(before.ravel(), after.ravel())[0, 1]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_ROOT / "rigid_gain_replication_voltage_effects.csv", index=False)

    populations = {
        "All reviewed (n=126)": np.ones(len(frame), dtype=bool),
        "Visual neural misses (n=35)": (frame.review_label == "neural") & (frame.status == "unmatched"),
        "Automated neural-like misses (n=12)": frame.automatic_neural_like & (frame.status == "unmatched"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    colors = ["#7f8c8d", "#4c78a8", "#72b7b2"]
    for (label, mask), color in zip(populations.items(), colors):
        values = frame.loc[mask, "peak_ratio"] * 100
        axes[0].hist(values, bins=np.linspace(65, 110, 24), histtype="step", linewidth=2, label=f"{label}; median {np.median(values):.1f}%", color=color)
        axes[1].scatter(np.abs(frame.loc[mask, "rigid_displacement_um"]), values, s=18, alpha=0.6, color=color, label=label)
    axes[0].axvline(100, color="black", linewidth=1, linestyle="--")
    axes[0].set(title="Peak waveform magnitude after interpolation", xlabel="Corrected / uncorrected peak (%)", ylabel="Events")
    axes[0].legend(frameon=False, fontsize=8)
    rho, pvalue = spearmanr(np.abs(frame.rigid_displacement_um), frame.peak_ratio, nan_policy="omit")
    axes[1].axhline(100, color="black", linewidth=1, linestyle="--")
    axes[1].set(title=f"Attenuation grows with applied shift (Spearman r={rho:.2f})", xlabel="Applied rigid displacement magnitude (µm)", ylabel="Peak retained (%)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("0.25× rigid resampling systematically attenuates reviewed raw events", fontsize=13, fontweight="bold")
    fig.savefig(OUTPUT_ROOT / "rigid_gain_replication_voltage_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(frame.groupby([frame.review_label.eq("neural") & frame.status.eq("unmatched"), frame.automatic_neural_like])[["peak_ratio", "rms_ratio", "trace_correlation"]].median())
    print(f"Spearman(abs displacement, peak ratio): r={rho:.4f}, p={pvalue:.4g}")


if __name__ == "__main__":
    main()
