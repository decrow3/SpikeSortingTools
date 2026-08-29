"""Plot voltage-level effects for events that switch recovery after 0.25x rigid correction."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_claimmask_window_sweep import event_local_samples, load_reference_settings
from testing.luke_motion_replication_sort import DEFAULT_REVIEW, OUTPUT_ROOT, WINDOW, recording_path

FIGURE_ROOT = Path(__file__).resolve().parent / "outputs/luke_motion_candidate_results"


def main() -> None:
    import spikeinterface.core as sc

    paired = pd.read_csv(OUTPUT_ROOT / "paired_event_recovery.csv")
    visual = paired[(paired.review_label == "neural") & (paired.status == "unmatched")]
    pivot = visual.pivot(index="review_id", columns="condition", values="recovered")
    switched = pivot[pivot.nunique(axis=1) > 1].index.tolist()
    review = pd.read_csv(DEFAULT_REVIEW).set_index("review_id").loc[switched]
    _, fs = load_reference_settings()
    samples = event_local_samples(review.sample_index.to_numpy(), WINDOW, fs)
    recordings = {
        "No correction": sc.load(recording_path("no_external_correction")),
        "0.25× rigid": sc.load(recording_path("rigid_gain_025")),
    }
    locations = recordings["No correction"].get_channel_locations()
    half = int(round(1.0e-3 * fs))
    time_ms = np.arange(-half, half + 1) / fs * 1e3

    fig, axes = plt.subplots(len(switched), 3, figsize=(12, 8), constrained_layout=True)
    summary_rows = []
    for row, (event_id, center) in enumerate(zip(switched, samples)):
        event = review.loc[event_id]
        local = np.flatnonzero(np.abs(locations[:, 1] - event.peak_depth_um) <= 120)
        traces = {
            name: rec.get_traces(start_frame=int(center - half), end_frame=int(center + half + 1), channel_ids=rec.channel_ids[local]).astype(float)
            for name, rec in recordings.items()
        }
        peak_local = int(np.argmin(np.abs(locations[local, 1] - event.peak_depth_um)))

        ax = axes[row, 0]
        for name, trace in traces.items():
            ax.plot(time_ms, trace[:, peak_local], linewidth=1.4, label=name)
        ax.axvline(0, color="0.5", linewidth=0.8)
        before_outcome = bool(pivot.loc[event_id, "no_external_correction"])
        after_outcome = bool(pivot.loc[event_id, "rigid_gain_025"])
        ax.set_title(f"{event_id}: {int(before_outcome)} → {int(after_outcome)}")
        ax.set(ylabel="Conditioned counts", xlabel="Time from event (ms)")
        if row == 0:
            ax.legend(frameon=False)

        common_limit = max(np.abs(np.concatenate([trace.ravel() for trace in traces.values()])))
        for col, (name, trace) in enumerate(traces.items(), start=1):
            ax = axes[row, col]
            image = ax.imshow(
                trace.T,
                aspect="auto",
                origin="lower",
                extent=[time_ms[0], time_ms[-1], locations[local, 1].min(), locations[local, 1].max()],
                cmap="RdBu_r",
                vmin=-common_limit,
                vmax=common_limit,
            )
            ax.axvline(0, color="black", linewidth=0.7)
            ax.set(title=name, xlabel="Time (ms)", ylabel="Depth (µm)")
            extrema = np.max(np.abs(trace), axis=0)
            summary_rows.append(
                {
                    "review_id": event_id,
                    "condition": name,
                    "recovered": before_outcome if name == "No correction" else after_outcome,
                    "peak_abs_counts": float(extrema.max()),
                    "peak_depth_um": float(locations[local[np.argmax(extrema)], 1]),
                    "local_rms_counts": float(np.sqrt(np.mean(trace**2))),
                }
            )
        fig.colorbar(image, ax=axes[row, 1:], shrink=0.75, label="Conditioned counts")

    fig.suptitle("Voltage-level effect at replication events whose sort recovery switches", fontsize=14, fontweight="bold")
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_ROOT / "rigid_gain_replication_switched_events.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(FIGURE_ROOT / "rigid_gain_replication_switched_event_traces.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
