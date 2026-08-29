"""Compare Luke motion with matched 120 s windows from the known-good Yates session."""

from __future__ import annotations

from pathlib import Path
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs/luke_motion_candidate_results"
LUKE_RUNS = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "motion_scale_sweep/runs"
)
YATES_ROOT = Path("/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16")
WINDOW_S = 120.0


def summarize_window(motion: np.ndarray, times: np.ndarray) -> dict[str, float]:
    rigid = np.mean(motion, axis=1)
    spread = np.ptp(motion, axis=1)
    speed = np.abs(np.diff(rigid) / np.diff(times))
    return {
        "rigid_excursion_p95_p5_um": float(np.percentile(rigid, 95) - np.percentile(rigid, 5)),
        "median_nonrigid_spread_um": float(np.median(spread)),
        "p95_nonrigid_spread_um": float(np.percentile(spread, 95)),
        "p99_rigid_speed_um_per_s": float(np.percentile(speed, 99)),
    }


def load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    return np.load(path / "motion.npy"), np.load(path / "time_bins.npy")


def build_rows() -> pd.DataFrame:
    rows = []
    estimator_paths = {
        "MEDiCINe": ("medicine_pipeline_default", "medicine"),
        "Kilosort-style": ("ks_pipeline_default", "ks-motion"),
    }
    for estimator, (luke_candidate, yates_folder) in estimator_paths.items():
        for probe in ("imec0", "imec1"):
            matches = glob.glob(str(LUKE_RUNS / probe / luke_candidate / "full_*"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {probe}/{luke_candidate} run, found {matches}")
            motion, times = load_arrays(Path(matches[0]))
            rows.append(
                {
                    "dataset": "Luke",
                    "probe": probe,
                    "estimator": estimator,
                    "window_start_s": 8160.0,
                    "window_duration_s": WINDOW_S,
                    "window_kind": "prespecified pathological",
                    **summarize_window(motion, times),
                }
            )
        for shank in ("shank1", "shank2"):
            motion, times = load_arrays(YATES_ROOT / f"{shank}-motion" / yates_folder)
            starts = np.arange(float(times.min()), float(times.max()) - WINDOW_S, WINDOW_S)
            for start in starts:
                keep = (times >= start) & (times < start + WINDOW_S)
                if keep.sum() < 10:
                    continue
                rows.append(
                    {
                        "dataset": "Yates",
                        "probe": shank,
                        "estimator": estimator,
                        "window_start_s": float(start),
                        "window_duration_s": WINDOW_S,
                        "window_kind": "nonoverlapping reference",
                        **summarize_window(motion[keep], times[keep]),
                    }
                )
    return pd.DataFrame(rows)


def render(rows: pd.DataFrame) -> None:
    metrics = [
        ("rigid_excursion_p95_p5_um", "Rigid excursion", "P95−P5 (µm)"),
        ("median_nonrigid_spread_um", "Median differential spread", "Across-depth P95−P5 (µm)"),
        ("p95_nonrigid_spread_um", "P95 differential spread", "Across-depth P95−P5 (µm)"),
        ("p99_rigid_speed_um_per_s", "P99 rigid speed", "µm/s"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8), constrained_layout=True)
    categories = [("Yates", "shank1"), ("Yates", "shank2"), ("Luke", "imec0"), ("Luke", "imec1")]
    labels = ["Yates\nshank 1", "Yates\nshank 2", "Luke\nimec0", "Luke\nimec1"]
    for row_index, estimator in enumerate(("MEDiCINe", "Kilosort-style")):
        estimator_rows = rows[rows.estimator == estimator]
        for column, (field, title, ylabel) in enumerate(metrics):
            ax = axes[row_index, column]
            for position, (dataset, probe) in enumerate(categories):
                values = estimator_rows[(estimator_rows.dataset == dataset) & (estimator_rows.probe == probe)][field].to_numpy(float)
                if dataset == "Yates":
                    box = ax.boxplot(values, positions=[position], widths=0.55, patch_artist=True, showfliers=False)
                    box["boxes"][0].set(facecolor="#d8e5f0", edgecolor="#32688e")
                    for element in ("whiskers", "caps", "medians"):
                        for line in box[element]:
                            line.set(color="#32688e", linewidth=1.3)
                    jitter = np.linspace(-0.12, 0.12, len(values))
                    ax.scatter(position + jitter, values, s=8, color="#32688e", alpha=0.28)
                else:
                    ax.scatter(position, values[0], s=75, color="#d17a22", edgecolor="#6f4514", zorder=4)
                    ax.text(position, values[0] * 1.04 + 0.02, f"{values[0]:.1f}", ha="center", fontsize=8)
            ax.set_xticks(range(4), labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="0.9")
            if row_index == 0:
                ax.set_title(title)
            if column == 0:
                ax.text(-0.22, 0.5, estimator, transform=ax.transAxes, rotation=90, va="center", ha="center", fontweight="bold")
    fig.suptitle(
        "Luke and Yates motion estimates in 120-second windows\n"
        "Yates boxes: all nonoverlapping windows; Luke points: prespecified 8,160–8,280 s pathological window",
        fontsize=14,
        fontweight="bold",
        linespacing=1.45,
    )
    fig.savefig(OUTPUT_ROOT / "luke_yates_motion_120s_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.to_csv(OUTPUT_ROOT / "luke_yates_motion_120s_metrics.csv", index=False)
    render(rows)
    summary = rows[rows.dataset == "Yates"].groupby(["estimator", "probe"])[
        ["rigid_excursion_p95_p5_um", "median_nonrigid_spread_um", "p95_nonrigid_spread_um", "p99_rigid_speed_um_per_s"]
    ].agg(["median", "max"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
