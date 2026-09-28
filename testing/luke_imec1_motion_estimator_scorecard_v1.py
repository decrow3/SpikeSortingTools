#!/usr/bin/env python
"""Create a compact, common-support imec1 lighthouse estimator scorecard."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "testing/outputs/luke_imec1_lighthouse_motion_comparison_v7"
ORDER = ["zero", "native_ap", "medicine", "medicine_rigid", "lfp80", "lfp260"]
LABELS = {
    "zero": "No motion",
    "native_ap": "Native AP",
    "medicine": "MEDiCINe",
    "medicine_rigid": "MEDiCINe rigid",
    "lfp80": "LFP80",
    "lfp260": "LFP260",
}
COLORS = {
    "zero": "#777777", "native_ap": "#0072B2", "medicine": "#CC79A7",
    "medicine_rigid": "#8E5EA2", "lfp80": "#E69F00", "lfp260": "#009E73",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input / "estimator_scorecard_common_support.png"

    metrics = pd.read_csv(args.input / "family_metrics.csv")
    support = pd.read_csv(args.input / "support_summary.csv")
    movement = metrics.loc[metrics.regime.eq("movement")].set_index("candidate").loc[ORDER]
    quiet = metrics.loc[metrics.regime.eq("quiet")].set_index("candidate").loc[ORDER]
    coverage = support.loc[support.grain.eq("all") & support.candidate.isin(ORDER)].set_index("candidate").loc[ORDER]

    table = pd.DataFrame({
        "estimator": ORDER,
        "label": [LABELS[name] for name in ORDER],
        "supported_event_fraction": coverage.supported_fraction.to_numpy(),
        "movement_rmse_um": movement.rmse_um.to_numpy(),
        "movement_mae_um": movement.mae_um.to_numpy(),
        "movement_p95_um": movement.p95_abs_error_um.to_numpy(),
        "movement_over_120_fraction": movement.exceed_120_fraction.to_numpy(),
        "quiet_rmse_um": quiet.rmse_um.to_numpy(),
    })
    zero_rmse = float(table.loc[table.estimator.eq("zero"), "movement_rmse_um"].iloc[0])
    zero_quiet = float(table.loc[table.estimator.eq("zero"), "quiet_rmse_um"].iloc[0])
    table["movement_skill_vs_zero"] = 1 - table.movement_rmse_um / zero_rmse
    table["quiet_rmse_added_vs_zero_um"] = table.quiet_rmse_um - zero_quiet
    table.to_csv(args.input / "estimator_scorecard_common_support.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = (
        ("movement_rmse_um", "Movement RMSE (µm)", zero_rmse),
        ("movement_mae_um", "Movement MAE (µm)", float(table.loc[table.estimator.eq("zero"), "movement_mae_um"].iloc[0])),
        ("movement_p95_um", "Movement P95 absolute residual (µm)", float(table.loc[table.estimator.eq("zero"), "movement_p95_um"].iloc[0])),
        ("quiet_rmse_um", "Quiet-period RMSE (µm)", zero_quiet),
    )
    y = np.arange(len(table))
    for ax, (column, title, baseline) in zip(axes.flat, panels):
        values = table[column].to_numpy(float)
        ax.barh(y, values, color=[COLORS[name] for name in ORDER], alpha=.86)
        ax.axvline(baseline, color="#555555", ls="--", lw=1, label="no-motion baseline")
        for yi, value in zip(y, values):
            ax.text(value, yi, f" {value:.1f}", va="center", fontsize=8)
        ax.set_yticks(y, table.label if ax in axes[:, 0] else [])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=.2)
        ax.set_xlim(0, max(values) * 1.16)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Luke0804 imec1 lighthouse estimator comparison\n"
        "Identical all-estimator support: 5 families, 36 movement increments, 16 scored quiet increments"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
