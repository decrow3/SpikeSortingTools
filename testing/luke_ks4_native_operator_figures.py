"""Render the preregistered Luke KS4 native-operator diagnostic figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_ks4_native_operator_audit import DEFAULT_OUTPUT


ARM_LABELS = {
    "moved_no_correction": "Moved, uncorrected",
    "moved_ks4_native_inverse": "KS4 native inverse",
    "moved_ks4_external_order_inverse": "KS4 external order",
    "moved_si_inverse": "SI inverse",
}


def render(root: Path = DEFAULT_OUTPUT) -> list[Path]:
    metrics = pd.read_csv(root / "case_metrics.csv")
    pairs = pd.read_csv(root / "pair_separability_metrics.csv")
    moved = metrics.loc[metrics.arm.isin(ARM_LABELS)].copy()
    moved["abs_displacement_um"] = moved.displacement_um.abs()
    colors = {
        "moved_no_correction": "#4c78a8",
        "moved_ks4_native_inverse": "#e45756",
        "moved_ks4_external_order_inverse": "#f2cf5b",
        "moved_si_inverse": "#72b7b2",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for arm in ARM_LABELS:
        summary = moved.loc[moved.arm.eq(arm)].groupby("abs_displacement_um").residual_fraction.median()
        axes[0, 0].plot(summary.index, summary.values, marker="o", label=ARM_LABELS[arm], color=colors[arm])
    axes[0, 0].set(title="Median waveform residual", xlabel="|displacement| (µm)", ylabel="Residual fraction")
    axes[0, 0].legend(frameon=False, fontsize=8)

    pivot = moved.loc[moved.arm.isin(["moved_no_correction", "moved_ks4_native_inverse"])].pivot_table(
        index=["template_id", "background", "generator", "displacement_um"],
        columns="arm",
        values="residual_fraction",
    ).reset_index()
    pivot["delta"] = pivot.moved_ks4_native_inverse - pivot.moved_no_correction
    pivot["abs_displacement_um"] = pivot.displacement_um.abs()
    for generator, group in pivot.groupby("generator", sort=True):
        summary = group.groupby("abs_displacement_um").delta.median()
        axes[0, 1].plot(summary.index, summary.values, marker="o", label=generator)
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set(
        title="Native KS4 residual change vs uncorrected",
        xlabel="|displacement| (µm)",
        ylabel="Δ residual (negative is better)",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)

    for arm in ARM_LABELS:
        summary = moved.loc[moved.arm.eq(arm)].groupby("abs_displacement_um").amplitude_retention.median()
        axes[1, 0].plot(summary.index, summary.values, marker="o", label=ARM_LABELS[arm], color=colors[arm])
    axes[1, 0].axhline(1, color="black", lw=1)
    axes[1, 0].set(title="Median amplitude retention", xlabel="|displacement| (µm)", ylabel="Recovered/reference")

    sign = pivot.assign(sign=np.where(pivot.displacement_um > 0, "positive", "negative")).groupby(
        ["generator", "sign"]
    ).delta.median().unstack()
    x = np.arange(len(sign))
    width = 0.36
    axes[1, 1].bar(x - width / 2, sign["negative"], width, label="negative displacement")
    axes[1, 1].bar(x + width / 2, sign["positive"], width, label="positive displacement")
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set_xticks(x, sign.index, rotation=20, ha="right")
    axes[1, 1].set(title="Signed-displacement robustness", ylabel="Median native Δ residual")
    axes[1, 1].legend(frameon=False, fontsize=8)
    recovery_path = root / "operator_recovery_curves.png"
    fig.savefig(recovery_path, dpi=180)
    plt.close(fig)

    summary = (
        pairs.loc[pairs.arm.isin(ARM_LABELS)]
        .groupby(["arm", "generator"])
        .agg(
            median_distance_retention=("distance_retention", "median"),
            p10_distance_retention=("distance_retention", lambda x: x.quantile(0.1)),
            median_separation_to_noise=("separation_to_noise", "median"),
            p10_separation_to_noise=("separation_to_noise", lambda x: x.quantile(0.1)),
        )
        .reset_index()
    )
    generators = sorted(summary.generator.unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    width = 0.19
    x = np.arange(len(generators))
    for index, arm in enumerate(ARM_LABELS):
        group = summary.loc[summary.arm.eq(arm)].set_index("generator").reindex(generators)
        position = x + (index - 1.5) * width
        axes[0].bar(position, group.median_distance_retention, width, label=ARM_LABELS[arm], color=colors[arm])
        axes[0].scatter(position, group.p10_distance_retention, marker="_", color="black", zorder=3)
        axes[1].bar(position, group.median_separation_to_noise, width, label=ARM_LABELS[arm], color=colors[arm])
        axes[1].scatter(position, group.p10_separation_to_noise, marker="_", color="black", zorder=3)
    for axis in axes:
        axis.set_xticks(x, generators, rotation=20, ha="right")
    axes[0].axhline(1, color="black", lw=1)
    axes[0].set(title="Fixed-pair distance retention", ylabel="Median; black mark = p10")
    axes[1].set(title="Reference-direction separation/noise", ylabel="Median; black mark = p10")
    axes[0].legend(frameon=False, fontsize=8)
    separation_path = root / "template_separability.png"
    fig.savefig(separation_path, dpi=180)
    plt.close(fig)

    stationary = metrics.loc[metrics.arm.eq("stationary_ks4_d0")]
    tax = pd.Series(
        {
            "Residual fraction": stationary.residual_fraction.median(),
            "Absolute amplitude error": np.abs(stationary.amplitude_retention - 1).median(),
            "Cosine loss": (1 - stationary.template_cosine).median(),
        }
    )
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar(tax.index, tax.values, color=["#e45756", "#f58518", "#72b7b2"])
    axis.set(title="Stationary KS4 dshift=0 registration tax", ylabel="Median paired degradation")
    axis.tick_params(axis="x", rotation=18)
    tax_path = root / "zero_shift_tax.png"
    fig.savefig(tax_path, dpi=180)
    plt.close(fig)
    return [recovery_path, separation_path, tax_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in render(args.root):
        print(path)


if __name__ == "__main__":
    main()
