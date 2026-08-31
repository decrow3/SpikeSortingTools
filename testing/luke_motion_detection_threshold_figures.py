"""Build compact evidence products for the Luke 5/6/7 threshold audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("testing/outputs/luke_motion_detection_threshold_factorial")
LABELS = {
    "quiet": "Quiet",
    "rapid_motion": "Rapid motion",
    "sustained_noise": "Sustained anomaly",
    "support_dropout": "Support dropout",
    "noise_plus_motion": "Noise + motion",
}
ESTIMATORS = {
    "dredge_300_200_cpu": "DREDGE",
    "decentralized_300_200_numpy": "Decentralized",
}


def main() -> None:
    summary = pd.read_csv(ROOT / "threshold_field_summary.csv")
    agreement = pd.read_csv(ROOT / "threshold_agreement.csv")
    cross = pd.read_csv(ROOT / "threshold_cross_estimator_agreement.csv")

    counts = summary[summary.estimator.eq("dredge_300_200_cpu")].pivot(
        index="regime", columns="threshold", values="peak_count"
    )
    retained = counts[7.0] / counts[5.0]
    rigid7 = agreement[agreement.threshold.eq(7.0)].pivot(
        index="regime", columns="estimator", values="rigid_correlation_vs_threshold5"
    )
    residual7 = agreement[agreement.threshold.eq(7.0)].pivot(
        index="regime", columns="estimator", values="nonrigid_correlation_vs_threshold5"
    )
    cross5 = cross[cross.threshold.eq(5.0)].set_index("regime")
    cross7 = cross[cross.threshold.eq(7.0)].set_index("regime")

    interpretations = {
        "quiet": "Small rigid drift is threshold-stable; local residual is weaker and method-dependent.",
        "rapid_motion": "Large rigid trajectory survives losing about two thirds of peaks and retains method consensus.",
        "sustained_noise": "Small rigid estimate persists, but residual structure changes strongly for decentralized.",
        "support_dropout": "Thresholding does not rescue failed cross-probe support; decentralized trajectory is least stable.",
        "noise_plus_motion": "Large rigid trajectory and method consensus remain strong; residual scale decreases modestly.",
    }
    rows = []
    for regime in LABELS:
        rows.append({
            "regime": regime,
            "threshold7_retained_fraction": float(retained[regime]),
            "dredge_rigid_r_threshold7_vs5": float(rigid7.loc[regime, "dredge_300_200_cpu"]),
            "decentralized_rigid_r_threshold7_vs5": float(rigid7.loc[regime, "decentralized_300_200_numpy"]),
            "dredge_residual_r_threshold7_vs5": float(residual7.loc[regime, "dredge_300_200_cpu"]),
            "decentralized_residual_r_threshold7_vs5": float(residual7.loc[regime, "decentralized_300_200_numpy"]),
            "cross_estimator_rigid_r_threshold5": float(cross5.loc[regime, "rigid_dredge_decentralized_correlation"]),
            "cross_estimator_rigid_r_threshold7": float(cross7.loc[regime, "rigid_dredge_decentralized_correlation"]),
            "interpretation": interpretations[regime],
        })
    evidence = pd.DataFrame(rows)
    evidence.to_csv(ROOT / "threshold_regime_evidence.csv", index=False)

    order = list(LABELS)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    axes[0].bar(x, [retained[r] for r in order], color="#607D8B")
    axes[0].set_ylabel("Fraction of threshold-5 peaks retained")
    axes[0].set_title("Threshold 7 retains about one third of detections")
    axes[0].set_ylim(0, 0.6)
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.36
    axes[1].bar(
        x - width / 2,
        [rigid7.loc[r, "dredge_300_200_cpu"] for r in order],
        width,
        label="DREDGE",
        color="#1565C0",
    )
    axes[1].bar(
        x + width / 2,
        [rigid7.loc[r, "decentralized_300_200_numpy"] for r in order],
        width,
        label="Decentralized",
        color="#EF6C00",
    )
    axes[1].axhline(0.9, color="black", linestyle="--", linewidth=1, alpha=0.65)
    axes[1].set_ylabel("Rigid correlation: threshold 7 vs 5")
    axes[1].set_title("Supported rigid fields survive the threshold change")
    axes[1].set_ylim(0.6, 1.01)
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)
    for axis in axes:
        axis.set_xticks(x, [LABELS[r] for r in order], rotation=25, ha="right")
    fig.suptitle("Luke 2025-08-04 historical-equivalent detection-threshold audit", fontsize=14)
    fig.savefig(ROOT / "threshold_robustness.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
