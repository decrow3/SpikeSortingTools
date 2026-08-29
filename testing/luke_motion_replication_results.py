"""Visualize the independent Luke rigid-gain replication."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPLICATION_ROOT = Path(
    "/mnt/NPX/Luke/20250804/"
    "dredge_pipeline_results_Luke0804_V2V1_g0_imec1/"
    "motion_candidate_replication/shared_template"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs/luke_motion_candidate_results"
LABELS = {
    "no_external_correction": "No correction",
    "rigid_gain_025": "0.25×, sigma 20 µm",
    "rigid_gain_025_sigma10": "0.25×, sigma 10 µm",
}
COLORS = {
    "no_external_correction": "#32688e",
    "rigid_gain_025": "#d17a22",
    "rigid_gain_025_sigma10": "#b89b26",
}


def main() -> None:
    scores = pd.read_csv(REPLICATION_ROOT / "replication_scores.csv")
    events = pd.read_csv(REPLICATION_ROOT / "paired_event_recovery.csv")
    visual = scores[scores.population == "visual_neural_unmatched"].set_index("condition")
    automatic = scores[scores.population == "automatic_neural_like_unmatched"].set_index("condition")
    neural = events[(events.review_label == "neural") & (events.status == "unmatched")]
    matrix = neural.pivot(index="review_id", columns="condition", values="recovered")
    order = matrix.sort_values(list(matrix.columns), ascending=False).index
    matrix = matrix.loc[order, list(LABELS)]

    plt.rcParams.update({"font.size": 9, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    conditions = list(LABELS)
    x = np.arange(len(conditions))

    ax = axes[0, 0]
    widths = 0.34
    vals_visual = [100 * visual.loc[c, "observed_recovery"] for c in conditions]
    vals_auto = [100 * automatic.loc[c, "observed_recovery"] for c in conditions]
    ax.bar(x - widths / 2, vals_visual, widths, label="Visual neural (n=35)", color="#4c78a8")
    ax.bar(x + widths / 2, vals_auto, widths, label="Automated neural-like (n=12)", color="#72b7b2")
    for xpos, value in zip(x - widths / 2, vals_visual):
        ax.text(xpos, value + 1.2, f"{int(round(value * 35 / 100))}/35", ha="center")
    for xpos, value in zip(x + widths / 2, vals_auto):
        ax.text(xpos, value + 1.2, f"{int(round(value * 12 / 100))}/12", ha="center")
    ax.set(ylim=(0, 108), ylabel="Recovered events (%)", xticks=x, xticklabels=[LABELS[c] for c in conditions])
    ax.set_title("Independent 240 s recovery replication")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[0, 1]
    metrics = ["n_ks_good", "median_contamination_pct"]
    metric_labels = ["KS-good units", "Median contamination (%)"]
    normalized = np.array([[visual.loc[c, m] for m in metrics] for c in conditions], float)
    quality_width = 0.24
    for i, c in enumerate(conditions):
        positions = np.arange(len(metrics)) + (i - 1) * quality_width
        ax.bar(positions, normalized[i], quality_width, color=COLORS[c], label=LABELS[c])
        for j, value in enumerate(normalized[i]):
            ax.text(j + (i - 1) * quality_width, value + 2, f"{value:.0f}", ha="center", fontsize=8)
    ax.set(xticks=np.arange(len(metrics)), xticklabels=metric_labels, ylim=(0, 145))
    ax.set_title("Unit-quality guardrails")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    image = ax.imshow(matrix.to_numpy(int).T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set(yticks=np.arange(len(matrix.columns)), yticklabels=[LABELS[c] for c in matrix.columns], xlabel="Visually neural missed events")
    ax.set_xticks([])
    ax.set_title("Paired visual-neural event recovery")
    fig.colorbar(image, ax=ax, ticks=[0, 1], label="Recovered")

    ax = axes[1, 1]
    learned = [visual.loc[c, "learned_detection_count"] / 1e6 for c in conditions]
    final = [visual.loc[c, "n_final_spikes"] / 1e6 for c in conditions]
    ax.bar(x - widths / 2, learned, widths, color="#9ecae9", label="Learned detections")
    ax.bar(x + widths / 2, final, widths, color="#3182bd", label="Final spikes")
    ax.set(xticks=x, xticklabels=[LABELS[c] for c in conditions], ylabel="Count (millions)", ylim=(0, 1.55))
    ax.set_title("Detection counts and coincidence")
    ax.legend(frameon=False)
    coincidence = [100 * visual.loc[c, "cross_unit_near_coincident_fraction"] for c in conditions]
    for i, value in enumerate(coincidence):
        ax.text(i, 0.05, f"Coinc. {value:.1f}%", ha="center", color="white", fontweight="bold", fontsize=8)

    fig.suptitle("Luke shared-window rigid correction: kriging-scale comparison", fontsize=14, fontweight="bold")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_ROOT / "rigid_gain_replication.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
