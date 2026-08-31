"""Build reviewed figures and evidence tables from the Luke input factorial."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testing.luke_motion_input_factorial import load_common_field
from testing.luke_motion_scale_characterization import decompose_spatial_field


ROOT = Path("testing/outputs/luke_motion_input_factorial")
REGIMES = ("quiet", "rapid_motion", "sustained_noise", "support_dropout", "noise_plus_motion")
ESTIMATORS = ("dredge_300_200_cpu", "decentralized_300_200_numpy", "iterative_300_200")
COLORS = {"dredge_300_200_cpu": "#506b3f", "decentralized_300_200_numpy": "#d08b27", "iterative_300_200": "#3569a8"}
LABELS = {"dredge_300_200_cpu": "DREDGE", "decentralized_300_200_numpy": "decentralized", "iterative_300_200": "iterative"}


def run_dir(regime: str, probe: str, estimator: str, condition: str = "full") -> Path:
    return ROOT / "runs" / regime / probe / estimator / condition


def rigid_trace(regime: str, probe: str, estimator: str, condition: str = "full") -> tuple[np.ndarray, np.ndarray]:
    field, times, depths = load_common_field(run_dir(regime, probe, estimator, condition), 120.0)
    rigid = np.asarray(decompose_spatial_field(field, depths)["rigid"])
    return times, rigid - np.median(rigid)


def trace_figure() -> None:
    fig, axes = plt.subplots(len(REGIMES), 1, figsize=(11, 13), sharex=True)
    for ax, regime in zip(axes, REGIMES):
        for probe, linestyle in (("imec0", "-"), ("imec1", "--")):
            for estimator in ESTIMATORS[:2]:
                times, trace = rigid_trace(regime, probe, estimator)
                ax.plot(
                    times,
                    trace,
                    color=COLORS[estimator],
                    linestyle=linestyle,
                    linewidth=1.6,
                    alpha=0.9,
                    label=f"{LABELS[estimator]} {probe}",
                )
        ax.axhline(0, color="#888888", linewidth=0.6)
        ax.set_ylabel("µm")
        ax.set_title(regime.replace("_", " "))
    axes[-1].set_xlabel("Seconds from window start")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4, frameon=False)
    fig.suptitle("Matched estimators and probes separate supported motion from support dropout", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ROOT / "regime_rigid_traces.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def robustness_figure(agreement: pd.DataFrame) -> None:
    conditions = (
        "random_half",
        "random_quarter",
        "high_amplitude_half",
        "exclude_synchronous",
        "exclude_bursty_seconds",
        "exclude_dominant_channel",
    )
    selected = agreement[
        agreement.scope.eq("condition_vs_full")
        & agreement.probe.eq("imec1")
        & agreement.left_estimator.isin(ESTIMATORS[:2])
        & agreement.condition.isin(conditions)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True, constrained_layout=True)
    for column, metric in enumerate(("rigid_correlation", "nonrigid_correlation")):
        for row, estimator in enumerate(ESTIMATORS[:2]):
            values = selected[selected.left_estimator.eq(estimator)].pivot(
                index="regime", columns="condition", values=metric
            ).reindex(index=REGIMES, columns=conditions)
            ax = axes[row, column]
            image = ax.imshow(values, vmin=-0.2, vmax=1.0, cmap="RdYlBu", aspect="auto")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    value = values.iloc[y, x]
                    if np.isfinite(value):
                        ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=8)
            ax.set_title(f"{LABELS[estimator]} — {metric.replace('_', ' ')}")
            labels = {
                "random_half": "random 50%",
                "random_quarter": "random 25%",
                "high_amplitude_half": "high-amp half",
                "exclude_synchronous": "exclude sync",
                "exclude_bursty_seconds": "exclude burst s",
                "exclude_dominant_channel": "exclude top ch",
            }
            ax.set_xticks(range(len(conditions)), [labels[value] for value in conditions], rotation=30, ha="right")
            ax.set_yticks(range(len(REGIMES)), [value.replace("_", " ") for value in REGIMES])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Correlation to full-input field", shrink=0.78, pad=0.02)
    fig.suptitle("Input perturbations expose regime- and estimator-specific instability")
    fig.savefig(ROOT / "input_perturbation_robustness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def cross_probe_figure(agreement: pd.DataFrame) -> None:
    values = agreement[
        agreement.scope.eq("cross_probe")
        & agreement.condition.eq("full")
        & agreement.left_estimator.isin(ESTIMATORS[:2])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(REGIMES))
    width = 0.35
    for index, estimator in enumerate(ESTIMATORS[:2]):
        part = values[values.left_estimator.eq(estimator)].set_index("regime").reindex(REGIMES)
        axes[0].bar(x + (index - 0.5) * width, part.rigid_correlation, width, label=LABELS[estimator], color=COLORS[estimator])
        axes[1].bar(x + (index - 0.5) * width, part.nonrigid_correlation, width, label=LABELS[estimator], color=COLORS[estimator])
    for ax, title in zip(axes, ("Rigid component", "Residual nonrigid component")):
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.axhline(0.65, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xticks(x, [value.replace("_", "\n") for value in REGIMES])
        ax.set_ylim(-0.25, 1.05)
        ax.set_title(title)
        ax.set_ylabel("imec0–imec1 correlation")
    axes[1].legend(frameon=False)
    fig.suptitle("Cross-probe support is strong for rapid and noise-plus-motion rigid trajectories")
    fig.tight_layout()
    fig.savefig(ROOT / "cross_probe_regime_support.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evidence_table(summary: pd.DataFrame, agreement: pd.DataFrame) -> pd.DataFrame:
    rows = []
    classes = {
        "quiet": ("small supported rigid drift", "Small 4–6 µm rigid excursion; nonrigid residual is not cross-probe reproducible."),
        "rapid_motion": ("supported motion", "Large 25–44 µm rigid excursion with method, subsample, exclusion, and cross-probe support."),
        "sustained_noise": ("small/ambiguous motion", "Rigid component is small and robust; local residual is not cross-probe supported."),
        "support_dropout": ("estimator failure risk", "Method and cross-probe agreement fail; DREDGE-only tail spread is not independently reproduced."),
        "noise_plus_motion": ("supported rigid motion; noise-biased residual", "Rigid trajectory survives masks and agrees across probes; dominant-channel removal alters residual structure."),
    }
    for regime in REGIMES:
        full = summary[(summary.regime.eq(regime)) & summary.condition.eq("full")]
        cross_method = agreement[
            agreement.scope.eq("cross_estimator")
            & agreement.regime.eq(regime)
            & agreement.condition.eq("full")
            & agreement.left_estimator.eq("decentralized_300_200_numpy")
            & agreement.right_estimator.eq("dredge_300_200_cpu")
        ]
        cross_probe = agreement[
            agreement.scope.eq("cross_probe")
            & agreement.regime.eq(regime)
            & agreement.condition.eq("full")
            & agreement.left_estimator.eq("dredge_300_200_cpu")
        ]
        dredge = full[full.estimator.eq("dredge_300_200_cpu")]
        rows.append(
            {
                "regime": regime,
                "classification": classes[regime][0],
                "dredge_rigid_excursion_range_um": f"{dredge.rigid_excursion_p95_p5_um.min():.1f}–{dredge.rigid_excursion_p95_p5_um.max():.1f}",
                "within_probe_dredge_dc_r_range": f"{cross_method.rigid_correlation.min():.2f}–{cross_method.rigid_correlation.max():.2f}",
                "cross_probe_dredge_r": float(cross_probe.rigid_correlation.iloc[0]),
                "cross_probe_dredge_residual_r": float(cross_probe.nonrigid_correlation.iloc[0]),
                "interpretation": classes[regime][1],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    summary = pd.read_csv(ROOT / "factorial_field_summary.csv")
    agreement = pd.read_csv(ROOT / "factorial_agreement.csv")
    trace_figure()
    robustness_figure(agreement)
    cross_probe_figure(agreement)
    evidence_table(summary, agreement).to_csv(ROOT / "regime_evidence.csv", index=False)


if __name__ == "__main__":
    main()
