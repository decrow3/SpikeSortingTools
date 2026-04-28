import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SWEEP_DIR = Path(
    os.environ.get(
        "PLOT_SWEEP_DIR",
        "/mnt/NPX/Luke/20260316/dredge_pipeline_results_Luke03162026_V2V1_RH_g0_imec1/shallow_sweep",
    )
)

RUN_CANDIDATES = ["default", "claim_tonly", "claim_spatial", "peel3", "peel2", "peel1"]
RUN_LABELS = {
    "default": "default\n(100 peels)",
    "claim_tonly": "claim_tonly\n(0.25 ms)",
    "claim_spatial": "claim_spatial\n(0.25 ms, 75 um)",
    "peel3": "peel3\n(3 peels)",
    "peel2": "peel2\n(2 peels)",
    "peel1": "peel1\n(1 peel)",
}
RUN_COLORS = {
    "default": "#8C2D1E",
    "claim_tonly": "#B34A3C",
    "claim_spatial": "#355C7D",
    "peel3": "#C97B2C",
    "peel2": "#6E8B74",
    "peel1": "#1F4E5F",
}

FS = 30_000.0
FINE_CCG_WINDOW_S = 5e-3
FINE_CCG_BIN_S = 0.2e-3
FINE_CCG_NEAR_ZERO_S = 0.5e-3
DUPLICATE_NEAR_ZERO_FRAC_THRESH = 0.05
DUPLICATE_ZERO_PEAK_RATIO_THRESH = 1.25

OUTPUT_PDF = SWEEP_DIR / "fig_peeling_dose_response.pdf"
OUTPUT_PNG = SWEEP_DIR / "fig_peeling_dose_response.png"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titlepad": 5,
        "axes.labelpad": 4,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "figure.dpi": 160,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def fine_ccg_for_pair(times_a_s, times_b_s):
    ta = np.sort(np.asarray(times_a_s, float))
    tb = np.sort(np.asarray(times_b_s, float))
    edges = np.arange(-FINE_CCG_WINDOW_S, FINE_CCG_WINDOW_S + FINE_CCG_BIN_S, FINE_CCG_BIN_S)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(ta) == 0 or len(tb) == 0:
        return centers, np.zeros(len(centers), dtype=int)

    dts = []
    left = 0
    right = 0
    for t in ta:
        while left < len(tb) and tb[left] < (t - FINE_CCG_WINDOW_S):
            left += 1
        if right < left:
            right = left
        while right < len(tb) and tb[right] <= (t + FINE_CCG_WINDOW_S):
            right += 1
        if right > left:
            dts.append(tb[left:right] - t)

    if not dts:
        return centers, np.zeros(len(centers), dtype=int)

    all_dt = np.concatenate(dts)
    counts, _ = np.histogram(all_dt, bins=edges)
    return centers, counts


def ecdf(values):
    values = np.sort(np.asarray(values, float))
    y = np.arange(1, len(values) + 1) / len(values)
    return values, y


def load_screen_df(run_name):
    csv_path = SWEEP_DIR / f"within_run_screen_{run_name}.csv"
    if not csv_path.exists() or not csv_path.read_text().strip():
        return pd.DataFrame(
            columns=[
                "unit_a",
                "unit_b",
                "near_zero_frac",
                "zero_peak_ratio",
                "total_pairs",
                "near_zero_pairs",
            ]
        )
    df = pd.read_csv(csv_path)
    return df


def summarize_run(df):
    valid = df["near_zero_frac"].notna() & df["zero_peak_ratio"].notna()
    flagged = (
        valid
        & (df["near_zero_frac"] >= DUPLICATE_NEAR_ZERO_FRAC_THRESH)
        & (df["zero_peak_ratio"] >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
    )
    return {
        "n_pairs": int(valid.sum()),
        "median_nzf": float(np.nanmedian(df.loc[valid, "near_zero_frac"])) if valid.any() else np.nan,
        "flagged_n": int(flagged.sum()),
        "flagged_frac": float(flagged.mean()) if valid.any() else np.nan,
    }


def format_median(value):
    return f"{value:.3f}" if np.isfinite(value) else "NA"


def load_summary_df():
    df = pd.read_csv(SWEEP_DIR / "sweep_summary.csv").set_index("run")
    return df


def detect_run_order(summary_df):
    run_order = []
    for run in RUN_CANDIDATES:
        csv_path = SWEEP_DIR / f"within_run_screen_{run}.csv"
        if run in summary_df.index and csv_path.exists():
            run_order.append(run)
    return run_order


def load_spikes_by_unit(run_name):
    sorter_out = SWEEP_DIR / f"run_{run_name}" / "kilosort4" / "sorter_output"
    spike_times = np.load(sorter_out / "spike_times.npy").astype(float) / FS
    spike_clusters = np.load(sorter_out / "spike_clusters.npy")
    by_unit = {}
    for uid in np.unique(spike_clusters):
        by_unit[int(uid)] = np.sort(spike_times[spike_clusters == uid])
    return by_unit


def choose_exemplar_pair(df):
    valid = df[df["near_zero_frac"].notna() & (df["total_pairs"] > 0)].copy()
    if valid.empty:
        return None
    flagged = valid[
        (valid["near_zero_frac"] >= DUPLICATE_NEAR_ZERO_FRAC_THRESH)
        & (valid["zero_peak_ratio"] >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
    ].copy()
    if not flagged.empty:
        flagged["score"] = flagged["near_zero_pairs"] * flagged["zero_peak_ratio"]
        flagged = flagged.sort_values(["score", "near_zero_pairs", "total_pairs"], ascending=[False, False, False])
        return flagged.iloc[0]
    valid = valid.sort_values(["total_pairs", "near_zero_frac"], ascending=[False, False])
    return valid.iloc[0]


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#333333")


def add_panel_label(ax, label):
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def main():
    summary_df = load_summary_df()
    run_order = detect_run_order(summary_df)
    if not run_order:
        raise RuntimeError(f"No within-run screen CSVs found in {SWEEP_DIR}")

    screen_dfs = {run: load_screen_df(run) for run in run_order}
    summary_rows = {run: summarize_run(df) for run, df in screen_dfs.items()}
    include_claimmask = any(run.startswith("claim_") for run in run_order)

    fig_width = max(11.0, 9.0 + 0.78 * len(run_order))
    fig = plt.figure(figsize=(fig_width, 8.3), facecolor="#F7F4EE")
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.25, 1.0],
        hspace=0.34,
        wspace=0.28,
    )

    ax_ecdf = fig.add_subplot(outer[0, 0], facecolor="#FBFAF7")
    ax_frac = fig.add_subplot(outer[0, 1], facecolor="#FBFAF7")
    ax_trade = fig.add_subplot(outer[1, 0], facecolor="#FBFAF7")

    ccg_grid = gridspec.GridSpecFromSubplotSpec(1, len(run_order), subplot_spec=outer[1, 1], wspace=0.18)
    ccg_axes = [fig.add_subplot(ccg_grid[0, i], facecolor="#FBFAF7") for i in range(len(run_order))]

    # Panel A: ECDFs of near-zero-lag fraction.
    for run in run_order:
        df = screen_dfs[run]
        values = df.loc[df["near_zero_frac"].notna(), "near_zero_frac"].to_numpy()
        x, y = ecdf(values)
        color = RUN_COLORS[run]
        ax_ecdf.plot(x, y, color=color, lw=2.4)

        med = summary_rows[run]["median_nzf"]
        if len(x) and np.isfinite(med):
            y_pos = np.interp(med, x, y)
            ax_ecdf.scatter([med], [y_pos], s=28, color=color, zorder=4, edgecolors="white", linewidth=0.8)

    ax_ecdf.axvline(DUPLICATE_NEAR_ZERO_FRAC_THRESH, color="#7A1F1F", ls="--", lw=1.0)
    ax_ecdf.text(
        DUPLICATE_NEAR_ZERO_FRAC_THRESH + 0.003,
        0.04,
        "duplicate threshold",
        color="#7A1F1F",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    ax_ecdf.set_xlim(0, 1.0)
    ax_ecdf.set_ylim(0, 1.02)
    ax_ecdf.set_xlabel("Near-zero-lag fraction across nearby within-run pairs")
    ax_ecdf.set_ylabel("Empirical cumulative fraction of pairs")
    if include_claimmask:
        ax_ecdf.set_title("Claim-mask runs sit below the default duplicate-burden distribution", loc="left")
    else:
        ax_ecdf.set_title("Duplicate-like pair structure collapses as max_peels is reduced", loc="left")
    handles = [plt.Line2D([0], [0], color=RUN_COLORS[run], lw=2.6) for run in run_order]
    labels = [
        f"{run}: {summary_rows[run]['flagged_n']}/{summary_rows[run]['n_pairs']} flagged, med={format_median(summary_rows[run]['median_nzf'])}"
        for run in run_order
    ]
    ax_ecdf.legend(handles, labels, loc="lower right", fontsize=7, handlelength=2.8)
    style_axis(ax_ecdf)
    add_panel_label(ax_ecdf, "A")

    # Panel B: monotonic summary dose-response.
    x_pos = np.arange(len(run_order))
    flagged_frac = [summary_rows[run]["flagged_frac"] for run in run_order]
    median_nzf = [summary_rows[run]["median_nzf"] for run in run_order]
    colors = [RUN_COLORS[run] for run in run_order]

    ax_frac.plot(x_pos, flagged_frac, color="#222222", lw=1.4, zorder=1)
    ax_frac.scatter(x_pos, flagged_frac, s=90, color=colors, edgecolors="white", linewidth=1.2, zorder=3)
    for xi, run in enumerate(run_order):
        flagged_n = summary_rows[run]["flagged_n"]
        n_pairs = summary_rows[run]["n_pairs"]
        if not np.isfinite(flagged_frac[xi]):
            continue
        ax_frac.text(
            xi,
            flagged_frac[xi] + 0.045,
            f"{flagged_n}/{n_pairs}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
        )

    ax_frac_2 = ax_frac.twinx()
    ax_frac_2.plot(x_pos, median_nzf, color="#8A5A44", lw=1.2, ls=":", zorder=1)
    finite_mask = np.isfinite(median_nzf)
    ax_frac_2.scatter(np.asarray(x_pos)[finite_mask], np.asarray(median_nzf)[finite_mask], s=36,
                      color="#F2E6D8", edgecolors="#8A5A44", linewidth=1.0, zorder=4)

    ax_frac.set_xticks(x_pos)
    ax_frac.set_xticklabels([RUN_LABELS[run] for run in run_order])
    ax_frac.set_ylim(0, 1.0)
    finite_median_nzf = [value for value in median_nzf if np.isfinite(value)]
    ax_frac_2.set_ylim(0, max(0.2, (max(finite_median_nzf) * 1.25) if finite_median_nzf else 0.2))
    ax_frac.set_ylabel("Flagged duplicate-like pair fraction")
    ax_frac_2.set_ylabel("Median near-zero-lag fraction", color="#8A5A44")
    ax_frac_2.tick_params(axis="y", colors="#8A5A44")
    if include_claimmask:
        ax_frac.set_title("Claim-mask intervention reaches low burden without collapsing to one peel", loc="left")
    else:
        ax_frac.set_title("The dose-response is monotonic in both burden and distribution shift", loc="left")
    style_axis(ax_frac)
    style_axis(ax_frac_2)
    add_panel_label(ax_frac, "B")

    # Panel C: burden vs downstream quality.
    ax_trade.axhspan(0.0, 0.2, color="#EFE7D8", zorder=0)
    for run in run_order:
        burden = summary_rows[run]["flagged_frac"]
        n_good = float(summary_df.loc[run, "n_good"])
        eff = float(summary_df.loc[run, "efficiency"])
        if not np.isfinite(burden):
            continue
        color = RUN_COLORS[run]
        ax_trade.scatter(burden, n_good, s=140 + 260 * eff, color=color, alpha=0.95,
                         edgecolors="white", linewidth=1.2, zorder=3)
        dx = 0.02 if run != "default" else -0.11
        dy = 0.25 if run != "peel1" else 0.1
        ha = "left" if run != "default" else "right"
        ax_trade.text(
            burden + dx,
            n_good + dy,
            f"{run}\neff={eff:.3f}",
            fontsize=7,
            color=color,
            ha=ha,
            va="bottom",
        )

    ax_trade.set_xlim(-0.02, 1.02)
    ax_trade.set_ylim(0, max(summary_df.loc[run_order, "n_good"].max() + 3, 24))
    ax_trade.set_xlabel("Flagged duplicate-like pair fraction")
    ax_trade.set_ylabel("KS4 good-labeled units")
    if include_claimmask:
        ax_trade.set_title("Claim-mask runs preserve downstream yield at much lower duplicate burden", loc="left")
    else:
        ax_trade.set_title("Cleaner duplicate burden aligns with cleaner downstream sorting", loc="left")
    style_axis(ax_trade)
    add_panel_label(ax_trade, "C")

    ax_trade.text(
        0.02,
        0.96,
        "Point size encodes efficiency",
        transform=ax_trade.transAxes,
        fontsize=7,
        color="#555555",
        ha="left",
        va="top",
    )

    # Panel D: exemplar CCGs.
    for ax, run in zip(ccg_axes, run_order):
        row = choose_exemplar_pair(screen_dfs[run])
        if row is None:
            ax.set_visible(False)
            continue
        by_unit = load_spikes_by_unit(run)
        unit_a = int(row["unit_a"])
        unit_b = int(row["unit_b"])
        centers, counts = fine_ccg_for_pair(by_unit[unit_a], by_unit[unit_b])
        color = RUN_COLORS[run]
        ax.bar(centers * 1e3, counts, width=FINE_CCG_BIN_S * 1e3, color=color, edgecolor="none", alpha=0.9)
        ax.axvline(0, color="#7A1F1F", ls="--", lw=0.9)
        ax.set_xlim(-FINE_CCG_WINDOW_S * 1e3, FINE_CCG_WINDOW_S * 1e3)
        ax.text(
            0.03,
            1.03,
            RUN_LABELS[run].replace("\n", " "),
            transform=ax.transAxes,
            fontsize=7,
            ha="left",
            va="bottom",
            color=color,
        )
        ax.set_xlabel("Lag (ms)")
        if ax is ccg_axes[0]:
            ax.set_ylabel("Pair count")
        else:
            ax.set_yticklabels([])
        ax.text(
            0.04,
            0.96,
            f"u{unit_a}×u{unit_b}\nnzf={row['near_zero_frac']:.3f}",
            transform=ax.transAxes,
            fontsize=6.5,
            ha="left",
            va="top",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FBFAF7", ec="none", alpha=0.9),
        )
        style_axis(ax)

    add_panel_label(ccg_axes[0], "D")
    ccg_axes[0].text(
        -0.02,
        1.18,
        "Exemplar nearby-pair CCGs show the same collapse at the raw signal level",
        transform=ccg_axes[0].transAxes,
        fontsize=9,
        ha="left",
        va="bottom",
        color="#111111",
    )

    if include_claimmask:
        suptitle = "Claim-mask intervention suppresses duplicate-like pair structure without sacrificing yield"
        subtitle = (
            "Pure peel reductions still show the expected burden collapse, but the cross-peel claim mask sits lower than default\n"
            "while preserving many more units than peel1, consistent with blocking re-claiming rather than disabling overlap recovery."
        )
    else:
        suptitle = "Later peeling passes create a monotonic, dose-dependent duplicate-like pair burden"
        subtitle = (
            "Within-run nearby-pair CCG structure collapses from default to peel3 to peel2 to peel1,\n"
            "linking extra residual passes directly to refractory-violating duplicate detections."
        )

    fig.suptitle(
        suptitle,
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )
    fig.text(0.055, 0.956, subtitle, ha="left", va="top", fontsize=8.5, color="#444444")

    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG)
    plt.close(fig)

    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()