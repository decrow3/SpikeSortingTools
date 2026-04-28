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
        "/mnt/NPX/Luke/20260302/dredge_pipeline_results_Luke03022026_V2V1_RH_g0_imec1/shallow_sweep_claimmask",
    )
)

DEFAULT_RUN = "default"
PEEL_RUN = "peel1"
CLAIM_RUN = "claim_spatial"
N_EXAMPLES = 2
MAX_EXPORT_EXAMPLES = int(os.environ.get("PLOT_MAX_EXPORT_EXAMPLES", "0"))
FS = 30_000.0
MATCH_TOL_S = 0.5e-3
TIME_BIN_S = 10.0
FINE_CCG_WINDOW_S = 5e-3
FINE_CCG_BIN_S = 0.2e-3
FINE_CCG_NEAR_ZERO_S = 0.5e-3
DUPLICATE_NEAR_ZERO_FRAC_THRESH = 0.05
DUPLICATE_ZERO_PEAK_RATIO_THRESH = 1.25

OUTPUT_PDF = SWEEP_DIR / "fig_claimmask_split_examples.pdf"
OUTPUT_PNG = SWEEP_DIR / "fig_claimmask_split_examples.png"
OUTPUT_DIR = SWEEP_DIR / "fig_claimmask_split_examples_gallery"
OUTPUT_INDEX_CSV = SWEEP_DIR / "fig_claimmask_split_examples_index.csv"

RUN_COLORS = {
    DEFAULT_RUN: "#8C2D1E",
    PEEL_RUN: "#1F4E5F",
    CLAIM_RUN: "#355C7D",
}
PAIR_COLORS = ["#B23A48", "#355C7D"]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titlepad": 5,
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


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#333333")


def add_panel_label(ax, label, x=-0.16, y=1.08):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#111111",
    )


def load_spikes_by_unit(run_name):
    sorter_out = SWEEP_DIR / f"run_{run_name}" / "kilosort4" / "sorter_output"
    spike_times = np.load(sorter_out / "spike_times.npy").astype(float) / FS
    spike_clusters = np.load(sorter_out / "spike_clusters.npy")
    by_unit = {}
    for uid in np.unique(spike_clusters):
        by_unit[int(uid)] = np.sort(spike_times[spike_clusters == uid])
    return by_unit


def load_labels(run_name):
    sorter_out = SWEEP_DIR / f"run_{run_name}" / "kilosort4" / "sorter_output"
    labels_df = pd.read_csv(sorter_out / "cluster_KSLabel.tsv", sep="\t")
    labels_df.columns = [c.strip() for c in labels_df.columns]
    return dict(zip(labels_df.iloc[:, 0].astype(int), labels_df.iloc[:, 1]))


def coincident_count(times_a_s, times_b_s, tol_s=MATCH_TOL_S):
    if len(times_a_s) == 0 or len(times_b_s) == 0:
        return 0
    ta = np.sort(np.asarray(times_a_s, float))
    tb = np.sort(np.asarray(times_b_s, float))
    idx = np.searchsorted(tb, ta)
    il = np.clip(idx - 1, 0, len(tb) - 1)
    ir = np.clip(idx, 0, len(tb) - 1)
    min_dist = np.minimum(np.abs(ta - tb[il]), np.abs(ta - tb[ir]))
    return int((min_dist <= tol_s).sum())


def top_matches(ref_times_s, other_by_unit, topk=3, min_spikes=100):
    rows = []
    ref_times_s = np.sort(np.asarray(ref_times_s, float))
    if len(ref_times_s) < min_spikes:
        return []
    for uid, other_times_s in other_by_unit.items():
        if len(other_times_s) < min_spikes:
            continue
        n_coinc = coincident_count(ref_times_s, other_times_s)
        if n_coinc == 0:
            continue
        rows.append(
            dict(
                unit=uid,
                coinc=n_coinc,
                ref_frac=n_coinc / len(ref_times_s),
                other_frac=n_coinc / len(other_times_s),
            )
        )
    rows.sort(key=lambda row: (row["ref_frac"], row["coinc"], row["other_frac"]), reverse=True)
    return rows[:topk]


def fine_ccg_for_pair(times_a_s, times_b_s):
    ta = np.sort(np.asarray(times_a_s, float))
    tb = np.sort(np.asarray(times_b_s, float))
    edges = np.arange(-FINE_CCG_WINDOW_S, FINE_CCG_WINDOW_S + FINE_CCG_BIN_S, FINE_CCG_BIN_S)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(ta) == 0 or len(tb) == 0:
        return dict(
            bin_centers_s=centers,
            counts=np.zeros(len(centers), dtype=int),
            total_pairs=0,
            near_zero_pairs=0,
            near_zero_frac=np.nan,
            zero_peak_ratio=np.nan,
        )

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
        counts = np.zeros(len(centers), dtype=int)
    else:
        counts, _ = np.histogram(np.concatenate(dts), bins=edges)

    nz_mask = np.abs(centers) <= FINE_CCG_NEAR_ZERO_S
    total_pairs = int(counts.sum())
    near_zero_pairs = int(counts[nz_mask].sum())
    near_zero_frac = (near_zero_pairs / total_pairs) if total_pairs else np.nan
    baseline = np.nanmean(counts[~nz_mask]) if (~nz_mask).any() else np.nan
    zero_peak = counts[np.argmin(np.abs(centers))] if len(counts) else np.nan
    zero_peak_ratio = (zero_peak / max(baseline, 1.0)) if total_pairs else np.nan
    return dict(
        bin_centers_s=centers,
        counts=counts,
        total_pairs=total_pairs,
        near_zero_pairs=near_zero_pairs,
        near_zero_frac=near_zero_frac,
        zero_peak_ratio=zero_peak_ratio,
    )


def pair_time_metrics(times_a_s, times_b_s, bins_s):
    c1, _ = np.histogram(times_a_s, bins=bins_s)
    c2, _ = np.histogram(times_b_s, bins=bins_s)
    union = c1 + c2
    frac1 = np.divide(c1, union, out=np.full_like(c1, np.nan, dtype=float), where=union > 0)
    valid = union > 0
    mid_bins = int(np.sum(valid & (frac1 >= 0.25) & (frac1 <= 0.75)))
    flips = int(np.sum(np.diff((frac1 > 0.5).astype(int)[valid]) != 0)) if valid.any() else 0
    segregation = float(np.nanmean(np.abs(c1 - c2) / np.maximum(union, 1))) if valid.any() else np.nan
    return c1, c2, union, frac1, mid_bins, flips, segregation


def collect_examples(default_screen_df, default_by_unit, peel1_by_unit, claim_by_unit, dedupe_families=False):
    duration_s = max(times[-1] for times in default_by_unit.values() if len(times))
    bins_s = np.arange(0, duration_s + TIME_BIN_S, TIME_BIN_S)

    flagged = default_screen_df[
        default_screen_df["near_zero_frac"].notna()
        & default_screen_df["zero_peak_ratio"].notna()
        & (default_screen_df["near_zero_frac"] >= DUPLICATE_NEAR_ZERO_FRAC_THRESH)
        & (default_screen_df["zero_peak_ratio"] >= DUPLICATE_ZERO_PEAK_RATIO_THRESH)
        & (default_screen_df["total_pairs"] > 0)
    ].copy()
    flagged["pair_score"] = flagged["near_zero_pairs"] * flagged["zero_peak_ratio"]
    flagged = flagged.sort_values("pair_score", ascending=False)

    candidates = []
    best_by_family = {}
    for _, row in flagged.iterrows():
        unit_a = int(row["unit_a"])
        unit_b = int(row["unit_b"])
        union_times_s = np.sort(np.concatenate([default_by_unit[unit_a], default_by_unit[unit_b]]))

        peel_matches = top_matches(union_times_s, peel1_by_unit)
        claim_matches = top_matches(union_times_s, claim_by_unit)
        if not peel_matches or not claim_matches:
            continue

        peel_top = peel_matches[0]
        claim_top = claim_matches[0]
        peel_second = peel_matches[1]["ref_frac"] if len(peel_matches) > 1 else 0.0
        claim_second = claim_matches[1]["ref_frac"] if len(claim_matches) > 1 else 0.0
        if peel_top["ref_frac"] < 0.55 or claim_top["ref_frac"] < 0.55:
            continue

        c1, c2, union, frac1, mid_bins, flips, segregation = pair_time_metrics(
            default_by_unit[unit_a], default_by_unit[unit_b], bins_s
        )
        quality = min(peel_top["ref_frac"], claim_top["ref_frac"]) - 0.5 * max(peel_second, claim_second)
        family_key = (peel_top["unit"], claim_top["unit"])
        candidate = dict(
            unit_a=unit_a,
            unit_b=unit_b,
            peel_unit=peel_top["unit"],
            claim_unit=claim_top["unit"],
            peel_ref_frac=peel_top["ref_frac"],
            peel_second_frac=peel_second,
            claim_ref_frac=claim_top["ref_frac"],
            claim_second_frac=claim_second,
            near_zero_frac=float(row["near_zero_frac"]),
            zero_peak_ratio=float(row["zero_peak_ratio"]),
            pair_score=float(row["pair_score"]),
            mid_bins=mid_bins,
            flips=flips,
            segregation=segregation,
            quality=quality,
        )
        rank = (mid_bins, flips, quality, row["pair_score"])
        candidate["family_key"] = f"{family_key[0]}->{family_key[1]}"
        candidate["_rank"] = rank
        candidates.append(candidate)

        prev = best_by_family.get(family_key)
        if prev is None or rank > prev["_rank"]:
            best_by_family[family_key] = candidate

    selected = best_by_family.values() if dedupe_families else candidates
    selected = sorted(selected, key=lambda row: row["_rank"], reverse=True)
    if not selected:
        raise RuntimeError("Could not find default split examples that collapse in peel1 and claim_spatial")
    return selected, bins_s


def format_example_name(example, rank):
    return (
        f"rank_{rank:03d}"
        f"__default_u{example['unit_a']}_u{example['unit_b']}"
        f"__{PEEL_RUN}_u{example['peel_unit']}"
        f"__{CLAIM_RUN}_u{example['claim_unit']}"
    )


def draw_example_row(fig, subplot_spec, example, bins_s,
                     default_by_unit, peel_by_unit, claim_by_unit,
                     default_labels, peel_labels, claim_labels,
                     show_panel_labels=False):
    t_mid_min = (bins_s[:-1] + bins_s[1:]) / 2 / 60
    sub = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subplot_spec,
        width_ratios=[1.9, 1.05, 1.05],
        height_ratios=[1.15, 0.95],
        hspace=0.30,
        wspace=0.28,
    )

    ax_counts = fig.add_subplot(sub[0, 0], facecolor="#FBFAF7")
    ax_ccg = fig.add_subplot(sub[0, 1], facecolor="#FBFAF7")
    ax_frac = fig.add_subplot(sub[1, 0], facecolor="#FBFAF7")
    ax_peel = fig.add_subplot(sub[1, 1], facecolor="#FBFAF7")
    ax_claim = fig.add_subplot(sub[1, 2], facecolor="#FBFAF7")

    unit_a = example["unit_a"]
    unit_b = example["unit_b"]
    peel_unit = example["peel_unit"]
    claim_unit = example["claim_unit"]

    default_a = default_by_unit[unit_a]
    default_b = default_by_unit[unit_b]
    peel_times = peel_by_unit[peel_unit]
    claim_times = claim_by_unit[claim_unit]
    c1, c2, union, frac1, _, _, _ = pair_time_metrics(default_a, default_b, bins_s)
    peel_counts, _ = np.histogram(peel_times, bins=bins_s)
    claim_counts, _ = np.histogram(claim_times, bins=bins_s)

    ax_counts.plot(t_mid_min, union, color="#111111", lw=1.3, ls="--", label="default union")
    ax_counts.plot(t_mid_min, c1, color=PAIR_COLORS[0], lw=1.1, label=f"default u{unit_a}")
    ax_counts.plot(t_mid_min, c2, color=PAIR_COLORS[1], lw=1.1, label=f"default u{unit_b}")
    ax_counts.set_ylabel(f"Counts / {TIME_BIN_S:.0f}s")
    ax_counts.set_xlabel("Time (min)")
    ax_counts.legend(loc="upper right", ncol=2)
    ax_counts.set_title(
        f"Default split pair: u{unit_a} ({default_labels.get(unit_a, 'unknown')}) and u{unit_b} ({default_labels.get(unit_b, 'unknown')})",
        loc="left",
    )

    ax_frac.plot(t_mid_min, frac1, color="#444444", lw=1.1)
    ax_frac.axhline(0.5, color="#AAAAAA", lw=0.8, ls=":")
    ax_frac.set_ylim(0, 1)
    ax_frac.set_ylabel(f"u{unit_a} fraction")
    ax_frac.set_xlabel("Time (min)")
    ax_frac.set_title(
        f"Within-default handoff: seg={example['segregation']:.2f}, flips={example['flips']}, mixed bins={example['mid_bins']}",
        loc="left",
        fontsize=8,
    )

    ccg = fine_ccg_for_pair(default_a, default_b)
    ax_ccg.bar(ccg["bin_centers_s"] * 1e3, ccg["counts"], width=FINE_CCG_BIN_S * 1e3,
               color=RUN_COLORS[DEFAULT_RUN], edgecolor="none")
    ax_ccg.axvline(0, color="#7A1F1F", ls="--", lw=0.9)
    ax_ccg.set_xlim(-FINE_CCG_WINDOW_S * 1e3, FINE_CCG_WINDOW_S * 1e3)
    ax_ccg.set_xlabel("Lag (ms)")
    ax_ccg.set_ylabel("Pair count")
    ax_ccg.set_title(
        f"Default CCG\nnzf={example['near_zero_frac']:.3f}, peak/base={example['zero_peak_ratio']:.1f}",
        color=RUN_COLORS[DEFAULT_RUN],
    )

    ax_peel.plot(t_mid_min, union, color="#777777", lw=1.2, ls="--", label="default union")
    ax_peel.plot(t_mid_min, peel_counts, color=RUN_COLORS[PEEL_RUN], lw=1.3, label=f"{PEEL_RUN} u{peel_unit}")
    ax_peel.set_xlabel("Time (min)")
    ax_peel.set_ylabel(f"Counts / {TIME_BIN_S:.0f}s")
    ax_peel.set_title(f"{PEEL_RUN} collapse to u{peel_unit} ({peel_labels.get(peel_unit, 'unknown')})",
                      color=RUN_COLORS[PEEL_RUN], fontsize=8.5)
    ax_peel.text(
        0.03,
        0.97,
        f"match={example['peel_ref_frac']:.2f}, second={example['peel_second_frac']:.2f}",
        transform=ax_peel.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=RUN_COLORS[PEEL_RUN],
        bbox=dict(boxstyle="round,pad=0.2", fc="#FBFAF7", ec="none", alpha=0.9),
    )

    ax_claim.plot(t_mid_min, union, color="#777777", lw=1.2, ls="--", label="default union")
    ax_claim.plot(t_mid_min, claim_counts, color=RUN_COLORS[CLAIM_RUN], lw=1.3, label=f"{CLAIM_RUN} u{claim_unit}")
    ax_claim.set_xlabel("Time (min)")
    ax_claim.set_ylabel(f"Counts / {TIME_BIN_S:.0f}s")
    ax_claim.set_title(f"{CLAIM_RUN} collapse to u{claim_unit} ({claim_labels.get(claim_unit, 'unknown')})",
                       color=RUN_COLORS[CLAIM_RUN], fontsize=8.5)
    ax_claim.text(
        0.03,
        0.97,
        f"match={example['claim_ref_frac']:.2f}, second={example['claim_second_frac']:.2f}",
        transform=ax_claim.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=RUN_COLORS[CLAIM_RUN],
        bbox=dict(boxstyle="round,pad=0.2", fc="#FBFAF7", ec="none", alpha=0.9),
    )

    if show_panel_labels:
        add_panel_label(ax_counts, "A")
        add_panel_label(ax_ccg, "B")
        add_panel_label(ax_peel, "C", x=-0.24, y=1.10)
    for ax in [ax_counts, ax_ccg, ax_frac, ax_peel, ax_claim]:
        style_axis(ax)


def export_example_gallery(examples, bins_s,
                           default_by_unit, peel_by_unit, claim_by_unit,
                           default_labels, peel_labels, claim_labels):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_examples = examples[:MAX_EXPORT_EXAMPLES] if MAX_EXPORT_EXAMPLES > 0 else examples
    print(f"Preparing to export {len(export_examples)} single-example figures to {OUTPUT_DIR}", flush=True)

    index_rows = []
    for rank, example in enumerate(export_examples, start=1):
        stem = format_example_name(example, rank)
        out_pdf = OUTPUT_DIR / f"{stem}.pdf"
        out_png = OUTPUT_DIR / f"{stem}.png"

        fig = plt.figure(figsize=(12.0, 4.8), facecolor="#F7F4EE")
        outer = gridspec.GridSpec(1, 1, figure=fig)
        draw_example_row(
            fig,
            outer[0],
            example,
            bins_s,
            default_by_unit,
            peel_by_unit,
            claim_by_unit,
            default_labels,
            peel_labels,
            claim_labels,
            show_panel_labels=True,
        )

        fig.suptitle(
            f"Default split pair u{example['unit_a']}-u{example['unit_b']} collapses under {PEEL_RUN} and {CLAIM_RUN}",
            x=0.055,
            y=0.992,
            ha="left",
            fontsize=13,
            fontweight="bold",
            color="#111111",
        )
        fig.text(
            0.055,
            0.965,
            f"Rank {rank}: default nzf={example['near_zero_frac']:.3f}, peak/base={example['zero_peak_ratio']:.1f}, "
            f"peel1 match={example['peel_ref_frac']:.2f}, claim_spatial match={example['claim_ref_frac']:.2f}",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#444444",
        )
        fig.savefig(out_pdf)
        fig.savefig(out_png)
        plt.close(fig)

        if rank == 1 or rank % 10 == 0 or rank == len(export_examples):
            print(f"Exported {rank}/{len(export_examples)} examples", flush=True)

        row = {k: v for k, v in example.items() if not k.startswith("_")}
        row.update(rank=rank, file_png=out_png.name, file_pdf=out_pdf.name)
        index_rows.append(row)

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(OUTPUT_INDEX_CSV, index=False)
    return len(export_examples)


def export_combined_figure(examples, bins_s,
                           default_by_unit, peel_by_unit, claim_by_unit,
                           default_labels, peel_labels, claim_labels):
    overview_examples = examples[:N_EXAMPLES]
    fig = plt.figure(figsize=(12.0, 4.8 * len(overview_examples)), facecolor="#F7F4EE")
    outer = gridspec.GridSpec(len(overview_examples), 1, figure=fig, hspace=0.36)

    for row_idx, example in enumerate(overview_examples):
        draw_example_row(
            fig,
            outer[row_idx],
            example,
            bins_s,
            default_by_unit,
            peel_by_unit,
            claim_by_unit,
            default_labels,
            peel_labels,
            claim_labels,
            show_panel_labels=(row_idx == 0),
        )

    fig.suptitle(
        "Concrete split examples collapse under peel1 and the cross-peel claim mask",
        x=0.055,
        y=0.992,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.055,
        0.965,
        "Each row starts from a default within-run duplicate pair chosen from the flagged nearby-pair screen, then shows\n"
        "that the same activity is captured by a single unit in both peel1 and claim_spatial rather than remaining over-split.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#444444",
    )

    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG)
    plt.close(fig)


def main():
    print(f"Loading split-example inputs from {SWEEP_DIR}", flush=True)
    default_screen_df = pd.read_csv(SWEEP_DIR / f"within_run_screen_{DEFAULT_RUN}.csv")
    default_by_unit = load_spikes_by_unit(DEFAULT_RUN)
    peel_by_unit = load_spikes_by_unit(PEEL_RUN)
    claim_by_unit = load_spikes_by_unit(CLAIM_RUN)
    default_labels = load_labels(DEFAULT_RUN)
    peel_labels = load_labels(PEEL_RUN)
    claim_labels = load_labels(CLAIM_RUN)

    print("Collecting candidate split examples...", flush=True)
    all_examples, bins_s = collect_examples(default_screen_df, default_by_unit, peel_by_unit, claim_by_unit)
    overview_examples, _ = collect_examples(default_screen_df, default_by_unit, peel_by_unit, claim_by_unit,
                                            dedupe_families=True)
    print(f"Found {len(all_examples)} total exportable examples and {len(overview_examples)} deduped overview families", flush=True)

    export_count = export_example_gallery(
        all_examples,
        bins_s,
        default_by_unit,
        peel_by_unit,
        claim_by_unit,
        default_labels,
        peel_labels,
        claim_labels,
    )
    export_combined_figure(
        overview_examples,
        bins_s,
        default_by_unit,
        peel_by_unit,
        claim_by_unit,
        default_labels,
        peel_labels,
        claim_labels,
    )

    chosen = [
        f"u{row['unit_a']}-u{row['unit_b']} -> {PEEL_RUN} u{row['peel_unit']} / {CLAIM_RUN} u{row['claim_unit']}"
        for row in overview_examples[:N_EXAMPLES]
    ]
    print(f"Exported {export_count} single-example figures to {OUTPUT_DIR}")
    print(f"Wrote example index CSV → {OUTPUT_INDEX_CSV}")
    print("Top combined-figure examples:")
    for line in chosen:
        print(f"  {line}")
    print(f"Saved {OUTPUT_PDF}")
    print(f"Saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()