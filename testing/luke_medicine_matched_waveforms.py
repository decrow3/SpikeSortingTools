"""Inspect cached physical-channel templates for matched RESCUE/MEDiCINe units.

This analysis uses only completed Kilosort outputs. It undoes Kilosort whitening,
compares waveforms on the shared physical-channel geometry, summarizes all
shared-interior reciprocal matches, and renders a stratified six-pair gallery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/cur/cur_output")
CAND = Path("/media/huklab/Data/luke_medicine_rigid_sort_20260909_v1/cur/cur_output")
PAIRS = Path("testing/outputs/luke_medicine_rigid_vs_rescue_fair_v1/reciprocal_primary_matches.csv")
DEPTH_RANGE_UM = (300.0, 3540.0)
TIME_LAGS = range(-3, 4)
DEPTH_SHIFTS_UM = range(-100, 101, 20)


def dewhiten_selected(root: Path, cluster_ids: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    templates = np.load(root / "templates.npy", mmap_mode="r")
    inverse = np.load(root / "whitening_mat_inv.npy")
    positions = np.load(root / "channel_positions.npy")
    selected = np.asarray(templates[np.asarray(cluster_ids, dtype=int)], dtype=np.float64)
    physical = selected @ np.asarray(inverse, dtype=np.float64)
    return {int(cid): wave for cid, wave in zip(cluster_ids, physical)}, positions.astype(float)


def _overlap_time(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag < 0:
        return a[-lag:], b[:lag]
    if lag > 0:
        return a[:-lag], b[lag:]
    return a, b


def aligned_similarity(
    baseline: np.ndarray,
    candidate: np.ndarray,
    baseline_positions: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    max_radius_um: float = 160.0,
) -> dict[str, float | int]:
    """Find the best geometry-preserving depth shift and small temporal lag."""
    b_ptp = np.ptp(baseline, axis=0)
    c_ptp = np.ptp(candidate, axis=0)
    b_peak = int(np.argmax(b_ptp))
    c_peak = int(np.argmax(c_ptp))
    b_lookup = {(float(x), float(y)): i for i, (x, y) in enumerate(baseline_positions)}
    best = None
    absolute = None
    for depth_shift in DEPTH_SHIFTS_UM:
        pairs = []
        center = float(baseline_positions[b_peak, 1])
        for ci, (x, y) in enumerate(candidate_positions):
            bi = b_lookup.get((float(x), float(y + depth_shift)))
            if bi is not None and abs(float(baseline_positions[bi, 1]) - center) <= max_radius_um:
                pairs.append((bi, ci))
        if len(pairs) < 8:
            continue
        bi = np.array([p[0] for p in pairs])
        ci = np.array([p[1] for p in pairs])
        for lag in TIME_LAGS:
            bw, cw = _overlap_time(baseline[:, bi], candidate[:, ci], lag)
            bv, cv = bw.reshape(-1), cw.reshape(-1)
            denom = np.linalg.norm(bv) * np.linalg.norm(cv)
            cosine = float(np.dot(bv, cv) / denom) if denom else np.nan
            gain = float(np.dot(bv, cv) / np.dot(bv, bv)) if np.dot(bv, bv) else np.nan
            record = {"waveform_cosine": cosine, "depth_shift_um": int(depth_shift),
                      "time_lag_samples": int(lag), "candidate_on_baseline_gain": gain,
                      "channels_compared": int(len(pairs))}
            if depth_shift == 0 and (absolute is None or cosine > absolute["waveform_cosine"]):
                absolute = record
            if best is None or cosine > best["waveform_cosine"]:
                best = record
    if best is None:
        raise RuntimeError("insufficient shared geometry for waveform comparison")
    best["absolute_coordinate_cosine"] = None if absolute is None else absolute["waveform_cosine"]
    best["baseline_peak_depth_um"] = float(baseline_positions[b_peak, 1])
    best["candidate_peak_depth_um"] = float(candidate_positions[c_peak, 1])
    best["peak_depth_difference_um"] = float(candidate_positions[c_peak, 1] - baseline_positions[b_peak, 1])
    best["peak_to_peak_ratio"] = float(c_ptp[c_peak] / b_ptp[b_peak]) if b_ptp[b_peak] else np.nan
    return best


def select_examples(pairs: pd.DataFrame) -> pd.DataFrame:
    selected: list[tuple[str, pd.Series]] = []
    used: set[tuple[int, int]] = set()

    def add(reason: str, candidates: pd.DataFrame, *, sort_column: str = "jaccard",
            ascending: bool = False, middle: bool = False) -> None:
        ordered = candidates.sort_values(sort_column, ascending=ascending)
        rows = ordered.iloc[[len(ordered) // 2]] if middle else ordered
        for _, row in rows.iterrows():
            key = (int(row.baseline_cluster), int(row.candidate_cluster))
            if key not in used:
                used.add(key)
                selected.append((reason, row))
                return
        raise RuntimeError(f"could not select a distinct example for {reason}")

    good_good = pairs[pairs.baseline_label.eq("good") & pairs.candidate_label.eq("good")]
    add("strongest good→good", good_good)
    add("median good→good", good_good, middle=True)
    add("representative good→MUA", pairs[pairs.baseline_label.eq("good") & pairs.candidate_label.eq("mua")], middle=True)
    add("largest refractory increase", pairs[pairs.baseline_label.eq("good")],
        sort_column="candidate_minus_baseline_refractory_violation_fraction_1_5ms")
    add("largest amplitude-CV increase", pairs[pairs.baseline_label.eq("good")],
        sort_column="candidate_minus_baseline_amplitude_cv")
    add("weakest reciprocal match", pairs, ascending=True)
    add("waveform/spatial outlier", pairs, sort_column="waveform_cosine", ascending=True)
    out = pd.DataFrame([dict(reason=reason, **row.to_dict()) for reason, row in selected])
    return out


def plot_geometry_waveform(ax, wave: np.ndarray, positions: np.ndarray, center_depth: float,
                           scale: float, color: str, title: str) -> None:
    keep = np.flatnonzero(np.abs(positions[:, 1] - center_depth) <= 140)
    t = np.linspace(-0.75, 1.25, wave.shape[0])
    for channel in keep:
        x, y = positions[channel]
        ax.plot(x + t * 12, y + wave[:, channel] * scale, color=color, lw=0.75)
    ax.set_xlim(-18, 78)
    ax.set_ylim(center_depth - 165, center_depth + 165)
    ax.set_xticks([])
    ax.set_ylabel("depth (µm)", fontsize=7)
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(axis="y", color="#e5e7eb", lw=0.5)


def render_gallery(selected: pd.DataFrame, bbank: dict[int, np.ndarray], cbank: dict[int, np.ndarray],
                   bpos: np.ndarray, cpos: np.ndarray, output: Path) -> None:
    fig, axes = plt.subplots(len(selected), 3, figsize=(12, 17), constrained_layout=True)
    for row_index, row in enumerate(selected.itertuples(index=False)):
        bw = bbank[int(row.baseline_cluster)]
        cw = cbank[int(row.candidate_cluster)]
        bpeak = int(np.argmax(np.ptp(bw, axis=0)))
        cpeak = int(np.argmax(np.ptp(cw, axis=0)))
        common_ptp = max(float(np.ptp(bw[:, bpeak])), float(np.ptp(cw[:, cpeak])))
        scale = 32.0 / common_ptp if common_ptp else 1.0
        plot_geometry_waveform(axes[row_index, 0], bw, bpos, bpos[bpeak, 1], scale, "#1f5a94",
                               f"RESCUE u{int(row.baseline_cluster)} ({row.baseline_label})")
        plot_geometry_waveform(axes[row_index, 1], cw, cpos, cpos[cpeak, 1], scale, "#c56a1a",
                               f"MEDiCINe u{int(row.candidate_cluster)} ({row.candidate_label})")
        ax = axes[row_index, 2]
        t = (np.arange(bw.shape[0]) - np.argmin(bw[:, bpeak])) / 30.0
        bt = bw[:, bpeak] / (np.max(np.abs(bw[:, bpeak])) or 1.0)
        ct = cw[:, cpeak] / (np.max(np.abs(cw[:, cpeak])) or 1.0)
        ct = np.roll(ct, -int(row.time_lag_samples))
        ax.plot(t, bt, color="#1f5a94", lw=1.5, label="RESCUE")
        ax.plot(t, ct, color="#c56a1a", lw=1.5, ls="--", label="MEDiCINe")
        ax.axhline(0, color="#9ca3af", lw=0.6)
        ax.set_title(f"{row.reason}\nJ={row.jaccard:.3f}; shape cos={row.waveform_cosine:.3f}; Δy={int(row.depth_shift_um):+d} µm", fontsize=9, loc="left")
        ax.set_xlabel("time from trough (ms)", fontsize=7)
        ax.set_ylabel("normalized voltage", fontsize=7)
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.tick_params(labelsize=7)
    fig.suptitle("Matched-unit physical-channel templates\nCommon 300–3540 µm cohort; pairwise shared amplitude scale in geometry panels", fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_summary(metrics: pd.DataFrame, selected: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    colors = np.where(metrics.baseline_label.eq("good") & metrics.candidate_label.eq("good"), "#1f5a94",
                      np.where(metrics.baseline_label.eq("good"), "#c56a1a", "#9ca3af"))
    axes[0].scatter(metrics.jaccard, metrics.waveform_cosine, c=colors, s=26, alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[0].set(xlabel="spike-train Jaccard", ylabel="best peak-relative waveform cosine",
                title="Timing agreement versus waveform shape")
    axes[1].scatter(metrics.peak_depth_difference_um, metrics.waveform_cosine, c=colors, s=26, alpha=0.8,
                    edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="#6b7280", lw=0.8)
    axes[1].set(xlabel="candidate − baseline peak depth (µm)", ylabel="best peak-relative waveform cosine",
                title="Peak displacement versus waveform shape")
    lookup = {(int(r.baseline_cluster), int(r.candidate_cluster)): r for r in selected.itertuples(index=False)}
    for ax in axes:
        ax.grid(color="#e5e7eb", lw=0.6)
        ax.tick_params(labelsize=8)
        for row in metrics.itertuples(index=False):
            key = (int(row.baseline_cluster), int(row.candidate_cluster))
            if key in lookup:
                ax.annotate(str(list(lookup).index(key) + 1),
                            (row.jaccard, row.waveform_cosine) if ax is axes[0] else (row.peak_depth_difference_um, row.waveform_cosine),
                            fontsize=7, xytext=(3, 3), textcoords="offset points")
    fig.suptitle("All 77 shared-interior reciprocal matches\nblue: good→good; orange: good→MUA; gray: MUA→MUA", fontsize=12)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(PAIRS)
    lo, hi = DEPTH_RANGE_UM
    pairs = pairs[pairs.baseline_median_depth_um.between(lo, hi) & pairs.candidate_median_depth_um.between(lo, hi)].copy()
    b_ids = pairs.baseline_cluster.astype(int).unique()
    c_ids = pairs.candidate_cluster.astype(int).unique()
    bbank, bpos = dewhiten_selected(BASE, b_ids)
    cbank, cpos = dewhiten_selected(CAND, c_ids)
    waveform_rows = []
    for row in pairs.itertuples(index=False):
        values = aligned_similarity(bbank[int(row.baseline_cluster)], cbank[int(row.candidate_cluster)], bpos, cpos)
        waveform_rows.append({"baseline_cluster": int(row.baseline_cluster), "candidate_cluster": int(row.candidate_cluster), **values})
    metrics = pairs.merge(pd.DataFrame(waveform_rows), on=["baseline_cluster", "candidate_cluster"], validate="one_to_one")
    selected = select_examples(metrics)
    metrics.to_csv(output / "matched_waveform_metrics.csv", index=False)
    selected.to_csv(output / "selected_waveform_pairs.csv", index=False)
    render_gallery(selected, bbank, cbank, bpos, cpos, output / "matched_waveform_gallery.png")
    render_summary(metrics, selected, output / "matched_waveform_summary.png")
    summary = {
        "schema_version": "luke-medicine-matched-waveforms-v1",
        "pairs": int(len(metrics)),
        "waveform_representation": "templates.npy @ whitening_mat_inv.npy on physical channels",
        "common_depth_range_um": list(DEPTH_RANGE_UM),
        "alignment_search": {"depth_shifts_um": [min(DEPTH_SHIFTS_UM), max(DEPTH_SHIFTS_UM), 20],
                             "time_lags_samples": [min(TIME_LAGS), max(TIME_LAGS)]},
        "median_best_waveform_cosine": float(metrics.waveform_cosine.median()),
        "median_absolute_coordinate_cosine": float(metrics.absolute_coordinate_cosine.median()),
        "fraction_best_waveform_cosine_ge_0_9": float(metrics.waveform_cosine.ge(0.9).mean()),
        "fraction_best_waveform_cosine_ge_0_8": float(metrics.waveform_cosine.ge(0.8).mean()),
        "median_peak_depth_difference_um": float(metrics.peak_depth_difference_um.median()),
        "median_best_depth_shift_um": float(metrics.depth_shift_um.median()),
        "median_peak_to_peak_ratio": float(metrics.peak_to_peak_ratio.median()),
        "fraction_peak_to_peak_ratio_below_1": float(metrics.peak_to_peak_ratio.lt(1).mean()),
        "label_transition_summary": {
            transition: {
                "pairs": int(len(frame)),
                "median_waveform_cosine": float(frame.waveform_cosine.median()),
                "median_peak_to_peak_ratio": float(frame.peak_to_peak_ratio.median()),
                "median_refractory_change": float(frame.candidate_minus_baseline_refractory_violation_fraction_1_5ms.median()),
                "median_amplitude_cv_change": float(frame.candidate_minus_baseline_amplitude_cv.median()),
            }
            for transition, frame in metrics.assign(
                transition=metrics.baseline_label + "→" + metrics.candidate_label
            ).groupby("transition")
        },
        "selected_pairs": selected[["reason", "baseline_cluster", "candidate_cluster", "baseline_label",
                                     "candidate_label", "jaccard", "waveform_cosine", "depth_shift_um",
                                     "time_lag_samples", "peak_to_peak_ratio"]].to_dict("records"),
        "limitations": [
            "Templates are sorter averages, not fresh raw-voltage snippets or within-session waveform distributions.",
            "Peak-relative cosine is optimized over ±100 µm and ±3 samples and is descriptive, not an identity test.",
            "Each sort has its own whitening estimate; inverse whitening returns physical channel space but does not remove preprocessing differences.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
