"""Cheap agreement check across the 16 distinct prior-lighthouse premerge clusters."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1"
PRE = ROOT / "testing/outputs/luke_medicine_motion_aware_merge_v1"
OUT = ROOT / "testing/outputs/luke_premerge_lighthouse_agreement_v1"
FS = 29999.835983263598
START_S, STOP_S, BIN_S = 930.0, 1030.0, 5.0


def correlation(x, y, minimum=4):
    keep = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[keep], y[keep])[0, 1]) if keep.sum() >= minimum else np.nan


def agreement_stats(matrix):
    pairwise = [correlation(matrix[i], matrix[j]) for i in range(len(matrix))
                for j in range(i + 1, len(matrix))]
    consensus = np.nanmedian(matrix, axis=0)
    leave_one_out = [correlation(matrix[i], np.nanmedian(np.delete(matrix, i, axis=0), axis=0))
                     for i in range(len(matrix))]
    return np.asarray(pairwise), consensus, np.asarray(leave_one_out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reconciliation = pd.read_csv(RECON / "prior_unit_reconciliation.csv")
    cluster_units = (reconciliation.groupby("dominant_premerge_cluster")["unit_id"]
                     .apply(lambda x: "/".join(map(str, sorted(x)))).to_dict())
    cids = np.array(sorted(cluster_units), dtype=int)
    st = np.load(PRE / "bounded_premerge/st.npy", mmap_mode="r")
    clu = np.asarray(np.load(PRE / "bounded_premerge/clu.npy", mmap_mode="r"), int)
    depth = np.asarray(np.load(PRE / "audit/observed_depth_um.npy", mmap_mode="r"), float)
    time_s = np.asarray(st[:, 0], float) / FS
    edges = np.arange(START_S, STOP_S + BIN_S, BIN_S)
    centers = (edges[:-1] + edges[1:]) / 2
    tracks = []
    for cid in cids:
        values = []
        for left, right in zip(edges[:-1], edges[1:]):
            take = (clu == cid) & (time_s >= left) & (time_s < right)
            values.append(np.median(depth[take]) if take.any() else np.nan)
        values = np.asarray(values)
        tracks.append(values - np.nanmedian(values))
    matrix = np.asarray(tracks)
    pairwise, consensus, leave_one_out = agreement_stats(matrix)

    rng = np.random.default_rng(20260909)
    null = []
    for _ in range(500):
        shifted = np.asarray([np.roll(row, int(rng.integers(row.size))) for row in matrix])
        p, c, loo = agreement_stats(shifted)
        null.append([np.nanmedian(p), np.nanmedian(loo), np.ptp(c)])
    null = np.asarray(null)
    observed = np.array([np.nanmedian(pairwise), np.nanmedian(leave_one_out), np.ptp(consensus)])
    p_upper = (1 + np.sum(null >= observed, axis=0)) / (len(null) + 1)

    rows = []
    for cid, row, loo in zip(cids, matrix, leave_one_out):
        rows.append({"premerge_cluster": cid, "prior_unit_labels": cluster_units[cid],
                     "observed_5s_bins": int(np.isfinite(row).sum()),
                     "displacement_span_um": float(np.nanmax(row) - np.nanmin(row)),
                     "leave_one_out_consensus_r": loo})
    pd.DataFrame(rows).to_csv(OUT / "per_cluster_agreement.csv", index=False)
    pd.DataFrame({"time_s": centers, "consensus_relative_depth_um": consensus,
                  "contributing_clusters": np.isfinite(matrix).sum(axis=0)}).to_csv(
                      OUT / "consensus_track.csv", index=False)

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[1.7, 1], layout="constrained")
    for row in matrix:
        ax.plot(centers, row, color="#8A939D", alpha=0.42, lw=1)
    ax.plot(centers, consensus, color="#2166AC", marker="o", lw=2.6,
            label="median across available clusters")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set(title="Centered premerge depth tracks", ylabel="relative depth (um)",
           xlim=(START_S, STOP_S))
    ax.legend(frameon=False)
    labels = [cluster_units[cid] for cid in cids]
    order = np.argsort(leave_one_out)
    colors = ["#2166AC" if value >= 0 else "#B7BDC5" for value in leave_one_out[order]]
    bx.barh(np.asarray(labels)[order], leave_one_out[order], color=colors)
    bx.axvline(0, color="#333333", lw=0.8)
    bx.set(title="Agreement with median of the other 15 clusters",
           xlabel="leave-one-out Pearson correlation", ylabel="prior unit label(s)")
    fig.suptitle("Shared movement among 16 distinct prior-lighthouse premerge representations\n"
                 "930-1030 s; 5-s medians; each cluster centered independently")
    fig.savefig(OUT / "01_shared_movement_agreement.png", dpi=180)
    fig.savefig(OUT / "01_shared_movement_agreement.pdf")
    plt.close(fig)

    summary = {
        "distinct_premerge_clusters": int(len(cids)),
        "median_pairwise_pearson_r": float(observed[0]),
        "positive_pairwise_fraction": float(np.mean(pairwise > 0)),
        "median_leave_one_out_consensus_r": float(observed[1]),
        "positive_leave_one_out_fraction": float(np.mean(leave_one_out > 0)),
        "leave_one_out_r_above_0p3_count": int(np.sum(leave_one_out > 0.3)),
        "consensus_peak_to_peak_um": float(observed[2]),
        "circular_shift_draws": 500,
        "circular_shift_p_upper": {
            "median_pairwise_r": float(p_upper[0]),
            "median_leave_one_out_r": float(p_upper[1]),
            "consensus_peak_to_peak": float(p_upper[2]),
        },
        "interpretation": "Weak common component, not strong pairwise agreement.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
