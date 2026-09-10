"""Run the cheap depth-blind lighthouse screen on Kilosort premerge clusters.

The preserved checkpoint ends after ``clustering_qr.run(mode='template')`` and
before ``template_matching.merging_function``.  This script reconstructs each
cluster's temporal waveform from ``Wall @ wPCA``, forms relative-channel
waveform graphs at 0.95/0.97/0.98, and only then reveals cached spike depth and
time.  It reads no voltage and launches no sorter.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing import luke_kilosort_depthblind_tracklet_linkage_v1 as v1
from testing import luke_kilosort_family_plausibility_v3 as v3
from testing import luke_kilosort_family_threshold_sweep_v4 as v4


CHECKPOINT = ROOT / "testing/outputs/luke_medicine_motion_aware_merge_v1"
PREMERGE = CHECKPOINT / "bounded_premerge"
AUDIT = CHECKPOINT / "audit"
OUT = ROOT / "testing/outputs/luke_kilosort_premerge_lighthouse_v5"
STATIC_CLU = CHECKPOINT / "identity_controls/static_clu.npy"
THRESHOLDS = v4.THRESHOLDS
INTERVAL_S = (930.0, 1030.0)
MIN_SNIPPET_SPIKES = 20
MIN_LOCAL_ENERGY_FRACTION = 0.80


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_temporal_templates(wall: np.ndarray, w_pca: np.ndarray) -> np.ndarray:
    """Convert cluster x channel x PC coefficients into cluster x time x channel."""
    wall = np.asarray(wall, dtype=np.float32)
    w_pca = np.asarray(w_pca, dtype=np.float32)
    if wall.ndim != 3 or w_pca.ndim != 2 or wall.shape[2] != w_pca.shape[0]:
        raise ValueError("Wall and wPCA dimensions disagree")
    return np.einsum("ncp,pt->ntc", wall, w_pca, optimize=True)


def relative_waveform_inventory(
    templates: np.ndarray, positions: np.ndarray, event_cids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    active, counts = np.unique(event_cids, return_counts=True)
    rows = []
    waves = []
    for cid, count in zip(active.astype(int), counts.astype(int)):
        if count < MIN_SNIPPET_SPIKES:
            continue
        full = np.asarray(templates[cid], dtype=np.float32)
        peak_sample, peak_channel = np.unravel_index(np.argmax(np.abs(full)), full.shape)
        peak_depth = float(positions[peak_channel, 1])
        patch_origin = 40.0 * np.floor(peak_depth / 40.0)
        keep = np.flatnonzero(
            (positions[:, 1] >= patch_origin + v1.PATCH_Y_MIN_UM - 1e-6)
            & (positions[:, 1] <= patch_origin + v1.PATCH_Y_MAX_UM + 1e-6)
        )
        keep = keep[np.lexsort((positions[keep, 0], positions[keep, 1] - patch_origin))]
        total_energy = float(np.sum(full.astype(np.float64) ** 2))
        local_energy = float(np.sum(full[:, keep].astype(np.float64) ** 2)) if len(keep) else 0.0
        local_fraction = local_energy / total_energy if total_energy > 0 else np.nan
        reason = "eligible"
        if len(keep) != 16:
            reason = "incomplete_probe_edge_patch"
        elif not np.isfinite(local_fraction) or local_fraction < MIN_LOCAL_ENERGY_FRACTION:
            reason = "diffuse_template"
        rows.append({
            "cid": cid,
            "snippet_spikes": count,
            "KSLabel": "premerge_unlabelled",
            "eligibility": reason,
            "local_energy_fraction": local_fraction,
            "template_peak_abs_ks_units": float(np.max(np.abs(full))),
            "peak_sample_full": int(peak_sample),
            "peak_channel": int(peak_channel),
            "template_peak_x_um": float(positions[peak_channel, 0]),
            "template_peak_depth_um": peak_depth,
            "patch_origin_depth_um": patch_origin,
        })
        if reason == "eligible":
            patch = full[:, keep]
            patch_peak = int(np.unravel_index(np.argmax(np.abs(patch)), patch.shape)[0])
            waves.append(v1.centered_crop(patch, patch_peak, v1.WAVEFORM_SAMPLES))
    inventory = pd.DataFrame(rows)
    eligible = inventory[inventory["eligibility"] == "eligible"].reset_index(drop=True)
    if len(eligible) != len(waves):
        raise RuntimeError("eligible inventory and waveform rows disagree")
    return inventory, np.asarray(waves, dtype=np.float32)


def maximum_spanning_tree_bottleneck(submatrix: np.ndarray) -> float:
    """Weakest selected edge in a maximum spanning tree (Kruskal)."""
    submatrix = np.asarray(submatrix, float)
    count = len(submatrix)
    if submatrix.shape != (count, count) or count < 2:
        return np.nan
    edges = sorted(
        [(submatrix[i, j], i, j) for i in range(count) for j in range(i + 1, count)],
        reverse=True,
    )
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    selected = []
    for score, i, j in edges:
        a, b = root(i), root(j)
        if a != b:
            parent[a] = b
            selected.append(float(score))
            if len(selected) == count - 1:
                break
    return min(selected) if len(selected) == count - 1 else np.nan


def add_graph_isolation(
    audits: pd.DataFrame, eligible_cids: np.ndarray, similarity: np.ndarray,
) -> pd.DataFrame:
    location = {int(cid): i for i, cid in enumerate(eligible_cids)}
    bottlenecks = []
    margins = []
    for row in audits.itertuples(index=False):
        indices = [location[int(cid)] for cid in str(row.member_cids).split()]
        bottleneck = maximum_spanning_tree_bottleneck(similarity[np.ix_(indices, indices)])
        bottlenecks.append(bottleneck)
        margins.append(bottleneck - float(row.max_cosine_to_outside_template))
    result = audits.copy()
    result["mst_bottleneck_cosine"] = bottlenecks
    result["mst_bottleneck_minus_external_cosine"] = margins
    return result


def static_merge_fate(audits: pd.DataFrame, premerge_clu: np.ndarray, static_clu: np.ndarray) -> pd.DataFrame:
    """Report whether ordinary final merging collapses members of coherent families."""
    if len(premerge_clu) != len(static_clu):
        raise ValueError("premerge and static labels disagree in length")
    mapping = {}
    for cid in np.unique(premerge_clu):
        labels = np.unique(static_clu[premerge_clu == cid])
        if len(labels) != 1:
            raise RuntimeError("one premerge cluster maps to multiple static clusters")
        mapping[int(cid)] = int(labels[0])
    rows = []
    coherent = audits[audits["depth_time_class"] == "depth_time_coherent"]
    for row in coherent.itertuples(index=False):
        members = list(map(int, str(row.member_cids).split()))
        static_labels = [mapping[cid] for cid in members]
        rows.append({
            "cosine_threshold": row.cosine_threshold,
            "threshold_family_id": row.threshold_family_id,
            "premerge_member_cids": row.member_cids,
            "static_output_labels": " ".join(map(str, static_labels)),
            "premerge_member_count": len(members),
            "distinct_static_output_labels": len(set(static_labels)),
            "member_pairs_collapsed_by_static_merge": len(members) - len(set(static_labels)),
            "all_members_remain_separate_after_static_merge": len(set(static_labels)) == len(members),
        })
    return pd.DataFrame(rows)


def save_moving_candidate_spotlight(
    audits: pd.DataFrame, members: pd.DataFrame, motion: pd.DataFrame, tracks: pd.DataFrame,
    frames: np.ndarray, event_cids: np.ndarray, depth_um: np.ndarray, fs: float,
) -> None:
    candidates = motion[motion["cosine_threshold"] == 0.97]
    if candidates.empty:
        return
    row = candidates.assign(
        displacement_span=lambda x: x["family_displacement_max_um"] - x["family_displacement_min_um"]
    ).sort_values("displacement_span", ascending=False).iloc[0]
    family_id = row["family_id"]
    audit = audits[(audits["cosine_threshold"] == 0.97) & (audits["family_id"] == family_id)].iloc[0]
    cids = members.loc[
        (members["cosine_threshold"] == 0.97) & (members["family_id"] == family_id), "cid"
    ].to_numpy(int)
    take = np.flatnonzero(np.isin(event_cids, cids))
    take = take[np.argsort(frames[take], kind="stable")]
    times = frames[take].astype(float) / fs
    depths = depth_um[take]
    segments = np.split(np.arange(len(times)), np.flatnonzero(np.diff(times) > 2.0) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, layout="constrained", height_ratios=(2.2, 1))
    for segment in segments:
        if len(segment) > 1:
            axes[0].plot(times[segment], depths[segment], color="#2E78B7", lw=0.5, alpha=0.25)
    axes[0].scatter(times, depths, s=6, color="#2E78B7", alpha=0.75, linewidths=0, rasterized=True)
    axes[0].set(ylabel="cached premerge spike depth (µm)", title=(
        f"{row.threshold_family_id}: {len(cids)} premerge tracklets · template span {audit.template_depth_span_um:.0f} µm · "
        f"MST bottleneck {audit.mst_bottleneck_cosine:.3f} · max rapid jump {audit.cross_cid_depth_jump_max_um:.1f} µm"
    ))
    family_track = tracks[
        (tracks["cosine_threshold"] == 0.97) & (tracks["threshold_family_id"] == row["threshold_family_id"])
        & (tracks["track_type"] == "kilosort_family")
    ]
    reference = tracks[
        (tracks["cosine_threshold"] == 0.97) & (tracks["track_type"] == "previous_lighthouse_median")
    ]
    axes[1].plot(family_track["time_s"], family_track["relative_depth_um"], "o-", color="#222222", lw=1.7, ms=4,
                 label="premerge family")
    axes[1].plot(reference["time_s"], reference["relative_depth_um"], "o-", color="#2E78B7", lw=1.2, ms=3,
                 label="previous lighthouse median")
    axes[1].axhline(0, color="0.5", lw=0.7)
    axes[1].set(xlabel="recording time (s)", ylabel="seed-relative depth (µm)", title=(
        f"5-s displacement: prior-cell median r={row.consensus_pearson_r:.2f}, shift p={row.consensus_circular_shift_p:.2g}; "
        f"best cell {int(row.best_lighthouse_unit_id)} r={row.best_pearson_r:.2f}, corrected p={row.best_library_match_circular_shift_p:.2g}"
    ))
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Kilosort premerge moving-family candidate; identity graph constructed without depth/time", fontsize=14)
    fig.savefig(OUT / "06_premerge_moving_candidate_spotlight.png", dpi=180)
    fig.savefig(OUT / "06_premerge_moving_candidate_spotlight.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((PREMERGE / "receipt.json").read_text())
    if not receipt.get("complete") or receipt.get("merging_function_called"):
        raise RuntimeError("checkpoint is not a complete premerge Kilosort state")
    ops_path = Path(receipt["source_ops"]["path"])
    ops = np.load(ops_path, allow_pickle=True).item()
    fs = float(ops["fs"])
    positions = np.column_stack([np.asarray(ops["xc"]), np.asarray(ops["yc"])]).astype(float)
    wall = np.load(PREMERGE / "Wall.npy", mmap_mode="r")
    st = np.load(PREMERGE / "st.npy", mmap_mode="r")
    clu = np.load(PREMERGE / "clu.npy", mmap_mode="r").astype(int)
    depths = np.load(AUDIT / "observed_depth_um.npy", mmap_mode="r")
    if not (len(st) == len(clu) == len(depths)):
        raise RuntimeError("premerge event arrays disagree")
    time_s_all = np.asarray(st[:, 0], float) / fs
    take = (time_s_all >= INTERVAL_S[0]) & (time_s_all < INTERVAL_S[1])
    frames = np.asarray(st[take, 0], dtype=np.int64)
    event_cids = np.asarray(clu[take], dtype=int)
    depth_um = np.asarray(depths[take], dtype=float)
    if not np.all(np.isfinite(depth_um)):
        raise RuntimeError("nonfinite cached premerge depth in target interval")
    xy = np.column_stack([np.zeros(len(depth_um)), depth_um])

    templates = reconstruct_temporal_templates(wall, np.asarray(ops["wPCA"]))
    inventory, waves = relative_waveform_inventory(templates, positions, event_cids)
    eligible = inventory[inventory["eligibility"] == "eligible"].reset_index(drop=True)
    similarity, best_lag = v1.best_lagged_cosine(waves)

    audits = []
    members = []
    edges = []
    for threshold in THRESHOLDS:
        audit, threshold_members, threshold_edges = v4.audit_components(
            threshold, eligible, similarity, frames, event_cids, xy, fs
        )
        audits.append(audit); members.append(threshold_members); edges.append(threshold_edges)
    audits = add_graph_isolation(
        pd.concat(audits, ignore_index=True), eligible["cid"].to_numpy(int), similarity
    )
    audits = v4.attach_baseline_parent(audits)
    members = pd.concat(members, ignore_index=True)
    edges = pd.concat(edges, ignore_index=True)
    survival = v4.baseline_survival(audits)

    summary_rows = []
    for threshold in THRESHOLDS:
        subset = audits[audits["cosine_threshold"] == threshold]
        summary_rows.append({
            "cosine_threshold": threshold,
            "multi_family_count": int(len(subset)),
            "templates_in_families": int(subset["member_count"].sum()),
            "largest_family_size": int(subset["member_count"].max()),
            "complete_link_family_count": int(subset["complete_link_at_threshold"].sum()),
            "depth_time_coherent_count": int(subset["depth_time_class"].eq("depth_time_coherent").sum()),
            "implausible_count": int(subset["depth_time_class"].eq("implausible").sum()),
            "underpowered_count": int(subset["depth_time_class"].eq("underpowered").sum()),
            "coherent_family_ids": " ".join(
                subset.loc[subset["depth_time_class"].eq("depth_time_coherent"), "threshold_family_id"]
            ),
        })
    threshold_summary = pd.DataFrame(summary_rows)
    lighthouse = pd.read_csv(v3.LIGHTHOUSE)
    pair, motion, tracks = v4.compare_motion(
        audits, members, frames, event_cids, depth_um, fs, lighthouse
    )
    static_clu = np.load(STATIC_CLU, mmap_mode="r")
    merge_fate = static_merge_fate(audits, np.asarray(clu), np.asarray(static_clu))

    inventory.to_csv(OUT / "premerge_template_inventory.csv", index=False)
    threshold_summary.to_csv(OUT / "threshold_summary.csv", index=False)
    audits.to_csv(OUT / "component_audit_all_thresholds.csv", index=False)
    members.to_csv(OUT / "members_all_thresholds.csv", index=False)
    edges.to_csv(OUT / "edges_all_thresholds.csv", index=False)
    survival.to_csv(OUT / "baseline_0p95_family_survival.csv", index=False)
    pair.to_csv(OUT / "coherent_family_by_lighthouse_all_thresholds.csv", index=False)
    motion.to_csv(OUT / "coherent_family_motion_all_thresholds.csv", index=False)
    tracks.to_csv(OUT / "binned_tracks_all_thresholds.csv", index=False)
    merge_fate.to_csv(OUT / "coherent_family_static_merge_fate.csv", index=False)
    np.savez_compressed(
        OUT / "waveform_similarity_matrix.npz",
        cid=eligible["cid"].to_numpy(int), cosine=similarity, best_lag_samples=best_lag,
    )

    old_out = v4.OUT
    try:
        v4.OUT = OUT
        v4.save_summary_figures(threshold_summary, survival, audits, motion, tracks)
        for threshold in THRESHOLDS:
            v4.save_joined_atlas(threshold, audits, members, frames, event_cids, depth_um, fs)
    finally:
        v4.OUT = old_out
    save_moving_candidate_spotlight(
        audits, members, motion, tracks, frames, event_cids, depth_um, fs
    )

    final_reference_summary = pd.read_csv(
        ROOT / "testing/outputs/luke_kilosort_family_threshold_sweep_v4/threshold_summary.csv"
    )
    comparison = threshold_summary.merge(
        final_reference_summary, on="cosine_threshold", suffixes=("_premerge", "_final_cid")
    )
    comparison.to_csv(OUT / "premerge_vs_final_cid_summary.csv", index=False)

    summary = {
        "status": "complete",
        "purpose": "depth-blind lighthouse family discovery on Kilosort clusters before final merging",
        "window_s": list(INTERVAL_S),
        "raw_voltage_read": False,
        "new_sort_run": False,
        "merging_function_called": False,
        "checkpoint_window_s": receipt["window_s"],
        "checkpoint_clusters": int(receipt["premerge_clusters"]),
        "snippet_events": int(len(frames)),
        "snippet_active_premerge_clusters": int(np.unique(event_cids).size),
        "eligible_premerge_templates": int(len(eligible)),
        "template_reconstruction": "einsum('ncp,pt->ntc', Wall, wPCA); validated separately against final templates",
        "threshold_summary": threshold_summary.to_dict(orient="records"),
        "coherent_family_static_merge_fate": merge_fate.to_dict(orient="records"),
        "limitations": [
            "Wall templates were learned by bounded 930-1230 s clustering, while event plausibility is shown for 930-1030 s.",
            "The reconstructed premerge clusters are a fresh bounded clustering from cached full-sort st/tF, not a historical checkpoint from the accepted final sort.",
            "Kilosort PC-energy depth is cached sorter-derived localization rather than an independent raw-voltage position.",
            "Depth/time coherence and agreement with previous lighthouse traces remain candidate screens, not biological identity proof.",
        ],
        "source_sha256": {str(path): sha256(path) for path in [
            PREMERGE / "receipt.json", PREMERGE / "Wall.npy", PREMERGE / "st.npy", PREMERGE / "clu.npy",
            AUDIT / "observed_depth_um.npy", STATIC_CLU, ops_path, v3.LIGHTHOUSE,
        ]},
    }
    summary["script_sha256"] = sha256(Path(__file__))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
