"""Rebuild depth-blind Kilosort families at cosine 0.95, 0.97, and 0.98.

This is a cached sensitivity analysis.  Each graph is rebuilt independently at
its threshold before depth/time is revealed.  It then applies the frozen v3
plausibility screen and compares surviving coherent families with the prior
lighthouse traces.  No voltage is read and no sorter is launched.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from testing import luke_kilosort_depthblind_tracklet_linkage_v1 as v1
from testing import luke_kilosort_depthblind_families_v2 as v2
from testing import luke_kilosort_family_plausibility_v3 as v3


OUT = ROOT / "testing/outputs/luke_kilosort_family_threshold_sweep_v4"
THRESHOLDS = (0.95, 0.97, 0.98)
MAX_JOIN_GAP_S = 2.0


def threshold_tag(threshold: float) -> str:
    return f"c{int(round(threshold * 100)):03d}"


def cross_cid_join_metrics(
    members: pd.DataFrame, frames: np.ndarray, event_cids: np.ndarray,
    depth_um: np.ndarray, fs: float,
) -> pd.DataFrame:
    time_s = frames.astype(float) / fs
    rows = []
    for family_id, group in members.groupby("family_id"):
        cids = group["cid"].to_numpy(int)
        take = np.flatnonzero(np.isin(event_cids, cids))
        take = take[np.argsort(frames[take], kind="stable")]
        times = time_s[take]
        ordered_cids = event_cids[take]
        depths = depth_um[take]
        valid = np.diff(times) <= MAX_JOIN_GAP_S
        cross = valid & (np.diff(ordered_cids) != 0)
        jumps = np.abs(np.diff(depths))[cross]
        rows.append({
            "family_id": family_id,
            "cross_cid_joined_intervals": int(cross.sum()),
            "cross_cid_depth_jump_median_um": float(np.median(jumps)) if len(jumps) else np.nan,
            "cross_cid_depth_jump_p95_um": float(np.quantile(jumps, 0.95)) if len(jumps) else np.nan,
            "cross_cid_depth_jump_max_um": float(np.max(jumps)) if len(jumps) else np.nan,
            "cross_cid_depth_jumps_ge_100_um": int(np.count_nonzero(jumps >= v3.RAPID_JUMP_UM)),
            "cross_cid_depth_jumps_ge_160_um": int(np.count_nonzero(jumps >= 160.0)),
        })
    return pd.DataFrame(rows)


def audit_components(
    threshold: float, eligible: pd.DataFrame, similarity: np.ndarray,
    frames: np.ndarray, event_cids: np.ndarray, xy: np.ndarray, fs: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_threshold = v2.FAMILY_THRESHOLD
    try:
        v2.FAMILY_THRESHOLD = threshold
        members, edges, families = v2.build_waveform_families(eligible, similarity)
    finally:
        v2.FAMILY_THRESHOLD = old_threshold
    members_revealed, families_revealed = v2.reveal_depth_and_time(
        members, families, eligible, frames, event_cids, xy, fs
    )
    joined = cross_cid_join_metrics(members_revealed, frames, event_cids, xy[:, 1], fs)
    audit = v3.waveform_predictors(families_revealed, members).merge(
        joined, on="family_id", validate="one_to_one"
    )
    audit = audit.rename(columns={"complete_link_at_0p95": "complete_link_at_threshold"})
    audit["rapid_jump_fraction_ge_100um"] = (
        audit["cross_cid_depth_jumps_ge_100_um"]
        / audit["cross_cid_joined_intervals"].replace(0, np.nan)
    )
    time_s = frames.astype(float) / fs
    coactive = []
    for family_id, group in members_revealed.groupby("family_id"):
        cids = group["cid"].to_numpy(int)
        take = np.isin(event_cids, cids)
        count, fraction, median_span = v3.coactive_depth_separation(
            time_s[take], xy[take, 1], event_cids[take], cids
        )
        coactive.append({
            "family_id": family_id,
            "coactive_one_second_bins": count,
            "separated_coactive_fraction_ge_80um": fraction,
            "coactive_median_between_cid_span_um": median_span,
        })
    audit = audit.merge(pd.DataFrame(coactive), on="family_id", validate="one_to_one")
    audit["depth_time_class"] = audit.apply(v3.classify_family, axis=1)
    audit.insert(0, "cosine_threshold", threshold)
    audit.insert(1, "threshold_family_id", threshold_tag(threshold) + "_" + audit["family_id"])
    audit["member_signature"] = audit["member_cids"].map(
        lambda value: " ".join(map(str, sorted(map(int, str(value).split()))))
    )
    members_revealed.insert(0, "cosine_threshold", threshold)
    members_revealed.insert(1, "threshold_family_id", threshold_tag(threshold) + "_" + members_revealed["family_id"])
    edges.insert(0, "cosine_threshold", threshold)
    return audit, members_revealed, edges


def attach_baseline_parent(all_audits: pd.DataFrame) -> pd.DataFrame:
    baseline = all_audits[all_audits["cosine_threshold"] == THRESHOLDS[0]]
    base_sets = {
        row.threshold_family_id: set(map(int, row.member_signature.split()))
        for row in baseline.itertuples(index=False)
    }
    parent_ids = []
    retention = []
    for row in all_audits.itertuples(index=False):
        current = set(map(int, row.member_signature.split()))
        overlaps = [(parent, len(current & members), members) for parent, members in base_sets.items()]
        parent, overlap, parent_members = max(overlaps, key=lambda item: item[1])
        parent_ids.append(parent if overlap else "")
        retention.append(len(current) / len(parent_members) if overlap else np.nan)
    result = all_audits.copy()
    result["baseline_0p95_parent"] = parent_ids
    result["baseline_member_retention_fraction"] = retention
    return result


def baseline_survival(audits: pd.DataFrame) -> pd.DataFrame:
    baseline = audits[audits["cosine_threshold"] == THRESHOLDS[0]]
    rows = []
    for parent in baseline.itertuples(index=False):
        original = set(map(int, parent.member_signature.split()))
        for threshold in THRESHOLDS:
            children = audits[
                (audits["cosine_threshold"] == threshold)
                & (audits["baseline_0p95_parent"] == parent.threshold_family_id)
            ]
            retained = set()
            for signature in children["member_signature"]:
                retained.update(map(int, signature.split()))
            together = any(set(map(int, signature.split())) == original for signature in children["member_signature"])
            rows.append({
                "baseline_family_id": parent.family_id,
                "baseline_depth_time_class": parent.depth_time_class,
                "cosine_threshold": threshold,
                "baseline_member_count": len(original),
                "members_retained_in_any_multitemplate_child": len(original & retained),
                "retained_fraction": len(original & retained) / len(original),
                "all_original_members_still_together": bool(together),
                "child_family_ids": " ".join(children["threshold_family_id"]),
                "coherent_child_family_ids": " ".join(
                    children.loc[children["depth_time_class"] == "depth_time_coherent", "threshold_family_id"]
                ),
            })
    return pd.DataFrame(rows)


def compare_motion(
    audits: pd.DataFrame, members: pd.DataFrame, frames: np.ndarray,
    event_cids: np.ndarray, depth_um: np.ndarray, fs: float, lighthouse: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_pair = []
    all_best = []
    track_rows = []
    time_s = frames.astype(float) / fs
    for threshold in THRESHOLDS:
        threshold_audit = audits[audits["cosine_threshold"] == threshold]
        coherent = threshold_audit.loc[
            threshold_audit["depth_time_class"] == "depth_time_coherent", "family_id"
        ].sort_values().tolist()
        threshold_members = members[members["cosine_threshold"] == threshold]
        if not coherent:
            continue
        family_tracks, prior_tracks, consensus, support = v3.make_tracks(
            coherent, threshold_members, time_s, event_cids, depth_um, lighthouse
        )
        pair, best = v3.compare_tracks(family_tracks, prior_tracks, consensus)
        pair.insert(0, "cosine_threshold", threshold)
        pair.insert(1, "threshold_family_id", threshold_tag(threshold) + "_" + pair["family_id"])
        best.insert(0, "cosine_threshold", threshold)
        best.insert(1, "threshold_family_id", threshold_tag(threshold) + "_" + best["family_id"])
        all_pair.append(pair)
        all_best.append(best)
        for family_id, series in family_tracks.items():
            for bin_index, value in series.items():
                track_rows.append({
                    "cosine_threshold": threshold, "threshold_family_id": threshold_tag(threshold) + "_" + family_id,
                    "track_type": "kilosort_family", "track_id": family_id, "bin": int(bin_index),
                    "time_s": v3.INTERVAL_S[0] + (bin_index + 0.5) * v3.TRACK_BIN_S,
                    "relative_depth_um": float(value),
                })
        for bin_index, value in consensus.items():
            track_rows.append({
                "cosine_threshold": threshold, "threshold_family_id": "reference",
                "track_type": "previous_lighthouse_median", "track_id": "median", "bin": int(bin_index),
                "time_s": v3.INTERVAL_S[0] + (bin_index + 0.5) * v3.TRACK_BIN_S,
                "relative_depth_um": float(value), "contributing_lighthouse_units": int(support.loc[bin_index]),
            })
    pair_table = pd.concat(all_pair, ignore_index=True) if all_pair else pd.DataFrame()
    best_table = pd.concat(all_best, ignore_index=True) if all_best else pd.DataFrame()
    return pair_table, best_table, pd.DataFrame(track_rows)


def save_joined_atlas(
    threshold: float, audit: pd.DataFrame, members: pd.DataFrame,
    frames: np.ndarray, event_cids: np.ndarray, depth_um: np.ndarray, fs: float,
) -> None:
    subset = audit[audit["cosine_threshold"] == threshold].sort_values(
        ["depth_time_class", "member_count"], ascending=[True, False]
    )
    threshold_members = members[members["cosine_threshold"] == threshold]
    times_all = frames.astype(float) / fs
    colors = {"depth_time_coherent": "#2E78B7", "implausible": "#D06B4C", "underpowered": "#888888"}
    tag = threshold_tag(threshold)
    with PdfPages(OUT / f"04_{tag}_joined_family_atlas.pdf") as pdf:
        for page_start in range(0, len(subset), 4):
            page = subset.iloc[page_start:page_start + 4]
            fig, axes = plt.subplots(len(page), 1, figsize=(14, 3 * len(page)), sharex=True, layout="constrained", squeeze=False)
            for ax, family in zip(axes[:, 0], page.itertuples(index=False)):
                cids = threshold_members.loc[
                    threshold_members["family_id"] == family.family_id, "cid"
                ].to_numpy(int)
                take = np.flatnonzero(np.isin(event_cids, cids))
                take = take[np.argsort(frames[take], kind="stable")]
                times = times_all[take]
                depths = depth_um[take]
                segments = v2.chronological_segments(times, MAX_JOIN_GAP_S)
                for segment in segments:
                    if len(segment) > 1:
                        ax.plot(times[segment], depths[segment], color=colors[family.depth_time_class], lw=0.5, alpha=0.25)
                ax.scatter(times, depths, s=4, color=colors[family.depth_time_class], alpha=0.7, linewidths=0, rasterized=True)
                ax.set(ylabel="depth (µm)", title=(
                    f"{tag}_{family.family_id} · {family.depth_time_class.replace('_', ' ')} · "
                    f"{family.member_count} CIDs [{family.member_cids}] · min pair cosine {family.all_pairs_min_cosine:.3f}"
                ))
                ax.grid(alpha=0.2)
            axes[-1, 0].set(xlabel="recording time (s)", xlim=v3.INTERVAL_S)
            fig.suptitle(f"Families rebuilt at cosine ≥{threshold:.2f}; chronological joins break at gaps >2 s", fontsize=13)
            pdf.savefig(fig, dpi=180)
            fig.savefig(OUT / f"04_{tag}_joined_family_atlas_page{page_start // 4 + 1}.png", dpi=180)
            plt.close(fig)


def save_summary_figures(
    summary: pd.DataFrame, survival: pd.DataFrame, audits: pd.DataFrame,
    motion: pd.DataFrame, tracks: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for column, title, ax in zip(
        ["multi_family_count", "templates_in_families", "largest_family_size"],
        ["Multi-template families", "Templates retained in families", "Largest component"], axes,
    ):
        ax.plot(summary["cosine_threshold"], summary[column], "o-", color="#2E78B7", lw=2)
        ax.set(xlabel="cosine threshold", ylabel="count", title=title)
        ax.grid(alpha=0.2)
    fig.suptitle("Family graph size after rebuilding at each threshold")
    fig.savefig(OUT / "01_threshold_graph_summary.png", dpi=180)
    fig.savefig(OUT / "01_threshold_graph_summary.pdf")
    plt.close(fig)

    baseline_order = audits[audits["cosine_threshold"] == 0.95].sort_values(
        ["depth_time_class", "family_id"]
    )["family_id"].tolist()
    matrix = survival.pivot(index="baseline_family_id", columns="cosine_threshold", values="retained_fraction").reindex(baseline_order)
    fig, ax = plt.subplots(figsize=(7, 9), layout="constrained")
    image = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), [f"≥{x:.2f}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set(xlabel="rebuilt graph threshold", ylabel="0.95 family",
           title="Fraction of each 0.95 family's members retained in any multi-template child")
    for y in range(len(matrix.index)):
        for x in range(len(matrix.columns)):
            ax.text(x, y, f"{matrix.iloc[y, x]:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if matrix.iloc[y, x] > 0.6 else "#222222")
    fig.colorbar(image, ax=ax, label="member retention fraction")
    fig.savefig(OUT / "02_baseline_family_survival.png", dpi=180)
    fig.savefig(OUT / "02_baseline_family_survival.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=(16, 5), sharex=True, sharey=True, layout="constrained")
    colors = {"depth_time_coherent": "#2E78B7", "implausible": "#D06B4C", "underpowered": "#888888"}
    for ax, threshold in zip(axes, THRESHOLDS):
        subset = audits[audits["cosine_threshold"] == threshold]
        for status, group in subset.groupby("depth_time_class"):
            ax.scatter(group["rapid_jump_fraction_ge_100um"], group["separated_coactive_fraction_ge_80um"],
                       s=25 + 8 * group["member_count"], color=colors[status], alpha=0.8, label=status.replace("_", " "))
            for row in group.itertuples(index=False):
                ax.annotate(row.family_id, (row.rapid_jump_fraction_ge_100um, row.separated_coactive_fraction_ge_80um),
                            xytext=(2, 2), textcoords="offset points", fontsize=7)
        ax.axvline(v3.MAX_RAPID_JUMP_FRACTION, color="0.35", ls="--", lw=1)
        ax.axhline(v3.MAX_SEPARATED_COACTIVE_FRACTION, color="0.35", ls="--", lw=1)
        ax.set(title=f"cosine ≥{threshold:.2f}", xlabel="rapid ≥100-µm jump fraction")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("coactive separated ≥80-µm fraction")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", frameon=False)
    fig.suptitle("Depth/time outcomes after independently rebuilding each waveform graph")
    fig.savefig(OUT / "03_threshold_depth_time_outcomes.png", dpi=180)
    fig.savefig(OUT / "03_threshold_depth_time_outcomes.pdf")
    plt.close(fig)

    if motion.empty:
        return
    high_motion = motion[motion["cosine_threshold"] > 0.95].copy()
    if high_motion.empty:
        return
    fig, axes = plt.subplots(len(high_motion), 1, figsize=(13, 3 * len(high_motion)), sharex=True, layout="constrained", squeeze=False)
    full_bins = np.arange(20)
    centers = v3.INTERVAL_S[0] + (full_bins + 0.5) * v3.TRACK_BIN_S
    for ax, row in zip(axes[:, 0], high_motion.itertuples(index=False)):
        family = tracks[(tracks["cosine_threshold"] == row.cosine_threshold)
                        & (tracks["threshold_family_id"] == row.threshold_family_id)
                        & (tracks["track_type"] == "kilosort_family")].set_index("bin")["relative_depth_um"].reindex(full_bins)
        reference = tracks[(tracks["cosine_threshold"] == row.cosine_threshold)
                           & (tracks["track_type"] == "previous_lighthouse_median")].set_index("bin")["relative_depth_um"].reindex(full_bins)
        ax.plot(centers, family, "o-", color="#222222", lw=1.7, ms=4, label=row.threshold_family_id)
        ax.plot(centers, reference, "o-", color="#2E78B7", lw=1.2, ms=3, label="prior lighthouse median")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set(ylabel="seed-relative depth (µm)", title=(
            f"{row.threshold_family_id}: consensus r={row.consensus_pearson_r:.2f}, shift p={row.consensus_circular_shift_p:.2g}; "
            f"best-cell r={row.best_pearson_r:.2f}, corrected p={row.best_library_match_circular_shift_p:.2g}"
        ))
        ax.grid(alpha=0.2); ax.legend(frameon=False)
    axes[-1, 0].set_xlabel("recording time (s)")
    fig.suptitle("Higher-threshold coherent families versus previous lighthouse displacement")
    fig.savefig(OUT / "05_high_threshold_motion_comparison.png", dpi=180)
    fig.savefig(OUT / "05_high_threshold_motion_comparison.pdf")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_summary = json.loads(v1.EVENT_SUMMARY.read_text())
    fs = float(event_summary["sampling_frequency_hz"])
    event_file = np.load(v1.EVENTS)
    frames = np.asarray(event_file["frames"])
    event_cids = np.asarray(event_file["cid"]).astype(int)
    xy = np.asarray(event_file["xy_um"])
    templates = np.load(v1.SOURCE / "templates.npy", mmap_mode="r")
    positions = np.load(v1.SOURCE / "channel_positions.npy")
    all_templates, waves = v1.make_templates(templates, positions, event_cids)
    eligible = all_templates[all_templates["eligibility"] == "eligible"].copy().reset_index(drop=True)
    similarity, _ = v1.best_lagged_cosine(waves)

    audits = []
    members = []
    edges = []
    for threshold in THRESHOLDS:
        audit, threshold_members, threshold_edges = audit_components(
            threshold, eligible, similarity, frames, event_cids, xy, fs
        )
        audits.append(audit); members.append(threshold_members); edges.append(threshold_edges)
    audits = attach_baseline_parent(pd.concat(audits, ignore_index=True))
    members = pd.concat(members, ignore_index=True)
    edges = pd.concat(edges, ignore_index=True)
    survival = baseline_survival(audits)
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
            "coherent_family_ids": " ".join(subset.loc[subset["depth_time_class"].eq("depth_time_coherent"), "threshold_family_id"]),
        })
    threshold_summary = pd.DataFrame(summary_rows)
    lighthouse = pd.read_csv(v3.LIGHTHOUSE)
    pair, motion, tracks = compare_motion(
        audits, members, frames, event_cids, xy[:, 1], fs, lighthouse
    )

    threshold_summary.to_csv(OUT / "threshold_summary.csv", index=False)
    audits.to_csv(OUT / "component_audit_all_thresholds.csv", index=False)
    members.to_csv(OUT / "members_all_thresholds.csv", index=False)
    edges.to_csv(OUT / "edges_all_thresholds.csv", index=False)
    survival.to_csv(OUT / "baseline_0p95_family_survival.csv", index=False)
    pair.to_csv(OUT / "coherent_family_by_lighthouse_all_thresholds.csv", index=False)
    motion.to_csv(OUT / "coherent_family_motion_all_thresholds.csv", index=False)
    tracks.to_csv(OUT / "binned_tracks_all_thresholds.csv", index=False)
    save_summary_figures(threshold_summary, survival, audits, motion, tracks)
    for threshold in (0.97, 0.98):
        save_joined_atlas(threshold, audits, members, frames, event_cids, xy[:, 1], fs)

    baseline_coherent = audits[
        (audits["cosine_threshold"] == 0.95) & audits["depth_time_class"].eq("depth_time_coherent")
    ]["threshold_family_id"].tolist()
    baseline_core_survival = survival[
        survival["baseline_family_id"].isin([value.split("_")[-1] for value in baseline_coherent])
    ]
    result = {
        "status": "complete",
        "raw_voltage_read": False,
        "new_spike_sort_run": False,
        "thresholds": list(THRESHOLDS),
        "threshold_summary": threshold_summary.to_dict(orient="records"),
        "baseline_coherent_survival": baseline_core_survival.to_dict(orient="records"),
        "interpretation": "threshold graphs rebuilt independently; depth/time classification remains post-hoc",
        "limitations": [
            "A higher cosine threshold can remove plausible families while retaining high-cosine implausible pairs.",
            "Connected components remain single-linkage unless all-pairs completeness is separately required.",
            "Depth/time coherence and lighthouse agreement are candidate screens, not identity validation.",
        ],
        "source_sha256": {
            str(v1.EVENTS): sha256(v1.EVENTS),
            str(v1.SOURCE / "templates.npy"): sha256(v1.SOURCE / "templates.npy"),
            str(v1.SOURCE / "channel_positions.npy"): sha256(v1.SOURCE / "channel_positions.npy"),
            str(v3.LIGHTHOUSE): sha256(v3.LIGHTHOUSE),
        },
    }
    result["script_sha256"] = sha256(Path(__file__))
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
