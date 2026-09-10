"""Reconcile 17 prior lighthouse identities with the Kilosort premerge screen.

This is a diagnostic, not a new identity matcher. Prior seed/event frames and
depths are used only to trace already-frozen lighthouse candidates into the
premerge partition. The output explains whether each candidate was absent at
detection, excluded from the waveform inventory, represented by a singleton,
or placed in a coherent/implausible multi-template family.
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

from testing import luke_kilosort_family_plausibility_v3 as v3


PREMERGE_ROOT = ROOT / "testing/outputs/luke_medicine_motion_aware_merge_v1"
SCREEN = ROOT / "testing/outputs/luke_kilosort_premerge_lighthouse_v5"
PRIOR = ROOT / "testing/outputs/luke_waveform_only_expansion_v1"
SEEDS = ROOT / "testing/outputs/luke_population_depth_v2/templates.npz"
OUT = ROOT / "testing/outputs/luke_premerge_prior_lighthouse_reconciliation_v1"
FS = 29999.835983263598
TIME_TOLERANCE_SAMPLES = 9
SEED_CLUSTER_MIN_EVENTS = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_events_to_premerge(
    target_frames: np.ndarray, target_depth_um: np.ndarray, pre_frames: np.ndarray,
    pre_depth_um: np.ndarray, pre_clu: np.ndarray,
) -> pd.DataFrame:
    """Map known events within +/-9 samples, using depth only to resolve candidates."""
    rows = []
    for frame, depth in zip(np.asarray(target_frames, int), np.asarray(target_depth_um, float)):
        lo = np.searchsorted(pre_frames, frame - TIME_TOLERANCE_SAMPLES, side="left")
        hi = np.searchsorted(pre_frames, frame + TIME_TOLERANCE_SAMPLES, side="right")
        candidates = np.arange(lo, hi)
        if not len(candidates):
            rows.append({"frame": frame, "matched": False, "premerge_cluster": np.nan,
                         "sample_error": np.nan, "depth_error_um": np.nan, "candidate_detections": 0})
            continue
        sample_error = np.abs(pre_frames[candidates] - frame)
        # Prefer exact temporal alignment; depth resolves duplicate detections at
        # the same closest sample without becoming a discovery feature.
        closest_time = np.min(sample_error)
        candidates = candidates[sample_error == closest_time]
        chosen = candidates[np.argmin(np.abs(pre_depth_um[candidates] - depth))]
        rows.append({
            "frame": frame,
            "matched": True,
            "premerge_cluster": int(pre_clu[chosen]),
            "sample_error": int(abs(pre_frames[chosen] - frame)),
            "depth_error_um": float(pre_depth_um[chosen] - depth),
            "candidate_detections": int(len(candidates)),
        })
    return pd.DataFrame(rows)


def family_lookup(audit: pd.DataFrame, members: pd.DataFrame, threshold: float) -> dict[int, tuple[str, str]]:
    selected_members = members[members["cosine_threshold"] == threshold]
    selected_audit = audit[audit["cosine_threshold"] == threshold].set_index("family_id")
    lookup = {}
    for row in selected_members.itertuples(index=False):
        status = selected_audit.loc[row.family_id, "depth_time_class"]
        lookup[int(row.cid)] = (row.threshold_family_id, status)
    return lookup


def closest_similarity(cid: int, similarity_ids: np.ndarray, similarity: np.ndarray) -> tuple[int | None, float]:
    hit = np.flatnonzero(similarity_ids == cid)
    if not len(hit):
        return None, np.nan
    index = int(hit[0])
    values = similarity[index].copy()
    values[index] = -np.inf
    closest = int(np.argmax(values))
    return int(similarity_ids[closest]), float(values[closest])


def track_for_clusters(
    clusters: list[int], pre_time_s: np.ndarray, pre_depth_um: np.ndarray, pre_clu: np.ndarray,
) -> pd.Series:
    take = np.isin(pre_clu, clusters) & (pre_time_s >= 930) & (pre_time_s < 1030)
    return v3.seed_center(v3.binned_median(pre_time_s[take], pre_depth_um[take], 5.0), 5.0)


def categorize(dominant_cid: int, lookup_097: dict[int, tuple[str, str]]) -> str:
    if dominant_cid not in lookup_097:
        return "dominant cluster is a 0.97 singleton"
    _, status = lookup_097[dominant_cid]
    if status == "depth_time_coherent":
        return "dominant cluster is in a coherent 0.97 family"
    if status == "implausible":
        return "dominant cluster is in an implausible 0.97 family"
    return "dominant cluster is in an underpowered 0.97 family"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(PRIOR / "candidate_audit.csv")
    candidates = candidates[candidates["selected"]].sort_values("seed_centroid_um")
    overlay = pd.read_csv(PRIOR / "overlay_events.csv")
    strict = overlay[overlay["evidence"] == "strict_accepted"].copy()
    seed_bank = np.load(SEEDS)

    pre_st = np.load(PREMERGE_ROOT / "bounded_premerge/st.npy", mmap_mode="r")
    pre_frames = np.asarray(pre_st[:, 0], dtype=np.int64)
    pre_time_s = pre_frames.astype(float) / FS
    pre_clu = np.asarray(np.load(PREMERGE_ROOT / "bounded_premerge/clu.npy", mmap_mode="r"), int)
    pre_depth = np.asarray(np.load(PREMERGE_ROOT / "audit/observed_depth_um.npy", mmap_mode="r"), float)
    inventory = pd.read_csv(SCREEN / "premerge_template_inventory.csv").set_index("cid")
    audit = pd.read_csv(SCREEN / "component_audit_all_thresholds.csv")
    members = pd.read_csv(SCREEN / "members_all_thresholds.csv")
    similarity_file = np.load(SCREEN / "waveform_similarity_matrix.npz")
    similarity_ids = similarity_file["cid"].astype(int)
    similarity = similarity_file["cosine"]
    lookup_095 = family_lookup(audit, members, 0.95)
    lookup_097 = family_lookup(audit, members, 0.97)

    event_rows = []
    cluster_rows = []
    unit_rows = []
    prior_tracks = {}
    dominant_tracks = {}
    union_tracks = {}
    for candidate in candidates.itertuples(index=False):
        unit_id = int(candidate.unit_id)
        seed_frames = seed_bank[f"unit_{unit_id}_seed_frames"].astype(int)
        seed_depths = np.full(len(seed_frames), float(candidate.seed_centroid_um))
        seed_map = map_events_to_premerge(seed_frames, seed_depths, pre_frames, pre_depth, pre_clu)
        seed_map.insert(0, "unit_id", unit_id)
        seed_map.insert(1, "source", "seed_spike")
        event_rows.append(seed_map)
        matched_seed = seed_map[seed_map["matched"]].copy()
        counts = matched_seed["premerge_cluster"].astype(int).value_counts()
        dominant = int(counts.index[0])
        dominant_count = int(counts.iloc[0])
        supported_clusters = sorted(counts[counts >= SEED_CLUSTER_MIN_EVENTS].index.astype(int).tolist())

        unit_strict = strict[strict["unit_id"] == unit_id]
        strict_map = map_events_to_premerge(
            unit_strict["frame"].to_numpy(int), unit_strict["centroid_um"].to_numpy(float),
            pre_frames, pre_depth, pre_clu,
        )
        strict_map.insert(0, "unit_id", unit_id)
        strict_map.insert(1, "source", "strict_lighthouse_event")
        strict_map["training"] = unit_strict["training"].to_numpy(bool)
        event_rows.append(strict_map)
        matched_strict = strict_map[strict_map["matched"]]

        for cid, count in counts.items():
            cid = int(cid)
            fam095, status095 = lookup_095.get(cid, ("singleton", "not_assessed"))
            fam097, status097 = lookup_097.get(cid, ("singleton", "not_assessed"))
            cluster_rows.append({
                "unit_id": unit_id, "premerge_cluster": cid, "seed_events_mapped": int(count),
                "seed_event_fraction": float(count / len(seed_frames)), "is_dominant": cid == dominant,
                "inventory_eligibility": inventory.loc[cid, "eligibility"] if cid in inventory.index else "below_20_spikes",
                "family_0p95": fam095, "family_0p95_status": status095,
                "family_0p97": fam097, "family_0p97_status": status097,
            })

        nearest_cid, nearest_score = closest_similarity(dominant, similarity_ids, similarity)
        fam095, status095 = lookup_095.get(dominant, ("singleton", "not_assessed"))
        fam097, status097 = lookup_097.get(dominant, ("singleton", "not_assessed"))
        prior_track = v3.seed_center(
            v3.binned_median(unit_strict["time_s"], unit_strict["relative_um"], 5.0), 5.0
        )
        dominant_track = track_for_clusters([dominant], pre_time_s, pre_depth, pre_clu)
        union_track = track_for_clusters(supported_clusters, pre_time_s, pre_depth, pre_clu)
        n_dom, r_dom, rho_dom = v3.pearson_and_spearman(dominant_track, prior_track)
        n_union, r_union, rho_union = v3.pearson_and_spearman(union_track, prior_track)
        prior_tracks[unit_id] = prior_track
        dominant_tracks[unit_id] = dominant_track
        union_tracks[unit_id] = union_track
        strict_in_seed_union = (
            matched_strict["premerge_cluster"].astype(int).isin(supported_clusters).mean()
            if len(matched_strict) else np.nan
        )
        unit_rows.append({
            "unit_id": unit_id,
            "cohort": candidate.cohort,
            "seed_events": len(seed_frames),
            "seed_events_with_premerge_detection": int(seed_map["matched"].sum()),
            "seed_detection_recovery_fraction": float(seed_map["matched"].mean()),
            "seed_mapped_cluster_count": int(len(counts)),
            "seed_supported_cluster_count_ge_2_events": int(len(supported_clusters)),
            "seed_supported_clusters": " ".join(map(str, supported_clusters)),
            "dominant_premerge_cluster": dominant,
            "dominant_seed_event_fraction": dominant_count / len(seed_frames),
            "dominant_inventory_eligibility": inventory.loc[dominant, "eligibility"] if dominant in inventory.index else "below_20_spikes",
            "dominant_closest_waveform_cluster": nearest_cid,
            "dominant_closest_waveform_cosine": nearest_score,
            "dominant_family_0p95": fam095,
            "dominant_family_0p95_status": status095,
            "dominant_family_0p97": fam097,
            "dominant_family_0p97_status": status097,
            "diagnostic_bucket": categorize(dominant, lookup_097),
            "strict_events": int(len(unit_strict)),
            "strict_events_with_premerge_detection": int(strict_map["matched"].sum()),
            "strict_detection_recovery_fraction": float(strict_map["matched"].mean()) if len(strict_map) else np.nan,
            "strict_events_in_seed_supported_cluster_union_fraction": strict_in_seed_union,
            "dominant_track_5s_bins": n_dom,
            "dominant_track_vs_prior_pearson_r": r_dom,
            "dominant_track_vs_prior_spearman_rho": rho_dom,
            "seed_cluster_union_track_5s_bins": n_union,
            "seed_cluster_union_vs_prior_pearson_r": r_union,
            "seed_cluster_union_vs_prior_spearman_rho": rho_union,
            "dominant_track_displacement_span_um": float(dominant_track.max() - dominant_track.min()),
            "seed_cluster_union_displacement_span_um": float(union_track.max() - union_track.min()),
        })

    unit_summary = pd.DataFrame(unit_rows).sort_values("unit_id")
    event_mapping = pd.concat(event_rows, ignore_index=True)
    mapped_clusters = pd.DataFrame(cluster_rows).sort_values(["unit_id", "seed_events_mapped"], ascending=[True, False])
    unit_summary.to_csv(OUT / "prior_unit_reconciliation.csv", index=False)
    event_mapping.to_csv(OUT / "prior_event_to_premerge_mapping.csv", index=False)
    mapped_clusters.to_csv(OUT / "prior_seed_cluster_mapping.csv", index=False)

    bucket_counts = unit_summary["diagnostic_bucket"].value_counts().rename_axis("diagnostic_bucket").reset_index(name="prior_candidates")
    bucket_counts.to_csv(OUT / "diagnostic_bucket_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    order = bucket_counts.sort_values("prior_candidates")
    ax.barh(order["diagnostic_bucket"], order["prior_candidates"], color="#2E78B7")
    for y, value in enumerate(order["prior_candidates"]):
        ax.text(value + 0.15, y, str(value), va="center")
    ax.set(xlabel="Prior lighthouse candidates", title="Where the 17 prior lighthouse candidates went in the 0.97 premerge screen")
    ax.set_xlim(0, max(order["prior_candidates"]) + 2); ax.grid(axis="x", alpha=0.2)
    fig.savefig(OUT / "01_reconciliation_buckets.png", dpi=180)
    fig.savefig(OUT / "01_reconciliation_buckets.pdf")
    plt.close(fig)

    with PdfPages(OUT / "02_prior_unit_premerge_tracks.pdf") as pdf:
        for page_start in range(0, len(unit_summary), 6):
            page = unit_summary.iloc[page_start:page_start + 6]
            fig, axes = plt.subplots(len(page), 1, figsize=(14, 2.7 * len(page)), sharex=True, layout="constrained", squeeze=False)
            full_bins = np.arange(20)
            centers = 930 + (full_bins + 0.5) * 5
            for ax, row in zip(axes[:, 0], page.itertuples(index=False)):
                ax.plot(centers, prior_tracks[row.unit_id].reindex(full_bins), "o-", color="#2E78B7", ms=3, lw=1.2,
                        label="prior strict lighthouse")
                ax.plot(centers, dominant_tracks[row.unit_id].reindex(full_bins), "o-", color="#222222", ms=3, lw=1.2,
                        label=f"dominant premerge cluster {row.dominant_premerge_cluster}")
                if row.seed_supported_cluster_count_ge_2_events > 1:
                    ax.plot(centers, union_tracks[row.unit_id].reindex(full_bins), "o--", color="#D17A22", ms=3, lw=1.0,
                            label="seed-supported cluster union")
                ax.axhline(0, color="0.6", lw=0.7)
                ax.set(ylabel="relative depth (µm)", title=(
                    f"Prior {row.unit_id}: {row.diagnostic_bucket}; dominant seed share {row.dominant_seed_event_fraction:.0%}; "
                    f"union/prior r={row.seed_cluster_union_vs_prior_pearson_r:.2f}"
                ))
                ax.grid(alpha=0.2); ax.legend(frameon=False, fontsize=7, ncol=3)
            axes[-1, 0].set(xlabel="recording time (s)", xlim=(930, 1030))
            fig.suptitle("Prior lighthouse identities traced into cached Kilosort premerge clusters", fontsize=13)
            pdf.savefig(fig, dpi=180)
            fig.savefig(OUT / f"02_prior_unit_premerge_tracks_page{page_start // 6 + 1}.png", dpi=180)
            plt.close(fig)

    unique_dominant = int(unit_summary["dominant_premerge_cluster"].nunique())
    summary = {
        "status": "complete",
        "prior_candidate_count": int(len(unit_summary)),
        "unique_dominant_premerge_clusters": unique_dominant,
        "seed_detection_recovery": {
            "matched": int(unit_summary["seed_events_with_premerge_detection"].sum()),
            "total": int(unit_summary["seed_events"].sum()),
        },
        "dominant_cluster_eligibility_counts": unit_summary["dominant_inventory_eligibility"].value_counts().to_dict(),
        "diagnostic_bucket_counts": unit_summary["diagnostic_bucket"].value_counts().to_dict(),
        "shared_dominant_cluster_candidates": (
            unit_summary.groupby("dominant_premerge_cluster")["unit_id"].apply(list)
            .loc[lambda x: x.map(len) > 1].to_dict()
        ),
        "interpretation": "The family screen emits multi-template components only; singleton representations were omitted, not lost at detection.",
        "limitations": [
            "The 17 prior lighthouse templates are provisional and not 17 proven-independent biological cells.",
            "Prior event depth is used only to resolve same-time premerge detections during reconciliation, not to construct premerge families.",
            "A singleton premerge representation can still be a useful lighthouse; this diagnostic does not validate its identity across depth.",
            "Track correlations compare two non-independent analyses of the same recording and strict prior support is sparse for some units.",
        ],
        "source_sha256": {str(path): sha256(path) for path in [
            PRIOR / "candidate_audit.csv", PRIOR / "overlay_events.csv", SEEDS,
            PREMERGE_ROOT / "bounded_premerge/st.npy", PREMERGE_ROOT / "bounded_premerge/clu.npy",
            PREMERGE_ROOT / "audit/observed_depth_um.npy", SCREEN / "component_audit_all_thresholds.csv",
            SCREEN / "members_all_thresholds.csv", SCREEN / "waveform_similarity_matrix.npz",
        ]},
    }
    summary["script_sha256"] = sha256(Path(__file__))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
