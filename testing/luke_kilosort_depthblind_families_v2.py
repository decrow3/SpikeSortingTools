"""Group cached Kilosort templates by a depth-blind cosine-threshold graph.

Unlike the v1 reciprocal-nearest-neighbour screen, every pair with cosine >=
0.95 becomes an edge. Connected components are candidate multi-tracklet
families. Absolute depth is excluded from graph construction and added only to
separate post-hoc tables and plots. Components are not assumed to be cliques;
edge density and all-pairs minimum similarity expose single-linkage chaining.
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
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from testing import luke_kilosort_depthblind_tracklet_linkage_v1 as v1


OUT = ROOT / "testing/outputs/luke_kilosort_depthblind_families_v2"
FAMILY_THRESHOLD = 0.95
SENSITIVITY_THRESHOLDS = (0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99)
MAX_JOIN_GAP_S = 2.0


def proposed_family_colors(families: pd.DataFrame) -> dict[str, tuple]:
    palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("Dark2").colors)
    ordered = families.sort_values("waveform_review_order")["family_id"].tolist()
    return {family_id: palette[i % len(palette)] for i, family_id in enumerate(ordered)}


def compact_member_cids(member_cids: str, limit: int = 8) -> str:
    cids = str(member_cids).split()
    if len(cids) <= limit:
        return " ".join(cids)
    return " ".join(cids[:limit]) + f" … (+{len(cids) - limit} more)"


def threshold_components(similarity: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    adjacency = similarity >= threshold
    np.fill_diagonal(adjacency, False)
    _, labels = connected_components(csr_matrix(adjacency), directed=False)
    sizes = np.bincount(labels)
    return labels, sizes


def family_ids(labels: np.ndarray, sizes: np.ndarray, cids: np.ndarray) -> dict[int, str]:
    multi = [label for label, size in enumerate(sizes) if size > 1]
    multi.sort(key=lambda label: int(np.min(cids[labels == label])))
    return {label: f"F{i + 1:03d}" for i, label in enumerate(multi)}


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def chronological_segments(times_s: np.ndarray, max_gap_s: float) -> list[np.ndarray]:
    """Indices for chronological line segments, broken at long time gaps."""
    if len(times_s) == 0:
        return []
    return [segment for segment in np.split(np.arange(len(times_s)), np.flatnonzero(np.diff(times_s) > max_gap_s) + 1) if len(segment)]


def build_waveform_families(
    eligible: pd.DataFrame,
    similarity: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cids = eligible["cid"].to_numpy(int)
    labels, sizes = threshold_components(similarity, FAMILY_THRESHOLD)
    ids = family_ids(labels, sizes, cids)
    member_rows: list[dict] = []
    edge_rows: list[dict] = []
    family_rows: list[dict] = []
    for label, family_id in ids.items():
        index = np.flatnonzero(labels == label)
        sub = similarity[np.ix_(index, index)]
        upper = sub[np.triu_indices(len(index), 1)]
        direct = upper >= FAMILY_THRESHOLD
        possible = len(upper)
        outside = np.flatnonzero(labels != label)
        external_max = float(np.max(similarity[np.ix_(index, outside)])) if len(outside) else np.nan
        member_cids = cids[index]
        for local_i, global_i in enumerate(index):
            inside_scores = sub[local_i].copy()
            inside_scores[local_i] = -np.inf
            member_rows.append(
                {
                    "family_id": family_id,
                    "cid": int(cids[global_i]),
                    "KSLabel": eligible.loc[global_i, "KSLabel"],
                    "snippet_spikes": int(eligible.loc[global_i, "snippet_spikes"]),
                    "local_energy_fraction": float(eligible.loc[global_i, "local_energy_fraction"]),
                    "template_peak_abs_ks_units": float(eligible.loc[global_i, "template_peak_abs_ks_units"]),
                    "direct_family_degree": int(np.count_nonzero(inside_scores >= FAMILY_THRESHOLD)),
                    "closest_family_cid": int(member_cids[np.argmax(inside_scores)]),
                    "closest_family_cosine": float(np.max(inside_scores)),
                }
            )
        for ai, bi in zip(*np.triu_indices(len(index), 1)):
            if sub[ai, bi] >= FAMILY_THRESHOLD:
                edge_rows.append(
                    {
                        "family_id": family_id,
                        "cid_a": int(member_cids[ai]),
                        "cid_b": int(member_cids[bi]),
                        "cosine": float(sub[ai, bi]),
                    }
                )
        labels_in_family = eligible.iloc[index]["KSLabel"]
        family_rows.append(
            {
                "family_id": family_id,
                "member_count": int(len(index)),
                "member_cids": " ".join(map(str, member_cids)),
                "good_count": int((labels_in_family == "good").sum()),
                "mua_count": int((labels_in_family == "mua").sum()),
                "total_snippet_spikes": int(eligible.iloc[index]["snippet_spikes"].sum()),
                "direct_edge_count": int(direct.sum()),
                "possible_edge_count": int(possible),
                "edge_density": float(direct.mean()),
                "all_pairs_min_cosine": float(np.min(upper)),
                "all_pairs_q25_cosine": float(np.quantile(upper, 0.25)),
                "all_pairs_median_cosine": float(np.median(upper)),
                "all_pairs_max_cosine": float(np.max(upper)),
                "max_cosine_to_outside_template": external_max,
                "complete_link_at_0p95": bool(np.all(direct)),
            }
        )
    members = pd.DataFrame(member_rows)
    edges = pd.DataFrame(edge_rows)
    families = pd.DataFrame(family_rows)
    families = families.sort_values(
        ["good_count", "edge_density", "member_count", "total_snippet_spikes"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    families.insert(1, "waveform_review_order", np.arange(1, len(families) + 1))
    return members, edges, families


def reveal_depth_and_time(
    members: pd.DataFrame,
    families: pd.DataFrame,
    eligible: pd.DataFrame,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = eligible.set_index("cid")
    times = frames.astype(np.float64) / fs
    bins = np.arange(v1.INTERVAL_S[0], v1.INTERVAL_S[1] + 1.0, 1.0)
    member_rows = []
    family_rows = []
    for family in families.itertuples(index=False):
        family_members = members[members["family_id"] == family.family_id]
        cids = family_members["cid"].to_numpy(int)
        count_vectors = []
        all_times = []
        all_depths = []
        for member in family_members.itertuples(index=False):
            cid = int(member.cid)
            take = event_cids == cid
            member_times = times[take]
            member_depths = xy[take, 1]
            count_vector = np.histogram(member_times, bins=bins)[0]
            count_vectors.append(count_vector)
            all_times.append(member_times)
            all_depths.append(member_depths)
            member_rows.append(
                {
                    **member._asdict(),
                    "template_peak_x_um": float(metadata.loc[cid, "template_peak_x_um"]),
                    "template_peak_depth_um": float(metadata.loc[cid, "template_peak_depth_um"]),
                    "spike_depth_median_um": float(np.median(member_depths)),
                    "spike_depth_p05_um": float(np.quantile(member_depths, 0.05)),
                    "spike_depth_p95_um": float(np.quantile(member_depths, 0.95)),
                    "median_time_s": float(np.median(member_times)),
                    "active_one_second_bins": int(np.count_nonzero(count_vector)),
                }
            )
        count_vectors = np.asarray(count_vectors)
        pair_correlations = [
            safe_correlation(count_vectors[i], count_vectors[j])
            for i in range(len(cids)) for j in range(i + 1, len(cids))
        ]
        merged_times = np.sort(np.concatenate(all_times))
        merged_depths = np.concatenate(all_depths)
        rate_hz = len(merged_times) / (v1.INTERVAL_S[1] - v1.INTERVAL_S[0])
        expected_short = 1 - np.exp(-rate_hz * v1.REFRACTORY_S)
        observed_short = v1.violation_fraction(merged_times, v1.REFRACTORY_S)
        template_depths = metadata.loc[cids, "template_peak_depth_um"].to_numpy(float)
        family_rows.append(
            {
                **family._asdict(),
                "template_depth_min_um": float(np.min(template_depths)),
                "template_depth_max_um": float(np.max(template_depths)),
                "template_depth_span_um": float(np.ptp(template_depths)),
                "spike_depth_p05_um": float(np.quantile(merged_depths, 0.05)),
                "spike_depth_p95_um": float(np.quantile(merged_depths, 0.95)),
                "spike_depth_p90_span_um": float(np.quantile(merged_depths, 0.95) - np.quantile(merged_depths, 0.05)),
                "family_active_one_second_bins": int(np.count_nonzero(np.sum(count_vectors, axis=0))),
                "median_pairwise_one_second_count_correlation": float(np.nanmedian(pair_correlations)),
                "max_pairwise_one_second_count_correlation": float(np.nanmax(pair_correlations)),
                "merged_refractory_violation_fraction_1ms": observed_short,
                "expected_poisson_short_isi_fraction_1ms": expected_short,
                "merged_to_poisson_short_isi_ratio": observed_short / expected_short if expected_short > 0 else np.nan,
            }
        )
    return pd.DataFrame(member_rows), pd.DataFrame(family_rows)


def save_figures(
    sensitivity: pd.DataFrame,
    families: pd.DataFrame,
    members_revealed: pd.DataFrame,
    eligible: pd.DataFrame,
    waves: np.ndarray,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(sensitivity["cosine_threshold"], sensitivity["multi_family_count"], "o-", color="#286EA6")
    axes[0].set(xlabel="Cosine edge threshold", ylabel="Multi-template families", title="Family count versus waveform threshold")
    axes[1].plot(sensitivity["cosine_threshold"], sensitivity["templates_in_families"], "o-", label="templates in families", color="#286EA6")
    axes[1].plot(sensitivity["cosine_threshold"], sensitivity["largest_family_size"], "o-", label="largest family", color="#D17A22")
    axes[1].set(xlabel="Cosine edge threshold", ylabel="Template count", title="Family coverage and largest component")
    for ax in axes:
        ax.axvline(FAMILY_THRESHOLD, color="0.35", ls="--", lw=1)
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.savefig(OUT / "01_component_threshold_sensitivity.png", dpi=180)
    fig.savefig(OUT / "01_component_threshold_sensitivity.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    point_size = 25 + 12 * families["member_count"]
    scatter = ax.scatter(
        families["edge_density"], families["all_pairs_min_cosine"], s=point_size,
        c=families["good_count"], cmap="Blues", edgecolor="0.2", linewidth=0.5,
    )
    for row in families.itertuples(index=False):
        ax.annotate(row.family_id, (row.edge_density, row.all_pairs_min_cosine), xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.axhline(FAMILY_THRESHOLD, color="0.35", ls="--", lw=1, label="all pairs also ≥0.95")
    ax.set(
        xlabel="Direct-edge density within connected component",
        ylabel="Minimum all-pairs cosine",
        title="Waveform-only family coherence at cosine-edge threshold 0.95",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Kilosort-good members")
    fig.savefig(OUT / "02_family_waveform_coherence.png", dpi=180)
    fig.savefig(OUT / "02_family_waveform_coherence.pdf")
    plt.close(fig)

    ordered = families.sort_values("waveform_review_order")
    times = frames.astype(np.float64) / fs
    family_colors = proposed_family_colors(families)
    with PdfPages(OUT / "03_all_family_depth_time_atlas.pdf") as pdf:
        for page_start in range(0, len(ordered), 6):
            page = ordered.iloc[page_start:page_start + 6]
            fig, axes = plt.subplots(len(page), 1, figsize=(14, 2.25 * len(page)), sharex=True, layout="constrained", squeeze=False)
            for ax, family in zip(axes[:, 0], page.itertuples(index=False)):
                family_members = members_revealed[members_revealed["family_id"] == family.family_id]
                for member in family_members.itertuples(index=False):
                    take = event_cids == int(member.cid)
                    ax.scatter(times[take], xy[take, 1], s=3, alpha=0.60, color=family_colors[family.family_id], linewidths=0, rasterized=True)
                ax.set_ylabel("depth (µm)")
                ax.set_title(
                    f"{family.family_id}: n={family.member_count}, good={family.good_count}, density={family.edge_density:.2f}, "
                    f"all-pair min={family.all_pairs_min_cosine:.3f} · member CIDs {compact_member_cids(family.member_cids)}", fontsize=9
                )
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            axes[-1, 0].set_xlabel("Recording time (s)")
            fig.suptitle(
                "Depth/time revealed after cosine ≥0.95 waveform-family construction\n"
                "Order uses waveform coherence, Kilosort label, and support—not depth",
                fontsize=12,
            )
            pdf.savefig(fig, dpi=180)
            page_number = page_start // 6 + 1
            fig.savefig(OUT / f"03_family_depth_time_atlas_page{page_number}.png", dpi=180)
            plt.close(fig)

    shortlist = families[
        families["complete_link_at_0p95"]
        & (families["member_count"] == 2)
        & (families["good_count"] >= 1)
        & (families["merged_to_poisson_short_isi_ratio"] < 3)
    ].sort_values(["good_count", "all_pairs_min_cosine"], ascending=False)
    if shortlist.empty:
        return
    wave_by_cid = {int(cid): waves[i] for i, cid in enumerate(eligible["cid"])}
    fig, axes = plt.subplots(len(shortlist), 4, figsize=(15, 3.15 * len(shortlist)), layout="constrained", squeeze=False)
    for row_axes, family in zip(axes, shortlist.itertuples(index=False)):
        family_members = members_revealed[members_revealed["family_id"] == family.family_id]
        cids = family_members["cid"].to_numpy(int)
        ax = row_axes[0]
        for cid in cids:
            take = event_cids == cid
            ax.scatter(times[take], xy[take, 1], s=7, alpha=0.65, color=family_colors[family.family_id], linewidths=0)
        ax.set(xlim=v1.INTERVAL_S, xlabel="time (s)", ylabel="cached depth (µm)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        a, b = wave_by_cid[cids[0]], wave_by_cid[cids[1]]
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        for panel, data, title in [
            (row_axes[1], a_norm, f"CID {cids[0]} normalized"),
            (row_axes[2], b_norm, f"CID {cids[1]} normalized"),
            (row_axes[3], a_norm - b_norm, "normalized difference"),
        ]:
            limit = float(max(np.max(np.abs(data)), np.finfo(float).eps))
            panel.imshow(data, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            panel.set(xlabel="relative channel", ylabel="centered sample", title=title)
        row_axes[0].set_title(
            f"{family.family_id}: cos={family.all_pairs_min_cosine:.3f}, good={family.good_count}/2, "
            f"template span={family.template_depth_span_um:.0f} µm, merged/Poisson={family.merged_to_poisson_short_isi_ratio:.2f} · CIDs {family.member_cids}",
            fontsize=8,
        )
    fig.suptitle(
        "Post-hoc review shortlist from cosine ≥0.95 complete-link pairs\n"
        "At least one Kilosort-good member and merged short-ISI ratio <3; sparse pairs remain underpowered",
        fontsize=12,
    )
    fig.savefig(OUT / "04_low_refractory_good_member_pairs.png", dpi=180)
    fig.savefig(OUT / "04_low_refractory_good_member_pairs.pdf")
    plt.close(fig)


def save_individual_family_scatters(
    families: pd.DataFrame,
    members_revealed: pd.DataFrame,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    """One unbinned per-spike time/depth scatter per waveform family."""
    scatter_dir = OUT / "family_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    times = frames.astype(np.float64) / fs
    family_colors = proposed_family_colors(families)
    index_rows = []
    for family in families.sort_values("waveform_review_order").itertuples(index=False):
        family_members = members_revealed[
            members_revealed["family_id"] == family.family_id
        ].sort_values("template_peak_depth_um")
        total = int(family_members["snippet_spikes"].sum())
        point_size = 10 if total < 300 else 5 if total < 2_000 else 2
        fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")
        plotted_depths = []
        for member in family_members.itertuples(index=False):
            cid = int(member.cid)
            take = event_cids == cid
            depths = xy[take, 1]
            plotted_depths.append(depths)
            ax.scatter(
                times[take], depths, s=point_size, alpha=0.72,
                color=family_colors[family.family_id], linewidths=0,
                rasterized=True,
            )
        depths = np.concatenate(plotted_depths)
        padding = max(10.0, 0.04 * float(np.ptp(depths)))
        ax.set(
            xlim=v1.INTERVAL_S,
            ylim=(float(np.min(depths) - padding), float(np.max(depths) + padding)),
            xlabel="Recording time (s)",
            ylabel="Kilosort per-spike depth (µm)",
            title=(
                f"{family.family_id} — proposed-family per-spike depth\n"
                f"930–1,030 s · cosine-edge threshold ≥{FAMILY_THRESHOLD:g} · "
                f"{family.member_count} source CIDs ({compact_member_cids(family.member_cids)}) · {total:,} spikes · all-pairs minimum {family.all_pairs_min_cosine:.3f}"
            ),
        )
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(color="0.88", linewidth=0.6)
        ax.text(
            0.0, -0.16,
            "Every marker is one cached spike_position; no binning, interpolation, trajectory, or motion estimate.",
            transform=ax.transAxes, fontsize=8, color="0.3",
        )
        png = scatter_dir / f"{family.family_id}_per_spike_depth.png"
        fig.savefig(png, dpi=180)
        plt.close(fig)
        index_rows.append(
            {
                "waveform_review_order": int(family.waveform_review_order),
                "family_id": family.family_id,
                "member_count": int(family.member_count),
                "member_cids": family.member_cids,
                "total_spikes": total,
                "family_color": to_hex(family_colors[family.family_id]),
                "relative_png": str(png.relative_to(OUT)),
            }
        )
    return pd.DataFrame(index_rows)


def save_combined_proposed_family_scatter(
    families: pd.DataFrame,
    members_revealed: pd.DataFrame,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> None:
    """All proposed families together, with one fixed color per family."""
    family_colors = proposed_family_colors(families)
    times = frames.astype(np.float64) / fs
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True, layout="constrained")
    bands = [(0, 960), (960, 1920), (1920, 2880), (2880, 3840)]
    for family in families.sort_values("waveform_review_order").itertuples(index=False):
        cids = members_revealed.loc[members_revealed["family_id"] == family.family_id, "cid"].to_numpy(int)
        take_family = np.isin(event_cids, cids)
        for ax, (lo, hi) in zip(axes, bands):
            take = take_family & (xy[:, 1] >= lo) & (xy[:, 1] < hi)
            if np.any(take):
                ax.scatter(
                    times[take], xy[take, 1], s=1.8, alpha=0.65,
                    color=family_colors[family.family_id], linewidths=0,
                    rasterized=True,
                )
    for ax, (lo, hi) in zip(axes, bands):
        ax.set(ylim=(lo, hi), ylabel="depth (µm)")
        ax.grid(color="0.9", linewidth=0.5)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[-1].set(xlim=v1.INTERVAL_S, xlabel="Recording time (s)")
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=5, color=family_colors[row.family_id], label=row.family_id)
        for row in families.sort_values("waveform_review_order").itertuples(index=False)
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, ncol=2, title="Proposed family")
    fig.suptitle(
        "Proposed waveform families — one point per spike, colored by family\n"
        "930–1,030 s · cosine-edge threshold ≥0.95 · source CIDs merged for display · Kilosort per-spike depth",
        fontsize=14,
    )
    fig.savefig(OUT / "05_all_proposed_families_per_spike_depth.png", dpi=180)
    fig.savefig(OUT / "05_all_proposed_families_per_spike_depth.pdf")
    plt.close(fig)


def save_joined_family_tracks(
    families: pd.DataFrame,
    members_revealed: pd.DataFrame,
    frames: np.ndarray,
    event_cids: np.ndarray,
    xy: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    """Chronologically join spikes within each proposed family, with gap breaks."""
    joined_dir = OUT / "family_joined"
    joined_dir.mkdir(parents=True, exist_ok=True)
    family_colors = proposed_family_colors(families)
    all_times = frames.astype(np.float64) / fs
    index_rows = []
    with PdfPages(OUT / "06_joined_proposed_family_tracks.pdf") as pdf:
        for family in families.sort_values("waveform_review_order").itertuples(index=False):
            cids = members_revealed.loc[
                members_revealed["family_id"] == family.family_id, "cid"
            ].to_numpy(int)
            take = np.flatnonzero(np.isin(event_cids, cids))
            order = np.argsort(frames[take], kind="stable")
            take = take[order]
            times = all_times[take]
            depths = xy[take, 1]
            ordered_cids = event_cids[take]
            gaps = np.diff(times)
            valid = gaps <= MAX_JOIN_GAP_S
            cross_cid = valid & (np.diff(ordered_cids) != 0)
            depth_jumps = np.abs(np.diff(depths))
            cross_jumps = depth_jumps[cross_cid]

            fig, ax = plt.subplots(figsize=(13, 5.5), layout="constrained")
            for segment in chronological_segments(times, MAX_JOIN_GAP_S):
                if len(segment) > 1:
                    ax.plot(
                        times[segment], depths[segment], color=family_colors[family.family_id],
                        lw=0.55, alpha=0.30, solid_capstyle="round", rasterized=True,
                    )
            point_size = 10 if len(take) < 300 else 5 if len(take) < 2_000 else 2
            ax.scatter(
                times, depths, s=point_size, color=family_colors[family.family_id],
                alpha=0.78, linewidths=0, rasterized=True,
            )
            padding = max(10.0, 0.04 * float(np.ptp(depths)))
            ax.set(
                xlim=v1.INTERVAL_S,
                ylim=(float(np.min(depths) - padding), float(np.max(depths) + padding)),
                xlabel="Recording time (s)",
                ylabel="Kilosort per-spike depth (µm)",
                title=(
                    f"{family.family_id} — chronological spike joins across proposed family\n"
                    f"source CIDs {compact_member_cids(family.member_cids)} · {len(take):,} spikes · lines join consecutive spikes only when Δt ≤{MAX_JOIN_GAP_S:g} s"
                ),
            )
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.grid(color="0.88", linewidth=0.6)
            ax.text(
                0.0, -0.16,
                "Lines are a display of chronological adjacency, not an interpolated trajectory or motion estimate; rapid vertical zigzags argue against merging.",
                transform=ax.transAxes, fontsize=8, color="0.3",
            )
            png = joined_dir / f"{family.family_id}_joined_spikes.png"
            fig.savefig(png, dpi=180)
            pdf.savefig(fig, dpi=180)
            plt.close(fig)
            index_rows.append(
                {
                    "waveform_review_order": int(family.waveform_review_order),
                    "family_id": family.family_id,
                    "member_cids": family.member_cids,
                    "spikes": int(len(take)),
                    "joined_intervals_le_2s": int(valid.sum()),
                    "cross_cid_joined_intervals": int(cross_cid.sum()),
                    "cross_cid_depth_jump_median_um": float(np.median(cross_jumps)) if len(cross_jumps) else np.nan,
                    "cross_cid_depth_jump_p95_um": float(np.quantile(cross_jumps, 0.95)) if len(cross_jumps) else np.nan,
                    "cross_cid_depth_jumps_ge_40_um": int(np.count_nonzero(cross_jumps >= 40)),
                    "cross_cid_depth_jumps_ge_100_um": int(np.count_nonzero(cross_jumps >= 100)),
                    "family_color": to_hex(family_colors[family.family_id]),
                    "relative_png": str(png.relative_to(OUT)),
                }
            )
    return pd.DataFrame(index_rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    similarity, best_lag = v1.best_lagged_cosine(waves)

    sensitivity_rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        labels, sizes = threshold_components(similarity, threshold)
        multi = sizes[sizes > 1]
        sensitivity_rows.append(
            {
                "cosine_threshold": threshold,
                "multi_family_count": int(len(multi)),
                "templates_in_families": int(multi.sum()),
                "largest_family_size": int(multi.max()) if len(multi) else 0,
                "direct_edges": int(np.count_nonzero(np.triu(similarity >= threshold, 1))),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    members, edges, families = build_waveform_families(eligible, similarity)
    members_revealed, families_revealed = reveal_depth_and_time(
        members, families, eligible, frames, event_cids, xy, fs
    )

    sensitivity.to_csv(OUT / "component_threshold_sensitivity.csv", index=False)
    members.to_csv(OUT / "family_members_waveform_only.csv", index=False)
    edges.to_csv(OUT / "family_edges_waveform_only.csv", index=False)
    families.to_csv(OUT / "family_summary_waveform_only.csv", index=False)
    members_revealed.to_csv(OUT / "family_members_depth_time_revealed.csv", index=False)
    families_revealed.to_csv(OUT / "family_summary_depth_time_revealed.csv", index=False)
    np.savez_compressed(
        OUT / "waveform_similarity_matrix.npz",
        cid=eligible["cid"].to_numpy(int), cosine=similarity, best_lag_samples=best_lag,
    )
    shortlist = families_revealed[
        families_revealed["complete_link_at_0p95"]
        & (families_revealed["member_count"] == 2)
        & (families_revealed["good_count"] >= 1)
        & (families_revealed["merged_to_poisson_short_isi_ratio"] < 3)
    ].copy()
    shortlist.to_csv(OUT / "posthoc_review_shortlist.csv", index=False)
    save_figures(sensitivity, families_revealed, members_revealed, eligible, waves, frames, event_cids, xy, fs)
    scatter_index = save_individual_family_scatters(
        families_revealed, members_revealed, frames, event_cids, xy, fs
    )
    scatter_index.to_csv(OUT / "family_scatter_index.csv", index=False)
    save_combined_proposed_family_scatter(
        families_revealed, members_revealed, frames, event_cids, xy, fs
    )
    joined_index = save_joined_family_tracks(
        families_revealed, members_revealed, frames, event_cids, xy, fs
    )
    joined_index.to_csv(OUT / "family_joined_index.csv", index=False)

    complete = int(families["complete_link_at_0p95"].sum())
    all_good = int((families["good_count"] == families["member_count"]).sum())
    summary = {
        "status": "complete",
        "purpose": "depth-blind multi-tracklet family screen from cached final Kilosort templates",
        "raw_voltage_read": False,
        "new_spike_sort_run": False,
        "interval_s": list(v1.INTERVAL_S),
        "eligible_templates": int(len(eligible)),
        "family_edge_rule": f"signed relative-waveform cosine >= {FAMILY_THRESHOLD}; no reciprocal-nearest restriction",
        "multi_template_families": int(len(families)),
        "templates_in_families": int(families["member_count"].sum()),
        "largest_family_size": int(families["member_count"].max()),
        "complete_link_families": complete,
        "chained_families": int(len(families) - complete),
        "all_good_families": all_good,
        "posthoc_low_refractory_good_member_pair_shortlist": int(len(shortlist)),
        "individual_family_scatterplots": int(len(scatter_index)),
        "joined_family_plots": int(len(joined_index)),
        "joined_line_max_gap_s": MAX_JOIN_GAP_S,
        "visual_review_leads": {
            "F010": "strongest cached lead: CIDs 308/316, both Kilosort-good, cosine 0.959, adjacent 40-um template origins, 155 events; plausible local split but not identity-validated",
            "F007": "large-excursion lead: CIDs 216/231, both Kilosort-good, cosine 0.985, 120-um template span, only 78 events",
            "F011": "large-excursion lead: CIDs 356/399, both Kilosort-good, cosine 0.984, 160-um template span, only 70 events",
        },
        "identity_excluded_features": "absolute depth, per-spike depth, spike time, motion estimate",
        "posthoc_reveal": "template depth, Kilosort per-spike depth/time, temporal correlation, merged refractory statistic",
        "interpretation_status": "candidate families for visual review; none automatically promoted as a biological identity",
        "limitations": [
            "Connected components use single-linkage and may chain templates whose mutual cosine is below 0.95; edge density and all-pairs minima are retained.",
            "Templates and CIDs are final Kilosort outputs learned from the full recording, not snippet-trained pre-clustering tracklets.",
            "Per-spike depth is Kilosort PC-feature-weighted position rather than independent raw-voltage localization.",
            "Exact 40-um recentering does not address intermediate spatial phase or unassigned spikes.",
        ],
        "source_sha256": {
            str(v1.EVENTS): sha256(v1.EVENTS),
            str(v1.SOURCE / "templates.npy"): sha256(v1.SOURCE / "templates.npy"),
            str(v1.SOURCE / "channel_positions.npy"): sha256(v1.SOURCE / "channel_positions.npy"),
        },
        "script_sha256": sha256(Path(__file__)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = "# Depth-blind Kilosort template families at cosine >=0.95\n\n"
    readme += "This v2 screen removes the v1 reciprocal-nearest restriction. Every qualifying waveform edge participates, so one family can contain many cached CIDs. Absolute depth and time are added only after the graph is frozen.\n\n"
    readme += f"- **{len(families)}** multi-template families contain **{families['member_count'].sum()}** of {len(eligible)} eligible templates.\n"
    readme += f"- Largest family: **{families['member_count'].max()}** templates.\n"
    readme += f"- Complete-link families: **{complete}**; chained families: **{len(families) - complete}**.\n"
    readme += f"- Families containing only Kilosort-good members: **{all_good}**.\n\n"
    readme += f"- Post-hoc low-refractory two-member review shortlist: **{len(shortlist)}** families; this is not an identity acceptance rule.\n\n"
    readme += "## Visual-review leads\n\n"
    readme += "- **F010 (CIDs 308/316)** is the strongest cached lead: both are Kilosort-good, cosine 0.959, their template origins are 40 um apart, and their 155 combined events span 66 um from p05 to p95. It could be one locally split unit, but the refractory check has little power at this event count.\n"
    readme += "- **F007 (216/231)** and **F011 (356/399)** are the interesting large-span leads: cosine 0.985/0.984 and template spans 120/160 um, but only 78/70 events. They remain nominations, not tracked cells.\n"
    readme += "- F002 demonstrates the chaining hazard: CID 79 links to 319, which links to 379, although the end templates have cosine 0.923 and lie at implausibly separated depths.\n\n"
    readme += "## Per-spike family scatterplots\n\n"
    readme += "`05_all_proposed_families_per_spike_depth.*` overlays all grouped spikes using one fixed color per proposed family. `family_scatter/` contains one single-color PNG per family; source CIDs appear only in the title. Every marker is one spike at `spike_positions.npy[:,1]`, with no connecting lines or binned summaries. `family_scatter_index.csv` maps family colors, review order, and source CIDs to filenames.\n\n"
    readme += "## Chronologically joined family plots\n\n"
    readme += f"`family_joined/` contains one plot per family with consecutive spikes joined across member CIDs when their time gap is at most {MAX_JOIN_GAP_S:g} s. `06_joined_proposed_family_tracks.pdf` collects all 24 plots. Lines are display adjacency only—not interpolation or a motion estimate—and remain deliberately jagged so implausible rapid depth switching is visible. `family_joined_index.csv` records cross-CID jump counts and sizes.\n\n"
    readme += "Start with `family_summary_waveform_only.csv` and `02_family_waveform_coherence.*`, then inspect every family in `03_all_family_depth_time_atlas.pdf`. A connected component is a candidate inventory, not proof that every member is the same cell.\n"
    (OUT / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
