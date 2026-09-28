#!/usr/bin/env python
"""Cached phase/global-rival audit for the imec1 sorter-free lighthouse bank.

This is a method audit, not a motion comparison. It reads only the frozen seed
templates and seed-event table from v2. Absolute depth is revealed only after
waveform-only groups have been frozen. No sorter product or motion estimate is
read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from testing.luke_imec1_dots_raw_lighthouse_check import OFFSETS, geometry_and_mapping, patches

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v2"
DEFAULT_OUTPUT = ROOT / "testing/outputs/luke_imec1_sorterfree_phase_audit_v1"
THRESHOLDS = (0.90, 0.95, 0.97)
LAGS = range(-3, 4)


def phase_centered_geometry(relative_geometry: np.ndarray, phase: int) -> np.ndarray:
    """Channel coordinates relative to the detected peak channel."""
    return relative_geometry - relative_geometry[int(phase)]


def overlap_indices(relative_geometry: np.ndarray, phase_a: int, phase_b: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact common channel coordinates after peak-channel recentering."""
    a = phase_centered_geometry(relative_geometry, phase_a)
    b = phase_centered_geometry(relative_geometry, phase_b)
    lookup = {tuple(v): i for i, v in enumerate(b.tolist())}
    pairs = [(i, lookup[tuple(v)]) for i, v in enumerate(a.tolist()) if tuple(v) in lookup]
    if not pairs:
        return np.empty(0, int), np.empty(0, int)
    ia, ib = zip(*pairs)
    return np.asarray(ia), np.asarray(ib)


def lagged_overlap_cosine(
    wave_a: np.ndarray,
    phase_a: int,
    wave_b: np.ndarray,
    phase_b: int,
    relative_geometry: np.ndarray,
) -> tuple[float, float, int]:
    """Best lagged cosine and minimum bidirectional energy coverage."""
    ia, ib = overlap_indices(relative_geometry, phase_a, phase_b)
    if len(ia) < 8:
        return np.nan, 0.0, 0
    total_a = float(np.square(wave_a).sum())
    total_b = float(np.square(wave_b).sum())
    ea = float(np.square(wave_a[:, ia]).sum())
    eb = float(np.square(wave_b[:, ib]).sum())
    coverage = min(ea / max(total_a, 1e-20), eb / max(total_b, 1e-20))
    best = -np.inf
    for lag in LAGS:
        if lag < 0:
            a = wave_a[-lag:, ia]
            b = wave_b[:lag, ib]
        elif lag > 0:
            a = wave_a[:-lag, ia]
            b = wave_b[lag:, ib]
        else:
            a = wave_a[:, ia]
            b = wave_b[:, ib]
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator:
            best = max(best, float(np.vdot(a, b) / denominator))
    return best, coverage, len(ia)


def similarity_matrices(templates: np.ndarray, phases: np.ndarray, relative_geometry: np.ndarray):
    n = len(templates)
    cosine = np.eye(n, dtype=float)
    coverage = np.ones((n, n), dtype=float)
    common_channels = np.full((n, n), templates.shape[2], dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            score, support, count = lagged_overlap_cosine(
                templates[i], int(phases[i]), templates[j], int(phases[j]), relative_geometry
            )
            cosine[i, j] = cosine[j, i] = score
            coverage[i, j] = coverage[j, i] = support
            common_channels[i, j] = common_channels[j, i] = count
    return cosine, coverage, common_channels


def complete_link_labels(cosine: np.ndarray, threshold: float) -> np.ndarray:
    safe = np.nan_to_num(cosine, nan=-1.0)
    safe = np.clip((safe + safe.T) / 2, -1, 1)
    np.fill_diagonal(safe, 1.0)
    distance = squareform(np.clip(1 - safe, 0, 2), checks=False)
    return fcluster(linkage(distance, method="complete"), 1 - threshold, criterion="distance")


def run(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    families = pd.read_csv(source / "families_after_depth_reveal.csv")
    seed = pd.read_csv(source / "seed_cluster_events.csv")
    with np.load(source / "family_templates.npz") as saved:
        templates = np.asarray(saved["waveforms"], dtype=float)
        relative_geometry = np.asarray(saved["relative_geometry"], dtype=float)
    phases = families.phase_id.to_numpy(int)
    cosine, coverage, common = similarity_matrices(templates, phases, relative_geometry)

    rows = []
    for i, family in families.iterrows():
        rivals = np.arange(len(families)) != i
        same = rivals & (phases == phases[i])
        cross = rivals & (phases != phases[i])
        same_index = np.flatnonzero(same)[np.nanargmax(cosine[i, same])]
        cross_index = np.flatnonzero(cross)[np.nanargmax(cosine[i, cross])]
        rows.append({
            "family_id": family.family_id,
            "phase_id": int(family.phase_id),
            "selected_depth_blind": bool(family.selected_depth_blind),
            "same_phase_rival_id": families.family_id.iloc[same_index],
            "same_phase_rival_cosine_recomputed": float(cosine[i, same_index]),
            "cross_phase_rival_id": families.family_id.iloc[cross_index],
            "cross_phase_rival_phase": int(phases[cross_index]),
            "cross_phase_rival_cosine": float(cosine[i, cross_index]),
            "cross_phase_overlap_energy_fraction": float(coverage[i, cross_index]),
            "cross_phase_common_channels": int(common[i, cross_index]),
            "global_rival_id": families.family_id.iloc[np.flatnonzero(rivals)[np.nanargmax(cosine[i, rivals])]],
            "global_rival_cosine": float(np.nanmax(cosine[i, rivals])),
            "seed_depth_p90_span_um": float(family.seed_depth_p90_span_um),
        })
    rival_table = pd.DataFrame(rows)
    rival_table["cross_phase_alias_ge_0p95"] = rival_table.cross_phase_rival_cosine >= 0.95
    rival_table.to_csv(output / "global_rival_audit.csv", index=False)

    memberships = []
    group_summaries = []
    for threshold in THRESHOLDS:
        labels = complete_link_labels(cosine, threshold)
        for i, label in enumerate(labels):
            memberships.append({"threshold": threshold, "group": int(label), "family_id": families.family_id.iloc[i]})
        mapped = seed.assign(group=seed.family_id.map(dict(zip(families.family_id, labels))))
        for label, members in families.assign(group=labels).groupby("group"):
            observations = mapped[mapped.group == label]
            q05 = float(observations.waveform_centroid_um.quantile(0.05))
            q95 = float(observations.waveform_centroid_um.quantile(0.95))
            group_summaries.append({
                "threshold": threshold,
                "group": int(label),
                "templates": len(members),
                "phases": ";".join(map(str, sorted(members.phase_id.unique()))),
                "selected_templates": int(members.selected_depth_blind.sum()),
                "seed_events": len(observations),
                "seed_depth_p90_span_um": q95 - q05,
                "seed_spatially_plausible_le_120um": bool(q95 - q05 <= 120),
            })
    pd.DataFrame(memberships).to_csv(output / "complete_link_memberships.csv", index=False)
    groups = pd.DataFrame(group_summaries)
    groups.to_csv(output / "complete_link_group_audit_after_depth_reveal.csv", index=False)
    np.savez_compressed(output / "phase_centered_similarity.npz", cosine=cosine, overlap_energy_fraction=coverage,
                        common_channels=common, family_ids=families.family_id.to_numpy(), phases=phases,
                        relative_geometry=relative_geometry)

    selected = rival_table[rival_table.selected_depth_blind]
    summary = {
        "schema": "luke0804-imec1-sorterfree-phase-audit-v1",
        "status": "complete",
        "source": str(source.resolve()),
        "motion_estimate_read": False,
        "sorter_input_read": False,
        "absolute_depth_used_for_grouping": False,
        "proposal_templates": len(families),
        "selected_templates": int(families.selected_depth_blind.sum()),
        "selected_with_cross_phase_alias_ge_0p95": int(selected.cross_phase_alias_ge_0p95.sum()),
        "median_selected_cross_phase_rival_cosine": float(selected.cross_phase_rival_cosine.median()),
        "complete_link": {
            str(t): {
                "groups": int(groups[groups.threshold == t].group.nunique()),
                "multi_template_groups": int((groups[groups.threshold == t].templates > 1).sum()),
                "groups_with_selected_template": int((groups[groups.threshold == t].selected_templates > 0).sum()),
                "selected_groups_seed_plausible_le_120um": int(((groups.threshold == t) & (groups.selected_templates > 0) & groups.seed_spatially_plausible_le_120um).sum()),
            } for t in THRESHOLDS
        },
        "interpretation": "Cached method audit only. Cross-phase aliases were invisible to v2 ranking and matching; depth spans were revealed only after waveform-only groups were frozen.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.source, args.output)


if __name__ == "__main__":
    main()
