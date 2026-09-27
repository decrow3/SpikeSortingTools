#!/usr/bin/env python3
"""Analyze BC disjoint-spike spatial split halves on preregistered domains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def waveform_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Metrics after restricting the channel dimension; center each channel in time."""
    if left.size == 0:
        return {k: np.nan for k in (
            "left_energy", "right_energy", "cross_product", "difference_energy",
            "raw_cosine", "centered_cosine", "normalized_agreement", "difference_fraction",
        )}
    l = left.astype(np.float64, copy=False).ravel()
    r = right.astype(np.float64, copy=False).ravel()
    el, er = float(l @ l), float(r @ r)
    cross = float(l @ r)
    denom = el + er
    raw_cos = cross / np.sqrt(el * er) if el > 0 and er > 0 else np.nan
    lc = left.astype(np.float64, copy=False) - left.mean(axis=0, keepdims=True)
    rc = right.astype(np.float64, copy=False) - right.mean(axis=0, keepdims=True)
    lcf, rcf = lc.ravel(), rc.ravel()
    cel, cer = float(lcf @ lcf), float(rcf @ rcf)
    centered_cos = float(lcf @ rcf) / np.sqrt(cel * cer) if cel > 0 and cer > 0 else np.nan
    diff = float((l - r) @ (l - r))
    return {
        "left_energy": el,
        "right_energy": er,
        "cross_product": cross,
        "difference_energy": diff,
        "raw_cosine": raw_cos,
        "centered_cosine": centered_cos,
        "normalized_agreement": 2.0 * cross / denom if denom > 0 else np.nan,
        "difference_fraction": diff / denom if denom > 0 else np.nan,
    }


def membership_checks(path: Path, unit_ids: np.ndarray) -> dict:
    z = np.load(path, allow_pickle=False)
    assert np.array_equal(z["unit_ids"], unit_ids)
    out = []
    all_disjoint = True
    for i, unit in enumerate(unit_ids):
        lo0, lo1 = z["left_offsets"][i:i+2]
        ro0, ro1 = z["right_offsets"][i:i+2]
        left = z["left_sorting_row_indices"][lo0:lo1]
        right = z["right_sorting_row_indices"][ro0:ro1]
        overlap = np.intersect1d(left, right).size
        all_disjoint &= overlap == 0
        out.append({"unit_id": int(unit), "left_count": int(left.size),
                    "right_count": int(right.size), "overlap_count": int(overlap)})
    return {
        "all_disjoint": bool(all_disjoint),
        "left_count_min": int(min(x["left_count"] for x in out)),
        "left_count_max": int(max(x["left_count"] for x in out)),
        "right_count_min": int(min(x["right_count"] for x in out)),
        "right_count_max": int(max(x["right_count"] for x in out)),
        "rows": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", type=Path, required=True)
    ap.add_argument("--domains", type=Path, required=True)
    ap.add_argument("--qualification", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gate = json.loads((args.extract / "full_template_reproduction_gate.json").read_text())
    if not gate.get("pass") or not gate.get("interpretation_allowed"):
        raise RuntimeError("Full-template reproduction gate did not pass; interpretation forbidden")

    split_path = args.extract / "split_half_spatial_templates.npz"
    membership_path = args.extract / "split_half_membership.npz"
    split = np.load(split_path, allow_pickle=False)
    domains = np.load(args.domains, allow_pickle=False)
    left, right = split["left_templates"], split["right_templates"]
    unit_ids = split["unit_ids"]
    assert np.array_equal(unit_ids, domains["unit_ids"])
    assert np.array_equal(split["channel_ids"], domains["channel_ids"])
    assert split["statistical_independence"].item() == 0
    states = domains["occupied_states_um"]
    state_masks = domains["state_source_domain"]
    intersection_masks = domains["all_state_source_domain"]
    checks = membership_checks(membership_path, unit_ids)
    assert checks["all_disjoint"]

    rows = []
    for i, unit in enumerate(unit_ids):
        total_energy = float(np.sum(left[i].astype(float) ** 2 + right[i].astype(float) ** 2))
        for si, state in enumerate(states):
            inside = state_masks[i, si]
            for label, mask in (("inside", inside), ("outside", ~inside)):
                metrics = waveform_metrics(left[i, :, mask], right[i, :, mask])
                rows.append({
                    "unit_id": int(unit), "state_um": float(state), "domain": label,
                    "primary_per_state_domain": True, "channel_count": int(mask.sum()),
                    "combined_energy_fraction_of_full_probe":
                        float((metrics["left_energy"] + metrics["right_energy"]) / total_energy)
                        if total_energy > 0 else np.nan,
                    **metrics,
                })
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output / "per_state_inside_outside_metrics.csv", index=False)

    # Secondary only: all-state intersection, retained to describe the old AZ bottleneck.
    secondary_rows = []
    for i, unit in enumerate(unit_ids):
        for label, mask in (("inside", intersection_masks[i]), ("outside", ~intersection_masks[i])):
            secondary_rows.append({
                "unit_id": int(unit), "domain": label, "primary_per_state_domain": False,
                "channel_count": int(mask.sum()), **waveform_metrics(left[i, :, mask], right[i, :, mask]),
            })
    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv(args.output / "descriptive_all_state_intersection_metrics.csv", index=False)

    inside = detail.query("domain == 'inside'")
    per_unit = inside.groupby("unit_id").agg(
        state_count=("state_um", "size"),
        min_inside_channels=("channel_count", "min"),
        median_inside_channels=("channel_count", "median"),
        min_centered_cosine=("centered_cosine", "min"),
        median_centered_cosine=("centered_cosine", "median"),
        min_normalized_agreement=("normalized_agreement", "min"),
        median_normalized_agreement=("normalized_agreement", "median"),
        max_difference_fraction=("difference_fraction", "max"),
        median_inside_energy_fraction=("combined_energy_fraction_of_full_probe", "median"),
    ).reset_index()
    qual = json.loads(args.qualification.read_text())
    old_five = set(qual["native_position_all_state_pass_unit_ids"])
    per_unit["previously_qualified_five"] = per_unit.unit_id.isin(old_five)
    per_unit.to_csv(args.output / "per_donor_primary_summary.csv", index=False)

    placement = pd.read_csv(args.qualification.parent / "placement_attempts.csv")
    base_by_unit = dict(zip(unit_ids.tolist(), domains["best_base_shift_um"].tolist()))
    at_frozen_base = placement[
        placement.apply(lambda row: row.unit_id in base_by_unit
                        and row.base_shift_um == base_by_unit[row.unit_id], axis=1)
    ][["unit_id", "base_shift_um", "minimum_retained_energy_fraction",
       "minimum_roundtrip_centered_cosine", "support_operator_pass", "failure_reason"]]
    comparison = per_unit.merge(at_frozen_base, on="unit_id", validate="one_to_one")
    comparison.to_csv(args.output / "az_support_vs_bc_reliability.csv", index=False)

    # Full-cohort state summary, preserving per-state domains rather than intersecting states.
    state_summary = inside.groupby("state_um").agg(
        donors=("unit_id", "size"),
        channels_median=("channel_count", "median"),
        centered_cosine_median=("centered_cosine", "median"),
        centered_cosine_p10=("centered_cosine", lambda x: np.quantile(x, .1)),
        normalized_agreement_median=("normalized_agreement", "median"),
        normalized_agreement_p10=("normalized_agreement", lambda x: np.quantile(x, .1)),
        difference_fraction_median=("difference_fraction", "median"),
        inside_energy_fraction_median=("combined_energy_fraction_of_full_probe", "median"),
    ).reset_index()
    state_summary.to_csv(args.output / "full_cohort_state_summary.csv", index=False)

    # Spatial energy profiles for requested examples plus cohort median.
    eps = np.finfo(float).tiny
    l_energy = np.sum(left.astype(float) ** 2, axis=1)
    r_energy = np.sum(right.astype(float) ** 2, axis=1)
    agreement_ch = 2 * np.sum(left.astype(float) * right.astype(float), axis=1) / (l_energy + r_energy + eps)
    depth = split["geometry_um"][:, 1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    examples = [407, 13, 435, 70]
    for ax, unit in zip(axes.ravel(), examples):
        i = int(np.flatnonzero(unit_ids == unit)[0])
        denom = max(float(max(l_energy[i].max(), r_energy[i].max())), eps)
        ax.plot(depth, l_energy[i] / denom, color=OKABE_ITO["blue"], ls="-", lw=1.2, label="left")
        ax.plot(depth, r_energy[i] / denom, color=OKABE_ITO["vermillion"], ls="--", lw=1.2, label="right")
        ax2 = ax.twinx()
        visible = (l_energy[i] + r_energy[i]) >= .001 * float((l_energy[i] + r_energy[i]).max())
        ax2.plot(depth, np.where(visible, agreement_ch[i], np.nan), color=OKABE_ITO["green"],
                 ls=":", lw=1.0, alpha=.8, label="agreement (signal channels)")
        ax.set(title=f"unit {unit}", xlabel="depth (µm)", ylabel="normalized energy")
        ax2.set_ylabel("per-channel agreement")
        ax2.set_ylim(-1.05, 1.05)
        if ax is axes[0, 0]:
            lines = ax.lines + ax2.lines
            ax.legend(lines, [x.get_label() for x in lines], frameon=False, loc="upper left")
    fig.suptitle("BC disjoint-spike spatial profiles (shared temporal basis; descriptive)")
    fig.savefig(args.output / "example_spatial_split_profiles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for state, group in inside.groupby("state_um"):
        axes[0].plot([state], [group.centered_cosine.median()], marker="o", color=OKABE_ITO["blue"])
        axes[1].plot([state], [group.normalized_agreement.median()], marker="o", color=OKABE_ITO["vermillion"])
    axes[0].plot(state_summary.state_um, state_summary.centered_cosine_median,
                 color=OKABE_ITO["blue"], ls="-", label="median")
    axes[0].plot(state_summary.state_um, state_summary.centered_cosine_p10,
                 color=OKABE_ITO["sky"], ls="--", label="P10")
    axes[1].plot(state_summary.state_um, state_summary.normalized_agreement_median,
                 color=OKABE_ITO["vermillion"], ls="-", label="median")
    axes[1].plot(state_summary.state_um, state_summary.normalized_agreement_p10,
                 color=OKABE_ITO["orange"], ls="--", label="P10")
    for ax, title in zip(axes, ("Centered cosine", "Normalized agreement")):
        ax.set(xlabel="motion state (µm)", ylabel=title, title=title)
        ax.legend(frameon=False)
    fig.suptitle("Primary per-state frozen support domains, n=90")
    fig.savefig(args.output / "full_cohort_spatial_reliability.png", dpi=180)
    plt.close(fig)

    outside = detail.query("domain == 'outside'")
    result = {
        "schema": "luke-bc-spatial-reliability-analysis-v1",
        "status": "complete",
        "interpretation_allowed": True,
        "reproduction_gate": gate,
        "units": int(unit_ids.size),
        "states_um": states.tolist(),
        "membership": {k: v for k, v in checks.items() if k != "rows"},
        "estimator_provenance": {
            "disjoint_spike_membership": True,
            "shared_temporal_basis": True,
            "shared_preprocessing_and_common_reference": True,
            "statistically_independent_halves": False,
            "qualification": "Cross-products and split-half agreement are descriptive because estimator error can be correlated.",
        },
        "primary_domain": "Each donor's frozen state-specific support domain; seven states are evaluated separately.",
        "secondary_domain": "All-state intersection is descriptive only and was not substituted for AZ per-state retention.",
        "inside_primary": {
            "centered_cosine_median": float(inside.centered_cosine.median()),
            "centered_cosine_p10": float(inside.centered_cosine.quantile(.1)),
            "normalized_agreement_median": float(inside.normalized_agreement.median()),
            "normalized_agreement_p10": float(inside.normalized_agreement.quantile(.1)),
            "difference_fraction_median": float(inside.difference_fraction.median()),
            "energy_fraction_median": float(inside.combined_energy_fraction_of_full_probe.median()),
        },
        "outside_primary": {
            "centered_cosine_median": float(outside.centered_cosine.median()),
            "normalized_agreement_median": float(outside.normalized_agreement.median()),
            "energy_fraction_median": float(outside.combined_energy_fraction_of_full_probe.median()),
        },
        "previous_az_outcome_unchanged": {
            "qualified_count": 5,
            "qualified_unit_ids": sorted(old_five),
            "hybrid_launch_permitted": False,
        },
        "support_relationship": {
            "spearman_retention_vs_min_split_agreement": float(
                comparison.minimum_retained_energy_fraction.corr(
                    comparison.min_normalized_agreement, method="spearman")),
            "donors_min_split_agreement_ge_0p90": int(
                (comparison.min_normalized_agreement >= .90).sum()),
            "donors_min_split_agreement_ge_0p95": int(
                (comparison.min_normalized_agreement >= .95).sum()),
        },
        "finding": (
            "The waveform core is repeatable on frozen per-state support domains, while diffuse "
            "outside-domain tails are weak and poorly repeatable. This supports a narrower, "
            "signal-supported per-state benchmark and an estimator redesign with independently "
            "learned bases or bootstrap uncertainty; it does not justify silently broadening the "
            "support, changing the cohort, or launching the hybrid."
        ),
        "recommendation_scope": "measurement-design evidence only; no hybrid launch or cohort change",
        "inputs": {
            "split_half_spatial_templates_sha256": sha256(split_path),
            "split_half_membership_sha256": sha256(membership_path),
            "frozen_domains_sha256": sha256(args.domains),
        },
    }
    (args.output / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")

    assets = [
        "analysis_summary.json", "per_state_inside_outside_metrics.csv",
        "descriptive_all_state_intersection_metrics.csv", "per_donor_primary_summary.csv",
        "az_support_vs_bc_reliability.csv", "full_cohort_state_summary.csv", "example_spatial_split_profiles.png",
        "full_cohort_spatial_reliability.png",
    ]
    with (args.output / "SHA256SUMS").open("w") as f:
        for name in assets:
            f.write(f"{sha256(args.output / name)}  {name}\n")


if __name__ == "__main__":
    main()
