#!/usr/bin/env python3
"""Explain the frozen AZ support blocker from already-saved arrays only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXTRACTION = ROOT / "testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/ba_attempt4"
QUALIFICATION = ROOT / "testing/outputs/luke_au_cpu_preparation/donor_qualification_v1"
OUTPUT = QUALIFICATION / "support_explanation"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(np.asarray(x))
    ry = rankdata(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def q(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    with np.load(EXTRACTION / "full_probe_final_label_templates.npz", allow_pickle=False) as z:
        templates = np.asarray(z["templates"], dtype=np.float32)
        unit_ids = np.asarray(z["unit_ids"], dtype=np.int64)
        geom = np.asarray(z["geometry_um"], dtype=np.float64)
    measurements = {
        int(row["unit_id"]): row
        for row in read_csv(EXTRACTION / "full_probe_candidate_measurements.csv")
    }
    splits = {
        int(row["unit_id"]): row
        for row in read_csv(EXTRACTION / "split_half_diagnostics.csv")
    }
    attempts = read_csv(QUALIFICATION / "placement_attempts.csv")
    attempt_by_unit: dict[int, list[dict]] = {}
    for row in attempts:
        attempt_by_unit.setdefault(int(row["unit_id"]), []).append(row)

    rows = []
    cumulative = []
    radii = np.arange(0.0, 1921.0, 40.0)
    for i, unit_id in enumerate(unit_ids):
        uid = int(unit_id)
        best = max(
            attempt_by_unit[uid],
            key=lambda row: float(row["minimum_retained_energy_fraction"]),
        )
        m = measurements[uid]
        s = splits[uid]
        channel_energy = np.square(templates[i], dtype=np.float64).sum(axis=0)
        total = float(channel_energy.sum())
        weights = channel_energy / total
        peak_depth = float(m["depth_um"])
        distance = np.abs(geom[:, 1] - peak_depth)
        cumulative.append([float(weights[distance <= radius].sum()) for radius in radii])
        sorted_weights = np.sort(weights)[::-1]
        effective_channels = float(1.0 / np.square(weights).sum())
        row = {
            "unit_id": uid,
            "qualified": best["placement_pass"] == "True",
            "best_base_shift_um": float(best["base_shift_um"]),
            "best_minimum_all_state_retention": float(best["minimum_retained_energy_fraction"]),
            "best_minimum_roundtrip_cosine": float(best["minimum_roundtrip_centered_cosine"]),
            "depth_um": peak_depth,
            "ptp_standardized": float(m["full_probe_ptp_standardized"]),
            "valid_rest_waveforms": int(m["valid_rest_waveforms"]),
            "split_centered_cosine": float(s["split_centered_cosine"]),
            "left_to_full_centered_cosine": float(s["left_to_full_centered_cosine"]),
            "right_to_full_centered_cosine": float(s["right_to_full_centered_cosine"]),
            "left_to_full_ptp_ratio": float(s["left_to_full_ptp_ratio"]),
            "right_to_full_ptp_ratio": float(s["right_to_full_ptp_ratio"]),
            "energy_within_200um_of_peak": float(weights[distance <= 200.0].sum()),
            "energy_within_400um_of_peak": float(weights[distance <= 400.0].sum()),
            "energy_within_800um_of_peak": float(weights[distance <= 800.0].sum()),
            "top4_channel_energy_fraction": float(sorted_weights[:4].sum()),
            "top16_channel_energy_fraction": float(sorted_weights[:16].sum()),
            "effective_energy_channel_count": effective_channels,
        }
        rows.append(row)

    cumulative = np.asarray(cumulative)
    passed = np.asarray([row["qualified"] for row in rows], dtype=bool)
    retention = np.asarray([row["best_minimum_all_state_retention"] for row in rows])
    split_cosine = np.asarray([row["split_centered_cosine"] for row in rows])
    ptp = np.asarray([row["ptp_standardized"] for row in rows])
    depth = np.asarray([row["depth_um"] for row in rows])
    valid_rest = np.asarray([row["valid_rest_waveforms"] for row in rows])
    local400 = np.asarray([row["energy_within_400um_of_peak"] for row in rows])
    effective_channels = np.asarray([row["effective_energy_channel_count"] for row in rows])

    high_stability = split_cosine >= 0.98
    high_stability_fail = high_stability & ~passed
    low_stability = split_cosine < 0.90
    near = int(np.argmin(np.where(passed, np.inf, np.abs(retention - 0.99))))
    median_fail_value = float(np.median(retention[~passed]))
    median_fail = int(np.argmin(np.where(passed, np.inf, np.abs(retention - median_fail_value))))
    worst = int(np.argmin(retention))
    strongest_pass = int(np.argmax(np.where(passed, ptp, -np.inf)))
    representative_indices = [strongest_pass, near, median_fail, worst]

    summary = {
        "schema": "luke-az-support-explanation-v1",
        "status": "exploratory_saved_array_explanation_complete",
        "decision": "keep AZ hybrid blocked; the saved outputs do not identify remote energy as coherent waveform versus estimation noise",
        "source_scope": {
            "voltage_read": False,
            "new_extraction": False,
            "new_template_fit": False,
            "main_full_probe_templates_available": True,
            "split_half_template_arrays_available": False,
            "split_half_scalar_diagnostics_available": True,
        },
        "counts": {
            "candidates": int(unit_ids.size),
            "qualified": int(passed.sum()),
            "rejected": int((~passed).sum()),
            "high_split_stability_ge_0p98": int(high_stability.sum()),
            "high_split_stability_but_rejected": int(high_stability_fail.sum()),
            "low_split_stability_lt_0p90": int(low_stability.sum()),
        },
        "associations_spearman": {
            "retention_vs_split_half_cosine": spearman(retention, split_cosine),
            "retention_vs_ptp": spearman(retention, ptp),
            "retention_vs_depth": spearman(retention, depth),
            "retention_vs_valid_rest_waveforms": spearman(retention, valid_rest),
            "retention_vs_energy_within_400um": spearman(retention, local400),
            "retention_vs_effective_energy_channel_count": spearman(retention, effective_channels),
        },
        "qualified_group": {
            "unit_ids": [int(unit_ids[i]) for i in np.flatnonzero(passed)],
            "depth_um": q(depth[passed]),
            "ptp_standardized": q(ptp[passed]),
            "split_centered_cosine": q(split_cosine[passed]),
            "energy_within_400um": q(local400[passed]),
            "best_retention": q(retention[passed]),
        },
        "rejected_group": {
            "depth_um": q(depth[~passed]),
            "ptp_standardized": q(ptp[~passed]),
            "split_centered_cosine": q(split_cosine[~passed]),
            "energy_within_400um": q(local400[~passed]),
            "best_retention": q(retention[~passed]),
        },
        "representatives": [
            {
                "unit_id": int(unit_ids[i]),
                "role": role,
                "qualified": bool(passed[i]),
                "best_retention": float(retention[i]),
                "split_centered_cosine": float(split_cosine[i]),
                "ptp_standardized": float(ptp[i]),
                "depth_um": float(depth[i]),
            }
            for i, role in zip(
                representative_indices,
                ["strongest qualified", "closest rejected to gate", "median rejected", "lowest retention"],
            )
        ],
        "identifiability_limit": (
            "Whole-template split-half cosine/PTP summaries cannot localize reproducibility to the remote spatial tails. "
            "Without retained split-half spatial templates, coherent broad support and stable/systematic or random estimation tails cannot be separated."
        ),
        "selection_bias_warning": (
            "The geometry-bound gate selects narrow, high-retention templates that fit every state inside the lower 182 sites. "
            "This conditions the donor bank on depth, footprint and SNR and is not a representative estimate of support failure among real neurons."
        ),
    }

    write_csv(args.output / "support_explanation_metrics.csv", rows)
    (args.output / "support_explanation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    blue = "#0072B2"
    orange = "#E69F00"
    vermillion = "#D55E00"
    grey = "#777777"

    ax = axes[0, 0]
    ax.scatter(split_cosine[~passed], retention[~passed], s=28, facecolors="none", edgecolors=grey, label="Rejected")
    ax.scatter(split_cosine[passed], retention[passed], s=48, marker="D", color=blue, label="Qualified")
    ax.axhline(0.99, color=vermillion, linestyle="--", linewidth=1.2, label="99% gate")
    ax.axvline(0.98, color=orange, linestyle=":", linewidth=1.2, label="High-stability reference")
    ax.set(xlabel="Split-half whole-template centered cosine", ylabel="Best minimum all-state retention", title="Retention versus overall split-half stability")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.18)

    ax = axes[0, 1]
    sizes = 18 + 2.2 * np.sqrt(np.maximum(ptp, 0))
    ax.scatter(depth[~passed], retention[~passed], s=sizes[~passed], facecolors="none", edgecolors=grey)
    ax.scatter(depth[passed], retention[passed], s=sizes[passed] + 18, marker="D", color=blue)
    ax.axhline(0.99, color=vermillion, linestyle="--", linewidth=1.2)
    ax.axvspan(2020, 3820, color=orange, alpha=0.08, label="Native target depth span")
    ax.set(xlabel="Native peak depth (µm)", ylabel="Best minimum all-state retention", title="Geometry gate selects a narrow depth/SNR subset")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(radii, np.median(cumulative[~passed], axis=0), color=grey, linestyle="--", linewidth=2, label="Rejected median (n=85)")
    ax.fill_between(radii, np.percentile(cumulative[~passed], 25, axis=0), np.percentile(cumulative[~passed], 75, axis=0), color=grey, alpha=0.14)
    ax.plot(radii, np.median(cumulative[passed], axis=0), color=blue, linestyle="-", linewidth=2, label="Qualified median (n=5)")
    ax.axhline(0.99, color=vermillion, linestyle=":", linewidth=1.2)
    ax.set(xlabel="Radius from native peak depth (µm)", ylabel="Cumulative main-template energy fraction", title="Main-template spatial concentration")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    styles = [
        (blue, "-"),
        (orange, "--"),
        (vermillion, "-."),
        ("#000000", ":"),
    ]
    for index, (color, linestyle), role in zip(
        representative_indices,
        styles,
        ["qualified", "near miss", "median reject", "lowest retention"],
    ):
        energy = np.square(templates[index], dtype=np.float64).sum(axis=0)
        rel_depth = geom[:, 1] - depth[index]
        by_depth = {}
        for value, e in zip(rel_depth, energy):
            by_depth[float(value)] = by_depth.get(float(value), 0.0) + float(e)
        xs = np.asarray(sorted(by_depth))
        ys = np.asarray([by_depth[x] for x in xs])
        ys /= ys.sum()
        ax.plot(xs, np.maximum(ys, 1e-8), color=color, linestyle=linestyle, linewidth=1.6, label=f"u{int(unit_ids[index])} {role}")
    ax.set_yscale("log")
    ax.set(xlabel="Depth relative to native peak (µm)", ylabel="Energy fraction per depth row", title="Saved main-template profiles cannot identify tail coherence")
    ax.grid(alpha=0.18, which="both")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("AZ support blocker — saved-array explanatory audit", fontsize=13)
    fig.savefig(args.output / "support_explanation.png", dpi=180)
    fig.savefig(args.output / "support_explanation.pdf")
    plt.close(fig)

    summary["artifacts"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(args.output.iterdir())
        if path.is_file() and path.name != "support_explanation_summary.json"
    }
    (args.output / "support_explanation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
