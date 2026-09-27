"""Read-only CF packet audit and compact physical-coordinate template figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--actual-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.packet / "MANIFEST.json").read_text())
    manifest_ok = all(
        (args.packet / row["path"]).stat().st_size == row["bytes"]
        and sha256(args.packet / row["path"]) == row["sha256"]
        for row in manifest["products"]
    )
    decisions = {
        name: load_npz(args.packet / f"{name}_decisions.npz")
        for name in ("actual_native", "zero_native", "actual_fixed", "zero_fixed")
    }
    actual, zero = load_npz(args.actual_template), load_npz(args.packet / "zero_template_data.npz")
    common = load_npz(args.packet / "common_operator.npz")
    aidx, zidx = common["actual_channel_indices"], common["zero_channel_indices"]
    mapping_ok = (
        np.array_equal(actual["registered_geom"][aidx], common["common_geom"])
        and np.array_equal(zero["registered_geom"][zidx], common["common_geom"])
    )
    observed = ((actual["spike_counts_by_channel"][:, aidx] > 0)
                & (zero["spike_counts_by_channel"][:, zidx] > 0))
    common_operator_ok = (
        np.array_equal(observed, common["observed_both"])
        and np.all(common["fixed_weights"][~observed] == 0)
        and np.allclose(common["fixed_weights"][observed], common["base_weights"][observed])
        and np.array_equal(common["base_weights"], decisions["zero_native"]["spatial_weights"])
    )

    upper = np.triu_indices(748, 1)
    audit = {"manifest_ok": manifest_ok, "physical_mapping_ok": mapping_ok,
             "common_operator_ok": common_operator_ok, "operators": {}}
    for name, data in decisions.items():
        distance = data["distances"]
        nearest = np.min(np.where(np.eye(748, dtype=bool), np.inf, distance), axis=1)
        nearest = nearest[np.isfinite(nearest)]
        audit["operators"][name] = {
            "finite_upper_pairs": int(np.isfinite(distance[upper]).sum()),
            "direct_edges": int(data["direct_force_mask"][upper].sum()),
            "linkage_relations": int(data["force_linkage_mask"][upper].sum()),
            "components": int(np.unique(data["force_component_ids"]).size),
            "nearest_neighbor_median": float(np.median(nearest)),
        }
    for prefix, first, second in (
        ("native", "actual_native", "zero_native"),
        ("fixed", "actual_fixed", "zero_fixed"),
    ):
        a, b = decisions[first], decisions[second]
        audit[f"{prefix}_direct_flips"] = int(
            np.count_nonzero(a["direct_force_mask"][upper] ^ b["direct_force_mask"][upper]))
        audit[f"{prefix}_linkage_flips"] = int(
            np.count_nonzero(a["force_linkage_mask"][upper] ^ b["force_linkage_mask"][upper]))
    audit["fixed_finite_domain_exact"] = bool(np.array_equal(
        np.isfinite(decisions["actual_fixed"]["distances"]),
        np.isfinite(decisions["zero_fixed"]["distances"])))

    rows = list(csv.DictReader(open(args.packet / "PER_UNIT_WAVEFORM_METRICS.csv")))
    values = np.array([float(row["waveform_cosine_distance"]) for row in rows])
    units = np.array([int(row["source_unit"]) for row in rows])
    order = np.lexsort((units, values))
    quantiles = (0.10, 0.50, 0.95)
    selected = [order[round(q * (len(order) - 1))] for q in quantiles]
    audit["figure_examples"] = [
        {"quantile": q, "source_unit": int(units[index]),
         "waveform_cosine_distance": float(values[index])}
        for q, index in zip(quantiles, selected, strict=True)
    ]

    blue, vermillion = "#0072B2", "#D55E00"
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 5.8), constrained_layout=True)
    geom = common["common_geom"]
    for column, (q, index) in enumerate(zip(quantiles, selected, strict=True)):
        unit = units[index]
        aw = actual["templates"][unit][:, aidx]
        zw = zero["templates"][unit][:, zidx]
        ptpa, ptpz = np.ptp(aw, axis=0), np.ptp(zw, axis=0)
        channel = int(np.argmax(np.maximum(ptpa, ptpz)))
        axes[0, column].plot(aw[:, channel], color=blue, linestyle="-", label="actual field")
        axes[0, column].plot(zw[:, channel], color=vermillion, linestyle="--", label="zero field")
        axes[0, column].set_title(
            f"q{int(q*100):02d}: unit {unit}\ncosine distance {values[index]:.3f}"
        )
        axes[0, column].set_xlabel("sample")
        axes[0, column].set_ylabel("template amplitude")
        axes[1, column].plot(ptpa, geom[:, 1], color=blue, linestyle="-", marker=".", ms=2)
        axes[1, column].plot(ptpz, geom[:, 1], color=vermillion, linestyle="--", marker=".", ms=2)
        axes[1, column].set_xlabel("peak-to-peak amplitude")
        axes[1, column].set_ylabel("physical depth (µm)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("CF actual-field vs zero-field templates on identical physical contacts")
    figure.savefig(args.output / "CF_PHYSICAL_TEMPLATE_EXAMPLES.png", dpi=180)
    plt.close(figure)
    (args.output / "CF_PACKET_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
