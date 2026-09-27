"""Independent saved-cache measurement audit for CK anchors; no voltage read."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from testing.cg_anchor_helper import (
    _conditioned_template,
    _cosine,
    half1_noise_std,
    score_anchor_reproducibility,
)
from testing.cj_force_gate_w2 import write_csv


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_values(first, second, blocks1, blocks2, support, noise, seed, count):
    rng = np.random.default_rng(seed)
    unique1, unique2 = np.unique(blocks1), np.unique(blocks2)
    values = []
    for _ in range(count):
        draw1 = rng.choice(unique1, len(unique1), replace=True)
        draw2 = rng.choice(unique2, len(unique2), replace=True)
        take1 = np.concatenate([np.flatnonzero(blocks1 == block) for block in draw1])
        take2 = np.concatenate([np.flatnonzero(blocks2 == block) for block in draw2])
        values.append(_cosine(
            _conditioned_template(first[take1], support, noise),
            _conditioned_template(second[take2], support, noise)))
    return np.asarray(values)


def synthetic_control(equal, seed=701, draws=300):
    rng = np.random.default_rng(seed)
    time = np.arange(121)
    base = -5 * np.exp(-0.5 * ((time - 42) / 2.5) ** 2)
    base += 2 * np.exp(-0.5 * ((time - 51) / 5.0) ** 2)
    template = base[:, None] * np.linspace(1.0, 0.45, 8)[None]
    first = template + rng.normal(0, 1.0, size=(100, 121, 8))
    second = template + rng.normal(0, 1.0, size=(100, 121, 8))
    if equal:
        blocks = np.repeat(np.arange(10), 10)
    else:
        sizes = np.array([1, 1, 1, 1, 1, 5, 10, 20, 25, 35])
        blocks = np.repeat(np.arange(10), sizes)
    result = score_anchor_reproducibility(
        first, second, blocks_half1=blocks, blocks_half2=blocks,
        support_channels=np.arange(8), n_bootstrap=draws, seed=seed)
    return {
        "point": result["point_reliability"],
        "bootstrap_lower": result["ci_lower"],
        "bootstrap_upper": result["ci_upper"],
        "point_minus_lower": result["point_reliability"] - result["ci_lower"],
        "block_sizes": np.bincount(blocks).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.ck_root
    manifest = json.loads((root / "MANIFEST.json").read_text())
    manifest_ok = True
    for item in manifest["products"]:
        path = root / item["path"]
        manifest_ok &= path.stat().st_size == item["bytes"]
        manifest_ok &= sha256(path) == item["sha256"]

    rows = list(csv.DictReader(open(root / "SELECTED_EVENTS.csv")))
    raw = np.load(root / "RAW_SNIPPETS.npy", mmap_mode="r")
    with np.load(root / "MASKS_AND_SELECTION.npz", allow_pickle=False) as data:
        selection = {key: data[key] for key in data.files}
    with np.load(root / "NOISE_STD.npz", allow_pickle=False) as data:
        saved_noise = {key: data[key] for key in data.files}
    with np.load(root / "BOOTSTRAP_SAMPLES.npz", allow_pickle=False) as data:
        saved_bootstrap = {key: data[key] for key in data.files}
    reliability = {int(row["unit"]): row for row in csv.DictReader(
        open(root / "ANCHOR_RELIABILITY.csv"))}
    if len(rows) != len(raw) or len(rows) != 1600:
        raise ValueError("selected-row/cache cardinality mismatch")
    fs = 29999.759166666667
    selected_indices = selection["selected_event_indices"]
    row_checks = {
        "event_index": all(int(row["event_index"]) == selected_indices[i]
                           for i, row in enumerate(rows)),
        "event_row_id": all(int(row["event_row_id"]) == selection[
            "selected_event_row_ids"][i] for i, row in enumerate(rows)),
        "time_sample": all(int(row["time_sample"]) == selection["times_samples"][
            selected_indices[i]] for i, row in enumerate(rows)),
        "channel": all(int(row["channel"]) == selection["channels"][
            selected_indices[i]] for i, row in enumerate(rows)),
        "session_time": all(abs(float(row["time_session_s"]) - (
            900 + int(row["time_sample"]) / fs)) < 1e-10 for row in rows),
    }

    diagnostics = []
    exact_units = []
    for unit in selection["selected_units"]:
        unit = int(unit)
        indices = [i for i, row in enumerate(rows) if int(row["unit"]) == unit]
        first_indices = [i for i in indices if int(rows[i]["half"]) == 1]
        second_indices = [i for i in indices if int(rows[i]["half"]) == 2]
        first, second = np.asarray(raw[first_indices]), np.asarray(raw[second_indices])
        blocks1 = np.array([int(rows[i]["block5"]) for i in first_indices])
        blocks2 = np.array([int(rows[i]["block5"]) for i in second_indices])
        support = np.array([int(value) for value in rows[indices[0]][
            "support_channels"].split(";")])
        noise = half1_noise_std(first)
        template1 = _conditioned_template(first, support, noise)
        template2 = _conditioned_template(second, support, noise)
        point = _cosine(template1, template2)
        mean_template = (template1 + template2) / 2
        difference = template1 - template2
        mean_energy = float(np.sum(mean_template ** 2))
        difference_energy = float(np.sum(difference ** 2))
        feature_power = mean_template.ravel() ** 2
        effective_dimension = float(feature_power.sum() ** 2 / np.sum(feature_power ** 2))
        ptp1 = np.ptp(np.median(first, axis=0), axis=0)
        ptp2 = np.ptp(np.median(second, axis=0), axis=0)
        depths = selection["physical_geom"][:, 1]
        centroid1 = float(np.sum(ptp1 * depths) / np.sum(ptp1))
        centroid2 = float(np.sum(ptp2 * depths) / np.sum(ptp2))
        peak_channel = support[int(np.argmax(ptp1[support] + ptp2[support]))]
        trough1 = int(np.argmin(np.median(first, axis=0)[:, peak_channel]))
        trough2 = int(np.argmin(np.median(second, axis=0)[:, peak_channel]))
        record = {
            "unit": unit, "point_recomputed": point,
            "point_saved": float(reliability[unit]["point_reliability"]),
            "point_abs_error": abs(point - float(reliability[unit]["point_reliability"])),
            "noise_array_equal": bool(np.array_equal(noise, saved_noise[f"unit_{unit}"])),
            "support_channels": len(support),
            "conditioned_feature_count": int(template1.size),
            "signal_effective_dimension": effective_dimension,
            "difference_to_mean_energy": difference_energy / mean_energy,
            "raw_ptp_depth_centroid_half2_minus_half1_um": centroid2 - centroid1,
            "raw_peak_trough_sample_half1": trough1,
            "raw_peak_trough_sample_half2": trough2,
            "observed_channel_median_half2_minus_half1": float(
                np.median([int(rows[i]["channel"]) for i in second_indices])
                - np.median([int(rows[i]["channel"]) for i in first_indices])),
        }
        if unit in (23, 120):
            values = bootstrap_values(
                first, second, blocks1, blocks2, support, noise, unit, 100)
            record["first100_bootstrap_array_equal"] = bool(np.array_equal(
                values, saved_bootstrap[f"unit_{unit}"][:100]))
            record["first100_bootstrap_max_abs_error"] = float(np.max(np.abs(
                values - saved_bootstrap[f"unit_{unit}"][:100])))
            exact_units.append(unit)
        diagnostics.append(record)

    write_csv(args.output / "UNIT_DIAGNOSTICS.csv", diagnostics)
    synthetic = {
        "frozen_before_output": True,
        "seed": 701, "draws": 300,
        "equal_blocks": synthetic_control(True),
        "unequal_blocks": synthetic_control(False),
    }
    (args.output / "SYNTHETIC_CONTROL.json").write_text(
        json.dumps(synthetic, indent=2) + "\n")
    summary = {
        "status": "complete_saved_cache_only",
        "manifest_all_products_valid": bool(manifest_ok),
        "raw_cache_shape": list(raw.shape), "raw_cache_dtype": str(raw.dtype),
        "selected_rows": len(rows), "row_clock_channel_checks": row_checks,
        "deterministic_exact_units": exact_units,
        "all_point_scores_exact": all(x["point_abs_error"] == 0 for x in diagnostics),
        "all_noise_arrays_exact": all(x["noise_array_equal"] for x in diagnostics),
        "two_unit_first100_bootstrap_exact": all(
            x.get("first100_bootstrap_array_equal", True) for x in diagnostics),
        "max_abs_raw_depth_centroid_change_um": max(abs(x[
            "raw_ptp_depth_centroid_half2_minus_half1_um"]) for x in diagnostics),
        "max_abs_observed_channel_median_change": max(abs(x[
            "observed_channel_median_half2_minus_half1"]) for x in diagnostics),
        "max_abs_peak_trough_sample_change": max(abs(x[
            "raw_peak_trough_sample_half2"] - x["raw_peak_trough_sample_half1"])
            for x in diagnostics),
        "median_difference_to_mean_energy": float(np.median([
            x["difference_to_mean_energy"] for x in diagnostics])),
        "median_effective_dimension": float(np.median([
            x["signal_effective_dimension"] for x in diagnostics])),
        "voltage_reads": 0, "raw_cache_only": True,
    }
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
