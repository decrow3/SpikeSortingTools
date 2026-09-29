"""Test target-specific full_st recovery over wider depth radii against time-shift nulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from testing.luke_cluster21_detection_lineage import detection_template_depths, matched_fraction
from testing.luke_cluster452_identity_replay import interval, sha256, template_depths


def circular_shift(values: np.ndarray, start: int, stop: int, shift: int) -> np.ndarray:
    width = stop - start
    if width <= 0:
        raise ValueError("window must have positive width")
    return start + ((np.asarray(values, dtype=np.int64) - start + shift) % width)


def radius_rows(anchor: np.ndarray, times: np.ndarray, depths: np.ndarray, target_depth: float,
                radii: list[float], tolerance: int, shifts: list[int], start: int,
                stop: int) -> list[dict]:
    rows = []
    for radius in radii:
        candidates = np.sort(times[np.abs(depths - target_depth) <= radius], kind="stable")
        observed = matched_fraction(anchor, candidates, tolerance)
        nulls = [
            matched_fraction(np.sort(circular_shift(anchor, start, stop, shift)), candidates, tolerance)
            for shift in shifts
        ]
        rows.append({
            "radius_um": radius, "candidate_events": int(len(candidates)),
            "observed_matched_fraction": observed,
            "null_matched_fractions": nulls,
            "maximum_null_matched_fraction": max(nulls),
            "observed_minus_max_null": observed - max(nulls),
        })
    base = rows[0]
    for row in rows:
        row["observed_gain_over_100um"] = (
            row["observed_matched_fraction"] - base["observed_matched_fraction"]
        )
        row["maximum_null_gain_over_100um"] = max(
            value - base_value
            for value, base_value in zip(row["null_matched_fractions"], base["null_matched_fractions"])
        )
        row["gain_above_max_null_gain"] = (
            row["observed_gain_over_100um"] - row["maximum_null_gain_over_100um"]
        )
    return rows


def run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    sorter = Path(config["sorter_output"])
    fs = float(config["sampling_frequency_hz"])
    tolerance = int(round(float(config["tolerance_ms"]) * 1e-3 * fs))
    radii = [float(value) for value in config["depth_radii_um"]]
    if radii[0] != 100.0 or radii != sorted(radii):
        raise RuntimeError("depth radii must be sorted and begin at the audited 100 um baseline")
    shifts = [int(round(float(value) * fs)) for value in config["circular_shift_null_s"]]
    target_depth = float(template_depths(Path(config["curated_output"]))[int(config["target_cluster"])])

    ops = np.load(sorter / "ops.npy", allow_pickle=True).item()
    detection_depth = detection_template_depths(ops)
    full_st = np.load(sorter / "full_st.npy", mmap_mode="r")
    full_times = np.asarray(full_st[:, 0], dtype=np.int64)
    template_ids = np.asarray(full_st[:, 1], dtype=np.int64)
    full_depths = detection_depth[template_ids]
    order = np.argsort(full_times, kind="stable")
    full_times, full_depths = full_times[order], full_depths[order]

    legacy = Path(config["legacy_curated"])
    legacy_times = np.load(legacy / "spike_times.npy", mmap_mode="r").reshape(-1)
    legacy_labels = np.load(legacy / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    anchor_times = np.array(
        legacy_times[legacy_labels == int(config["legacy_anchor_cluster"])], dtype=np.int64, copy=True
    )
    anchor_times.sort(kind="stable")

    windows = []
    for window in config["windows"]:
        start = int(round(float(window["start_s"]) * fs))
        stop = int(round(float(window["stop_s"]) * fs))
        anchor = anchor_times[interval(anchor_times, start, stop)]
        selected = interval(full_times, start, stop)
        rows = radius_rows(anchor, full_times[selected], full_depths[selected], target_depth,
                           radii, tolerance, shifts, start, stop)
        windows.append({"name": window["name"], "role": window["role"],
                        "anchor_events": int(len(anchor)), "radii": rows})

    failing = next(row for row in windows if row["role"] == "failing")
    rule = config["spatial_escape_support_rule"]
    qualifying = [
        row["radius_um"] for row in failing["radii"][1:]
        if row["observed_gain_over_100um"] >= float(rule["minimum_failing_observed_gain_over_100um"])
        and row["gain_above_max_null_gain"] >= float(rule["minimum_failing_gain_above_max_null_gain"])
    ]
    verdict = "target_specific_spatial_escape_supported" if qualifying else "wider_radius_recovery_not_supported"
    result = {
        "schema": config["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "target_cluster": int(config["target_cluster"]),
        "legacy_anchor_cluster": int(config["legacy_anchor_cluster"]), "target_depth_um": target_depth,
        "windows": windows, "qualifying_radii_um": qualifying,
        "v1_interpretation": "Whole-probe time-only coincidence is descriptive and not target-specific because it lacked an event-density null.",
        "sort_launched": False, "raw_voltage_read": False, "production_changed": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster553_radial_detection_null.v2.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster553_radial_detection_null_v2"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
