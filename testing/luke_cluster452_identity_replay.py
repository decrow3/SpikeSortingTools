"""Frozen cached-sort identity redistribution check for rescue cluster 452.

The legacy sort supplies an independent event anchor. Candidate labels are
selected from template peak depth before any event counts in the four endpoint
windows are inspected. No voltage is read and no labels are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_pairs(first: np.ndarray, second: np.ndarray, tolerance: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic maximum-cardinality pairing for sorted interval events."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    first_order = np.argsort(first, kind="stable")
    second_order = np.argsort(second, kind="stable")
    a, b = first[first_order], second[second_order]
    ai: list[int] = []
    bi: list[int] = []
    i = j = 0
    while i < len(a) and j < len(b):
        delta = int(a[i]) - int(b[j])
        if delta < -tolerance:
            i += 1
        elif delta > tolerance:
            j += 1
        else:
            ai.append(int(first_order[i]))
            bi.append(int(second_order[j]))
            i += 1
            j += 1
    return np.asarray(ai, dtype=np.int64), np.asarray(bi, dtype=np.int64)


def template_depths(root: Path) -> np.ndarray:
    templates = np.load(root / "templates.npy", mmap_mode="r")
    geometry = np.load(root / "channel_positions.npy")
    peak_channel = np.argmax(np.ptp(templates, axis=1), axis=1)
    return np.asarray(geometry[peak_channel, 1], dtype=float)


def template_cosine(first: np.ndarray, second: np.ndarray, max_lag: int = 2) -> tuple[float, int]:
    best = (-np.inf, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = first[-lag:], second[:lag]
        elif lag > 0:
            a, b = first[:-lag], second[lag:]
        else:
            a, b = first, second
        av, bv = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
        denominator = np.linalg.norm(av) * np.linalg.norm(bv)
        value = float(np.dot(av, bv) / denominator) if denominator else np.nan
        if np.isfinite(value) and value > best[0]:
            best = (value, lag)
    return best


def extract_candidates(root: Path, candidate_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.load(root / "spike_times.npy", mmap_mode="r").reshape(-1)
    labels = np.load(root / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    mask = np.isin(labels, candidate_ids)
    selected_times = np.asarray(times[mask], dtype=np.int64)
    selected_labels = np.asarray(labels[mask], dtype=np.int64)
    order = np.argsort(selected_times, kind="stable")
    return selected_times[order], selected_labels[order]


def interval(values: np.ndarray, start: int, stop: int) -> slice:
    return slice(np.searchsorted(values, start, side="left"), np.searchsorted(values, stop, side="left"))


def refractory_fraction(times: np.ndarray, threshold: int) -> float:
    times = np.sort(np.asarray(times, dtype=np.int64))
    return float(np.mean(np.diff(times) < threshold)) if len(times) > 1 else 0.0


def match_summary(anchor: np.ndarray, target_times: np.ndarray, target_labels: np.ndarray, tolerance: int) -> dict[str, Any]:
    anchor_hit, target_hit = exclusive_pairs(anchor, target_times, tolerance)
    counts = pd.Series(target_labels[target_hit]).value_counts() if len(target_hit) else pd.Series(dtype=int)
    denominator = max(len(anchor), 1)
    return {
        "anchor_events": int(len(anchor)),
        "matched_events": int(len(anchor_hit)),
        "matched_fraction": float(len(anchor_hit) / denominator),
        "label_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "label_fractions": {str(int(k)): float(v / denominator) for k, v in counts.items()},
    }


def select_anchor(anchor_rows: list[dict[str, Any]], gates: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Select only among anchors eligible under the frozen reference gates."""
    eligible = [
        row for row in anchor_rows
        if row["reference_anchor_events"] >= int(gates["minimum_reference_events"])
        and row["reference_matched_fraction"] >= float(gates["minimum_observed_fraction"])
        and row["excess_over_null"] >= float(gates["minimum_excess_over_shift_null"])
        and (row["target_is_dominant"] or not gates["target_must_be_dominant_rescue_label"])
    ]
    if eligible:
        return max(
            eligible,
            key=lambda row: (row["reference_matched_fraction"], row["reference_target_fraction"]),
        ), True
    return max(
        anchor_rows,
        key=lambda row: (row["reference_target_fraction"], row["reference_matched_fraction"]),
    ), False


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"output exists; preserve prior evidence: {output}")
    config = json.loads(config_path.read_text())
    fs = float(config["matching"]["sampling_frequency_hz"])
    duration_frames = int(round(float(config["matching"]["duration_s"]) * fs))
    tolerance = int(round(float(config["matching"]["tolerance_ms"]) * 1e-3 * fs))
    target_id = int(config["target"]["cluster_id"])
    rescue_root = Path(config["inputs"]["rescue_curated"])
    legacy_root = Path(config["inputs"]["legacy_curated"])

    rescue_depth = template_depths(rescue_root)
    legacy_depth = template_depths(legacy_root)
    target_depth = float(rescue_depth[target_id])
    radius = float(config["candidate_rule"]["maximum_depth_difference_um"])
    rescue_ids = np.flatnonzero(np.abs(rescue_depth - target_depth) <= radius)
    legacy_ids = np.flatnonzero(np.abs(legacy_depth - target_depth) <= radius)
    rescue_times, rescue_labels = extract_candidates(rescue_root, rescue_ids)
    legacy_times, legacy_labels = extract_candidates(legacy_root, legacy_ids)

    windows = []
    for window in config["windows"]:
        windows.append({**window, "start_frame": int(round(window["start_s"] * fs)),
                        "stop_frame": int(round(window["stop_s"] * fs))})
    reference = [w for w in windows if w["role"] == "reference"]
    reference_start = min(w["start_frame"] for w in reference)
    reference_stop = max(w["stop_frame"] for w in reference)
    rescue_reference = interval(rescue_times, reference_start, reference_stop)

    anchor_rows = []
    for legacy_id in legacy_ids:
        keep = legacy_labels == legacy_id
        anchor_all = legacy_times[keep]
        anchor = anchor_all[interval(anchor_all, reference_start, reference_stop)]
        observed = match_summary(
            anchor, rescue_times[rescue_reference], rescue_labels[rescue_reference], tolerance
        )
        null_fractions = []
        for shift_s in config["matching"]["circular_shift_null_s"]:
            shifted = np.sort((anchor + int(round(float(shift_s) * fs))) % duration_frames)
            null_fractions.append(match_summary(shifted, rescue_times, rescue_labels, tolerance)["matched_fraction"])
        null_median = float(np.median(null_fractions))
        target_fraction = observed["label_fractions"].get(str(target_id), 0.0)
        anchor_rows.append({
            "legacy_cluster": int(legacy_id), "reference_anchor_events": observed["anchor_events"],
            "reference_matched_fraction": observed["matched_fraction"],
            "reference_target_fraction": target_fraction, "null_fractions": null_fractions,
            "null_median_fraction": null_median,
            "excess_over_null": observed["matched_fraction"] - null_median,
            "target_is_dominant": bool(observed["label_counts"] and
                int(max(observed["label_counts"], key=observed["label_counts"].get)) == target_id),
        })
    gates = config["anchor_rule"]
    selected, anchor_pass = select_anchor(anchor_rows, gates)
    anchor_rows.sort(key=lambda row: (row["reference_matched_fraction"], row["reference_target_fraction"]), reverse=True)
    selected_id = int(selected["legacy_cluster"])
    anchor_all = legacy_times[legacy_labels == selected_id]

    window_rows = []
    for window in windows:
        a = anchor_all[interval(anchor_all, window["start_frame"], window["stop_frame"])]
        r = interval(rescue_times, window["start_frame"], window["stop_frame"])
        summary = match_summary(a, rescue_times[r], rescue_labels[r], tolerance)
        window_rows.append({"name": window["name"], "role": window["role"], **summary})

    reference_rows = [row for row in window_rows if row["role"] == "reference"]
    failing_rows = [row for row in window_rows if row["role"] == "failing"]
    all_partner_ids = sorted(set().union(*(map(int, row["label_fractions"].keys()) for row in window_rows)) - {target_id})
    rescue_templates = np.load(rescue_root / "templates.npy", mmap_mode="r")
    refractory_frames = int(round(float(config["redistribution_support"]["refractory_period_ms"]) * 1e-3 * fs))
    target_train = rescue_times[rescue_labels == target_id]
    target_rvf = refractory_fraction(target_train, refractory_frames)
    partner_rows = []
    for partner in all_partner_ids:
        ref_fraction = float(np.mean([row["label_fractions"].get(str(partner), 0.0) for row in reference_rows]))
        fail_fractions = [row["label_fractions"].get(str(partner), 0.0) for row in failing_rows]
        target_ref = float(np.mean([row["label_fractions"].get(str(target_id), 0.0) for row in reference_rows]))
        target_fail = float(np.mean([row["label_fractions"].get(str(target_id), 0.0) for row in failing_rows]))
        cosine, lag = template_cosine(rescue_templates[target_id], rescue_templates[partner])
        partner_train = rescue_times[rescue_labels == partner]
        union_rvf = refractory_fraction(np.r_[target_train, partner_train], refractory_frames)
        increase = union_rvf - max(target_rvf, refractory_fraction(partner_train, refractory_frames))
        rule = config["redistribution_support"]
        supported = bool(
            anchor_pass
            and all(value >= float(rule["minimum_partner_fraction_in_each_failing_window"]) for value in fail_fractions)
            and np.mean(fail_fractions) - ref_fraction >= float(rule["minimum_partner_fraction_increase_from_reference"])
            and target_ref - target_fail >= float(rule["minimum_target_fraction_decrease_from_reference"])
            and cosine >= float(rule["minimum_template_cosine"])
            and increase <= float(rule["maximum_union_refractory_increase"])
        )
        partner_rows.append({
            "partner_cluster": int(partner), "reference_fraction": ref_fraction,
            "failing_fractions": fail_fractions, "failing_mean_fraction": float(np.mean(fail_fractions)),
            "target_reference_fraction": target_ref, "target_failing_fraction": target_fail,
            "template_cosine": cosine, "template_best_lag_samples": lag,
            "target_rvf": target_rvf, "partner_rvf": refractory_fraction(partner_train, refractory_frames),
            "union_rvf": union_rvf, "union_rvf_increase": increase,
            "all_support_rules_pass": supported,
        })
    partner_rows.sort(key=lambda row: row["failing_mean_fraction"] - row["reference_fraction"], reverse=True)
    supported = [row for row in partner_rows if row["all_support_rules_pass"]]
    verdict = "identity_redistribution_supported" if len(supported) == 1 else (
        "ambiguous_multiple_supported_partners" if len(supported) > 1 else "identity_redistribution_not_supported"
    )

    output.mkdir(parents=True)
    pd.DataFrame(anchor_rows).to_json(output / "legacy_anchor_candidates.json", orient="records", indent=2)
    pd.DataFrame(partner_rows).to_json(output / "rescue_partner_candidates.json", orient="records", indent=2)
    (output / "window_matches.json").write_text(json.dumps(window_rows, indent=2) + "\n")
    result = {
        "schema": config["schema"], "status": "complete", "verdict": verdict,
        "config_sha256": sha256(config_path), "target_cluster": target_id,
        "target_template_depth_um": target_depth,
        "rescue_spatial_candidate_count": int(len(rescue_ids)),
        "legacy_spatial_candidate_count": int(len(legacy_ids)),
        "selected_legacy_anchor": selected, "anchor_gates_pass": bool(anchor_pass),
        "supported_partner_ids": [row["partner_cluster"] for row in supported],
        "raw_voltage_read": False, "sort_launched": False, "production_changed": False,
        "interpretation": (
            f"A negative result rejects this frozen retained-output identity explanation for "
            f"cluster {target_id}; it does not prove that detection or waveform integrity is normal."
        )
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/luke_cluster452_identity_replay.v1.json"))
    parser.add_argument("--output", type=Path, default=Path("testing/outputs/luke_cluster452_identity_replay_v1"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
