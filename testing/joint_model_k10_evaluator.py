#!/usr/bin/env python3
"""K10 saved-output evaluator for the joint two-frame benchmark.

This module is intentionally independent of DARTsort and raw recordings.  It
matches truth events to each output cluster separately using session-clock time
and per-event placement, then reports split, merge, frame and overlap endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "joint-model-k10-evaluator-result-v3"
VALID_FRAMES = {"tissue", "probe"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_1d(name: str, value: Any, length: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and array.size != length:
        raise ValueError(f"{name} length differs from companion arrays")
    return array


def _integer_samples(name: str, value: Any, length: int | None = None) -> np.ndarray:
    array = _as_1d(name, value, length)
    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain exact integer samples")
    if np.issubdtype(array.dtype, np.integer):
        if array.size and (
            np.any(array < np.iinfo(np.int64).min)
            or np.any(array > np.iinfo(np.int64).max)
        ):
            raise ValueError(f"{name} exceeds the int64 sample domain")
    elif np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError(f"{name} must contain exact integer samples")
        if array.size and (
            np.any(array < -(2.0**63)) or np.any(array >= 2.0**63)
        ):
            raise ValueError(f"{name} exceeds the int64 sample domain")
    else:
        raise ValueError(f"{name} must contain exact integer samples")
    return array.astype(np.int64, copy=False)


def validate_inputs(
    truth: Mapping[str, Any], output: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    required_truth = {
        "event_id", "unit_id", "sample_index", "placement_um", "frame",
        "amplitude", "is_overlap", "decoy_group",
    }
    required_output = {
        "sample_index", "cluster_id", "placement_um", "assigned_frame", "amplitude",
    }
    if missing := sorted(required_truth - set(truth)):
        raise ValueError(f"truth fields missing: {missing}")
    if missing := sorted(required_output - set(output)):
        raise ValueError(f"output fields missing: {missing}")

    n_truth = _as_1d("truth.event_id", truth["event_id"]).size
    n_output = _as_1d("output.sample_index", output["sample_index"]).size
    checked_truth = {key: _as_1d(f"truth.{key}", truth[key], n_truth) for key in required_truth}
    checked_output = {key: _as_1d(f"output.{key}", output[key], n_output) for key in required_output}
    checked_truth["sample_index"] = _integer_samples(
        "truth.sample_index", truth["sample_index"], n_truth
    )
    checked_output["sample_index"] = _integer_samples(
        "output.sample_index", output["sample_index"], n_output
    )
    checked_truth["unit_id"] = _integer_samples("truth.unit_id", truth["unit_id"], n_truth)
    checked_output["cluster_id"] = _integer_samples(
        "output.cluster_id", output["cluster_id"], n_output
    )
    if np.unique(checked_truth["event_id"]).size != n_truth:
        raise ValueError("truth.event_id must be unique")
    for namespace, arrays in (("truth", checked_truth), ("output", checked_output)):
        for field in ("placement_um", "amplitude"):
            values = arrays[field].astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{namespace}.{field} contains nonfinite values")
            arrays[field] = values
    if np.any(checked_truth["amplitude"] <= 0):
        raise ValueError("truth.amplitude must be positive")
    checked_truth["is_overlap"] = checked_truth["is_overlap"].astype(bool)
    checked_truth["frame"] = checked_truth["frame"].astype(str)
    checked_output["assigned_frame"] = checked_output["assigned_frame"].astype(str)
    if not set(checked_truth["frame"]).issubset(VALID_FRAMES):
        raise ValueError("truth.frame contains an unknown label")
    if not set(checked_output["assigned_frame"]).issubset(VALID_FRAMES):
        raise ValueError("output.assigned_frame contains an unknown label")

    time_tolerance = config.get("time_tolerance_samples")
    if time_tolerance != 15:
        raise ValueError("time_tolerance_samples must equal the frozen inclusive value 15")
    spatial_tolerance = config.get("spatial_tolerance_um")
    if not isinstance(spatial_tolerance, (int, float)) or not np.isfinite(spatial_tolerance):
        raise ValueError("spatial_tolerance_um is required and must be finite")
    if spatial_tolerance < 0:
        raise ValueError("spatial_tolerance_um must be nonnegative")
    bounds = config.get("session_bounds_samples")
    if not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise ValueError("session_bounds_samples must be a two-element half-open interval")
    exact_bounds = _integer_samples("session_bounds_samples", bounds, 2)
    lo, hi = map(int, exact_bounds)
    if lo >= hi:
        raise ValueError("invalid session bounds")
    for namespace, samples in (
        ("truth", checked_truth["sample_index"]),
        ("output", checked_output["sample_index"]),
    ):
        if samples.size and (samples.min() < lo or samples.max() >= hi):
            raise ValueError(f"{namespace} samples fall outside session bounds")
    shifts = _integer_samples("null_shifts_samples", config.get("null_shifts_samples", []))
    if shifts.size == 0 or np.any(shifts == 0):
        raise ValueError("nonzero null_shifts_samples are required")
    if config.get("clock_frame") != "session_acquisition_samples":
        raise ValueError("clock_frame must be session_acquisition_samples")
    clock_identity = config.get("clock_identity")
    if not isinstance(clock_identity, str) or not clock_identity:
        raise ValueError("a nonempty clock_identity is required")
    normalized_config = {
        "time_tolerance_samples": 15,
        "spatial_tolerance_um": float(spatial_tolerance),
        "session_bounds_samples": [lo, hi],
        "null_shifts_samples": shifts.tolist(),
        "clock_frame": "session_acquisition_samples",
        "clock_identity": clock_identity,
        "capture_fraction": 0.05,
        "above_chance_rule": "observed_matches_strictly_gt_max_shifted_null",
    }
    return checked_truth, checked_output, normalized_config


def maximum_cardinality_pairs(
    truth_samples: np.ndarray,
    truth_placements: np.ndarray,
    output_samples: np.ndarray,
    output_placements: np.ndarray,
    *,
    time_tolerance_samples: int,
    spatial_tolerance_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic maximum-cardinality matching on the two-gate graph."""
    ts = np.asarray(truth_samples, dtype=np.int64)
    tz = np.asarray(truth_placements, dtype=float)
    os = np.asarray(output_samples, dtype=np.int64)
    oz = np.asarray(output_placements, dtype=float)
    truth_order = np.argsort(ts, kind="stable")
    adjacency: dict[int, list[int]] = {}
    for ti in truth_order:
        candidates = np.flatnonzero(
            (np.abs(os - ts[ti]) <= time_tolerance_samples)
            & (np.abs(oz - tz[ti]) <= spatial_tolerance_um)
        )
        adjacency[int(ti)] = sorted(
            map(int, candidates),
            key=lambda oi: (abs(int(os[oi] - ts[ti])), abs(float(oz[oi] - tz[ti])), oi),
        )

    output_owner: dict[int, int] = {}
    truth_owner: dict[int, int] = {}
    for root in map(int, truth_order):
        queue = [root]
        visited_truth = {root}
        visited_output: set[int] = set()
        parent_output: dict[int, int] = {}
        free_output = None
        for ti in queue:
            for oi in adjacency[ti]:
                if oi in visited_output:
                    continue
                visited_output.add(oi)
                parent_output[oi] = ti
                if oi not in output_owner:
                    free_output = oi
                    break
                next_truth = output_owner[oi]
                if next_truth not in visited_truth:
                    visited_truth.add(next_truth)
                    queue.append(next_truth)
            if free_output is not None:
                break
        if free_output is None:
            continue
        oi = free_output
        while True:
            ti = parent_output[oi]
            previous_output = truth_owner.get(ti)
            output_owner[oi] = ti
            truth_owner[ti] = oi
            if previous_output is None:
                break
            del output_owner[previous_output]
            oi = previous_output
    pairs = sorted(((ti, oi) for oi, ti in output_owner.items()), key=lambda pair: pair[0])
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int64),
    )


def _circular_shift(samples: np.ndarray, shift: int, lo: int, hi: int) -> np.ndarray:
    return ((np.asarray(samples, dtype=np.int64) - lo + int(shift)) % (hi - lo)) + lo


def _cluster_frame(frames: np.ndarray, cluster_id: int) -> str:
    labels = np.unique(frames)
    if labels.size != 1:
        raise ValueError(f"cluster {cluster_id} has multiple assigned_frame labels")
    return str(labels[0])


def evaluate_saved_outputs(
    truth: Mapping[str, Any],
    output: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    overlap_residuals: Mapping[str, Any] | None = None,
    background: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    truth, output, config = validate_inputs(truth, output, config)
    lo, hi = config["session_bounds_samples"]
    clusters = sorted(map(int, np.unique(output["cluster_id"])))
    cluster_rows = {cluster: np.flatnonzero(output["cluster_id"] == cluster) for cluster in clusters}
    units = sorted(map(int, np.unique(truth["unit_id"])))
    unit_rows = {unit: np.flatnonzero(truth["unit_id"] == unit) for unit in units}
    per_unit: list[dict[str, Any]] = []
    pair_cache: dict[tuple[int, int], dict[str, Any]] = {}

    for unit in units:
        trows = unit_rows[unit]
        ts, tz = truth["sample_index"][trows], truth["placement_um"][trows]
        cluster_scores: dict[int, dict[str, Any]] = {}
        for cluster in clusters:
            orows = cluster_rows[cluster]
            ti, oi_local = maximum_cardinality_pairs(
                ts, tz,
                output["sample_index"][orows], output["placement_um"][orows],
                time_tolerance_samples=config["time_tolerance_samples"],
                spatial_tolerance_um=config["spatial_tolerance_um"],
            )
            oi = orows[oi_local]
            tp = int(ti.size)
            fp = int(orows.size - tp)
            fn = int(trows.size - tp)
            denominator = tp + fp + fn
            null_counts = []
            for shift in config["null_shifts_samples"]:
                shifted = _circular_shift(ts, shift, lo, hi)
                nti, _ = maximum_cardinality_pairs(
                    shifted, tz,
                    output["sample_index"][orows], output["placement_um"][orows],
                    time_tolerance_samples=config["time_tolerance_samples"],
                    spatial_tolerance_um=config["spatial_tolerance_um"],
                )
                null_counts.append(int(nti.size))
            score = {
                "cluster_id": cluster,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "accuracy": float(tp / denominator) if denominator else 0.0,
                "capture": float(tp / trows.size) if trows.size else 0.0,
                "null_match_counts": null_counts,
                "null_max": max(null_counts),
                "above_chance": bool(tp > max(null_counts)),
                "matched_truth_row_indices": trows[ti].tolist(),
                "matched_output_row_indices": oi.tolist(),
            }
            cluster_scores[cluster] = score
            pair_cache[(unit, cluster)] = score
        positive_clusters = [cluster for cluster, score in cluster_scores.items() if score["tp"] > 0]
        best = min(
            positive_clusters,
            key=lambda cluster: (
                -cluster_scores[cluster]["accuracy"],
                -cluster_scores[cluster]["tp"],
                cluster_scores[cluster]["fp"],
                cluster,
            ),
        ) if positive_clusters else None
        qualifying = [
            cluster for cluster, score in cluster_scores.items()
            if score["capture"] > config["capture_fraction"] and score["above_chance"]
        ]
        truth_frames = np.unique(truth["frame"][trows])
        if truth_frames.size != 1:
            raise ValueError(f"truth unit {unit} has multiple frame labels")
        assigned_frame = None
        if best is not None:
            assigned_frame = _cluster_frame(
                output["assigned_frame"][cluster_rows[best]], best
            )
        pooled_ti, pooled_oi = maximum_cardinality_pairs(
            ts, tz,
            output["sample_index"], output["placement_um"],
            time_tolerance_samples=config["time_tolerance_samples"],
            spatial_tolerance_um=config["spatial_tolerance_um"],
        )
        gross_cluster_tp = sum(score["tp"] for score in cluster_scores.values())
        per_unit.append({
            "truth_unit_id": unit,
            "truth_event_count": int(trows.size),
            "truth_frame": str(truth_frames[0]),
            "best_output_cluster": best,
            "assigned_frame": assigned_frame,
            "best_score": cluster_scores.get(best),
            "capturing_clusters": qualifying,
            "split": len(qualifying) > 1,
            "pooled_output_unique_tp": int(pooled_ti.size),
            "pooled_output_matched_truth_row_indices": trows[pooled_ti].tolist(),
            "pooled_output_matched_output_row_indices": pooled_oi.tolist(),
            "gross_per_cluster_tp": int(gross_cluster_tp),
            "cross_cluster_duplicate_credit_count": int(
                gross_cluster_tp - pooled_ti.size
            ),
        })

    global_ti, global_oi = maximum_cardinality_pairs(
        truth["sample_index"], truth["placement_um"],
        output["sample_index"], output["placement_um"],
        time_tolerance_samples=config["time_tolerance_samples"],
        spatial_tolerance_um=config["spatial_tolerance_um"],
    )
    credited_by_unit = {
        int(unit): int(np.sum(truth["unit_id"][global_ti] == unit)) for unit in units
    }
    credited_by_cluster = {
        int(cluster): int(np.sum(output["cluster_id"][global_oi] == cluster))
        for cluster in clusters
    }
    gross_pairwise_tp = sum(score["tp"] for score in pair_cache.values())
    per_unit_pooled_tp = sum(row["pooled_output_unique_tp"] for row in per_unit)
    cross_cluster_duplicate_credit_count = sum(
        row["cross_cluster_duplicate_credit_count"] for row in per_unit
    )
    cross_unit_competing_credit_count = int(per_unit_pooled_tp - global_ti.size)

    unit_by_id = {row["truth_unit_id"]: row for row in per_unit}
    many_to_one: dict[int, list[int]] = {}
    for row in per_unit:
        if row["best_output_cluster"] is not None:
            many_to_one.setdefault(row["best_output_cluster"], []).append(row["truth_unit_id"])
    for row in per_unit:
        best = row["best_output_cluster"]
        merged_with = []
        if best is not None:
            for other in units:
                if other == row["truth_unit_id"]:
                    continue
                score = pair_cache[(other, best)]
                if score["capture"] > config["capture_fraction"] and score["above_chance"]:
                    merged_with.append(other)
        row["merged_with_truth_units"] = merged_with
        row["merge"] = bool(merged_with)
        row["many_to_one_truth_units"] = many_to_one.get(best, []) if best is not None else []

    overlap_rows = np.flatnonzero(truth["is_overlap"])
    overlap_detection = overlap_assignment = overlap_frame = 0
    overlap_amplitude_errors: list[float] = []
    for trow in overlap_rows:
        unit = int(truth["unit_id"][trow])
        unit_result = unit_by_id[unit]
        compatible = (
            (np.abs(output["sample_index"] - truth["sample_index"][trow]) <= config["time_tolerance_samples"])
            & (np.abs(output["placement_um"] - truth["placement_um"][trow]) <= config["spatial_tolerance_um"])
        )
        if np.any(compatible):
            overlap_detection += 1
        best = unit_result["best_output_cluster"]
        if best is None:
            continue
        score = pair_cache[(unit, best)]
        if int(trow) not in score["matched_truth_row_indices"]:
            continue
        overlap_assignment += 1
        position = score["matched_truth_row_indices"].index(int(trow))
        orow = score["matched_output_row_indices"][position]
        if output["assigned_frame"][orow] == truth["frame"][trow]:
            overlap_frame += 1
        overlap_amplitude_errors.append(
            abs(float(output["amplitude"][orow] / truth["amplitude"][trow] - 1.0))
        )

    probe_units = [row for row in per_unit if row["truth_frame"] == "probe"]
    tissue_units = [row for row in per_unit if row["truth_frame"] == "tissue"]
    decoy_groups: dict[str, list[int]] = {}
    for unit, rows in unit_rows.items():
        labels = {str(label) for label in truth["decoy_group"][rows] if str(label)}
        if len(labels) > 1:
            raise ValueError(f"truth unit {unit} has multiple decoy groups")
        if labels:
            decoy_groups.setdefault(next(iter(labels)), []).append(unit)
    false_merge_groups = []
    for group, group_units in sorted(decoy_groups.items()):
        bests = [unit_by_id[unit]["best_output_cluster"] for unit in group_units]
        if len(group_units) > 1 and None not in bests and len(set(bests)) < len(bests):
            false_merge_groups.append(group)

    def fraction(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    result = {
        "schema": SCHEMA,
        "config": config,
        "units": per_unit,
        "many_to_one": [
            {"cluster_id": cluster, "truth_unit_ids": unit_ids}
            for cluster, unit_ids in sorted(many_to_one.items()) if len(unit_ids) > 1
        ],
        "global_event_credit": {
            "truth_event_count": int(truth["sample_index"].size),
            "output_event_count": int(output["sample_index"].size),
            "unique_tp": int(global_ti.size),
            "unique_recall": fraction(int(global_ti.size), int(truth["sample_index"].size)),
            "gross_pairwise_tp_not_recovery": int(gross_pairwise_tp),
            "per_unit_pooled_tp": int(per_unit_pooled_tp),
            "cross_cluster_duplicate_credit_count": int(
                cross_cluster_duplicate_credit_count
            ),
            "cross_unit_competing_credit_count": cross_unit_competing_credit_count,
            "matched_truth_row_indices": global_ti.tolist(),
            "matched_output_row_indices": global_oi.tolist(),
            "credited_by_truth_unit": credited_by_unit,
            "credited_by_output_cluster": credited_by_cluster,
        },
        "frames": {
            "probe_fixed_recall": fraction(
                sum(row["assigned_frame"] == "probe" for row in probe_units), len(probe_units)
            ),
            "tissue_fixed_retention": fraction(
                sum(row["assigned_frame"] == "tissue" for row in tissue_units), len(tissue_units)
            ),
            "false_probe_fixed_rate": fraction(
                sum(row["assigned_frame"] == "probe" for row in tissue_units), len(tissue_units)
            ),
            "decoy_false_merge_rate": fraction(len(false_merge_groups), len(decoy_groups)),
            "false_merge_decoy_groups": false_merge_groups,
        },
        "overlap": {
            "truth_event_count": int(overlap_rows.size),
            "detection_recall": fraction(overlap_detection, int(overlap_rows.size)),
            "best_cluster_assignment_recall": fraction(overlap_assignment, int(overlap_rows.size)),
            "frame_assignment_recall": fraction(overlap_frame, int(overlap_rows.size)),
            "median_abs_relative_amplitude_error": (
                float(np.median(overlap_amplitude_errors)) if overlap_amplitude_errors else None
            ),
            "residual_energy": dict(overlap_residuals) if overlap_residuals is not None else None,
        },
        "background": dict(background) if background is not None else None,
        "cost": dict(cost) if cost is not None else None,
        "scope": "saved-output fixture/evaluation only; no biological identity or arm-benefit claim",
    }
    return result


def _finite_real_scalar(name: str, value: Any) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or np.issubdtype(array.dtype, np.bool_)
        or not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def selection_aware_arm_report(
    development: Mapping[str, Mapping[str, Any]],
    selected_holdout: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select on development only, then judge one selected arm on held-out CIs."""
    baseline = config.get("baseline_arm")
    candidates = config.get("candidate_arms")
    primary = config.get("selection_primary_endpoint")
    selection_direction = config.get("selection_direction")
    rules = config.get("endpoint_rules")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError("baseline_arm must be a nonempty string")
    if (
        not isinstance(candidates, Sequence)
        or isinstance(candidates, (str, bytes))
        or not candidates
        or any(not isinstance(arm, str) or not arm for arm in candidates)
        or len(set(candidates)) != len(candidates)
        or baseline in candidates
    ):
        raise ValueError("candidate_arms must be unique nonempty strings excluding baseline")
    if not isinstance(primary, str) or not primary:
        raise ValueError("selection_primary_endpoint must be a nonempty string")
    if selection_direction not in {"higher", "lower"}:
        raise ValueError("selection_direction must be higher or lower")
    if not isinstance(rules, Mapping) or not rules:
        raise ValueError("endpoint_rules must be a nonempty mapping")

    development_values: dict[str, float] = {}
    for arm in candidates:
        if arm not in development or primary not in development[arm]:
            raise ValueError("every candidate requires the development primary endpoint")
        development_values[arm] = _finite_real_scalar(
            "development primary endpoint", development[arm][primary]
        )
    if selection_direction == "higher":
        selected = min(candidates, key=lambda arm: (-development_values[arm], arm))
    else:
        selected = min(candidates, key=lambda arm: (development_values[arm], arm))

    required_holdout_keys = {"candidate_arm", "baseline_arm", "endpoints"}
    if not isinstance(selected_holdout, Mapping) or set(selected_holdout) != required_holdout_keys:
        raise ValueError(
            "holdout must contain exactly one selected-arm-versus-baseline summary"
        )
    if selected_holdout.get("candidate_arm") != selected:
        raise ValueError("holdout candidate_arm must equal the development-selected arm")
    if selected_holdout.get("baseline_arm") != baseline:
        raise ValueError("holdout baseline_arm differs from the frozen baseline")
    endpoint_values = selected_holdout.get("endpoints")
    if not isinstance(endpoint_values, Mapping) or set(endpoint_values) != set(rules):
        raise ValueError("holdout endpoints must exactly match endpoint_rules")

    endpoint_results: dict[str, dict[str, Any]] = {}
    for endpoint in sorted(rules):
        rule = rules[endpoint]
        value = endpoint_values[endpoint]
        if not isinstance(rule, Mapping) or rule.get("direction") not in {"higher", "lower"}:
            raise ValueError(f"endpoint rule {endpoint} has invalid direction")
        try:
            margin = _finite_real_scalar(
                f"endpoint rule {endpoint} practical loss margin",
                rule.get("practical_loss_margin"),
            )
        except ValueError as exc:
            raise ValueError(f"endpoint rule {endpoint} has invalid practical loss margin") from exc
        if margin < 0:
            raise ValueError(f"endpoint rule {endpoint} has invalid practical loss margin")
        if not isinstance(value, Mapping):
            raise ValueError(f"holdout endpoint {endpoint} must be a mapping")
        try:
            estimate = _finite_real_scalar(f"holdout {endpoint} estimate", value["estimate"])
            ci_low = _finite_real_scalar(f"holdout {endpoint} ci_low", value["ci_low"])
            ci_high = _finite_real_scalar(f"holdout {endpoint} ci_high", value["ci_high"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"holdout endpoint {endpoint} lacks numeric CI fields") from exc
        if not ci_low <= estimate <= ci_high:
            raise ValueError(f"holdout endpoint {endpoint} has an invalid confidence interval")
        n_units_array = _integer_samples(f"holdout.{endpoint}.n_units", [value.get("n_units")], 1)
        n_units = int(n_units_array[0])
        if n_units <= 0:
            raise ValueError(f"holdout endpoint {endpoint} n_units must be positive")
        direction = rule["direction"]
        if direction == "higher":
            if ci_low >= -margin:
                decision = "PASS_NONINFERIOR"
            elif ci_high < -margin:
                decision = "FAIL_PRACTICAL_LOSS"
            else:
                decision = "INCONCLUSIVE"
        else:
            if ci_high <= margin:
                decision = "PASS_NONINFERIOR"
            elif ci_low > margin:
                decision = "FAIL_PRACTICAL_LOSS"
            else:
                decision = "INCONCLUSIVE"
        endpoint_results[endpoint] = {
            "direction": direction,
            "practical_loss_margin": margin,
            "effect_convention": "candidate_minus_baseline",
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_units": n_units,
            "decision": decision,
        }

    decisions = [row["decision"] for row in endpoint_results.values()]
    if "FAIL_PRACTICAL_LOSS" in decisions:
        overall = "FAIL_PRACTICAL_LOSS"
    elif all(decision == "PASS_NONINFERIOR" for decision in decisions):
        overall = "PASS_NONINFERIOR"
    else:
        overall = "INCONCLUSIVE"
    return {
        "schema": "joint-model-k10-selection-aware-arm-report-v1",
        "selection_source": "development_only",
        "baseline_arm": baseline,
        "candidate_arms": list(candidates),
        "selection_primary_endpoint": primary,
        "selection_direction": selection_direction,
        "development_primary_values": development_values,
        "selected_arm": selected,
        "holdout_scope": "selected_arm_vs_baseline_only",
        "endpoints": endpoint_results,
        "overall_decision": overall,
        "equivalence_claim": False,
        "superiority_claim": False,
        "inconclusive_is_equivalence": False,
        "minimum_unit_floor": None,
    }


def _load_npz(path: Path, prefix: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            name[len(prefix):]: np.asarray(archive[name])
            for name in archive.files if name.startswith(prefix)
        }


def write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".k10-result.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = evaluate_saved_outputs(
        _load_npz(args.arrays, "truth_"),
        _load_npz(args.arrays, "output_"),
        spec["config"],
        overlap_residuals=spec.get("overlap_residuals"),
        background=spec.get("background"),
        cost=spec.get("cost"),
    )
    result["provenance"] = {
        "arrays": str(args.arrays.resolve()),
        "arrays_sha256": sha256_path(args.arrays),
        "spec": str(args.spec.resolve()),
        "spec_sha256": sha256_path(args.spec),
        "evaluator_sha256": sha256_path(Path(__file__).resolve()),
    }
    write_json_create_only(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
